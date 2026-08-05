"""Mini-swe-agent runner for the blackbox SWE-agent recipe.

Agent runs inside a remote sandbox via sidecar tool image mount.
The runner creates the sandbox, pipes task config via stdin, parses
the result from stdout, and evaluates reward in the same sandbox.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shlex
import time
from pathlib import Path

import httpx

from examples.blackbox_recipes.mini_swe_agent.dataset import extract_image
from examples.blackbox_recipes.mini_swe_agent.reward import build_reward_context, evaluate_in_env
from examples.blackbox_recipes.sandbox_client import (
    SandboxClient,
    extract_upstream,
    is_dead_connection,
    repair_yr_connection,
    rewrite_gateway_url,
)
from uni_agent.gateway.session import SessionHandle

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

DEFAULT_TOOL_IMAGE = "swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest"

# ── Keepalive / retry knobs ─────────────────────────────────────────────
# The netentsec proxy drops the CONNECT tunnel after ~30 s of idle.
# A background task that runs a harmless command every KEEPALIVE_INTERVAL
# seconds keeps the yr RPC connection alive during long agent runs.
KEEPALIVE_INTERVAL = int(os.getenv("SANDBOX_KEEPALIVE_INTERVAL", "20"))
# Max attempts per sample when the sandbox connection dies mid-run.
MAX_ATTEMPTS = int(os.getenv("SANDBOX_MAX_ATTEMPTS", "2"))


class SandboxConnectionDead(Exception):
    """Raised when the shared yr RPC connection is detected as dead.

    The caller (mini_swe_agent_runner) catches this, rebuilds the yr
    connection via :func:`repair_yr_connection`, and retries the sample.
    """


async def _keepalive_loop(
    sandbox: SandboxClient,
    stop_event: asyncio.Event,
    dead_event: asyncio.Event,
    sample_index: int,
) -> None:
    """Periodically run ``pwd`` inside *sandbox* until *stop_event* is set.

    Logs every probe.  On failure, sets *dead_event* so the runner can
    detect the connection loss and rebuild.
    """
    n = 0
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=KEEPALIVE_INTERVAL)
            return  # stopped
        except asyncio.TimeoutError:
            pass  # time to send a keepalive probe

        n += 1
        try:
            r = await sandbox.run("pwd", timeout=10)
            if r.exit_code == 0:
                logger.info("[sample %d] keepalive #%d OK (pwd)", sample_index, n)
            else:
                logger.warning(
                    "[sample %d] keepalive #%d FAILED rc=%d — connection may be dead: %s",
                    sample_index, n, r.exit_code, r.stderr[:120],
                )
                if is_dead_connection(r):
                    dead_event.set()
                    # Connection is gone — stop probing so we don't spawn
                    # more in-flight RPCs that would linger after repair.
                    logger.warning("[sample %d] keepalive #%d dead — stopping keepalive", sample_index, n)
                    return
        except Exception as e:
            logger.warning("[sample %d] keepalive #%d exception: %s", sample_index, n, e)
            dead_event.set()
            return


class SandboxEnvForReward:
    """Adapts :class:`Sandbox` to the async env interface used by
    reward specs (``communicate``, ``write_file``, ``read_file``).
    """

    def __init__(self, sandbox):
        self._sandbox = sandbox

    async def communicate(self, input: str, timeout=600, check="ignore", error_msg="Command failed") -> str:
        result = await self._sandbox.run(input, timeout=int(timeout))
        if check == "raise" and result.exit_code != 0:
            raise RuntimeError(f"{error_msg}: {result.stdout[:200]}")
        return result.stdout

    async def write_file(self, path: str | Path, content: str) -> None:
        encoded = base64.b64encode(content.encode()).decode()
        await self.communicate(f"echo {encoded} | base64 -d > {path}", check="raise", error_msg=f"write {path}")

    async def read_file(self, path: str | Path, **_) -> str:
        return await self.communicate(f"cat {path}")


def _extract_task(raw_prompt) -> str:
    """Extract task text from raw_prompt (str or message list)."""
    if isinstance(raw_prompt, str):
        return raw_prompt
    return next(
        (m["content"] for m in raw_prompt if isinstance(m, dict) and m.get("role") == "user"),
        str(raw_prompt),
    )


def _build_task_config(
    *,
    task: str,
    gateway_url: str,
) -> dict:
    """Build the task config passed to run_agent.py via stdin."""
    agent_gateway_url = rewrite_gateway_url(gateway_url)
    step_limit = int(os.environ.get("AGENT_MAX_TURNS", "100"))
    return {
        "task": task,
        "gateway_url": agent_gateway_url,
        "agent": {
            "step_limit": step_limit,
        },
    }


def build_agent_command(
    *,
    config_b64: str,
    conda_env: str = "testbed",
) -> str:
    """Build the command that runs run_agent.py inside the sandbox."""
    conda_prefix = f"/opt/miniconda3/envs/{conda_env}"
    run_agent_env = (
        f"CONDA_DEFAULT_ENV={shlex.quote(conda_env)} "
        f"CONDA_PREFIX={shlex.quote(conda_prefix)} "
        f"PATH={shlex.quote(conda_prefix + '/bin')}:/opt/miniconda3/bin:$PATH "
        "PIP_DISABLE_PIP_VERSION_CHECK=1 "
        "PIP_PROGRESS_BAR=off"
    )
    return (
        "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy; "
        f"env {run_agent_env} sh -c 'echo \"[mini_swe] shell env: CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV "
        'CONDA_PREFIX=$CONDA_PREFIX PATH=$PATH" >&2; '
        'echo "[mini_swe] python=$(command -v python) pip=$(command -v pip)" >&2\' ; '
        f"printf %s {shlex.quote(config_b64)} | base64 -d | "
        f"env {run_agent_env} "
        "/opt/mini-swe-agent/bin/python /opt/mini-swe-agent/bin/run_agent.py"
    )


async def mini_swe_agent_runner(
    *,
    raw_prompt,
    session: SessionHandle,
    sample_index: int,
    tools_kwargs: dict | None = None,
    tool_image: str = DEFAULT_TOOL_IMAGE,
    run_timeout: int = 7200,
    conda_env: str = "testbed",
    sandbox_max_retries: int = 10,
    **kwargs,
) -> None:
    """Run mini-swe-agent inside a sandbox with sidecar tool mount.

    Flow (per attempt):
        1. Create remote sandbox with mini-swe-agent sidecar
        2. Pipe task config to run_agent.py via stdin
        3. Parse agent result from stdout
        4. Evaluate reward in the same sandbox
        5. Post reward_info for the framework reward path

    Connection-death resilience:
        The netentsec proxy drops the CONNECT tunnel after ~30s idle.  If the
        shared yr RPC connection dies mid-run (keepalive failure / SDK
        timeout signature), the attempt is aborted, the yr connection is
        rebuilt (:func:`repair_yr_connection`), and the sample is retried
        from scratch (up to ``SANDBOX_MAX_ATTEMPTS``, default 2).
    """
    tools_kwargs = tools_kwargs or {}
    logger.info("mini_swe_agent_runner called, sample_index=%d", sample_index)

    # Extract task text and sandbox config (image from parquet)
    task = _extract_task(raw_prompt)
    logger.info("task extracted, %d chars", len(task))

    env_config = tools_kwargs.get("env", {})
    image = extract_image(env_config)
    if not image:
        raise ValueError(f"No sandbox image found in tools_kwargs.env for sample {sample_index}")

    # Gateway URL — extract upstream for tunnel
    gateway_url = session.base_url
    if not gateway_url:
        raise ValueError(f"gateway_url is empty for sample {sample_index}")

    upstream = extract_upstream(gateway_url)
    task_config = _build_task_config(
        task=task,
        gateway_url=gateway_url,
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            logger.warning("[sample %d] retrying attempt %d/%d after connection repair", sample_index, attempt, MAX_ATTEMPTS)
        try:
            await _run_attempt(
                raw_prompt=raw_prompt,
                session=session,
                sample_index=sample_index,
                tools_kwargs=tools_kwargs,
                image=image,
                tool_image=tool_image,
                upstream=upstream,
                task_config=task_config,
                run_timeout=run_timeout,
                conda_env=conda_env,
                sandbox_max_retries=sandbox_max_retries,
            )
            return  # success
        except SandboxConnectionDead as e:
            logger.warning(
                "[sample %d] attempt %d/%d failed: sandbox connection dead: %s",
                sample_index, attempt, MAX_ATTEMPTS, e,
            )
            repair_yr_connection(reason=f"sample {sample_index} attempt {attempt} connection dead")
            if attempt >= MAX_ATTEMPTS:
                logger.error("[sample %d] all %d attempts failed (connection death)", sample_index, MAX_ATTEMPTS)
                raise


async def _run_attempt(
    *,
    raw_prompt,
    session: SessionHandle,
    sample_index: int,
    tools_kwargs: dict,
    image: str,
    tool_image: str,
    upstream: str,
    task_config: dict,
    run_timeout: int,
    conda_env: str,
    sandbox_max_retries: int,
) -> None:
    """One sandbox lifecycle attempt.  Raises :class:`SandboxConnectionDead`
    when the shared yr connection is detected dead."""
    env_config = tools_kwargs.get("env", {})

    sandbox = await SandboxClient.create(
        image=image,
        sidecar_image=tool_image,
        upstream=upstream,
        max_retries=int(sandbox_max_retries),
    )
    sandbox_id = sandbox.sandbox_id
    logger.info("Sandbox created (image=%s, sandbox_id=%s)", image, sandbox_id)

    dead_event = asyncio.Event()
    try:
        # Run post_setup_cmd if provided (e.g. git checkout correct commit)
        post_setup_cmd = env_config.get("post_setup_cmd", "")
        if post_setup_cmd:
            logger.info("Running post_setup_cmd (%d chars)...", len(post_setup_cmd))
            r = await sandbox.run(post_setup_cmd, timeout=600)
            if r.exit_code != 0:
                logger.warning("post_setup_cmd failed (rc=%d): %s", r.exit_code, r.stdout[:200])
            else:
                logger.info("post_setup_cmd done")

        # Run agent inside sandbox — pipe config via base64-encoded stdin.
        config_b64 = base64.b64encode(json.dumps(task_config).encode()).decode()
        agent_cmd = build_agent_command(config_b64=config_b64, conda_env=conda_env)
        logger.info("[sample %d] starting agent inside sandbox", sample_index)
        logger.info("Running agent inside sandbox: sample=%d cmd_preview=%s... timeout=%ds",sample_index, agent_cmd[:100], run_timeout,)
        probe = await sandbox.run("printf sandbox_alive", timeout=10)
        logger.info("1 [sample %d] sandbox_probe rc=%d stdout=%r stderr=%r", sample_index, probe.exit_code, probe.stdout, probe.stderr)
        if is_dead_connection(probe):
            logger.warning("[sample %d] initial probe dead — connection lost at session start", sample_index)
            raise SandboxConnectionDead("initial probe dead")

        # Start keepalive background task to prevent proxy idle timeout
        keepalive_stop = asyncio.Event()
        keepalive_task = asyncio.create_task(_keepalive_loop(sandbox, keepalive_stop, dead_event, sample_index))

        t0 = time.perf_counter()
        agent_result = await sandbox.run(agent_cmd, timeout=int(run_timeout))
        elapsed = time.perf_counter() - t0

        # Stop keepalive
        keepalive_stop.set()
        try:
            await asyncio.wait_for(keepalive_task, timeout=5)
        except asyncio.TimeoutError:
            keepalive_task.cancel()

        logger.info(
            "[sample %d] agent process finished: rc=%d (%.1fs)",
            sample_index,
            agent_result.exit_code,
            elapsed,
        )

        # Connection-death check: keepalive reported failure OR agent result
        # carries the SDK timeout signature → retry from scratch.
        if dead_event.is_set():
            logger.warning("[sample %d] connection dead detected via keepalive", sample_index)
            raise SandboxConnectionDead("keepalive failure")
        if is_dead_connection(agent_result):
            logger.warning("[sample %d] connection dead detected via agent result: %s",
                           sample_index, agent_result.stderr[:120])
            raise SandboxConnectionDead("agent result dead")

        logger.info("Running agent inside sandbox: start parse agent result from stdout",)

        # Parse agent result from stdout
        agent_info = _parse_agent_result(agent_result.stdout, sample_index)
        logger.info(
            "[sample %d] agent: exit_status=%s, submission=%d chars",
            sample_index,
            agent_info.get("exit_status"),
            len(agent_info.get("submission", "")),
        )
        logger.info("Running agent inside sandbox: evaluate reward in the same sandbox",)

        # Evaluate reward in the same sandbox (reduced probes: 2 instead of 6)
        metadata, eval_timeout = build_reward_context(tools_kwargs)
        probe = await sandbox.run("printf sandbox_alive", timeout=10)
        logger.info("2 [sample %d] sandbox_probe rc=%d stdout=%r stderr=%r", sample_index, probe.exit_code, probe.stdout, probe.stderr)
        if is_dead_connection(probe):
            raise SandboxConnectionDead("reward pre-probe dead")
        t0 = time.perf_counter()
        reward_env = SandboxEnvForReward(sandbox)
        score, eval_result = await evaluate_in_env(reward_env, metadata, eval_timeout)
        probe = await sandbox.run("printf sandbox_alive", timeout=10)
        logger.info("3 [sample %d] sandbox_probe rc=%d stdout=%r stderr=%r", sample_index, probe.exit_code, probe.stdout, probe.stderr)
        logger.info(
            "[sample %d] reward done: score=%s, resolved=%s (%.1fs)",
            sample_index,
            score,
            eval_result.get("resolved"),
            time.perf_counter() - t0,
        )

        reward_info = {"reward_score": score, **eval_result}
        if not session.reward_info_url:
            raise ValueError(f"reward_info_url is empty for session {session.session_id}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(session.reward_info_url, json={"reward_info": reward_info})
            response.raise_for_status()

    except SandboxConnectionDead:
        raise  # caller handles repair + retry
    except Exception as e:
        logger.warning("Mini-swe-agent runner failed for sample %d (sandbox_id=%s): %s", sample_index, sandbox_id, e)
        raise
    finally:
        try:
            await sandbox.cleanup()
        except Exception:
            pass


def _parse_agent_result(stdout: str, sample_index: int) -> dict:
    """Parse agent result JSON from run_agent.py stdout.

    litellm may print error messages to stdout, polluting the output.
    The last line starting with '{' is the result JSON.
    """
    stdout = stdout.strip()
    if not stdout:
        return {"exit_status": "error", "submission": ""}
    # Try the last line that looks like JSON first
    lines = [ln.strip() for ln in stdout.split("\n") if ln.strip()]
    for line in reversed(lines):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    # Fallback: try entire stdout
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        logger.warning("[sample %d] Failed to parse agent result (full stdout): %s", sample_index, stdout[:1000])
        return {"exit_status": "error", "submission": ""}
