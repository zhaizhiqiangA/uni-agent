"""Mini-swe-agent runner for the blackbox SWE-agent recipe.

Uses third-party minisweagent components (DefaultAgent, DockerEnvironment,
LitellmModel) with gateway-based LLM routing. Computes reward in-process
and passes it via the gateway's complete_session endpoint.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Any

from uni_agent.trainer.framework.types import SessionHandle, SessionRuntime

from examples.swe_agent_blackbox.reward import build_reward_context, evaluate_in_env

logger = logging.getLogger(__name__)

try:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.models.litellm_model import LitellmModel

    _SWEBENCH_CONFIG = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml"))
except ImportError:
    _SWEBENCH_CONFIG = None


# =====================================================================
# DockerEnvForReward: adapts sync DockerEnvironment to async interface
# =====================================================================


class DockerEnvForReward:
    """Adapts minisweagent's sync DockerEnvironment to async interface for reward specs."""

    def __init__(self, docker_env):
        self._env = docker_env

    async def communicate(self, input: str, timeout=60, check="ignore", error_msg="Command failed") -> str:
        result = await asyncio.to_thread(self._env.execute, {"command": input}, timeout=int(timeout))
        output, rc = result.get("output", ""), result.get("returncode", 0)
        if check == "raise" and rc != 0:
            raise RuntimeError(f"{error_msg}: {output[:200]}")
        return output

    async def write_file(self, path: str | Path, content: str) -> None:
        encoded = base64.b64encode(content.encode()).decode()
        await self.communicate(f"echo {encoded} | base64 -d > {path}", check="raise", error_msg=f"write {path}")

    async def read_file(self, path: str | Path, **_) -> str:
        return await self.communicate(f"cat {path}")


# =====================================================================
# Agent runner
# =====================================================================


async def mini_swe_agent_runner(
    *,
    raw_prompt,
    session: SessionHandle,
    sample_index: int,
    session_runtime: SessionRuntime,
    tools_kwargs: dict | None = None,
) -> None:
    """Run mini-swe-agent's DefaultAgent through the gateway with in-process reward."""
    if _SWEBENCH_CONFIG is None:
        raise ImportError("minisweagent is required for mini_swe_agent_runner")

    tools_kwargs = tools_kwargs or {}
    logger.info("mini_swe_agent_runner called, sample_index=%d", sample_index)

    # 1. Extract task text
    task = raw_prompt if isinstance(raw_prompt, str) else next(
        (m["content"] for m in raw_prompt if isinstance(m, dict) and m.get("role") == "user"),
        str(raw_prompt),
    )
    logger.info("task extracted, %d chars", len(task))

    # 2. Create DockerEnvironment
    env_config = tools_kwargs.get("env", {})
    image = env_config.get("image", "")
    if not image:
        raise ValueError(f"No Docker image found in tools_kwargs.env for sample {sample_index}")

    env_cfg = dict(_SWEBENCH_CONFIG.get("environment", {}))
    env_cfg.pop("environment_class", None)
    env_cfg["image"] = image
    env_cfg["container_timeout"] = "2h"
    env_cfg.setdefault("env", {})["GIT_PAGER"] = "cat"
    docker_env = DockerEnvironment(**env_cfg)
    logger.info("Docker container started: %s", docker_env.container_id[:12])

    # 2b. Run post_setup_cmd if provided
    post_setup_cmd = env_config.get("post_setup_cmd", "")
    if post_setup_cmd:
        logger.info("Running post_setup_cmd (%d chars)...", len(post_setup_cmd))
        result = docker_env.execute({"command": post_setup_cmd}, timeout=120)
        rc = result.get("returncode", 0)
        if rc != 0:
            logger.warning("post_setup_cmd failed (rc=%d): %s", rc, result.get("output", "")[:200])
        else:
            logger.info("post_setup_cmd done")

    # 3. Prepare metadata
    metadata, eval_timeout = build_reward_context(tools_kwargs)

    try:
        # 4. Create LitellmModel pointing at gateway
        model_cfg = dict(_SWEBENCH_CONFIG.get("model", {}))
        model_cfg.update({
            "model_name": "openai/default",
            "model_kwargs": {
                "api_base": session.base_url,
                "api_key": "not-needed",
                "drop_params": True,
            },
            "cost_tracking": "ignore_errors",
        })
        model = LitellmModel(**model_cfg)

        # 5. Create DefaultAgent
        agent_cfg = dict(_SWEBENCH_CONFIG.get("agent", {}))
        agent_cfg["step_limit"] = int(os.environ.get("SWE_AGENT_MAX_TURNS", str(agent_cfg.get("step_limit", 250))))
        agent_cfg["cost_limit"] = 0
        agent = DefaultAgent(model, docker_env, **agent_cfg)

        # 6. Run agent in thread (DefaultAgent is synchronous)
        logger.info("starting DefaultAgent.run()...")
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, lambda: agent.run(task=task))

        exit_status = info.get("exit_status", "unknown")
        submission = info.get("submission", "")
        logger.info(
            "agent finished: exit_status=%s, steps=%d, submission=%d chars",
            exit_status, agent.n_calls, len(submission),
        )

        # 7. Evaluate reward
        reward_env = DockerEnvForReward(docker_env)
        score, eval_result = await evaluate_in_env(reward_env, metadata, eval_timeout)
        logger.info("reward: score=%s, resolved=%s", score, eval_result.get("resolved"))

        # 8. Signal completion with reward_info
        reward_info = {"reward_score": score, **eval_result}
        await session_runtime.complete_session(session.session_id, reward_info=reward_info)

    except Exception as e:
        logger.warning("Mini-swe-agent runner failed for sample %d: %s", sample_index, e)
        try:
            await session_runtime.complete_session(session.session_id, reward_info={"reward_score": 0.0})
        except Exception:
            pass
        try:
            docker_env.cleanup()
        except Exception:
            pass
        raise
