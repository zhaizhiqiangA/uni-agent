"""Black-box Codex CLI agent executed inside a task sandbox."""

from __future__ import annotations

import base64
import json
import logging
import shlex
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)


def build_agent_command(
    *,
    task_b64: str,
    tool_script: str,
    gateway_url: str,
    model_name: str,
    api_key: str = "EMPTY",
    project_dir: str = "/testbed",
    conda_env: str = "testbed",
    agent_home: str = "/root/.agent-home",
) -> str:
    """Build the shell command that pipes a task into the sidecar entrypoint."""
    conda_prefix = f"/opt/miniconda3/envs/{conda_env}"
    run_env = (
        f"CONDA_DEFAULT_ENV={shlex.quote(conda_env)} "
        f"CONDA_PREFIX={shlex.quote(conda_prefix)} "
        f"PATH={shlex.quote(conda_prefix + '/bin')}:/opt/miniconda3/bin:$PATH "
        f"CODEX_API_BASE={shlex.quote(gateway_url)} "
        f"CODEX_MODEL={shlex.quote(model_name)} "
        f"CODEX_API_KEY={shlex.quote(api_key)} "
        f"CODEX_PROJECT_DIR={shlex.quote(project_dir)} "
        f"CODEX_HOME={shlex.quote(agent_home)} "
        "NO_PROXY='*' no_proxy='*' HTTP_PROXY='' HTTPS_PROXY='' "
        "PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_PROGRESS_BAR=off"
    )
    return (
        "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy; "
        f"printf %s {shlex.quote(task_b64)} | base64 -d | "
        f"env {run_env} bash {shlex.quote(tool_script)}"
    )


def parse_agent_result(stdout: str, exit_code: int) -> dict[str, Any]:
    """Normalize the CLI JSONL stream into the black-box agent result shape."""
    if exit_code == -1:
        return {"exit_status": "timeout", "ok": False, "content": "", "error": "agent process timed out"}

    events: list[dict[str, Any]] = []
    final_content = ""
    errors: list[str] = []
    process_exit_code = exit_code
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        event_type = event.get("type")
        if event_type == "process.completed":
            value = event.get("exit_code")
            if isinstance(value, int):
                process_exit_code = value
        elif event_type in {"error", "turn.failed"}:
            error = event.get("error")
            errors.append(
                str(error.get("message") if isinstance(error, dict) else error or event.get("message") or event)
            )

        item = event.get("item")
        if isinstance(item, dict) and item.get("type") in {"agent_message", "message"}:
            text = item.get("text") or item.get("content") or ""
            if isinstance(text, str):
                final_content = text

    ok = process_exit_code == 0
    result: dict[str, Any] = {
        "exit_status": "ok" if ok else "error",
        "ok": ok,
        "content": final_content,
        "event_count": len(events),
    }
    if not ok:
        result["error"] = errors[-1] if errors else f"agent exited with code {process_exit_code}"
    return result


class CodexConfig(AgentConfig):
    """Launch parameters for the CLI sidecar."""

    name: str = "codex"
    run_timeout: float = Field(default=7200.0, description="Wallclock cap (s) on the agent process.")
    conda_env: str = Field(default="testbed", description="Task repository conda environment.")
    tool_script: str = Field(description="In-sandbox sidecar entrypoint path.")
    agent_home: str = Field(default="/root/.agent-home", description="Per-sandbox CLI state directory.")


@register_agent("codex")
class CodexAgent(Agent):
    """Run the CLI in the sandbox against ``config.model``."""

    config_model = CodexConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
        workdir: str | None = None,
    ) -> AgentResult:
        cfg: CodexConfig = self.config  # type: ignore[assignment]
        if cfg.model.base_url is None:
            raise ValueError("codex: config.model.base_url is not set (the gateway policy endpoint)")
        if not cfg.model.model_name:
            raise ValueError("codex: config.model.model_name is not set")
        task = self._extract_task(messages)
        project_dir = workdir or "/testbed"
        command = build_agent_command(
            task_b64=base64.b64encode(task.encode()).decode(),
            tool_script=cfg.tool_script,
            gateway_url=cfg.model.base_url,
            model_name=cfg.model.model_name,
            api_key=cfg.model.api_key,
            project_dir=project_dir,
            conda_env=cfg.conda_env,
            agent_home=cfg.agent_home,
        )
        result = await sandbox.exec_shell(command, timeout=cfg.run_timeout, workdir=project_dir)
        agent_info = parse_agent_result(result.stdout or "", result.exit_code)
        logger.info(
            "codex: done exit_status=%s content=%d chars rc=%s",
            agent_info.get("exit_status"),
            len(agent_info.get("content", "")),
            result.exit_code,
        )
        return AgentResult(
            output=agent_info,
            transcript=list(messages),
            info={"exit_status": agent_info.get("exit_status"), "ok": agent_info.get("ok", False)},
            finished=agent_info.get("ok") is True,
        )

    @staticmethod
    def _extract_task(messages: list[dict[str, Any]]) -> str:
        if len(messages) > 2:
            raise ValueError(f"codex accepts at most 2 messages (system?, user), got {len(messages)}")
        problem = next((message.get("content") for message in messages if message.get("role") == "user"), None)
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("codex requires a 'user' message (the problem statement)")
        return problem
