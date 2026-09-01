from __future__ import annotations

import asyncio
import base64
import json
import re

import pytest

from uni_agent.agents.base import AgentResult, ModelConfig
from uni_agent.agents.codex.agent import CodexAgent, CodexConfig, build_agent_command, parse_agent_result
from uni_agent.sandbox.base import ExecResult


class FakeSandbox:
    def __init__(self, *, stdout: str = "", exit_code: int = 0):
        self.stdout = stdout
        self.exit_code = exit_code
        self.calls: list[dict] = []

    async def exec_shell(self, script, *, timeout=None, workdir=None, env=None):
        self.calls.append({"script": script, "timeout": timeout, "workdir": workdir, "env": env})
        return ExecResult(exit_code=self.exit_code, stdout=self.stdout, stderr="")


def make_agent(base_url="http://gateway:8000/v1", **kwargs):
    kwargs.setdefault("tool_script", "/opt/codex/bin/run_agent.sh")
    kwargs.setdefault("model", ModelConfig(base_url=base_url, model_name="policy"))
    return CodexAgent(CodexConfig(**kwargs))


def decoded_task(command: str) -> str:
    match = re.search(r"printf %s ([^ ]+) \| base64 -d", command)
    assert match
    return base64.b64decode(match.group(1)).decode()


def test_build_agent_command_pipes_prompt_and_runtime_settings():
    command = build_agent_command(
        task_b64=base64.b64encode(b"fix 'this'").decode(),
        tool_script="/opt/codex/bin/run_agent.sh",
        gateway_url="http://127.0.0.1:38197/sessions/s1/v1",
        model_name="policy",
        api_key="key",
    )

    assert decoded_task(command) == "fix 'this'"
    assert "CODEX_API_BASE=http://127.0.0.1:38197/sessions/s1/v1" in command
    assert "CODEX_MODEL=policy" in command
    assert "CODEX_HOME=/root/.agent-home" in command
    assert "CONDA_DEFAULT_ENV=testbed" in command


def test_parse_agent_result_uses_process_completed_status():
    stdout = "\n".join(
        [
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}}),
            json.dumps({"type": "process.completed", "exit_code": 3}),
        ]
    )

    assert parse_agent_result(stdout, 0) == {
        "exit_status": "error",
        "ok": False,
        "content": "done",
        "event_count": 2,
        "error": "agent exited with code 3",
    }


def test_parse_agent_result_timeout():
    assert parse_agent_result("", -1) == {
        "exit_status": "timeout",
        "ok": False,
        "content": "",
        "error": "agent process timed out",
    }


def test_run_returns_agent_result_and_passes_tunnel_url_unchanged():
    stdout = "\n".join(
        [
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "fixed"}}),
            json.dumps({"type": "process.completed", "exit_code": 0}),
        ]
    )
    sandbox = FakeSandbox(stdout=stdout)
    messages = [{"role": "user", "content": "fix the issue"}]

    result = asyncio.run(
        make_agent(base_url="http://127.0.0.1:38197/sessions/abc/v1").run(
            sandbox=sandbox,
            messages=messages,
            workdir="/repo",
        )
    )

    assert isinstance(result, AgentResult)
    assert result.finished is True
    assert result.output["content"] == "fixed"
    assert result.transcript == messages
    assert len(sandbox.calls) == 1
    assert sandbox.calls[0]["workdir"] == "/repo"
    assert "CODEX_API_BASE=http://127.0.0.1:38197/sessions/abc/v1" in sandbox.calls[0]["script"]
    assert "CODEX_PROJECT_DIR=/repo" in sandbox.calls[0]["script"]


def test_run_marks_child_failure_unfinished_even_when_wrapper_exits_zero():
    sandbox = FakeSandbox(stdout=json.dumps({"type": "process.completed", "exit_code": 2}))
    result = asyncio.run(make_agent().run(sandbox=sandbox, messages=[{"role": "user", "content": "task"}]))

    assert result.finished is False
    assert result.info == {"exit_status": "error", "ok": False}


def test_validation_requires_endpoint_model_and_single_user_prompt():
    with pytest.raises(ValueError, match="base_url"):
        asyncio.run(
            CodexAgent(CodexConfig(tool_script="/opt/codex/bin/run_agent.sh")).run(
                sandbox=FakeSandbox(),
                messages=[{"role": "user", "content": "task"}],
            )
        )

    with pytest.raises(ValueError, match="model_name"):
        asyncio.run(
            make_agent().from_config(
                CodexConfig(
                    tool_script="/opt/codex/bin/run_agent.sh",
                    model=ModelConfig(base_url="http://gateway/v1"),
                )
            ).run(sandbox=FakeSandbox(), messages=[{"role": "user", "content": "task"}])
        )

    with pytest.raises(ValueError, match="requires a 'user' message"):
        asyncio.run(make_agent().run(sandbox=FakeSandbox(), messages=[{"role": "system", "content": "system"}]))

    with pytest.raises(ValueError, match="at most 2 messages"):
        asyncio.run(
            make_agent().run(
                sandbox=FakeSandbox(),
                messages=[
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "one"},
                    {"role": "user", "content": "two"},
                ],
            )
        )
