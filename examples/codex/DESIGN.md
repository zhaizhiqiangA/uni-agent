# Codex recipe design

## Scope

The integration is intentionally limited to the same seams used by the existing
black-box recipes. It does not replace or migrate any community runtime layer.

Changed areas:

```text
examples/codex/
uni_agent/agents/codex/
uni_agent/agents/registry.py
tests/uni_agent/agents/codex/
```

## Execution path

```text
run_task
  -> existing SWE task
  -> existing OpenYuanRong sandbox
  -> CodexAgent
  -> /opt/codex/bin/run_agent.sh
  -> local Responses-to-Chat bridge
  -> existing Gateway /v1/chat/completions
  -> existing reward and trajectory finalization
```

## Responsibilities

- `uni_agent/agents/codex/agent.py` validates the task prompt, passes runtime
  model settings through environment variables, launches the sidecar, and
  normalizes the JSONL result.
- `Dockerfile.codex-tool` packages the native CLI and the recipe-local bridge.
- `responses_proxy.py` converts Responses messages and tool calls to the
  Gateway's existing Chat Completions contract, then converts the completed
  response back to JSON or SSE Responses events.
- `run_agent.sh` creates isolated CLI state under the task repository, starts
  the local bridge, executes the CLI, and emits the child exit status as a final
  JSONL event.
- `task_config_codex.yaml` selects the existing task and sandbox providers and
  mounts the sidecar.
- `run_train.sh` is the existing black-box training entry with recipe-specific
  names and task config only.

## Failure semantics

The sandbox provider may hide stdout when a command exits non-zero. To preserve
structured CLI errors without changing the sandbox implementation,
`run_agent.sh` always exits zero after emitting:

```json
{"type": "process.completed", "exit_code": 1}
```

`CodexAgent` uses that event as the authoritative process status and sets
`finished=False` when it is non-zero.
