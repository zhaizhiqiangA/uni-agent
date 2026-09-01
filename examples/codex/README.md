# Codex black-box recipe

This recipe runs the Codex CLI inside the SWE task sandbox as a mounted sidecar.
It follows the same integration boundary as the other black-box agents:

- `uni_agent/agents/codex/` contains only the host-side agent adapter;
- `examples/codex/` contains the sidecar image, sandbox entrypoint, task config,
  and training entry;
- existing framework, gateway, sandbox, task, tool, and logging implementations
  are reused without modification.

## Build the sidecar

```bash
bash examples/codex/build_tool.sh \
  --version 0.147.0 \
  --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

The image is mounted at `/opt/codex`. The sandbox entrypoint is:

```text
/opt/codex/bin/run_agent.sh
```

## Protocol bridge

The CLI uses the OpenAI Responses API, while the community Gateway exposes
Chat Completions. `responses_proxy.py` runs inside the sidecar and translates:

```text
Codex Responses API
  -> local sidecar bridge
  -> existing session /v1/chat/completions
```

The bridge is recipe-local; the community Gateway is not changed.

## Training

```bash
bash examples/codex/run_train.sh
```

The task defaults live in `examples/codex/task_config_codex.yaml`. Runtime model
bindings and the OpenYuanRong reverse tunnel are injected by the existing
`uni_agent.framework.task_runner.run_task` path.
