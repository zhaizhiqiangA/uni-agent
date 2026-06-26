# Mini-SWE-Agent In-Sandbox Execution

## Overview

`mini_swe` and `claude_code` both run inside the SWE-bench sandbox through a
sidecar tool image. The external runner creates the sandbox, mounts the selected
tool image, starts the agent process, and evaluates the reward in the same
sandbox.

For `mini_swe`, the agent executes commands through `LocalEnvironment` (local
bash) inside the sandbox and calls the LLM through the gateway URL passed in via
stdin. For `claude_code`, the runner starts the Claude Code CLI from the sidecar
image and points it at the same Anthropic-compatible gateway.

The `mini_swe` tool image uses
[python-build-standalone](https://github.com/astral-sh/python-build-standalone)
to build an isolated Python environment. The Claude Code tool image uses a Node
builder to install the Claude Code npm package. Both images use a minimal
`FROM scratch` final stage, so the sandbox base image does not need to provide
Python, Node, or npm for the sidecar tool runtime.

**Supported runners:**

| runner | Description |
|--------|-------------|
| `uniagent` | Original SWE-agent runner |
| `mini_swe` | mini-swe-agent sidecar runner |
| `claude_code` | Claude Code sidecar runner |

**Supported sandbox types:**

| Type | Description |
|------|-------------|
| OpenYuanRong (`"openyuanrong"`) | Uses `akernel_sdk.Mount` and `sandbox.commands.run()` |

At runtime, the selected runner depends directly on its tool image. The tool
image does not need to be extracted into a host directory ahead of time.

## Architecture

```text
[Rollouter Host: mini_swe_agent_runner]
  |
  |-- _create_sandbox(image, sidecar_image)
  |     `-- openyuanrong: Sandbox(mounts=[Mount(target="/opt/<tool>", ...)])
  |
  |-- sandbox.run("<tool entrypoint>")
  |     `-- [Inside Sandbox]
  |           /opt/mini-swe-agent/bin/python3.12
  |           stdin <- task config JSON (task, gateway_url, agent)
  |           commands run inside the SWE-bench sandbox
  |           stdout -> runner-specific execution result
  |
  |-- parse agent result
  |-- SandboxEnvForReward(sandbox) -> evaluate_in_env()
  `-- POST session.reward_info_url
```

## Prerequisites

1. **OpenYuanRong** - set `OPENYUANRONG_SERVER_ADDRESS` and `OPENYUANRONG_TOKEN`.
2. **Runner tool image** - build the selected tool image and push it to a remote
   registry if the sandbox service cannot access local Docker images.

## 1. Build Tool Image

`mini_swe` is injected into the SWE-bench sandbox as a sidecar tool image. Use
`build_tool.sh` to build it.

| runner | Default tool image | Dockerfile | Sandbox mount path | Image contents | Mirror option |
|--------|--------------------|------------|--------------------|----------------|---------------|
| `mini_swe` | `mini-swe-agent-tool:latest` | `Dockerfile.mini-swe-agent-tool` | `/opt/mini-swe-agent` | Standalone Python 3.12, `mini-swe-agent`, `litellm`, and `run_agent.py` | `--pip-index` / `PIP_INDEX_URL` |

### mini_swe Tool Image

`mini_swe` is the default build target:

```bash
# Use the default PyPI source.
bash examples/swe_agent_blackbox/build_tool.sh

# Use a custom PyPI mirror.
bash examples/swe_agent_blackbox/build_tool.sh --pip-index https://pypi.tuna.tsinghua.edu.cn/simple/

# Build and push to a remote registry.
bash examples/swe_agent_blackbox/build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

The `mini_swe` image uses `python-build-standalone` to build an isolated Python
runtime. The final `FROM scratch` image contains only the files needed under
`/opt/mini-swe-agent`, and it does not depend on the Python version installed in
the sandbox base image.

After pushing the image, point runtime inference at it with `SWE_AGENT_TOOL_IMAGE`:

```bash
SWE_AGENT_TOOL_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest \
RUNNER=mini_swe \
bash examples/swe_agent_blackbox/scripts/run_infer.sh
```

### Combined Build Options

Image tags, mirrors, and registries can be combined:

```bash
bash examples/swe_agent_blackbox/build_tool.sh \
    --tool mini_swe \
    --pip-index https://pypi.tuna.tsinghua.edu.cn/simple/ \
    --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

The build script:

1. Selects the Dockerfile and default image name from `--tool`:
   - `mini_swe` -> `mini-swe-agent-tool:latest`
2. Tags and pushes the image when `--registry` is provided.

The tool image is a sidecar runtime dependency, not a SWE-bench task base image.
The `mini_swe` Python runtime is fully isolated from the sandbox container's
Python.

### Build Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TOOL_IMAGE` | `mini-swe-agent-tool` | Image name; the default changes with `TOOL_KIND` |
| `TOOL_TAG` | `latest` | Image tag |
| `TOOL_VERSION` | `latest` | Tool package version |
| `PIP_INDEX_URL` | unset, use PyPI | pip index URL; equivalent to `--pip-index` |
| `TOOL_KIND` | `mini_swe` | Tool kind |

## 2. Inference With OpenYuanRong Sandbox

### Using run_infer.sh

```bash
cd "$(git rev-parse --show-toplevel)"

RUNNER=mini_swe \
SWE_AGENT_TOOL_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest \
MODEL_PATH=$HOME/models/Qwen3.5-9B \
DATA_PATH=$HOME/data/swe_agent/r2e_gym.parquet \
MAX_SAMPLES=1 \
TP=1 \
bash examples/swe_agent_blackbox/scripts/run_infer.sh
```

### Calling Python Directly

```bash
python examples/swe_agent_blackbox/parallel_infer.py \
    --model-path ~/models/Qwen3.5-9B \
    --data-path ~/data/swe_agent/r2e_gym.parquet \
    --max-samples 1 \
    --runner mini_swe \
    --max-turns 100 \
    --tensor-parallel-size 1
```

## 3. Inference

### Environment Variables

```bash
export OPENYUANRONG_SERVER_ADDRESS="6.2.179.37:8888"
export OPENYUANRONG_TOKEN="<your-token>"
export DEPLOYMENT=openyuanrong
```

### Run mini_swe

```bash
RUNNER=mini_swe \
OPENYUANRONG_SERVER_ADDRESS="6.2.179.37:8888" \
OPENYUANRONG_TOKEN="<token>" \
DEPLOYMENT=openyuanrong \
SWE_AGENT_TOOL_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest \
bash examples/swe_agent_blackbox/scripts/run_infer.sh
```

## 4. Training (Fully Async)

```bash
OPENYUANRONG_SERVER_ADDRESS="6.2.179.37:8888" \
OPENYUANRONG_TOKEN="<token>" \
MODEL_PATH=~/models/Qwen3.5-9B \
bash examples/swe_agent_blackbox/scripts/run_train_megatron_async.sh
```

The training YAML keeps `mini_swe` as the default runner:

```yaml
agent_runner_fqn: examples.swe_agent_blackbox.mini_swe_agent_runner.mini_swe_agent_runner
```

## 5. Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SWE_AGENT_MAX_TURNS` | `100` | Max agent steps |
| `SWE_AGENT_TOOL_IMAGE` | `swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest` | Sidecar tool image |
