# RFC: Blackbox Agent Integration — Train Any Agent Without Modifying Its Internals

## Summary

This RFC proposes a generic recipe for integrating **arbitrary third-party agents** into the uni-agent training pipeline as black boxes. The training infrastructure treats the agent as an opaque entity that communicates with the policy model solely through a gateway. The gateway intercepts every LLM call, collects token-level trajectories, and feeds them to the RL trainer — all without any knowledge of the agent's internal control flow (tool orchestration, prompting strategy, state management).

Two reference implementations are provided: one built with uni-agent components, and one wrapping the third-party mini-swe-agent.

Both training and inference share a unified reward path: the runner evaluates reward in-process and passes `reward_info` via `complete_session()`, then `compute_score()` is called through the reward worker (`RewardLoopWorker` → `NaiveRewardManager` → `compute_score`).

## Motivation

Many mature agent frameworks (OpenHands, SWE-agent, mini-swe-agent, etc.) already have well-tuned interaction loops, tool integrations, and prompting strategies. Rewriting them to fit a specific training framework is costly and fragile.

A **blackbox** approach solves this: plug any agent into the training pipeline, and the gateway-transparent architecture ensures the RL trainer can observe and optimize every LLM call while the agent logic stays untouched.

This enables:
- **Zero-cost agent migration**: bring your existing agent, write a thin runner adapter, start training.
- **Agent-agnostic training**: swap between different agent implementations without changing training config.
- **Decoupled iteration**: improve agent logic and training hyperparameters independently.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Training Infrastructure                        │
│                                                                       │
│  ┌─────────┐     ┌──────────────────────────┐    ┌────────────────┐  │
│  │  GRPO /  │────▶│    AgentFramework         │───▶│  Reward Worker │  │
│  │  PPO     │     │  (RolloutAdapter)          │    │ compute_score()│  │
│  └─────────┘     └─────────┬──────────────────┘    └────────────────┘  │
│                             │                               ▲           │
│                      _run_session()                          │           │
│                             │                                │           │
│                 ┌───────────▼───────────┐                    │           │
│                 │   agent_runner()      │                    │           │
│                 │ (you implement this)   │                    │           │
│                 │                        │                    │           │
│                 │  1. Parse prompt       │                    │           │
│                 │  2. Start env (Docker)  │                    │           │
│                 │  3. Run your agent     │                    │           │
│                 │  4. Report completion  │────────────────────┘           │
│                 └────────────────────────┘  (optional: reward_info)      │
│                             │                                           │
│                     agent calls LLM                                     │
│                             │                                           │
│                   ┌─────────▼─────────┐                                 │
│                   │     Gateway       │  intercepts every LLM call,      │
│                   │                   │  collects token trajectories,    │
│                   │                   │  routes to vLLM / SGLang         │
│                   └───────────────────┘                                 │
└──────────────────────────────────────────────────────────────────────┘
```

The gateway is the key enabler: it sits between the agent and the policy model, making every LLM request observable to the trainer. The agent is unaware of this interception — it simply calls an OpenAI-compatible API endpoint. The trainer uses the collected token sequences (prompts, completions, logprobs) to compute policy gradients and update the model.

### Reward Computation

The reward flow is unified across training and inference. The runner evaluates reward in the same Docker environment the agent used, then passes `reward_info` via `complete_session()`. The reward worker calls `compute_score()`, which reads the pre-computed score from `extra_info`.

```
agent_runner() → runs agent → evaluates reward in same env → complete_session(reward_info={...})
                                                                           │
                                                                 SWEAgentFramework._score_trajectories()
                                                                 merges reward_info → extra_info
                                                                           │
                                                                 RewardLoopWorker → NaiveRewardManager
                                                                           │
                                                                 compute_score()
                                                                 reads extra_info["reward_score"]
```

This design avoids spawning a second container for reward evaluation and keeps `compute_score()` as the single reward entry point for both training and inference.

### Key Contracts

| Interface | Who implements | Responsibility |
|-----------|---------------|----------------|
| `agent_runner()` | **You** | Run your agent, evaluate reward, call `complete_session(reward_info={...})` |
| `compute_score()` | **You** | Extract reward score from `extra_info["reward_score"]`, return `{"score": float}` |
| `AgentFramework` subclass | **You** | Override `_score_trajectories` to merge `reward_info` into `extra_info` |

The framework handles everything else: LLM serving, gateway routing, rollout batching, RL advantages, checkpointing.

## Integration Guide

### Step 1: Write an agent runner

The runner is an async function with this signature:

```python
async def my_agent_runner(
    *,
    raw_prompt,                          # str or list[dict] — the task
    session: SessionHandle,              # contains session_id, base_url (gateway endpoint)
    sample_index: int,                   # sample index in the batch
    session_runtime: SessionRuntime,     # call complete_session() when done
    tools_kwargs: dict | None = None,    # per-sample config from dataset
    **kwargs,                            # any extra runner_kwargs from training config
) -> None:
```

**Your runner must:**

1. **Parse the prompt** — `raw_prompt` is a string or chat message list.
2. **Create an environment** — typically a Docker container with the task setup.
3. **Point your agent's LLM at the gateway** — use `session.base_url` as the OpenAI-compatible API endpoint. The agent treats it as a standard LLM API; the gateway handles interception transparently.
4. **Run your agent** — call its existing loop, unmodified.
5. **Report completion** — call `session_runtime.complete_session(session.session_id)` (Mode A) or `session_runtime.complete_session(session.session_id, reward_info={...})` (Mode B).

**Error handling:** If the runner fails, call `complete_session(reward_info={"reward_score": 0.0})` before re-raising, so the framework doesn't hang.

### Step 2: Write compute_score

`compute_score()` reads the pre-computed score that the runner injected via `complete_session(reward_info={...})`.

```python
def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> dict:
    score = 0.0
    if extra_info and "reward_score" in extra_info:
        score = float(extra_info["reward_score"])
    return {"score": score}
```

### Step 3: Subclass AgentFramework

Override `_score_trajectories` to merge `reward_info` from the runner into `extra_info` so `compute_score()` can access it:

```python
from uni_agent.trainer.framework.framework import OpenAICompatibleAgentFramework

class MyFramework(OpenAICompatibleAgentFramework):
    async def _score_trajectories(self, session_trajectories, sample_fields):
        if session_trajectories and session_trajectories[-1].reward_info:
            reward_info = session_trajectories[-1].reward_info
            extra_info = dict(sample_fields.get("extra_info") or {})
            sample_fields = {**sample_fields, "extra_info": {**extra_info, **reward_info}}
        return await super()._score_trajectories(session_trajectories, sample_fields)
```

### Step 4: Write training config

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable: true
    custom:
      agent_framework:
        agent_loop_manager_class: uni_agent.trainer.framework.entry.AgentFrameworkRolloutAdapter
        framework_class_fqn: my_recipe.framework.MyFramework
        agent_runner_fqn: my_recipe.runner.my_agent_runner
        completion_timeout_seconds: 600
        agent_runner_kwargs: {}
```

### Step 5: Prepare dataset

The recipe uses a custom `RLHFDataset` subclass (`SWEBenchDataset`) that injects the verl-standard `data_source` and `reward_model` fields at load time, so the parquet files don't need to include them explicitly.

```yaml
data:
  custom_cls:
    path: pkg://examples.swe_agent_blackbox.dataset
    name: SWEBenchDataset
```

Parquet format with columns:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | `str` or `list[dict]` | Task description / chat messages |
| `agent_name` | `str` | Agent identifier (metadata) |
| `extra_info` | `dict` | Must contain `tools_kwargs` with env and reward config |

Example `extra_info.tools_kwargs`:
```json
{
  "env": {
    "image": "my-task-image:latest",
    "post_setup_cmd": "cd /testbed && git checkout abc123"
  },
  "reward": {
    "name": "swe_bench",
    "metadata": {"instance_id": "repo__id-123", "patch": "diff --git ..."}
  }
}
```

### Step 6: Launch training

```bash
python3 -m verl.trainer.main_ppo \
    --config-name=my_recipe \
    --config-path=my_recipe/config \
    actor_rollout_ref.model.path=/path/to/model \
    data.train_files="['/path/to/train.parquet']" \
    ...
```

## Reference Implementations

### Runner built with uni-agent components

Uses uni-agent's built-in `AgentInteraction` loop with `OpenAICompatibleChatModel` pointing at the gateway. Demonstrates:
- `AgentEnv` + `AgentEnvConfig` for Docker sandbox management
- `ToolsManager` for tool call parsing (configured via `agent_config.yaml`)
- In-process reward evaluation via `evaluate_in_env()`

Files: `examples/swe_agent_blackbox/agent_runner.py`

### Runner wrapping mini-swe-agent (third-party)

Wraps `minisweagent`'s `DefaultAgent` + `DockerEnvironment` + `LitellmModel`. Demonstrates:
- Adapting a sync third-party agent to the async runner interface
- `DockerEnvForReward` adapter bridging sync DockerEnvironment to async reward spec interface
- Running the agent in a thread executor (`loop.run_in_executor`)
- In-process reward evaluation via `evaluate_in_env()`

Files: `examples/swe_agent_blackbox/mini_swe_agent_runner.py`

### Shared reward infrastructure

- `reward.py` — `build_reward_context()`, `compute_score()`, `evaluate_in_env()`
- `framework.py` — `SWEAgentFramework` subclass injecting `reward_info` into `extra_info`
- `dataset.py` — `SWEBenchDataset` injecting `data_source` and `reward_model` fields
- `parallel_infer.py` — Standalone inference runner with `RewardLoopWorker` for `compute_score()`

## Runner Checklist

When integrating a new agent, verify:

| Item | Check |
|------|-------|
| Runner signature | `async def runner(*, raw_prompt, session, sample_index, session_runtime, tools_kwargs=None, **kwargs) -> None` |
| LLM routing | Agent's LLM client points at `session.base_url` (gateway) |
| Completion | `await session_runtime.complete_session(reward_info={...})` called on success **and** failure |
| Reward | Runner evaluates reward in-process, passes `reward_info["reward_score"]`. `compute_score()` reads it from `extra_info` |
| Cleanup | Docker/env resources cleaned up in `finally` block |
| Env config | `DeployConfig` discriminator `type` field present (default: `"local"`) |
| Tool parser | `agent_config.yaml` must specify `tool_parser` (e.g., `qwen3_coder`). Inference CLI defaults to `qwen3_coder` |

## Limitations & Future Work

- **Single-turn reward only**: Current flow computes reward once after the agent finishes. Token-level or step-level reward is not yet supported.
- **No trajectory visibility during training**: The framework sees token sequences through the gateway but not the agent's tool calls or intermediate states. This is by design (black box) but limits some training approaches.
- **Gateway coupling**: The agent's LLM must use an OpenAI-compatible API and point at the gateway. Agents with custom LLM backends need adaptation.
- **Sync agent overhead**: Third-party sync agents (like mini-swe-agent) run via `run_in_executor`, which blocks a thread per session. For high concurrency, async-native agents perform better.
- **Tool parser required**: The gateway must have a tool parser configured (e.g., `qwen3_coder`) to extract structured tool calls from model output. Without it, multi-turn tool calling breaks.
