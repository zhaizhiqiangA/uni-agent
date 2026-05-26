# Blackbox SWE-Agent Recipe v2 设计文档

> 适配 `uni_agent.trainer` 新框架，参考 `deepeyes_with_gateway` 配置模式

## 一、架构变化

```
旧: YAML → SWERolloutAdapter → 手动创建 GatewayRuntime + Framework
             agent_runner + reward_fn 分别传入

新: YAML → AgentFrameworkRolloutAdapter (通用)
         → from_config 自动创建 GatewayRuntime
         → agent_runner_fqn 加载 runner
         → reward worker + compute_score 读取 reward_info
```

关键变化：
- 不再需要自定义 `trainer_adapter.py`
- `agent_runner` 负责：跑 agent + 计算 reward + 调用 complete_session
- `reward_fn` 消失 → 改为极薄的 `compute_score`，从 `extra_info` 读 reward
- Gateway 依赖从 `uni_agent.blackbox` 迁移到 `uni_agent.trainer`

## 二、文件清单

所有文件位于 `/home/dyp/recipe/uni-agent/`。

### 新建

| 文件 | 说明 |
|------|------|
| `examples/swe_agent_blackbox/framework.py` | SWEAgentFramework 子类，把 reward_info 注入 extra_info |
| `examples/swe_agent_blackbox/agent_runner.py` | uniagent runner：跑 agent + reward + complete_session |
| `examples/swe_agent_blackbox/reward.py` | 极薄 compute_score |

### 重写

| 文件 | 说明 |
|------|------|
| `examples/swe_agent_blackbox/mini_swe_agent_runner.py` | mini-swe-agent runner：同上流程 |

### 修改

| 文件 | 说明 |
|------|------|
| `examples/swe_agent_blackbox/utils.py` | 精简：保留 config loading + deep_merge + reward 评估 |
| `examples/swe_agent_blackbox/config/swe_agent_blackbox.yaml` | 新配置格式 |
| `examples/swe_agent_blackbox/parallel_infer.py` | 迁移到新框架 import |
| `examples/swe_agent_blackbox/dataset.py` | 补充 SWE-rebench 数据源支持 |

### 删除

| 文件 | 说明 |
|------|------|
| `examples/swe_agent_blackbox/trainer_adapter.py` | 不再需要 |

### 不变

| 文件 | 说明 |
|------|------|
| `examples/swe_agent_blackbox/config/agent_config.yaml` | agent 配置不变 |
| `examples/swe_agent_blackbox/config/parallel_infer.yaml` | Hydra 配置不变 |

## 三、数据流

```
1. agent_runner(raw_prompt, session, tools_kwargs)
   ├── 解析 task, 创建 Docker env (AgentEnv / DockerEnvironment)
   ├── 运行 SWE agent (LLM 调用走 gateway)
   ├── 在同一 Docker env 中运行 SWE-bench / R2E-Gym / SWE-rebench 评测
   ├── HTTP POST /sessions/{id}/complete
   │   {reward_info: {reward_score: 1.0, eval_completed: True, ...}}
   └── 关闭 env

2. framework._run_session (基类)
   ├── wait_for_completion (已 COMPLETED, 立即返回)
   └── finalize_session → trajectories 带 reward_info

3. SWEAgentFramework._score_trajectories (子类覆写)
   ├── 合并 reward_info → sample_fields["extra_info"]
   └── 调用父类 → _trajectory_to_reward_dataproto → RewardLoopWorker

4. compute_score(data_source, solution_str, ground_truth, extra_info)
   └── return float(extra_info.get("reward_score", 0.0))

5. 框架设置 reward_score → _write_session_trajectories_to_tq → 训练管线
```

## 四、各文件详细设计

### 4.1 `framework.py` — 子类（5 行核心代码）

```python
from uni_agent.trainer.framework.framework import OpenAICompatibleAgentFramework

class SWEAgentFramework(OpenAICompatibleAgentFramework):
    async def _score_trajectories(self, session_trajectories, sample_fields):
        if session_trajectories and session_trajectories[-1].reward_info:
            reward_info = session_trajectories[-1].reward_info
            extra_info = dict(sample_fields.get("extra_info") or {})
            sample_fields = {**sample_fields, "extra_info": {**extra_info, **reward_info}}
        return await super()._score_trajectories(session_trajectories, sample_fields)
```

- 不修改基类
- 通过 `framework_class_fqn` 在 YAML 中指定

### 4.2 `agent_runner.py` — uniagent runner

```python
async def swe_agent_runner(
    *, raw_prompt, session: SessionHandle, sample_index: int,
    tools_kwargs: dict | None = None,
    agent_config_path: str | None = None,  # YAML agent_runner_kwargs
):
    tools_kwargs = tools_kwargs or {}
    agent_config = load_agent_config(agent_config_path) if agent_config_path else {}

    # 1. 解析 task
    messages = _parse_messages(raw_prompt)

    # 2. 创建 AgentEnv (Docker + swerex)
    env = _create_agent_env(f"swe_bb_{sample_index}", tools_kwargs, agent_config)
    await env.start()

    # 3. 创建 OpenAICompatibleChatModel → session.base_url
    # 4. 创建 ToolsManager + AgentInteraction
    # 5. 运行
    try:
        model = OpenAICompatibleChatModel(base_url=session.base_url, ...)
        interaction = AgentInteraction(env=env, model=model, ...)
        await interaction.run(task=task, messages=messages)

        # 6. 计算 reward
        reward_config = tools_kwargs.get("reward", {})
        metadata = {"data_source": reward_config.get("name"), "reward_model": reward_config.get("metadata")}
        score, eval_result = await evaluate_in_env(env, metadata, eval_timeout)

        # 7. 调用 complete_session
        reward_info = {"reward_score": score, **eval_result}
        await _http_complete_session(session, reward_info)
    except Exception:
        metadata = {..., "_failed": True}
        await _http_complete_session(session, {"reward_score": 0.0})
        raise
    finally:
        await env.close()
```

### 4.3 `mini_swe_agent_runner.py` — mini-swe-agent runner

流程同上，差异在：
- 使用 `DockerEnvironment` + `LitellmModel` + `DefaultAgent`（来自 minisweagent）
- `DockerEnvForReward` 适配器
- 不需要 `agent_config_path`（使用 mini-swe-agent 内置配置）

### 4.4 `utils.py` — 共享工具

保留：
- `load_agent_config(path)` — 加载 YAML 配置
- `_deep_merge(base, override)` — 配置合并
- `_create_agent_env(...)` — AgentEnv 创建

新增：
- `evaluate_in_env(env, metadata, eval_timeout)` — 共享 reward 评测
  - 根据 `data_source` 选择 `SWEBenchRewardSpec` / `R2EGymRewardSpec` / `SWERebenchRewardSpec`
  - 调用 `spec.compute_reward()` → `(resolved, result)`
- `_http_complete_session(session, reward_info)` — HTTP POST 到 gateway
  - URL: `{session.base_url.rsplit("/v1", 1)[0]}/complete`

### 4.5 `reward.py` — compute_score

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if extra_info and "reward_score" in extra_info:
        return float(extra_info["reward_score"])
    return 0.0
```

### 4.6 `config/swe_agent_blackbox.yaml` — 新配置

关键变化：

```yaml
actor_rollout_ref:
  rollout:
    agent:
      # 通用 adapter，不再需要自定义
      agent_loop_manager_class: uni_agent.trainer.framework.entry.AgentFrameworkRolloutAdapter
    custom:
      agent_framework:
        # 框架子类，把 reward_info 注入 extra_info
        framework_class_fqn: examples.swe_agent_blackbox.framework.SWEAgentFramework
        # uniagent runner FQN
        agent_runner_fqn: examples.swe_agent_blackbox.agent_runner.swe_agent_runner
        gateway_count: 1
        completion_timeout_seconds: 600
        max_concurrent_sessions: 2
        agent_runner_kwargs:
          agent_config_path: examples/swe_agent_blackbox/config/agent_config.yaml

reward:
  custom_reward_function:
    path: examples.swe_agent_blackbox.reward
    name: compute_score
```

### 4.7 `dataset.py` — 补充 SWE-rebench

在数据集加载/镜像名映射中增加 `swe_rebench` 类型的支持。

### 4.8 `parallel_infer.py` — 迁移新框架

- Import 从 `uni_agent.blackbox` → `uni_agent.trainer`
- 使用 `SWEAgentFramework` 替代直接构造
- `reward_fn` → 不再需要，reward 在 agent_runner 内完成

## 五、框架依赖关系

```
uni_agent.trainer.framework.types     → SessionHandle, Trajectory, SessionRuntime
uni_agent.trainer.framework.framework → OpenAICompatibleAgentFramework
uni_agent.trainer.framework.entry     → AgentFrameworkRolloutAdapter
uni_agent.trainer.gateway.runtime     → GatewayServingRuntime

uni_agent.reward.registry             → load_reward_spec
uni_agent.reward.swe_bench            → SWEBenchRewardSpec
uni_agent.reward.swe_rebench          → SWERebenchRewardSpec
uni_agent.reward.r2e_gym              → R2EGymRewardSpec

uni_agent.interaction.env             → AgentEnv, AgentEnvConfig
uni_agent.interaction.interaction     → AgentInteraction
uni_agent.interaction.model           → OpenAICompatibleChatModel
uni_agent.interaction.tools_manager   → ToolsManager
```

## 六、实施顺序

1. 创建 `examples/swe_agent_blackbox/` 目录结构
2. 实现 `utils.py` — 共享工具（config loading, reward eval, complete_session）
3. 实现 `framework.py` — 子类
4. 实现 `reward.py` — compute_score
5. 实现 `agent_runner.py` — uniagent runner
6. 实现 `mini_swe_agent_runner.py` — mini-swe-agent runner
7. 修改 `config/swe_agent_blackbox.yaml`
8. 补充 `dataset.py` — SWE-rebench 支持
9. 迁移 `parallel_infer.py`
10. 删除 `trainer_adapter.py`
11. 更新 `.claude.md` 和项目文档
