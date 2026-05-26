# Blackbox SWE-Agent Recipe v2 — 特性设计

## 一、概述

Blackbox SWE-Agent Recipe 是一个基于 `uni_agent.trainer` 新框架的 SWE-bench 训练 recipe。
核心思路：agent runner 在进程内完成 LLM 推理和 reward 评估，通过 gateway 的 `complete_session` 端点传递 reward 信息，
由框架统一走 reward worker 的 `compute_score` 路径，无需自定义 trainer adapter。

## 二、架构

```
YAML 配置
  └→ AgentFrameworkRolloutAdapter (通用，无需自定义)
       └→ SWEAgentFramework (薄子类，注入 reward_info 到 extra_info)
            ├→ GatewayServingRuntime (LLM 推理路由)
            ├→ agent_runner (uniagent / mini-swe-agent)
            │    ├→ 运行 agent (LLM 调用走 gateway)
            │    ├→ evaluate_in_env (reward 评估)
            │    └→ HTTP POST /complete (传递 reward_info)
            └→ reward worker → compute_score (从 extra_info 读取 reward_score)
```

关键设计决策：
- **无自定义 trainer adapter**：使用通用 `AgentFrameworkRolloutAdapter`
- **reward 在 agent_runner 内计算**：agent 跑完后在同一 Docker 环境中执行评测
- **reward_info 注入 extra_info**：`SWEAgentFramework` 子类覆写 `_score_trajectories`，合并 reward_info 到 sample_fields
- **compute_score 统一 reward 路径**：训练和推理均通过 `RewardLoopWorker` → `NaiveRewardManager` → `compute_score` 计算最终 reward

## 三、模块说明

### 3.1 `framework.py` — SWEAgentFramework

继承 `OpenAICompatibleAgentFramework`，覆写 `_score_trajectories`：
- 从 `session_trajectories[-1].reward_info` 提取 reward 信息
- 合并到 `sample_fields["extra_info"]` 中
- 调用父类 `_score_trajectories` 走标准 reward worker 路径

**统一 reward 路径**：训练和推理均通过 `RewardLoopWorker` → `NaiveRewardManager` → `compute_score` 计算 reward。基类不再跳过 in-process reward，`SWEAgentFramework._score_trajectories` 将 runner 传来的 `reward_info` 合并到 `extra_info`，由 `compute_score` 读取 `extra_info["reward_score"]` 产生最终分数。推理模式由 `parallel_infer.py` 创建 `RewardLoopWorker` Ray actor 实现。

### 3.2 `agent_runner.py` — Uniagent Runner

使用白盒交互组件：
- `AgentEnv` — Docker 环境管理
- `OpenAICompatibleChatModel` — LLM 调用（指向 gateway）
- `ToolsManager` — 工具管理（parser 从 agent_config 读取）
- `AgentInteraction` — 交互循环

流程：解析 task → 创建 env → 运行 agent → evaluate_in_env → session_runtime.complete_session → 关闭 env

框架通过 `_run_session` 将 `session_runtime` 传递给 agent_runner，runner 直接调用 `session_runtime.complete_session(session_id, reward_info=...)` 通知完成，无需关心 gateway HTTP 细节。

私有 helper：`load_agent_config`, `_create_agent_env`

**R2E 镜像兼容**：R2E-Gym 镜像的 swerex 安装在 `/opt/swerex-venv/bin/python3`（Python 3.11 venv），而非默认的 `python3`（Python 3.8）。`_create_agent_env` 会对 R2E 镜像（image 名含 `r2e`）自动覆盖 command 为 `/opt/swerex-venv/bin/python3 -m swerex.server`。

### 3.3 `mini_swe_agent_runner.py` — Mini-SWE-Agent Runner

使用第三方 minisweagent 组件：
- `DockerEnvironment` — Docker 环境
- `LitellmModel` — LLM 调用（指向 gateway）
- `DefaultAgent` — agent 循环

包含 `DockerEnvForReward` 适配器（sync → async 接口适配）

**`_FixedCmdDockerEnvironment`**：DockerEnvironment 子类，根据镜像 ENTRYPOINT 动态调整容器 CMD 格式。sweb 镜像 ENTRYPOINT 为 `/bin/bash` 时使用 `-lc "sleep <timeout>"`（走 bash 内建 sleep），其他镜像（如 R2E-Gym 使用 nvidia_entrypoint.sh）使用原始 `sleep <timeout>`。

**`SWE_AGENT_MAX_TURNS`**：通过环境变量控制 `DefaultAgent` 的 `step_limit`，限制 agent 最大迭代步数。

**错误处理**：runner 异常时调用 `complete_session(reward=0)` 后 re-raise，framework 将 session 标记为 failed。上下文溢出等不可恢复错误会快速传播，不产生无意义的重试。

### 3.4 `reward.py` — compute_score + evaluate_in_env

- `compute_score(data_source, solution_str, ground_truth, extra_info)` — 从 `extra_info["reward_score"]` 读取分数
- `evaluate_in_env(env, metadata, eval_timeout)` — 统一 reward 评测
  - 根据 `data_source` 自动选择 reward spec（SWE-bench / SWE-rebench / R2E-Gym）
- `_get_reward_spec(data_source)` — reward spec 查找

### 3.5 Gateway 消息归一化与错误处理

**`_normalize_tool_call_arguments`**：gateway 在处理请求时自动将 `tool_calls[].function.arguments` 从 JSON 字符串解析为 dict。OpenAI API 规范定义 arguments 为 JSON 字符串，但部分 chat template（如 Qwen）通过 `|items` 迭代需要 dict 对象。此归一化在 `_normalize_message` 中执行，确保所有经过 gateway 的消息格式一致，同时修复了前缀匹配（`_is_request_context_prefix`）和模板编码的问题。

**`_backend.generate()` 异常映射**：gateway 捕获 vLLM 后端的所有异常，按 OpenAI API 规范映射为 HTTP 状态码：`ValueError`（如上下文长度超限）→ 400（`invalid_request_error`），其他异常 → 500（`internal_error`）。这避免了 agent 侧（litellm/tenacity）对不可恢复错误的盲目重试。

### 3.6 `parallel_infer.py` — 推理入口

独立推理脚本，使用 `_MockReplayBuffer` 避免训练依赖。
支持 `--runner uniagent|mini_swe` 选择 runner 类型。
内含 `load_swe_dataset` 数据集加载逻辑（支持 SWE-bench、SWE-rebench、R2E-Gym，自动将远程 registry 镜像名映射为本地名）。

**异常容错**：`generate_sequences` 抛 `RuntimeError`（如所有 rollout 失败）时捕获异常并输出 `Resolved 0 / N` 统计，不会导致进程崩溃。

**推理模式 reward 传播**：推理模式下无 `reward_loop_worker_handles`，框架在 `_run_session` 中直接从 `trajectory.reward_info`（由 agent_runner 通过 `complete_session` 传入）提取 `reward_score` 并设置到 trajectory 上。`parallel_infer.py` 从 TQ store 读取 `rm_scores[-1, -1]`（最后一个 trajectory 的最后一个 token 位置）获取 score。

### 3.7 `dataset.py` — SWEBenchDataset

继承 `RLHFDataset`，覆写 `__getitem__` 注入 verl 标准 reward 字段：
- `data_source`：从 `extra_info.tools_kwargs.reward.name` 提取（如 `swe_bench`、`r2e_gym`）
- `reward_model`：`{"ground_truth": {}}`（占位，实际 reward 由 agent_runner 计算）

这些字段是 `NaiveRewardManager.run_single()` 的硬性要求（`non_tensor_batch["data_source"]` 和 `non_tensor_batch["reward_model"]["ground_truth"]`），缺失会导致 KeyError。

训练时通过 `swe_agent_blackbox.yaml` 的 `data.custom_cls` 配置指定使用此类；推理时 `parallel_infer.py` 的 `_inject_reward_fields` 实现相同逻辑。

## 四、配置文件

### `config/swe_agent_blackbox.yaml` — 训练配置

关键配置项：
- `agent_loop_manager_class`: `uni_agent.trainer.framework.entry.AgentFrameworkRolloutAdapter`
- `framework_class_fqn`: `examples.swe_agent_blackbox.framework.SWEAgentFramework`
- `agent_runner_fqn`: `examples.swe_agent_blackbox.agent_runner.swe_agent_runner`
- `reward.custom_reward_function`: 指向 `compute_score`
- `algorithm.adv_estimator`: `grpo`
- `data.custom_cls`: 指向 `SWEBenchDataset`，注入 `data_source` 和 `reward_model` 字段

### `config/agent_config.yaml` — Agent 配置

定义 env、interaction、tools、tool_parser 等 agent 行为参数。`tool_parser` 字段指定工具调用解析器（默认 `qwen3_coder`）。

### `config/parallel_infer.yaml` — Hydra 推理配置

推理模式的 Hydra 配置，由 `parallel_infer.py` 使用。

### `config/swe_agent_blackbox_megatron.yaml` — Megatron 训练配置

基于 `swe_agent_blackbox.yaml`，切换为 Megatron 后端：
- `defaults` 中 `override model_engine: megatron`，自动加载 megatron_actor/critic/ref/engine 配置
- actor 使用 `megatron` 引擎（TP=4, sequence_parallel, distributed_optimizer, bfloat16, core_attn recompute）
- 去掉 FSDP 的 `fsdp_config`，改用 `megatron` 配置块
- 其余部分（rollout、data、algorithm、reward、agent_framework）与 FSDP 版一致

### `scripts/run_train_megatron.sh` — Megatron 训练启动脚本

基于 `run_train.sh`，新增 `MEGATRON_TP`/`MEGATRON_PP`/`MEGATRON_CP` 环境变量控制 Megatron 并行度。
默认 `--config-name=swe_agent_blackbox_megatron`，通过命令行 override 传入 `actor.megatron.tensor_model_parallel_size` 等参数。

## 五、依赖关系

```
uni_agent.trainer.framework.types     → SessionHandle, Trajectory
uni_agent.trainer.framework.framework → OpenAICompatibleAgentFramework
uni_agent.trainer.framework.entry     → AgentFrameworkRolloutAdapter
uni_agent.trainer.gateway.runtime     → GatewayServingRuntime

uni_agent.reward.registry             → REWARD_SPEC_REGISTRY
uni_agent.reward.swe_bench            → SWEBenchRewardSpec
uni_agent.reward.swe_rebench          → SWERebenchRewardSpec
uni_agent.reward.r2e_gym              → R2EGymRewardSpec

uni_agent.interaction.env             → AgentEnv, AgentEnvConfig
uni_agent.interaction.interaction     → AgentInteraction
uni_agent.interaction.model           → OpenAICompatibleChatModel
uni_agent.interaction.tools_manager   → ToolsManager, ToolsManagerConfig

verl.tools.tool_registry              → initialize_tools_from_config
verl.utils.dataset.rl_dataset         → RLHFDataset (SWEBenchDataset 的基类)
verl.experimental.reward_loop         → RewardLoopWorker, RewardLoopManager
```

## 六、数据流

```
1. agent_runner(raw_prompt, session, tools_kwargs)
   ├── 创建 Docker env
   ├── 运行 agent (LLM 走 gateway)
   ├── evaluate_in_env → reward spec → (resolved, result)
   ├── POST /complete {reward_info: {reward_score, ...}}
   └── 关闭 env

2. SWEAgentFramework._score_trajectories
   ├── 合并 reward_info → sample_fields["extra_info"]
   └── 父类 → _trajectory_to_reward_dataproto → RewardLoopWorker

3. compute_score(data_source, solution_str, ground_truth, extra_info)
   └── return float(extra_info["reward_score"])

4. 框架 → _write_session_trajectories_to_tq → 训练管线
```

## 七、目录结构

```
examples/swe_agent_blackbox/
├── __init__.py
├── framework.py              # SWEAgentFramework 子类
├── agent_runner.py           # Uniagent runner
├── mini_swe_agent_runner.py  # Mini-swe-agent runner
├── reward.py                 # compute_score + evaluate_in_env
├── dataset.py                # SWEBenchDataset (注入 verl 标准字段)
├── parallel_infer.py         # 推理入口（含数据集加载）
├── config/
│   ├── swe_agent_blackbox.yaml  # FSDP 训练配置
│   ├── swe_agent_blackbox_megatron.yaml  # Megatron 训练配置
│   ├── agent_config.yaml        # Agent 配置
│   └── parallel_infer.yaml      # Hydra 推理配置
├── scripts/
│   ├── run_train.sh          # FSDP 训练启动脚本
│   ├── run_train_megatron.sh # Megatron 训练启动脚本
│   └── run_infer.sh          # 推理启动脚本
└── docs/
    ├── DESIGN.md             # 本文档
    ├── GUIDE.md              # 执行指南
    └── VERIFICATION.md       # 验证清单
```
