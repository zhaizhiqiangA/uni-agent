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
- **compute_score 极薄**：只从 `extra_info["reward_score"]` 读取浮点值

## 三、模块说明

### 3.1 `framework.py` — SWEAgentFramework

继承 `OpenAICompatibleAgentFramework`，覆写 `_score_trajectories`：
- 从 `session_trajectories[-1].reward_info` 提取 reward 信息
- 合并到 `sample_fields["extra_info"]` 中
- 调用父类 `_score_trajectories` 走标准 reward worker 路径

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

### 3.4 `reward.py` — compute_score + evaluate_in_env

- `compute_score(data_source, solution_str, ground_truth, extra_info)` — 从 `extra_info["reward_score"]` 读取分数
- `evaluate_in_env(env, metadata, eval_timeout)` — 统一 reward 评测
  - 根据 `data_source` 自动选择 reward spec（SWE-bench / SWE-rebench / R2E-Gym）
- `_get_reward_spec(data_source)` — reward spec 查找

### 3.5 `parallel_infer.py` — 推理入口

独立推理脚本，使用 `_MockReplayBuffer` 避免训练依赖。
支持 `--runner uniagent|mini_swe` 选择 runner 类型。
内含 `load_swe_dataset` 数据集加载逻辑（支持 SWE-bench、SWE-rebench、R2E-Gym，自动将远程 registry 镜像名映射为本地名）。

## 四、配置文件

### `config/swe_agent_blackbox.yaml` — 训练配置

关键配置项：
- `agent_loop_manager_class`: `uni_agent.trainer.framework.entry.AgentFrameworkRolloutAdapter`
- `framework_class_fqn`: `examples.swe_agent_blackbox.framework.SWEAgentFramework`
- `agent_runner_fqn`: `examples.swe_agent_blackbox.agent_runner.swe_agent_runner`
- `reward.custom_reward_function`: 指向 `compute_score`
- `algorithm.adv_estimator`: `grpo`

### `config/agent_config.yaml` — Agent 配置

定义 env、interaction、tools、tool_parser 等 agent 行为参数。

### `config/parallel_infer.yaml` — Hydra 推理配置

推理模式的 Hydra 配置，由 `parallel_infer.py` 使用。

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
├── parallel_infer.py         # 推理入口（含数据集加载）
├── config/
│   ├── swe_agent_blackbox.yaml  # 训练配置
│   ├── agent_config.yaml        # Agent 配置
│   └── parallel_infer.yaml      # Hydra 推理配置
├── scripts/
│   ├── run_train.sh          # 训练启动脚本
│   └── run_infer.sh          # 推理启动脚本
└── docs/
    ├── DESIGN.md             # 本文档
    ├── GUIDE.md              # 执行指南
    └── VERIFICATION.md       # 验证清单
```
