# Blackbox SWE-Agent Recipe v2 — 验证清单

## 一、静态验证（无需 GPU）

### 1.1 模块 import

| # | 检查项 | 验证方法 | 状态 |
|---|--------|----------|------|
| 1 | `framework.py` — SWEAgentFramework 子类 import 通过 | `python -c "from examples.swe_agent_blackbox.framework import SWEAgentFramework"` | ⬜ |
| 2 | `agent_runner.py` — swe_agent_runner + load_agent_config import 通过 | `python -c "from examples.swe_agent_blackbox.agent_runner import swe_agent_runner, load_agent_config"` | ⬜ |
| 3 | `mini_swe_agent_runner.py` — mini_swe_agent_runner import 通过（需 minisweagent） | `python -c "from examples.swe_agent_blackbox.mini_swe_agent_runner import mini_swe_agent_runner"` | ⬜ |
| 4 | `reward.py` — compute_score + evaluate_in_env import 通过 | `python -c "from examples.swe_agent_blackbox.reward import compute_score, evaluate_in_env"` | ⬜ |
| 5 | `parallel_infer.py` 顶层 import 通过 | `python -c "import examples.swe_agent_blackbox.parallel_infer"` | ⬜ |
| 6 | 训练配置中 FQN 可动态加载 | `python -c "from examples.swe_agent_blackbox.framework import SWEAgentFramework; from examples.swe_agent_blackbox.agent_runner import swe_agent_runner"` | ⬜ |
| 7 | `reward.custom_reward_function` FQN 可加载 | `python -c "from examples.swe_agent_blackbox.reward import compute_score"` | ⬜ |

### 1.2 类型签名兼容性

| # | 检查项 | 详情 | 状态 |
|---|--------|------|------|
| 8 | agent_runner 签名与框架 `_run_session` 调用匹配 | 框架传入 `raw_prompt, session: SessionHandle, sample_index, session_runtime: SessionRuntime, **runner_kwargs`；runner 签名 `(*, raw_prompt, session, sample_index, session_runtime, tools_kwargs=None, agent_config_path=None)` — 通过 `runner_kwargs` 传递 `tools_kwargs` 和 `agent_config_path` | ⬜ |
| 9 | mini_swe_agent_runner 签名与框架调用匹配 | 签名 `(*, raw_prompt, session, sample_index, session_runtime, tools_kwargs=None)` — 兼容 | ⬜ |
| 10 | `SWEAgentFramework._score_trajectories` 签名与父类匹配 | 父类签名 `(self, session_trajectories: list[Trajectory], sample_fields: dict)` — 子类签名一致 | ⬜ |
| 11 | `SessionRuntime.complete_session` 调用正确 | 两个 runner 均调用 `await session_runtime.complete_session(session.session_id, reward_info=...)` — 与 Protocol 定义 `async def complete_session(self, session_id: str, reward_info: dict | None = None)` 一致 | ⬜ |
| 12 | `Trajectory.reward_info` 字段存在 | Trajectory dataclass 必需字段为 `prompt_ids, response_ids, response_mask`，可选字段含 `reward_info: dict[str, Any] = field(default_factory=dict)` — `_score_trajectories` 中 `session_trajectories[-1].reward_info` 可访问 | ⬜ |

### 1.3 Reward 数据流

| # | 检查项 | 详情 | 状态 |
|---|--------|------|------|
| 13 | Reward spec registry key 匹配 | `swe_bench` → `SWEBenchRewardSpec`，`swe_rebench` → `SWERebenchRewardSpec`，`r2e_gym` → `R2EGymRewardSpec`；数据集 `tools_kwargs.reward.name` 需对应这些 key | ⬜ |
| 14 | `evaluate_in_env` 传入 env 接口满足 spec 需求 | `SWEBenchRewardSpec` 用 `communicate` + `write_file` + `read_file`；`SWERebenchRewardSpec` 用 `communicate` + `write_file`；`R2EGymRewardSpec` 用 `communicate` + `write_file`。`AgentEnv` 和 `DockerEnvForReward` 均实现了这三个方法 | ⬜ |
| 15 | `compute_score` 正确读取 reward_score | `float(extra_info["reward_score"])` — 与 `evaluate_in_env` 返回 `score=1.0/0.0` 匹配 | ⬜ |
| 16 | reward_info → extra_info 注入链完整 | runner `complete_session(reward_info={"reward_score": score, **eval_result})` → framework `_score_trajectories` 合并到 `sample_fields["extra_info"]` → `compute_score` 从 `extra_info["reward_score"]` 读取 | ⬜ |

### 1.4 Config 文件

| # | 检查项 | 验证方法 | 状态 |
|---|--------|----------|------|
| 17 | `swe_agent_blackbox.yaml` Hydra 语法正确 | `python -c "from hydra import compose, initialize_config_dir; ..."` | ⬜ |
| 18 | `parallel_infer.yaml` Hydra 语法正确 | 同上 | ⬜ |
| 19 | `agent_config.yaml` YAML 语法正确 + 结构合理 | `python -c "import yaml; yaml.safe_load(open('...'))"` | ⬜ |
| 20 | 训练配置中 `agent_loop_manager_class` FQN 正确 | `uni_agent.trainer.framework.entry.AgentFrameworkRolloutAdapter` | ⬜ |
| 21 | 训练配置中 `framework_class_fqn` FQN 正确 | `examples.swe_agent_blackbox.framework.SWEAgentFramework` | ⬜ |
| 22 | 训练配置中 `agent_runner_fqn` FQN 正确 | `examples.swe_agent_blackbox.agent_runner.swe_agent_runner` | ⬜ |

### 1.5 脚本

| # | 检查项 | 详情 | 状态 |
|---|--------|------|------|
| 23 | `run_train.sh` 参数传递完整 | 环境变量 → Hydra override 参数覆盖关系正确 | ⬜ |
| 24 | `run_infer.sh` 参数传递完整 | CLI 参数与 `parallel_infer.py` argparse 参数一一对应 | ⬜ |
| 25 | `run_infer.sh` 参数传递完整（含 --runner） | `RUNNER` 环境变量传递给 `--runner` 参数，默认 `uniagent` | ⬜ |

### 1.6 逻辑审查

| # | 检查项 | 详情 | 状态 |
|---|--------|------|------|
| 26 | `DockerEnvForReward` 接口覆盖所有 reward spec 需求 | `communicate` + `write_file` + `read_file` 覆盖所有 3 个 spec 的调用，无需 `close` | ⬜ |
| 27 | `_create_agent_env` 正确构建 `AgentEnvConfig` | 从 `agent_config` 取基础 config，用 `tools_kwargs.env` override `deployment.image/command`，其余 shallow merge | ⬜ |
| 28 | `load_swe_dataset` + `_remap_image_to_local` 镜像名映射正确 | 远程 `registry.example.com/path/sweb.eval.x86_64.repo__id:tag` → 本地 `sweb.eval.x86_64.repo__id:latest` | ⬜ |
| 29 | `parallel_infer.py` 中 `_tq_mock` 替换安全 | `_dummy_kv_put` / `_dummy_kv_batch_put` 替换 transfer queue 方法，推理模式不写入训练队列 | ⬜ |
| 30 | `_detect_tool_parser` 逻辑覆盖 Qwen3 / Hermes | `<function=` + `<parameter=` → `qwen3_coder`；`"name"` pattern → `hermes`；其他 → None | ⬜ |
| 31 | `AgentInteraction` 构造参数完整 | `run_id, env, model, tools_manager, messages, action_timeout, timeout_budget, max_turns` 均从 agent_config 读取，有合理默认值 | ⬜ |
| 32 | `AgentInteraction` 的 `timeout_budget` 传 `-1` 是否有效 | agent_runner 默认 `timeout_budget=-1`，需确认框架是否将 `-1` 视为"无限制" | ⬜ |

## 二、集成验证（需 GPU 环境）

### 2.1 训练端到端

| # | 检查项 | 状态 |
|---|--------|------|
| 33 | `run_train.sh` 端到端运行不报错 | ⬜ |
| 34 | GRPO reward 正确传导（train tensorboard 观察 reward 曲线） | ⬜ |
| 35 | Checkpoint 保存/加载正常 | ⬜ |

### 2.2 推理端到端

| # | 检查项 | 状态 |
|---|--------|------|
| 36 | `run_infer.sh` 端到端运行（uniagent runner） | ⬜ |
| 37 | `run_infer.sh` 端到端运行（mini_swe runner） | ⬜ |
| 38 | 推理结果 resolve rate 合理 | ⬜ |

### 2.3 Reward 评测

| # | 检查项 | 状态 |
|---|--------|------|
| 39 | SWE-bench reward spec 评测正确 | ⬜ |
| 40 | SWE-rebench reward spec 评测正确 | ⬜ |
| 41 | R2E-Gym reward spec 评测正确 | ⬜ |

### 2.4 数据集加载

| # | 检查项 | 状态 |
|---|--------|------|
| 42 | SWE-bench parquet 加载 + 镜像映射 | ⬜ |
| 43 | SWE-rebench parquet 加载 + 镜像映射 | ⬜ |
| 44 | R2E-Gym parquet 加载 + 镜像映射 | ⬜ |

## 三、已知待确认项

| # | 项目 | 详情 | 优先级 |
|---|------|------|--------|
| ~~45~~ | ~~`run_infer.sh` 缺少 `--runner` 参数传递~~ | **已修复**：已添加 `RUNNER` 环境变量 + `--runner` 传递 | ~~低~~ |
| ~~46~~ | ~~`AgentInteraction.timeout_budget=-1` 行为~~ | **已修复**：默认值改为 `300`，-1 实际语义是"timeout 即终止"而非"无限" | ~~中~~ |
| 47 | `agent_config.yaml` 中 `max_turns: 3` 过低 | 作为默认值可能不够，训练脚本会 override 为 100，但推理默认可能也需调高 | 低 |
