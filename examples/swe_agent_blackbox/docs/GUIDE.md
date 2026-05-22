# Blackbox SWE-Agent Recipe v2 — 执行指南

## 一、环境准备

### 1.1 基础依赖

```bash
# Python 3.12+
pip install -e .  # 安装 uni-agent 及其依赖
cd verl && pip install -e . && cd ..  # 安装 verl

# 额外依赖
pip install pyarrow  # parallel_infer.py 数据集加载
pip install minisweagent   # 仅 mini_swe runner 需要
```

### 1.2 Docker

SWE-bench/SWE-rebench/R2E-Gym 评测需要 Docker 环境。
镜像命名规则：`sweb.eval.x86_64.<repo>__<id>:latest`

### 1.3 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SWE_AGENT_MAX_TURNS` | `100` | Agent 最大交互轮数 |
| `SWE_AGENT_EVAL_TIMEOUT` | `600` | Reward 评测超时（秒） |
| `VERL_LOGGING_LEVEL` | `INFO` | 日志级别 |
| `SWE_AGENT_LOG_TRAJECTORY` | `0` | 设为 `1` 时 gateway 打印每条 request/response 的消息详情（调试用） |
| `NCCL_P2P_DISABLE` | `1` | NCCL 配置 |
| `NCCL_SHM_DISABLE` | `1` | NCCL 配置 |

## 二、数据集准备

支持 parquet 格式的 SWE-bench、SWE-rebench、R2E-Gym 数据集。

数据集需包含字段：`prompt`、`agent_name`、`extra_info.tools_kwargs`（含 `env.image` 和 `reward` 配置）。

```bash
# 数据集路径示例
~/data/swe_agent/swe_bench_verified.parquet
~/data/swe_agent/swe_rebench.parquet
~/data/r2e_gym/r2e_gym.parquet
```

## 三、模型权重

推荐模型：`Qwen3-Coder-30B-A3B-Instruct`

需确保模型路径下 `tokenizer_config.json` 包含 tool-use 相关 chat template，以便自动检测 tool_parser。

## 四、训练

### 4.1 启动训练

```bash
bash examples/swe_agent_blackbox/scripts/run_train.sh
```

### 4.2 关键参数

通过环境变量覆盖默认值：

```bash
# 模型 & 数据
MODEL_PATH=~/models/Qwen3-Coder-30B-A3B-Instruct
TRAIN_DATA=~/data/swe_agent/swe_bench_verified.parquet
VAL_DATA=~/data/swe_agent/swe_bench_verified.parquet

# 硬件
NNODES=1 NGPUS_PER_NODE=8

# 训练
TRAIN_BATCH_SIZE=8
TOTAL_EPOCHS=30
ACTOR_LR=3e-4

# Rollout
ENGINE=vllm
TP=4
N=1

# Agent
MAX_TURNS=100
COMPLETION_TIMEOUT=600
```

### 4.3 训练配置

训练配置文件：`examples/swe_agent_blackbox/config/swe_agent_blackbox.yaml`

关键配置说明：
- `algorithm.adv_estimator=grpo` — 使用 GRPO 算法
- `actor_rollout_ref.rollout.multi_turn.enable=true` — 启用多轮
- `actor_rollout_ref.rollout.custom.agent_framework.framework_class_fqn` — 框架子类
- `actor_rollout_ref.rollout.custom.agent_framework.agent_runner_fqn` — agent runner

## 五、推理

### 5.1 启动推理

```bash
bash examples/swe_agent_blackbox/scripts/run_infer.sh
```

### 5.2 关键参数

```bash
MODEL_PATH=~/models/Qwen3-Coder-30B-A3B-Instruct
DATA_PATH=~/data/swe_agent/swe_bench_verified.parquet
MAX_SAMPLES=10        # -1 表示全部
N=8                   # 每个 sample 的 rollout 数
ENGINE=vllm
TP=4
```

### 5.3 选择 Runner

```bash
# Uniagent runner（默认）
RUNNER=uniagent

# Mini-swe-agent runner
RUNNER=mini_swe
```

通过 `--runner` 参数或 `parallel_infer.py` CLI 选择。

## 六、故障排查

### 6.1 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: verl.tools.utils` | verl import 路径错误 | 确认使用 `verl.tools.tool_registry` |
| `No Docker image found` | 数据集缺少 env.image | 检查 parquet 中 extra_info.tools_kwargs.env.image |
| `minisweagent ImportError` | 未安装 minisweagent | `pip install minisweagent` 或使用 `--runner uniagent` |
| Gateway timeout | agent 运行超时 | 增大 `COMPLETION_TIMEOUT` / `SWE_AGENT_MAX_TURNS` |
| agent_runner signature 不匹配 | 缺少 session_runtime 参数 | 确保框架版本 >= 本次提交，框架会自动传入 |
| NCCL 集群失败 | NCCL 配置问题 | 确认 `NCCL_P2P_DISABLE=1` |

### 6.2 日志

设置 `VERL_LOGGING_LEVEL=DEBUG` 获取详细日志。

### 6.3 Tool Parser 自动检测

`parallel_infer.py` 会根据 `tokenizer_config.json` 的 chat template 自动检测：
- `<function=` + `<parameter=` → `qwen3_coder`
- `"name"` pattern → `hermes`
- 其他 → 无 parser（需手动 `--tool-parser`）
