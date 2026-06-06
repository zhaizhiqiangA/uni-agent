# Blackbox SWE-Agent 全异步训练接入设计

> 日期: 2026-06-06
> 状态: Draft
> 前置: verl 已包含 PR #6628 (TQ 全异步路径)

## 1. 目标

将 blackbox SWE-agent (Gateway + 子进程 agent_runner) 接入 verl 的 TQ 全异步训练路径
(`FullyAsyncRollouterTQ` + `FullyAsyncTrainerTQ`)。

沿用 sync 路径 (`main_ppo_sync`) 同样的模式：在 YAML 中通过 `agent_loop_manager_class`
注册 adapter 类，由 verl 框架从配置读取并实例化。

## 2. 架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TQ 全异步路径 (blackbox)                           │
│                                                                      │
│  FullyAsyncTaskRunner (verl 原生，不动)                               │
│    │                                                                 │
│    ├─ FullyAsyncRollouterTQ (verl 原生，仅加 ~5 行配置读取)          │
│    │     │                                                           │
│    │     ├─ FullyAsyncLLMServerManager                               │
│    │     │     └─ FullyAsyncLLMServerClient (partial rollout resume) │
│    │     │                                                           │
│    │     ├─ agent_loop_manager_class ← 从 YAML 配置读取 ──────┐     │
│    │     │     └─ FullyAsyncAgentFrameworkRolloutAdapter ←────┘     │
│    │     │           ├─ GatewayServingRuntime (注入上面的 client)    │
│    │     │           ├─ SWEAgentFramework → agent_runner (子进程)    │
│    │     │           └─ generate_sequences_single() → 写 TQ        │
│    │     │                                                           │
│    │     └─ _feed_samples() → acquire_slot → pending_queue          │
│    │           _processor_worker → generate_sequences_single()       │
│    │             → release_slot                                      │
│    │                                                                 │
│    └─ FullyAsyncTrainerTQ (verl 原生，不动)                           │
│          └─ PPOTrainer 多重继承 → 从 TQ 读数据 → train              │
│                                                                      │
│  数据流:                                                              │
│    dataloader → acquire_slot → generate → Gateway → TQ               │
│      → RB.poll → trainer.sample → PPO pipeline → update_weights     │
└──────────────────────────────────────────────────────────────────────┘
```

## 3. 改动范围

### 3.1 verl 侧 (~10 行)

两处 `__init__` 中 `agent_loop_manager_class` 的赋值，改为从配置读取
（与 `main_ppo_sync.py:737-741` 完全一致的 pattern）：

**文件 1**: `verl/experimental/fully_async_policy/fully_async_rollouter.py`
```python
# __init__ 中，改前 (line 485):
self.agent_loop_manager_class = FullyAsyncAgentLoopManager

# 改后:
_manager_fqn = config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
if _manager_fqn:
    from verl.utils.import_utils import load_class_from_fqn
    self.agent_loop_manager_class = load_class_from_fqn(_manager_fqn)
# else: 保持默认 FullyAsyncAgentLoopManager（已在上面赋值）
```

**文件 2**: `verl/experimental/fully_async_policy/fully_async_rollouter_tq.py`
```python
# __init__ 中，改前 (line 119):
self.agent_loop_manager_class = FullyAsyncAgentLoopManagerTQ

# 改后（在 super().__init__() 之后再覆盖）:
_manager_fqn = config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
if _manager_fqn:
    from verl.utils.import_utils import load_class_from_fqn
    self.agent_loop_manager_class = load_class_from_fqn(_manager_fqn)
# else: 保持默认 FullyAsyncAgentLoopManagerTQ（已在 super().__init__() 中赋值）
```

### 3.2 uni-agent 侧

| 文件 | 类型 | 说明 |
|------|------|------|
| `uni_agent/trainer/framework/async_entry.py` | 新增 (~50 行) | `FullyAsyncAgentFrameworkRolloutAdapter` |
| `uni_agent/trainer/framework/framework.py` | 修改 (~3 行) | `replay_buffer` 改为可选 |
| `examples/.../swe_agent_blackbox_megatron_async.yaml` | 新增 | 全异步 YAML 配置 |
| `examples/.../run_train_megatron_async.sh` | 新增 | 启动脚本 |

## 4. 详细设计

### 4.1 `FullyAsyncAgentFrameworkRolloutAdapter`

位于 `uni_agent/trainer/framework/async_entry.py`。

```python
class FullyAsyncAgentFrameworkRolloutAdapter:
    """TQ 全异步路径的 blackbox agent adapter。

    用法: 在 YAML 中指定
      actor_rollout_ref.rollout.agent.agent_loop_manager_class:
        uni_agent.trainer.framework.async_entry.FullyAsyncAgentFrameworkRolloutAdapter

    接口合同 (与 FullyAsyncAgentLoopManagerTQ 一致):
    - create(config, llm_client, reward_loop_worker_handles, teacher_client)
    - generate_sequences_single(prompts) -> None (写 TQ)
    """

    def __init__(self):
        self.framework = None

    @classmethod
    @auto_await
    async def create(
        cls, *, config, llm_client, teacher_client=None,
        reward_loop_worker_handles=None, **_,
    ):
        del teacher_client
        # llm_client 是 FullyAsyncLLMServerClient (由 _init_async_rollout_manager 传入)
        # Gateway 接受外部注入的 client → 自动获得 partial rollout resume
        framework = await build_agent_framework(
            config=config,
            llm_client=llm_client,
            replay_buffer=None,  # TQ 全异步路径不需要本地 RB
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        instance = cls()
        instance.framework = framework
        return instance

    async def generate_sequences_single(self, prompts):
        """TQ 异步路径入口。内部写 TQ，返回 None。"""
        return await self.framework.generate_sequences(prompts)

    async def generate_sequences(self, prompts):
        """兼容 validate 路径。"""
        return await self.framework.generate_sequences(prompts)
```

**关键设计点**:

- **Partial Rollout Resume 透明化**: `llm_client` 是 `FullyAsyncLLMServerClient`
  （由 `_init_async_rollout_manager()` 传入），`GatewayServingRuntime` 接受外部注入的
  client，因此 Gateway 内部所有 LLM 调用自动获得 partial rollout resume。当 trainer 做
  weight sync 打断生成时，client 自动 resume，对 agent_runner 完全透明。

- **不需要本地 ReplayBuffer**: TQ 全异步路径下，RB 状态由 rollouter 的
  `acquire_slot()` / `release_slot()` 管理，不需要 framework 层的 `.add()` 调用。
  需配合 `framework.py` 的小改动（4.2 节）。

- **不创建 AgentLoopWorker**: 直接使用 Gateway session 运行 agent_runner 子进程，
  复用 `OpenAICompatibleAgentFramework` 的 TQ 写入逻辑。

### 4.2 `framework.py` 改动

`OpenAICompatibleAgentFramework.generate_sequences()` 中两处修改：

```python
# 改前 (line 211-213):
if self._replay_buffer is None:
    raise RuntimeError("OpenAICompatibleAgentFramework requires replay_buffer...")
# ...
self._replay_buffer.add(
    partition_id,
    {str(uid): {"global_steps": global_steps, "status": "running"} for uid in uid_values},
)

# 改后:
if self._rollout_config is None:
    raise RuntimeError("OpenAICompatibleAgentFramework requires rollout_config...")
# ...
if self._replay_buffer is not None:
    self._replay_buffer.add(
        partition_id,
        {str(uid): {"global_steps": global_steps, "status": "running"} for uid in uid_values},
    )
```

sync 路径不受影响（仍传 replay_buffer 非 None）。

### 4.3 YAML 配置

`swe_agent_blackbox_megatron_async.yaml` 关键字段：

```yaml
actor_rollout_ref:
  hybrid_engine: false
  nccl_timeout: 9600

  rollout:
    mode: async
    calculate_log_probs: true
    enable_sleep_mode: true
    free_cache_engine: true
    enable_chunked_prefill: true
    checkpoint_engine:
      backend: nccl

    multi_turn:
      enable: true
      max_assistant_turns: 1
      max_parallel_calls: 1
      format: qwen3_coder

    agent:
      num_workers: 8
      agent_loop_manager_class: uni_agent.trainer.framework.async_entry.FullyAsyncAgentFrameworkRolloutAdapter

    custom:
      agent_framework:
        framework_class_fqn: examples.swe_agent_blackbox.framework.SWEAgentFramework
        agent_runner_fqn: examples.swe_agent_blackbox.agent_runner.swe_agent_runner
        gateway_count: 1
        completion_timeout_seconds: 600
        max_concurrent_sessions: 32
        agent_runner_kwargs:
          agent_config_path: examples/swe_agent_blackbox/config/agent_config.yaml

  actor:
    use_rollout_log_probs: true
    use_dynamic_bsz: true
    ppo_mini_batch_size: 16
    megatron:
      param_offload: true
      grad_offload: true
      optimizer_offload: true
      use_mbridge: true

  ref:
    megatron:
      param_offload: false

data:
  train_batch_size: 0
  gen_batch_size: 1
  return_raw_chat: true
  trust_remote_code: true
  custom_cls:
    path: pkg://examples.swe_agent_blackbox.dataset
    name: SWEBenchDataset

algorithm:
  adv_estimator: grpo
  rollout_correction:
    bypass_mode: true

async_training:
  use_trainer_do_validate: false
  staleness_threshold: 1.0
  trigger_parameter_sync_step: 4
  require_batches: 1
  partial_rollout: true

transfer_queue:
  enable: true

rollout:
  nnodes: 1
  n_gpus_per_node: 8
  total_rollout_steps: 100000
```

### 4.4 启动脚本

入口与现有 `run_train_megatron.sh` 一致，使用 verl 原生的 `fully_async_main`：

```bash
ray job submit \
    --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "'$(pwd)'"}' \
    -- python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name=swe_agent_blackbox_megatron_async \
    --config-path="$(pwd)/examples/swe_agent_blackbox/config" \
    hydra.searchpath=[pkg://verl.trainer.config] \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    data.train_files="['${TRAIN_DATA}']" \
    data.val_files="['${VAL_DATA}']" \
    # ... 其余环境变量覆盖同 run_train_megatron.sh
```

## 5. 数据流详解

### 5.1 训练步

```
1. Rollouter._feed_samples():
   - dataloader 迭代 → batch_dict (bsz=1)
   - 注入 uid, __rollout_n__, global_steps
   - RB.acquire_slot() (双层流控)
   - put to pending_queue

2. Rollouter._processor_worker():
   - get from pending_queue
   - 调用 adapter.generate_sequences_single(batch)

3. Adapter → OpenAICompatibleAgentFramework.generate_sequences():
   - 从 batch 提取 raw_prompt, uid, global_steps
   - 为每个 prompt 创建 n 个 session (rollout.n)
   - 每个 session:
     a. Gateway.create_session()
     b. agent_runner() → 子进程执行 SWE-agent
     c. agent_runner 通过 HTTP 调用 Gateway /v1/chat/completions
        → Gateway.generate() → FullyAsyncLLMServerClient.generate()
        (自动处理 partial rollout resume)
     d. Gateway.finalize_session() → trajectories
   - _score_trajectories() → reward
   - _write_session_trajectories_to_tq() → tq.async_kv_batch_put()
   - tq.async_kv_put(uid, status="finished")

4. RB.release_slot()

5. Trainer (FullyAsyncTrainerTQ):
   - RB.wait_and_sample() → KVBatchMeta
   - kv_batch_get → 完整张量数据
   - PPOTrainer 管线: _compute_old_log_prob → _compute_advantage → _update_actor
   - update_weights() → NCCL 同步权重到 Rollouter GPUs
   - RB.reset_staleness() → 重置版本窗口
```

### 5.2 错误处理

- `_run_prompt_sessions_to_tq()` 用 `asyncio.gather(*tasks, return_exceptions=True)` 捕获每个
  session 的异常
- 部分成功时：成功的 session 写 TQ，uid 标记 "finished"；全部失败则标记 "failure"
- 全部失败时 `generate_sequences()` 抛 RuntimeError
- `_process_single_sample_streaming()` 捕获异常，仍调用 `release_slot()`
- RB 的 `wait_and_sample()` 不为 "failure" 的 uid 等待

## 6. 验证计划

1. **配置解析**: YAML 加载验证
2. **端到端训练**: 小规模数据 + Qwen3.5-4B 跑 2 步验证数据流
3. **Partial Rollout**: 验证 weight sync 打断生成后能正常恢复

## 7. 风险与待确认项

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Gateway + `FullyAsyncLLMServerClient` 兼容性 | Gateway.generate() 需正确处理 stop_reason="aborted" | 测试验证 weight sync 场景 |
| TQ key 格式一致性 | Adapter 写的 key 格式需与 Trainer 读取格式匹配 | 复用 `_write_session_trajectories_to_tq()`，与 `AgentLoopWorkerTQ` 一致 |
| `replay_buffer` 改动向后兼容 | sync 路径不受影响 | `if self._replay_buffer is not None` 条件守卫 |
