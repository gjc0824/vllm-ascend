# Layered Prefill × 异步调度（Async Scheduling）冲突评估与解除方案

## 1. 背景与结论

Layered Prefill 在 vllm（`layer_prefill tp eager` → `Support chunked layered prefill with prefix caching` 共 6 个 commit）与 vllm-ascend（`layer_prefill tp eager` → `Enable layered prefill with prefix caching` 共 8 个 commit，均为 2026-08-26 之后）中实现了按 layer group 跨 step 推进 Prefill、Decode 每步全量前向的调度方式。此前该特性与 `--async-scheduling` 通过调度器门禁、平台门禁互斥。

本评估逐条核对了 vLLM V1 异步调度的完整协议（`EngineCore.step_with_batch_queue`、`AsyncScheduler` 占位符协议、worker 两阶段 RPC、`prev_sampled_token_ids` GPU 回填通道）后得出结论：

- **PP=1 时协议不冲突**。异步调度下 worker 收到的 RPC 严格保持 `execute(N), sample(N), execute(N+1), sample(N+1)` 成对顺序（`vllm/v1/engine/core.py`），单槽 `execute_model_state`、frontier 时序、`input_batch` 交换与 PP 一收一发语义全部保持成立。真正的冲突是 4 个具体问题（C1–C4），全部有局部修复。
- **PP>1 时确实冲突**。异步 PP 把采样 token 的回传从"调度器回显"换成 GPU 广播 ring，且 V1 runner 下异步 PP 本身无 overlap 收益，本期维持 fail-closed 门禁（C4）。

## 2. 异步调度协议要点（与 layered 相关的部分）

1. **两阶段 worker API**：`execute_model` 返回 `None` 并挂起 `ExecuteModelState`，随后由 `sample_tokens(grammar_output)` 收尾。每步恰好一次 execute + 一次 sample，顺序严格交替。
2. **占位符协议**（`AsyncScheduler._update_after_schedule`）：每个本步会采样的非 prefill-chunk 请求在调度时 `num_output_placeholders += num_sampled_tokens_per_step`；worker 侧采样结果先以 `-1` 占位写入 `token_ids_cpu` 与 `req_state.output_token_ids`；`update_from_output` 交付真实 token 时 `num_output_placeholders -= len(new_token_ids)` 并断言非负。
3. **GPU 回填通道**：`input_batch.prev_sampled_token_ids` + `prev_req_id_to_index` 让下一步 `_prepare_inputs` 直接用上一步采样结果做 GPU patch，CPU 侧 `token_ids_cpu` 惰性对齐。
4. **batch queue**：PP≤1 时 `max_concurrent_batches=2`，即调度(N+1) 先于 update_from_output(N) 执行。

## 3. 冲突清单

| # | 严重度 | 冲突 | 位置 | 处理 |
| --- | --- | --- | --- | --- |
| C1 | 崩溃 | Layered P 请求在 `_schedule_regular` 返回之后才追加进 `num_scheduled_tokens`，`AsyncScheduler._update_after_schedule` 从未见到它，`num_output_placeholders` 保持 0；final group 交付 1 个 token 时驱动占位符到 -1，断言失败（若请求在该步结束则必崩） | `vllm/v1/core/sched/async_scheduler.py` | `AsyncScheduler._update_after_layered_schedule` 覆写：`plan.is_sampling_step` 时补 `num_output_placeholders += num_sampled_tokens_per_step` |
| C2 | 数值错误 | P 请求的首次采样在 `p_input_batch` 上发生，采样结果以 `-1` 占位写入 `req_state.output_token_ids`；下一步该请求迁回主 batch 时 `add_request` 会用 -1 作为 input id。同步路径采样后立即写真实 id，无此问题 | `vllm_ascend/worker/model_runner_v1.py` 采样 bookkeeping | 新增 `_commit_layered_sampled_tokens`：P 子批次为采样步时，从该步 CPU 已就绪的 `output.sampled_token_ids` 解析真实 id 并回填 `req_state.output_token_ids` 尾部占位符 |
| C3 | 性能 | `_execute_layered_step` 中 3 处全设备 `torch.npu.synchronize()` 将异步调度的 CPU/GPU 重叠完全串行化 | `vllm_ascend/worker/model_runner_v1.py`（P forward 后、PP clone 前、D/P 之间） | 收窄为主流级 `torch.npu.current_stream().synchronize()`（P forward 全部入队于主流，DSA 等内部 fork 均已事件闭环；独立 commit） |
| C4 | 范围 | PP>1 + async + layered：异步 PP 的 GPU 采样广播 ring 与 layered 一收一发 packed payload 不兼容；且 V1 runner + PP>1 下异步本身无 overlap 收益 | `vllm/v1/core/sched/scheduler.py`、`vllm_ascend/platform.py` | 维持 fail-closed：门禁改为仅拒绝 `async_scheduling and pipeline_parallel_size > 1` |

## 4. 已排除的疑似冲突

以下条目经过核对，**不构成冲突**，记录以免后续重复排查：

- **单槽 `execute_model_state` / frontier 时序**：layered 的 `execute_model` 返回挂起状态、`sample_tokens` 收尾，与异步两阶段协议同构；RPC FIFO 配对顺序保证每个请求同一时刻只有一个 in-flight frontier。
- **cache_blocks 发布时机**：layered 在 chunk 完成 group 的调度时发布前缀缓存，本版本常规 prefill 同样在 `allocate_slots` 时发布（`kv_cache_manager.py`），安全性靠 GPU stream FIFO 顺序与 block 引用计数保证。异步下的重复发布是幂等 no-op；intermediate group 只重发布已完成的 `chunk_start` 前缀，不会提前发布 partial KV。
- **乐观记账**：`num_in_flight_tokens` 调度时累加、`update_from_output` 按输出 FIFO 排空，layered 语义与异步现有协议一致。
- **KV 块复用/释放**：异步的 deferred-free fence 已覆盖两 in-flight batch 窗口，layered 走同一 allocator。
- **Structured output**：layered 准入门禁排除 structured-output 请求，P 行不参与延迟采样；D 行走标准异步路径。
- **P→D 迁移的 `all_token_ids` 缺省**：`_update_states` 异步恢复分支改为先判断 `req_id in req_data.all_token_ids`；layered P 请求因上一 step 已被调度而不携带该 payload，其首个采样 token 由 worker 侧 C2 修复直接解决。

## 5. 修改设计

### 5.1 vllm 仓库

- `vllm/v1/core/sched/scheduler.py` — `_layered_prefill_supported_for_scheduler`：由整体拒绝 `async_scheduling` 改为仅拒绝 `async_scheduling and PP>1`；async+PP=1 允许 layered。
- `vllm/v1/core/sched/async_scheduler.py` — 新增 `_update_after_layered_schedule` 覆写（C1）：调用 `super()` 后，仅当 `plan.is_sampling_step` 时补占位符；intermediate group 步无输出不加。
- `vllm/v1/worker/gpu_model_runner.py` — `_update_states` 异步恢复分支增加 `all_token_ids` 存在性守卫，避免 P 请求迁移步 KeyError。

### 5.2 vllm-ascend 仓库

- `vllm_ascend/platform.py` — 门禁改为仅拒绝 async+PP>1；注释块同步更新（async+PP=1 支持的前提：V1 runner 每 worker 至多一个 in-flight execute；async PP 广播 ring 不兼容 layered payload；DBO 仍禁用）。
- `vllm_ascend/worker/model_runner_v1.py` — C2 修复（`_commit_layered_sampled_tokens`），仅在 P 子批次的采样步调用。

## 6. 剩余限制

- **PP>1**：与 async 仍互斥（C4）。解除需要 layered PP 步的采样 token 参与（或绕开）异步 PP 广播 ring，且 V1 runner 下无收益，暂无计划。
- **DBO**：多 in-flight frontier 与 layered 单 frontier 假设冲突，维持禁用。
- **C3 性能收窄**为独立 commit：三处全设备 sync 收窄为主流级同步；主流之外的跨流依赖（HCCL communicator 内部顺序、DSA overlap 流的事件闭环、ACL graph replay 的 update_stream 语义）均由各自机制保证。

## 7. 验证

- `vllm`：`tests/v1/core/test_layered_prefill.py`（async × layered 占位符协议回归、async+PP=1 放行、async+PP>1 回退常规调度）与 `tests/v1/core/test_async_scheduler.py`、`tests/v1/core/test_scheduler.py` 全部通过。
- `vllm-ascend`：`tests/ut/test_platform.py`（async+PP=1 接受、async+PP=2 拒绝）、`tests/ut/worker/test_layered_sampled_tokens_commit.py`（占位符回填）、`tests/ut/models/test_layered_prefill.py`、`tests/ut/patch/platform/test_patch_pp_mtp.py` 全部通过。
- NPU E2E（TP=2、Qwen2.5-0.5B、`group_token_target=512`、含 2.5k-token 分 chunk 提示与 P/D 混批）：layered+async+eager 与 layered+async+`FULL_DECODE_ONLY`（D 全图、P eager）的输出与 layered+sync+eager 参考**逐字一致**（greedy、`ignore_eos`、24 token 全序列比对）。抢占/中止生命周期的 frontier 清理由 vllm 既有 scheduler 单测覆盖，未在本次 E2E 中单独加压。
