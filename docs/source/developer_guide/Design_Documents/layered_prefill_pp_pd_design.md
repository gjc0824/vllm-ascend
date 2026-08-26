# PP + PD 混部的 Layered Prefill 适配设计

## 1. 文档状态

- 状态：设计评审稿，仅描述方案，不包含代码实现。
- 目标：在 `pipeline parallelism (PP)` 与 `prefill/decode (P/D)` 混部的单个引擎中，让 Prefill 请求沿 layer group 推进，同时让 Decode 请求每一步执行完整模型，从而降低 PP stage 的时延不均衡和流水气泡。
- 适用仓库：当前工作区中的 `vllm` 与 `vllm-ascend`。
- 论文：From Tokens to Layers: Redefining Stall-Free Scheduling for MoE Serving with Layered Prefill，arXiv:2510.08055v2。

本文把论文已经验证的结论、针对 PP 的工程推导、以及尚未验证的假设分开书写。论文实现运行在 vLLM 衍生的单引擎、TP-only、CUDA 环境，没有实现或评估 PP/NPU；因此下文关于 PP + Ascend 的部分是适配设计，不是论文原文结论。

## 2. 结论先行

### 2.1 论文的核心不是跳过层

Layered Prefill（分层 Prefill）不会丢弃模型计算，也不是 early-exit 或 layer skipping。它改变的是调度轴：

1. 将 Decoder-only Transformer 的连续层划分为 `N_lg` 个连续 layer group。
2. 每次迭代只有一个指定 group 对 Prefill 行执行 Prefill，同时对 Decode 行执行 Decode。
3. 其余 group 只对 Decode 行执行 Decode。
4. Prefill 行在后续迭代推进到下一个 group；经过 `N_lg` 次迭代后，Prompt 的每个 layer 恰好执行一次。

因此，Decode 每一步仍穿过完整模型，Prefill 不再以 token chunk 的方式重复穿过全模型。收益主要来自 MoE expert 权重和 KV 访问的重复加载减少，而不是减少理论 FLOPs。

### 2.2 对当前 PP + PD 混部目标的判断

当前 vLLM 的调度、KV 进度、模型 forward 和 PP 通信都默认“一个请求的 token 在一次 forward 中完成当前本地所有层”。只修改 `num_scheduled_tokens` 或复用现有 Dynamic CPP 预测器，不能得到正确的按层推进：

- SchedulerOutput 没有 layer group、layer frontier 或“本步只提交计算、不提交 token 进度”的语义。
- PP worker 每个 `SchedulerOutput` 只做一次前向、一收一发 `IntermediateTensors`。
- 模型 forward 固定遍历当前 PP rank 的全部本地层。
- Attention/KV metadata 默认每个层都为 scheduled token 写 KV。
- 最后 PP rank 会按普通 batch 取 logits；如果不增加 mask，Prefill 行会在尚未完成最后一组时错误地产生 token。

这是一项跨 Scheduler、Worker/PP 协议、模型层循环、Attention/KV 和计时模型的架构特性，不能作为一个小型 Ascend scheduler patch 安全完成。

### 2.3 最可能被社区接受的路线

建议采用“上游最小通用接口 + Ascend 实验实现”的路线：

- 先在上游 vLLM 抽象出 Layered Prefill 的执行计划和模型能力接口，避免在 vllm-ascend 中复制整个 `Scheduler.schedule()`。
- 第一阶段只支持 **PD-mixed**（`kv_role=kv_both` 或无 KV connector）、PP=1 的参考语义、一个 Prefill cohort，使用 eager；保留 DP=1 以保证各 EP rank 的 scheduler plan 一致，但允许 TP>1。启用 EP 时在 TP/EP group 内使用 AlltoAll dispatcher，TP-only 则保留现有 AllGather MoE 路径。PP>1、DP+EP、PD 分离、SP/PCP/DCP、异步调度、DBO、Speculative/MTP、Mamba/hybrid、Multimodal 和 LoRA 后置。
- 先用一个明确的 MoE/GQA 模型验证语义和收益；Dense 模型默认回退到普通 chunked prefill，因为论文的 Dense 消融中 Layered Prefill 反而更慢。
- 论文的 one-group-per-iteration 是正确性基线；最终目标可以在此基础上动态选择连续的 `k` 个 group，使本步 Prefill 计算落在 Decode 可用时延预算内。
- 只有在 eager 版本的正确性、TP/EP collective 顺序、PP stage balance 和 TBT 指标稳定后，才增加 ACLGraph/编译图、DP+EP、Prefix/KV pool 和 PD 分离支持。

## 3. 论文理解

### 3.1 问题：chunked prefill 在 MoE 中产生 sparsity erosion

Chunked Prefill 沿 token 维度切分 Prompt。每个 chunk 都从第一个 Transformer layer 重新经过完整层栈，因此同一个请求会反复触发各层的 MoE router 和 expert 权重访问。

论文以 Qwen3-30B-A3B 为例：128 个 expert、top-k=8 时，batch size 为 128 平均每个 expert 只有约 8 个 token，远低于现代加速器的 ridge point。小 chunk 既不能让每个 expert 获得足够大的计算 batch，又会让更多 expert 被重复加载，执行转为 memory-bound。

论文的 microbenchmark 观察为：

- 8192-token 输入、chunk=512 时，MoE 占 Prefill runtime 的一半以上，runtime 超过 500 ms。
- chunk 增大到 4096--8192 时，MoE load 降到 100 GB 以下，runtime 接近 200 ms。
- 但单纯增大 chunk 会拉长一次迭代，Qwen arXiv 任务的 p99 TBT 从 512-token chunk 的 48.4 ms 增至 2048-token chunk 的 129 ms，出现效率与 TBT 的直接冲突。

### 3.2 Layered Prefill 的执行规则

设连续层组为 `G_0, G_1, ..., G_(N_lg-1)`，每个请求的 Prompt 行为 `P`，活跃生成请求行为 `D`。第 `t` 次迭代的逻辑为：

```text
G_t:       P + D
其他 group: D
```

这里的 `P + D` 是同一 layer group 内的混合 batch：

- `P` 行使用完整 Prompt query，生成当前 group 的 KV，并更新 hidden/residual。
- `D` 行使用一个或若干 decode query，照常更新所有 layer 的 KV。
- 对 `P` 行，未选中的 layer 必须真正 bypass，不能只在层末尾覆盖输出；否则仍会执行 attention/MoE，并且可能错误写 KV。

论文实现会保存每个 Prefill 请求在当前 group 之后的 hidden/residual，下一次迭代从该 frontier 继续，而不是重新从 embedding 或第一个 layer 开始。

### 3.3 Layer group 数量和 token chunk 的关系

论文使用如下经验式选择 group 数：

```text
N_lg(L) = max(1, ceil(L / 512))
```

其中 512 只是与 chunked baseline 公平比较的参考 token 粒度，不是算法常数。论文也明确指出 Layered Prefill 与 token chunking 正交：超长输入可以用较大的 token chunk（例如 8192）并在每个 chunk 内沿 layer group 推进。

论文实现会把同时到达的短请求合并到同一个 layer stage，避免一个 batch 内存在多个不同的 `prefill_compute_layers`。

### 3.4 论文收益和边界

论文报告的代表性结果：

| 指标 | 论文观察 |
| --- | --- |
| Expert load | ShareGPT 下降约 12%，arXiv 长 Prompt 下降约 39% |
| Qwen arXiv 1.3 req/s | mean TTFT 2.80 s -> 1.24 s；p99 TTFT 8.65 s -> 4.10 s |
| Qwen 可用吞吐 | 最高满足 SLO 的 request rate 约 1.3 -> 1.6 req/s |
| Energy/token | Qwen 约下降 22%，GPT-OSS 约下降 20% |
| `N_lg` trade-off | group 数越大，每步 P 工作越少，TBT 越稳，但 TTFT 越高 |

论文的 Dense Qwen3-8B 消融中 Layered Prefill 的 TTFT/TBT 均劣于 chunked baseline。因此 vllm-ascend 不应把它作为所有模型的默认调度策略，必须具备模型能力检测和普通路径回退。

论文参考实现为 [scale-snu/layered-prefill](https://github.com/scale-snu/layered-prefill)，它是 Nano vLLM，不是当前 vLLM 主线的 PP 实现。其关键工程手段是 per-request stage queue、保存 hidden/residual、对 P/D 行拆分后只运行指定 P layers，并为不同 layer stage 捕获图。

## 4. 当前框架的事实基线

### 4.1 vLLM Scheduler 是 token-progress scheduler

当前 vLLM V1 Scheduler 在 `vllm/vllm/v1/core/sched/scheduler.py:439-450` 明确说明：没有独立的“Prefill phase/Decode phase”，每个请求通过 `num_computed_tokens` 追赶 `num_tokens_with_spec`。主要路径如下：

- `vllm/vllm/v1/core/sched/scheduler.py:516-555` 计算本步 `num_new_tokens`。
- `vllm/vllm/v1/core/sched/scheduler.py:575-637` 为这些 token 分配 KV slots 并填充 `num_scheduled_tokens`。
- `vllm/vllm/v1/core/sched/scheduler.py:1317-1343` 在执行后更新 token progress，并依据 token 是否完成设置 `request.is_prefill_chunk`。

这套语义无法表达“本步执行了 q 个 Prompt token 的第 g 组层，但 token 尚未穿过完整模型”。必须增加独立的 layer frontier 和 token commit 语义。

### 4.2 SchedulerOutput 没有 layer 计划

`vllm/vllm/v1/core/sched/output.py:193-210` 的关键字段只有：

- `req_id -> num_scheduled_tokens`
- 所有请求的 `total_num_scheduled_tokens`
- KV、encoder、speculative 等现有 metadata

没有 `layer_group_id`、`layer_start/end`、`prefill_stage`、`commit_tokens` 或持久化 activation 标识。直接把 layer 信息塞进现有 token 字段会破坏 KV allocator、InputBatch、PP token broadcast 和统计逻辑。

### 4.3 当前 PP forward 是完整本地层栈

模型通过 `make_layers()` 和 `get_pp_indices()` 把全局层静态划分给 PP rank。以 DeepSeek 为例，`vllm/vllm/model_executor/models/deepseek_v2.py:1433-1515`：

- 第一个 rank 做 embedding。
- 每个 rank 循环自己的 `[start_layer, end_layer)` 全部层。
- 非末 rank 只返回 `IntermediateTensors(hidden_states, residual)`。
- 末 rank 做 norm 并产生最终 hidden state。

Llama/Qwen 等模型具有同样的完整本地 layer loop。当前 forward API 没有 runtime layer range 或 per-row layer mask。

### 4.4 PP worker 是“一步一收一发”

上游 `vllm/vllm/v1/worker/gpu_worker.py:1019-1107` 和 Ascend 的镜像实现 `vllm_ascend/worker/worker.py:625-691` 都遵循：

1. 非首 PP rank 对当前 forward `irecv_tensor_dict()`。
2. ModelRunner 执行一次完整 model forward。
3. 非末 PP rank 对输出 `isend_tensor_dict()`。
4. EngineCore 随后进入 sample/update。

`vllm/vllm/v1/engine/core.py:595-614` 也把 execute、sample、scheduler update 作为一个 step 的连续生命周期。Layered Prefill 需要让一个请求在多个 step 中保留 activation frontier，并在非最终 group 时禁止 Prefill 行采样。

### 4.5 Ascend V1/V2 的边界

- `vllm_ascend/worker/model_runner_v1.py:1777-1798` 入口仍是一次 `execute_model(SchedulerOutput, IntermediateTensors)`。
- `:2124-2126` 与 `:2629-2659` 都只调用一次 `self.model(...)`。
- `:2131-2163` 按现有 PP 语义返回 IntermediateTensors 或计算 logits。
- `vllm_ascend/worker/v2/model_runner.py:70-200` 主要复用上游 V2 runner，不能绕过上游模型/PP 协议直接获得 layerwise 语义。

因此第一版应锁定 Ascend V1 ModelRunner；V2 在通用接口稳定后再接入。

### 4.6 已有 Dynamic CPP 只能复用“方法”，不能复用“语义”

vllm-ascend 已有：

- `core/scheduler_profiling_chunk.py`：复制 Scheduler.schedule，在 token budget 上叠加时间预算。
- `core/profiling_chunk_predictor.py`：根据 `(chunk_size, history_length)` 拟合时延。
- `patch/platform/patch_profiling_chunk.py` 与 `worker.py:profile_prefill_latency`：启动 profiling 和在线 timing。

这些代码对 Layered Prefill 有价值的部分是“实测、预测、在线校准”框架；但预测变量应改为 layer group 成本，Scheduler 输出也必须增加 layer plan。继续复制一份完整 `schedule()` 会带来上游 drift，且与 balance、batch-job、short-request-first 等 scheduler subclass 互相覆盖。

### 4.7 “Layerwise”已有含义，不能混用

vllm-ascend 已有 `MooncakeLayerwiseConnector`、layerwise KV pool 和 layerwise KV offload。它们的 layerwise 指 KV cache 按物理层传输/存储；不等同于本文的 Prefill 按 layer group 调度。

本文建议对外使用 `layered_prefill` 或 `layer_group_prefill`，不要把调度策略命名为 `layerwise`，避免与已有 KV transfer 语义冲突。

## 5. PP + PD 混部的目标语义

### 5.1 明确 PD-mixed 与 PD-disaggregated

本设计的第一目标是 **PD-mixed**：同一个引擎同时有 Prefill 和 Decode 请求，通常对应 `kv_role=kv_both` 或没有 KV connector。vllm-ascend 当前也把 `enable_balance_scheduling` 限定为此模式。

这与 P/D 分离不同：

- PD-disaggregated 中 P 和 D 是不同引擎/节点，P 的最终 KV 通过 connector 传给 D。
- Layered Prefill 的 hidden frontier 需要在 P 引擎内跨迭代保存；现有 Mooncake connector 只定义 KV 的传输/释放，不承载中间 hidden/residual 或 layer frontier。

第一版不应同时解决两个问题。PD 分离支持放到后续协议扩展中。

### 5.2 每步的目标计划

将全局层切成固定的连续 groups `G[0..N-1]`。一个调度 step 生成如下计划：

```text
decode rows:  每个 PP stage 执行自己的全部本地层
prefill rows: 只执行 [group_cursor, group_cursor + k) 的连续 groups
              未选中的本地层 bypass，不写 P 行 KV
```

其中 `k` 是本步推进的 group 数：

- 论文基线固定 `k=1`。
- 目标实现根据当前 Decode batch、Prompt query 长度、历史长度和各 PP stage 的实测时延选择最大的安全 `k`。
- 当没有活跃 Decode 请求时，不强行把 P 拆成很多 step；应回退为完整 Prefill 或普通 chunked Prefill，避免无意义地放大 TTFT。

一个 batch 内的 P 请求第一版必须属于同一个 layer cohort，即共享相同的 group partition 和 `[group_cursor, group_end)`。不同长度请求可以进入不同 cohort 队列，不能在一个模型 forward 中隐式混合不同 layer mask。

### 5.3 PP stage 边界与 layer group 边界

推荐分阶段处理：

1. **MVP**：每个 group 完全落在一个 PP stage 内，group 边界与 `VLLM_PP_LAYER_PARTITION` 对齐；一个 step 可推进同一 stage 内的多个连续子组。
2. **第二阶段**：允许一个计划包含跨多个 PP stage 的连续 groups，但要求所有 rank 使用同一个全局计划并维持严格的 collective 顺序。
3. **后续**：研究动态重分区和 stage-aware group packing；不把 `get_pp_indices()` 的静态层划分和 runtime 调度重叠在同一个首发版本中。

stage-aligned 不是算法必须条件，而是 PP 首发的工程约束。它可以把“当前 P 工作落在哪个 stage”表达成明确的 owner rank，显著降低跨轮 activation 恢复的复杂度。

### 5.4 PP activation frontier 的建议协议

标准 PP 每步会让 P 行从 rank 0 重新开始，这会丢失前一 group 的 hidden state。建议在 worker 侧增加 `LayeredPrefillState`：

```text
request_id
cohort_id
group_cursor / group_end
frontier_owner_pp_rank
hidden_states[q, hidden_size]
residual[q, hidden_size] (如模型需要)
positions / token-row mapping
```

在每一个普通 PP forward 中：

- owner 之前的 rank 对 P 行只转发占位 activation；owner rank 用本地保存的 frontier 替换 P 行输入。
- owner rank 执行当前计划覆盖的本地 layer groups，并保存新的 frontier。
- owner 之后的 rank 对 P 行继续转发结果，但只在其本地范围与计划相交时执行 P layer。
- D 行从 rank 0 到最后 rank 按普通完整模型路径执行。
- 所有 rank 仍必须参与相同次数、相同顺序的 PP/TP/EP collective；bypass 只改变 P 行计算，不改变 collective 拓扑。

这可以保留“一步一收一发”的基本 PP transport，但需要在模型输入选择、P row mask、frontier 生命周期和最后 rank 的 logits mask 上增加协议。若未来采用真正多 microbatch 的异步 PP，则需要为每个 in-flight batch 配置独立的 frontier slot，不能复用一个全局 tensor。

### 5.5 Token progress 与 layer progress 分离

建议把以下两个概念严格分开：

- `query_tokens`：本步实际送入 attention/model 的 P/D token 数，用于构造 input 和 attention metadata。
- `logical_token_commit`：请求完整穿过最后一个 layer group 后，才把 Prompt token 数提交给普通 `num_computed_tokens`。

非最终 group 的 P step：

- 可以有 `query_tokens=q`。
- `logical_token_commit=0`。
- 不产生 P 行输出 token。
- 不向 Prefix Cache 或远端 KV connector 发布“完整可复用”的请求。

最终 group 的 P step：

- `logical_token_commit=q`。
- 计算 norm/LM head 和首 token。
- 请求转入 Decode running 状态。

KV blocks 可以在首个 group admission 时一次性预留，后续 group 只复用相同 slot；如果沿用当前每步 allocate 的逻辑，第二个 group 会误把同一批 Prompt 当成新 token。allocator 和 zeroing 逻辑需要显式区分“新 block 分配”和“已有 block 的新 layer 写入”。

## 6. 推荐的接口设计

### 6.1 上游通用数据结构

建议向上游提出一个可选的 `PrefillExecutionPlan`，而不是在 Ascend 中 monkey-patch dataclass：

```python
@dataclass
class PrefillExecutionPlan:
    # 每个 active cohort 的 request ids，首发限制为一个 cohort。
    req_ids: tuple[str, ...]
    group_start: int
    group_end: int              # [start, end)
    num_groups: int
    query_tokens: dict[str, int]
    commit_tokens: bool
    cohort_id: int
```

`SchedulerOutput` 增加可选的 `prefill_execution_plan` 字段。默认值为 `None` 时，现有 token scheduler 完全不变。真正实现时字段命名应遵循上游 review 意见；上面是语义草案，不是要求原样采用。

### 6.2 请求和 Worker 状态

EngineCore/Scheduler 侧需要保存：

- layer cohort 和固定 partition 的 ID；
- 当前 group cursor、总 group 数、下一次允许推进的范围；
- 已分配但尚未完整提交的 KV blocks；
- preemption、abort、finish 时的 partial state。

Worker/ModelRunner 侧需要保存：

- 每个 request 的 frontier hidden/residual；
- frontier 当前 owner PP rank；
- batch reorder 后的 request-to-row 映射；
- 需要在下一次执行中清理的 state。

EngineCore 不应直接持有 GPU activation；调度进程只持有元数据，GPU tensor 由 worker state manager 管理。

### 6.3 模型能力接口

不要把 layer mask 作为所有模型的隐式全局变量。建议增加类似以下能力检查和上下文：

```python
class SupportsLayeredPrefill(Protocol):
    def forward_layered_prefill(
        self,
        ...,
        prefill_plan: PrefillExecutionPlan,
        prefill_state: LayeredPrefillState | None,
    ) -> ...: ...
```

默认模型能力为 false，Scheduler 在启动时 fail-closed 或回退普通 chunked path。支持模型必须明确保证：

- P 行只在计划覆盖的层执行；
- D 行仍执行全部本地层；
- 非活跃 P 层不写 KV、不触发 MoE/attention/通信 kernel；
- residual、aux hidden state、norm、logits 和模型特有状态在 group 边界正确传递。

第一版可以为 Qwen3 MoE GQA 提供实现；不要把 DeepSeek MLA、Mamba 或所有自定义模型一次性纳入通用路径。

### 6.4 Attention/KV 接口

当前 attention metadata 按 batch 构造一次，并在每个 layer 中复用。Layered Prefill 至少需要：

- P/D row 划分或 row compaction；
- 当前 layer 是否对 P rows 写 KV 的 mask；
- 每个 KV group/layer 的 active status；
- P rows 在 inactive layer 的 bypass 路径；
- 最终 group 才触发完整 Prefix Cache store / remote KV completion。

Ascend 的 MLA/GQA、FlashComm、KV pool、zeroing、block copy、layerwise store 都要逐一验证。不能假设把一个全局 `is_prefill` 变成 `False` 就能解决 layer-specific KV 写入。

## 7. 自适应 group 数和 runtime timing

### 7.1 两个容易混淆的参数

需要区分：

- `N_lg`：一个 Prefill request/cohort 的总 group 数，决定 Prompt 要跨多少个调度 step。
- `k_t`：第 t 个 step 实际推进的连续 group 数，决定本步向 Decode 增加多少 P 工作。

论文固定 `k_t=1`；本项目最终目标是根据实际时延动态选择 `k_t`，而不是在 scheduler 中简单写死 `N_lg=ceil(L/512)`。

### 7.2 Group partition 的建议

不要每个请求都创建一套任意 layer partition。建议：

1. 启动时按 PP partition 生成有限个可捕获的 group layouts，例如 `N_lg ∈ {1, 2, 4, 8, 16}`。
2. 请求按 Prompt 长度选择最接近的 layout；`N_lg` 不超过模型 hidden layer 数，也不超过配置的 `max_groups`。
3. 每个 layout 的 group 边界保持连续、稳定，并尽量与 PP stage 边界对齐。
4. 在固定 layout 中以 `k_t` 合并相邻 groups，不修改 layer 权重归属。

这样能避免“每个长度都触发一次模型重编译/ACLGraph capture”，也让时延 predictor 有可解释的 group ID。

### 7.3 预算模型

对 PP rank `r`，设 Decode-only 的基准时间为 `D_r`，当前 P 计划中落在该 rank 的 groups 的预测增量为 `P_r(g, q, h, b)`，其中 q 是 Prompt query length，h 是历史长度，b 是混合 batch 信息。一个保守的本步约束为：

```text
step_time = max_r(D_r + sum(P_r(...)))
step_time <= decode_target + allowed_prefill_slack
```

调度器从当前 group cursor 开始，选择最大的连续 `k_t`，使上式成立。若实际 step 超预算，下一个 step 减少 `k_t`；若连续多个 step 留有预算，再逐步增加 `k_t`。

这里的目标不是让每个 stage 的绝对时间完全相同，而是限制“P 增量叠加到最慢 stage 后”的最大值，避免某个 stage 因 P 参与而成为新的 pipeline bubble 来源。

### 7.4 Predictor 设计

现有 `ChunkSizePredictor` 的二次函数可作为起点，但不能直接把 layer group 当 token chunk：

- Attention 成本仍依赖 `q * (q + h)`。
- MoE 成本依赖当前 group 的 expert 类型、routing、TP/EP 配置和有效 token 数。
- 不同 PP rank 的 layer group 成本不同。
- Decode batch size 和 P/D 混合比例会改变可用 slack。

建议新增独立的 `LayerGroupCostModel`：

- 启动 warmup：对每个允许 layout 和若干 q/h 桶执行真实 eager forward。
- runtime：记录每个 PP rank 的 step duration、P cohort、group range、q/h、D batch size。
- 用 EMA/分桶统计作为首发模型；数据足够后再拟合带交互项的回归模型。
- 预测失败、数据不足或模型变更时，回退到 `k_t=1` 或普通 chunked scheduler。

现有 CPP 的 `torch.npu.synchronize()` timing 会产生同步开销。在线校准必须设置采样频率、warmup 期和停止条件，并在文档中披露测量开销，不能在每个 NPU hot path 无条件同步。

## 8. 适配量评估

这里不使用虚假的 LOC 估算；代码量主要取决于支持的模型数量和 NPU kernel 能否复用。按社区可拆分的模块估算如下：

| 模块 | 主要落点 | 必要工作 | 规模/风险 |
| --- | --- | --- | --- |
| Scheduler 数据协议 | 上游 `vllm/vllm/v1/core/sched/{output,scheduler,request}` | layer cursor、cohort、query/commit 分离、partial KV reservation、preempt/abort 生命周期 | 大；建议 upstream PR |
| Scheduler policy | 上游 hook + Ascend `core/layered_prefill_scheduler.py` | 选择 `N_lg`、`k_t`、cohort queue，与 FCFS/priority 规则整合 | 中到大 |
| PP worker transport | 上游 `v1/worker/gpu_worker.py`、Ascend `worker.py` | frontier owner、P row 注入/转发、in-flight state、collective 顺序 | 大；正确性高风险 |
| ModelRunner | 上游 GPU runner + Ascend `model_runner_v1.py` | plan 传入、query rows、非最终 logits mask、状态释放 | 大 |
| 模型 layer loop | Qwen3 MoE 等支持模型、必要的 patch | P/D 行拆分、选定层执行、residual/aux state | 大；模型相关 |
| Attention/KV | Ascend attention、KV pool、slot mapping、zeroing | layer-specific KV 写入和 bypass，最终 commit 才 cache-complete | 大；kernel 相关 |
| Runtime profiling | Ascend predictor/worker timing | per-rank/group 成本、EMA/回归、预算控制 | 中 |
| Graph/compile | ACLGraph、torch.compile、forward context | 每种 layout/range 的 capture 或首发 eager gate | 中到大 |
| Connector/prefix cache | Mooncake/AscendStore/Prefix Cache | partial KV/frontier 协议、layer completion、失败恢复 | 很大；后置 |
| 测试与观测 | `tests/ut`、`tests/e2e`、metrics | 正确性、PP 通信、时延、bubble、expert load、内存 | 中到大 |

按合理的社区提交拆分，至少是以下 5 类 PR，而不是一个 Ascend-only PR：

1. 上游 execution plan 和 token commit 接口。
2. 上游 `SupportsLayeredPrefill`/PP frontier 语义与一个参考模型。
3. vllm-ascend eager + PP/TP + PD-mixed 实现。
4. Ascend timing、NPU kernel、graph 优化。
5. Prefix/KV connector、EP 和高级特性扩展。

## 9. 与现有特性的冲突和处理策略

| 特性 | 冲突原因 | 首发策略 |
| --- | --- | --- |
| `enable_chunked_prefill` | token chunk 与 layer group 是正交的，但两者同时改变 query/KV progress，组合会放大状态空间 | 首发只支持 layered-only；后续固定大 token chunk + layer groups |
| `profiling_chunk_config`/CPP | 现有 scheduler subclass 已复制完整 `schedule()`；两者不能互相覆盖 | 不叠加 subclass；改成统一 `PrefillSchedulingPolicy`，或首发互斥 |
| `enable_balance_scheduling` | balance scheduler 修改 DP admission，layer scheduler 修改 P cohort/时延预算 | 首发互斥；后续在同一 policy 中定义先 balance DP 再分配 groups 的顺序 |
| `async_scheduling` | PP 中存在多个 in-flight batch；frontier、PP send ring 和 sample/update 顺序难以保持一致 | 首发强制 `--no-async-scheduling` |
| DBO/dual batch overlap | 一个 step 被拆成多个 microbatch，P group cursor 可能跨 microbatch 交错 | 首发禁用 |
| ACLGraph/CUDA Graph/torch.compile | 动态 layer range、row mask、P/D shape 会导致 graph 组合爆炸；图通常捕获完整 layer loop | 首发 eager；后续每个 layout/group range 单独 capture |
| `VLLM_PP_LAYER_PARTITION` | 不同 stage layer 数和 group 边界不一致会产生新的 stage bottleneck | 首发要求显式校验，group 尽量 stage-aligned |
| TP | 同一个 plan 下各 TP rank 可保持相同 layer 顺序 | 首发保留，作为 PP/TP 基线 |
| EP/DeepEP/EPLB | P 行过滤后 all-to-all token 数和 expert routing 需在所有 rank 保持一致；EPLB 统计也会改变 | PP=1 仅支持 DP=1 + eager AlltoAll 基线；MC2/FusedMC2、DP+EP、EPLB 后置 |
| SP/PCP/DCP/CP | row 维切分或 context cache 分片会改变 frontier row mapping；Ascend PCP 本身已有约束 | 首发禁用；后续先处理 DCP，再评估其他 CP |
| Speculative/MTP/Eagle | token progress、draft slots、PP sampled-token broadcast 假设每步有完整 token 结果 | 首发禁用 |
| Mamba/hybrid cache | 除 KV 外还有 recurrent/SSM state，layer 跳转不能只保存 hidden/residual | 首发禁用 |
| Multimodal/encoder-decoder | encoder budget、embedding、cross-attention cache 与 P layer frontier 不同 | 首发禁用 |
| LoRA | P/D 行可能有不同 LoRA batch，layer mask 与动态 adapter batching 组合复杂 | 首发禁用或只允许单 LoRA cohort |
| Prefix Cache/APC | 中间 group 产生的 KV 不是完整可复用前缀；提前发布会读到 partial KV | 首发关闭；只在 final commit 写入完整 cache |
| MooncakeLayerwiseConnector | 当前 layerwise 是 KV 传输，不是 hidden frontier 调度；现有协议不传 partial activation | 首发仅 PD-mixed；后续独立设计 frontier/完成度协议 |
| KV pool/offload/recompute | partial layer 写入、slot 复用和异步 offload 的完成顺序需要新 fencing | 首发关闭；后续按 layer completion 扩展 |
| `short_request_first`/DyntraLB/batch-job scheduler | 都可能替换 scheduler class 或改变 waiting queue 顺序 | 启动时明确互斥，避免后加载的 scheduler 覆盖 layered policy |
| 量化、多流 MoE、shared expert DP | kernel 可能假设连续完整 token batch | 首发只选择已经验证的模型/量化组合，逐项加回 |

特别需要注意 vllm-ascend 当前 `_setup_worker_and_scheduler()` 的 scheduler 选择顺序：profiling、batch-job、DyntraLB 等配置可能覆盖前一个 scheduler class。新特性必须改为组合式 hook 或在启动时拒绝冲突配置，不能依赖 import 顺序。

## 10. 分阶段实施计划

### Phase 0：可观测性和基线，不改变执行语义

目标是先证明“为什么需要 layer groups”，而不是先写 layer mask：

- 在 PP + PD-mixed 下记录每个 PP rank 的 Decode-only、普通 hybrid Prefill、各 layer 区间耗时。
- 记录 stage idle、step max/min、P/D token 数、Prompt 长度、KV block 数和 NPU HBM。
- 对 Qwen3 MoE 等目标模型统计 expert load/dispatch bytes；建立普通 chunked baseline。
- 验证 `VLLM_PP_LAYER_PARTITION` 对 stage balance 的影响。

验收：能用真实 trace 说明 P 参与导致哪个 stage 成为瓶颈，并得到每个局部 layer group 的成本曲线。

### Phase 1：PP=1 的语义参考实现

使用一个支持模型实现论文算法的最小闭环：

- 一个 P cohort，固定 `k=1`。
- P/D row 拆分，保存 GPU hidden/residual frontier。
- 非 final group 不取 P logits、不提交 token progress。
- 关闭 Prefix Cache、KV connector、Speculative、Graph 和所有复杂模型状态。
- 对同一输入比较普通 full prefill 与 layered prefill 的 logits、KV 和生成 token。

这一步不是最终性能目标，而是把 SchedulerOutput、request lifecycle、model layer loop 和 KV commit 语义固定下来。

### Phase 2：PP + TP + PD-mixed eager MVP

在 Phase 1 的参考模型上接入 PP：

- 要求 PP>1、TP 可用；是否保留 EP 取决于 Phase 1 AlltoAll 基线能否提供全局 plan/collective 顺序保证，首个 PP PR 可以先关闭 EP。
- group 边界与 PP stage 对齐；每步一个全局 plan。
- 每个 stage 增加 frontier owner 和 P row 注入/转发。
- 继续使用普通 PP 一收一发，暂时关闭 async/DBO。
- 首发固定 `k=1`，只验证 stage time/bubble 是否改善。

验收不只看 TTFT：必须同时通过 PP 多 rank correctness、P-only、D-only、P+D 混合、P 在中间 group 时无错误输出、请求取消/抢占/finish 等测试。

### Phase 3：自适应 `k_t` 和 Ascend timing

- 引入 per-rank `LayerGroupCostModel` 和 Decode slack budget。
- 启动 profiling 使用少量 layout/q/h 桶，不复制 CPP 的 64 次 token profiling 作为固定规则。
- runtime 使用 EMA 和受控在线校准；发生超预算时降低 `k_t`，连续低于预算时增加 `k_t`。
- 增加短 Prompt、无 Decode、不同 Prompt 长度 cohort 的 fallback。

验收：相比 token chunk baseline，TBT p99 不恶化，PP stage max-min 和 bubble 下降，TTFT/吞吐在目标 MoE trace 上有稳定收益。

### Phase 4：NPU kernel、图和模型扩展

按模型/硬件逐项增加：

- Ascend attention 的 layer-specific KV write mask。
- Qwen3/DeepSeek 等 MoE kernel 的 P/D row compaction。
- 每个允许 layout 的 ACLGraph/compile capture。
- EP/all-to-all、shared expert、多种量化和 DCP。

任何一个扩展都必须保留 eager fallback，不能因为图或 kernel 不支持而静默执行全层 P。

### Phase 5：Prefix/KV pool 和 PD 分离

后续另起设计：

- 定义 per-request layer completion/partial KV 的 connector metadata。
- 设计 P hidden frontier 是否需要跨 engine（通常不应跨 engine，D 只在完整 KV 可用后开始 Decode）。
- MooncakeLayerwiseConnector/AscendStore 只在 layer completion 事件和 slot lifetime 明确后接入。
- 对 P/D 分离评估 layered scheduling 的收益是否仍高于纯 P PP/CPP；不能直接把 PD-mixed 的实现复制过去。

## 11. 配置和回退建议

建议新增独立的 `scheduler_config.layered_prefill_config`，不要复用 `profiling_chunk_config`，也不要未经 review 新增环境变量。可包含：

| 参数 | 建议默认值 | 说明 |
| --- | --- | --- |
| `enabled` | `false` | feature gate，默认零行为变化 |
| `mode` | `"adaptive"` | `"one_group"` 作为论文基线，`"adaptive"` 支持 `k_t` |
| `group_token_target` | `512` | 仅用于选择初始 layout，不是硬编码算法常数 |
| `allowed_num_groups` | `[1,2,4,8,16]` | 预先构建的 layout，需受层数/PP 限制 |
| `max_groups_per_step` | `1` | 首发建议固定为 1，Phase 3 再开放 |
| `target_decode_slack_ms` | `auto` | 从 Decode-only stage timing 推导 |
| `warmup_steps` | `auto` | Predictor 进入 runtime 前的稳定期 |
| `online_calibration` | `false` | 避免默认每步 NPU 同步；显式开启才测量 |
| `require_pd_mixed` | `true` | 首发拒绝 producer/consumer connector role |
| `require_eager` | `true` | graph 支持成熟后再放宽 |

启动校验应 fail-closed：

- `pipeline_parallel_size > 1`；
- `kv_role` 为 `kv_both` 或未配置；
- 模型实现 `SupportsLayeredPrefill`；
- 不启用首发互斥特性；
- group layout 与 PP partition 合法；
- 如果校验失败且用户未强制 enable，则记录原因并回退普通 chunked；如果用户强制 enable，则明确报错。

## 12. 正确性不变量

每个实现阶段都必须保持以下不变量：

1. 每个 P token 的每个 Transformer layer 恰好执行一次；不能少层，也不能重复层。
2. D 行每个 step 都执行完整本地层范围。
3. 非 final group 不产生 P 输出 token，不提前更新 `num_computed_tokens`。
4. 只有已完成全部 layer groups 的 KV 才能被 Prefix Cache、KV connector 或 Decode 请求视为完整 KV。
5. 所有 PP/TP/EP rank 对同一 batch 使用相同的 global plan、group range 和 collective 顺序。
6. request reorder、preemption、abort、finish 和 DP rank 迁移都不会把一个 request 的 frontier 与另一个 request 混合。
7. P/D 行的 logits index、sampling output 和 scheduler update 一一对应；P 中间 group 的空输出不是错误。
8. fallback 到普通 scheduler 时，不能留下 layer frontier、partial KV reservation 或禁用状态。

## 13. 验证指标和测试计划

### 13.1 正确性

- 相同 Prompt、Sampling seed 下，full prefill、chunked prefill、layered prefill 的最终 logits/生成 token 一致。
- P-only、D-only、P+D 混合、P 在每个 group、最后 group、无 Decode slack 等边界。
- PP=1/2/4、不同 `VLLM_PP_LAYER_PARTITION`，验证 collective 次数和 activation shape。
- 取消、抢占、KV block 不足、Prefix Cache 关闭/命中、服务重启后的状态释放。

### 13.2 性能

- TTFT mean/p50/p95/p99、TBT mean/p95/p99、E2E、request rate/SLO attainment。
- 每个 PP rank 的 Decode-only time、P 增量、step max-min、idle/bubble ratio。
- `N_lg` 和 `k_t` 的 sensitivity；Prompt 长度、Decode batch size、P/D 比例。
- MoE expert load bytes、all-to-all bytes、NPU HBM bandwidth、energy/token（若硬件接口可用）。
- Layered activation memory、KV block reservation、frontier 生命周期峰值。

### 13.3 建议基准

首批选择一个可稳定运行的 MoE/GQA 模型，例如 Qwen3-30B-A3B 的 Ascend 支持版本；先用 TP-only 的 PP stage，再逐步加 EP。工作负载至少包括：

- 512、2K、8K、32K 固定 Prompt；
- ShareGPT 类短/中 Prompt；
- arXiv/RAG 类长 Prompt；
- 无 Decode、低 Decode 并发、高 Decode 并发；
- 多种 P/D 到达率和请求长度混合。

结果必须与普通 vLLM-Ascend chunked prefill、静态 PP、现有 Dynamic CPP 分别比较，不能只与未经调优的基线比较。

## 14. 待社区评审的关键问题

1. 首发是否接受只支持 stage-aligned layer groups，还是一开始就要求跨 PP stage 的 group？
2. P frontier 是全部保留在 GPU，还是需要一个可选的 NPU/CPU offload 机制？后者可能抵消收益。
3. `k_t` 的预算目标应以 TBT SLO、Decode-only step time，还是最慢 PP stage 的 EMA 为主？
4. 允许一个 step 同时推进跨多个 PP stage 的多个 group，还是保持“一个 active PP owner/stage per step”以简化 collective？
5. 首发模型选择 Qwen3 MoE GQA、采用受限的 AlltoAll EP，还是直接针对当前 Ascend 主力 DeepSeek MLA？后者需要额外处理 MLA/indexer KV state。
6. 上游 vLLM 是否接受 `PrefillExecutionPlan` 和 `SupportsLayeredPrefill` 这样的通用 API；若不接受，Ascend-only patch 只能作为实验分支，不能承诺长期兼容。

## 15. Phase 1：PP=1 代码级实现蓝图

本节把前面的架构方案收敛为一个可以拆 PR 的 PP=1 参考实现。它仍然是设计，不是本次提交中的代码；代码落地时应先在上游 vLLM 建立最小协议，再在 vllm-ascend 接入 NPU runner。

### 15.1 首阶段的范围和验收边界

第一阶段建议明确为以下配置组合：

- V1 engine、PP=1、eager execution；完全关闭 CUDA/ACL graph、torch.compile、async scheduling、DBO、Speculative/MTP、Mamba/hybrid、CP/PCP/DCP、LoRA、Multimodal、Prefix Cache、KV Connector 和 KV offload。
- 一个 Prefill cohort 加任意 Decode 请求。首个语义版本可以限制一个正在推进的 Prefill request；Decode 请求仍可正常批处理。
- 固定 `N_lg` 和每步 `k=1`。`N_lg` 先按 prompt 长度分桶并限制在模型层数以内，不在这一阶段加入 runtime timing 或动态合并多个 group。
- 首个模型选择 Qwen3 MoE 的 Ascend 支持版本，先覆盖 `TP>=1, DP=1`；TP 与 EP 可以同时启用，但必须保证所有 TP/EP rank 进入相同的 D/P forward 序列。多个 DP group 和动态 EPLB 在 plan 同步完成后加入。
- EP 首发使用已有 eager AlltoAll dispatcher 作为正确性基线；TP-only 保持 AllGather。MC2/FusedMC2 需要验证 active token mask 和固定 global batch 后再开启；不应因为自动选择了不支持的 communicator 而静默退回全层 Prefill。

PP=1 版本的目标是证明以下闭环，而不是立即达到论文性能：

1. Prompt 的每个 token 在每个 layer 恰好计算一次。
2. Decode 每一步仍执行完整模型。
3. 中间 group 不产生 Prefill 输出、不推进逻辑 token/KV completion。
4. 最后 group 的 logits、采样 token 和 KV 状态与普通 full Prefill 一致。
5. EP 各 rank 的 layer/collective 顺序一致，EP=1 与 EP>1 生成结果一致。

### 15.2 为什么首版采用两个 eager 子批次

当前 Attention backend 以一次 `InputBatch` 的 `query_start_loc`、`slot_mapping` 和 block table 为单位构造 metadata；如果在每一个 layer 内把 P 行动态屏蔽，还需要为每层重建 metadata 或改 NPU attention kernel。首版建议采用下面的语义参考路径：

```text
一个 scheduler step
  ├─ Decode view：所有 D request，调用现有 full model forward（全部层）
  └─ Prefill view：当前 P cohort，调用 layered forward（仅 [group_start, group_end)）
```

两次 forward 都在同一 worker、同一 NPU stream、同一 step 内按固定顺序执行。P/D 子批次不共享输入 buffer，但共享 KV cache；每个 view 各自构造合法的 attention metadata。这样可以直接复用现有 attention/KV 写入和 EP dispatcher，避免在第一版同时引入 `active_row_mask` kernel。

该路径在 P/D 混合时会有两次 model-call，吞吐不会等于论文的混合 batch；它的价值是先固定调度和状态语义。Phase 1 通过后，再实现单次 forward 的 P/D row compaction：对每层只取 `D rows | active P rows`，并把两次调用合并为一次或少数几次 kernel 调度。

### 15.3 跨进程计划和 worker 状态

建议在上游新增 `vllm/vllm/v1/core/layered_prefill.py`，只放可序列化的 CPU metadata：

```python
@dataclass(frozen=True)
class LayerGroupRange:
    group_id: int
    start: int                 # inclusive, global layer index
    end: int                   # exclusive


@dataclass(frozen=True)
class LayeredPrefillPlan:
    version: int
    cohort_id: int
    group_id: int
    num_groups: int
    group_start: int
    group_end: int
    prefill_req_ids: tuple[str, ...]
    # Query length of each P request in this step. It is normally the
    # entire uncached prompt, not a newly allocated token count.
    query_tokens: dict[str, int]
    # 0 for an intermediate group, q for the final group.
    commit_tokens: dict[str, int]
    reuse_kv_blocks: bool

    @property
    def is_final_group(self) -> bool:
        return self.group_id + 1 == self.num_groups
```

把 `SchedulerOutput` 扩展为：

```python
layered_prefill_plan: LayeredPrefillPlan | None = None
```

默认值为 `None` 时，现有 scheduler/worker 行为完全不变。不要把 `group_start` 临时挂到 `SchedulerOutput` 的动态属性上，也不要把它编码到 `num_scheduled_tokens`；后者会被 KV allocator、PP token broadcast 和统计逻辑同时解释。

GPU worker 侧另建非序列化的状态：

```python
@dataclass
class LayeredFrontier:
    req_id: str
    group_id: int
    query_len: int
    hidden_states: torch.Tensor       # [query_len, hidden_size]
    residual: torch.Tensor | None     # same shape for residual-style models


class LayeredPrefillStateStore:
    by_req_id: dict[str, LayeredFrontier]
```

`LayeredFrontier` 只能存于 ModelRunner/worker GPU 进程，Scheduler 进程只保存 `group_id` 和生命周期 metadata。索引必须使用 `req_id` 或稳定的 request-state index，不能直接使用本 step 的 batch row；batch reorder、preemption 和 request admission 都会改变 row 顺序。

### 15.4 Request 和 Scheduler 的具体改动

#### Request 字段

在 `vllm/vllm/v1/request.py:Request` 增加 CPU-only 字段，建议默认关闭：

```python
self.layered_prefill_enabled = False
self.layered_prefill_cohort_id = -1
self.layered_prefill_group_id = 0
self.layered_prefill_num_groups = 0
self.layered_prefill_query_tokens = 0
self.layered_prefill_kv_reserved = False
```

这些字段不替代 `num_computed_tokens`。含义必须固定为：

- `num_computed_tokens`：已经完整穿过所有必要 layer 的逻辑 token 前缀；
- `layered_prefill_group_id`：当前 frontier 已完成的 group 数；
- `layered_prefill_query_tokens`：每次 layered P view 要重放的 prompt query 长度；
- `layered_prefill_kv_reserved`：prompt 对应 token slots 是否已经分配。

#### 调度入口

不要复制一份完整的 `Scheduler.schedule()`。在 `schedule()` 的 running/waiting request 选择阶段抽象出一个 `PrefillSchedulingPolicy` hook：

```python
plan = self.layered_policy.try_schedule(
    running=self.running,
    waiting=self.waiting,
    token_budget=token_budget,
    decode_requests=scheduled_running_reqs,
)
```

首版 policy 的状态机如下：

```text
WAITING, group=0, kv_reserved=False
  -- admit + allocate_slots(q) --> RUNNING, group=0, kv_reserved=True
RUNNING, group=g < N-1
  -- reuse existing blocks --> RUNNING, group=g+1
RUNNING, group=N-1
  -- commit q + sample --> DECODING (ordinary RUNNING semantics)
```

每个 P step 都把 `num_scheduled_tokens[req_id]` 设为 `q`，因为 runner 确实要处理 q 个 prompt query；但只有第一次调用 `kv_cache_manager.allocate_slots(request, q)`。后续 step：

- `req_to_new_blocks[req_id]` 必须表达“无新增 block”，不能把全部旧 block 当成新 block 再 append；
- `CachedRequestData` 增加一个明确的 `reuse_existing_blocks` 或等价标志；
- ModelRunner 继续使用已有 block table，slot mapping 每次按相同的 prompt positions 计算；
- Phase 1 关闭 prefix cache，避免“部分 layer 已写入但 token prefix 被视为 cache hit”的歧义。

在 `_update_after_schedule()` 中使用 plan 的 `commit_tokens`，而不是无条件加 `num_scheduled_tokens`：

```python
for req_id, scheduled_q in output.num_scheduled_tokens.items():
    request = self.requests[req_id]
    commit_q = (
        output.layered_prefill_plan.commit_tokens.get(req_id, 0)
        if output.layered_prefill_plan is not None
        else scheduled_q
    )
    request.num_computed_tokens += commit_q
    request.num_in_flight_tokens += scheduled_q
    request.is_prefill_chunk = (
        request.num_computed_tokens < request.num_tokens
    )
```

`num_in_flight_tokens` 仍按真实 query 数增加，等 `update_from_output()` 消费本 step 后归零；`num_computed_tokens` 在中间 group 保持不变。这样可以保留当前同步 EngineCore 的 execute/update 生命周期，同时不让 scheduler 误以为 prompt 已完成。

`_make_cached_request_data()`、`get_grammar_bitmask()` 和 `_inflight_prefills` 需要识别 layered plan：

- 中间 group 不发送重复的“新 token” payload；worker 使用 request-state 中已经缓存的 prompt token ids。
- 中间 group 的 P request 永远不进入 structured-output grammar sampling。
- P request 保留在 `_inflight_prefills`，直到最终 group 的 commit 发生。
- D request 的 token progress 和 grammar 行为不变。

首版 cohort 规则要简单且可证明：一个 plan 中所有 P request 必须具有相同 `cohort_id`、`num_groups` 和 `group_id`。不满足条件的新 P request 留在 waiting queue，不能在同一模型调用中隐式混合不同 layer range。

#### Preempt/abort/finish

扩展 `_preempt_request()` 和 `finish_requests()`：

1. 释放该 request 的全部预留 block。
2. 从 `LayeredPrefillStateStore` 删除 frontier。
3. 将 group cursor 重置为 0；恢复后从完整 prompt 重新开始，不尝试使用失效的 partial activation。
4. 如果 abort 发生在 forward 与 update 之间，worker 下一次 `finish_requests()` 必须先清理 frontier，再复用 request-state slot。

这比保存 partial activation 后做抢占恢复更保守，但不会引入 hidden/KV 状态错配。

### 15.5 PP=1 ModelRunner 的调用链

`vllm/vllm/v1/worker/gpu/model_runner.py` 和 `vllm-ascend/vllm_ascend/worker/model_runner_v1.py` 都在 `execute_model()` 形成一次 `InputBatch`。建议在构造普通 `model_inputs` 后、调用完整 `self.model(...)` 前增加显式分支：

```python
plan = scheduler_output.layered_prefill_plan
if plan is None:
    model_output = self._model_forward(model_inputs)
else:
    model_output = self._execute_layered_pp1(
        scheduler_output=scheduler_output,
        plan=plan,
        input_batch=input_batch,
        attn_metadata=attn_metadata,
        slot_mappings_by_layer=slot_mappings_by_layer,
    )
```

`_execute_layered_pp1()` 的建议顺序：

1. `LayeredBatchView.from_input_batch(..., request_ids=decode_req_ids)`，调用现有完整模型 forward，得到 D hidden/logits 输入；
2. 从 `input_batch` 为 P cohort 构造独立 view，保留原 request-state index、prompt positions、block table 和 slot mapping；
3. 若 `group_id == 0`，P model call 从 prompt embedding 开始；否则从 `LayeredFrontier.hidden_states/residual` 开始；
4. 调用模型的 `forward_layered_prefill(layer_start, layer_end, ...)`，只执行计划范围；
5. 中间 group 把返回的 hidden/residual 写回 state store，不计算 P norm/LM head；最终 group 才生成 P hidden，并标记 P row 可采样；
6. 把 D/P 的最终 hidden、eligible row 和 request index 写入 `LayeredExecuteState`，供 `sample_tokens()` 使用。

`LayeredBatchView` 是 worker-local 的轻量视图，不要复制完整 `RequestState`：

```python
@dataclass
class LayeredBatchView:
    req_ids: list[str]
    state_indices: torch.Tensor
    num_scheduled_tokens: np.ndarray
    query_start_loc: torch.Tensor
    input_ids: torch.Tensor | None
    positions: torch.Tensor
    block_tables: tuple[torch.Tensor, ...]
    slot_mappings: tuple[torch.Tensor, ...]
```

它必须能由已有 `BlockTables.gather_block_tables()` 和 `compute_slot_mappings()` 生成，确保 Phase 1 不改 Attention kernel。P view 的 positions 在所有 group 都是 prompt 的 `[0, q)`；D view 的 positions 由普通 `num_computed_tokens` 和 sampled token 状态生成。

`execute_model_state` 需要增加 `layered_state`，不能只存一个完整 batch 的 hidden：

```python
@dataclass
class LayeredExecuteState:
    decode_input_batch: LayeredBatchView | None
    prefill_input_batch: LayeredBatchView | None
    decode_hidden: torch.Tensor | None
    prefill_hidden: torch.Tensor | None
    sample_req_ids: list[str]
    intermediate_prefill_req_ids: list[str]
```

#### Sampling 语义

当前 `Sampler.__call__()` 假设每个 `InputBatch` row 都有一个 logits。首版不要给中间 P row 构造伪 logits 再事后清空，因为这会污染 sampling state、penalty state 或 PP token cache。增加一个 worker-local `SampleBatchView`，只包含以下 request：

- 所有 Decode request；
- 最终 group 的 Prefill request；
- 不包含中间 group 的 Prefill request。

调用 sampler 后，按原 scheduler request 顺序重建 `ModelRunnerOutput.sampled_token_ids`；中间 P request 对应空 list。只有最终 P group 的 `postprocess_num_computed_tokens()` 才把 q 写入 worker GPU token counter。PP=1 不需要广播 sampled token，但这套输出形状要为后续 PP 设计保留显式 mask。

### 15.6 支持模型的代码接口

建议在 `vllm/vllm/model_executor/models/interfaces.py` 增加 opt-in protocol，默认能力为 false：

```python
class SupportsLayeredPrefill(Protocol):
    num_hidden_layers: int

    def forward_layered_prefill(
        self,
        *,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        layer_start: int,
        layer_end: int,
        frontier: tuple[torch.Tensor, torch.Tensor | None] | None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> LayeredForwardOutput: ...


@dataclass
class LayeredForwardOutput:
    hidden_states: torch.Tensor
    residual: torch.Tensor | None
    is_final_layer: bool
```

实际接口可以按上游 review 改名，但必须具备显式 layer range，不能依赖 forward context 中的隐式全局变量。Scheduler 启动时检查 `isinstance(model, SupportsLayeredPrefill)`；能力检查失败时回退普通 chunked prefill 或在显式 `force` 下报错。

#### Qwen3 MoE 的最小改法

在 `vllm/vllm/model_executor/models/qwen3_moe.py` 中将逻辑放在 `Qwen3MoeModel`，顶层 `Qwen3MoeForCausalLM` 只做委托：

```python
def forward_layered_prefill(..., layer_start, layer_end, frontier):
    if frontier is None:
        hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        hidden_states, residual = frontier

    for global_idx in range(layer_start, layer_end):
        local_idx = global_idx - self.start_layer
        layer = self.layers[local_idx]
        hidden_states, residual = layer(positions, hidden_states, residual)

    return LayeredForwardOutput(hidden_states, residual, layer_end == self.end_layer)
```

PP=1 时 `start_layer=0`、`end_layer=num_hidden_layers`，因此不需要 `PPMissingLayer`。真实实现还要：

- 对 `layer_start/end` 做边界和连续性校验；
- group 0 的第一层使用 `residual=None`，后续 group 使用保存的 residual；
- 中间 group 不调用 `self.norm`；最终 group 在输出 logits 前调用 norm；
- 只在 active layer 中调用 `Qwen3MoeDecoderLayer`，未选中的层完全不进入 attention、router、expert 或 KV write；
- 保留全局 `layer_idx`，因为 attention 的 `layer_idx` 和 EP/EPLB 统计不能使用从零开始的局部下标。

Dense Qwen/Llama 可以先实现同一协议作为 reference correctness model，但默认策略仍应对 Dense 回退普通路径；论文表明 Dense 模型不一定受益。

### 15.7 KV/Attention 的首阶段实现细节

采用 P/D 两个子批次后，首版不需要给 attention 增加 layer mask：

- D view 每个 layer 都使用 D rows 的 slot mapping，行为与现有 decode 一致。
- P view 只被传给当前 group 的 layer loop，因此该 view 的每个 layer 都是 active；其他 layer 根本不调用。
- P 每个 group 使用相同 prompt positions 和相同 block table；slot 是按 token position 复用，不是重新 allocate。
- 每个 P request 的每个 layer 只允许出现一次 `store_kv_cache`；可以在测试模式给 attention layer 加计数 hook 验证。
- 只有最终 group 完成后，scheduler 才允许 `cache_blocks()`、Prefix Cache、connector 或 Decode request 将该 prompt 视为完整。

因此 Phase 1 的 KV 适配主要是 allocator 生命周期和 view 构造，不是 NPU kernel 改造。后续合并 P/D row 时才需要在 `slot_mappings_by_layer` 中加入 active-row mask，并确认 `attention_v1.py`、`mla_v1.py`、SFA/FA3 backend 对非连续 row 的支持。

### 15.8 EP 支持策略

PP=1、TP/EP>1 时，所有 TP/EP rank 都持有完整的 layer loop；attention/非专家权重按 TP 现有方式分片，routed expert 权重和 token dispatcher 按 EP 分片。首版利用两个子批次把 TP/EP 约束降到可验证范围：

1. Scheduler plan 必须在 TP/EP group 内完全一致；Phase 1 暂时要求 DP=1，避免不同 DP rank 的 P cohort 和 query 数不一致。TP>1 不改变 plan，只要求每个 TP rank 执行相同的 layer/collective 序列。
2. 每个 step 的调用序列固定为 `D full forward -> P active-group forward`；所有 EP rank 以相同顺序进入每一个 MoE layer 的 dispatch/combine collective。
3. D/P 两个 forward 分别进入 `set_ascend_forward_context()`，使 `num_tokens` 和 communicator 选择对应当前子批次，而不是把 P+D 的总 token 数误传给 MC2 capacity 判断。
4. 启用 EP 时首发强制已有 AlltoAll dispatcher；TP-only 保持 AllGather。MC2/FusedMC2 只有在 active rows、padding、`global_bs` 和 `x_active_mask` 的测试通过后才放开。
5. `dynamic_eplb`、EPLB heat collection 和 expert placement 迁移首发关闭；静态 EP expert map 不受影响。否则同一 prompt 被拆成多个 group 会改变负载统计的时间窗口。
6. 如果某个 group 没有 P rows，所有 EP rank 都跳过 P forward；不能只让 rank 0 跳过，否则下一次 MoE collective 会死锁。
7. 测试必须比较 EP=1 与 EP=2/4 的 logits、sampled token、每层 routing shape 和 all-to-all send/recv count；任何 rank 的 active layer 序列不同都视为 fatal error。

这里的“支持 EP”表示支持现有 EP dispatcher 在 eager、固定 batch 语义下执行 layer group，不表示已经支持 DP+EP、sequence-parallel MoE、EPLB 动态迁移或 FusedMC2。后续单次混合 forward 需要为每个 EP collective 传递一致的 active token mask；这是独立的 kernel/通信适配工作。

### 15.9 具体文件和 PR 拆分

建议按以下顺序提交，避免把 Ascend runner 与 scheduler 状态一次性耦合：

| PR | 文件范围 | 交付物 |
| --- | --- | --- |
| 1 | 上游 `vllm/vllm/v1/core/layered_prefill.py`、`request.py`、`sched/output.py`、`sched/scheduler.py` | plan、group cursor、commit token、一次性 KV reservation、preempt/abort 状态；默认关闭 |
| 2 | 上游 `vllm/vllm/v1/worker/gpu/input_batch.py`、`worker/gpu/model_runner.py`、`worker/gpu/sample/*` | LayeredBatchView、frontier store、双子批次执行、sample mask、eager 生命周期 |
| 3 | 上游 `model_executor/models/interfaces.py`、`models/qwen3_moe.py` | `SupportsLayeredPrefill` 和 Qwen3 MoE 参考实现；CPU/GPU 数值测试 |
| 4 | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py`、`ascend_forward_context.py`、EP dispatcher 相关测试 | NPU eager 子批次、AlltoAll EP、communicator 选择和 HBM frontier budget |
| 5 | `vllm-ascend/tests/ut`、`tests/e2e`、metrics | PP=1 P/D、KV、抢占、EP 多 rank、性能基线和 feature gate |

如果上游暂时不接受通用 protocol，vllm-ascend 可以在实验分支实现 PR 2/4，但应把 `LayeredPrefillPlan` 放在独立兼容层，不要复制完整 scheduler；否则每次上游 scheduler 改动都需要手工同步。

### 15.10 Phase 1 测试用例和通过标准

#### Scheduler 单元测试

- `N_lg=1` 与普通 Prefill 的 `SchedulerOutput` 和 token progress 等价。
- `N_lg=2/4`：连续 step 的 `group_id` 为 `0..N-1`；前 `N-1` 步 `commit_tokens=0`，最终步为 q。
- 中间步 `num_computed_tokens` 不变、`num_in_flight_tokens` 在 update 后归零。
- 后续 group 不新增 block；preempt 后重新从 group 0 分配。
- 不同 cohort 不会出现在同一个 plan；D request 不受 layered cursor 影响。

#### Model/Runner 正确性测试

- tiny Qwen3 MoE，固定 seed，比较 full Prefill 和 layered Prefill 最终 logits/生成 token。
- 每个 group 单独执行一次 layer hook，验证每个 P token/layer 计数为 1。
- 中间 group 没有 P `compute_logits`、norm、sampling 或 KV-complete 事件。
- P-only、D-only、P+D 混合；P request 在最后 group 才出现首个生成 token。
- request reorder、finish、abort、OOM/preempt 后 frontier 不泄漏或串 request。

#### EP 多进程测试

- EP=1、EP=2、EP=4 的 final logits 容差一致；至少覆盖一个 MoE layer 位于每个 group 边界的情况。
- 每个 rank 记录 `(step_id, group_id, layer_idx, comm_type, active_tokens)`，集合后逐项相同。
- 路由 token 数、AlltoAll split、combine 输出 shape 与普通 full Prefill 对比。
- 任意 rank 禁用某个 P group 时，测试应失败并明确报 collective-order mismatch，而不是挂死。

#### 首发性能/资源门禁

- 记录 frontier HBM：`query_len * hidden_size * sizeof(dtype) * (hidden + residual)`；超过配置上限直接回退普通 Prefill。
- 记录 D/P 两个子批次耗时、每层耗时、MoE dispatch bytes、AlltoAll bytes、TTFT/TBT。
- Phase 1 不以论文吞吐为验收标准；必须先满足 correctness、无 KV 重复写、无 EP collective mismatch 和可控内存。

### 15.11 Phase 1 之后的优化顺序

1. 在同一 `LayeredBatchView` 中合并 P/D rows，逐层使用 `D rows | active P rows`，去掉双 model-call。
2. 为 Attention metadata 增加 row compaction/active mask；验证 FA3/SFA/MLA 和按层 KV write。
3. 为 EP dispatcher 增加统一 active mask，放开 MC2/FusedMC2、DP+EP 和 dynamic EPLB。
4. 引入 measured group cost、Decode slack 和 `k>1`；保持 `k=1` fallback。
5. 最后再做 ACLGraph/compile capture；图 key 至少包含 layout、group range、P/D row shape、EP communicator mode。

## 16. 参考资料

- [From Tokens to Layers: arXiv 2510.08055](https://arxiv.org/abs/2510.08055)
- [Layered Prefill reference implementation](https://github.com/scale-snu/layered-prefill)
- 当前工作区 vLLM PP 实现：`vllm/vllm/model_executor/models/utils.py`（跨仓库源码路径）
- [vLLM-Ascend Pipeline Parallel guide](../../user_guide/feature_guide/pipeline_parallel.md)
- [vLLM-Ascend Dynamic Chunked Pipeline Parallel design](dynamic_chunked_pipeline_parallel.md)
- [vLLM-Ascend Disaggregated Prefill design](disaggregated_prefill.md)
