# Layered Prefill 开发计划

## 1. 目的和状态

- 状态：执行中。
- 基线日期：2026-08-27。
- 适用仓库：当前工作区中的 `vllm` 与 `vllm-ascend`。
- 架构和语义来源：[PP + PD 混部的 Layered Prefill 适配设计](layered_prefill_pp_pd_design.md)。

本文只维护当前实现状态、优先级、依赖和验收条件。接口背景、协议推导和长期兼容性分析保留在设计文档中。

当前结论是：**不等待 P group 图即可开始受限范围的收益测试，但 Phase 1 还没有完成。** D 子批次已经可以使用 `FULL_DECODE_ONLY` ACLGraph，P 子批次继续 eager。TP=4 长输入精度、KV/lifecycle 和 frontier 资源门禁必须与性能测试并行解决。下一实现里程碑是 DP=1、TP/EP>1 的 eager EP 基线，并要求 ordinary/layered 共用同一个通信 selector；P graph 只有在后续 profiling 证明 P eager 是主要瓶颈后才启动。FusedMC2 放在 P/D 单 forward 之后，因为它依赖最终混合 batch 的 `global_bs`、padding 和 active-row mask；非 fused MC2 随 EP 标准 selector 一起验证。

## 2. 当前基线

### 2.1 已实现

| 能力 | 当前状态 |
| --- | --- |
| Scheduler/Request 协议 | 已有 layer plan、group cursor、final-only token commit 和 KV block 复用语义 |
| Model | Qwen3-MoE 支持连续 partial-layer forward 和 hidden/residual frontier |
| Runner | 同一 step 固定执行 `D full forward -> P active-group forward` 两个子批次 |
| 并行范围 | V1、PP=1、DP=1；TP-only 已运行 TP=2/4；一个 P request；固定 `k=1` |
| MoE 通信 | 当前 TP-only 复用 AllGather；Layered EP 的统一 selector 尚未完成，代码中的 layered AlltoAll override 只能视为过渡实现 |
| 图模式 | D 子批次可 replay `FULL_DECODE_ONLY`；P 子批次强制 eager |
| 验证工具 | 工作区 `verify_layered_prefill_correctness.py` 支持 eager P 参照和 graph D 参照分离比较 |

### 2.2 已有证据和未解决问题

| 配置 | 结果 | 判断 |
| --- | --- | --- |
| TP=2，2103-token，多 group，D-only graph | P 与 eager reference 一致，D 与 graph baseline 一致，确认 graph replay | 可进入性能基线 |
| TP=4，633-token，2-group，D-only graph | 生成文本正常，确认 D graph replay | 可进入短输入性能基线，但仍需 logits/KV 自动验收 |
| TP=4，2103-token，2-group，全 eager | P 输出与普通 eager baseline 不一致，D 输出一致 | Phase 1 精度阻塞项 |
| TP=4，2103-token，5-group，D-only graph | P 输出不一致，并有一个 D 输出与普通 graph baseline 不一致 | 需要同时检查 P frontier 和 D/P view/batch invariance |

“文本正常”不是最终正确性标准。正式验收应同时检查最终 token、top-1、logits 容差、layer/KV 执行次数和跨请求状态隔离。

### 2.3 尚未实现或未验收

- P/D 单 forward 和逐层 row compaction。
- PP>1 的 global plan、frontier owner 和跨 stage 转发。
- DP=1、EP=2/4 多进程正确性、标准通信 selector、active shape 和 collective-order 验证。
- DP>1 的 cohort/plan 同步和 frontier 归属。
- 单次 forward 后的 FusedMC2；非 fused MC2 要先覆盖标准 selector 选择的路径。
- 自适应 `k_t`、group cost model 和 Decode slack 控制。
- P group ACLGraph。
- Prefix Cache、KV pool/offload、PD-disaggregated connector 和其他首发互斥特性。

## 3. 执行原则

1. 正确性和性能使用不同门禁。已验证配置可以开始测性能，但不能用局部性能结果替代 Phase 1 正确性验收。
2. 先定位 TP=4 长输入差异，再扩大 Prompt、group 和并发矩阵；不要把数值问题归因于尚未实现的 P graph。
3. EP 首发固定 DP=1、PP=1、eager、`k=1`，通信后端由普通 vLLM-Ascend selector 决定；Layered Prefill 不拥有独立的 AlltoAll/MC2 选择逻辑。
4. 性能数据必须拆分 D graph、P eager、双 model-call、attention、MoE dispatch、selector 结果和通信成本，避免把参考实现的固定开销误判为算法上限。
5. FusedMC2 要等单次 forward 的 active-row/global-batch 语义稳定；PP>1 是验证“降低 PP bubble”目标的后续必要里程碑。
6. 兼容性默认 fail-closed。未进入当前里程碑的特性可以继续拒绝启动，不要求一次性全部放开。
7. 每个里程碑都保留普通 full/chunked prefill fallback；不得静默让 P 请求执行错误的全层或 partial-layer 路径。

## 4. 里程碑和验收

### M0：Phase 1 正确性和资源闭环

优先级：P0。允许与 M1 中已验证配置的性能测量并行。

| ID | 状态 | 工作项 | 验收条件 |
| --- | --- | --- | --- |
| M0.1 | 进行中 | TP=4 长输入精度定位 | 记录每个 group 边界 hidden/residual、最终 logits/top-k；定位首次偏差层；eager P 与 eager baseline、graph D 与 graph baseline 达到既定数值容差且 greedy token 一致 |
| M0.2 | 待办 | layer/KV 恰好一次验证 | 使用 layer/attention hook 证明每个 P token 的每层只执行和写 KV 一次，中间 group 不产生 norm/logits/sample/KV-complete 事件 |
| M0.3 | 待办 | 请求生命周期 E2E | request reorder、finish、abort、OOM、preempt、重启后 frontier 不泄漏、不串请求，恢复时从 group 0 安全重算 |
| M0.4 | 待办 | KV allocator 和 completion | Prompt block 只分配一次，后续 group 只复用；partial KV 不被 Decode、Prefix Cache 或 connector 视为完整 |
| M0.5 | 待办 | frontier HBM 门禁 | 记录 hidden/residual 峰值和 request 归属；超过预算时回退普通 Prefill，并清理 frontier、KV reservation 和 layered 状态 |
| M0.6 | 待办 | 正确性矩阵自动化 | 覆盖 P-only、D-only、P+D，TP=1/2/4，633/2103-token，2/4/5 groups，eager/decode-graph，连续多请求 |

M0 退出条件：上述项目全部通过；TP=4 长输入不存在未解释的 token 差异；失败路径不会遗留 partial state。bit-exact 不是跨执行后端的强制条件，但容差必须先由普通 eager/graph baseline 标定，不能为 Layered Prefill 单独放宽。

### M1：TP=2/4 收益基线和决策

优先级：P0。先使用 M0 已经验证的配置，不等待 P graph。

测试至少包含以下对照：

1. 普通 Prefill + 全 eager，作为语义参照，不用于评价 Decode TBT 上限。
2. 普通 Prefill + Decode graph，作为主要性能 baseline。
3. Layered Prefill + D graph/P eager，作为当前候选实现。
4. `N_lg=1` 或普通 chunked prefill，检查 Layered 调度和双调用的固定开销。

需要输出：

- TTFT、TBT mean/p95/p99、E2E、吞吐和 SLO attainment。
- D-only graph 时间、P group eager 时间、两次 model-call/host launch 时间和 mixed-step 总时间。
- Attention、MoE dispatch/combine、TP collective、HBM 峰值和 graph replay 证据。
- TP=2/4，短/长 Prompt，不同 Decode batch 和 P/D 到达比例。

M1 决策条件：

- 如果当前实现已有稳定收益，进入 M2，并保留当前路径作为回归 baseline。
- 如果收益不明显，但 P eager 或双 model-call 是主要额外成本，先完成 M2 后复测，不直接否定算法。
- 如果 M2 后在目标 MoE workload 上仍无稳定收益，暂停 PP/DP 扩展，重新评估模型、layer layout 和 workload 假设。
- 只有 profiling 明确显示 P eager 的可捕获开销是主要瓶颈，才建立 P graph 工作项；不得仅因“eager 较慢”跳过分项证据。

### M2：DP=1 的 TP/EP eager + 统一通信

依赖：M0 通过；可与 M1 的已验证配置性能测量并行，但顺序上先于单次 forward、FusedMC2、PP 和 DP>1。

- 配置固定为 V1、PP=1、DP=1、EP=2/4，TP 按模型支持覆盖 1/2/4，使用 eager、一个 P cohort、`k=1`；验证脚本必须真正打开 expert parallel，不能继续使用 `--no-enable-expert-parallel`。
- 删除/禁用 `get_layered_prefill_moe_comm_override()` 一类按 layered 强制 AlltoAll 的临时逻辑。ordinary forward 和 layered forward 的每一次 MoE 调用都走同一个 `select_moe_comm_method`（或上游等价 selector），复用相同硬件、容量、padding 和 fused 开关判断。
- 当前 `platform.py` 对 layered + MC2/FusedMC2 仍有启动拒绝门禁：Phase 2 只解除非 fused MC2 的 layered 专用拒绝，并让 selector 决定是否使用；FusedMC2 的门禁保留到 M4，不能通过配置绕过其 active-row/global-batch 契约。
- D/P 子批次的 `num_tokens` 可以不同，因此 selector 可能自然选择 AllGather、非 fused MC2 或 AlltoAll；记录实际 `comm_type`，不以 layered 标志改写结果。
- 覆盖 EP=1/2/4、633/2103 tokens、2/4/5 groups；校验 logits、文本、routing shape、AlltoAll/MC2 split、collective 顺序和无 hang。
- 非 fused MC2 在标准 selector 选中时一并验证；FusedMC2 不在本里程碑实现。

退出条件：EP 多进程与 EP=1 在既定容差内一致，所有 rank 的 `(step, group, layer, comm_type, active_tokens)` 计划一致，失败时快速报错而不是 collective 超时。

### M3：P/D 单 forward 和 row compaction

依赖：M0 通过；M2 的 EP selector/collective 基线稳定。

- 建立单个 `LayeredBatchView`，保留稳定的 request 到 row 映射。
- 每层执行 `D rows | active P rows`，inactive P rows 不进入 attention、router、expert 或 KV write；为 Attention metadata 和 `slot_mappings_by_layer` 增加 active-row/compaction 语义。
- 保证 logits/sample 只覆盖所有 D rows 和 final-group P rows，frontier、KV 和 collective 顺序与双子批次路径一致。
- 对比双子批次路径的 logits、KV、通信后端/次数、TBT 和 host/kernel 调度开销。

退出条件：双子批次与单 forward 在既定容差内语义一致；无重复 KV；报告能量化去掉双 model-call 的收益。

### M4：FusedMC2

依赖：M3 的单 forward active-row/global-batch 语义稳定。

- 固定最终混合 batch 的 `global_bs`、padding、`x_active_mask`、input/output split 和输出压缩规则。
- 只让标准 selector 在设备/容量条件满足时选择 FusedMC2；失败或不支持时回退普通 MC2/AlltoAll。
- 覆盖 TP/EP=2/4、不同 active-row 比例、量化/非量化和 graph/eager 边界；不新增 layered 专用通信分支。

退出条件：FusedMC2 与普通 selector 路径的 logits/KV/routing 一致，且在目标 workload 上有可重复的通信或时延收益。

### M5：PP=2/4 的 stage-aligned MVP

依赖：M0 通过；建议 M3 通过后开始，M4 可按设备可用性并行。

- group 边界先与 PP stage 对齐，所有 rank 使用同一个 global plan；定义 frontier owner 和 P row 注入/转发。
- 保持普通 PP 一收一发、P eager、`k=1`，暂时关闭 async/DBO；EP 是否在 PP 首发开启取决于 M2 的 collective 证据。
- 覆盖 PP=2/4、不同合法 layer partition、P-only/D-only/P+D、取消/抢占/finish，并记录各 PP rank step time、max-min、idle/bubble ratio。

退出条件：多 rank 计划和 collective 顺序一致，文本/logits/KV 正确，并证明 PP stage bubble 相对 baseline 的变化。

### M6：自适应 `k_t`

依赖：M5 提供稳定的 per-rank timing。

- 实现 `LayerGroupCostModel`，按 layout、Prompt 长度和 Decode batch 建立有限桶。
- 使用最慢 PP rank 的 Decode slack 选择连续 `k_t`，运行时以 EMA 受控校准；超预算时减小，连续低于预算时增加。
- 无 Decode、短 Prompt、冷启动、数据不足或模型不支持时回退 `k=1`/普通 chunked，避免热路径逐 group NPU 到 CPU 同步。

退出条件：目标 workload 的 TBT p99 不超过预设 SLO，且相对固定 `k=1` 在 TTFT、吞吐或 PP bubble 上有稳定收益。

### M7：DP>1 及 DP+EP

依赖：M3 的 active-row 语义和 M5 的 global plan 稳定；M2 的标准 selector 已覆盖 EP。

- 先实现 DP-only：定义 cohort admission、plan 广播、request/frontier rank 归属和迁移策略；禁止未定义的 partial-frontier 跨 rank 迁移。
- 再组合 EP=1/2/4；所有 DP/TP/EP rank 的 global plan、active token shape 和 collective 顺序必须一致。
- 在已验证的 MC2/AlltoAll 路径上测量 DP/EP 收益；dynamic EPLB、SP/PCP/DCP 后置。

退出条件：DP/EP 多进程正确性通过，收益报告能区分 TP collective、EP 通信和 Layered 调度贡献。

### M8：条件优化和生态兼容

以下项目不阻塞 M0-M7，应按 profiling 和实际部署需求逐项立项：

- P group ACLGraph/compile capture。图 key 至少包含 layout、group range、P/D row shape 和 PP/EP communicator mode，并保留 eager fallback。
- DeepSeek MLA、FA3/SFA、量化、DCP、shared expert 和其他模型/kernel 组合。
- Prefix Cache、KV pool/offload、recompute 和 layer completion fencing。
- PD-disaggregated connector 的 partial-KV completion 协议；hidden frontier 默认仍留在 P engine。
- async scheduling、DBO、Speculative/MTP、Mamba/hybrid、Multimodal、LoRA 和其他 scheduler policy。

## 5. 推荐提交顺序

| 顺序 | 交付物 | 主要仓库 |
| --- | --- | --- |
| 1 | TP=4 分层精度探针和回归用例 | `vllm-ascend`，必要时 `vllm` 模型层 |
| 2 | KV/lifecycle/frontier HBM 测试与 fallback | `vllm` + `vllm-ascend` |
| 3 | TP=2/4 标准性能脚本和报告 | 工作区脚本 + `vllm-ascend` benchmark/test |
| 4 | DP=1、TP/EP=2/4 的统一通信 selector 和非 fused MC2/AlltoAll 多进程验证 | `vllm` + `vllm-ascend` |
| 5 | P/D 单 forward、row compaction 和 attention active rows | `vllm` + `vllm-ascend` |
| 6 | FusedMC2（依赖单 forward 的 global batch/active mask） | `vllm-ascend` |
| 7 | PP=2/4 global plan 和 frontier transport | `vllm` + `vllm-ascend` |
| 8 | measured group cost 和自适应 `k_t` | 优先上游 `vllm` 通用接口 |
| 9 | DP>1 plan 同步、DP+EP 多进程验证和收益矩阵 | `vllm` + `vllm-ascend` |
| 10 | 经 profiling 批准的 P graph/模型/connector 扩展 | 以 NPU 和部署需求拆分 |

每个提交都应默认关闭新行为，并包含失败路径测试。跨仓库协议应优先落在上游通用对象中，Ascend 侧只维护 NPU runner、graph、通信和 kernel 适配，避免复制完整 Scheduler。

## 6. 项目完成定义

Layered Prefill 不能只以“能生成文本”或单个 TP benchmark 宣布完成。目标版本至少应满足：

1. M0 正确性、生命周期、KV 和资源门禁全部通过。
2. M2 DP=1、TP/EP 多进程正确性通过，ordinary/layered 共用 selector，非 fused MC2/AlltoAll 均有可追溯选择和 fallback。
3. M3 单 forward 语义完成，或数据证明双子批次是可接受的正式路径；M4 FusedMC2 仅在其契约满足时启用。
4. PP=2/4 correctness 和 bubble 指标通过，证明设计目标而不只是 PP=1/TP 局部收益。
5. 固定 `k=1` 和自适应 `k_t` 都有安全 fallback。
6. 目标 TP/DP/EP 组合有独立的正确性和收益数据，未支持组合在启动时明确拒绝。
7. P graph 是否实现由 profiling 决定；未实现时 D graph + P eager 仍是受支持且可回退的执行模式。
