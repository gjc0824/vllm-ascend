# Single-forward layered prefill

## Execution contract

For a supported layered scheduler step, the V1 runner prepares one persistent
batch and invokes one layer traversal. Decode rows execute every decoder layer;
Prefill rows join that traversal only over the scheduled layer range. No decoder
layer is evaluated twice in a step.

The existing model adapter supplies embedding, layer invocation, and final
normalization/hyper-connection transitions. Model forward files and attention
implementations are unchanged. Input preparation, request lifecycle updates,
attention metadata builders, logits selection, and sampling reuse the normal
runner paths.

The runner keeps two metadata views:

- The original mixed batch for the active Prefill layer group.
- A compact Decode view built from common attention metadata by independent,
  cached backend builders. Derived backend metadata is never manually sliced.

Decode padding follows the existing uniform-decode contract. Physical padding
has query rows, zero sequence lengths/block tables, and invalid KV slots. The
builder still receives the real request count. Omitting those query rows can
leave padded attention outputs unwritten and contaminate real quantization rows.

Intermediate Prefill rows use the normal sampling discard mask. Frontiers contain
only real Prefill activations, are keyed by request ID, and reuse compatible
storage. Completion and preemption use the existing frontier cleanup.

## Decode group graphs

Without graphs, inactive-P Decode layers add substantial eager launch overhead.
`LayeredDecodeGraphCache` records those layer groups using their ordinary eager
operators. This does not register or replace the full-model attention graph
update handles. Pure Decode steps retain the normal full-model graph path.

Graph keys include exact tensor layouts, alias structure, all host-side values,
layer bounds, and MoE communication method. CPU metadata changes cannot replay
an incompatible graph. Graph inputs have private stable storage. Metadata Python
wrappers are copied; runner-owned Decode tensor storage is retained strongly and
refreshed when input addresses change. This avoids duplicating large constant
tables and does not alias the mixed builders' mutable workspaces.

Outputs retain strong references while graphs share a pool. The cache is bounded
to 512 graph entries and 64 metadata layouts. Unsupported metadata and new shapes
beyond the limit execute eagerly. KV-cache initialization invalidates the cache.

Capture records work and the first invocation replays it once. There is no warmup
forward on live request KV/state caches. The graph capture context synchronizes
on a cache miss; there is no added device-wide D/P boundary synchronization on
steady-state replay. Cold-shape capture cost must be reported separately from
steady-state performance.

## Configuration

The fields are under `additional_config.scheduler_config.layered_prefill_config`:

| Field | Default | Meaning |
| --- | --- | --- |
| `single_forward` | `true` | Use row compaction on supported steps. `false` preserves the split D/P reference path. |
| `single_forward_decode_graph` | `true` | Cache Decode layer groups when graph execution is enabled. `false` uses eager groups. |

For the tested TP4 configuration, use `max-num-seqs=64`, sufficient token budget
for the complete P query plus D rows (16448 for 16K input), and
`compilation_config.cudagraph_mode="FULL_DECODE_ONLY"` with
`require_eager=false`.

The single-forward implementation is restricted to synchronous DP1/PP1,
non-speculative, decoder-only execution with one active P request and one token
per D request. DBO, independent PCP/DCP, LoRA, hybrid state, prompt embeddings,
auxiliary/routed-expert outputs, KV-sharing fast prefill, and sparse KV offload
are not enabled by this implementation. Existing split execution remains the
fallback within supported platform configurations. Layered prefill is currently
restricted to DP1 at platform validation; the pre-session DP implementation is
archived separately and is not part of these commits. Request capacity must
accommodate TP-aligned Decode padding.

DSA-CP query sharding within TP is distinct from independent PCP/DCP and was
validated on DeepSeek-V4-Flash. Other model/backend combinations require their
own hardware validation; the implementation does not imply universal graph
compatibility.

## Validation and performance status

Unit tests cover 2D residual and 3D hyper-connection states, intermediate/final
groups, padding, metadata ownership, storage reuse, graph group routing, and
configuration parsing. The remote experiment contains single-request, concurrent,
and full 64-request token comparisons plus fixed 16384-input/256-output benchmarks.

Correctness and performance are separate gates. A successful token comparison
must not be described as a throughput improvement. See the experiment report for
the measured results, cold graph capture conditions, failed experiments, and any
remaining acceptance failures.

The 2026-09-20 DeepSeek-V4-Flash TP4 experiment did **not** meet the performance
gate. At 16384 input tokens, 256 output tokens, and concurrency 64, three-run
mean output throughput was 194.88 tokens/s for single-forward Layer[4], versus
228.22 for Chunk4160 and 202.05 for split-forward Layer[4]. This path remains
experimental; it is not a recommended throughput replacement for Chunk Prefill.
Serial and simultaneous-request fixtures matched, including 64 full 256-token
outputs, but one staggered-arrival fixture differed from its schedule-dependent
baseline. Do not describe this as universal bitwise equivalence.

A subsequent final-source run on idle NPU2-5 produced only 61/64 exact outputs;
its non-layered Chunk baseline also developed concurrent-output anomalies.
That run did not establish a stable acceptance baseline, so final correctness
and performance acceptance remain incomplete. The passing fixtures above refer
specifically to the earlier NPU0-3 run, not to every TP4 card placement.

The full report is `SINGLE_FORWARD_REPORT.md` in the remote experiment directory:
`/home/g00955623/layer_prefill/experiments/single_forward_20260920/`.
