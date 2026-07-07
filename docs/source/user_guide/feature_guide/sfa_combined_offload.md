# SFA Combined KV Cache Offload Guide

## Overview

SFA combined KV cache offload is an experimental GLM sparse flash attention
(SFA) path that enables prefill/prefix offload and decode offload at the same
time.

The feature uses `MultiConnector` with two child connectors:

- `AscendStoreConnector` handles prefill and prefix layerwise store/load.
- `SFAKVOffloadConnector` handles SFA decode CPU offload and sparse top-k
  token loading.

The two connectors intentionally stay separate because they use different host
memory pools and different load/store paths. `AscendMultiConnector` fans out
prefill save events to both connectors, routes prefix-cache behavior to
`AscendStoreConnector`, and forwards decode-only hooks to
`SFAKVOffloadConnector`.

## Requirements

- V1 engine.
- SFA sparse attention model, such as GLM models with `index_topk` metadata.
- Layerwise KV cache reuse enabled through `use_layerwise`.
- `additional_config.use_offload=true`.
- Context parallelism is not covered by this combined offload path.
- Sparse C8 indexer cache is not covered by the SFA alias layout.

## Architecture

### Connector responsibilities

`AscendStoreConnector` is responsible for the prefill/prefix cache path. In
combined SFA offload, its decode layerwise save/load path is disabled because
decode cache loading is handled by the SFA decode path. This avoids conflicting
decode loads from the prefix-cache connector.

`SFAKVOffloadConnector` owns the decode CPU cache. It allocates CPU KV blocks
for completed SFA blocks and uses an LRU resident workspace on the NPU for
sparse decode attention. It does not operate on indexer cache groups.

`AscendMultiConnector` keeps both connectors active. For a prefill forward, it
allows both connectors to see the save event. For a decode-only forward, it
skips `AscendStoreConnector` layerwise load/save and keeps the decode path on
`SFAKVOffloadConnector`.

### Hybrid SFA KV cache layout

SFA uses two logical cache roles:

- real KV cache for K/nope and rope/PE values;
- indexer cache for sparse top-k selection.

The combined offload layout keeps one real-KV group and splits the indexer
cache into multiple groups. The indexer group count is inferred from the total
number of SFA layers:

```text
indexer_group_count = total_layers // 4 + 1
```

Layers are assigned to indexer groups by `layer_id % indexer_group_count`. For
example, with 78 layers, `indexer_group_count` is 20:

```text
indexer group 0: layers 0, 20, 40, 60
indexer group 1: layers 1, 21, 41, 61
...
```

This keeps each indexer group on a stable block table and slot mapping. The
real KV group remains separate and continues to use layerwise cache reuse. The
K tensor and indexer tensor share the same underlying storage through the SFA
indexer alias layout, so the indexer view does not require a separate physical
K allocation.

The metadata stays generic:

- `block_table_tensors_by_group`
- `slot_mappings_by_group`

There is no special `indexer_slot_mapping` metadata. SFA forward code selects
the correct block table and slot mapping by KV cache group.

### SFA offload cache tuple

When `use_offload` is enabled, each SFA layer receives a seven-entry KV cache
tuple:

| Index | Tensor | Role |
|---|---|---|
| 0 | `k_cache` | normal reused real-KV K/nope cache |
| 1 | `v_cache` | normal reused rope/PE cache |
| 2 | `dsa_k_cache` | indexer alias view over K storage |
| 3 | `topk_buffer_k` | per-layer LRU resident K buffer for CPU decode offload |
| 4 | `topk_buffer_v` | per-layer LRU resident rope/PE buffer for CPU decode offload |
| 5 | `tail_k_cache` | per-request NPU tail K buffer, two blocks per request |
| 6 | `tail_v_cache` | per-request NPU tail rope/PE buffer, two blocks per request |

The normal KV cache at indices `0:2` remains the write target for the layerwise
reused real-KV path. The LRU resident buffers at indices `3:5` are filled from
the SFA decode CPU cache. The tail buffers at indices `5:7` keep the latest
decode blocks resident on NPU.

### Tail window

The tail window is fixed to two blocks. For each SFA layer, the NPU tail buffer
is allocated per active request:

```text
tail_k_cache: [max_num_seqs * 2, block_size, 1, kv_lora_rank]
tail_v_cache: [max_num_seqs * 2, block_size, 1, qk_rope_head_dim]
```

Every SFA forward scatters newly produced KV tokens into the tail buffers using
the request index and token position:

```text
tail_physical_block = token_to_req * 2 + logical_block_id % 2
tail_slot = tail_physical_block * block_size + position % block_size
```

During decode attention, top-k tokens are split into two regions:

- tokens in the latest two logical blocks read from the NPU tail buffers;
- older tokens read from the CPU offload cache through the LRU resident
  top-k buffers.

This split lets the normal real-KV cache continue layerwise reuse while the
latest decode window remains available for SFA attention.

## Configuration

### Combined prefill and decode offload

The following example enables the combined path with `AscendStoreConnector` and
`SFAKVOffloadConnector` under `MultiConnector`:

```bash
vllm serve /path/to/glm-model \
    --served-model-name glm5 \
    --block-size 128 \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "use_layerwise": true,
            "layerwise_num_shared_buffers": 4,
            "layerwise_independent_layers": [],
            "connectors": [
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "backend": "memcache",
                        "use_layerwise": true,
                        "save_decode_cache": false,
                        "lookup_rpc_port": "0",
                        "layerwise_num_shared_buffers": 4,
                        "layerwise_independent_layers": []
                    }
                },
                {
                    "kv_connector": "SFAKVOffloadConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "use_layerwise": true
                    }
                }
            ]
        }
    }' \
    --additional-config '{
        "use_offload": true,
        "lru_resident_cache_config": {
            "enabled": true,
            "buffer_size": 4096,
            "topk": 2048
        },
        "sfa_kv_offload_cpu_cache_config": {
            "dram_size_gb": 2,
            "cache_budget_ratio": 0.9
        }
    }'
```

`save_decode_cache=false` is recommended for the `AscendStoreConnector` child.
In the combined SFA path, `AscendMultiConnector` also patches this setting for
layerwise AscendStore children so the decode-only forward is not interrupted by
the prefill/prefix connector.

### Full decode only graph mode

Combined SFA offload supports `FULL_DECODE_ONLY` graph mode. Add the graph
configuration to the same command that enables the combined KV transfer config.
Do not add `--enforce-eager` when validating graph mode.

```bash
vllm serve /path/to/glm-model \
    --compilation-config '{
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [16]
    }' \
    --additional-config '{
        "ascend_compilation_config": {
            "enable_npugraph_ex": true,
            "enable_static_kernel": false
        },
        "use_offload": true,
        "lru_resident_cache_config": {
            "enabled": true,
            "buffer_size": 4096,
            "topk": 2048
        },
        "sfa_kv_offload_cpu_cache_config": {
            "dram_size_gb": 2,
            "cache_budget_ratio": 0.9
        }
    }'
```

`AscendMultiConnector` ignores the `AscendStoreConnector` piecewise graph
requirement only for this combined SFA decode-offload configuration. If another
child connector requires piecewise graph mode, the normal compatibility check
still applies.

## Configuration Parameters

### KV transfer config

- `kv_connector`: Use `"MultiConnector"` for combined offload.
- `kv_role`: Use `"kv_both"`.
- `use_layerwise`: Enables layerwise cache reuse and SFA hybrid grouping.
- `layerwise_num_shared_buffers`: Number of physical real-KV reuse tensors.
  The combined SFA path currently uses `4`.
- `layerwise_independent_layers`: Layers excluded from layerwise reuse. Use an
  empty list for the current SFA combined path.
- `AscendStoreConnector.backend`: Host backend for the prefill/prefix cache,
  such as `"memcache"`.
- `AscendStoreConnector.save_decode_cache`: Set to `false` in combined mode.
- `SFAKVOffloadConnector.use_layerwise`: Must be `true`.

### Additional config

- `use_offload`: Must be `true` to allocate the SFA offload buffers and enable
  decode offload behavior.
- `lru_resident_cache_config.enabled`: Enables the LRU resident decode
  workspace.
- `lru_resident_cache_config.buffer_size`: Number of resident token slots per
  top-k row. It must be positive, greater than or equal to `topk`, and divisible
  by `block_size`.
- `lru_resident_cache_config.topk`: Number of sparse top-k token indices used
  by SFA decode offload.
- `sfa_kv_offload_cpu_cache_config.dram_size_gb`: CPU DRAM budget per rank for
  the SFA decode offload cache.
- `sfa_kv_offload_cpu_cache_config.cache_budget_ratio`: Fraction of that CPU
  DRAM budget used for the block pool. It must be in `(0, 1]`.

The SFA decode CPU block count is computed from the CPU budget:

```text
cpu_blocks =
    floor(dram_size_bytes * cache_budget_ratio
          / (block_size * (kv_lora_rank + qk_rope_head_dim)
             * dtype_size * num_layers))
```

At least two CPU blocks must be available.

## Memory Notes

The SFA hybrid NPU KV cache uses vLLM's normal `num_blocks` calculation with
the total number of SFA layers. Layerwise reuse then reduces the number of
physical real-KV tensors. This keeps the four reused real-KV tensors within the
profiled NPU memory budget instead of simply multiplying the final block count
after allocation.

The tail buffer is an additional resident NPU copy. Its per-layer memory cost
is approximately:

```text
max_num_seqs * 2 * block_size
    * (kv_lora_rank + qk_rope_head_dim)
    * dtype_size
```

The LRU resident buffers also consume NPU memory:

```text
max_num_topk_rows * buffer_size
    * (kv_lora_rank + qk_rope_head_dim)
    * dtype_size
```

where `max_num_topk_rows` is bounded by `max_num_batched_tokens` and
`max_num_seqs * decode_width`.

## Verification

Check the startup log for these messages:

```text
SFA hybrid KV cache groups: <N> indexer groups + 1 KV group
Layerwise SFA hybrid KV cache reuse
KV offload allocate <N> cpu blocks
CUDAGraphMode.FULL_DECODE_ONLY
```

For `FULL_DECODE_ONLY`, also confirm that the log does not contain an override
from `FULL_DECODE_ONLY` to `PIECEWISE`.

During requests, SFA decode offload logs should show `load_kv_token_wise`.
Successful online serving should report normal `/health`, `/v1/models`, and
`/v1/chat/completions` responses.

## Troubleshooting

- If startup falls back to `PIECEWISE`, check whether another child connector
  still requires piecewise graph mode.
- If `lru_resident_cache_config.buffer_size` validation fails, make sure it is
  greater than or equal to `topk` and divisible by `block_size`.
- If CPU cache allocation fails because too few CPU blocks are available,
  increase `sfa_kv_offload_cpu_cache_config.dram_size_gb` or
  `cache_budget_ratio`. If host memory is exhausted, reduce the configured CPU
  DRAM budget or request concurrency.
- If decode accuracy changes near block boundaries, verify that tail buffer
  updates, logical block rotation, and full-block CPU save sources are aligned.
  The current CPU save path reads the normal reused KV cache, while decode
  attention reads the latest window from the explicit tail buffers.
