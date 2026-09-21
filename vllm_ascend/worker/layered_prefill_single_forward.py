# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Runner-owned row compaction for one mixed layered-prefill forward.

Only common attention metadata is sliced here. Each backend's normal builder
owns the derived metadata, with separate builders for the compact Decode view.
Model-specific state transitions remain in the existing model adapter.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.core.layered_prefill import LayeredFrontier, make_layer_group_ranges

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.worker.layered_prefill_decode_graph import LayeredDecodeGraphCache


class LayeredPrefillSingleForward:
    """Persistent Decode builders; no per-step activations are retained here."""

    def __init__(self, runner):
        self.runner = runner
        self.decode_builders: dict[tuple[int, int], Any] = {}
        self.decode_graph_cache = None

    @staticmethod
    def supported(runner) -> bool:
        parallel = runner.parallel_config
        return (
            parallel.data_parallel_size == 1
            and parallel.pipeline_parallel_size == 1
            and not parallel.enable_dbo
            and runner.dcp_size == 1
            and not runner.use_dcp
            and runner.speculative_config is None
            and not runner.use_async_scheduling
            and not runner.supports_mm_inputs
            and not runner.is_pooling_model
            and not getattr(runner, "_has_gdn", False)
            and not getattr(getattr(runner, "model_config", None), "is_hybrid", False)
            and not runner.sparse_kv_offload_enabled
            and not runner.dynamic_eplb
            and not runner.use_aux_hidden_state_outputs
            and not getattr(runner.cache_config, "kv_sharing_fast_prefill", False)
            and not getattr(
                getattr(runner, "model_config", None),
                "enable_return_routed_experts",
                False,
            )
            and not getattr(runner, "enable_prompt_embeds", False)
            and getattr(runner, "lora_config", None) is None
            and runner._pad_for_sequence_parallelism(runner.max_num_reqs - 1) <= runner.max_num_reqs
            and runner.cache_config.mamba_cache_mode != "align"
        )

    def builder(self, kv_gid, attn_gid, group, original):
        key = (kv_gid, attn_gid)
        if key not in self.decode_builders:
            # Builders own mutable workspaces (slot mappings, compressed-cache
            # metadata, etc.). Sharing them would corrupt the mixed view.
            self.decode_builders[key] = type(original)(
                kv_cache_spec=original.kv_cache_spec,
                layer_names=group.layer_names,
                vllm_config=self.runner.vllm_config,
                device=self.runner.device,
            )
        return self.decode_builders[key]

    def forward_decode(self, plan, start, end, state, input_ids, positions):
        runner = self.runner
        compilation = getattr(runner, "compilation_config", None)
        use_graph = (
            compilation is not None
            and compilation.cudagraph_mode != CUDAGraphMode.NONE
            and not runner.model_config.enforce_eager
            and runner.ascend_config.scheduler_config.layered_prefill_config.single_forward_decode_graph
        )
        adapter = runner.layered_prefill_model_adapter
        if not use_graph:
            for idx in range(start, end):
                state = adapter._forward_layer(adapter.layers[idx], positions, *state, input_ids)
            return state
        if self.decode_graph_cache is None:
            self.decode_graph_cache = LayeredDecodeGraphCache(runner)
        # Use the same boundaries for every active P group, so D groups are
        # reusable across the other steps of a P request, including ramp-up.
        for group in make_layer_group_ranges(adapter.end_layer, plan.num_groups):
            if start <= group.start and group.end <= end:
                state = self.decode_graph_cache.forward(
                    group.start,
                    group.end,
                    state,
                    input_ids,
                    positions,
                )
        return state

    def prepare(self, plan, num_tokens_padded):
        return LayeredPrefillBatch(self, plan, num_tokens_padded)


class LayeredPrefillBatch:
    """Ephemeral D/P row layout and the compact Decode attention metadata."""

    def __init__(self, executor, plan, num_tokens_padded):
        self.executor = executor
        self.runner = runner = executor.runner
        self.plan = plan
        self.num_tokens_padded = num_tokens_padded
        req_ids = runner.input_batch.req_ids
        if len(plan.prefill_req_ids) != 1:
            raise ValueError("Single-forward layered prefill requires one P request")
        self.p_req_id = plan.prefill_req_ids[0]
        self.p_index = req_ids.index(self.p_req_id)
        starts = runner.query_start_loc.cpu[: len(req_ids) + 1].tolist()
        self.p_start, self.p_end = starts[self.p_index : self.p_index + 2]
        self.p_tokens = self.p_end - self.p_start
        if self.p_tokens != plan.query_tokens[self.p_req_id]:
            raise ValueError("Layered P query length does not match prepared inputs")
        self.d_requests = [i for i in range(len(req_ids)) if i != self.p_index]
        self.d_rows = [row for i in self.d_requests for row in range(starts[i], starts[i + 1])]
        if any(starts[i + 1] - starts[i] != 1 for i in self.d_requests):
            raise ValueError("Single-forward D view requires one token per request")
        self.d_tokens = len(self.d_rows)
        self.d_padded = runner._pad_for_sequence_parallelism(self.d_tokens)
        self.d_indices = torch.tensor(self.d_rows, dtype=torch.long, device=runner.device)
        self.d_req_indices = torch.tensor(self.d_requests, dtype=torch.long, device=runner.device)
        self.decode_metadata: dict[str, Any] = {}
        self.decode_ratios: dict[Any, Any] = {}
        if not plan.is_sampling_step:
            mask = runner.discard_request_mask.np[: len(req_ids)].copy()
            mask[self.p_index] = True
            indices = np.flatnonzero(mask)
            runner._restore_layered_sampling_masks((indices, len(indices), mask))

    def _decode_tokens(self, value, *, dim=0, fill=0):
        selected = value.index_select(dim, self.d_indices)
        if self.d_padded == self.d_tokens:
            return selected
        shape = list(selected.shape)
        shape[dim] = self.d_padded - self.d_tokens
        return torch.cat((selected, selected.new_full(shape, fill)), dim=dim)

    def compact_metadata(self, common):
        """Compact the backend-independent contract, never backend metadata."""
        if common.context_parallel_metadata is not None:
            raise ValueError("Single-forward compaction does not support DCP metadata")
        updates = {}
        for name in (
            "seq_lens",
            "seq_lens_cpu",
            "_seq_lens_cpu",
            "seq_lens_cpu_upper_bound",
            "num_computed_tokens_cpu",
            "_num_computed_tokens_cpu",
            "is_prefilling",
            "block_table_tensor",
            "group_len",
            "group_key_idx",
            "group_key_cache_idx",
            "dcp_local_seq_lens",
            "dcp_local_seq_lens_cpu",
            "encoder_seq_lens",
            "encoder_seq_lens_cpu",
            "req_ids_tensor",
        ):
            value = getattr(common, name, None)
            if value is not None:
                selected = (
                    value[self.d_requests] if value.device.type == "cpu" else value.index_select(0, self.d_req_indices)
                )
                if self.d_padded > self.d_tokens:
                    padding = selected.new_zeros((self.d_padded - self.d_tokens, *selected.shape[1:]))
                    selected = torch.cat((selected, padding))
                updates[name] = selected
        # Match the runner's uniform-Decode padding contract: every physical
        # token has a query row, while dummy requests have zero KV length and
        # invalid slots. Leaving padding outside query_start_loc can leave
        # backend output rows unwritten and poison quantization tiles.
        query_cpu = torch.arange(self.d_padded + 1, dtype=common.query_start_loc_cpu.dtype)
        updates.update(
            query_start_loc=query_cpu.to(common.query_start_loc.device, non_blocking=True),
            query_start_loc_cpu=query_cpu,
            num_reqs=self.d_padded,
            num_actual_tokens=self.d_tokens,
            num_input_tokens=self.d_padded,
            max_query_len=1,
            max_seq_len=int(updates["_seq_lens_cpu"].max()),
            actual_seq_lengths_q=list(range(1, self.d_padded + 1)),
            slot_mapping=self._decode_tokens(common.slot_mapping, fill=-1),
            positions=self._decode_tokens(common.positions),
            positions_cpu=(common.positions_cpu[self.d_rows] if common.positions_cpu is not None else None),
            attn_state=type(common.attn_state).DecodeOnly,
            graph_pad_size=-1,
        )
        return replace(common, **updates)

    @contextmanager
    def _context(self, metadata, positions, num_tokens, actual_tokens, layer_start):
        runner = self.runner
        update_cos_sin(positions)
        with set_ascend_forward_context(
            metadata,
            runner.vllm_config,
            num_tokens=num_tokens,
            num_actual_tokens=actual_tokens,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            model_instance=runner.model,
            has_sinks=runner._has_sinks,
        ):
            runner._set_layered_prefill_moe_layer_offset(layer_start)
            yield

    @staticmethod
    def _join(d_state, p_state, p_start, p_end, d_indices, num_tokens):
        result = []
        for d, p in zip(d_state, p_state):
            if d is None and p is None:
                result.append(None)
                continue
            if p is None or (d is None and d_indices.numel()):
                raise ValueError("D/P residual contracts differ at the same layer")
            mixed = p.new_zeros((num_tokens, *p.shape[1:]))
            mixed[p_start:p_end].copy_(p)
            if d is not None:
                mixed.index_copy_(0, d_indices, d[: d_indices.numel()])
            result.append(mixed)
        return tuple(result)

    def _save_frontier(self, state):
        store = self.runner.layered_prefill_state
        old = store.get(self.p_req_id)
        saved = []
        for value, previous in zip(state, (old.hidden_states, old.residual) if old else (None, None)):
            if value is None:
                saved.append(None)
            elif previous is not None and previous.shape == value.shape:
                previous.copy_(value)
                saved.append(previous)
            else:
                saved.append(value.clone())
        store.put(
            LayeredFrontier(
                req_id=self.p_req_id,
                group_id=self.plan.group_id + 1,
                query_len=self.p_tokens,
                hidden_states=saved[0],
                residual=saved[1],
            )
        )

    def forward(self, input_ids, positions, inputs_embeds, mixed_metadata):
        """Traverse each decoder layer at most once with its active token rows."""
        runner, plan = self.runner, self.plan
        adapter = runner.layered_prefill_model_adapter
        if adapter is None:
            raise RuntimeError("Missing layered model adapter")
        adapter._validate_range(plan.group_start, plan.group_end)
        if (plan.group_end == adapter.end_layer) != plan.is_final_group:
            raise ValueError("Layered final-group flag does not match the model layer range")
        frontier = runner.layered_prefill_state.get(self.p_req_id)
        if plan.group_id == 0:
            if frontier is not None:
                raise RuntimeError("Unexpected frontier for the first P group")
            p_state = adapter._prepare_initial_state(
                input_ids=input_ids[self.p_start : self.p_end],
                inputs_embeds=inputs_embeds[self.p_start : self.p_end] if inputs_embeds is not None else None,
                intermediate_tensors=None,
                frontier=None,
            )
        else:
            if frontier is None or frontier.group_id != plan.group_id or frontier.query_len != self.p_tokens:
                raise RuntimeError("Missing or mismatched layered activation frontier")
            p_state = (frontier.hidden_states, frontier.residual)

        d_ids = self._decode_tokens(input_ids) if self.d_tokens else None
        d_positions = self._decode_tokens(positions, dim=positions.ndim - 1) if self.d_tokens else None
        d_state = (None, None)
        if self.d_tokens:
            d_state = adapter._prepare_initial_state(
                input_ids=d_ids,
                inputs_embeds=self._decode_tokens(inputs_embeds) if inputs_embeds is not None else None,
                intermediate_tensors=None,
                frontier=None,
            )

        # Context changes happen only at group boundaries, not every layer.
        # The existing backend/stream dependencies order the work; no device
        # synchronize or independent P and D model call is necessary.
        for start, end, mixed in (
            (adapter.start_layer, plan.group_start, False),
            (plan.group_start, plan.group_end, True),
            (plan.group_end, adapter.end_layer, False),
        ):
            if start == end or (not mixed and not self.d_tokens):
                continue
            state = (
                self._join(
                    d_state,
                    p_state,
                    self.p_start,
                    self.p_end,
                    self.d_indices,
                    self.num_tokens_padded,
                )
                if mixed
                else d_state
            )
            ids, pos = (input_ids, positions) if mixed else (d_ids, d_positions)
            metadata = mixed_metadata if mixed else self.decode_metadata
            padded = self.num_tokens_padded if mixed else self.d_padded
            actual = self.p_tokens + self.d_tokens if mixed else self.d_tokens
            with self._context(metadata, pos, padded, actual, start):
                if mixed:
                    for idx in range(start, end):
                        state = adapter._forward_layer(adapter.layers[idx], pos, *state, ids)
                else:
                    state = self.executor.forward_decode(plan, start, end, state, ids, pos)
                if mixed:
                    p_state = tuple(x[self.p_start : self.p_end] if x is not None else None for x in state)
                    if not plan.is_final_group:
                        self._save_frontier(p_state)
                    if self.d_tokens:
                        d_state = tuple(self._decode_tokens(x) if x is not None else None for x in state)
                else:
                    d_state = state
                if end == adapter.end_layer:
                    # Finalize in the same physical layout/context as the last
                    # layer (important for sequence-parallel final norms).
                    final, _ = adapter._finalize(*state)
                    if mixed:
                        runner.layered_prefill_state.clear(self.p_req_id)
                        return final
                    output = final.new_zeros((self.num_tokens_padded, *final.shape[1:]))
                    output.index_copy_(0, self.d_indices, final[: self.d_tokens])
                    return output
        # A P-only non-final group has no sampled logits.
        return p_state[0]
