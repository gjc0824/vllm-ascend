# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Runner-owned row compaction for one mixed layered-prefill forward.

Only common attention metadata is sliced here. Each backend's normal builder
owns the derived metadata, with separate builders for the compact Decode view.
Model-specific state transitions remain in the existing model adapter.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from vllm.config import CompilationMode, CUDAGraphMode, set_current_vllm_config
from vllm.forward_context import get_forward_context
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
        self.compiled_groups: dict[tuple, Any] = {}
        self.metadata_buffers: dict[tuple, torch.Tensor] = {}
        self.index_buffers: dict[tuple, torch.Tensor] = {}

    def indices(self, length, *, dtype, device):
        key = (dtype, device)
        value = self.index_buffers.get(key)
        if value is None or value.numel() < length:
            capacity = max(length, getattr(self.runner, "max_num_reqs", length) + 1)
            value = torch.arange(capacity, dtype=dtype, device=device)
            self.index_buffers[key] = value
        return value[:length]

    def metadata_buffer(self, key, value, rows):
        shape = (rows, *value.shape[1:])
        target = self.metadata_buffers.get(key)
        if (
            target is None
            or target.shape[1:] != value.shape[1:]
            or target.shape[0] < rows
            or target.dtype != value.dtype
            or target.device != value.device
        ):
            capacity = max(rows, getattr(self.runner, "max_num_reqs", rows))
            target = value.new_empty((capacity, *shape[1:]))
            self.metadata_buffers[key] = target
        return target[:rows]

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

    def forward_mixed(self, start, end, state, input_ids, positions):
        runner = self.runner
        adapter = runner.layered_prefill_model_adapter
        compilation = getattr(runner, "compilation_config", None)
        use_compile = (
            getattr(compilation, "mode", None) == CompilationMode.VLLM_COMPILE
            and not runner.model_config.enforce_eager
            and runner.ascend_config.scheduler_config.layered_prefill_config.single_forward_compile
        )
        if not use_compile:
            for idx in range(start, end):
                state = adapter._forward_layer(adapter.layers[idx], positions, *state, input_ids)
            return state

        # Lazy import: compilation infrastructure is not needed by fallback
        # runners. Compilation does not add a warmup forward on live KV caches.
        from vllm.compilation.backends import set_model_tag

        from vllm_ascend.worker.layered_prefill_compiled import (
            LayeredPrefillCompiledGroup,
        )

        # Python-side communication dispatch must not be frozen from a small
        # MC2 batch and reused for a large AllGather batch (or conversely).
        comm = str(get_forward_context().moe_comm_type)
        key = (start, end, comm, state[1] is None)
        group = self.compiled_groups.get(key)
        if group is None:
            # The AOT decorator hashes the function and config, not module
            # attributes or backend prefix. Isolate each layer/comm contract
            # in BOTH caches without mutating the full model's configuration.
            config = copy(runner.vllm_config)
            config.additional_config = dict(config.additional_config or {})
            config.additional_config["layered_prefill_compile_key"] = key
            config.compilation_config = copy(config.compilation_config)
            config.compilation_config.cache_dir = ""
            config.compilation_config.local_cache_dir = None
            config.compilation_config.traced_files = set(config.compilation_config.traced_files)
        else:
            config = group.vllm_config
        with (
            set_current_vllm_config(config),
            set_model_tag(f"layered_prefill_{start}_{end}_{comm}_{state[1] is None}"),
        ):
            if group is None:
                group = LayeredPrefillCompiledGroup(
                    adapter=adapter,
                    start=start,
                    end=end,
                    vllm_config=config,
                )
                self.compiled_groups[key] = group
            return group(*state, positions, input_ids)

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
        self.decode_is_prefix = self.p_index == len(req_ids) - 1
        if self.decode_is_prefix:
            self.d_indices = executor.indices(self.d_tokens, dtype=torch.long, device=runner.device)
            self.d_req_indices = self.d_indices
        else:
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
        selected = (
            value.narrow(dim, 0, self.d_tokens) if self.decode_is_prefix else value.index_select(dim, self.d_indices)
        )
        if self.d_padded == self.d_tokens:
            return selected
        shape = list(selected.shape)
        shape[dim] = self.d_padded - self.d_tokens
        return torch.cat((selected, selected.new_full(shape, fill)), dim=dim)

    def _compact_rows(self, value, key, *, tokens=False, fill=0):
        target = self.executor.metadata_buffer(key, value, self.d_padded)
        if self.decode_is_prefix:
            target[: self.d_tokens].copy_(value[: self.d_tokens])
        elif value.device.type == "cpu":
            rows = self.d_rows if tokens else self.d_requests
            target[: self.d_tokens].copy_(value[rows])
        else:
            indices = self.d_indices if tokens else self.d_req_indices
            torch.index_select(value, 0, indices, out=target[: self.d_tokens])
        if self.d_padded > self.d_tokens:
            target[self.d_tokens :].fill_(fill)
        return target

    def compact_metadata(self, common, kv_cache_gid=0):
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
                updates[name] = self._compact_rows(value, (kv_cache_gid, name))
        # Match the runner's uniform-Decode padding contract: every physical
        # token has a query row, while dummy requests have zero KV length and
        # invalid slots. Leaving padding outside query_start_loc can leave
        # backend output rows unwritten and poison quantization tiles.
        query_cpu = self.executor.indices(
            self.d_padded + 1,
            dtype=common.query_start_loc_cpu.dtype,
            device=common.query_start_loc_cpu.device,
        )
        updates.update(
            query_start_loc=self.executor.indices(
                self.d_padded + 1,
                dtype=common.query_start_loc.dtype,
                device=common.query_start_loc.device,
            ),
            query_start_loc_cpu=query_cpu,
            num_reqs=self.d_padded,
            num_actual_tokens=self.d_tokens,
            num_input_tokens=self.d_padded,
            max_query_len=1,
            max_seq_len=int(updates["_seq_lens_cpu"].max()),
            actual_seq_lengths_q=list(range(1, self.d_padded + 1)),
            slot_mapping=self._compact_rows(
                common.slot_mapping,
                (kv_cache_gid, "slot_mapping"),
                tokens=True,
                fill=-1,
            ),
            positions=self._compact_rows(common.positions, (kv_cache_gid, "positions"), tokens=True),
            positions_cpu=(
                self._compact_rows(common.positions_cpu, (kv_cache_gid, "positions_cpu"), tokens=True)
                if common.positions_cpu is not None
                else None
            ),
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
    def _join(
        d_state,
        p_state,
        p_start,
        p_end,
        d_indices,
        num_tokens,
        *,
        decode_is_prefix=False,
    ):
        result = []
        for d, p in zip(d_state, p_state):
            if d is None and p is None:
                result.append(None)
                continue
            if p is None or (d is None and d_indices.numel()):
                raise ValueError("D/P residual contracts differ at the same layer")
            mixed = p.new_empty((num_tokens, *p.shape[1:]))
            # Real rows are overwritten below; only TP padding needs clearing.
            actual = p_end - p_start + d_indices.numel()
            if num_tokens > actual:
                mixed[actual:].zero_()
            mixed[p_start:p_end].copy_(p)
            if d is not None:
                if decode_is_prefix:
                    mixed[: d_indices.numel()].copy_(d[: d_indices.numel()])
                else:
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
                    decode_is_prefix=self.decode_is_prefix,
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
                    state = self.executor.forward_mixed(start, end, state, ids, pos)
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
