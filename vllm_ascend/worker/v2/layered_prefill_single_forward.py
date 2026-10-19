# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""One MRV2 input preparation and one traversal of the active decoder layers.

Decode metadata is rebuilt through the normal backend builders. Its tensors
and builders are private: preparing the compact view must not overwrite the
mixed view, including backend-owned compressed-cache scratch.
"""

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.v1.core.layered_prefill import LayeredFrontier
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.rotary_embedding import update_cos_sin

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata


class LayeredV2Buffers:
    """Runner-owned storage reused only after the previous step was sampled."""

    def __init__(self):
        self.tensors = {}
        self.row_indices = {}
        self.uniform_indices = {}

    def arange(self, length, dtype, device):
        key = (dtype, str(device))
        value = self.uniform_indices.get(key)
        if value is None or value.numel() < length:
            value = torch.arange(length, dtype=dtype, device=device)
            self.uniform_indices[key] = value
        return value[:length]

    def indices(self, rows, device):
        key = (tuple(rows), str(device))
        cached = self.row_indices.get(key)
        if cached is None:
            cached = torch.tensor(rows, dtype=torch.long, device=device)
            self.row_indices[key] = cached
        return cached

    def empty(self, key, value, shape):
        target = self.tensors.get(key)
        if target is None or target.shape != shape or target.dtype != value.dtype or target.device != value.device:
            target = value.new_empty(shape)
            self.tensors[key] = target
        return target

    def copy(self, key, value):
        target = self.empty(key, value, value.shape)
        target.copy_(value)
        return target

    @staticmethod
    def _contiguous_start(rows, upper_bound):
        count = len(rows)
        if count == 0:
            return 0
        if isinstance(rows, range):
            start = rows.start
            contiguous = count == 1 or rows.step == 1
        else:
            iterator = iter(rows)
            start = next(iterator)
            contiguous = all(row == start + offset for offset, row in enumerate(iterator, 1))
        if contiguous and start >= 0 and start + count <= upper_bound:
            return start
        return None

    def select(self, key, value, rows, padded=None, fill=0):
        count = len(rows)
        padded = count if padded is None else padded
        target = self.empty(key, value, (padded, *value.shape[1:]))
        start = self._contiguous_start(rows, value.shape[0])
        if start is not None:
            target[:count].copy_(value[start : start + count])
        elif count:
            index = self.indices(rows, value.device)
            torch.index_select(value, 0, index, out=target[:count])
        if padded > count:
            target[count:].fill_(fill)
        return target

    def detach_state(self, state):
        return state._replace(
            hidden_states=(None if state.hidden_states is None else self.copy("decode_hidden", state.hidden_states)),
            aux_hidden_states=(
                None
                if state.aux_hidden_states is None
                else [self.copy(("decode_aux", i), value) for i, value in enumerate(state.aux_hidden_states)]
            ),
        )


class LayeredV2SingleForward:
    def __init__(self, runner, executor):
        self.runner = runner
        self.executor = executor
        self.buffers = LayeredV2Buffers()
        self.decode_groups = None

    @staticmethod
    def supports_plan(plan):
        # A one-token P chunk makes DSV4 mixed metadata DecodeOnly. Its
        # cached RoPE views then alias the global runtime buffer overwritten
        # by the compact D builder, even though the builders are private.
        # Keep these steps on the established separate D/P execution path.
        return (len(plan.prefill_req_ids) == 1 or bool(getattr(plan, "boundary_mode", None))) and all(
            int(plan.query_tokens.get(req_id, 0)) > 1 for req_id in plan.prefill_req_ids
        )

    @staticmethod
    def supported(runner, *, allow_pipeline_parallel=False):
        parallel = runner.parallel_config
        model = runner.model_config
        return (
            (allow_pipeline_parallel or parallel.pipeline_parallel_size == 1)
            and parallel.data_parallel_size == 1
            and not getattr(parallel, "enable_dbo", False)
            and getattr(parallel, "decode_context_parallel_size", 1) == 1
            and getattr(parallel, "prefill_context_parallel_size", 1) == 1
            and runner.speculative_config is None
            and not getattr(runner, "use_async_scheduling", False)
            and not getattr(getattr(runner, "scheduler_config", None), "async_scheduling", False)
            and not getattr(runner, "supports_mm_inputs", False)
            and not getattr(model, "is_multimodal_model", False)
            and not getattr(model, "is_hybrid", False)
            and getattr(model, "runner_type", "generate") == "generate"
            and not getattr(model, "enable_return_routed_experts", False)
            and not getattr(runner, "use_aux_hidden_state_outputs", False)
            and not getattr(runner, "enable_prompt_embeds", False)
            and not getattr(runner, "sparse_kv_offload_enabled", False)
            and not getattr(runner, "dynamic_eplb", False)
            and not getattr(runner, "_has_gdn", False)
            and not getattr(
                runner.layered_prefill_model_adapter.backbone,
                "aux_hidden_state_layers",
                (),
            )
            and getattr(runner, "lora_config", None) is None
            and not getattr(parallel, "enable_eplb", False)
            and not getattr(runner.cache_config, "kv_sharing_fast_prefill", False)
            and getattr(runner.cache_config, "mamba_cache_mode", None) != "align"
            and runner._pad_for_sequence_parallelism(runner.max_num_reqs - 1) <= runner.max_num_reqs
        )

    def _private_groups(self):
        if self.decode_groups is None:
            self.decode_groups = []
            for groups in self.runner.attn_groups:
                private = []
                for group in groups:
                    original = group.get_metadata_builder(0)
                    builder = type(original)(
                        kv_cache_spec=original.kv_cache_spec,
                        layer_names=group.layer_names,
                        vllm_config=self.runner.vllm_config,
                        device=self.runner.device,
                    )
                    private.append(
                        SimpleNamespace(
                            layer_names=group.layer_names,
                            get_metadata_builder=lambda index, b=builder: b,
                        )
                    )
                self.decode_groups.append(private)
        return self.decode_groups

    def prepare_decode(self, prepared, plan, *, build_attention_metadata=True):
        """Compact input/common metadata fields, never derived backend metadata."""
        batch = prepared.input_batch
        if len(plan.prefill_req_ids) != 1 and not getattr(plan, "boundary_mode", None):
            raise ValueError("Single-forward requires exactly one prefill request outside boundary mode")
        p_ids = set(plan.prefill_req_ids)
        p_indices = [i for i, req_id in enumerate(batch.req_ids[: batch.num_reqs]) if req_id in p_ids]
        if len(p_indices) != len(p_ids) or not p_indices:
            raise ValueError("Single-forward batch is missing a prefill request")
        if p_indices != list(range(p_indices[0], p_indices[-1] + 1)):
            raise ValueError("Single-forward requires contiguous prefill request rows")
        p_start = int(batch.query_start_loc_np[p_indices[0]])
        p_end = int(batch.query_start_loc_np[p_indices[-1] + 1])
        for index in p_indices:
            if (
                int(batch.query_start_loc_np[index + 1] - batch.query_start_loc_np[index])
                != plan.query_tokens[batch.req_ids[index]]
            ):
                raise ValueError("Single-forward P query length does not match plan")
        requests = [i for i in range(batch.num_reqs) if i not in p_indices]
        if any(int(batch.num_scheduled_tokens[i]) != 1 for i in requests):
            raise ValueError("Single-forward Decode requires one token per request")
        rows = [int(batch.query_start_loc_np[i]) for i in requests]
        if not rows:
            return None, p_start, p_end
        count = len(rows)
        padded = self.runner._pad_for_sequence_parallelism(count)
        b = self.buffers
        # At most the current request/token layouts are retained. Variable
        # prompt lengths must not grow a lifetime cache of index tensors.
        b.row_indices.clear()
        updates = {}
        for name in (
            "idx_mapping",
            "expanded_idx_mapping",
            "expanded_local_pos",
            "seq_lens_cpu_upper_bound",
            "prompt_lens",
        ):
            value = getattr(batch, name)
            updates[name] = None if value is None else b.select(name, value, requests)
        for name in (
            "idx_mapping_np",
            "num_scheduled_tokens",
            "num_computed_tokens_np",
            "prefill_len_np",
            "num_computed_prefill_tokens_np",
            "is_prefilling_np",
            "max_seq_len_np",
        ):
            value = getattr(batch, name)
            updates[name] = None if value is None else value[requests].copy()
        for name in ("input_ids", "positions", "is_padding"):
            updates[name] = b.select(name, getattr(batch, name), rows, padded, fill=(name == "is_padding"))
        updates["seq_lens"] = b.select("seq_lens", batch.seq_lens, requests, padded)
        seq_np = b.select("seq_lens_cpu", torch.from_numpy(batch.seq_lens_np), requests, padded).numpy()
        query_cpu = b.arange(padded + 1, torch.int32, "cpu")
        query_np = query_cpu.numpy()
        query = b.arange(padded + 1, torch.int32, batch.query_start_loc.device)
        updates.update(
            req_ids=[batch.req_ids[i] for i in requests],
            num_reqs=count,
            num_reqs_after_padding=padded,
            num_tokens=count,
            num_tokens_after_padding=padded,
            query_start_loc=query,
            query_start_loc_np=query_np,
            seq_lens_np=seq_np,
            dcp_local_seq_lens=None,
            attn_state=AscendAttentionState.DecodeOnly,
            logits_indices=b.arange(count, torch.long, batch.input_ids.device),
            cu_num_logits=query[: count + 1],
            cu_num_logits_np=query_np[: count + 1],
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
        )
        compact = replace(batch, **updates)
        if not build_attention_metadata:
            # A whole-stage mixed traversal needs only the compact sampling
            # contract. No Decode layers consume backend metadata or KV slots.
            return (
                SimpleNamespace(
                    input_batch=compact,
                    attn_metadata=None,
                    slot_mappings_by_layer=None,
                    rows=rows,
                ),
                p_start,
                p_end,
            )
        blocks = tuple(b.select(("block", i), block, requests, padded) for i, block in enumerate(prepared.block_tables))
        slots = b.empty(
            "slot_mappings",
            prepared.slot_mappings,
            (prepared.slot_mappings.shape[0], padded),
        )
        for i, slot in enumerate(prepared.slot_mappings):
            slots[i].copy_(b.select(("slot", i), slot, rows, padded, fill=-1))
        # All physical decode rows have one query; dummy rows have zero KV
        # length and invalid slots, ensuring every attention output is written.
        metadata = build_attn_metadata(
            attn_groups=self._private_groups(),
            num_reqs=padded,
            num_reqs_actual=count,
            num_tokens=padded,
            query_start_loc_gpu=query,
            query_start_loc_cpu=query_cpu,
            max_query_len=1,
            seq_lens=compact.seq_lens,
            max_seq_len=int(seq_np.max()),
            block_tables=blocks,
            slot_mappings=slots,
            kv_cache_config=self.runner.kv_cache_config,
            seq_lens_np=seq_np,
            positions=compact.positions,
            attn_state=compact.attn_state,
            num_actual_tokens=count,
            num_input_tokens=padded,
            is_prefilling=torch.zeros(padded, dtype=torch.bool),
        )
        decode = SimpleNamespace(
            input_batch=compact,
            attn_metadata=metadata,
            slot_mappings_by_layer=build_slot_mappings_by_layer(slots, self.runner.kv_cache_config),
            rows=rows,
        )
        return decode, p_start, p_end

    @contextmanager
    def context(self, prepared, start):
        runner = self.runner
        batch = prepared.input_batch
        update_cos_sin(batch.positions)
        with set_ascend_forward_context(
            prepared.attn_metadata,
            runner.vllm_config,
            num_tokens=batch.num_tokens_after_padding,
            num_actual_tokens=batch.num_tokens,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(
                num_tokens=batch.num_tokens_after_padding,
                has_lora=False,
                num_active_loras=0,
            ),
            model_instance=runner.model,
            skip_compiled=False,
            moe_comm_token_count=batch.num_tokens,
        ):
            ctx = get_forward_context()
            ctx.slot_mapping = prepared.slot_mappings_by_layer
            ctx.is_padding = batch.is_padding
            runner._set_layered_prefill_moe_layer_offset(start)
            yield

    def save_frontier(self, req_id, plan, state):
        store = self.runner.layered_prefill_state
        old = store.get(req_id)
        saved = []
        for value, previous in zip(state, (old.hidden_states, old.residual) if old else (None, None)):
            if value is None:
                saved.append(None)
            elif (
                previous is not None
                and previous.shape == value.shape
                and previous.dtype == value.dtype
                and previous.device == value.device
            ):
                previous.copy_(value)
                saved.append(previous)
            else:
                saved.append(value.clone())
        store.put(
            LayeredFrontier(
                req_id=req_id,
                group_id=getattr(plan, "resume_group_id", plan.group_id + 1),
                cohort_id=plan.cohort_id if getattr(plan, "boundary_mode", None) else None,
                query_len=state[0].shape[0],
                hidden_states=saved[0],
                residual=saved[1],
            )
        )

    def forward(self, prepared, plan):
        if len(plan.prefill_req_ids) != 1 or not self.supports_plan(plan):
            raise ValueError("Single-forward requires one P request with more than one query token")
        runner, adapter = self.runner, self.runner.layered_prefill_model_adapter
        adapter._validate_range(plan.group_start, plan.group_end)
        if (plan.group_end == adapter.end_layer) != plan.is_final_group:
            raise ValueError("Single-forward final-group flag does not match layer range")
        decode, p_start, p_end = self.prepare_decode(prepared, plan)
        batch = prepared.input_batch
        p_id = plan.prefill_req_ids[0]
        frontier = runner.layered_prefill_state.get(p_id)
        if plan.group_id == 0:
            if frontier is not None:
                raise RuntimeError("Unexpected frontier for first P group")
            p_state = adapter._prepare_initial_state(
                input_ids=batch.input_ids[p_start:p_end],
                frontier=None,
                inputs_embeds=None,
                intermediate_tensors=None,
            )
        else:
            if frontier is None or frontier.group_id != plan.group_id or frontier.query_len != p_end - p_start:
                raise RuntimeError("Missing or mismatched layered activation frontier")
            p_state = (frontier.hidden_states, frontier.residual)
        d_state = (None, None)
        if decode is not None:
            d_state = adapter._prepare_initial_state(
                input_ids=decode.input_batch.input_ids,
                frontier=None,
                inputs_embeds=None,
                intermediate_tensors=None,
            )
        final = None
        for start, end, mixed in (
            (adapter.start_layer, plan.group_start, False),
            (plan.group_start, plan.group_end, True),
            (plan.group_end, adapter.end_layer, False),
        ):
            if start == end or (not mixed and decode is None):
                continue
            active = prepared if mixed else decode
            if mixed:
                joined = []
                for i, (d, p) in enumerate(zip(d_state, p_state)):
                    if d is None and p is None:
                        joined.append(None)
                        continue
                    if p is None or (d is None and decode is not None):
                        raise ValueError("D/P residual contracts differ at the same layer")
                    value = self.buffers.empty(("joined", i), p, (batch.num_tokens_after_padding, *p.shape[1:]))
                    value[p_start:p_end].copy_(p)
                    value[batch.num_tokens :].zero_()
                    if decode is not None:
                        if decode.rows == list(range(len(decode.rows))):
                            value[: len(decode.rows)].copy_(d[: len(decode.rows)])
                        else:
                            indices = self.buffers.indices(decode.rows, value.device)
                            value.index_copy_(0, indices, d[: len(decode.rows)])
                    joined.append(value)
                state = tuple(joined)
            else:
                state = d_state
            with self.context(active, start):
                ids, positions = (
                    active.input_batch.input_ids,
                    active.input_batch.positions,
                )
                state = (
                    self.executor.forward_mixed(start, end, state, ids, positions)
                    if mixed
                    else self.executor.forward_decode(plan, start, end, state, ids, positions)
                )
                if mixed:
                    p_state = tuple(None if x is None else x[p_start:p_end] for x in state)
                    if not plan.is_final_group:
                        self.save_frontier(p_id, plan, p_state)
                    if decode is not None and not plan.is_final_group:
                        d_state = tuple(
                            None
                            if x is None
                            else self.buffers.select(
                                ("activation", i),
                                x,
                                decode.rows,
                                decode.input_batch.num_tokens_after_padding,
                            )
                            for i, x in enumerate(state)
                        )
                else:
                    d_state = state
                if end == adapter.end_layer:
                    final, _ = adapter._finalize(*state)
                    if mixed:
                        runner.layered_prefill_state.clear(p_id)
        # Return final output in its natural layout; the caller chooses the
        # combined sampler only for the final group of the final prompt chunk.
        if decode is not None and plan.is_final_group and not plan.is_sampling_step:
            final = self.buffers.select(
                "decode_final",
                final,
                decode.rows,
                decode.input_batch.num_tokens_after_padding,
            )
        return final, decode
