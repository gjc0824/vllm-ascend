# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""One traversal of a PP stage with a shared D/P layer group."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.cudagraph_utils import get_uniform_token_count
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.model_runner import ExecuteModelState, sort_batch_req_ids

from vllm_ascend.worker.v2.layered_prefill import (
    LayeredV2ExecuteModelState,
    subset_scheduler_output,
)
from vllm_ascend.worker.v2.layered_prefill_single_forward import LayeredV2SingleForward


class LayeredV2PPStageForward:
    """Keep the ordinary D/P wire and sampling contracts outside the stage."""

    def __init__(self, runner):
        self.runner = runner
        self.views = runner._layered_single_forward_executor

    @staticmethod
    def supported(runner, plan, d_req_ids, scheduler_output):
        adapter = runner.layered_prefill_model_adapter
        boundary = bool(getattr(plan, "boundary_mode", None))
        local_start = max(adapter.start_layer, plan.group_start)
        local_end = min(adapter.end_layer, plan.group_end)
        return (
            bool(getattr(runner, "_single_forward", False))
            and not getattr(runner, "_fuse_mixed_batch", False)
            and not getattr(runner, "_same_layer_batch", False)
            and runner.parallel_config.pipeline_parallel_size == 2
            and 2 <= plan.num_groups <= adapter.num_hidden_layers
            and 0 <= plan.group_id < plan.num_groups
            and bool(d_req_ids)
            and LayeredV2SingleForward.supports_plan(plan)
            and LayeredV2SingleForward.supported(runner, allow_pipeline_parallel=True)
            and getattr(runner.vllm_config, "kv_transfer_config", None) is None
            and not getattr(runner, "is_encoder_decoder", False)
            and local_start < local_end
            and (
                boundary
                or (
                    adapter.start_layer <= plan.group_start < plan.group_end <= adapter.end_layer
                    and runner._layered_pp_group_owner(plan) == (0 if runner.is_first_pp_rank else 1)
                )
            )
            and plan.is_final_group == (getattr(plan, "resume_group_id", plan.group_id + 1) == plan.num_groups)
            and plan.is_final_group == (plan.group_end == adapter.num_hidden_layers)
            and all(scheduler_output.num_scheduled_tokens[r] == 1 for r in d_req_ids)
        )

    def decode_wire_tokens(self, count):
        runner = self.runner
        # Native downstream stages copy exactly their dispatched D bucket.
        # Compact mixed-stage D metadata instead pads only to a TP multiple.
        descriptor, _ = dispatch_cg_and_sync_dp(
            runner.cudagraph_manager,
            count,
            count,
            get_uniform_token_count(count, count, 1),
            runner.dp_size,
            runner.dp_rank,
            need_eager=False,
            num_active_loras=0,
        )
        if descriptor.num_tokens < count:
            raise RuntimeError("PP Decode descriptor cannot hold its logical rows")
        return descriptor.num_tokens

    def forward_stage(
        self,
        prepared,
        plan,
        d_intermediate,
        p_intermediate,
        wire_tokens,
        *,
        trace=False,
    ):
        """Execute layers once and own both outputs before any metadata reuse."""
        runner = self.runner
        adapter = runner.layered_prefill_model_adapter
        views = self.views
        local_start = max(plan.group_start, adapter.start_layer)
        local_end = min(plan.group_end, adapter.end_layer)
        partial_stage = (local_start, local_end) != (
            adapter.start_layer,
            adapter.end_layer,
        )
        decode, p_start, p_end = views.prepare_decode(prepared, plan, build_attention_metadata=partial_stage)
        if decode is None:
            raise RuntimeError("PP mixed stage requires Decode rows")
        batch = prepared.input_batch
        d_count = decode.input_batch.num_tokens
        p_count = p_end - p_start
        p_ids = [req_id for req_id in batch.req_ids[: batch.num_reqs] if req_id in plan.prefill_req_ids]

        if runner.is_first_pp_rank:
            d_state = adapter._prepare_initial_state(
                input_ids=decode.input_batch.input_ids,
                frontier=None,
                inputs_embeds=None,
                intermediate_tensors=None,
            )
        else:
            if d_intermediate is None:
                raise RuntimeError("PP mixed stage did not receive Decode activation")
            if any(v.shape[0] != wire_tokens for v in d_intermediate.tensors.values()):
                raise RuntimeError("PP Decode activation does not match native wire width")
            d_state = adapter._prepare_initial_state(
                input_ids=None,
                frontier=None,
                inputs_embeds=None,
                intermediate_tensors=IntermediateTensors(
                    {
                        key: views.buffers.select(
                            ("pp_d_input", key),
                            value,
                            range(d_count),
                            decode.input_batch.num_tokens_after_padding,
                        )
                        if partial_stage
                        else value[:d_count]
                        for key, value in d_intermediate.tensors.items()
                    }
                ),
            )
        _, frontier, embeds, intermediate, source = runner._prepare_layered_p_activation(
            plan,
            req_ids=p_ids,
            num_tokens_padded=p_count,
            inputs_embeds=None,
            intermediate_tensors=p_intermediate,
        )
        runner._record_layered_activation(plan, source)
        if getattr(plan, "boundary_mode", None) and intermediate is not None:
            # The upstream stage sends TP-padded P rows. Mixed metadata owns
            # only logical P rows; drop transport padding before joining D/P.
            if any(value.shape[0] < p_count for value in intermediate.tensors.values()):
                raise RuntimeError("Boundary PP activation is shorter than its logical query")
            intermediate = IntermediateTensors({key: value[:p_count] for key, value in intermediate.tensors.items()})
        p_state = adapter._prepare_initial_state(
            input_ids=batch.input_ids[p_start:p_end],
            frontier=frontier,
            inputs_embeds=embeds,
            intermediate_tensors=intermediate,
        )
        if p_state[0].shape[0] != p_count:
            raise RuntimeError("PP Prefill frontier does not match logical query rows")

        if local_start > adapter.start_layer:
            with views.context(decode, adapter.start_layer):
                d_state = runner._layered_group_executor.forward_decode(
                    plan,
                    adapter.start_layer,
                    local_start,
                    d_state,
                    decode.input_batch.input_ids,
                    decode.input_batch.positions,
                )

        joined = []
        for index, (d_value, p_value) in enumerate(zip(d_state, p_state)):
            if d_value is None and p_value is None:
                joined.append(None)
                continue
            if d_value is None or p_value is None:
                raise RuntimeError("PP D/P residual contracts disagree at stage entry")
            value = views.buffers.empty(
                ("pp_joined", index),
                p_value,
                (batch.num_tokens_after_padding, *p_value.shape[1:]),
            )
            value.zero_()
            value[p_start:p_end].copy_(p_value)
            d_start = views.buffers._contiguous_start(decode.rows, value.shape[0])
            if d_start is not None:
                value[d_start : d_start + d_count].copy_(d_value[:d_count])
            else:
                value.index_copy_(
                    0,
                    views.buffers.indices(decode.rows, value.device),
                    d_value[:d_count],
                )
            joined.append(value)
        with views.context(prepared, local_start):
            group_trace = torch.profiler.record_function("layered_pp_mixed_group") if trace else nullcontext()
            with group_trace:
                state = runner._layered_group_executor.forward_mixed(
                    local_start,
                    local_end,
                    tuple(joined),
                    batch.input_ids,
                    batch.positions,
                )
            if plan.is_final_group and runner.is_last_pp_rank:
                state = adapter._finalize(*state)
            # Own P at its frontier before D continues through the suffix.
            # Only the global final group applies the model's final norm to P.
            p_output = tuple(
                None
                if value is None
                else views.buffers.select(
                    ("pp_p_output", index),
                    value,
                    range(p_start, p_end),
                    runner._pad_for_sequence_parallelism(p_count),
                )
                for index, value in enumerate(state)
            )
            d_output = tuple(
                None
                if value is None
                else views.buffers.select(
                    ("pp_d_output", index),
                    value,
                    decode.rows,
                    decode.input_batch.num_tokens_after_padding if local_end < adapter.end_layer else wire_tokens,
                )
                for index, value in enumerate(state)
            )
        if local_end < adapter.end_layer:
            with views.context(decode, local_end):
                d_output = runner._layered_group_executor.forward_decode(
                    plan,
                    local_end,
                    adapter.end_layer,
                    d_output,
                    decode.input_batch.input_ids,
                    decode.input_batch.positions,
                )
                if runner.is_last_pp_rank:
                    d_output = adapter._finalize(*d_output)
                d_output = tuple(
                    None
                    if value is None
                    else views.buffers.select(("pp_d_wire", index), value, range(d_count), wire_tokens)
                    for index, value in enumerate(d_output)
                )
        retain = not plan.is_final_group and (
            not getattr(plan, "boundary_mode", None)
            or runner._layered_next_group_owner(plan) == (0 if runner.is_first_pp_rank else 1)
        )
        offset = 0
        for p_id in p_ids:
            count = plan.query_tokens[p_id]
            if retain:
                views.save_frontier(
                    p_id, plan, tuple(None if x is None else x[offset : offset + count] for x in p_output)
                )
            else:
                runner.layered_prefill_state.clear(p_id)
            offset += count
        runner._synchronize_layered_compute()
        return decode, d_output, p_output

    def execute(self, scheduler_output, plan, d_req_ids, d_intermediate, p_intermediate):
        runner = self.runner
        wire_tokens = self.decode_wire_tokens(len(d_req_ids))
        prepared = runner._prepare_layered_batch(runner._strip_lifecycle_fields(scheduler_output))
        expected_order = sort_batch_req_ids({req_id: 1 for req_id in d_req_ids}, runner.decode_query_len)
        actual_order = [req_id for req_id in prepared.input_batch.req_ids if req_id in set(d_req_ids)]
        if actual_order != expected_order:
            raise RuntimeError("PP mixed and native Decode request order disagree")
        runner.kv_connector.pre_forward(scheduler_output)
        profiler_config = getattr(runner.vllm_config, "profiler_config", None)
        if getattr(profiler_config, "profiler", None) is None:
            decode, d_output, p_output = self.forward_stage(prepared, plan, d_intermediate, p_intermediate, wire_tokens)
        else:
            ring_plan = getattr(scheduler_output, "ring_step_plan", None)
            label = (
                f"layered_pp_prefill/step={getattr(ring_plan, 'step_id', -1)}"
                f"/group={plan.group_id}"
                f"/layers={plan.group_start}:{plan.group_end}"
                f"/P={','.join(plan.prefill_req_ids)}"
            )
            # Keep the outer label's request identity compatible with the
            # existing overlap analyzer; distinguish mixed work by nesting.
            with (
                torch.profiler.record_function(label),
                torch.profiler.record_function("layered_pp_owner_mixed"),
            ):
                decode, d_output, p_output = self.forward_stage(
                    prepared,
                    plan,
                    d_intermediate,
                    p_intermediate,
                    wire_tokens,
                    trace=True,
                )
        finished = scheduler_output.finished_req_ids
        d_state = ExecuteModelState(
            input_batch=decode.input_batch,
            attn_metadata=decode.attn_metadata,
            slot_mappings_by_layer=decode.slot_mappings_by_layer,
            hidden_states=(
                self.views.buffers.select(
                    "pp_d_sampling",
                    d_output[0],
                    list(range(decode.input_batch.num_tokens)),
                    decode.input_batch.num_tokens_after_padding,
                )
                if runner.is_last_pp_rank
                else None
            ),
            aux_hidden_states=None,
            finished_req_ids=finished,
        )
        if plan.is_sampling_step:
            # Reuse the native P input-batch contract for sampling. Outputs and
            # compact D metadata already own storage before these builders run.
            p_schedule = subset_scheduler_output(
                scheduler_output,
                list(plan.prefill_req_ids),
                layered_plan=plan,
                include_one_time_updates=False,
            )
            with runner._layered_p_buffers():
                p_prepared = runner._prepare_layered_batch(p_schedule, build_attention_metadata=False)
            mixed_p_order = [req_id for req_id in prepared.input_batch.req_ids if req_id in plan.prefill_req_ids]
            if list(p_prepared.input_batch.req_ids) != mixed_p_order:
                raise RuntimeError("PP Prefill sampling order disagrees with activation order")
            p_state = ExecuteModelState(
                input_batch=p_prepared.input_batch,
                attn_metadata=p_prepared.attn_metadata,
                slot_mappings_by_layer=p_prepared.slot_mappings_by_layer,
                hidden_states=p_output[0] if runner.is_last_pp_rank else None,
                aux_hidden_states=None,
                finished_req_ids=finished,
            )
        else:
            p_state = ExecuteModelState(
                input_batch=SimpleNamespace(req_ids=list(plan.prefill_req_ids), num_reqs=len(plan.prefill_req_ids)),
                attn_metadata=None,
                slot_mappings_by_layer=None,
                hidden_states=None,
                aux_hidden_states=None,
                finished_req_ids=finished,
            )
        runner.execute_model_state = LayeredV2ExecuteModelState(
            scheduler_output=scheduler_output,
            d_state=d_state,
            p_state=p_state,
            sample_p=bool(plan.is_sampling_step),
            single_forward=True,
            prefill_req_ids=tuple(plan.prefill_req_ids),
        )
        counters = runner._layered_counters()
        counters["pp_stage_mixed_steps"] = counters.get("pp_stage_mixed_steps", 0) + 1
        if runner.is_last_pp_rank:
            return None

        def intermediate(state):
            return IntermediateTensors(
                {name: value for name, value in zip(("hidden_states", "residual"), state) if value is not None}
            )

        return runner._combine_layered_pp_intermediate(
            intermediate(d_output),
            runner._layered_nonfused_pp_payload(plan, intermediate(p_output)),
        )
