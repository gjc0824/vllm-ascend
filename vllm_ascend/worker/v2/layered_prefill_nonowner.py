# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Transport and sampling metadata for PP stages with no active P layers."""

from types import SimpleNamespace

import numpy as np
from vllm.v1.core.layered_prefill import LayeredForwardOutput
from vllm.v1.worker.gpu.model_runner import ExecuteModelState

from vllm_ascend.worker.v2.layered_prefill import store_req_frontiers
from vllm_ascend.worker.v2.layered_prefill_single_forward import LayeredV2SingleForward


def can_compact_internal_prefill_payload(runner, plan):
    """A shared configuration/plan decision, independent of rank-local execution.

    PP0 retains an internal group's frontier for its own next group. PP1
    first consumes the frontier at the stage handoff, so an internal group
    needs only a one-row transport fence. The row's value has no reader.
    Do not use local adapter capabilities, D presence or frontier state here:
    the sender and receiver must make the same transport decision.
    """
    parallel = getattr(runner, "parallel_config", None)
    config = getattr(
        getattr(getattr(runner, "ascend_config", None), "scheduler_config", None),
        "layered_prefill_config",
        None,
    )
    extensions = getattr(
        getattr(getattr(runner, "ascend_config", None), "scheduler_config", None),
        "layered_prefill_extensions",
        None,
    )
    scheduler = getattr(runner, "scheduler_config", None)
    owner = getattr(runner, "_layered_pp_group_owner", None)
    next_owner = getattr(runner, "_layered_next_group_owner", None)
    return bool(
        getattr(config, "enabled", False)
        and not getattr(config, "fuse_mixed_batch", False)
        and not getattr(config, "same_layer_batch", False)
        # The P-only relay shortcut returns before waiting on received
        # tensors. Keep its existing transport contract outside this path.
        and (getattr(plan, "boundary_mode", None) or not getattr(extensions, "skip_relay_stages", False))
        and getattr(parallel, "pipeline_parallel_size", 1) == 2
        and getattr(parallel, "data_parallel_size", 1) == 1
        and not getattr(parallel, "enable_dbo", False)
        and scheduler is not None
        and not getattr(scheduler, "async_scheduling", True)
        and getattr(runner, "speculative_config", None) is None
        and getattr(runner.vllm_config, "kv_transfer_config", None) is None
        and getattr(plan, "num_groups", 0) >= 2
        and 0 <= getattr(plan, "group_id", -1) < plan.num_groups - 1
        and 0 <= getattr(plan, "group_start", -1) < getattr(plan, "group_end", 0)
        and not plan.is_final_group
        and not plan.is_sampling_step
        and (len(plan.prefill_req_ids) == 1 or bool(getattr(plan, "boundary_mode", None)))
        and all(
            plan.query_tokens[req_id] > (0 if getattr(plan, "boundary_mode", None) else 1)
            for req_id in plan.prefill_req_ids
        )
        and callable(owner)
        and callable(next_owner)
        and owner(plan) == 0
        and next_owner(plan) == 0
    )


def run_internal_prefill_fence(runner, scheduler_output, plan, intermediate_tensors):
    """Wait for a one-row P fence, discard its value, and preserve lifecycle."""
    if not can_compact_internal_prefill_payload(runner, plan) or runner.is_first_pp_rank:
        return None
    req_ids = list(plan.prefill_req_ids)
    queries = {req_id: plan.query_tokens[req_id] for req_id in req_ids}
    if scheduler_output.num_scheduled_tokens != queries or scheduler_output.total_num_scheduled_tokens != sum(
        queries.values()
    ):
        raise RuntimeError("Internal P fence schedule disagrees with its plan")
    if intermediate_tensors is None:
        raise RuntimeError("Internal P fence requires one-row wire metadata")
    tensors = intermediate_tensors.tensors
    if "hidden_states" not in tensors or any(value.ndim == 0 or value.shape[0] != 1 for value in tensors.values()):
        raise RuntimeError("Internal P fence requires one-row activation tensors")
    runner.kv_connector.pre_forward(scheduler_output)
    runner._record_layered_activation(plan, "internal_fence")
    return (
        ExecuteModelState(
            input_batch=SimpleNamespace(req_ids=req_ids, num_reqs=len(req_ids)),
            attn_metadata=None,
            slot_mappings_by_layer=None,
            hidden_states=None,
            aux_hidden_states=None,
            finished_req_ids=scheduler_output.finished_req_ids,
        ),
        LayeredForwardOutput(
            tensors["hidden_states"][:0],
            None if tensors.get("residual") is None else tensors["residual"][:0],
            False,
        ),
    )


def run_nonowner_prefill(runner, scheduler_output, plan, intermediate_tensors):
    """Return None outside the validated two-stage, stage-local group envelope.

    Decode execution, output ownership, stream completion and the sampled-token
    collective remain in the caller. This only removes P work on an empty local
    layer range; it never substitutes another request's activation.
    """
    # Boundary plans use the general local-range path, including one-token chunks.
    if getattr(plan, "boundary_mode", None):
        return None
    if (
        runner.parallel_config.pipeline_parallel_size != 2
        or getattr(runner, "_fuse_mixed_batch", False)
        or getattr(runner, "_same_layer_batch", False)
        or not LayeredV2SingleForward.supports_plan(plan)
        or not LayeredV2SingleForward.supported(runner, allow_pipeline_parallel=True)
        or getattr(runner.vllm_config, "kv_transfer_config", None) is not None
        or getattr(runner, "is_encoder_decoder", False)
    ):
        return None
    num_groups = int(plan.num_groups)
    group_id = int(plan.group_id)
    group_start = int(plan.group_start)
    group_end = int(plan.group_end)
    if (
        num_groups < 2
        or not 0 <= group_id < num_groups
        or group_start < 0
        or group_end <= group_start
        or bool(plan.is_final_group) != (group_id + 1 == num_groups)
    ):
        return None

    owner = runner._layered_pp_group_owner(plan)
    rank = 0 if runner.is_first_pp_rank else 1
    # Stage ownership comes from actual layer partitions, including uneven
    # partitions and odd group counts; group indices do not identify a rank.
    if owner not in (0, 1) or rank == owner:
        return None
    adapter = runner.layered_prefill_model_adapter
    if (owner == 0 and group_end > adapter.start_layer) or (owner == 1 and group_start < adapter.end_layer):
        return None
    req_ids = list(plan.prefill_req_ids)
    req_id = req_ids[0]
    query = int(plan.query_tokens[req_id])
    if (
        set(scheduler_output.num_scheduled_tokens) != {req_id}
        or scheduler_output.num_scheduled_tokens[req_id] != query
        or scheduler_output.total_num_scheduled_tokens != query
    ):
        raise RuntimeError("Non-owner P schedule disagrees with its layer plan")
    sample_p = bool(plan.is_sampling_step)
    if rank == 1 and sample_p:
        raise RuntimeError("Non-owner final PP stage cannot sample an unfinished group")

    if sample_p:
        # Non-last GPUModelRunner.sample_tokens needs the normal InputBatch for
        # receive masks and postprocess_num_computed_tokens, not attention data.
        batch = runner._prepare_layered_batch(scheduler_output, build_attention_metadata=False).input_batch
    else:
        batch = SimpleNamespace(req_ids=req_ids, num_reqs=len(req_ids))

    runner.kv_connector.pre_forward(scheduler_output)
    if rank == 1:
        padded = runner._pad_for_sequence_parallelism(query)
        _, frontier, embeds, received, source = runner._prepare_layered_p_activation(
            plan,
            req_ids=req_ids,
            num_tokens_padded=padded,
            inputs_embeds=None,
            intermediate_tensors=intermediate_tensors,
        )
        runner._record_layered_activation(plan, source)
        hidden, residual = adapter._prepare_initial_state(
            input_ids=None,
            frontier=frontier,
            inputs_embeds=embeds,
            intermediate_tensors=received,
        )
        if hidden.shape[0] != padded or (residual is not None and residual.shape[0] != padded):
            raise RuntimeError("Non-owner P activation does not match padded query rows")
        # Preserve the existing compiled-P owned-copy policy. In particular,
        # do not retain a view into reusable recv/transport scratch.
        store_req_frontiers(
            runner.layered_prefill_state,
            req_ids,
            np.array([0, query], dtype=np.int32),
            hidden,
            residual,
            group_id + 1,
            batch_req_ids=req_ids,
            keep_ids=req_ids,
            reuse_existing=True,
        )
    else:
        # The owner restores its local frontier. No P tensor from this rank is
        # consumed, so preserve the existing zero-row wire contract directly.
        runner.layered_prefill_state.clear(req_id)
        hidden, residual = adapter.make_transport_frontier(0, runner.model_config.dtype, runner.device)
        runner._record_layered_activation(plan, "transport_frontier")
    return (
        ExecuteModelState(
            input_batch=batch,
            attn_metadata=None,
            slot_mappings_by_layer=None,
            hidden_states=None,
            aux_hidden_states=None,
            finished_req_ids=scheduler_output.finished_req_ids,
        ),
        LayeredForwardOutput(hidden, residual, bool(plan.is_final_group)),
    )
