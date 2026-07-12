from types import SimpleNamespace

from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import (
    RequestTracker,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_scheduler import (
    CPUBlockManager,
    SFAKVOffloadlScheduler,
    _num_required_cpu_blocks,
)


def test_mtp_decode_allocates_cpu_lookahead_across_block_boundary() -> None:
    assert _num_required_cpu_blocks(
        num_tokens_after_step=128,
        block_size=128,
        decode_width=4,
        is_decode_step=True,
    ) == 2


def test_mtp_decode_does_not_overallocate_before_block_boundary() -> None:
    assert _num_required_cpu_blocks(
        num_tokens_after_step=124,
        block_size=128,
        decode_width=4,
        is_decode_step=True,
    ) == 1


def test_prefill_and_non_spec_decode_keep_existing_block_count() -> None:
    assert _num_required_cpu_blocks(
        num_tokens_after_step=128,
        block_size=128,
        decode_width=4,
        is_decode_step=False,
    ) == 1
    assert _num_required_cpu_blocks(
        num_tokens_after_step=128,
        block_size=128,
        decode_width=1,
        is_decode_step=True,
    ) == 1


def test_cached_mtp_decode_mirrors_hbm_lookahead_block_to_cpu() -> None:
    scheduler = object.__new__(SFAKVOffloadlScheduler)
    scheduler.real_kv_cache_group_id = 0
    scheduler._block_size = 128
    scheduler.decode_width = 4
    scheduler._preempted_req_ids = set()
    scheduler._unfinished_request_ids = {"req"}
    scheduler._unfinished_requests = {
        "req": (SimpleNamespace(num_prompt_tokens=100), []),
    }
    scheduler.cpu_block_manager = CPUBlockManager(8)
    existing_cpu_blocks = scheduler.cpu_block_manager.allocate_block(1)
    scheduler._request_trackers = {
        "req": RequestTracker(
            req_id="req",
            allocated_block_ids_npu=[10],
            allocated_block_ids_cpu=existing_cpu_blocks,
        ),
    }

    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req"],
            new_block_ids=[[11]],
            num_computed_tokens=[124],
        ),
        num_scheduled_tokens={"req": 4},
    )

    scheduler.build_connector_meta(scheduler_output)

    tracker = scheduler._request_trackers["req"]
    assert tracker.allocated_block_ids_npu == [10, 11]
    assert tracker.allocated_block_ids_cpu == [1, 2]
