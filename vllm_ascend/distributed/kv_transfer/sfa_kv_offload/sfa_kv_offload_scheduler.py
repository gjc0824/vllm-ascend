from abc import ABC
from collections import deque
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request

from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import (
    get_sfa_kv_offload_cpu_block_num,
    SFAKVOffloadConnectorMetadata,
    ReqMeta,
    RequestTracker,
)


def _num_finalized_scheduled_tokens(scheduler_output: SchedulerOutput, req_id: str) -> int:
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
    draft_tokens = scheduler_output.scheduled_spec_decode_tokens.get(req_id, [])
    return max(num_scheduled_tokens - len(draft_tokens), 0)


def _num_covered_blocks(num_tokens: int, block_size: int) -> int:
    return (num_tokens + block_size - 1) // block_size


def _is_sfa_indexer_group(kv_cache_group) -> bool:
    return bool(kv_cache_group.layer_names) and all(
        ".indexer.k_cache" in layer_name
        for layer_name in kv_cache_group.layer_names
    )


def get_sfa_real_kv_group_id(kv_cache_config: KVCacheConfig | None) -> int:
    if kv_cache_config is None:
        return -1
    real_group_ids = [
        group_id
        for group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups)
        if not _is_sfa_indexer_group(kv_cache_group)
    ]
    if len(real_group_ids) != 1:
        raise ValueError(
            "SFA KV offload expected exactly one real KV cache group, "
            f"got {real_group_ids}"
        )
    return real_group_ids[0]


class CPUBlockManager(ABC):
    def __init__(self, block_num: int) -> None:
        self.block_num = block_num
        self.block_pool = deque(range(1, block_num))

    def allocate_block(self, new_block_num: int) -> list[int]:
        # logger.info(f'>>>>> pool scheduler allocate cpu block, require: {new_block_num}, resource: {len(self.block_pool)}')
        if len(self.block_pool) < new_block_num:
            raise ValueError(
                "No enough SFA KV offload CPU blocks to allocate, "
                f"requested={new_block_num}, available={len(self.block_pool)}, "
                f"capacity={self.block_num}"
            )
        allocated_blocks = []
        for _ in range(new_block_num):
            allocated_blocks.append(self.block_pool.popleft())
        return allocated_blocks

    def free(self, to_free_blocks: list[int]):
        self.block_pool.extend(to_free_blocks)


class SFAKVOffloadlScheduler:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        use_layerwise,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        self.use_layerwise = use_layerwise
        self.kv_cache_config = kv_cache_config
        hf_text_config = getattr(vllm_config.model_config, "hf_text_config", None)
        hf_config = getattr(vllm_config.model_config, "hf_config", hf_text_config)
        self.hf_config = hf_text_config or hf_config
        init_ascend_config(vllm_config)
        ascend_config = get_ascend_config()
        self.use_offload = ascend_config.use_offload
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.pcp_size = getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1)
        self.dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)
        self.decode_width = 1
        if vllm_config.speculative_config is not None:
            self.decode_width += vllm_config.speculative_config.num_speculative_tokens
        self.group_block_sizes = self._infer_group_block_sizes(vllm_config, kv_cache_config)
        self.real_kv_cache_group_id = get_sfa_real_kv_group_id(kv_cache_config)
        self._block_size = self.group_block_sizes[self.real_kv_cache_group_id]

        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}
        self._preempted_req_ids: set[str] = set()
        self._unfinished_requests: dict[str, tuple[Request, list[list[int]]]] = {}
        self._unfinished_request_ids: set[str] = set()

        num_layers = self._infer_num_offload_layers(vllm_config)
        cpu_block_num, cpu_cache_size = get_sfa_kv_offload_cpu_block_num(
            vllm_config,
            self.hf_config,
            self._block_size,
            num_layers,
        )
        logger.info(
            "SFA KV offload scheduler CPU block capacity: %s blocks, "
            "%.2f GiB across %s layers.",
            cpu_block_num,
            cpu_cache_size / 1024 / 1024 / 1024,
            num_layers,
        )
        self.cpu_block_manager = CPUBlockManager(cpu_block_num)

    def _infer_num_offload_layers(self, vllm_config: "VllmConfig") -> int:
        if self.kv_cache_config is not None and self.real_kv_cache_group_id >= 0:
            real_group = self.kv_cache_config.kv_cache_groups[
                self.real_kv_cache_group_id
            ]
            if real_group.layer_names:
                return len(real_group.layer_names)
        return vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )

    def _infer_group_block_sizes(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: KVCacheConfig | None,
    ) -> list[int]:
        block_sizes: list[int] = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            block_sizes.append(kv_cache_spec.block_size)
        return block_sizes

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        """
        """
        local_block_ids: list[list[int]] = []

        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        self._unfinished_request_ids.add(request.request_id)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.
        """
        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self._unfinished_request_ids.discard(finished_req_id)
            self._preempted_req_ids.discard(finished_req_id)

        for req_id in scheduler_output.preempted_req_ids:
            self._preempted_req_ids.update(scheduler_output.preempted_req_ids)
            self._request_trackers.pop(req_id, None)
            self._unfinished_requests.pop(req_id, None)

        meta = SFAKVOffloadConnectorMetadata(self._unfinished_request_ids, scheduler_output.preempted_req_ids)

        for request in scheduler_output.scheduled_new_reqs:
            block_ids_npu = request.block_ids[self.real_kv_cache_group_id].copy()
            num_tokens_to_compute = request.num_computed_tokens + _num_finalized_scheduled_tokens(
                scheduler_output, request.req_id)
            num_cpu_blocks = min(_num_covered_blocks(num_tokens_to_compute, self._block_size), len(block_ids_npu))
            block_ids_cpu = self.cpu_block_manager.allocate_block(num_cpu_blocks)
            request_tracker = RequestTracker(
                req_id=request.req_id,
                allocated_block_ids_npu=block_ids_npu,
                allocated_block_ids_cpu=block_ids_cpu,
                offload_src_hbm_ids=block_ids_npu[:num_cpu_blocks],
                offload_dst_cpu_ids=block_ids_cpu,
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                num_new_offload_blocks=len(request_tracker.offload_src_hbm_ids),
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            # resumed request
            new_block_ids_npu = cached_reqs.new_block_ids[i]
            if isinstance(new_block_ids_npu, tuple):
                new_block_ids_npu = new_block_ids_npu[self.real_kv_cache_group_id]
            elif new_block_ids_npu is None:
                new_block_ids_npu = []
            if req_id in self._preempted_req_ids:
                raise ValueError('preempted reqs not implemented')
            # decode/chunked request
            else:
                request_tracker = self._request_trackers[req_id]
                num_new_tokens = _num_finalized_scheduled_tokens(scheduler_output, req_id)
                req_tuple = self._unfinished_requests.get(req_id)
                if req_tuple:
                    request = req_tuple[0]
                else:
                    raise ValueError(
                        f"Request {req_id} is not in _unfinished_requests, but it is scheduled to be cached"
                    )
                num_computed_token = cached_reqs.num_computed_tokens[i]
                num_tokens_after_step = num_computed_token + num_new_tokens
                num_blocks_after_step = _num_covered_blocks(num_tokens_after_step, self._block_size)
                num_offloaded_blocks = len(request_tracker.allocated_block_ids_cpu)
                target_num_blocks = min(
                    num_blocks_after_step,
                    len(request_tracker.allocated_block_ids_npu) + len(new_block_ids_npu),
                )
                num_new_cpu_blocks = max(target_num_blocks - num_offloaded_blocks, 0)
                new_block_ids_cpu = self.cpu_block_manager.allocate_block(num_new_cpu_blocks)
                request_tracker.update(new_block_ids_npu, new_block_ids_cpu)
                is_decode_step = (
                    0 < num_new_tokens <= self.decode_width
                    and num_computed_token >= request.num_prompt_tokens
                )
                if is_decode_step:
                    # In the buffer hybrid layout, indexer cache aliases the
                    # real-K storage. A decode-step whole-block HBM refresh can
                    # copy alias-polluted historical tokens into CPU. Decode
                    # tokens are patched into CPU cache token-wise after real-KV
                    # re-materialization in attention.
                    offload_start = target_num_blocks
                elif num_new_tokens > 0 and target_num_blocks > 0:
                    # Refresh every real-KV block touched by this prefill/chunk
                    # step. A small final prefill chunk can be decode-width-sized
                    # but still needs whole-block save semantics.
                    offload_start = min(num_computed_token // self._block_size, target_num_blocks - 1)
                else:
                    offload_start = max(target_num_blocks - num_new_cpu_blocks, 0)
                if target_num_blocks > len(request_tracker.allocated_block_ids_npu):
                    raise ValueError(
                        "SFA KV offload target block count exceeds NPU block table: "
                        f"req_id={req_id}, target={target_num_blocks}, "
                        f"npu={request_tracker.allocated_block_ids_npu}"
                    )
                request_tracker.offload_src_hbm_ids = request_tracker.allocated_block_ids_npu[
                    offload_start:target_num_blocks
                ]
                request_tracker.offload_dst_cpu_ids = request_tracker.allocated_block_ids_cpu[
                    offload_start:target_num_blocks
                ]

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    num_new_offload_blocks=len(request_tracker.offload_src_hbm_ids),
                )
            if req_meta is not None:
                meta.add_request(req_meta)
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        tracker = self._request_trackers.get(request.request_id)
        if tracker is not None:
            self.cpu_block_manager.free(tracker.allocated_block_ids_cpu)
        return False, None
