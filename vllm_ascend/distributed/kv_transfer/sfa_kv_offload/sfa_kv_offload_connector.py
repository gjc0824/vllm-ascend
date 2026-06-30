from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.forward_context import ForwardContext
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_scheduler import SFAKVOffloadlScheduler
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_worker import SFAKVOffloadWorker

class SFAKVOffloadConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        self.use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", False)

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = SFAKVOffloadlScheduler(vllm_config, self.use_layerwise, kv_cache_config)
        else:
            self.connector_worker = SFAKVOffloadWorker(
                vllm_config,
                self.use_layerwise,
                kv_cache_config,
            )

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        # TODO support prefix cache
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        # sfa offload, 0 for indexer and 1 for ori kv_cache
        return self.request_finished(request, block_ids[-1])

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        self.connector_worker.start_load_kv(metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # In sfa kv offload, we use prepare_lru_resident_and_load instead of wait_for_layer_load
        return

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        self.connector_worker.save_kv_layer(layer_name)

    def wait_for_save(self):
        self.connector_worker.wait_for_save()

    def prepare_lru_resident_and_load(
        self,
        layer_name: str,
        num_tokens: int,
        num_reqs: int,
        topk_indices: torch.Tensor,
        current_slots: torch.Tensor,
        req_ids: torch.Tensor,
        token_to_req: torch.Tensor | None = None,
        tokens_per_req: torch.Tensor | None = None,
        capturing: bool = False,
    ) -> bool:
        return self.connector_worker.prepare_lru_resident_and_load(
            layer_name,
            num_tokens,
            num_reqs,
            topk_indices,
            current_slots,
            req_ids,
            token_to_req,
            tokens_per_req,
            capturing,
        )

    def set_req_ids(self, req_ids: list):
        return self.connector_worker.set_req_ids(req_ids)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        # In sfa kv offload, we don't need delay free, thus no need to return finished_send/recv too.
        return (set(), set())
