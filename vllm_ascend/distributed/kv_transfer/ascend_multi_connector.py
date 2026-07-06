import copy
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    SupportsHMA,
    supports_hma,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import MooncakeLayerwiseConnector

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class AscendMultiConnector(MultiConnector, SupportsHMA):
    _ASCEND_STORE_CONNECTORS = {
        "AscendStoreConnector",
        "MooncakeConnectorStoreV1",
    }

    @classmethod
    def _get_connector_classes_and_configs(cls, vllm_config: "VllmConfig"):
        """Disable AscendStore decode cache traffic in combined SFA offload.

        In the combined prefill+decode offload path, AscendStore owns
        prefill/prefix layerwise store/load while SFAKVOffloadConnector owns
        decode CPU offload and resident-cache loading. AscendStore's layerwise
        decode cache save path also creates decode load specs, which conflicts
        with the explicit SFA tail-buffer path, so turn it off only for this
        connector combination.
        """
        assert vllm_config.kv_transfer_config is not None
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        child_configs = extra_config.get("connectors", [])
        has_sfa_decode_offload = any(
            child.get("kv_connector") == "SFAKVOffloadConnector"
            for child in child_configs
        )
        has_ascend_store = any(
            child.get("kv_connector") in cls._ASCEND_STORE_CONNECTORS
            for child in child_configs
        )
        if not has_sfa_decode_offload or not has_ascend_store:
            return super()._get_connector_classes_and_configs(vllm_config)

        patched_config = copy.copy(vllm_config)
        patched_kv_transfer_config = copy.copy(
            vllm_config.kv_transfer_config
        )
        patched_extra_config = copy.deepcopy(extra_config)
        for child in patched_extra_config.get("connectors", []):
            if child.get("kv_connector") not in cls._ASCEND_STORE_CONNECTORS:
                continue
            child_extra_config = child.setdefault(
                "kv_connector_extra_config", {}
            )
            if child_extra_config.get("use_layerwise", False):
                child_extra_config["save_decode_cache"] = False

        patched_kv_transfer_config.kv_connector_extra_config = (
            patched_extra_config
        )
        patched_config.kv_transfer_config = patched_kv_transfer_config
        return super()._get_connector_classes_and_configs(patched_config)

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        self._all_support_hma = all(supports_hma(c) for c in self._connectors)
        assert vllm_config.scheduler_config.disable_hybrid_kv_cache_manager or self._all_support_hma, (
            "HMA should not be enabled unless all sub-connectors support it"
        )

    @staticmethod
    def _is_sfa_decode_offload_connector(connector: Any) -> bool:
        return connector.__class__.__name__ == "SFAKVOffloadConnector"

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        empty_blocks = blocks.new_empty()
        for i, c in enumerate(self._connectors):
            if (
                i == chosen_connector
                or isinstance(c, MooncakeLayerwiseConnector)
                or self._is_sfa_decode_offload_connector(c)
            ):
                # Forward call to the chosen connector (if any).
                c.update_state_after_alloc(request, blocks, num_external_tokens)
            else:
                # Call with empty blocks for other connectors.
                c.update_state_after_alloc(request, empty_blocks, 0)

    def prepare_lru_resident_and_load(
        self,
        layer_name: str,
        num_tokens: int,
        num_reqs: int,
        topk_indices: torch.Tensor,
        current_slots: torch.Tensor,
        req_ids: torch.Tensor,
        token_to_req: torch.Tensor | None = None,
        capturing: bool = False,
    ) -> bool:
        handled = False
        for c in self._connectors:
            hook = getattr(c, "prepare_lru_resident_and_load", None)
            if hook is None:
                continue
            handled = bool(
                hook(
                    layer_name,
                    num_tokens,
                    num_reqs,
                    topk_indices,
                    current_slots,
                    req_ids,
                    token_to_req,
                    capturing,
                )
            ) or handled
        return handled

    def set_req_ids(self, req_ids: list[str]) -> None:
        for c in self._connectors:
            hook = getattr(c, "set_req_ids", None)
            if hook is not None:
                hook(req_ids)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # Recompute offload may contain an unhashed partial block that other
        # prefix-cache connectors cannot restore. Give its request state
        # priority regardless of connector ordering.
        for i, connector in enumerate(self._connectors):
            has_preempted_request = getattr(connector, "has_preempted_request", None)
            if has_preempted_request is None or not has_preempted_request(request.request_id):
                continue
            tokens, load_async = connector.get_num_new_matched_tokens(request, num_computed_tokens)
            if tokens is None:
                return None, False
            if tokens > 0:
                self._requests_to_connector[request.request_id] = i
                return tokens, load_async
            break

        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_before_preempt(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
        num_computed_tokens: int,
    ) -> bool:
        offloaded = False
        for c in self._connectors:
            hook = getattr(c, "update_state_before_preempt", None)
            if hook is not None:
                offloaded = bool(hook(request, block_ids, num_computed_tokens)) or offloaded
        return offloaded

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if not self._all_support_hma:
            assert len(block_ids) == 1, "HMA with multiple kv_cache_groups requires all sub-connectors to support HMA"
            return super().request_finished(request, block_ids[0])

        async_saves = 0
        kv_txfer_params = None
        for c in self._connectors:
            async_save, txfer_params = cast(SupportsHMA, c).request_finished_all_groups(request, block_ids)
            if async_save:
                async_saves += 1
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    raise RuntimeError("Only one connector can produce KV transfer params")
                kv_txfer_params = txfer_params
        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1

        self._requests_to_connector.pop(request.request_id, None)

        return async_saves > 0, kv_txfer_params
