from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


DEFAULT_CPU_DRAM_SIZE_BYTES = 64 * 1024 * 1024 * 1024
DEFAULT_CPU_CACHE_BUDGET_RATIO = 0.9
DEFAULT_K_CACHE_DIM = 512
DEFAULT_V_CACHE_DIM = 64


def get_sfa_kv_offload_cpu_dram_size(vllm_config) -> int:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    cpu_cache_config: dict[str, Any] = additional_config.get(
        "sfa_kv_offload_cpu_cache_config", {}
    )

    if "dram_size_bytes" in cpu_cache_config:
        return int(cpu_cache_config["dram_size_bytes"])
    if "sfa_kv_offload_cpu_dram_size_bytes" in additional_config:
        return int(additional_config["sfa_kv_offload_cpu_dram_size_bytes"])

    dram_size_gb = cpu_cache_config.get(
        "dram_size_gb",
        additional_config.get("sfa_kv_offload_cpu_dram_size_gb", None),
    )
    if dram_size_gb is None:
        return DEFAULT_CPU_DRAM_SIZE_BYTES
    return int(float(dram_size_gb) * 1024 * 1024 * 1024)


def get_sfa_kv_offload_cpu_cache_budget_ratio(vllm_config) -> float:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    cpu_cache_config: dict[str, Any] = additional_config.get(
        "sfa_kv_offload_cpu_cache_config", {}
    )
    ratio = float(
        cpu_cache_config.get(
            "cache_budget_ratio",
            additional_config.get(
                "sfa_kv_offload_cpu_cache_budget_ratio",
                DEFAULT_CPU_CACHE_BUDGET_RATIO,
            ),
        )
    )
    if ratio <= 0 or ratio > 1:
        raise ValueError(
            "SFA KV offload CPU cache budget ratio must be in (0, 1], "
            f"got {ratio}"
        )
    return ratio


def get_sfa_kv_offload_cache_dims(hf_config) -> tuple[int, int]:
    return (
        int(getattr(hf_config, "kv_lora_rank", DEFAULT_K_CACHE_DIM)),
        int(getattr(hf_config, "qk_rope_head_dim", DEFAULT_V_CACHE_DIM)),
    )


def get_sfa_kv_offload_cpu_block_num(
    vllm_config,
    hf_config,
    block_size: int,
    num_layers: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    dram_size_bytes: int | None = None,
) -> tuple[int, int]:
    dram_size_bytes = (
        get_sfa_kv_offload_cpu_dram_size(vllm_config)
        if dram_size_bytes is None
        else dram_size_bytes
    )
    budget_ratio = get_sfa_kv_offload_cpu_cache_budget_ratio(vllm_config)
    k_dim, v_dim = get_sfa_kv_offload_cache_dims(hf_config)
    bytes_per_block = block_size * (k_dim + v_dim) * dtype.itemsize * num_layers
    if bytes_per_block <= 0:
        raise ValueError(
            "Invalid SFA KV offload CPU block size calculation: "
            f"block_size={block_size}, k_dim={k_dim}, v_dim={v_dim}, "
            f"dtype={dtype}, num_layers={num_layers}"
        )
    usable_bytes = int(dram_size_bytes * budget_ratio)
    cpu_block_num = usable_bytes // bytes_per_block
    if cpu_block_num < 2:
        raise ValueError(
            "SFA KV offload CPU cache is too small to allocate usable blocks: "
            f"dram_size_bytes={dram_size_bytes}, budget_ratio={budget_ratio}, "
            f"bytes_per_block={bytes_per_block}"
        )
    return int(cpu_block_num), int(cpu_block_num * bytes_per_block)


@dataclass
class RequestTracker:
    req_id: str
    allocated_block_ids_npu: list[int]
    allocated_block_ids_cpu: list[int]

    def update(
        self,
        new_block_ids_npu: list[int],
        new_block_ids_cpu: list[int],
    ) -> None:
        """Update the request tracker when a running request is scheduled again."""
        self.allocated_block_ids_npu.extend(new_block_ids_npu)
        self.allocated_block_ids_cpu.extend(new_block_ids_cpu)


@dataclass
class ReqMeta:
    req_id: str
    block_ids_npu: list[int]
    block_ids_cpu: list[int]
    num_new_offload_blocks: int = 0

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        num_new_offload_blocks: int = 0,
    ) -> ReqMeta | None:
        """Create the request metadata from a request tracker."""
        return ReqMeta(
            req_id=tracker.req_id,
            block_ids_npu=tracker.allocated_block_ids_npu,
            block_ids_cpu=tracker.allocated_block_ids_cpu,
            num_new_offload_blocks=num_new_offload_blocks,
        )


class SFAKVOffloadConnectorMetadata(KVConnectorMetadata):
    def __init__(
            self,
            unfinished_request_ids: set[str],
            preempted_req_ids: set[str] | None,
        ):
        self.requests: list[ReqMeta] = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    def add_request(self, req_meta: ReqMeta) -> None:
        self.requests.append(req_meta)


@dataclass
class LayerMultiBlockReqMeta:
    req_id: str
    layer_id: int
    block_ids_npu: list[int] | None = None
    block_ids_cpu: list[int] | None = None
    cache_npu: tuple[torch.Tensor, torch.Tensor] | None = None
    cache_cpu: tuple[torch.Tensor, torch.Tensor] | None = None
