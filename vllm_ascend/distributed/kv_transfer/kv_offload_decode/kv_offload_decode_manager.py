from memfabric_hybrid import offload
import numpy as np
import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.ascend_config import KVOffloadDecodeConfig


class KVOffloadDecodeManager:
    """
    A manager responsible to the offload KV cache.
    It enlarge the availble memory that scheduler can see,
    so we can schedule longer max_model_len or larger decode batch size.
    No more scheduling logic: we reuse the original block_table/slot_mapping.
    """
    _CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024

    @staticmethod
    def _align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
        data_ptr = tensor.data_ptr()
        aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
        offset = (aligned_addr - data_ptr) // tensor.element_size()
        return tensor[int(offset):]

    @classmethod
    def _empty_aligned_cpu_tensor(
        cls,
        shape: list[int],
        dtype: torch.dtype,
        alignment: int = _CPU_CACHE_ALIGNMENT,
    ) -> torch.Tensor:
        num_elements = int(np.prod(shape))
        extra_elements = cdiv(alignment, torch.empty((), dtype=dtype).element_size())
        tensor = offload.empty([num_elements + extra_elements], dtype=dtype, pin_memory=True)
        return cls._align_memory(tensor, alignment)[:num_elements].view(shape)

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        kv_offload_decode_config: KVOffloadDecodeConfig,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.kv_offload_decode_config = kv_offload_decode_config

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self.num_layers = model_config.get_num_layers(parallel_config)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.block_size = self._infer_group_block_sizes(self.kv_cache_config)

        logger.info(
            f"KVOffloadManager start init CPU KV pool with {kv_offload_decode_config.dram_size_per_dp_GB} "
            "GB dram per dp group, it might be time consuming, please wait."
        )
        config = offload.OffloadConfig()
        config.device_id = torch_npu.npu.current_device()
        config.size = kv_offload_decode_config.dram_size_per_dp_GB * 1024 * 1024 * 1024
        config.world_size = self.tp_size
        config.rank_id = self.tp_rank
        offload.initialize(config)
        self.tp_group.barrier()

    def _infer_group_block_sizes(
        self,
        kv_cache_config: KVCacheConfig | None,
    ) -> int:
        assert len(kv_cache_config.kv_cache_groups) == 1, "Hybrid KV is not supported."
        kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        return kv_cache_spec.block_size

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ):
        pass

    def offload_new_kv(
        self,
        slot_mapping: torch.Tensor,
        k_cache_cpu: torch.Tensor,
        v_cache_cpu: torch.Tensor,
        k_cache_npu: torch.Tensor, # for prefill, cache_npu[slot] -> cache_cpu[slot]
        v_cache_npu: torch.Tensor, # for prefill, cache_npu[slot] -> cache_cpu[slot]
        k: torch.Tensor, # for decode, k/v -> cache_cpu[slot]
        v: torch.Tensor, # for decode, k/v -> cache_cpu[slot]
        has_prefill: bool = False,
    ):
        # TODO remove prefill related part after PD disaggregate is ready.
        if has_prefill:
            # simple cache_cpu[slot] = cache_npu[slot].to('cpu')
            return
        # normal case for decode, offload.sparse_copy
        pass

    def onload_topk_kv(
        self,
    ):
        # original onload kv logic
        pass


_KV_OFFLOAD_DECODE_MANAGER: KVOffloadDecodeManager = None


def init_kv_offload_decode_manager(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    kv_offload_decode_config: KVOffloadDecodeConfig,
):
    global _KV_OFFLOAD_DECODE_MANAGER
    if _KV_OFFLOAD_DECODE_MANAGER is None:
        _KV_OFFLOAD_DECODE_MANAGER = KVOffloadDecodeManager(
            vllm_config,
            kv_cache_config,
            kv_offload_decode_config,
        )
    return _KV_OFFLOAD_DECODE_MANAGER


def get_kv_offload_decode_manager():
    assert _KV_OFFLOAD_DECODE_MANAGER is not None, "KV offload manager is not initialized."
    return _KV_OFFLOAD_DECODE_MANAGER
