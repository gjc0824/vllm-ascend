from __future__ import annotations

import os
import threading
from typing import Optional
from collections.abc import Generator

import numpy as np
import torch
from torch.utils.cpp_extension import load
import torch_npu
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.utils import CpuGpuBuffer
from memfabric_hybrid import h2d

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import (
    get_sfa_kv_offload_cache_dims,
    get_sfa_kv_offload_cpu_block_num,
    get_sfa_kv_offload_cpu_dram_size,
    SFAKVOffloadConnectorMetadata,
    LayerMultiBlockReqMeta,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.kv_transfer import (
    KVCacheStoreLayerSendingThread,
    KVTransferThread,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_scheduler import (
    get_sfa_real_kv_group_id,
)

_SUBSCRIBED_COMPUTE_STREAMS = set()
_SFA_DEBUG = bool(int(os.getenv("VLLM_ASCEND_SFA_DEBUG", "0")))
_SFA_PROBE = bool(int(os.getenv("VLLM_ASCEND_SFA_PROBE", "1")))
_SFA_FORCE_LRU_MISS = bool(
    int(os.getenv("VLLM_ASCEND_SFA_FORCE_LRU_MISS", "0"))
)
_SFA_DEBUG_BUILD = "all_cpu_inline_token_patch_20260709"


def _probe_enabled(count: int) -> bool:
    return count <= 12 or count in (16, 32, 64, 96, 128)


def _debug_tensor_head(tensor: torch.Tensor, limit: int = 8):
    if tensor.numel() == 0:
        return []
    return tensor.reshape(-1)[:limit].detach().cpu().tolist()


def _debug_list_head(values: list[int], limit: int = 8) -> list[int]:
    return values[:limit]


def _debug_synchronize_npu() -> None:
    try:
        torch_npu.npu.synchronize()
    except Exception:
        torch_npu.npu.current_stream().synchronize()


def get_subscribed_compute_streams() -> set:
    return _SUBSCRIBED_COMPUTE_STREAMS

def _is_current_stream_capturing() -> bool:
    for npu_runtime in (getattr(torch_npu, "npu", None), getattr(torch, "npu", None)):
        if npu_runtime is None:
            continue
        for attr_name in ("is_current_stream_capturing", "_is_current_stream_capturing"):
            capture_state = getattr(npu_runtime, attr_name, None)
            if not callable(capture_state):
                continue
            try:
                if bool(capture_state()):
                    return True
            except Exception:
                continue
    return False

# cpu sparse attn kernel related
# TODO maybe implement this in vllm custom op framework
os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
# cache_dir = "/root/.cache/torch_extensions/py311_cpu/cpu_sparse_attn"
# if os.path.exists(cache_dir):
#     shutil.rmtree(cache_dir)
#     print(f"已清理缓存目录: {cache_dir}")
ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
npu_include_path = os.path.join(ascend_home, "include")
npu_lib_path = os.path.join(ascend_home, "lib64")
if not os.path.exists(npu_lib_path):
    npu_lib_path = os.path.join(ascend_home, "lib")
torch_npu_path = os.path.dirname(torch_npu.__file__)
torch_npu_include = os.path.join(torch_npu_path, "include")
torch_npu_lib_path = os.path.join(torch_npu_path, "lib")
os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
os.environ['CXX'] = 'clang++'
os.environ['CC'] = 'clang'
abs_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(abs_path, "cpu_sparse_attn.cpp")
logger.info(
    "SFA_DEBUG_BUILD worker version=%s file=%s debug=%s",
    _SFA_DEBUG_BUILD,
    os.path.abspath(__file__),
    _SFA_DEBUG,
)
logger.info(f'>>>>> load cpu_sparse_attn from src: {src_path}')
cpu_sparse_attn = None
cpu_sparse_attn = load(
    name="cpu_sparse_attn",
    sources=[src_path],
    extra_cflags=[
        "-O3",
        "-std=c++20",
        "-fopenmp",
        "-march=armv8.2-a+sve+fp16+bf16",
        # "-march=native",
        "-fPIC",
        f"-I{npu_include_path}",
        f"-I{torch_npu_include}",
    ],
    extra_ldflags=[
        "-fopenmp",
        f"-L{npu_lib_path}",
        "-lascendcl",
        f"-L{torch_npu_lib_path}",
        "-ltorch_npu",
    ],
    verbose=True,  # 添加 verbose 查看编译过程
)


class SFAKVOffloadWorker:
    # The main class for the cache engine.

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwize: bool,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        hf_text_config = getattr(model_config, "hf_text_config", None)
        hf_config = getattr(model_config, "hf_config", hf_text_config)
        self.hf_config = hf_text_config or hf_config
        self.dp_rank = parallel_config.data_parallel_rank
        self.use_mla = False
        if hasattr(model_config, "use_mla") and isinstance(model_config.use_mla, bool) and model_config.use_mla:
            self.use_mla = True
        self.use_sparse = hasattr(model_config.hf_text_config, "index_topk")
        self.use_layerwise = use_layerwize
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        ascend_config = get_ascend_config()
        self.use_offload = ascend_config.use_offload

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.group_block_sizes = self._infer_group_block_sizes(vllm_config, kv_cache_config)
        self.real_kv_cache_group_id = get_sfa_real_kv_group_id(kv_cache_config)
        self.block_size = self.group_block_sizes[self.real_kv_cache_group_id]

        self.current_layer_save = 0
        self.current_layer_load = 0
        self.num_target_layers = model_config.get_num_layers(parallel_config)
        self.num_offload_layers = self.num_target_layers
        self.num_layers = self.num_offload_layers
        self.offload_layer_names: list[str] = []
        self.layer_name_to_offload_id: dict[str, int] = {}

        if self.use_mla:
            self.num_kv_head = 1
        else:
            self.num_kv_head = model_config.get_total_num_kv_heads()

        self.kv_send_thread: KVTransferThread | None = None
        self.layer_save_tasks: list[list[LayerMultiBlockReqMeta]] = []
        self.pending_save_layer_ids: set[int] = set()
        self.submitted_save_layer_ids: set[int] = set()
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        decode_width = 1
        if vllm_config.speculative_config is not None:
            decode_width += vllm_config.speculative_config.num_speculative_tokens
        self.max_num_topk_rows = min(
            self.max_num_tokens,
            self.max_num_reqs * decode_width,
        )
        lru_resident_config = ascend_config.lru_resident_cache_config
        self.sfa_sparse_topk = lru_resident_config.topk
        self.lru_resident_capacity = lru_resident_config.buffer_size

        head_num = 1
        head_dim_k, head_dim_v = get_sfa_kv_offload_cache_dims(self.hf_config)
        dtype = torch.bfloat16
        self.token_size_bytes_k = head_num * head_dim_k * dtype.itemsize
        self.token_size_bytes_v = head_num * head_dim_v * dtype.itemsize
        self.max_model_len = vllm_config.model_config.max_model_len
        max_block_num = cdiv(self.max_model_len, self.block_size)
        self.cpu_block_table = CpuGpuBuffer(self.max_num_reqs, max_block_num, dtype=torch.int32, device='npu', pin_memory=True)
        self.cpu_block_table_host_buffer = torch.zeros([self.max_num_reqs, max_block_num], dtype=torch.int32, device='cpu', pin_memory=True)
        self.lru_expanded_block_table_cpu = torch.empty(
            [self.max_num_topk_rows, max_block_num],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.actual_seq_len_q = torch.arange(self.max_num_reqs, dtype=torch.int32, device='cpu', pin_memory=True) + 1
        self.req_ids = []
        self.cpu_blocks_by_req: dict[str, int] = {}

        self.cpu_sparse_attn = cpu_sparse_attn

        self.load_stream = None
        self.load_stream = torch_npu.npu.Stream()
        self.save_stream = None
        self.side_compute_stream = torch_npu.npu.Stream()
        self.allocate_dram_size = get_sfa_kv_offload_cpu_dram_size(vllm_config)
        h2d.initialize(self.tp_rank, self.allocate_dram_size)

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

    @staticmethod
    def _as_cache_tuple(cache_or_caches) -> tuple[torch.Tensor, ...]:
        if isinstance(cache_or_caches, torch.Tensor):
            return (cache_or_caches,)
        return tuple(cache_or_caches)

    def _register_offload_layers(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.offload_layer_names = [
            layer_name
            for layer_name, cache_or_caches in kv_caches.items()
            if len(self._as_cache_tuple(cache_or_caches)) >= 5
        ]
        if not self.offload_layer_names:
            raise ValueError("SFA KV Offload did not find SFA KV cache layers.")

        self.num_offload_layers = len(self.offload_layer_names)
        self.num_layers = self.num_offload_layers
        self.layer_name_to_offload_id = {
            layer_name: layer_id
            for layer_id, layer_name in enumerate(self.offload_layer_names)
        }
        self.layer_save_tasks = [[] for _ in range(self.num_layers)]
        self.pending_save_layer_ids.clear()
        self.submitted_save_layer_ids.clear()

        logger.info(
            "SFA KV offload registered %s layers (%s target layers).",
            self.num_layers,
            self.num_target_layers,
        )
        if self.tp_rank == 0:
            preview_layer_names = self.offload_layer_names[:4]
            if len(self.offload_layer_names) > 4:
                preview_layer_names += ["..."] + self.offload_layer_names[-4:]
            logger.info("SFA KV offload layer names: %s", preview_layer_names)

    def _get_offload_layer_id(self, layer_name: str) -> int:
        layer_id = self.layer_name_to_offload_id.get(layer_name)
        if layer_id is None:
            registered_layers = ", ".join(self.offload_layer_names[:8])
            if len(self.offload_layer_names) > 8:
                registered_layers += ", ..."
            raise KeyError(
                "SFA KV offload layer is not registered, "
                f"layer_name={layer_name}, registered_layers=[{registered_layers}]"
            )
        return layer_id

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache_tuple = self._as_cache_tuple(first_kv_cache_tuple)
        first_kv_cache = first_kv_cache_tuple[0]

        self.num_blocks = (
            self.kv_cache_config.num_blocks if self.kv_cache_config is not None else first_kv_cache.shape[0]
        )
        logger.info("num_blocks: %s", self.num_blocks)

        logger.info(
            "Registering KV_Caches. use_mla: %s, use_sparse: %s, shape %s",
            self.use_mla,
            self.use_sparse,
            first_kv_cache.shape,
        )

        if self.use_sparse and self.use_offload:
            self._register_offload_layers(kv_caches)
            self.k_caches_npu: list[torch.Tensor] = []
            self.v_caches_npu: list[torch.Tensor] = []
            self.topk_buffers_k: list[torch.Tensor] = []
            self.topk_buffers_v: list[torch.Tensor] = []
            for layer_name in self.offload_layer_names:
                cache_or_caches = self._as_cache_tuple(kv_caches[layer_name])
                # SFA offload tuple: CPU saves normal KV [0:2], including
                # refreshed tail blocks, and decode LRU loads fill top-k
                # buffers [3:5].
                assert len(cache_or_caches) >= 5
                self.k_caches_npu.append(cache_or_caches[0])
                self.v_caches_npu.append(cache_or_caches[1])
                self.topk_buffers_k.append(cache_or_caches[3])
                self.topk_buffers_v.append(cache_or_caches[4])
                if _SFA_DEBUG and self.tp_rank == 0 and len(self.k_caches_npu) <= 2:
                    logger.info(
                        "SFA_DEBUG worker cache_layout layer=%s k=%s v=%s "
                        "indexer=%s topk_k=%s topk_v=%s",
                        layer_name,
                        tuple(cache_or_caches[0].shape),
                        tuple(cache_or_caches[1].shape),
                        tuple(cache_or_caches[2].shape),
                        tuple(cache_or_caches[3].shape),
                        tuple(cache_or_caches[4].shape),
                    )

            if self.use_layerwise:
                ready_event = threading.Event()
                self.layer_save_finished_events = [threading.Event() for _ in range(self.num_layers)]
                self.kv_send_thread = KVCacheStoreLayerSendingThread(
                    self.block_size,
                    self.num_layers,
                    self.tp_rank,
                    ready_event,
                    self.layer_save_finished_events,
                )
                self.kv_send_thread.start()
                ready_event.wait()
            else:
                raise ValueError("SFA KV Offload only support layerwise now.")

            head_dim_k, head_dim_v = get_sfa_kv_offload_cache_dims(self.hf_config)
            cpu_block_num, cpu_cache_size_single_card = (
                get_sfa_kv_offload_cpu_block_num(
                    self.vllm_config,
                    self.hf_config,
                    self.block_size,
                    self.num_layers,
                    dram_size_bytes=self.allocate_dram_size,
                )
            )
            logger.info(
                "KV offload allocate %s cpu blocks, size = %.2f GB per rank "
                "(npu_blocks=%s, layers=%s)",
                cpu_block_num,
                cpu_cache_size_single_card / 1024 / 1024 / 1024,
                self.num_blocks,
                self.num_layers,
            )
            if cpu_cache_size_single_card > self.allocate_dram_size:
                raise ValueError(
                    f"Needed cpu memory ({cpu_cache_size_single_card / 1024 / 1024 / 1024} GB/rank) is greater than "
                    f"available cpu memory ({self.allocate_dram_size / 1024 / 1024 / 1024} GB/rank), "
                    "try to decrease gpu_memory_utilization or allocate more cpu memory during init."
                )
            self.k_caches_cpu: list[torch.Tensor] = [
                h2d.empty(
                    [cpu_block_num, self.block_size, 1, head_dim_k],
                    dtype=torch.bfloat16,
                    pin_memory=True,
                )
                for _ in range(self.num_layers)
            ]
            self.v_caches_cpu: list[torch.Tensor] = [
                h2d.empty(
                    [cpu_block_num, self.block_size, 1, head_dim_v],
                    dtype=torch.bfloat16,
                    pin_memory=True,
                )
                for _ in range(self.num_layers)
            ]

            # topk cache reuse related
            self.lru_workspace_threads = 8
            self.lru_topk_indices_cpu = torch.empty(
                [self.max_num_topk_rows, self.sfa_sparse_topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_token_to_req_cpu = torch.empty(
                [self.max_num_topk_rows],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_slot_to_token_cpu_list = [torch.full(
                [self.max_num_topk_rows, self.lru_resident_capacity],
                -1,
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            ) for _ in range(self.num_layers)]
            self.lru_slots_cpu_list = [torch.arange(
                self.lru_resident_capacity,
                dtype=torch.int32,
                device='cpu',
            ).view(1, -1).repeat(self.max_num_topk_rows, 1).pin_memory() for _ in range(self.num_layers)]
            self.lru_current_slots_cpu = torch.empty(
                [self.max_num_topk_rows, self.sfa_sparse_topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_miss_count_cpu_list = [torch.empty(
                [self.max_num_topk_rows],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            ) for _ in range(self.num_layers)]
            self.lru_miss_tokens_cpu_list = [torch.empty(
                [self.max_num_topk_rows, self.sfa_sparse_topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            ) for _ in range(self.num_layers)]
            self.lru_miss_slots_cpu_list = [torch.empty(
                [self.max_num_topk_rows, self.sfa_sparse_topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            ) for _ in range(self.num_layers)]
            self.lru_req_ids_cpu = torch.empty([self.max_num_topk_rows], dtype=torch.int64, device='cpu', pin_memory=True)
            self.lru_last_req_ids_cpu_list = [torch.full(
                [self.max_num_topk_rows],
                -1,
                dtype=torch.int64,
                device='cpu',
                pin_memory=True,
            ) for _  in range(self.num_layers)]
            self.lru_token_mark_workspace = torch.zeros(
                [self.lru_workspace_threads, self.max_model_len],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_token_pos_workspace = torch.full(
                [self.lru_workspace_threads, self.max_model_len],
                -1,
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_slot_workspace = torch.empty(
                [self.lru_workspace_threads, self.lru_resident_capacity * 3],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_miss_position_workspace = torch.empty(
                [self.lru_workspace_threads, self.sfa_sparse_topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_epochs = torch.zeros(
                [self.lru_workspace_threads],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )

            self.lru_req_ids_ptr = self.lru_req_ids_cpu.data_ptr()
            self.lru_last_req_ids_ptrs = [lru_last_req_ids_cpu.data_ptr() for lru_last_req_ids_cpu in self.lru_last_req_ids_cpu_list]
            self.lru_topk_indices_ptr = self.lru_topk_indices_cpu.data_ptr()
            self.lru_token_to_req_ptr = self.lru_token_to_req_cpu.data_ptr()
            self.lru_slot_to_token_ptrs = [lru_slot_to_token_cpu.data_ptr() for lru_slot_to_token_cpu in self.lru_slot_to_token_cpu_list]
            self.lru_slots_ptrs = [lru_slots_cpu.data_ptr() for lru_slots_cpu in self.lru_slots_cpu_list]
            self.lru_current_slots_ptr = self.lru_current_slots_cpu.data_ptr()
            self.lru_miss_count_ptrs = [lru_miss_count_cpu.data_ptr() for lru_miss_count_cpu in self.lru_miss_count_cpu_list]
            self.lru_miss_tokens_ptrs = [lru_miss_tokens_cpu.data_ptr() for lru_miss_tokens_cpu in self.lru_miss_tokens_cpu_list]
            self.lru_miss_slots_ptrs = [lru_miss_slots_cpu.data_ptr() for lru_miss_slots_cpu in self.lru_miss_slots_cpu_list]
            self.lru_token_mark_workspace_ptr = self.lru_token_mark_workspace.data_ptr()
            self.lru_token_pos_workspace_ptr = self.lru_token_pos_workspace.data_ptr()
            self.lru_slot_workspace_ptr = self.lru_slot_workspace.data_ptr()
            self.lru_miss_position_workspace_ptr = self.lru_miss_position_workspace.data_ptr()
            self.lru_epochs_ptr = self.lru_epochs.data_ptr()

            # Keep integer staging dtypes aligned with their NPU sources.
            # Capture accepts raw D2H copies here, but D2H+dtype conversion
            # synchronizes the captured stream on Ascend.
            self.token_update_slots_cpu = torch.empty(
                [self.max_num_topk_rows],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.token_update_positions_cpu = torch.empty(
                [self.max_num_topk_rows],
                dtype=torch.int64,
                device='cpu',
                pin_memory=True,
            )
            self.token_update_token_to_req_cpu = torch.empty(
                [self.max_num_topk_rows],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.token_update_k_cpu = torch.empty(
                [self.max_num_topk_rows, head_dim_k],
                dtype=torch.bfloat16,
                device='cpu',
                pin_memory=True,
            )
            self.token_update_v_cpu = torch.empty(
                [self.max_num_topk_rows, head_dim_v],
                dtype=torch.bfloat16,
                device='cpu',
                pin_memory=True,
            )

            # sparse h2d (batch_copy related)
            self.addr_k_bases: list[int] = [t.data_ptr() for t in self.topk_buffers_k]
            self.addr_v_bases: list[int] = [t.data_ptr() for t in self.topk_buffers_v]
            self.gvas_k_bases: list[int] = [t.data_ptr() for t in self.k_caches_cpu]
            self.gvas_v_bases: list[int] = [t.data_ptr() for t in self.v_caches_cpu]

            gvas_buffer_offset = 0
            gvas_buffer_size_bytes = self.max_num_topk_rows * self.sfa_sparse_topk * 2 * 8 # 2: k+v, 8: int64
            addr_buffer_offset = gvas_buffer_offset + gvas_buffer_size_bytes
            addr_buffer_size_bytes = self.max_num_topk_rows * self.sfa_sparse_topk * 2 * 8
            size_buffer_offset = addr_buffer_offset + addr_buffer_size_bytes
            size_buffer_size_bytes = self.max_num_topk_rows * self.sfa_sparse_topk * 2 * 4 # 2: k+v, 4: int32
            num_tokens_buffer_offset = size_buffer_offset + size_buffer_size_bytes
            num_tokens_buffer_size_bytes = 4
            batch_copy_args_buffer_size_bytes = gvas_buffer_size_bytes + addr_buffer_size_bytes + size_buffer_size_bytes + num_tokens_buffer_size_bytes
            self.batch_copy_args_buffer_cpu = torch.zeros([batch_copy_args_buffer_size_bytes], dtype=torch.int8, device='cpu', pin_memory=True)
            self.batch_copy_args_buffer_npu = torch.zeros([batch_copy_args_buffer_size_bytes], dtype=torch.int8, device='npu')

            self.gvas_buffer_cpu = self.batch_copy_args_buffer_cpu[gvas_buffer_offset:gvas_buffer_offset + gvas_buffer_size_bytes].view(torch.int64)
            self.addr_buffer_cpu = self.batch_copy_args_buffer_cpu[addr_buffer_offset:addr_buffer_offset + addr_buffer_size_bytes].view(torch.int64)
            self.size_buffer_cpu = self.batch_copy_args_buffer_cpu[size_buffer_offset:size_buffer_offset + size_buffer_size_bytes].view(torch.int32)
            self.num_tokens_buffer_cpu = \
                self.batch_copy_args_buffer_cpu[num_tokens_buffer_offset:num_tokens_buffer_offset + num_tokens_buffer_size_bytes].view(torch.int32)
            assert self.gvas_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.sfa_sparse_topk * 2])
            assert self.addr_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.sfa_sparse_topk * 2])
            assert self.size_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.sfa_sparse_topk * 2])
            assert self.num_tokens_buffer_cpu.shape == torch.Size([1])

            self.gvas_buffer_npu = self.batch_copy_args_buffer_npu[gvas_buffer_offset:gvas_buffer_offset + gvas_buffer_size_bytes].view(torch.int64)
            self.addr_buffer_npu = self.batch_copy_args_buffer_npu[addr_buffer_offset:addr_buffer_offset + addr_buffer_size_bytes].view(torch.int64)
            self.size_buffer_npu = self.batch_copy_args_buffer_npu[size_buffer_offset:size_buffer_offset + size_buffer_size_bytes].view(torch.int32)
            self.num_tokens_buffer_npu = \
                self.batch_copy_args_buffer_npu[num_tokens_buffer_offset:num_tokens_buffer_offset + num_tokens_buffer_size_bytes].view(torch.int32)
            assert self.gvas_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.sfa_sparse_topk * 2])
            assert self.addr_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.sfa_sparse_topk * 2])
            assert self.size_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.sfa_sparse_topk * 2])
            assert self.num_tokens_buffer_npu.shape == torch.Size([1])

    def _debug_verify_lru_copy(
        self,
        layer_id: int,
        num_tokens: int,
        cpu_block_table: torch.Tensor,
    ) -> None:
        if not (_SFA_DEBUG and self.tp_rank == 0 and layer_id < 2 and num_tokens > 0):
            return
        try:
            miss_count = self.lru_miss_count_cpu_list[layer_id][:num_tokens]
            miss_tokens = self.lru_miss_tokens_cpu_list[layer_id][:num_tokens]
            miss_slots = self.lru_miss_slots_cpu_list[layer_id][:num_tokens]
            samples = []
            max_rows = min(num_tokens, 2)
            for row in range(max_rows):
                count = max(0, int(miss_count[row].item()))
                count = min(count, self.sfa_sparse_topk)
                for miss_idx in range(count):
                    token = int(miss_tokens[row, miss_idx].item())
                    slot = int(miss_slots[row, miss_idx].item())
                    if token < 0 or slot < 0 or slot >= self.lru_resident_capacity:
                        continue
                    block_id = token // self.block_size
                    if block_id < 0 or block_id >= cpu_block_table.shape[1]:
                        continue
                    table_row = min(row, cpu_block_table.shape[0] - 1)
                    cpu_block_id = int(cpu_block_table[table_row, block_id].item())
                    offset = token % self.block_size
                    k_src = float(
                        self.k_caches_cpu[layer_id][cpu_block_id, offset]
                        .float()
                        .abs()
                        .sum()
                        .item()
                    )
                    v_src = float(
                        self.v_caches_cpu[layer_id][cpu_block_id, offset]
                        .float()
                        .abs()
                        .sum()
                        .item()
                    )
                    k_dst = float(
                        self.topk_buffers_k[layer_id][row, slot]
                        .float()
                        .abs()
                        .sum()
                        .detach()
                        .cpu()
                        .item()
                    )
                    v_dst = float(
                        self.topk_buffers_v[layer_id][row, slot]
                        .float()
                        .abs()
                        .sum()
                        .detach()
                        .cpu()
                        .item()
                    )
                    samples.append(
                        (
                            row,
                            token,
                            slot,
                            cpu_block_id,
                            offset,
                            k_src,
                            k_dst,
                            abs(k_src - k_dst),
                            v_src,
                            v_dst,
                            abs(v_src - v_dst),
                        )
                    )
                    if len(samples) >= 4:
                        break
                if len(samples) >= 4:
                    break
            if samples:
                logger.info(
                    "SFA_DEBUG worker lru_verify layer=%s samples=%s",
                    layer_id,
                    samples,
                )
        except Exception:
            logger.exception("SFA_DEBUG worker lru_verify failed layer=%s", layer_id)

    def start_load_kv(self, metadata: SFAKVOffloadConnectorMetadata):
        # return
        self.current_layer_save = 0
        self.current_layer_load = 0
        req_id_to_block_ids: dict[str, list[int]] = {}
        for layer_save_task in self.layer_save_tasks:
            layer_save_task.clear()
        self.pending_save_layer_ids.clear()
        self.submitted_save_layer_ids.clear()
        for event in getattr(self, "layer_save_finished_events", []):
            event.clear()
        for req_id in metadata.preempted_req_ids or set():
            self.cpu_blocks_by_req.pop(req_id, None)
        for request in metadata.requests:
            req_id_to_block_ids[request.req_id] = request.block_ids_cpu
            self.cpu_blocks_by_req[request.req_id] = len(request.block_ids_cpu)
            if _SFA_DEBUG and self.tp_rank == 0:
                logger.info(
                    "SFA_DEBUG worker metadata req=%s cpu_blocks=%s "
                    "new_offload=%s src=%s dst=%s npu_total=%s",
                    request.req_id,
                    len(request.block_ids_cpu),
                    request.num_new_offload_blocks,
                    _debug_list_head(request.offload_src_hbm_ids),
                    _debug_list_head(request.offload_dst_cpu_ids),
                    len(request.block_ids_npu),
                )
            if request.num_new_offload_blocks <= 0:
                continue # no new blocks to save
            self.process_layer_data(request)
        num_save_layers = sum(1 for layer_save_task in self.layer_save_tasks if layer_save_task)
        self.num_save_tasks = sum(len(layer_save_task) for layer_save_task in self.layer_save_tasks)
        if self.tp_rank == 0:
            logger.info(
                f'>>>>> start load kv, reqs num: {len(metadata.requests)}, '
                f'save layer num = {num_save_layers}, save task num = {self.num_save_tasks}'
            )

        # generate block_table for load
        # NOTE reqs in self.req_ids and metadata.requests may not be in same order,
        # use reqs from self.req_ids (order of actual batch) to compute block_table.
        num_reqs = len(self.req_ids)
        if not req_id_to_block_ids:
            if num_reqs and self.tp_rank == 0:
                logger.info(
                    "SFA KV offload start_load_kv has no active request "
                    "metadata; clearing %s stale req ids.",
                    num_reqs,
                )
            self.req_ids = []
            self.cpu_block_table.copy_to_gpu(0)
            return

        cpu_block_table_np = self.cpu_block_table.np[:num_reqs]
        cpu_block_table_np.fill(0)
        missing_req_ids: list[str] = []
        for i, req_id in enumerate(self.req_ids[:num_reqs]):
            cpu_block_ids = req_id_to_block_ids.get(req_id)
            if cpu_block_ids is None:
                missing_req_ids.append(req_id)
                continue
            cpu_block_table_np[i][:len(cpu_block_ids)] = np.array([cpu_block_ids], dtype=np.int32)
        if _SFA_DEBUG and self.tp_rank == 0:
            logger.info(
                "SFA_DEBUG worker cpu_block_table num_reqs=%s req_tail=%s "
                "row0=%s blocks_by_req=%s",
                num_reqs,
                [req_id[-8:] for req_id in self.req_ids[:num_reqs]],
                cpu_block_table_np[0][:16].tolist() if num_reqs else [],
                {
                    req_id[-8:]: len(req_id_to_block_ids.get(req_id, []))
                    for req_id in self.req_ids[:num_reqs]
                },
            )
        if missing_req_ids and self.tp_rank == 0:
            logger.info(
                "SFA KV offload skipped %s stale req ids absent from "
                "metadata while building CPU block table.",
                len(missing_req_ids),
            )
        self.cpu_block_table.copy_to_gpu(num_reqs)
        if _SFA_PROBE and self.tp_rank == 0:
            probe_count = getattr(self, "_sfa_probe_start_load_count", 0) + 1
            self._sfa_probe_start_load_count = probe_count
            if _probe_enabled(probe_count):
                logger.info(
                    "SFA_PROBE start_load count=%s reqs=%s req_tail=%s "
                    "metadata_reqs=%s row0=%s row1=%s blocks_by_req=%s "
                    "save_tasks=%s",
                    probe_count,
                    num_reqs,
                    [req_id[-8:] for req_id in self.req_ids[:num_reqs]],
                    [request.req_id[-8:] for request in metadata.requests],
                    cpu_block_table_np[0][:8].tolist() if num_reqs else [],
                    cpu_block_table_np[1][:8].tolist() if num_reqs > 1 else [],
                    {
                        req_id[-8:]: len(req_id_to_block_ids.get(req_id, []))
                        for req_id in self.req_ids[:num_reqs]
                    },
                    [
                        len(layer_save_task)
                        for layer_save_task in self.layer_save_tasks[:4]
                    ],
                )

    def get_num_cpu_blocks(self, req_ids: list[str]) -> dict[str, int] | None:
        result = {req_id: self.cpu_blocks_by_req[req_id] for req_id in req_ids if req_id in self.cpu_blocks_by_req}
        return result or None

    def clear_finished_req_ids(self, finished_req_ids: set[str]) -> None:
        for req_id in finished_req_ids:
            self.cpu_blocks_by_req.pop(req_id, None)

    def save_cpu(self, layer_id: int | None = None) -> None:
        if layer_id is None:
            layer_id = self.current_layer_save
            self.current_layer_save += 1
            if self.current_layer_save == self.num_layers:
                self.current_layer_save = 0
        if layer_id < 0 or layer_id >= self.num_layers:
            raise ValueError(f"SFA KV offload layer id out of range: {layer_id}")
        if not self.layer_save_tasks[layer_id]:
            return
        if layer_id in self.submitted_save_layer_ids:
            return
        assert self.kv_send_thread is not None
        self.pending_save_layer_ids.add(layer_id)
        self.submitted_save_layer_ids.add(layer_id)
        if _SFA_DEBUG and self.tp_rank == 0 and layer_id < 2:
            logger.info(
                "SFA_DEBUG worker save_submit layer=%s tasks=%s mappings=%s",
                layer_id,
                len(self.layer_save_tasks[layer_id]),
                [
                    (
                        task.req_id[-8:],
                        _debug_list_head(task.block_ids_npu, 4),
                        _debug_list_head(task.block_ids_cpu, 4),
                    )
                    for task in self.layer_save_tasks[layer_id][:4]
                ],
            )
        self.kv_send_thread.add_request(list(self.layer_save_tasks[layer_id]))

    def save_kv_layer(self, layer_name: str) -> None:
        if _is_current_stream_capturing():
            return
        self.save_cpu(self._get_offload_layer_id(layer_name))

    def ensure_layer_saved(self, layer_name: str) -> None:
        if _is_current_stream_capturing():
            return
        layer_id = self._get_offload_layer_id(layer_name)
        if not self.layer_save_tasks[layer_id]:
            return
        assert self.kv_send_thread is not None
        current_stream = torch_npu.npu.current_stream()
        ready_event = current_stream.record_event()
        self.kv_send_thread.save_stream.wait_event(ready_event)
        if _SFA_DEBUG and self.tp_rank == 0 and layer_id < 2:
            logger.info(
                "SFA_DEBUG worker ensure_before layer=%s tasks=%s",
                layer_id,
                len(self.layer_save_tasks[layer_id]),
            )
        self.save_cpu(layer_id)
        event = self.layer_save_finished_events[layer_id]
        while not event.wait(timeout=1):
            logger.info(f'>>>>> layer {layer_id} waiting for pre-attention save')
        event.clear()
        self.pending_save_layer_ids.discard(layer_id)
        self.submitted_save_layer_ids.discard(layer_id)
        self.layer_save_tasks[layer_id].clear()
        if _SFA_DEBUG and self.tp_rank == 0 and layer_id < 2:
            logger.info("SFA_DEBUG worker ensure_done layer=%s", layer_id)

    def update_cpu_kv_tokens(
        self,
        layer_name: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        token_to_req: torch.Tensor | None = None,
        num_tokens: int | None = None,
    ) -> bool:
        if _is_current_stream_capturing():
            return False
        layer_id = self._get_offload_layer_id(layer_name)
        if num_tokens is None:
            num_tokens = int(slot_mapping.shape[0])
        num_tokens = min(
            int(num_tokens),
            int(slot_mapping.shape[0]),
            int(positions.shape[0]),
        )
        if num_tokens <= 0:
            return False
        if not self.req_ids:
            return False
        assert self.kv_send_thread is not None

        slot_mapping_cpu = slot_mapping[:num_tokens].detach().cpu().to(torch.int64)
        positions_cpu = positions[:num_tokens].detach().cpu().to(torch.int64)
        if token_to_req is None:
            token_to_req_cpu = torch.arange(num_tokens, dtype=torch.int64)
        else:
            token_to_req_cpu = token_to_req[:num_tokens].detach().cpu().to(torch.int64)
        slot_indices = slot_mapping[:num_tokens].clamp_min(0).to(torch.int64)
        key_flat = key_cache.view(-1, key_cache.shape[-1])
        value_flat = value_cache.view(-1, value_cache.shape[-1])
        key_cpu = self.k_caches_cpu[layer_id]
        value_cpu = self.v_caches_cpu[layer_id]
        key_values_cpu = torch.index_select(key_flat, 0, slot_indices).detach().cpu()
        value_values_cpu = torch.index_select(value_flat, 0, slot_indices).detach().cpu()

        return self._update_cpu_kv_tokens_from_cpu(
            layer_name,
            layer_id,
            slot_mapping_cpu,
            positions_cpu,
            token_to_req_cpu,
            key_values_cpu,
            value_values_cpu,
            num_tokens,
            key_cpu,
            value_cpu,
            strict=True,
        )

    def _update_cpu_kv_tokens_from_cpu(
        self,
        layer_name: str,
        layer_id: int,
        slot_mapping_cpu: torch.Tensor,
        positions_cpu: torch.Tensor,
        token_to_req_cpu: torch.Tensor,
        key_values_cpu: torch.Tensor,
        value_values_cpu: torch.Tensor,
        num_tokens: int,
        key_cpu: torch.Tensor,
        value_cpu: torch.Tensor,
        strict: bool,
    ) -> bool:
        if not self.req_ids:
            return False
        src_slots: list[int] = []
        dst_cpu_blocks: list[int] = []
        dst_offsets: list[int] = []
        src_indices: list[int] = []
        samples = []
        cpu_block_table = self.cpu_block_table.np
        for token_idx in range(num_tokens):
            slot = int(slot_mapping_cpu[token_idx].item())
            if slot < 0:
                continue
            req_row = int(token_to_req_cpu[token_idx].item())
            if req_row < 0 or req_row >= len(self.req_ids):
                if not strict:
                    continue
                raise RuntimeError(
                    "SFA CPU token update request row out of range: "
                    f"layer={layer_name}, row={req_row}, reqs={len(self.req_ids)}"
                )
            position = int(positions_cpu[token_idx].item())
            if position < 0:
                continue
            logical_block = position // self.block_size
            if logical_block >= cpu_block_table.shape[1]:
                if not strict:
                    continue
                raise RuntimeError(
                    "SFA CPU token update block index out of range: "
                    f"layer={layer_name}, position={position}, block={logical_block}, "
                    f"table_width={cpu_block_table.shape[1]}"
                )
            cpu_block = int(cpu_block_table[req_row, logical_block])
            if cpu_block <= 0:
                if not strict:
                    continue
                raise RuntimeError(
                    "SFA CPU token update missing CPU block: "
                    f"layer={layer_name}, req={self.req_ids[req_row]}, "
                    f"position={position}, block={logical_block}"
                )
            offset = position % self.block_size
            src_slots.append(slot)
            dst_cpu_blocks.append(cpu_block)
            dst_offsets.append(offset)
            src_indices.append(token_idx)
            if len(samples) < 4:
                samples.append(
                    (
                        req_row,
                        position,
                        slot,
                        cpu_block,
                        offset,
                    )
                )

        if not src_slots:
            return False

        for src_idx, cpu_block, offset in zip(src_indices, dst_cpu_blocks, dst_offsets):
            key_cpu[cpu_block, offset, 0].copy_(key_values_cpu[src_idx])
            value_cpu[cpu_block, offset, 0].copy_(value_values_cpu[src_idx])
        if _SFA_PROBE and self.tp_rank == 0 and layer_id == 0:
            probe_count = getattr(self, "_sfa_probe_token_update_count", 0) + 1
            self._sfa_probe_token_update_count = probe_count
            if _probe_enabled(probe_count):
                logger.info(
                    "SFA_PROBE token_update count=%s layer=%s tokens=%s "
                    "req_tail=%s samples=%s",
                    probe_count,
                    layer_id,
                    len(src_slots),
                    [req_id[-8:] for req_id in self.req_ids[:8]],
                    samples,
                )
        if _SFA_DEBUG and self.tp_rank == 0 and layer_id < 2:
            logger.info(
                "SFA_DEBUG worker token_update layer=%s tokens=%s samples=%s",
                layer_id,
                len(src_slots),
                samples,
            )
        return True

    def wait_for_save(self):
        assert self.use_layerwise
        if not self.pending_save_layer_ids:
            # no save tasks, no need to wait
            return
        for layer_id in sorted(self.pending_save_layer_ids):
            event = self.layer_save_finished_events[layer_id]
            while not event.wait(timeout=1):
                logger.info(f'>>>>> layer {layer_id} waiting for save')
            event.clear()
        self.pending_save_layer_ids.clear()
        self.submitted_save_layer_ids.clear()

    def set_req_ids(self, req_ids: list):
        self.req_ids = req_ids

    def prepare_lru_resident_and_load_cpu(self, args):
        (
            num_reqs,
            miss_count,
            miss_tokens,
            miss_slots,
            lru_req_ids_ptr,
            lru_last_req_ids_ptr,
            lru_topk_indices_ptr,
            lru_slot_to_token_ptr,
            lru_slots_ptr,
            lru_current_slots_ptr,
            lru_miss_count_ptr,
            lru_miss_tokens_ptr,
            lru_miss_slots_ptr,
            block_table,
            block_size,
            token_size_bytes_k,
            token_size_bytes_v,
            gvas_k_bases,
            gvas_v_bases,
            addr_k_bases,
            addr_v_bases,
            lru_token_mark_workspace_ptr,
            lru_token_pos_workspace_ptr,
            lru_slot_workspace_ptr,
            lru_miss_position_workspace_ptr,
            lru_epochs_ptr,
            gvas_buffer,
            addr_buffer,
            size_buffer,
            num_tokens_buffer,
            layer_id,
            do_offload,
            token_update_args,
        ) = args
        token_updated = False
        num_update_tokens_debug = 0
        if token_update_args is not None:
            (
                num_update_tokens,
                update_slots_cpu,
                update_positions_cpu,
                update_token_to_req_cpu,
                update_k_cpu,
                update_v_cpu,
                key_cpu,
                value_cpu,
            ) = token_update_args
            num_update_tokens_debug = int(num_update_tokens)
            if update_token_to_req_cpu is None:
                update_token_to_req_cpu = torch.arange(
                    num_update_tokens,
                    dtype=torch.int64,
                )
            token_updated = self._update_cpu_kv_tokens_from_cpu(
                self.offload_layer_names[layer_id],
                layer_id,
                update_slots_cpu,
                update_positions_cpu,
                update_token_to_req_cpu,
                update_k_cpu,
                update_v_cpu,
                num_update_tokens,
                key_cpu,
                value_cpu,
                strict=False,
            )
        force_lru_miss = _SFA_FORCE_LRU_MISS and num_reqs > 0
        if force_lru_miss:
            self.lru_last_req_ids_cpu_list[layer_id][:num_reqs].fill_(-1)
        cpu_sparse_attn.lru_resident_compact(
            lru_req_ids_ptr,
            lru_last_req_ids_ptr,
            lru_topk_indices_ptr,
            lru_slot_to_token_ptr,
            lru_slots_ptr,
            lru_current_slots_ptr,
            lru_miss_count_ptr,
            lru_miss_tokens_ptr,
            lru_miss_slots_ptr,
            lru_token_mark_workspace_ptr,
            lru_token_pos_workspace_ptr,
            lru_slot_workspace_ptr,
            lru_miss_position_workspace_ptr,
            lru_epochs_ptr,
            num_reqs,
            self.sfa_sparse_topk,
            self.lru_resident_capacity,
            self.max_model_len,
            self.lru_workspace_threads,
            self.lru_workspace_threads,
        )
        num_tokens_to_load = cpu_sparse_attn.compute_lru_resident_addrs(
            miss_count,
            miss_tokens,
            miss_slots,
            block_table,
            block_size,
            token_size_bytes_k,
            token_size_bytes_v,
            gvas_k_bases,
            gvas_v_bases,
            addr_k_bases,
            addr_v_bases,
            self.lru_resident_capacity,
            self.lru_workspace_threads,
            gvas_buffer,
            addr_buffer,
            size_buffer,
            num_tokens_buffer,
        )
        if _SFA_PROBE and self.tp_rank == 0 and layer_id == 0:
            probe_count = getattr(self, "_sfa_probe_lru_cpu_count", 0) + 1
            self._sfa_probe_lru_cpu_count = probe_count
            if _probe_enabled(probe_count):
                logger.info(
                    "SFA_PROBE lru_cpu count=%s layer=%s rows=%s "
                    "do_offload=%s force_lru_miss=%s token_update_args=%s update_tokens=%s "
                    "token_updated=%s load_tokens=%s req_ids=%s miss_count=%s "
                    "topk0=%s current_slots0=%s block_row0=%s "
                    "num_tokens_buffer=%s",
                    probe_count,
                    layer_id,
                    num_reqs,
                    do_offload,
                    force_lru_miss,
                    token_update_args is not None,
                    num_update_tokens_debug,
                    token_updated,
                    int(num_tokens_to_load),
                    self.lru_req_ids_cpu[:min(num_reqs, 8)].tolist(),
                    miss_count[:min(num_reqs, 8)].tolist(),
                    self.lru_topk_indices_cpu[:1, :8].reshape(-1).tolist()
                    if num_reqs
                    else [],
                    self.lru_current_slots_cpu[:1, :8].reshape(-1).tolist()
                    if num_reqs
                    else [],
                    block_table[0, :8].tolist() if num_reqs else [],
                    num_tokens_buffer[:min(num_reqs, 8)].tolist(),
                )

        if _SFA_DEBUG and self.tp_rank == 0 and layer_id < 2:
            miss_counts_head = miss_count[:min(num_reqs, 4)].tolist()
            miss_tokens_head = (
                miss_tokens[:1, :min(self.sfa_sparse_topk, 8)].tolist()
                if num_reqs
                else []
            )
            miss_slots_head = (
                miss_slots[:1, :min(self.sfa_sparse_topk, 8)].tolist()
                if num_reqs
                else []
            )
            logger.info(
                "SFA_DEBUG worker lru_cpu layer=%s num_reqs=%s load_tokens=%s "
                "miss_count=%s miss_tokens0=%s miss_slots0=%s block_row0=%s "
                "do_offload=%s force_lru_miss=%s",
                layer_id,
                num_reqs,
                int(num_tokens_to_load),
                miss_counts_head,
                miss_tokens_head,
                miss_slots_head,
                block_table[0, :16].tolist() if num_reqs else [],
                do_offload,
                force_lru_miss,
            )

        if not do_offload and layer_id == 0 and self.tp_rank == 0:
            logger.info(f'>>>>> load_kv_token_wise, num_tokens_to_load={num_tokens_to_load}')

        if do_offload:
            # in graph mode, we don't want to interrupt graph twice (since it's time consuming),
            # so we start offload here instead of original maybe_save_kv.
            self.save_cpu(layer_id)
            if layer_id in self.pending_save_layer_ids:
                event = self.layer_save_finished_events[layer_id]
                while not event.wait(timeout=1):
                    logger.info(f'>>>>> layer {layer_id} waiting for graph pre-attention save')
                event.clear()
                self.pending_save_layer_ids.discard(layer_id)
                self.submitted_save_layer_ids.discard(layer_id)
                self.layer_save_tasks[layer_id].clear()

    def prepare_lru_resident_and_load(
        self,
        layer_name: str,
        num_tokens: int,
        num_reqs: int,
        topk_indices_npu: torch.Tensor,
        current_slots_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        token_to_req_npu: torch.Tensor | None = None,
        capturing: bool = False,
        key_cache: torch.Tensor | None = None,
        value_cache: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        num_decode_tokens: int | None = None,
    ) -> bool:
        capturing = capturing or _is_current_stream_capturing()
        layer_id = self._get_offload_layer_id(layer_name)
        topk = self.sfa_sparse_topk
        capacity = self.lru_resident_capacity
        if topk > self.sfa_sparse_topk or capacity > self.lru_resident_capacity:
            raise ValueError(
                "LRU resident tensors exceed configured workspace, "
                f"topk={topk}, capacity={capacity}, "
                f"configured_topk={self.sfa_sparse_topk}, "
                f"configured_capacity={self.lru_resident_capacity}"
            )
        if num_tokens > self.max_num_topk_rows:
            raise ValueError(
                "SFA offload topk rows exceed configured workspace, "
                f"num_tokens={num_tokens}, max_num_topk_rows={self.max_num_topk_rows}"
            )
        cpu_block_table_reqs = self.cpu_block_table_host_buffer[:num_reqs]
        cpu_block_table_reqs.copy_(self.cpu_block_table.gpu[:num_reqs], non_blocking=capturing)
        if token_to_req_npu is not None:
            token_to_req_cpu = self.lru_token_to_req_cpu[:num_tokens]
            token_to_req_cpu.copy_(token_to_req_npu[:num_tokens], non_blocking=capturing)
            cpu_block_table_expanded = torch.index_select(
                self.cpu_block_table.gpu[:num_reqs], 0, token_to_req_npu[:num_tokens].to(torch.int64))
            cpu_block_table = self.lru_expanded_block_table_cpu[:num_tokens]
            cpu_block_table.copy_(cpu_block_table_expanded, non_blocking=capturing)
        else:
            cpu_block_table = cpu_block_table_reqs
        topk_indices_cpu = self.lru_topk_indices_cpu[:num_tokens]
        topk_indices_cpu.copy_(topk_indices_npu[:num_tokens], non_blocking=capturing)
        req_ids_cpu = self.lru_req_ids_cpu[:num_tokens]
        req_ids_cpu.copy_(req_ids_npu[:num_tokens], non_blocking=capturing)
        token_update_args = None
        if (
            key_cache is not None
            and value_cache is not None
            and slot_mapping is not None
            and positions is not None
            and num_decode_tokens is not None
        ):
            num_update_tokens = min(
                int(num_decode_tokens),
                int(num_tokens),
                int(slot_mapping.shape[0]),
                int(positions.shape[0]),
                self.max_num_topk_rows,
            )
            if num_update_tokens > 0:
                update_slots_cpu = self.token_update_slots_cpu[:num_update_tokens]
                update_positions_cpu = self.token_update_positions_cpu[:num_update_tokens]
                update_slots_cpu.copy_(
                    slot_mapping[:num_update_tokens],
                    non_blocking=capturing,
                )
                update_positions_cpu.copy_(
                    positions[:num_update_tokens],
                    non_blocking=capturing,
                )
                if token_to_req_npu is None:
                    update_token_to_req_cpu = self.token_update_token_to_req_cpu[:num_update_tokens]
                    update_token_to_req_cpu.copy_(
                        torch.arange(num_update_tokens, dtype=torch.int32),
                    )
                else:
                    update_token_to_req_cpu = self.token_update_token_to_req_cpu[:num_update_tokens]
                    update_token_to_req_cpu.copy_(
                        token_to_req_npu[:num_update_tokens],
                        non_blocking=capturing,
                    )
                slot_indices = slot_mapping[:num_update_tokens].clamp_min(0).to(torch.int64)
                key_values_npu = torch.index_select(
                    key_cache.view(-1, key_cache.shape[-1]),
                    0,
                    slot_indices,
                )
                value_values_npu = torch.index_select(
                    value_cache.view(-1, value_cache.shape[-1]),
                    0,
                    slot_indices,
                )
                update_k_cpu = self.token_update_k_cpu[:num_update_tokens]
                update_v_cpu = self.token_update_v_cpu[:num_update_tokens]
                update_k_cpu.copy_(key_values_npu, non_blocking=capturing)
                update_v_cpu.copy_(value_values_npu, non_blocking=capturing)
                token_update_args = (
                    num_update_tokens,
                    update_slots_cpu,
                    update_positions_cpu,
                    update_token_to_req_cpu,
                    update_k_cpu,
                    update_v_cpu,
                    self.k_caches_cpu[layer_id],
                    self.v_caches_cpu[layer_id],
                )
        if _SFA_DEBUG and self.tp_rank == 0 and layer_id < 2:
            logger.info(
                "SFA_DEBUG worker lru_prepare layer=%s tokens=%s reqs=%s "
                "capturing=%s token_to_req=%s token_update=%s topk0=%s "
                "req_ids=%s block_row0=%s",
                layer_id,
                num_tokens,
                num_reqs,
                capturing,
                token_to_req_npu is not None,
                token_update_args is not None,
                _debug_tensor_head(topk_indices_cpu[:1, :min(topk, 8)]),
                _debug_tensor_head(req_ids_cpu[:min(num_tokens, 8)]),
                cpu_block_table[:1, :16].reshape(-1).tolist() if num_reqs else [],
            )

        args = (
            num_tokens,
            self.lru_miss_count_cpu_list[layer_id][:num_tokens],
            self.lru_miss_tokens_cpu_list[layer_id][:num_tokens],
            self.lru_miss_slots_cpu_list[layer_id][:num_tokens],
            self.lru_req_ids_ptr,
            self.lru_last_req_ids_ptrs[layer_id],
            self.lru_topk_indices_ptr,
            self.lru_slot_to_token_ptrs[layer_id],
            self.lru_slots_ptrs[layer_id],
            self.lru_current_slots_ptr,
            self.lru_miss_count_ptrs[layer_id],
            self.lru_miss_tokens_ptrs[layer_id],
            self.lru_miss_slots_ptrs[layer_id],
            cpu_block_table,
            self.block_size,
            self.token_size_bytes_k,
            self.token_size_bytes_v,
            self.gvas_k_bases[layer_id],
            self.gvas_v_bases[layer_id],
            self.addr_k_bases[layer_id],
            self.addr_v_bases[layer_id],
            self.lru_token_mark_workspace_ptr,
            self.lru_token_pos_workspace_ptr,
            self.lru_slot_workspace_ptr,
            self.lru_miss_position_workspace_ptr,
            self.lru_epochs_ptr,
            self.gvas_buffer_cpu,
            self.addr_buffer_cpu,
            self.size_buffer_cpu,
            self.num_tokens_buffer_cpu,
            layer_id,
            capturing,
            token_update_args,
        )

        if capturing:
            if self.layer_save_tasks[layer_id]:
                self.pending_save_layer_ids.add(layer_id)
            current_compute_stream = torch_npu.npu.current_stream()
            subscribed_compute_streams = get_subscribed_compute_streams()
            if current_compute_stream not in subscribed_compute_streams:
                torch_npu.npu._subscribe_report(current_compute_stream)
                subscribed_compute_streams.add(current_compute_stream)
            torch_npu.npu._launch_host_func(
                current_compute_stream,
                self.prepare_lru_resident_and_load_cpu,
                args,
            )
        else:
            self.prepare_lru_resident_and_load_cpu(args)

        self.batch_copy_args_buffer_npu.copy_(self.batch_copy_args_buffer_cpu, non_blocking=capturing)
        copy_ret = h2d.batch_copy(
            self.gvas_buffer_npu,
            self.addr_buffer_npu,
            self.size_buffer_npu,
            self.num_tokens_buffer_npu,
            self.topk_buffers_k[0].device,
        )
        if _SFA_DEBUG:
            _debug_synchronize_npu()
            if self.tp_rank == 0 and layer_id < 2:
                logger.info(
                    "SFA_DEBUG worker h2d_done layer=%s ret=%s count=%s "
                    "gvas_head=%s addr_head=%s size_head=%s",
                    layer_id,
                    copy_ret,
                    _debug_tensor_head(self.num_tokens_buffer_cpu),
                    _debug_tensor_head(self.gvas_buffer_cpu[:4]),
                    _debug_tensor_head(self.addr_buffer_cpu[:4]),
                    _debug_tensor_head(self.size_buffer_cpu[:4]),
                )
                self._debug_verify_lru_copy(layer_id, num_tokens, cpu_block_table)

        current_slots_cpu = self.lru_current_slots_cpu[:num_tokens]
        current_slots_npu[:num_tokens].copy_(current_slots_cpu, non_blocking=capturing)
        if _SFA_DEBUG and self.tp_rank == 0 and layer_id < 2:
            logger.info(
                "SFA_DEBUG worker lru_done layer=%s current_slots0=%s "
                "num_tokens_buffer=%s",
                layer_id,
                _debug_tensor_head(current_slots_cpu[:1, :min(topk, 8)]),
                _debug_tensor_head(self.num_tokens_buffer_cpu),
            )
        return True

    def process_layer_data(self, request: ReqMeta) -> Generator[
        Optional[torch.Tensor], None, None]:
        """
        Generate kv offload related metadata.
        """
        num_new_offload_blocks = request.num_new_offload_blocks
        if num_new_offload_blocks <= 0:
            return
        block_ids_npu = request.offload_src_hbm_ids or request.block_ids_npu[-num_new_offload_blocks:]
        block_ids_cpu = request.offload_dst_cpu_ids or request.block_ids_cpu[-num_new_offload_blocks:]
        if len(block_ids_npu) != len(block_ids_cpu):
            raise ValueError(
                "SFA KV offload block mapping size mismatch: "
                f"req_id={request.req_id}, npu={block_ids_npu}, cpu={block_ids_cpu}"
            )
        if _SFA_DEBUG and self.tp_rank == 0:
            logger.info(
                "SFA_DEBUG worker process req=%s new_offload=%s src=%s dst=%s",
                request.req_id,
                num_new_offload_blocks,
                _debug_list_head(block_ids_npu),
                _debug_list_head(block_ids_cpu),
            )

        for layer_id in range(self.num_layers):
            req_meta_save = LayerMultiBlockReqMeta(
                request.req_id,
                layer_id,
                block_ids_npu=block_ids_npu,
                block_ids_cpu=block_ids_cpu,
                cache_npu=(self.k_caches_npu[layer_id], self.v_caches_npu[layer_id]),
                cache_cpu=(self.k_caches_cpu[layer_id], self.v_caches_cpu[layer_id]),
            )
            self.layer_save_tasks[layer_id].append(req_meta_save)
