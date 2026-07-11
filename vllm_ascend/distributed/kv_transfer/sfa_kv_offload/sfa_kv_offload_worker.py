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
        for request in metadata.requests:
            req_id_to_block_ids[request.req_id] = request.block_ids_cpu
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
        if missing_req_ids and self.tp_rank == 0:
            logger.info(
                "SFA KV offload skipped %s stale req ids absent from "
                "metadata while building CPU block table.",
                len(missing_req_ids),
            )
        self.cpu_block_table.copy_to_gpu(num_reqs)

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
        ready_event = torch_npu.npu.current_stream().record_event()
        self.kv_send_thread.save_stream.wait_event(ready_event)
        self.pending_save_layer_ids.add(layer_id)
        self.submitted_save_layer_ids.add(layer_id)
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
        self.save_cpu(layer_id)
        event = self.layer_save_finished_events[layer_id]
        while not event.wait(timeout=1):
            logger.info(f'>>>>> layer {layer_id} waiting for pre-attention save')
        event.clear()
        self.pending_save_layer_ids.discard(layer_id)
        self.submitted_save_layer_ids.discard(layer_id)
        self.layer_save_tasks[layer_id].clear()

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

        if not src_slots:
            return False

        for src_idx, cpu_block, offset in zip(src_indices, dst_cpu_blocks, dst_offsets):
            key_cpu[cpu_block, offset, 0].copy_(key_values_cpu[src_idx])
            value_cpu[cpu_block, offset, 0].copy_(value_values_cpu[src_idx])
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
            if update_token_to_req_cpu is None:
                update_token_to_req_cpu = torch.arange(
                    num_update_tokens,
                    dtype=torch.int64,
                )
            self._update_cpu_kv_tokens_from_cpu(
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
        update_token_indices_npu: torch.Tensor | None = None,
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
            source_rows = min(
                int(num_tokens),
                int(slot_mapping.shape[0]),
                int(positions.shape[0]),
            )
            max_update_tokens = min(
                source_rows,
                self.max_num_topk_rows,
            )
            if update_token_indices_npu is not None:
                max_update_tokens = min(
                    max_update_tokens,
                    int(update_token_indices_npu.shape[0]),
                )
            num_update_tokens = min(
                int(num_decode_tokens),
                max_update_tokens,
            )
            if num_update_tokens > 0:
                if update_token_indices_npu is not None:
                    raw_update_indices = update_token_indices_npu[
                        :num_update_tokens
                    ].to(torch.int64)
                    valid_update_mask = (
                        (raw_update_indices >= 0)
                        & (raw_update_indices < source_rows)
                    )
                    update_indices = raw_update_indices.clamp(
                        min=0,
                        max=source_rows - 1,
                    )
                    update_slots_npu = torch.index_select(
                        slot_mapping,
                        0,
                        update_indices,
                    )
                    update_slots_npu = torch.where(
                        valid_update_mask,
                        update_slots_npu,
                        torch.full_like(update_slots_npu, -1),
                    )
                    update_positions_npu = torch.index_select(
                        positions,
                        0,
                        update_indices,
                    )
                    update_token_to_req_npu = (
                        torch.index_select(
                            token_to_req_npu,
                            0,
                            update_indices,
                        )
                        if token_to_req_npu is not None
                        else None
                    )
                else:
                    update_slots_npu = slot_mapping[:num_update_tokens]
                    update_positions_npu = positions[:num_update_tokens]
                    update_token_to_req_npu = (
                        token_to_req_npu[:num_update_tokens]
                        if token_to_req_npu is not None
                        else None
                    )
                update_slots_cpu = self.token_update_slots_cpu[:num_update_tokens]
                update_positions_cpu = self.token_update_positions_cpu[:num_update_tokens]
                update_slots_cpu.copy_(
                    update_slots_npu,
                    non_blocking=capturing,
                )
                update_positions_cpu.copy_(
                    update_positions_npu,
                    non_blocking=capturing,
                )
                if update_token_to_req_npu is None:
                    update_token_to_req_cpu = self.token_update_token_to_req_cpu[:num_update_tokens]
                    update_token_to_req_cpu.copy_(
                        torch.arange(num_update_tokens, dtype=torch.int32),
                    )
                else:
                    update_token_to_req_cpu = self.token_update_token_to_req_cpu[:num_update_tokens]
                    update_token_to_req_cpu.copy_(
                        update_token_to_req_npu,
                        non_blocking=capturing,
                    )
                slot_indices = update_slots_npu.clamp_min(0).to(torch.int64)
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
        h2d.batch_copy(
            self.gvas_buffer_npu,
            self.addr_buffer_npu,
            self.size_buffer_npu,
            self.num_tokens_buffer_npu,
            self.topk_buffers_k[0].device,
        )

        current_slots_cpu = self.lru_current_slots_cpu[:num_tokens]
        current_slots_npu[:num_tokens].copy_(current_slots_cpu, non_blocking=capturing)
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
