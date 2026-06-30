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
from zbal import zbal_init, zbal_uninit, empty_tensor, batch_copy, zbal_h2d_init

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import (
    SFAKVOffloadConnectorMetadata,
    LayerMultiBlockReqMeta,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.kv_transfer import (
    KVCacheStoreLayerSendingThread,
    KVTransferThread,
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
        self.block_size = self.group_block_sizes[-1] # only offload kv cache

        self.current_layer_save = 0
        self.current_layer_load = 0
        self.num_layers = model_config.get_num_layers(parallel_config)

        if self.use_mla:
            self.num_kv_head = 1
        else:
            self.num_kv_head = model_config.get_total_num_kv_heads()

        self.kv_send_thread: KVTransferThread | None = None

        self.layer_save_tasks = [[] for _ in range(self.num_layers)]
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        lru_resident_config = ascend_config.lru_resident_cache_config
        self.sfa_sparse_topk = lru_resident_config.topk
        self.lru_resident_capacity = lru_resident_config.buffer_size

        # TODO get from config
        head_num = 1
        head_dim_k = 512
        head_dim_v = 64
        dtype = torch.bfloat16
        self.token_size_bytes_k = head_num * head_dim_k * dtype.itemsize
        self.token_size_bytes_v = head_num * head_dim_v * dtype.itemsize
        self.max_model_len = vllm_config.model_config.max_model_len
        max_block_num = cdiv(self.max_model_len, self.block_size)
        self.cpu_block_table = CpuGpuBuffer(self.max_num_reqs, max_block_num, dtype=torch.int32, device='npu', pin_memory=True)
        self.cpu_block_table_host_buffer = torch.zeros([self.max_num_reqs, max_block_num], dtype=torch.int32, device='cpu', pin_memory=True)
        self.actual_seq_len_q = torch.arange(self.max_num_reqs, dtype=torch.int32, device='cpu', pin_memory=True) + 1
        self.req_ids = []

        self.cpu_sparse_attn = cpu_sparse_attn

        self.load_stream = None
        self.load_stream = torch_npu.npu.Stream()
        self.save_stream = None
        self.side_compute_stream = torch_npu.npu.Stream()
        self.kv_cache_config.num_blocks
        allocate_dram_size = 40 * 1024 * 1024 * 1024 # 64GB, TODO get from config
        zbal_h2d_init(allocate_dram_size, self.max_num_reqs * self.sfa_sparse_topk * 2)

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

        if self.use_sparse and self.use_offload:
            self.k_caches_npu: list[torch.Tensor] = []
            self.v_caches_npu: list[torch.Tensor] = []
            self.topk_buffers_k: list[torch.Tensor] = []
            self.topk_buffers_v: list[torch.Tensor] = []
            for cache_or_caches in kv_caches.values():
                assert len(cache_or_caches) == 5
                self.k_caches_npu.append(cache_or_caches[0])
                self.v_caches_npu.append(cache_or_caches[1])
                self.topk_buffers_k.append(cache_or_caches[3])
                self.topk_buffers_v.append(cache_or_caches[4])

            npu_block_num = self.num_blocks
            # we need 4 * npu_blocks of cpu_blocks to fully store all offload blocks (dskv32, 512/128)
            # but you may want to set this to 1 in debug case in case of allocating to much dram
            # TODO remove this and directly compute from model config before merge
            cpu_block_num_multiple = 1
            cpu_block_num = npu_block_num * cpu_block_num_multiple
            cpu_cache_size_single_card = cpu_block_num * self.block_size * (512 + 64) * torch.bfloat16.itemsize * self.num_layers
            logger.info(f'KV offload allocate {cpu_block_num} cpu blocks, size = {cpu_cache_size_single_card / 1024 / 1024 / 1024} GB per rank')
            self.k_caches_cpu: list[torch.Tensor] = [empty_tensor([cpu_block_num, self.block_size, 1, 512], dtype=torch.bfloat16, pin_memory=True) for _ in range(self.num_layers)]
            self.v_caches_cpu: list[torch.Tensor] = [empty_tensor([cpu_block_num, self.block_size, 1, 64], dtype=torch.bfloat16, pin_memory=True) for _ in range(self.num_layers)]

            # topk cache reuse related
            self.lru_workspace_threads = 8
            self.lru_topk_indices_cpu = torch.empty(
                [self.max_num_reqs, self.sfa_sparse_topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_slot_to_token_cpu_list = [torch.full(
                [self.max_num_reqs, self.lru_resident_capacity],
                -1,
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            ) for _ in range(self.num_layers)]
            self.lru_slots_cpu_list = [torch.arange(
                self.lru_resident_capacity,
                dtype=torch.int32,
                device='cpu',
            ).view(1, -1).repeat(self.max_num_reqs, 1).pin_memory() for _ in range(self.num_layers)]
            self.lru_current_slots_cpu = torch.empty(
                [self.max_num_reqs, self.sfa_sparse_topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            )
            self.lru_miss_count_cpu_list = [torch.empty(
                [self.max_num_reqs],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            ) for _ in range(self.num_layers)]
            self.lru_miss_tokens_cpu_list = [torch.empty(
                [self.max_num_reqs, self.sfa_sparse_topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            ) for _ in range(self.num_layers)]
            self.lru_miss_slots_cpu_list = [torch.empty(
                [self.max_num_reqs, self.sfa_sparse_topk],
                dtype=torch.int32,
                device='cpu',
                pin_memory=True,
            ) for _ in range(self.num_layers)]
            self.lru_req_ids_cpu = torch.empty([self.max_num_reqs], dtype=torch.int64, device='cpu', pin_memory=True)
            self.lru_last_req_ids_cpu_list = [torch.full(
                [self.max_num_reqs],
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

            # sparse h2d (batch_copy related)
            self.addr_k_bases: list[int] = [t.data_ptr() for t in self.topk_buffers_k]
            self.addr_v_bases: list[int] = [t.data_ptr() for t in self.topk_buffers_v]
            self.gvas_k_bases: list[int] = [t.data_ptr() for t in self.k_caches_cpu]
            self.gvas_v_bases: list[int] = [t.data_ptr() for t in self.v_caches_cpu]

            gvas_buffer_offset = 0
            gvas_buffer_size_bytes = self.max_num_reqs * self.sfa_sparse_topk * 2 * 8 # 2: k+v, 8: int64
            addr_buffer_offset = gvas_buffer_offset + gvas_buffer_size_bytes
            addr_buffer_size_bytes = self.max_num_reqs * self.sfa_sparse_topk * 2 * 8
            size_buffer_offset = addr_buffer_offset + addr_buffer_size_bytes
            size_buffer_size_bytes = self.max_num_reqs * self.sfa_sparse_topk * 2 * 4 # 2: k+v, 4: int32
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
            assert self.gvas_buffer_cpu.shape == torch.Size([self.max_num_reqs * self.sfa_sparse_topk * 2])
            assert self.addr_buffer_cpu.shape == torch.Size([self.max_num_reqs * self.sfa_sparse_topk * 2])
            assert self.size_buffer_cpu.shape == torch.Size([self.max_num_reqs * self.sfa_sparse_topk * 2])
            assert self.num_tokens_buffer_cpu.shape == torch.Size([1])

            self.gvas_buffer_npu = self.batch_copy_args_buffer_npu[gvas_buffer_offset:gvas_buffer_offset + gvas_buffer_size_bytes].view(torch.int64)
            self.addr_buffer_npu = self.batch_copy_args_buffer_npu[addr_buffer_offset:addr_buffer_offset + addr_buffer_size_bytes].view(torch.int64)
            self.size_buffer_npu = self.batch_copy_args_buffer_npu[size_buffer_offset:size_buffer_offset + size_buffer_size_bytes].view(torch.int32)
            self.num_tokens_buffer_npu = \
                self.batch_copy_args_buffer_npu[num_tokens_buffer_offset:num_tokens_buffer_offset + num_tokens_buffer_size_bytes].view(torch.int32)
            assert self.gvas_buffer_npu.shape == torch.Size([self.max_num_reqs * self.sfa_sparse_topk * 2])
            assert self.addr_buffer_npu.shape == torch.Size([self.max_num_reqs * self.sfa_sparse_topk * 2])
            assert self.size_buffer_npu.shape == torch.Size([self.max_num_reqs * self.sfa_sparse_topk * 2])
            assert self.num_tokens_buffer_npu.shape == torch.Size([1])

    def start_load_kv(self, metadata: SFAKVOffloadConnectorMetadata):
        # return
        self.current_layer_save = 0
        self.current_layer_load = 0
        req_id_to_block_ids: dict[str, list[int]] = {}
        for layer_save_task in self.layer_save_tasks:
            layer_save_task.clear()
        for request in metadata.requests:
            req_id_to_block_ids[request.req_id] = request.block_ids_cpu
            if request.num_new_offload_blocks <= 0:
                continue # no new blocks to save
            self.process_layer_data(request)
        self.num_save_tasks = len(self.layer_save_tasks[0])
        if self.tp_rank == 0:
            logger.info(f'>>>>> start load kv, reqs num: {len(metadata.requests)}, save task num = {len(self.layer_save_tasks[0])}')

        # generate block_table for load
        # NOTE reqs in self.req_ids and metadata.requests may not be in same order,
        # use reqs from self.req_ids (order of actual batch) to compute block_table.
        num_reqs = len(self.req_ids)
        cpu_block_table_np = self.cpu_block_table.np[:num_reqs]
        cpu_block_table_np.fill(0)
        for i, req_id in enumerate(self.req_ids[:num_reqs]):
            cpu_block_ids = req_id_to_block_ids[req_id]
            cpu_block_table_np[i][:len(cpu_block_ids)] = np.array([cpu_block_ids], dtype=np.int32)
        self.cpu_block_table.copy_to_gpu(num_reqs)

    def save_cpu(self):
        self.kv_send_thread.add_request(self.layer_save_tasks[self.current_layer_save])
        self.current_layer_save += 1
        if self.current_layer_save == self.num_layers:
            self.current_layer_save = 0

    def save_kv_layer(self) -> None:
        self.save_cpu()

    def wait_for_save(self):
        assert self.use_layerwise
        if self.num_save_tasks == 0:
            # no save tasks, no need to wait
            return
        for layer_id, event in enumerate(self.layer_save_finished_events):
            is_finish = event.wait(timeout=1)
            if not is_finish:
                logger.info(f'>>>>> layer {layer_id} wait for save timeout')
            event.clear()
 
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
            do_offload,
        ) = args
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

        if not do_offload and self.current_layer_load == 0 and self.tp_rank == 0:
            logger.info(f'>>>>> load_kv_token_wise, num_tokens_to_load={num_tokens_to_load}')

        if do_offload:
            # in graph mode, we don't want to interrupt graph twice (since it's time consuming),
            # so we start offload here instead of original maybe_save_kv.
            self.save_cpu()

    def prepare_lru_resident_and_load(
        self,
        layer_name: str,
        num_reqs: int,
        topk_indices_npu: torch.Tensor,
        current_slots_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        capturing: bool = False,
    ) -> bool:
        capturing = capturing or _is_current_stream_capturing()
        topk = self.sfa_sparse_topk
        capacity = self.lru_resident_capacity
        if topk > self.sfa_sparse_topk or capacity > self.lru_resident_capacity:
            raise ValueError(
                "LRU resident tensors exceed configured workspace, "
                f"topk={topk}, capacity={capacity}, "
                f"configured_topk={self.sfa_sparse_topk}, "
                f"configured_capacity={self.lru_resident_capacity}"
            )
        cpu_block_table = self.cpu_block_table_host_buffer[:num_reqs]
        cpu_block_table.copy_(self.cpu_block_table.gpu[:num_reqs], non_blocking=capturing)
        topk_indices_cpu = self.lru_topk_indices_cpu[:num_reqs]
        topk_indices_cpu.copy_(topk_indices_npu[:num_reqs], non_blocking=capturing)
        req_ids_cpu = self.lru_req_ids_cpu[:num_reqs]
        req_ids_cpu.copy_(req_ids_npu[:num_reqs], non_blocking=capturing)

        args = (
            num_reqs,
            self.lru_miss_count_cpu_list[self.current_layer_load][:num_reqs],
            self.lru_miss_tokens_cpu_list[self.current_layer_load][:num_reqs],
            self.lru_miss_slots_cpu_list[self.current_layer_load][:num_reqs],
            self.lru_req_ids_ptr,
            self.lru_last_req_ids_ptrs[self.current_layer_load],
            self.lru_topk_indices_ptr,
            self.lru_slot_to_token_ptrs[self.current_layer_load],
            self.lru_slots_ptrs[self.current_layer_load],
            self.lru_current_slots_ptr,
            self.lru_miss_count_ptrs[self.current_layer_load],
            self.lru_miss_tokens_ptrs[self.current_layer_load],
            self.lru_miss_slots_ptrs[self.current_layer_load],
            cpu_block_table,
            self.block_size,
            self.token_size_bytes_k,
            self.token_size_bytes_v,
            self.gvas_k_bases[self.current_layer_load],
            self.gvas_v_bases[self.current_layer_load],
            self.addr_k_bases[self.current_layer_load],
            self.addr_v_bases[self.current_layer_load],
            self.lru_token_mark_workspace_ptr,
            self.lru_token_pos_workspace_ptr,
            self.lru_slot_workspace_ptr,
            self.lru_miss_position_workspace_ptr,
            self.lru_epochs_ptr,
            self.gvas_buffer_cpu,
            self.addr_buffer_cpu,
            self.size_buffer_cpu,
            self.num_tokens_buffer_cpu,
            capturing,
        )

        if capturing:
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
        batch_copy(
            self.gvas_buffer_npu,
            self.addr_buffer_npu,
            self.size_buffer_npu,
            self.num_tokens_buffer_npu,
            self.topk_buffers_k[0].device,
        )

        current_slots_cpu = self.lru_current_slots_cpu[:num_reqs]
        current_slots_npu[:num_reqs, :topk].copy_(current_slots_cpu, non_blocking=capturing)

        self.current_layer_load += 1
        if self.current_layer_load == self.num_layers:
            self.current_layer_load = 0
        return True

    def process_layer_data(self, request: ReqMeta) -> Generator[
        Optional[torch.Tensor], None, None]:
        """
        Generate kv offload related metadata.
        """
        num_new_offload_blocks = request.num_new_offload_blocks
        block_ids_npu = request.block_ids_npu
        block_ids_cpu = request.block_ids_cpu
        if len(block_ids_npu) > len(block_ids_cpu):
            # in most cases block_ids_npu has one more unfull block, remove it
            block_ids_npu = block_ids_npu[:-1]
        assert len(block_ids_npu) == len(block_ids_cpu)
        block_ids_npu = block_ids_npu[-num_new_offload_blocks:]
        block_ids_cpu = block_ids_cpu[-num_new_offload_blocks:]

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
