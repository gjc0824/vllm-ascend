import os
from itertools import product

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

logger = init_logger(__name__)
_GRAPH_DEBUG = bool(int(os.getenv("VLLM_ASCEND_GRAPH_DEBUG", "1")))
_ORIG_INITIALIZE_CUDAGRAPH_KEYS = CudagraphDispatcher.initialize_cudagraph_keys


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.lower() in ("1", "true", "yes", "on")


def _sequence_parallel_padding_enabled(self) -> bool:
    pass_config = getattr(self.compilation_config, "pass_config", None)
    if bool(getattr(pass_config, "enable_sp", False)):
        return True

    additional_config = getattr(self.vllm_config, "additional_config", None)
    if isinstance(additional_config, dict):
        if bool(additional_config.get("enable_flashcomm1", False)):
            return True
        if bool(additional_config.get("enable_sp_by_pass", False)):
            return True

    return _env_flag_enabled("VLLM_ASCEND_ENABLE_FLASHCOMM1")


def _create_padded_batch_descriptor(
    self,
    num_tokens: int,
    uniform_decode: bool,
    has_lora: bool,
    num_active_loras: int = 0,
) -> BatchDescriptor:
    max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
    uniform_decode_query_len = self.uniform_decode_query_len
    num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]

    # FULL mode should not be treated as uniform decode
    if (
        uniform_decode
        and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL)
        and self.cudagraph_mode != CUDAGraphMode.FULL
    ):
        num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
        assert num_tokens_padded % uniform_decode_query_len == 0
    else:
        uniform_decode = False
        num_reqs = min(num_tokens_padded, max_num_seqs)

    return BatchDescriptor(
        num_tokens=num_tokens_padded,
        num_reqs=num_reqs,
        uniform=uniform_decode,
        has_lora=has_lora,
        num_active_loras=num_active_loras,
    )


def _reachable_decode_capture_sizes(
    self,
    uniform_decode_query_len: int,
) -> list[int]:
    capture_sizes = self.compilation_config.cudagraph_capture_sizes
    max_size = self.compilation_config.max_cudagraph_capture_size
    if not capture_sizes or max_size is None:
        return []

    tp_size = self.vllm_config.parallel_config.tensor_parallel_size
    if tp_size <= 1 or not _sequence_parallel_padding_enabled(self):
        return []

    max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
    graph_sizes: set[int] = set()

    for num_reqs in range(1, max_num_seqs + 1):
        raw_tokens = uniform_decode_query_len * num_reqs
        dispatch_tokens = _round_up(raw_tokens, tp_size)
        if dispatch_tokens > max_size:
            continue
        graph_size = self._bs_to_padded_graph_size[dispatch_tokens]
        if graph_size in capture_sizes:
            graph_sizes.add(graph_size)

    return sorted(graph_sizes)


def _initialize_cudagraph_keys(
    self,
    cudagraph_mode: CUDAGraphMode,
    uniform_decode_query_len: int = 1,
):
    _ORIG_INITIALIZE_CUDAGRAPH_KEYS(
        self,
        cudagraph_mode,
        uniform_decode_query_len,
    )

    if not (
        cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        and cudagraph_mode.separate_routine()
    ):
        return

    decode_capture_sizes = _reachable_decode_capture_sizes(
        self,
        uniform_decode_query_len,
    )
    if not decode_capture_sizes:
        return

    lora_cases = self._get_lora_cases()
    for bs, num_active_loras in product(decode_capture_sizes, lora_cases):
        self.add_cudagraph_key(
            CUDAGraphMode.FULL,
            self._create_padded_batch_descriptor(
                bs,
                True,
                num_active_loras > 0,
                num_active_loras,
            ),
        )

    if _GRAPH_DEBUG:
        logger.info_once(
            "SFA_DEBUG cudagraph_decode_keys sizes=%s max_num_seqs=%s "
            "tp_size=%s query_len=%s",
            tuple(decode_capture_sizes),
            self.vllm_config.scheduler_config.max_num_seqs,
            self.vllm_config.parallel_config.tensor_parallel_size,
            uniform_decode_query_len,
        )


CudagraphDispatcher._create_padded_batch_descriptor = _create_padded_batch_descriptor
CudagraphDispatcher.initialize_cudagraph_keys = _initialize_cudagraph_keys
