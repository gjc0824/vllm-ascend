#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_input_batch.py
#

import numpy as np
import torch
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.pool.metadata import PoolingStates
from vllm.v1.sample.logits_processor import BatchUpdateBuilder, LogitsProcessors
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_ascend.worker.block_table import MultiGroupBlockTable


class NPUInputBatch(InputBatch):
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],  # The block_size of each kv cache group
        kernel_block_sizes: list[list[int]],
        logitsprocs: LogitsProcessors | None = None,
        logitsprocs_need_output_token_ids: bool = False,
        is_spec_decode: bool = False,
        is_pooling_model: bool = False,
        num_speculative_tokens: int = 0,
        cp_kv_cache_interleave_size: int = 1,
    ):
        self.is_pooling_model = is_pooling_model
        self.is_spec_decode = is_spec_decode
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size
        self._block_sizes = block_sizes.copy()
        self._kernel_block_sizes = [sizes.copy() for sizes in kernel_block_sizes]
        self._num_speculative_tokens = num_speculative_tokens
        self._cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

        self._req_ids: list[str | None] = []
        self.req_id_to_index: dict[str, int] = {}

        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        # This buffer is not directly transferred to the GPU, so it does not
        # need to be pinned.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        self.is_token_ids_tensor = torch.zeros(
            (max_num_reqs, max_model_len), device="cpu", dtype=bool, pin_memory=False
        )
        self.is_token_ids = self.is_token_ids_tensor.numpy()
        # Store prompt embeddings per request to avoid OOM from large upfront
        # allocation if max_model_len is big.
        # Maps req_index -> tensor of shape (num_prompt_tokens, hidden_size)
        self.req_prompt_embeds: dict[int, torch.Tensor] = {}
        self.num_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_tokens_no_spec = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.num_computed_tokens_cpu = self.num_computed_tokens_cpu_tensor.numpy()

        # Block table.
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            block_sizes=block_sizes,
            num_speculative_tokens=num_speculative_tokens,
            kernel_sizes=kernel_block_sizes,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )

        # Sampling-related.
        self.temperature = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)
        self.temperature_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory
        )
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()

        self.top_p = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)
        self.top_p_cpu_tensor = torch.empty((max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory)
        self.top_p_cpu = self.top_p_cpu_tensor.numpy()
        self.top_p_reqs: set[str] = set()

        self.top_k = torch.empty((max_num_reqs,), dtype=torch.int32, device=device)
        self.top_k_cpu_tensor = torch.empty((max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=pin_memory)
        self.top_k_cpu = self.top_k_cpu_tensor.numpy()
        self.top_k_reqs: set[str] = set()

        # IDs of requests which do not support spec decoding
        self.spec_decode_unsupported_reqs: set[str] = set()

        # Frequency penalty related data structures
        self.frequency_penalties = torch.empty((max_num_reqs,), dtype=torch.float, device=device)
        self.frequency_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.frequency_penalties_cpu = self.frequency_penalties_cpu_tensor.numpy()
        self.frequency_penalties_reqs: set[str] = set()

        # Presence penalty related data structures
        self.presence_penalties = torch.empty((max_num_reqs,), dtype=torch.float, device=device)
        self.presence_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.presence_penalties_cpu = self.presence_penalties_cpu_tensor.numpy()
        self.presence_penalties_reqs: set[str] = set()

        # Repetition penalty related data structures
        self.repetition_penalties = torch.empty((max_num_reqs,), dtype=torch.float, device=device)
        self.repetition_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.repetition_penalties_cpu = self.repetition_penalties_cpu_tensor.numpy()
        self.repetition_penalties_reqs: set[str] = set()

        # Speculative decoding
        self.num_accepted_tokens_cpu_tensor = torch.ones(
            (max_num_reqs,), dtype=torch.int64, device="cpu", pin_memory=pin_memory
        )
        self.num_accepted_tokens_cpu = self.num_accepted_tokens_cpu_tensor.numpy()

        # lora related
        self.request_lora_mapping = np.zeros((self.max_num_reqs,), dtype=np.int64)
        self.lora_id_to_request_ids: dict[int, set[str]] = {}
        self.lora_id_to_lora_request: dict[int, LoRARequest] = {}

        # req_index -> generator
        # NOTE(woosuk): The indices of the requests that do not have their own
        # generator should not be included in the dictionary.
        self.generators: dict[int, torch.Generator] = {}

        self.num_logprobs: dict[str, int] = {}

        # To accumulate prompt logprobs tensor chunks across prefill steps.
        self.in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors] = {}

        # Internal representation of per-step batch state changes, used for
        # reordering persistent batch and generating logitsprocs batch state
        # updates. Should reset each step.
        self.batch_update_builder = BatchUpdateBuilder()

        # TODO convert this to LogitsProcessor
        self.has_allowed_token_ids: set[str] = set()
        # NOTE(lufang): In the mask tensor, if the corresponding token allowed,
        # the value is False. Since we use masked_fill_ to set -inf.
        self.allowed_token_ids_mask: torch.Tensor | None = None
        self.allowed_token_ids_mask_cpu_tensor: torch.Tensor | None = None

        # req_index -> bad_words_token_ids
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        self.logits_processing_needs_token_ids = np.zeros(max_num_reqs, dtype=bool)

        self.req_output_token_ids: list[list[int] | None] = []

        # Store provided logitsprocs. If none are provided, initialize empty
        # data structure
        self.logitsprocs = logitsprocs or LogitsProcessors()
        self.logitsprocs_need_output_token_ids = logitsprocs_need_output_token_ids

        # Store last speculative tokens for sampler.
        self.spec_token_ids: list[list[int]] = [[] for _ in range(max_num_reqs)]

        # This is updated each time the batch constituents change.
        self.sampling_metadata = self._make_sampling_metadata()

        # for pooling models
        self.pooling_params: dict[str, PoolingParams] = {}
        self.pooling_states: dict[str, PoolingStates] = {}

        # Cached reference to the GPU tensor of previously sampled tokens
        self.prev_sampled_token_ids: torch.Tensor | None = None
        self.prev_req_id_to_index: dict[str, int] | None = None
        # These are used to update output_token_ids with real sampled
        # ids from prior step, if required by current sampling params
        # (e.g. penalties).
        self.sampled_token_ids_cpu: torch.Tensor | None = None
        self.async_copy_ready_event: torch.Event | None = None

    def clone_for_vpp_sampling(self) -> "NPUInputBatch":
        """Clone the sampling-visible batch state for a yielded VPP batch."""
        clone = NPUInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_batched_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.vocab_size,
            block_sizes=self._block_sizes,
            kernel_block_sizes=self._kernel_block_sizes,
            logitsprocs=self.logitsprocs,
            logitsprocs_need_output_token_ids=self.logitsprocs_need_output_token_ids,
            is_spec_decode=self.is_spec_decode,
            is_pooling_model=self.is_pooling_model,
            num_speculative_tokens=self._num_speculative_tokens,
            cp_kv_cache_interleave_size=self._cp_kv_cache_interleave_size,
        )

        clone._req_ids = self._req_ids.copy()
        clone.req_id_to_index = self.req_id_to_index.copy()
        clone.token_ids_cpu_tensor.copy_(self.token_ids_cpu_tensor)
        clone.is_token_ids_tensor.copy_(self.is_token_ids_tensor)
        clone.req_prompt_embeds = self.req_prompt_embeds.copy()
        clone.num_tokens = self.num_tokens.copy()
        clone.num_tokens_no_spec = self.num_tokens_no_spec.copy()
        clone.num_prompt_tokens = self.num_prompt_tokens.copy()
        clone.num_computed_tokens_cpu_tensor.copy_(self.num_computed_tokens_cpu_tensor)

        for src_table, dst_table in zip(
            self.block_table.block_tables, clone.block_table.block_tables
        ):
            dst_table.num_blocks_per_row = src_table.num_blocks_per_row.copy()
            dst_table.block_table.cpu.copy_(src_table.block_table.cpu)
            dst_table.block_table.gpu.copy_(src_table.block_table.gpu)
            dst_table.slot_mapping.cpu.copy_(src_table.slot_mapping.cpu)
            dst_table.slot_mapping.gpu.copy_(src_table.slot_mapping.gpu)

        clone.temperature_cpu_tensor.copy_(self.temperature_cpu_tensor)
        clone.greedy_reqs = self.greedy_reqs.copy()
        clone.random_reqs = self.random_reqs.copy()

        clone.top_p_cpu_tensor.copy_(self.top_p_cpu_tensor)
        clone.top_p_reqs = self.top_p_reqs.copy()

        clone.top_k_cpu_tensor.copy_(self.top_k_cpu_tensor)
        clone.top_k_reqs = self.top_k_reqs.copy()

        clone.spec_decode_unsupported_reqs = self.spec_decode_unsupported_reqs.copy()

        clone.frequency_penalties_cpu_tensor.copy_(self.frequency_penalties_cpu_tensor)
        clone.frequency_penalties_reqs = self.frequency_penalties_reqs.copy()

        clone.presence_penalties_cpu_tensor.copy_(self.presence_penalties_cpu_tensor)
        clone.presence_penalties_reqs = self.presence_penalties_reqs.copy()

        clone.repetition_penalties_cpu_tensor.copy_(self.repetition_penalties_cpu_tensor)
        clone.repetition_penalties_reqs = self.repetition_penalties_reqs.copy()

        clone.num_accepted_tokens_cpu_tensor.copy_(self.num_accepted_tokens_cpu_tensor)

        clone.request_lora_mapping = self.request_lora_mapping.copy()
        clone.lora_id_to_request_ids = {
            lora_id: request_ids.copy()
            for lora_id, request_ids in self.lora_id_to_request_ids.items()
        }
        clone.lora_id_to_lora_request = self.lora_id_to_lora_request.copy()

        clone.generators = self.generators.copy()
        clone.num_logprobs = self.num_logprobs.copy()
        clone.in_progress_prompt_logprobs_cpu = self.in_progress_prompt_logprobs_cpu.copy()

        clone.has_allowed_token_ids = self.has_allowed_token_ids.copy()
        if self.allowed_token_ids_mask_cpu_tensor is not None:
            clone.allowed_token_ids_mask_cpu_tensor = self.allowed_token_ids_mask_cpu_tensor.clone()
        if self.allowed_token_ids_mask is not None:
            clone.allowed_token_ids_mask = self.allowed_token_ids_mask.clone()

        clone.bad_words_token_ids = {
            req_idx: [token_ids.copy() for token_ids in bad_words]
            for req_idx, bad_words in self.bad_words_token_ids.items()
        }
        clone.logits_processing_needs_token_ids = self.logits_processing_needs_token_ids.copy()
        clone.req_output_token_ids = self.req_output_token_ids.copy()
        clone.spec_token_ids = [token_ids.copy() for token_ids in self.spec_token_ids]
        clone.pooling_params = self.pooling_params.copy()
        clone.pooling_states = self.pooling_states.copy()
        clone.prev_sampled_token_ids = self.prev_sampled_token_ids
        clone.prev_req_id_to_index = (
            None if self.prev_req_id_to_index is None else self.prev_req_id_to_index.copy()
        )
        clone.sampled_token_ids_cpu = self.sampled_token_ids_cpu
        clone.async_copy_ready_event = self.async_copy_ready_event
        clone.sampling_metadata = clone._make_sampling_metadata()
        return clone
