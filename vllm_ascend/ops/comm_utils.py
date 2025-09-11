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
import torch
import torch.distributed
import torch.distributed as dist
import torch_npu
from vllm.forward_context import get_forward_context

COMM_STREAM = None


def get_sp_metadata_context():
    attn_metadata = get_forward_context().attn_metadata
    sp_metadata = None
    enable_sp = False
    if attn_metadata is not None and attn_metadata.prefill is not None:
        if hasattr(attn_metadata.prefill, 'sp_metadata'):
            sp_metadata = attn_metadata.prefill.sp_metadata if attn_metadata.prefill.sp_metadata is not None else None
            enable_sp = sp_metadata.enable_sp if sp_metadata is not None else False
    return sp_metadata, enable_sp

def async_all_to_all(input_,
                     output_split_sizes,
                     input_split_sizes,
                     group,
                     event=None):
    if output_split_sizes is None:
        # Equal split (all2all)
        a2a_out = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.npu.current_device(),
        )

    if event:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(
                device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            event.wait()
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True)
    else:
        handle = dist.all_to_all_single(a2a_out,
                                        input_.contiguous(),
                                        output_split_sizes=output_split_sizes,
                                        input_split_sizes=input_split_sizes,
                                        group=group,
                                        async_op=True)
    return input_, a2a_out, handle
