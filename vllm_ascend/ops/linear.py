"""
Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
This file is a part of the vllm-ascend project.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Optional, Union, Tuple, Callable

import torch
import torch_npu
from torch import nn
from torch.nn.parameter import Parameter
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter)
from vllm.model_executor.layers.linear import (WEIGHT_LOADER_V2_SUPPORTED,
                                               ColumnParallelLinear,
                                               LinearBase,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear,
                                               UnquantizedLinearMethod,
                                               ReplicatedLinear)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP
from vllm.model_executor.utils import set_weight_attrs
from vllm.forward_context import get_forward_context

from vllm_ascend.distributed.parallel_state import (
    get_mlp_tensor_model_parallel_rank,
    get_mlp_tensor_model_parallel_world_size, 
    get_mlp_tp_group,
    is_sp_enabled)
from vllm_ascend.quantization.quant_config import AscendLinearMethod
from vllm_ascend.quantization.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.ascend_config import get_ascend_config


class AscendMlpColumnParallelLinear(ColumnParallelLinear):
    """Linear layer with column parallelism.

    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        output_sizes: Optional[list[int]] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
    ):
        # Divide the weight matrix along the last dimension.
        if prefix.find("gate_up_proj") != -1:
            self.tp_size = get_mlp_tensor_model_parallel_world_size()
            self.tp_rank = get_mlp_tensor_model_parallel_rank()
            self.enable_mlp_optimze = True
        else:
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_rank = get_tensor_model_parallel_rank()
            self.enable_mlp_optimze = False
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size)
                for output_size in self.output_sizes
            ]
        LinearBase.__init__(self,
                            input_size,
                            output_size,
                            skip_bias_add,
                            params_dtype,
                            quant_config,
                            prefix,
                            return_bias=return_bias)

        self.gather_output = gather_output

        if output_sizes is None:
            output_sizes = [output_size]

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)


class AscendMlpRowParallelLinear(RowParallelLinear):
    """Linear layer with row parallelism.
    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
    ):
        if prefix.find("down_proj") != -1:
            self.tp_size = get_mlp_tensor_model_parallel_world_size()
            self.tp_rank = get_mlp_tensor_model_parallel_rank()
            self.enable_mlp_optimze = True
        else:
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_rank = get_tensor_model_parallel_rank()
            self.enable_mlp_optimze = False
        # Divide the weight matrix along the first dimension.
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        LinearBase.__init__(self,
                            input_size,
                            output_size,
                            skip_bias_add,
                            params_dtype,
                            quant_config,
                            prefix,
                            return_bias=return_bias)

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        input_,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        forward_context = get_forward_context()
        self.enable_sp = forward_context.enable_sp
        attn_metadata = forward_context.attn_metadata
        is_prefill = attn_metadata.num_prefills if attn_metadata else False
        if self.enable_mlp_optimze and not self.enable_sp:
            tp_rank = get_mlp_tensor_model_parallel_rank()
            if self.input_is_parallel:
                input_parallel = input_
            else:
                tp_rank = get_mlp_tensor_model_parallel_rank()
                splitted_input = split_tensor_along_last_dim(
                    input_, num_partitions=self.tp_size)
                input_parallel = splitted_input[tp_rank].contiguous()
            # Matrix multiply.
            assert self.quant_method is not None
            # Only fuse bias add into GEMM for rank 0 (this ensures that
            # bias will not get added more than once in TP>1 case)
            bias_ = None if (self.tp_rank > 0
                             or self.skip_bias_add) else self.bias
            output_parallel = self.quant_method.apply(self,
                                                      input_parallel,
                                                      bias=bias_)
            output = get_mlp_tp_group().reduce_scatter(output_parallel, 0)
            # output = output[:num_tokens,:]
            # dispose_tensor(output_parallel)
        else:
            if self.input_is_parallel:
                input_parallel = input_
            else:
                tp_rank = get_tensor_model_parallel_rank()
                splitted_input = split_tensor_along_last_dim(
                    input_, num_partitions=self.tp_size)
                input_parallel = splitted_input[tp_rank].contiguous()

            # Matrix multiply.
            assert self.quant_method is not None
            # Only fuse bias add into GEMM for rank 0 (this ensures that
            # bias will not get added more than once in TP>1 case)
            bias_ = None if (self.tp_rank > 0
                             or self.skip_bias_add) else self.bias
            output_parallel = self.quant_method.apply(self,
                                                      input_parallel,
                                                      bias=bias_)
            if self.reduce_results and self.enable_sp and is_prefill:
                sp_size = get_tensor_model_parallel_world_size()
                original_len = input_.shape[0]
                reminder = original_len % sp_size
                if reminder != 0:
                    padding_len = sp_size - reminder
                    output_parallel = nn.functional.pad(output_parallel, (0, 0, 0, padding_len), mode='constant', value=0)
                output = tensor_model_parallel_reduce_scatter(output_parallel.movedim(0, -1)).movedim(-1, 0)
            elif self.reduce_results and self.tp_size > 1:
                output = tensor_model_parallel_all_reduce(output_parallel)
            else:
                output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias


class AscendMlpMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
    ):
        self.output_sizes = output_sizes
        if prefix.find("gate_up_proj") != -1:
            self.tp_size = get_mlp_tensor_model_parallel_world_size()
            self.tp_rank = get_mlp_tensor_model_parallel_rank()
            self.enable_mlp_optimze = True
        else:
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_rank = get_tensor_model_parallel_rank()
            self.enable_mlp_optimze = False
        assert all(output_size % self.tp_size == 0
                   for output_size in output_sizes)
        AscendMlpColumnParallelLinear.__init__(self,
                                               input_size=input_size,
                                               output_size=sum(output_sizes),
                                               bias=bias,
                                               gather_output=gather_output,
                                               skip_bias_add=skip_bias_add,
                                               params_dtype=params_dtype,
                                               quant_config=quant_config,
                                               prefix=prefix,
                                               return_bias=return_bias)

    def forward(
        self,
        input_,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None
        # self.global_batch_size = vllm_config.scheduler_config.max_num_seqs
        # Matrix multiply.
        assert self.quant_method is not None
        if self.enable_mlp_optimze:
            input2_ = get_mlp_tp_group().all_gather(input_, 0)
            output = self.quant_method.apply(self, input2_, bias)
        else:
            output_parallel = self.quant_method.apply(self, input_, bias)
            if self.gather_output:
                # All-gather across the partitions.
                output = tensor_model_parallel_all_gather(output_parallel)
            else:
                output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias


class CustomDeepseekV2MergedReplicatedLinear(ReplicatedLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size,
                         sum(output_sizes),
                         bias=bias,
                         quant_config=quant_config,
                         prefix=prefix)

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, loaded_shard_id: int):
        # With no support for GGUF format yet.
        assert not getattr(param, "is_gguf_weight", False)
        assert not getattr(param, "is_gguf_weight_type", False)

        assert loaded_shard_id < len(self.output_sizes)
        shard_offset = sum(self.output_sizes[:loaded_shard_id])
        shard_size = self.output_sizes[loaded_shard_id]
        shard = param.data.narrow(param.output_dim, shard_offset, shard_size)

        assert shard.size() == loaded_weight.size(), (
            f"Tried to load weights of size {loaded_weight.size()}"
            f"to a parameter shard of id {loaded_shard_id} size {shard.size()}"
        )
        shard.copy_(loaded_weight)


class CustomDeepseekV2SiluAndMul(SiluAndMul):

    def __init__(self,
                 *,
                 weight_scale: Optional[Callable[[], torch.Tensor]] = None):
        super().__init__()
        self.weight_scale = weight_scale

    def forward_oot(self, x: Union[torch.Tensor, Tuple[torch.Tensor,
                                                       torch.Tensor]]):
        if isinstance(x, tuple):
            assert self.weight_scale is not None
            # For AscendW8A8DynamicLinearMethod:
            # a dynamic scale is passed along with the quantized value.
            quantized_x, dynamic_scale = x
            return torch_npu.npu_dequant_swiglu_quant(
                x=quantized_x,
                weight_scale=self.weight_scale(),
                activation_scale=dynamic_scale,
                activate_left=True,
                quant_mode=1)
        else:
            return super().forward_oot(x)
        

class AscendDeepseekV2MLP(DeepseekV2MLP):
        
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.enable_sp = is_sp_enabled()
        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled
        enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        force_replicate = False
        self.enable_multistream_moe = \
            ascend_config.torchair_graph_config.enable_multistream_moe and \
            self.torchair_graph_enabled
        force_replicate = self.enable_multistream_moe or enable_shared_expert_dp
        if not force_replicate and not self.enable_sp:
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size, [intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_up_proj")
            self.down_proj = RowParallelLinear(intermediate_size,
                                               hidden_size,
                                               bias=False,
                                               quant_config=quant_config,
                                               reduce_results=reduce_results,
                                               prefix=f"{prefix}.down_proj")
        else:
            self.gate_up_proj = CustomDeepseekV2MergedReplicatedLinear(
                hidden_size, [intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_up_proj")
            self.down_proj = ReplicatedLinear(intermediate_size,
                                              hidden_size,
                                              bias=False,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        quant_method = self.gate_up_proj.quant_method
        if isinstance(quant_method, UnquantizedLinearMethod):
            self.act_fn = CustomDeepseekV2SiluAndMul()
        elif (isinstance(quant_method, AscendLinearMethod) and isinstance(
                quant_method.quant_method, AscendW8A8DynamicLinearMethod)):
            # TODO(sdmyzlp): Currently preserved as before:
            # 1. The only quantization supported for silu is W8A8Dynamic
            # 2. Output dtype of gate_up/down is fixed to be int32/bfloat16
            #
            # Maybe one can implement a better and more general configuration
            # scheme, e.g. by somehow passing around the tweaked `quant_config`
            self.act_fn = CustomDeepseekV2SiluAndMul(
                # Use lazy binding, for `weight_scale_fp32` is accessible
                # only after `process_weights_after_loading`.
                weight_scale=lambda: self.gate_up_proj.weight_scale_fp32)
            # To be consumed by AscendW8A8DynamicLinearMethod.apply()
            self.gate_up_proj._ascend_quant_config = {
                "output_dtype": torch.int32,
                "pertoken_scale": False,
                "return_scale": True,
            }
            self.down_proj._ascend_quant_config = {
                "output_dtype": torch.bfloat16,
                "pertoken_scale": True,
                "return_scale": False,
            }
        else:
            raise NotImplementedError(
                f"Quantization with [{type(quant_method)}] is NOT supported")

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x