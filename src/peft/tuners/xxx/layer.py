# Copyright 2023-present the HuggingFace Inc. team.
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
from __future__ import annotations

from typing import Any, Dict, Optional, Union
import warnings

import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer, _get_in_out_features, check_adapters_to_merge
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.config import ArrowConfig
from peft.utils.other import transpose


class BufferWrapper(nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.register_buffer("buffer", tensor.contiguous())
        # self.original_shape = base_layer.weight.shape
        
    def forward(self, *args, **kwargs):
        return self.buffer


class XXXLayer(LoraLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names: tuple[str, ...] = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = ("r", "lora_alpha", "scaling", "lora_dropout", "xxx_stiff_basis_a", "xxx_stiff_basis_a_quant_state", "xxx_stiff_basis_b", "xxx_stiff_basis_b_quant_state")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__(base_layer=base_layer, **kwargs)
        self.xxx_stiff_basis_a = nn.ModuleDict({})  # Currently there is no `BufferDict` in torch.nn, this is a workaround to ensure proper device handling
        self.xxx_stiff_basis_a_quant_state = {}
        self.xxx_init_lora_a = {}
        # self.xxx_init_lora_a = nn.ModuleDict({})
        self.xxx_stiff_basis_b = nn.ModuleDict({})
        self.xxx_stiff_basis_b_quant_state = {}
        self.xxx_init_lora_b = {}
        # self.xxx_init_lora_b = nn.ModuleDict({})


    def update_layer(
        self,
        adapter_name: str,
        xxx_stiff_basis: Dict[str, torch.Tensor],
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora: bool = False,
        use_dora: bool = False,
        use_alora: bool = False,
        use_qalora: bool = False,
        lora_bias: bool = False,
        arrow_config: ArrowConfig = None,
        qalora_group_size: int = 32,
        inference_mode: bool = False,
        **kwargs,
    ):
        # # collect the kwargs
        # kwargs = locals().copy()
        # del kwargs["self"]

        self.xxx_stiff_basis_a[adapter_name] = BufferWrapper(xxx_stiff_basis["stiff_basis_a"])
        self.xxx_stiff_basis_b[adapter_name] = BufferWrapper(xxx_stiff_basis["stiff_basis_b"])
        # TODO: sanity check for shape of xxx_stiff_basis
        self.xxx_stiff_basis_a_quant_state[adapter_name] = xxx_stiff_basis["quant_state_a"]
        self.xxx_stiff_basis_b_quant_state[adapter_name] = xxx_stiff_basis["quant_state_b"]
        self.xxx_init_lora_a[adapter_name] = xxx_stiff_basis["init_lora_a"]
        self.xxx_init_lora_b[adapter_name] = xxx_stiff_basis["init_lora_b"]
        # self.xxx_init_lora_a[adapter_name] = BufferWrapper(xxx_stiff_basis["init_lora_a"])
        # self.xxx_init_lora_b[adapter_name] = BufferWrapper(xxx_stiff_basis["init_lora_b"])

        super().update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            use_alora=use_alora,
            lora_bias=lora_bias,
            arrow_config=arrow_config,
            qalora_group_size=qalora_group_size,
            inference_mode=inference_mode,
        )

        self.lora_A[adapter_name].weight.data = xxx_stiff_basis["init_lora_a"].data
        self.lora_B[adapter_name].weight.data = xxx_stiff_basis["init_lora_b"].data


class Linear(LoraLinear, XXXLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        xxx_stiff_basis: Dict[str, torch.Tensor],
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        use_alora: bool = False,
        arrow_config: ArrowConfig = None,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)
        XXXLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            xxx_stiff_basis,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            use_alora=use_alora,
            lora_bias=lora_bias,
            arrow_config=arrow_config,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    # def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    #     """
    #     Merge the active adapter weights into the base weights

    #     Args:
    #         safe_merge (`bool`, *optional*):
    #             If True, the merge operation will be performed in a copy of the original weights and check for NaNs
    #             before merging the weights. This is useful if you want to check if the merge operation will produce
    #             NaNs. Defaults to `False`.
    #         adapter_names (`list[str]`, *optional*):
    #             The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
    #             to `None`.
    #     """
    #     adapter_names = check_adapters_to_merge(self, adapter_names)
    #     if not adapter_names:
    #         # no adapter to merge
    #         return

    #     for active_adapter in adapter_names:
    #         if active_adapter in self.xxx_delta_weight.keys():
    #             base_layer = self.get_base_layer()
    #             if safe_merge:
    #                 # Note that safe_merge will be slower than the normal merge
    #                 # because of the copy operation.
    #                 orig_weight = base_layer.weight.data.clone()
    #                 orig_dtype = orig_weight.dtype
    #                 delta_weight = self.get_delta_weight(active_adapter)
    #                 orig_weight += delta_weight.to(orig_dtype)

    #                 if not torch.isfinite(orig_weight).all():
    #                     raise ValueError(
    #                         f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
    #                     )

    #                 base_layer.weight.data = orig_weight

    #             else:
    #                 delta_weight = self.get_delta_weight(active_adapter)
    #                 base_layer.weight.data += delta_weight

    #             self.merged_adapters.append(active_adapter)

    # def unmerge(self) -> None:
    #     """
    #     This method unmerges all merged adapter layers from the base weights.
    #     """

    #     if not self.merged:
    #         warnings.warn("Already unmerged. Nothing to do.")
    #         return
    #     while len(self.merged_adapters) > 0:
    #         active_adapter = self.merged_adapters.pop()
    #         if active_adapter in self.xxx_delta_weight.keys():
    #             weight = self.get_base_layer().weight
    #             orig_dtype = weight.dtype
    #             delta_weight = self.get_delta_weight(active_adapter)
    #             weight.data -= delta_weight.to(orig_dtype)

    # def get_delta_weight(self, adapter) -> torch.Tensor:
    #     """
    #     Compute the delta weight for the given adapter.

    #     Args:
    #         adapter (str):
    #             The name of the adapter for which the delta weight should be computed.
    #     """
    #     device = self.xxx_delta_weight[adapter].device
    #     dtype = self.xxx_delta_weight[adapter].dtype

    #     # In case users wants to merge the adapter weights that are in
    #     # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    #     # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
    #     cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

    #     delta_weight = self.xxx_delta_weight[adapter]
    #     mask_indice = self.xxx_stiff_basis_w_mask_indice[adapter]()

    #     if cast_to_fp32:
    #         delta_weight = delta_weight.float()
        
    #     output_tensor = torch.sparse_coo_tensor(
    #         indices=mask_indice,
    #         values=delta_weight,
    #         size=(self.out_features, self.in_features),
    #         dtype=delta_weight.dtype,
    #         device=delta_weight.device
    #     )

    #     if cast_to_fp32:
    #         output_tensor = output_tensor.to(dtype=dtype)

    #         # cast back the weights
    #         self.xxx_delta_weight[adapter].data = delta_weight.to(dtype)
    #         # TODO: check whether this is necessary, whether we should also cast back stiff_basisobian

    #     return output_tensor

    # def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    #     self._check_forward_args(x, *args, **kwargs)

    #     if self.disable_adapters:
    #         if self.merged:
    #             self.unmerge()
    #         result = self.base_layer(x, *args, **kwargs)
    #     elif self.merged:
    #         result = self.base_layer(x, *args, **kwargs)
    #     else:
    #         result = self.base_layer(x, *args, **kwargs)
    #         torch_result_dtype = result.dtype

    #         xxx_delta_weight_keys = self.xxx_delta_weight.keys()
    #         for active_adapter in self.active_adapters:
    #             if active_adapter not in xxx_delta_weight_keys:
    #                 continue

    #             delta_weight = self.xxx_delta_weight[active_adapter]
    #             mask_indice = self.xxx_stiff_basis_w_mask_indice[active_adapter]()
    #             x = self._cast_input_dtype(x, delta_weight.dtype)
    #             delta_weight = torch.sparse_coo_tensor(
    #                 indices=mask_indice,
    #                 values=delta_weight,
    #                 size=(self.out_features, self.in_features),
    #                 dtype=delta_weight.dtype,
    #                 device=delta_weight.device
    #             )
    #             result = result + x @ delta_weight.T

    #         result = result.to(torch_result_dtype)

    #     return result

    # def __repr__(self) -> str:
    #     rep = super().__repr__()
    #     return "xxx." + rep
