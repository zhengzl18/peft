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
from peft.tuners.lora.layer import LoraLayer, VARIANT_KWARG_KEYS
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.config import ArrowConfig
from peft.utils.other import transpose


class BufferWrapper(nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.register_buffer("buffer", tensor)
        # self.original_shape = base_layer.weight.shape
        
    def forward(self, *args, **kwargs):
        return self.buffer


class XXXLayer(LoraLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names: tuple[str, ...] = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # ============== Seperate proj =================
    # All names of other parameters that may contain adapter-related parameters
    # other_param_names: tuple[str, ...] = ("r", "lora_alpha", "scaling", "lora_dropout", "xxx_stiff_basis_a", "xxx_stiff_basis_a_quant_state", "xxx_stiff_basis_b", "xxx_stiff_basis_b_quant_state",)
    # ============== Combined proj =================
    # other_param_names: tuple[str, ...] = ("r", "lora_alpha", "scaling", "lora_dropout", "xxx_stiff_basis", "xxx_stiff_basis_quant_state",)
    # ============== Zero out delta weights =================
    # other_param_names: tuple[str, ...] = ("r", "lora_alpha", "scaling", "lora_dropout", "xxx_stiff_basis_a", "xxx_stiff_basis_a_quant_state", "xxx_stiff_basis_b", "xxx_stiff_basis_b_quant_state", "xxx_delta_lora_a", "xxx_delta_lora_b")
    # ============== W proj =================
    other_param_names: tuple[str, ...] = ("r", "lora_alpha", "scaling", "lora_dropout",)

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__(base_layer=base_layer, **kwargs)
        # ============== Seperate proj =================
        # self.xxx_stiff_basis_a = nn.ModuleDict({})  # Currently there is no `BufferDict` in torch.nn, this is a workaround to ensure proper device handling
        # self.xxx_stiff_basis_b = nn.ModuleDict({})
        # self.xxx_stiff_basis_a_quant_state = {}
        # self.xxx_stiff_basis_b_quant_state = {}
        # self.xxx_init_lora_a = {}
        # self.xxx_init_lora_b = {}
        # ============== Combined proj =================
        # self.xxx_stiff_basis = nn.ModuleDict({})
        # self.xxx_stiff_basis_quant_state = {}
        # self.xxx_init_lora_a = {}
        # self.xxx_init_lora_b = {}
        # ============== W proj =================
        # self.xxx_stiff_basis_w_a = nn.ModuleDict({})  # Currently there is no `BufferDict` in torch.nn, this is a workaround to ensure proper device handling
        # self.xxx_stiff_basis_w_b = nn.ModuleDict({})
        # self.xxx_stiff_basis_k_inv = nn.ModuleDict({})
        # self.xxx_prev_lora_a = nn.ModuleDict({})
        # self.xxx_prev_lora_b = nn.ModuleDict({})


        # ============== Zero out delta weights =================
        # self.xxx_delta_lora_a = nn.ModuleDict({})
        # self.xxx_delta_lora_b = nn.ModuleDict({})


    def update_layer(
        self,
        adapter_name: str,
        xxx_stiff_basis: Dict[str, torch.Tensor],
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        lora_bias: bool = False,
        inference_mode: bool = False,
        **kwargs,
    ):
        # ============== Seperate proj =================
        # self.xxx_stiff_basis_a[adapter_name] = BufferWrapper(xxx_stiff_basis["stiff_basis_a"])
        # self.xxx_stiff_basis_b[adapter_name] = BufferWrapper(xxx_stiff_basis["stiff_basis_b"])
        # self.xxx_stiff_basis_a_quant_state[adapter_name] = xxx_stiff_basis["quant_state_a"]
        # self.xxx_stiff_basis_b_quant_state[adapter_name] = xxx_stiff_basis["quant_state_b"]
        # ============== Combined proj =================
        # self.xxx_stiff_basis[adapter_name] = BufferWrapper(xxx_stiff_basis["stiff_basis"])
        # self.xxx_stiff_basis_quant_state[adapter_name] = xxx_stiff_basis["quant_state"]
        # ============== W proj =================
        self.xxx_stiff_basis_w_a = xxx_stiff_basis["jac_w_a"]
        self.xxx_stiff_basis_w_b = xxx_stiff_basis["jac_w_b"]
        self.xxx_stiff_basis_k_inv = xxx_stiff_basis["K_inv"]
        # self.xxx_stiff_basis_w_a[adapter_name] = BufferWrapper(xxx_stiff_basis["jac_w_a"])
        # self.xxx_stiff_basis_w_b[adapter_name] = BufferWrapper(xxx_stiff_basis["jac_w_b"])
        # self.xxx_stiff_basis_k_inv[adapter_name] = BufferWrapper(xxx_stiff_basis["K_inv"])
        


        # TODO: sanity check for shape of xxx_stiff_basis
        # ============== ortho/pissa init =================
        # self.xxx_init_lora_a[adapter_name] = xxx_stiff_basis["init_lora_a"]
        # self.xxx_init_lora_b[adapter_name] = xxx_stiff_basis["init_lora_b"]
        # ============== svd init zero A=================
        # self.xxx_init_lora_a[adapter_name] = torch.zeros_like(xxx_stiff_basis["init_lora_a"])
        # self.xxx_init_lora_b[adapter_name] = xxx_stiff_basis["init_lora_b"]
        # ============== svd init zero B=================
        # self.xxx_init_lora_a[adapter_name] = xxx_stiff_basis["init_lora_a"]
        # self.xxx_init_lora_b[adapter_name] = torch.zeros_like(xxx_stiff_basis["init_lora_b"])
        # ============== Zero out delta weights =================
        # self.xxx_delta_lora_a[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        # self.xxx_delta_lora_b[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)

        super().update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=False,
            lora_bias=lora_bias,
            inference_mode=inference_mode,
        )
        assert self.lora_variant == {}, "XXXLayer does not support LoRA variants."

        # ============== ortho init =================
        # self.lora_A[adapter_name].weight.data = self.lora_A[adapter_name].weight.data.to(torch.float32)
        # self.lora_B[adapter_name].weight.data = self.lora_B[adapter_name].weight.data.to(torch.float32)
        # self.lora_A[adapter_name].weight.data.copy_(xxx_stiff_basis["init_lora_a"].data)
        # self.lora_B[adapter_name].weight.data.copy_(xxx_stiff_basis["init_lora_b"].data)
        # ============== pissa init =================
        self.lora_A[adapter_name].weight.data = self.lora_A[adapter_name].weight.data.to(torch.float32)
        self.lora_B[adapter_name].weight.data = self.lora_B[adapter_name].weight.data.to(torch.float32)
        self.lora_A[adapter_name].weight.data.copy_(xxx_stiff_basis["init_lora_a"].data)
        self.lora_B[adapter_name].weight.data.copy_(xxx_stiff_basis["init_lora_b"].data)
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        weight = transpose(weight.to(torch.float32), self.fan_in_fan_out)
        weight = weight.data - self.scaling[adapter_name] * self.lora_B[adapter_name].weight.data @ self.lora_A[adapter_name].weight.data
        weight = transpose(weight.to(dtype), self.fan_in_fan_out)
        self.get_base_layer().weight.data.copy_(weight)
        # ============== svd init zero A=================
        # self.lora_A[adapter_name].weight.data = self.lora_A[adapter_name].weight.data.to(torch.float32)
        # self.lora_B[adapter_name].weight.data = self.lora_B[adapter_name].weight.data.to(torch.float32)
        # self.lora_A[adapter_name].weight.data.zero_()
        # self.lora_B[adapter_name].weight.data.copy_(xxx_stiff_basis["init_lora_b"].data)
        # ============== svd init zero B=================
        # self.lora_A[adapter_name].weight.data = self.lora_A[adapter_name].weight.data.to(torch.float32)
        # self.lora_B[adapter_name].weight.data = self.lora_B[adapter_name].weight.data.to(torch.float32)
        # self.lora_A[adapter_name].weight.data.copy_(xxx_stiff_basis["init_lora_a"].data)
        # self.lora_B[adapter_name].weight.data.zero_()
        # ============== W proj =================
        device = self.xxx_stiff_basis_w_a.device
        self.xxx_prev_lora_a = self.lora_A[adapter_name].weight.data.clone().float().to(device)
        self.xxx_prev_lora_b = self.lora_B[adapter_name].weight.data.clone().float().to(device)

        # ============== Zero out delta weights =================
        # self.xxx_delta_lora_a[adapter_name].weight.data.zero_()
        # self.xxx_delta_lora_b[adapter_name].weight.data.zero_()
        # self.xxx_delta_lora_a[adapter_name].weight.requires_grad = False
        # self.xxx_delta_lora_b[adapter_name].weight.requires_grad = False


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
            lora_bias=lora_bias,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    # ============== Zero out delta weights =================
    # def forward(self, x, *args, **kwargs):
    #     self._check_forward_args(x, *args, **kwargs)
    #     adapter_names = kwargs.pop("adapter_names", None)
    #     variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}  # don't pass these to base_layer

    #     if self.disable_adapters:
    #         if self.merged:
    #             self.unmerge()
    #         result = self.base_layer(x, *args, **kwargs)
    #     elif adapter_names is not None:
    #         result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **variant_kwargs, **kwargs)
    #     elif self.merged:
    #         result = self.base_layer(x, *args, **kwargs)
    #     else:
    #         result = self.base_layer(x, *args, **kwargs)
    #         torch_result_dtype = result.dtype

    #         lora_A_keys = self.lora_A.keys()
    #         for active_adapter in self.active_adapters:
    #             if active_adapter not in lora_A_keys:
    #                 continue

    #             lora_A = self.lora_A[active_adapter]
    #             lora_B = self.lora_B[active_adapter]
    #             assert (self.xxx_delta_lora_a[active_adapter].weight.data == lora_A.weight.data - self.xxx_init_lora_a[active_adapter].data).all(), "Delta weights are not consistent with lora weights and initial lora weights."
    #             assert (self.xxx_delta_lora_b[active_adapter].weight.data == lora_B.weight.data - self.xxx_init_lora_b[active_adapter].data).all(), "Delta weights are not consistent with lora weights and initial lora weights."
    #             delta_lora_a = self.xxx_delta_lora_a[active_adapter]
    #             delta_lora_b = self.xxx_delta_lora_b[active_adapter]
    #             dropout = self.lora_dropout[active_adapter]
    #             scaling = self.scaling[active_adapter]
    #             x = self._cast_input_dtype(x, lora_A.weight.dtype)       
    #             dropouted_x = dropout(x)   
    #             result = result + lora_B(lora_A(dropouted_x)) * scaling - delta_lora_b(delta_lora_a(dropouted_x)) * scaling

    #         result = result.to(torch_result_dtype)

    #     return result

    # def get_delta_weight(self, adapter) -> torch.Tensor:
    #     """
    #     Compute the delta weight for the given adapter.

    #     Args:
    #         adapter (str):
    #             The name of the adapter for which the delta weight should be computed.
    #     """
    #     device = self.lora_B[adapter].weight.device
    #     dtype = self.lora_B[adapter].weight.dtype

    #     # In case users wants to merge the adapter weights that are in
    #     # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    #     # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
    #     cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

    #     weight_A = self.lora_A[adapter].weight
    #     weight_B = self.lora_B[adapter].weight
    #     self.xxx_delta_lora_a[adapter].weight.data = weight_A.data - self.xxx_init_lora_a[adapter].data
    #     self.xxx_delta_lora_b[adapter].weight.data = weight_B.data - self.xxx_init_lora_b[adapter].data
    #     delta_weight_a = self.xxx_delta_lora_a[adapter].weight
    #     delta_weight_b = self.xxx_delta_lora_b[adapter].weight


    #     if cast_to_fp32:
    #         weight_A = weight_A.float()
    #         weight_B = weight_B.float()
    #         delta_weight_a = delta_weight_a.float()
    #         delta_weight_b = delta_weight_b.float()

    #     output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter] - transpose(delta_weight_b @ delta_weight_a, self.fan_in_fan_out) * self.scaling[adapter]

    #     if cast_to_fp32:
    #         output_tensor = output_tensor.to(dtype=dtype)

    #         # cast back the weights
    #         self.lora_A[adapter].weight.data = weight_A.to(dtype)
    #         self.lora_B[adapter].weight.data = weight_B.to(dtype)
    #         self.xxx_delta_lora_a[adapter].weight.data = delta_weight_a.to(dtype)
    #         self.xxx_delta_lora_b[adapter].weight.data = delta_weight_b.to(dtype)

    #     return output_tensor
    # ============== Zero out delta weights =================


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
