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

import warnings
from typing import Optional
import torch
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
)
from peft.utils import TRANSFORMERS_MODELS_TO_XXX_TARGET_MODULES_MAPPING

from .layer import XXXLayer, Linear
from peft.tuners.lora.layer import LoraLayer, ParamWrapper
from peft.tuners.lora.model import LoraModel
from peft.utils.other import get_pattern_key


class XXXModel(LoraModel):
    prefix: str = "lora_"
    tuner_layer_cls = XXXLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_XXX_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        xxx_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        *,
        parameter_name: Optional[str] = None,
    ) -> None:
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        if xxx_config.target_parameters:
            # Right now, unfortunately, we don't support multiple adapters with target_parameters on the same model.
            other_configs_use_target_params = any(
                conf.target_parameters for key, conf in self.peft_config.items() if key != adapter_name
            )
            if other_configs_use_target_params:
                raise ValueError(
                    f"Adding a LoRA config with `target_parameters={xxx_config.target_parameters}` but there are "
                    "already other LoRA adapters on this model that use `target_parameters`. At the moment, only "
                    "one LoRA adapter per model with `target_parameters` is allowed."
                )

        assert hasattr(target, "xxx_stiff_basis"), f"Jacobian has not been initialized for {target}. Please run preprocess_xxx first."
        xxx_stiff_basis = target.xxx_stiff_basis
        del target.xxx_stiff_basis

        # Regexp matching - Find key which matches current target_name in patterns provided
        r_key = get_pattern_key(xxx_config.rank_pattern.keys(), current_key)
        alpha_key = get_pattern_key(xxx_config.alpha_pattern.keys(), current_key)
        r = xxx_config.rank_pattern.get(r_key, xxx_config.r)
        alpha = xxx_config.alpha_pattern.get(alpha_key, xxx_config.lora_alpha)

        kwargs = {
            "xxx_stiff_basis": xxx_stiff_basis,
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": xxx_config.lora_dropout,
            "fan_in_fan_out": xxx_config.fan_in_fan_out,
            "init_lora_weights": xxx_config.init_lora_weights,
            "use_rslora": xxx_config.use_rslora,
            "use_dora": xxx_config.use_dora,
            "use_alora": xxx_config.alora_invocation_tokens is not None,
            "use_qalora": xxx_config.use_qalora,
            "qalora_group_size": xxx_config.qalora_group_size,
            "ephemeral_gpu_offload": xxx_config.runtime_config.ephemeral_gpu_offload,
            "lora_bias": xxx_config.lora_bias,
            "arrow_config": xxx_config.arrow_config,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
            "parameter_name": parameter_name,
        }

        # if the target is a ParamWrapper, we nest it to allow targeting multiple nn.Parameter on the same module
        # wrap_target_param = isinstance(target, ParamWrapper) and (adapter_name in target.lora_A)
        if isinstance(target, XXXLayer):
            target.update_layer(
                adapter_name,
                xxx_stiff_basis,
                r=r,
                lora_alpha=alpha,
                lora_dropout=xxx_config.lora_dropout,
                init_lora_weights=xxx_config.init_lora_weights,
                use_rslora=xxx_config.use_rslora,
                use_dora=xxx_config.use_dora,
                lora_bias=xxx_config.lora_bias,
                arrow_config=xxx_config.arrow_config,
                inference_mode=xxx_config.inference_mode,
            )
        else:
            device_map = self.model.hf_device_map if hasattr(self.model, "hf_device_map") else None
            new_module = self._create_new_module(xxx_config, adapter_name, target, device_map=device_map, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    # def _create_and_replace(
    #     self,
    #     xxx_config,
    #     adapter_name,
    #     target,
    #     target_name,
    #     parent,
    #     current_key,
    #     *,
    #     parameter_name: Optional[str] = None,
    # ) -> None:
    #     if current_key is None:
    #         raise ValueError("Current Key shouldn't be `None`")

    #     if xxx_config.target_parameters:
    #         # Right now, unfortunately, we don't support multiple adapters with target_parameters on the same model.
    #         other_configs_use_target_params = any(
    #             conf.target_parameters for key, conf in self.peft_config.items() if key != adapter_name
    #         )
    #         if other_configs_use_target_params:
    #             raise ValueError(
    #                 f"Adding a LoRA config with `target_parameters={xxx_config.target_parameters}` but there are "
    #                 "already other LoRA adapters on this model that use `target_parameters`. At the moment, only "
    #                 "one LoRA adapter per model with `target_parameters` is allowed."
    #             )

    #     assert hasattr(target, "xxx_stiff_basis"), f"Jacobian has not been initialized for {target}. Please run preprocess_xxx first."
    #     xxx_stiff_basis = target.xxx_stiff_basis
    #     del target.xxx_stiff_basis

    #     kwargs = {
    #         "xxx_stiff_basis": xxx_stiff_basis,
    #         "fan_in_fan_out": xxx_config.fan_in_fan_out,
    #     }


    #     if isinstance(target, XXXLayer):
    #         target.update_layer(
    #             adapter_name,
    #             xxx_stiff_basis=xxx_stiff_basis,
    #             inference_mode=xxx_config.inference_mode,
    #         )
    #     else:
    #         device_map = self.model.hf_device_map if hasattr(self.model, "hf_device_map") else None
    #         new_module = self._create_new_module(xxx_config, adapter_name, target, device_map=device_map, **kwargs)
    #         if adapter_name not in self.active_adapters:
    #             # adding an additional adapter: it is not automatically trainable
    #             new_module.requires_grad_(False)
    #         self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(xxx_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = xxx_config.fan_in_fan_out = False
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`."
            )

        new_module = Linear(target, adapter_name, **kwargs)

        return new_module

