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


class XXXModel(BaseTuner):
    prefix: str = "xxx_"
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

        assert hasattr(target, "xxx_jacobian"), f"Jacobian has not been initialized for {target}. Please run preprocess_xxx first."
        xxx_jacobian = target.xxx_jacobian
        del target.xxx_jacobian

        kwargs = {
            "xxx_jacobian": xxx_jacobian,
            "fan_in_fan_out": xxx_config.fan_in_fan_out,
        }


        if isinstance(target, XXXLayer):
            target.update_layer(
                adapter_name,
                xxx_jacobian=xxx_jacobian,
                inference_mode=xxx_config.inference_mode,
            )
        else:
            device_map = self.model.hf_device_map if hasattr(self.model, "hf_device_map") else None
            new_module = self._create_new_module(xxx_config, adapter_name, target, device_map=device_map, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

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


    def _prepare_adapter_config(self, peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] in self.target_module_mapping:
                peft_config.target_modules = set(self.target_module_mapping[model_config["model_type"]])
            elif not peft_config.target_parameters:
                raise ValueError("Please specify `target_modules` or `target_parameters`in `peft_config`")
        return peft_config
