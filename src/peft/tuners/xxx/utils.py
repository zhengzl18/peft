# Copyright 2024-present the HuggingFace Inc. team.
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

# Reference code: https://github.com/iboing/CorDA/blob/main/cordalib/decomposition.py
# Reference paper: https://huggingface.co/papers/2406.05223

import os
from collections.abc import Iterable
from typing import Dict, List, Optional
import warnings

from peft.peft_model import PeftModel, PeftModelForCausalLM
from peft.tuners.xxx.config import XXXConfig, XXXPreprocessConfig
from peft.tuners.xxx.layer import XXXLayer
import torch
import torch.nn as nn
from torch.nn import Linear
from tqdm import tqdm

from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.model import LoraModel
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.utils.other import get_pattern_key
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise, QuantState


IMPORTANCE_SCORE_MODE = {
    "abs": lambda x: torch.abs(x),
    "square": lambda x: x ** 2,
}
EPS = 1e-20
    

def target_modules(model: nn.Module, config: XXXConfig) -> Iterable[nn.Module]:
    """
    Iterate over CorDA target name and modules of a model. A module is a target if its name is in
    `config.target_modules` and is `nn.Linear`.
    """
    if isinstance(model, PeftModel):
        model = model.get_base_model()
    for name, module in model.named_modules():
        # todo: change LoraModel to XXXModel
        if LoraModel._check_target_module_exists(config, name) and isinstance(module, (Linear, LoraLinear)):
            yield name, module
        # elif LoraModel._check_target_module_exists(config, name) and isinstance(module, LoraLinear):
        #     yield name, module

# def target_params(model: nn.Module, config: XXXConfig) -> Iterable[nn.Parameter]:
#     # TODO: maybe support bias
#     for name, module in target_modules(model, config):
#         yield f"{name}.weight", module.weight

def get_model_device(model: nn.Module) -> str:
    if hasattr(model, "module"):  # Handle DeepSpeed/DataParallel
        model = model.module
    return next(iter(model.parameters())).device.type

def preprocess_xxx(
    model: nn.Module,
    xxx_config: XXXConfig,
    local_rank: int = 0,
):
    """
    Build necessary XXX fields for a model.

    For each `M * N` linear layer, a `M * M` jacobian matrix will be built temporarily during the preprocessing
    process, consuming roughly another `2 * MODEL_SIZE` memory for typical LLMs if model weight is FP16 and jacobian
    is FP32. If that's too much, consider specifying `use_float16_for_jacobian` in `preprocess_config`.

    Args:
        model (`nn.Module`):
            Model to preprocess.
        xxx_config (`XXXConfig`):
            XXX configuration of the model. `preprocess_config` should be set.
    """
    stiff_basis_path = xxx_config.preprocess_config.stiff_basis_path
    assert stiff_basis_path is not None, "stiff_basis_path in preprocess_config should be specified for XXX preprocessing."
    quantize_stiff_basis = xxx_config.preprocess_config.quantize_stiff_basis
    if quantize_stiff_basis:
        torch.serialization.add_safe_globals([QuantState])
    
    for module_name, module in target_modules(model, xxx_config):
        file_name = module_name.replace('.', '-')
        
        if not os.path.exists(f"{stiff_basis_path}/{file_name}.pt"):
            raise FileNotFoundError(f"Stiff basis file for {module_name} not found in {stiff_basis_path}, run preprocess.py to build stiff basis first.")

        print(f"Loading stiff basis from {stiff_basis_path}/{file_name}.pt on cuda:{local_rank} ...")
        stiff_basis = torch.load(f"{stiff_basis_path}/{file_name}.pt", map_location=f"cuda:{local_rank}")
        
        if quantize_stiff_basis:
            assert "quant_state_a" in stiff_basis and "quant_state_b" in stiff_basis, "quantize_stiff_basis is True but quant_state not found in the loaded stiff basis."
        else:
            assert "quant_state_a" not in stiff_basis and "quant_state_b" not in stiff_basis, "quantize_stiff_basis is False but quant_state found in the loaded stiff basis."
            stiff_basis['stiff_basis_a'] = stiff_basis['stiff_basis_a'].to(model.dtype)
            stiff_basis['stiff_basis_b'] = stiff_basis['stiff_basis_b'].to(model.dtype)

        module.xxx_stiff_basis = stiff_basis

def calculate_jacobian(
    model: nn.Module,
    config: XXXConfig,
    data_loader: List[Dict[str, torch.Tensor]],
):
    save_path = config.preprocess_config.jacobian_path
    assert isinstance(save_path, str), f"jacobian_path in preprocess_config is expected to be a string for calculating and saving jacobians, got {type(config.jacobian_path)}."

    model.train()
    os.makedirs(save_path, exist_ok=True)
    for name, module in target_modules(model, config):
        assert '-' not in name
        file_name = name.replace('.', '-')

        if os.path.exists(f"{save_path}/{file_name}.pt"):
            print(f"Jacobian file for {save_path}/{name} already exists, skipping.")
            continue
        
        print(f"Calculating jacobian for {save_path}/{name} ...")
        for param in model.parameters():
            param.requires_grad = False
        grads_a = []
        grads_b = []
        module.lora_A.xxx.weight.requires_grad = True  # Only compute gradient for the target parameter
        module.lora_B.xxx.weight.requires_grad = True
        
        for data in tqdm(data_loader):
            data = {k: v.to(model.device) for k, v in data.items()}
            model.zero_grad()
            outputs = model(**data)
            outputs.loss.backward()
            assert module.lora_A.xxx.weight.grad is not None
            assert module.lora_B.xxx.weight.grad is not None
            grads_a.append(module.lora_A.xxx.weight.grad.reshape(-1).clone().cpu())
            grads_b.append(module.lora_B.xxx.weight.grad.reshape(-1).clone().cpu())
        model.zero_grad()
        
        # stack grads into jacobian matrix
        jac_a = torch.stack(grads_a, dim=0)
        jac_b = torch.stack(grads_b, dim=0)
        torch.save(dict(lora_a=jac_a, lora_b=jac_b), f"{save_path}/{file_name}.pt")

@torch.no_grad()
def calculate_stiff_basis(
    model: nn.Module,
    config: XXXConfig,
):
    jacobian_paths = config.preprocess_config.jacobian_path
    assert isinstance(jacobian_paths, List), \
        f"jacobian_path in preprocess_config is expected to be a List[str] for calculating and saving stiff basis, got {type(jacobian_paths)}."
    save_path = config.preprocess_config.stiff_basis_path
    assert save_path is not None, \
        "stiff_basis_path in preprocess_config should be specified for calculating and saving stiff basis."
    r_stiff_basis = config.preprocess_config.r_stiff_basis
    adaptive_r_stiff_basis = config.preprocess_config.adaptive_r_stiff_basis
    if adaptive_r_stiff_basis:
        cumulative_energy_threshold = config.preprocess_config.cumulative_energy_threshold
        min_r_stiff_basis = config.preprocess_config.min_r_stiff_basis

    assert r_stiff_basis is not None, \
        "r_stiff_basis in preprocess_config should be specified for calculating and saving stiff basis."
    quantize_stiff_basis = config.preprocess_config.quantize_stiff_basis

    os.makedirs(save_path, exist_ok=True)
    for name, module in target_modules(model, config):
        assert '-' not in name
        file_name = name.replace('.', '-')

        if os.path.exists(f"{save_path}/{file_name}.pt"):
            print(f"Stiff basis file for {save_path}/{name} already exists, skipping.")
            continue

        print(f"Calculating stiff basis for {name} ...")

        full_jacobian_a = []
        full_jacobian_b = []
        for jacobian_path in jacobian_paths:
            if not os.path.exists(f"{jacobian_path}/{file_name}.pt"):
                raise FileNotFoundError(f"Jacobian file for {name} not found in {jacobian_path}, cannot calculate stiff basis.")

            jac = torch.load(f"{jacobian_path}/{file_name}.pt", map_location=get_model_device(model))
            full_jacobian_a.append(jac['lora_a'])
            full_jacobian_b.append(jac['lora_b'])
        full_jacobian_a = torch.cat(full_jacobian_a, dim=0).to(torch.float32)
        full_jacobian_b = torch.cat(full_jacobian_b, dim=0).to(torch.float32)
            
        if full_jacobian_a.shape[0] < r_stiff_basis:
            warnings.warn(
                f"r_stiff_basis {r_stiff_basis} is larger than the total number of knowledge dataset samples {full_jacobian_a.shape[0]} for parameter {name}."
            )
        
        U, S, _ = torch.linalg.svd(full_jacobian_a.T, full_matrices=False)
        if adaptive_r_stiff_basis:
            max_r_stiff_basis = r_stiff_basis
            cumulative_energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
            r_stiff_basis = (torch.searchsorted(cumulative_energy, cumulative_energy_threshold) + 1).clip(min_r_stiff_basis, max_r_stiff_basis)
            print(f"Adjusted r_stiff_basis to {r_stiff_basis} for parameter {name} based on energy threshold {cumulative_energy_threshold}.")
        stiff_basis_a = U[:, :r_stiff_basis]
        
        U, S, _ = torch.linalg.svd(full_jacobian_b.T, full_matrices=False)
        if adaptive_r_stiff_basis:
            max_r_stiff_basis = r_stiff_basis
            cumulative_energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
            r_stiff_basis = (torch.searchsorted(cumulative_energy, cumulative_energy_threshold) + 1).clip(min_r_stiff_basis, max_r_stiff_basis)
            print(f"Adjusted r_stiff_basis to {r_stiff_basis} for parameter {name} based on energy threshold {cumulative_energy_threshold}.")
        stiff_basis_b = U[:, :r_stiff_basis]
        
        if quantize_stiff_basis:
            quantized_stiff_basis_a, quant_state_a = quantize_blockwise(stiff_basis_a)
            quantized_stiff_basis_b, quant_state_b = quantize_blockwise(stiff_basis_b)
            torch.save(dict(
                init_lora_a=module.lora_A.xxx.weight,
                init_lora_b=module.lora_B.xxx.weight,
                stiff_basis_a=quantized_stiff_basis_a, 
                quant_state_a=quant_state_a,
                stiff_basis_b=quantized_stiff_basis_b, 
                quant_state_b=quant_state_b
            ), f"{save_path}/{file_name}.pt")
        else:
            torch.save(dict(stiff_basis_a=stiff_basis_a, stiff_basis_b=stiff_basis_b), f"{save_path}/{file_name}.pt")


class ProjectionCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        for _, module in model.named_modules():
            if isinstance(module, XXXLayer):
                lora_a = module.lora_A[model.active_adapter].weight
                lora_b = module.lora_B[model.active_adapter].weight
                init_lora_a = module.xxx_init_lora_a[model.active_adapter]
                init_lora_b = module.xxx_init_lora_b[model.active_adapter]
                delta_a = lora_a - init_lora_a
                delta_b = lora_b - init_lora_b
                stiff_basis_a = module.xxx_stiff_basis_a[model.active_adapter]()
                stiff_basis_b = module.xxx_stiff_basis_b[model.active_adapter]()
                quant_state_a = module.xxx_stiff_basis_a_quant_state[model.active_adapter]
                quant_state_b = module.xxx_stiff_basis_b_quant_state[model.active_adapter]
                stiff_basis_a = dequantize_blockwise(stiff_basis_a, quant_state_a)
                stiff_basis_b = dequantize_blockwise(stiff_basis_b, quant_state_b)
                # print("\nnorm before projection:", delta_weight.data.norm().item())
                # print(torch.isfinite(delta_weight).all())
                projected_delta_a = delta_a.data.reshape(-1) - delta_a.data.reshape(-1) @ stiff_basis_a @ stiff_basis_a.T
                projected_delta_b = delta_b.data.reshape(-1) - delta_b.data.reshape(-1) @ stiff_basis_b @ stiff_basis_b.T
                lora_a.data = projected_delta_a.reshape(lora_a.shape) + init_lora_a.data
                lora_b.data = projected_delta_b.reshape(lora_b.shape) + init_lora_b.data
                # print("norm after projection:", delta_weight.data.norm().item())
        return control