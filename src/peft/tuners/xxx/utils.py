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

import math
import os
from collections.abc import Iterable
from einops import rearrange
import time
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
        
        # if quantize_stiff_basis:
        #     assert "quant_state_a" in stiff_basis and "quant_state_b" in stiff_basis, "quantize_stiff_basis is True but quant_state not found in the loaded stiff basis."
        # else:
        #     assert "quant_state_a" not in stiff_basis and "quant_state_b" not in stiff_basis, "quantize_stiff_basis is False but quant_state found in the loaded stiff basis."
        #     stiff_basis['stiff_basis_a'] = stiff_basis['stiff_basis_a'].to(model.dtype)
        #     stiff_basis['stiff_basis_b'] = stiff_basis['stiff_basis_b'].to(model.dtype)

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
        torch.save(
            dict(
                init_lora_a=module.lora_A.xxx.weight.data,
                init_lora_b=module.lora_B.xxx.weight.data,
                lora_a=jac_a, 
                lora_b=jac_b
            ), 
            f"{save_path}/{file_name}.pt"
        )

def calculate_jacobian_w(
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
        module.weight.requires_grad = True

        grads_w_b = []
        grads_w_a = []
        r = config.preprocess_config.r_jac_approx
        norms = []
        maxs = []
        ratios = []
        for data in tqdm(data_loader):
            data = {k: v.to(model.device) for k, v in data.items()}
            model.zero_grad()
            outputs = model(**data)
            outputs.loss.backward()
            assert module.weight.grad is not None

            U, S, V = torch.svd_lowrank(module.weight.grad, q=r)
            B = (U @ torch.diag(torch.sqrt(S)))
            A = (torch.diag(torch.sqrt(S)) @ V.T)
            # B = (U @ torch.diag(torch.sqrt(S))).to(torch.float16)
            # A = (torch.diag(torch.sqrt(S)) @ V.T).to(torch.float16)
            norms.append((B @ A - module.weight.grad).norm().item())
            maxs.append((B @ A - module.weight.grad).abs().max().item())
            ratios.append(S[0].item() / S[-1].item())
            grads_w_b.append(B.cpu())
            grads_w_a.append(A.cpu())
        model.zero_grad()
        print(f"Avg norm: {sum(norms) / len(norms)}")
        print(f"Avg max: {sum(maxs) / len(maxs)}")
        print(f"Avg ratio: {sum(ratios) / len(ratios)}")
        
        # stack grads into jacobian matrix
        jac_w_a = torch.stack(grads_w_a, dim=0)  # shape: (num_samples, out_features, in_features)
        jac_w_b = torch.stack(grads_w_b, dim=0)  # shape: (num_samples, out_features, in_features)

        BB = torch.einsum('pkr, qks -> pqrs', B, B)
        AA = torch.einsum('prm, qsm -> pqrs', A, A)
        K = torch.einsum('pqrs, pqrs -> pq', BB, AA)
        K_double = K.to(torch.float64)
        K_inv_double = torch.linalg.pinv(K_double)

        torch.save(
            dict(
                jac_w_a=jac_w_a, 
                jac_w_b=jac_w_b,
                K_inv=K_inv_double.to(torch.float32)
            ), 
            f"{save_path}/{file_name}.pt"
        )

def calculate_jacobian_svd_init(
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
        grads_w = []
        module.weight.requires_grad = True  

        for data in tqdm(data_loader):
            data = {k: v.to(model.device) for k, v in data.items()}
            model.zero_grad()
            outputs = model(**data)
            outputs.loss.backward()
            assert module.weight.grad is not None
            grads_w.append(module.weight.grad.clone().cpu())
        model.zero_grad()
        
        # stack grads into jacobian matrix
        jac_w = torch.stack(grads_w, dim=0)  # shape: (num_samples, out_features, in_features)
        del grads_w
        mean_jac_w = jac_w.mean(dim=0)  # shape: (out_features, in_features)
        U, _, Vh = torch.linalg.svd(mean_jac_w, full_matrices=False)
        r = config.r
        init_lora_a = Vh[:r, :].clone()  # shape: (r, in_features)
        init_lora_b = U[:, :r].clone()  # shape: (out_features, r)
        del U, Vh
        jac_a = (init_lora_b.T.to('cuda') @ jac_w.to('cuda')).reshape(len(data_loader), -1).cpu()  # shape: (num_samples, r * in_features)
        jac_b = (jac_w.to('cuda') @ init_lora_a.T.to('cuda')).reshape(len(data_loader), -1).cpu()  # shape: (num_samples, out_features * r)
        del jac_w

        torch.save(
            dict(
                init_lora_a=init_lora_a,
                init_lora_b=init_lora_b,
                lora_a=jac_a, 
                lora_b=jac_b
            ), 
            f"{save_path}/{file_name}.pt"
        )
        del jac_a, jac_b, init_lora_a, init_lora_b

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
        init_lora_a = None
        init_lora_b = None
        for jacobian_path in jacobian_paths:
            if not os.path.exists(f"{jacobian_path}/{file_name}.pt"):
                raise FileNotFoundError(f"Jacobian file for {name} not found in {jacobian_path}, cannot calculate stiff basis.")

            jac = torch.load(f"{jacobian_path}/{file_name}.pt", map_location=get_model_device(model))
            full_jacobian_a.append(jac['lora_a'])
            full_jacobian_b.append(jac['lora_b'])
            if init_lora_a is None:
                init_lora_a = jac.get('init_lora_a', None)
            else:
                assert init_lora_a == jac.get('init_lora_a', None), f"init_lora_a is not consistent across different jacobian files for {name}."
            if init_lora_b is None:
                init_lora_b = jac.get('init_lora_b', None)
            else:
                assert init_lora_b == jac.get('init_lora_b', None), f"init_lora_b is not consistent across different jacobian files for {name}."
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
        # flip = (stiff_basis_a.T @ mean_jac_a).sign()
        # stiff_basis_a = flip.unsqueeze(0) * stiff_basis_a
        # stiff_basis_a = full_jacobian_a.T[:, :r_stiff_basis]
        # stiff_basis_a, _ = torch.linalg.qr(stiff_basis_a)  # Orthonormalize the stiff basis to improve numerical stability
        
        U, S, _ = torch.linalg.svd(full_jacobian_b.T, full_matrices=False)
        if adaptive_r_stiff_basis:
            max_r_stiff_basis = r_stiff_basis
            cumulative_energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
            r_stiff_basis = (torch.searchsorted(cumulative_energy, cumulative_energy_threshold) + 1).clip(min_r_stiff_basis, max_r_stiff_basis)
            print(f"Adjusted r_stiff_basis to {r_stiff_basis} for parameter {name} based on energy threshold {cumulative_energy_threshold}.")
        stiff_basis_b = U[:, :r_stiff_basis]
        # flip = (stiff_basis_b.T @ mean_jac_b).sign()
        # stiff_basis_b = flip.unsqueeze(0) * stiff_basis_b
        # stiff_basis_b = full_jacobian_b.T[:, :r_stiff_basis]
        # stiff_basis_b, _ = torch.linalg.qr(stiff_basis_b)
        
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

@torch.no_grad()
def calculate_stiff_basis_w(
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

        full_jacobian_w_a = []
        full_jacobian_w_b = []
        for jacobian_path in jacobian_paths:
            if not os.path.exists(f"{jacobian_path}/{file_name}.pt"):
                raise FileNotFoundError(f"Jacobian file for {name} not found in {jacobian_path}, cannot calculate stiff basis.")

            jac = torch.load(f"{jacobian_path}/{file_name}.pt", map_location=get_model_device(model))
            full_jacobian_w_a.append(jac['jac_w_a'])
            full_jacobian_w_b.append(jac['jac_w_b'])
        full_jacobian_w_a = torch.cat(full_jacobian_w_a, dim=0).to(torch.float32)
        full_jacobian_w_b = torch.cat(full_jacobian_w_b, dim=0).to(torch.float32)
            
        assert full_jacobian_w_a.shape[0] == r_stiff_basis
        
        stiff_basis_w_a = full_jacobian_w_a
        stiff_basis_w_b = full_jacobian_w_b
       
        
        if quantize_stiff_basis:
            quantized_stiff_basis_w_a, quant_state_w_a = quantize_blockwise(stiff_basis_w_a)
            quantized_stiff_basis_w_b, quant_state_w_b = quantize_blockwise(stiff_basis_w_b)
            torch.save(dict(
                stiff_basis_w_a=quantized_stiff_basis_w_a, 
                quant_state_w_a=quant_state_w_a,
                stiff_basis_w_b=quantized_stiff_basis_w_b, 
                quant_state_w_b=quant_state_w_b
            ), f"{save_path}/{file_name}.pt")
        else:
            torch.save(dict(stiff_basis_w_a=stiff_basis_w_a, stiff_basis_w_b=stiff_basis_w_b), f"{save_path}/{file_name}.pt")


class ProjectionCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """
    counter = 0

    def __init__(self, local_rank: int = 0, proj_interval: int = 1):
        super().__init__()
        self.local_rank = local_rank
        self.proj_interval = proj_interval

    def _project(self, model):
        for n, module in model.named_modules():
            if isinstance(module, XXXLayer):
                lora_a = module.lora_A[model.active_adapter].weight.data
                lora_b = module.lora_B[model.active_adapter].weight.data
                prev_lora_a = module.xxx_prev_lora_a
                prev_lora_b = module.xxx_prev_lora_b
                scaling = module.scaling[model.active_adapter]

                # ============== W proj =================
                stiff_basis_w_a = module.xxx_stiff_basis_w_a.float()
                stiff_basis_w_b = module.xxx_stiff_basis_w_b.float()
                _, r, din = stiff_basis_w_a.shape
                _, dout, _ = stiff_basis_w_b.shape
                stiff_basis_w_a_flat = stiff_basis_w_a.reshape(-1, din)
                stiff_basis_w_b_T_flat = stiff_basis_w_b.transpose(1, 2).reshape(-1, dout)
                k_inv = module.xxx_stiff_basis_k_inv

                # u_flat = ((stiff_basis_w_b_T_flat @ lora_b @ lora_a * scaling) * stiff_basis_w_a_flat).sum(dim=1)
                # u = u_flat.view(-1, r).sum(dim=1)
                # stiff_basis_w_b_weighted = stiff_basis_w_b * (k_inv @ u).view(-1, 1, 1)
                # P = rearrange(stiff_basis_w_b_weighted, 'm out r -> out (m r)') @ \
                #     rearrange(stiff_basis_w_a, 'm r in -> (m r) in')  # shape: (out_features, in_features)

                # delta_w_proj = scaling * lora_b @ lora_a - P

                # u_flat = ((stiff_basis_w_b_T_flat @ lora_b @ lora_a) * scaling * stiff_basis_w_a_flat).sum(dim=1)
                u_flat = ((stiff_basis_w_b_T_flat @ lora_b @ lora_a - stiff_basis_w_b_T_flat @ prev_lora_b @ prev_lora_a) * scaling * stiff_basis_w_a_flat).sum(dim=1)
                u = u_flat.view(-1, r).sum(dim=1)
                
                # delta_w_proj = scaling * (lora_b @ lora_a)
                # delta_w_proj = scaling * (lora_b @ lora_a - prev_lora_b @ prev_lora_a)
                weights = (k_inv @ u).view(-1, 1, 1)
                B_weighted_flat = (stiff_basis_w_b * weights).permute(1, 0, 2).reshape(dout, -1)
                # delta_w_proj.addmm_(B_weighted_flat, stiff_basis_w_a_flat, beta=1.0, alpha=-1.0)

                with torch.no_grad():
                    base_weight = module.base_layer.weight
                    orig_dtype = base_weight.dtype
                    
                    P = B_weighted_flat @ stiff_basis_w_a_flat
                    updated_weight_fp32 = base_weight.data.float() - P
                    if self.local_rank == 0:
                        # swallowed_ratio_bf16 = (updated_weight_fp32.to(torch.bfloat16) == base_weight.data.to(torch.bfloat16)).float().mean().item()
                        swallowed_ratio = ((updated_weight_fp32.to(orig_dtype) == base_weight.data) & (P.abs() > 1e-12)).float().mean().item()
                        w_norm = base_weight.data.norm().item()
                        ba_norm = (lora_b @ lora_a * scaling).norm().item()
                        p_norm = P.norm().item()
                        # print(f"Delta K: {(lora_b @ lora_a - prev_lora_b @ prev_lora_a).abs().mean().item():.10f}")
                        # print(f"{n} P float32: {P.abs().mean().item():.10f}")
                        # print(f"{n} W float32: {base_weight.data.float().abs().mean().item():.10f}")
                        # print(f"{n} W-P float32: {updated_weight_fp32.abs().mean().item():.10f}")
                        print(f"{n} w_norm: {w_norm}")
                        print(f"{n} ba_norm: {ba_norm}")
                        print(f"{n} p_norm: {p_norm}")
                        print(f"{n} swallowed_ratio_fp32: {swallowed_ratio}")

                        # print(f"{n} P bfloat16: {P.to(torch.bfloat16).abs().mean().item():.10f}")
                        # print(f"{n} W bfloat16: {base_weight.data.to(torch.bfloat16).abs().mean().item():.10f}")
                        # print(f"{n} W-P bfloat16: {updated_weight_fp32.to(torch.bfloat16).abs().mean().item():.10f}")
                        # print(f"{n} swallowed_ratio_bf16: {swallowed_ratio_bf16}")

                    
                    base_weight.data.copy_(updated_weight_fp32.to(orig_dtype))

                    module.xxx_prev_lora_a.copy_(lora_a)
                    module.xxx_prev_lora_b.copy_(lora_b)

                # U, S, V = torch.svd_lowrank(delta_w_proj, q=module.r[model.active_adapter])
                # U, S, V = torch.svd_lowrank(delta_w_proj + scaling * init_lora_b @ init_lora_a, q=module.r[model.active_adapter])
                # lora_b_svd = (U @ torch.diag(torch.sqrt(S / scaling)))
                # lora_a_svd = (torch.diag(torch.sqrt(S / scaling)) @ V.T)

                # delta_W = scaling * lora_b_svd @ lora_a_svd - scaling * init_lora_b @ init_lora_a
                # mean_err = (delta_W.flatten() @ (stiff_basis_w_b[:64] @ stiff_basis_w_a[:64]).reshape(-1, dout * din).T).abs().mean()
                # print(f"svd err wo T: {mean_err.item():.8f}")

                # with torch.no_grad():
                #     # 防弹判定：如果旧的 lora_b 几乎为 0，说明 Adam 根本还没积累动量
                #     # 此时直接使用 SVD 的结果，不需要任何对齐！
                #     if torch.norm(lora_b) < 1e-7:
                #         lora_b_new = lora_b_svd
                #         lora_a_new = lora_a_svd
                #     else:
                #         # 求解正交普氏问题：寻找纯旋转矩阵 T 使得 B_svd @ T 贴近 B_old
                #         # 1. 计算协方差矩阵 M
                #         M = lora_b_svd.T @ lora_b
                        
                #         # 2. 对 M 进行微型 SVD (r x r 级别，极快)
                #         U_proc, _, Vh_proc = torch.linalg.svd(M, full_matrices=False)
                        
                #         # 3. 构造正交变换矩阵 T
                #         T = U_proc @ Vh_proc
                        
                #         # 4. 应用变换！
                #         # 因为 T 是绝对正交的，所以 T 的逆就是 T 的转置 (T.T)
                #         lora_b_new = lora_b_svd @ T
                #         lora_a_new = T.T @ lora_a_svd
                # # ==========================================================

                # with torch.no_grad():
                #     # --- 第一步：固定 lora_a，更新 lora_b ---
                #     cov_A = lora_a @ lora_a.T
                #     cov_A.diagonal().add_(1e-7) # 阻尼防奇异
                #     inv_cov_A = torch.linalg.inv(cov_A)
                    
                #     # 结合律魔法：计算 P @ lora_a.T
                #     # (m*r, din) @ (din, r) -> (m*r, r)
                #     A_proj = stiff_basis_w_a_flat @ lora_a.T 
                #     # (dout, m*r) @ (m*r, r) -> (dout, r)
                #     P_A_T = B_weighted_flat @ A_proj 
                    
                #     # 目标 W_target @ A.T
                #     target_W_A_T = lora_b @ cov_A - P_A_T / scaling
                #     lora_b_new = target_W_A_T @ inv_cov_A
                    
                #     # --- 第二步：固定刚刚更新的 lora_b_new，更新 lora_a ---
                #     cov_B = lora_b_new.T @ lora_b_new
                #     cov_B.diagonal().add_(1e-7)
                #     inv_cov_B = torch.linalg.inv(cov_B)
                    
                #     # 结合律魔法：计算 lora_b_new.T @ P
                #     # (r, dout) @ (dout, m*r) -> (r, m*r)
                #     B_proj = lora_b_new.T @ B_weighted_flat
                #     # (r, m*r) @ (m*r, din) -> (r, din)
                #     B_P = B_proj @ stiff_basis_w_a_flat
                    
                #     # lora_b_new.T @ 目标 W_target
                #     B_T_target_W = (lora_b_new.T @ lora_b) @ lora_a - B_P / scaling
                #     lora_a_new = inv_cov_B @ B_T_target_W

                # delta_W = scaling * lora_b_new @ lora_a_new
                # mean_err = (delta_W.flatten() @ (stiff_basis_w_b[:64] @ stiff_basis_w_a[:64]).reshape(-1, dout * din).T).abs().mean()
                # print(f"svd err: {mean_err.item():.8f}")

                # delta_W = scaling * lora_b_svd @ lora_a_svd
                # mean_err = (delta_W.flatten() @ (stiff_basis_w_b[:64] @ stiff_basis_w_a[:64]).reshape(-1, dout * din).T).abs().mean()
                # print(f"svd err wo align: {mean_err.item():.8f}")

                # module.lora_A[model.active_adapter].weight.data.copy_(lora_a_svd.data)
                # module.lora_B[model.active_adapter].weight.data.copy_(lora_b_svd.data)


    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.counter += 1
        if self.counter % self.proj_interval != 0:
            return control
        self._project(kwargs["model"])
        return control
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Ensure projection is applied at the end of training
        self._project(kwargs["model"])
        return control


# class ProjectionCallback(TrainerCallback):
#     """
#     A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
#     """

#     def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
#         model = kwargs["model"]
#         for _, module in model.named_modules():
#             if isinstance(module, XXXLayer):
#                 lora_a = module.lora_A[model.active_adapter].weight
#                 lora_b = module.lora_B[model.active_adapter].weight
#                 init_lora_a = module.xxx_init_lora_a[model.active_adapter]
#                 init_lora_b = module.xxx_init_lora_b[model.active_adapter]
#                 delta_a = lora_a - init_lora_a
#                 delta_b = lora_b - init_lora_b

#                 # ============== Seperate proj =================
#                 stiff_basis_a = module.xxx_stiff_basis_a[model.active_adapter]()
#                 stiff_basis_b = module.xxx_stiff_basis_b[model.active_adapter]()
#                 quant_state_a = module.xxx_stiff_basis_a_quant_state[model.active_adapter]
#                 quant_state_b = module.xxx_stiff_basis_b_quant_state[model.active_adapter]
#                 stiff_basis_a = dequantize_blockwise(stiff_basis_a, quant_state_a)
#                 stiff_basis_b = dequantize_blockwise(stiff_basis_b, quant_state_b)
#                 projected_delta_a = delta_a.data.reshape(-1) - delta_a.data.reshape(-1) @ stiff_basis_a @ stiff_basis_a.T
#                 projected_delta_b = delta_b.data.reshape(-1) - delta_b.data.reshape(-1) @ stiff_basis_b @ stiff_basis_b.T
#                 lora_a.data = projected_delta_a.reshape(lora_a.shape) + init_lora_a.data
#                 lora_b.data = projected_delta_b.reshape(lora_b.shape) + init_lora_b.data

#                 # ============== Combined proj =================
#                 # stiff_basis = module.xxx_stiff_basis[model.active_adapter]()
#                 # quant_state = module.xxx_stiff_basis_quant_state[model.active_adapter]
#                 # stiff_basis = dequantize_blockwise(stiff_basis, quant_state)
#                 # delta = torch.cat([delta_a.reshape(-1), delta_b.reshape(-1)], dim=0)
#                 # stiff_basis = torch.cat([stiff_basis_a, stiff_basis_b], dim=0)
#                 # projected_delta = delta.data - delta.data @ stiff_basis @ stiff_basis.T
#                 # lora_a.data = projected_delta[:delta_a.numel()].reshape(lora_a.shape) + init_lora_a.data
#                 # lora_b.data = projected_delta[delta_a.numel():].reshape(lora_b.shape) + init_lora_b.data
#                 # ============== Combined proj norm together =================
#                 # stiff_basis = module.xxx_stiff_basis[model.active_adapter]()
#                 # quant_state = module.xxx_stiff_basis_quant_state[model.active_adapter]
#                 # stiff_basis = dequantize_blockwise(stiff_basis, quant_state)
#                 # delta = torch.cat([delta_a.reshape(-1), delta_b.reshape(-1)], dim=0)
#                 # projected_delta = delta.data - delta.data @ stiff_basis @ stiff_basis.T
#                 # lora_a.data = projected_delta[:delta_a.numel()].reshape(lora_a.shape) + init_lora_a.data
#                 # lora_b.data = projected_delta[delta_a.numel():].reshape(lora_b.shape) + init_lora_b.data
#                 # ============== Zero out delta weights =================
#                 # module.xxx_delta_lora_a[model.active_adapter].weight.data.copy_(lora_a.data - init_lora_a.data)
#                 # module.xxx_delta_lora_b[model.active_adapter].weight.data.copy_(lora_b.data - init_lora_b.data)
#                 # ============== Zero out delta weights =================
#                 # print("norm after projection:", delta_weight.data.norm().item())
#         return control