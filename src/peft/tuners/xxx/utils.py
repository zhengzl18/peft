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

from peft.tuners.xxx.config import XXXConfig, XXXPreprocessConfig
from peft.tuners.xxx.layer import XXXLayer
import torch
import torch.nn as nn
from tqdm import tqdm

from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.model import LoraModel
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
    for name, module in model.named_modules():
        # todo: change LoraModel to XXXModel
        if LoraModel._check_target_module_exists(config, name) and isinstance(module, nn.Linear):
            yield name, module

def target_params(model: nn.Module, config: XXXConfig) -> Iterable[nn.Parameter]:
    # TODO: maybe support bias
    for name, module in target_modules(model, config):
        yield f"{name}.weight", module.weight

def get_model_device(model: nn.Module) -> str:
    if hasattr(model, "module"):  # Handle DeepSpeed/DataParallel
        model = model.module
    return next(iter(model.parameters())).device.type

def get_param_downsample_mask(
    name: str,
    num_params: int,
    n_param_downsample_rate: float,
    fwd_importance_sampling: bool = False,
    fwd_importance_score_path: Optional[str] = None,
    bwd_importance_sampling: bool = False,
    bwd_importance_score_path: Optional[str] = None,
) -> torch.Tensor:
    if n_param_downsample_rate > 1.0 or n_param_downsample_rate <= 0.0:
        raise ValueError("n_param_downsample_rate should be in (0.0, 1.0].")
    elif n_param_downsample_rate == 1.0:
        mask = torch.arange(num_params).clone()  # Use all params
        return mask
    else:
        k = int(num_params * n_param_downsample_rate)
        if not fwd_importance_sampling and not bwd_importance_sampling:
            mask = torch.randperm(num_params)[:k].clone()
            return mask
        
        fwd_importance_score = None
        bwd_importance_score = None
        if fwd_importance_sampling:
            assert fwd_importance_score_path is not None, "fwd_importance_score_path should be specified when fwd_importance_sampling is True."
            fwd_importance_score = torch.load(
                f"{fwd_importance_score_path}/{name.replace('.', '-')}.pt",
            )
        if bwd_importance_sampling:
            assert bwd_importance_score_path is not None, "bwd_importance_score_path should be specified when bwd_importance_sampling is True."
            bwd_importance_score = torch.load(
                f"{bwd_importance_score_path}/{name.replace('.', '-')}.pt",
            ) + EPS
        
        if fwd_importance_score is None:
            fwd_importance_score = torch.ones_like(bwd_importance_score)
        if bwd_importance_score is None:
            bwd_importance_score = torch.ones_like(fwd_importance_score)
        
        score = fwd_importance_score / bwd_importance_score
        # take top-k indices
        _, mask = torch.topk(score, k=k, largest=True, sorted=False)
        return mask

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
    fwd_importance_sampling = xxx_config.preprocess_config.fwd_importance_sampling
    bwd_importance_sampling = xxx_config.preprocess_config.bwd_importance_sampling
    
    for module_name, module in target_modules(model, xxx_config):
        module_stiff_basis = {}
        for param_name, _ in target_params(model, xxx_config):
            if not param_name.startswith(module_name):
                continue
            file_name = param_name.replace('.', '-')
            if fwd_importance_sampling:
                file_name += "_fwd"
            if bwd_importance_sampling:
                file_name += "_bwd"
            
            if not os.path.exists(f"{stiff_basis_path}/{file_name}.pt"):
                raise FileNotFoundError(f"Stiff basis file for {param_name} not found in {stiff_basis_path}, run preprocess.py to build stiff basis first.")

            print(f"Loading stiff basis from {stiff_basis_path}/{file_name}.pt on cuda:{local_rank} ...")
            stiff_basis = torch.load(f"{stiff_basis_path}/{file_name}.pt", map_location=f"cuda:{local_rank}")
            
            if quantize_stiff_basis:
                assert "quant_state" in stiff_basis, "quantize_stiff_basis is True but quant_state not found in the loaded stiff basis."
            else:
                assert "quant_state" not in stiff_basis, "quantize_stiff_basis is False but quant_state found in the loaded stiff basis."
                stiff_basis['stiff_basis'] = stiff_basis['stiff_basis'].to(model.dtype)
            # items = [(k, v) for k, v in stiff_basis.items()]
            # for k, v in items:
            #     if "mask" not in k:
            #         jac[k] = quantized_jac
            #         jac["quant_state"] = quant_state
            #     else:
            #         stiff_basis[k] = v
            module_stiff_basis[param_name.split('.')[-1]] = stiff_basis
        module.xxx_stiff_basis = module_stiff_basis

def calculate_jacobian(
    model: nn.Module,
    config: XXXConfig,
    data_loader: List[Dict[str, torch.Tensor]],
    mask_path: Optional[str] = None,
):
    save_path = config.preprocess_config.jacobian_path
    assert isinstance(save_path, str), f"jacobian_path in preprocess_config is expected to be a string for calculating and saving jacobians, got {type(config.jacobian_path)}."
    n_param_downsample_rate = config.preprocess_config.n_param_downsample_rate
    fwd_importance_sampling = config.preprocess_config.fwd_importance_sampling
    bwd_importance_sampling = config.preprocess_config.bwd_importance_sampling
    fwd_importance_score_path = config.preprocess_config.fwd_importance_score_path
    bwd_importance_score_path = config.preprocess_config.bwd_importance_score_path

    model.train()
    os.makedirs(save_path, exist_ok=True)
    for name, p in target_params(model, config):
        assert len(p.shape) <= 2
        assert '-' not in name
        file_name = name.replace('.', '-')
        if fwd_importance_sampling:
            file_name += f"_fwd"
        if bwd_importance_sampling:
            file_name += f"_bwd"

        if os.path.exists(f"{save_path}/{file_name}.pt"):
            print(f"Jacobian file for {save_path}/{name} already exists, skipping.")
            continue
        
        print(f"Calculating jacobian for {save_path}/{name} ...")
        for param in model.parameters():
            param.requires_grad = False
        grads = []
        p.requires_grad = True  # Only compute gradient for the target parameter

        if mask_path is not None:
            assert os.path.exists(f"{mask_path}/{file_name}.pt"), f"Mask file for {name} not found in {mask_path}."
            mask = torch.load(f"{mask_path}/{file_name}.pt", map_location=get_model_device(model))['mask']
        else:
            mask = get_param_downsample_mask(
                name,
                p.numel(), 
                n_param_downsample_rate,
                fwd_importance_sampling,
                fwd_importance_score_path,
                bwd_importance_sampling,
                bwd_importance_score_path
            )

        for data in tqdm(data_loader):
            data = {k: v.to(model.device) for k, v in data.items()}
            model.zero_grad()
            outputs = model(**data)
            outputs.loss.backward()
            assert p.grad is not None
            grads.append(p.grad.reshape(-1)[mask].clone().cpu())
        model.zero_grad()
        
        # stack grads into jacobian matrix
        jac = torch.stack(grads, dim=0)
        torch.save(dict(jac=jac, mask=mask), f"{save_path}/{file_name}.pt")

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
    fwd_importance_sampling = config.preprocess_config.fwd_importance_sampling
    bwd_importance_sampling = config.preprocess_config.bwd_importance_sampling
    quantize_stiff_basis = config.preprocess_config.quantize_stiff_basis

    os.makedirs(save_path, exist_ok=True)
    for name, p in target_params(model, config):
        assert len(p.shape) <= 2
        assert '-' not in name
        file_name = name.replace('.', '-')
        if fwd_importance_sampling:
            file_name += f"_fwd"
        if bwd_importance_sampling:
            file_name += f"_bwd"

        if os.path.exists(f"{save_path}/{file_name}.pt"):
            print(f"Stiff basis file for {save_path}/{name} already exists, skipping.")
            continue

        print(f"Calculating stiff basis for {name} ...")

        full_jacobian = []
        mask = None
        for jacobian_path in jacobian_paths:
            if not os.path.exists(f"{jacobian_path}/{file_name}.pt"):
                raise FileNotFoundError(f"Jacobian file for {name} not found in {jacobian_path}, cannot calculate stiff basis.")

            jac = torch.load(f"{jacobian_path}/{file_name}.pt", map_location=get_model_device(model))
            if mask is None:
                mask = jac['mask']
            else:
                assert torch.equal(mask, jac['mask']), "Masks from different jacobian files do not match."
            full_jacobian.append(jac['jac'])
        full_jacobian = torch.cat(full_jacobian, dim=0).to(torch.float32)
            
        if full_jacobian.shape[0] < r_stiff_basis:
            warnings.warn(
                f"r_stiff_basis {r_stiff_basis} is larger than the total number of knowledge dataset samples {full_jacobian.shape[0]} for parameter {name}."
            )
        U, S, _ = torch.linalg.svd(full_jacobian.T, full_matrices=False)
        if adaptive_r_stiff_basis:
            max_r_stiff_basis = r_stiff_basis
            cumulative_energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
            r_stiff_basis = (torch.searchsorted(cumulative_energy, cumulative_energy_threshold) + 1).clip(min_r_stiff_basis, max_r_stiff_basis)
            print(f"Adjusted r_stiff_basis to {r_stiff_basis} for parameter {name} based on energy threshold {cumulative_energy_threshold}.")
        stiff_basis = U[:, :r_stiff_basis]
        
        if quantize_stiff_basis:
            quantized_stiff_basis, quant_state = quantize_blockwise(stiff_basis)
            torch.save(dict(stiff_basis=quantized_stiff_basis, quant_state=quant_state, mask=mask), f"{save_path}/{file_name}.pt")
        else:
            torch.save(dict(stiff_basis=stiff_basis, mask=mask), f"{save_path}/{file_name}.pt")

def calculate_importance_score(
    model: nn.Module,
    data_loader: List[Dict[str, torch.Tensor]],
    config: XXXConfig,
    mode: str,
    save_path: str,
):
    model.train()
    os.makedirs(save_path, exist_ok=True)
    grad = {}
    assert mode in IMPORTANCE_SCORE_MODE, f"Unsupported importance score mode: {mode}"
    for param in model.parameters():
        param.requires_grad = False
    for name, p in target_params(model, config):
        if os.path.exists(f"{save_path}/{name.replace('.', '-')}.pt"):
            print(f"Importance score file for {name} already exists, skipping.")
            continue
        p.requires_grad = True  # Only compute gradient for the target parameter
        grad[name] = torch.zeros_like(p).reshape(-1).cpu()
    
    if len(grad) == 0:
        return
    else:
        for data in tqdm(data_loader):
            data = {k: v.to(model.device) for k, v in data.items()}
            model.zero_grad()
            outputs = model(**data)
            outputs.loss.backward()

            for name in grad:
                p = model.get_parameter(name)
                assert p.grad is not None
                grad[name] += IMPORTANCE_SCORE_MODE[mode](p.grad).reshape(-1).clone().cpu()
        model.zero_grad()

        for name in grad:
            grad[name] /= len(data_loader)
            assert '-' not in name
            torch.save(grad[name], f"{save_path}/{name.replace('.', '-')}.pt")


class ProjectionCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        for _, module in model.named_modules():
            if isinstance(module, XXXLayer):
                delta_weight = module.xxx_delta_weight[model.active_adapter]
                stiff_basis = module.xxx_stiff_basis_w[model.active_adapter]()
                quant_state = module.xxx_stiff_basis_w_quant_state[model.active_adapter]
                stiff_basis = dequantize_blockwise(stiff_basis, quant_state)
                # print("\nnorm before projection:", delta_weight.data.norm().item())
                # print(torch.isfinite(delta_weight).all())
                delta_weight.data = delta_weight.data - delta_weight.data @ stiff_basis @ stiff_basis.T
                # print("norm after projection:", delta_weight.data.norm().item())
        return control