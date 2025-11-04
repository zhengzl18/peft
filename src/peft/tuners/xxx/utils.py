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
from typing import Any, Callable, Dict, List, Optional

from peft.tuners.xxx.config import XXXConfig, XXXPreprocessConfig
import torch
import torch.nn as nn
from tqdm import tqdm

from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.model import LoraModel
from peft.utils.other import get_pattern_key


IMPORTANCE_SCORE_MODE = {
    "abs": lambda x: torch.abs(x),
    "square": lambda x: x ** 2,
}
EPS = 1e-20
    

def target_modules(model: nn.Module, config: LoraConfig) -> Iterable[nn.Module]:
    """
    Iterate over CorDA target name and modules of a model. A module is a target if its name is in
    `config.target_modules` and is `nn.Linear`.
    """
    for name, module in model.named_modules():
        if LoraModel._check_target_module_exists(config, name) and isinstance(module, nn.Linear):
            yield name, module

def target_params(model: nn.Module, config: LoraConfig) -> Iterable[nn.Parameter]:
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
    knowledge_loader: Optional[List[Dict[str, torch.Tensor]]] = None,
    local_rank: int = 0,
):
    """
    Build necessary CorDA fields for a model.

    For each `M * N` linear layer, a `M * M` jacobian matrix will be built temporarily during the preprocessing
    process, consuming roughly another `2 * MODEL_SIZE` memory for typical LLMs if model weight is FP16 and jacobian
    is FP32. If that's too much, consider specifying `use_float16_for_jacobian` in `preprocess_config`.

    Args:
        model (`nn.Module`):
            Model to preprocess.
        lora_config (`LoraConfig`):
            Lora configuration of the model. `preprocess_config` should be set.
        run_model (`Optional[Callable[[], None]]`):
            Callback to run the model when building jacobian. Typically you should run model inference on your sample
            dataset in this callback. Experiments have shown that when token count per sample is 2048, hidden dimension
            is 4096, collecting 256 distinct samples is enough. If you collect too few or too repetitive samples, the
            jacobian matrix may be low-ranked and unstabilize preprocessing. You can estimate sample count as
            `HIDDEN_DIM / TOKEN_PER_SAMPLE * 128`. `run_model` can be `None` only if jacobian file in
            `preprocess_config` is already created.
        hooked_model (`Optional[nn.Module]`):
            Model to hook when building jacobian. If none, original model will be hooked. This is only useful when
            you want to hook a different model than the one you are training, typically you should leave this `None`.

    Upon completion, the following fields are set for each target module:
        eigens.S_WC (`torch.Tensor`):
            Singular values of the weight matrix.
        eigens.U_WC (`torch.Tensor`):
            Left singular vectors of the weight matrix.
        eigens.V_WC (`torch.Tensor`):
            Right singular vectors of the weight matrix, multiplied by inverse of jacobian matrix.
    """
    sloppy_basis_path = xxx_config.preprocess_config.sloppy_basis_path
    jacobian_path = xxx_config.preprocess_config.jacobian_path
    eigen_path = xxx_config.preprocess_config.eigen_path
    fwd_importance_sampling = xxx_config.preprocess_config.fwd_importance_sampling
    bwd_importance_sampling = xxx_config.preprocess_config.bwd_importance_sampling
    use_cache = False

    # If cache exists, skip building
    if sloppy_basis_path is not None and os.path.exists(sloppy_basis_path) and os.listdir(sloppy_basis_path):
        use_cache = True
        for name, module in target_modules(model, xxx_config):
            file_name = name.replace('.', '-')
            if fwd_importance_sampling:
                file_name += "_fwd"
            if bwd_importance_sampling:
                file_name += "_bwd"
            if not os.path.exists(f"{sloppy_basis_path}/{file_name}.pt"):
                print(f"Sloppy basis file for {name} not found in {sloppy_basis_path}, rebuilding all.")
                use_cache = False
                break
            else:
                print(f"Loading sloppy basis for {name} on cuda:{local_rank} ...")
                # sloppy_basis = torch.load(f"{sloppy_basis_path}/{file_name}.pt", map_location=get_model_device(model))
                sloppy_basis = torch.load(f"{sloppy_basis_path}/{file_name}.pt", map_location=f"cuda:{local_rank}")
                for k, v in sloppy_basis.items():
                    if "mask" not in k:
                        # print(v.dtype)
                        sloppy_basis[k] = v.to(model.dtype)
                    else:
                        sloppy_basis[k] = v
                module.xxx_sloppy_basis = sloppy_basis
    
    if not use_cache:
        # Specify CorDA rank for each layer
        for name, module in target_modules(model, xxx_config):
            r_key = get_pattern_key(xxx_config.rank_pattern.keys(), name)
            module.rank = xxx_config.rank_pattern.get(r_key, xxx_config.r)

        # Calculate jacobian matrix
        # if not os.path.exists(jacobian_path) or not os.listdir(jacobian_path):
        calculate_jacobian(
            model, 
            knowledge_loader, 
            xxx_config, 
            jacobian_path,
        )

        # Calculate eigens
        # calculate_eigens(model, xxx_config, jacobian_path, eigen_path)

        # Crop CorDA eigens so that there's less to save
        calculate_sloppy_basis(model, xxx_config, jacobian_path, sloppy_basis_path, eigen_path=eigen_path, target_jacobian_path=None)

        # if sloppy_basis_path is not None:
        #     os.makedirs(sloppy_basis_path, exist_ok=True)

        for name, module in target_modules(model, xxx_config):
            # Load sloppy basis from disk
            file_name = name.replace('.', '-')
            if fwd_importance_sampling:
                file_name += "_fwd"
            if bwd_importance_sampling:
                file_name += "_bwd"
            sloppy_basis = torch.load(f"{sloppy_basis_path}/{file_name}.pt")
            module.xxx_sloppy_basis = sloppy_basis
    

def calculate_jacobian(
    model: nn.Module,
    knowledge_loader: List[Dict[str, torch.Tensor]],
    config: XXXConfig,
    jacobian_path: str,
):
    n_param_downsample_rate = config.preprocess_config.n_param_downsample_rate
    fwd_importance_sampling = config.preprocess_config.fwd_importance_sampling
    bwd_importance_sampling = config.preprocess_config.bwd_importance_sampling
    fwd_importance_score_path = config.preprocess_config.fwd_importance_score_path
    bwd_importance_score_path = config.preprocess_config.bwd_importance_score_path
    
    model.train()
    os.makedirs(jacobian_path, exist_ok=True)
    for name, p in target_params(model, config):
        assert len(p.shape) <= 2
        assert '-' not in name
        file_name = name.replace('.', '-')
        if fwd_importance_sampling:
            file_name += f"_fwd"
        if bwd_importance_sampling:
            file_name += f"_bwd"
        
        if os.path.exists(f"{jacobian_path}/{file_name}.pt"):
            print(f"Jacobian file for {name} already exists, skipping.")
            continue
        print(f"Calculating jacobian for {name} ...")
        for param in model.parameters():
            param.requires_grad = False
        grads = []
        p.requires_grad = True  # Only compute gradient for the target parameter

        mask = get_param_downsample_mask(
            name,
            p.numel(), 
            n_param_downsample_rate,
            fwd_importance_sampling,
            fwd_importance_score_path,
            bwd_importance_sampling,
            bwd_importance_score_path
        )
        # mask = torch.randperm(p.numel())[:int(p.numel() * n_param_downsample_rate)].clone()  # Sample params based on downsample rate
        for data in tqdm(knowledge_loader):
            data = {k: v.to(model.device) for k, v in data.items()}
            model.zero_grad()
            outputs = model(**data)
            outputs.loss.backward()
            assert p.grad is not None
            grads.append(p.grad.reshape(-1)[mask].clone().cpu())
        model.zero_grad()
        
        # stack grads into jacobian matrix
        jac = torch.stack(grads, dim=0)
        torch.save(dict(jac=jac, mask=mask), f"{jacobian_path}/{file_name}.pt")

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

@torch.no_grad()
def calculate_eigens(
    model: nn.Module,
    config: LoraConfig,
    jacobian_path: str,
    eigen_path: Optional[str] = None,
):
    """Call collect_eigens_for_layer and store result in key `eigens` of each layer."""
    for name, _ in target_params(model, config):
        module_name = '.'.join(name.split('.')[:-1])
        r_key = get_pattern_key(config.rank_pattern.keys(), module_name)
        rank = config.rank_pattern.get(r_key, config.r)
        try:
            jac = torch.load(f"{jacobian_path}/{name.replace('.', '-')}.pt", map_location=get_model_device(model))
        except FileNotFoundError:
            raise FileNotFoundError(f"Jacobian file for {name} not found in {jacobian_path}.")
        if jac.shape[0] <= jac.shape[1] - rank:
            # Data samples num is smaller than param dim,
            # no need to perform svd
            pass
        else:
            # TODO: svd
            assert eigen_path is not None
            pass

@torch.no_grad()
def calculate_sloppy_basis(
    model: nn.Module,
    config: LoraConfig,
    jacobian_path: str,
    sloppy_basis_path: str,
    target_jacobian_path: Optional[str] = None,
    eigen_path: Optional[str] = None,
):
    fwd_importance_sampling = config.preprocess_config.fwd_importance_sampling
    bwd_importance_sampling = config.preprocess_config.bwd_importance_sampling
    # all_module_sloppy_basis = {}
    os.makedirs(sloppy_basis_path, exist_ok=True)
    for name, _ in target_params(model, config):
        param_name = name.split('.')[-1]
        param_file_name = name.replace('.', '-')
        module_name = '.'.join(name.split('.')[:-1])
        module_file_name = module_name.replace('.', '-')
        if fwd_importance_sampling:
            param_file_name += "_fwd"
            module_file_name += "_fwd"
        if bwd_importance_sampling:
            param_file_name += "_bwd"
            module_file_name += "_bwd"
        if os.path.exists(f"{sloppy_basis_path}/{module_file_name}.pt"):
            print(f"Sloppy basis file for {name} already exists, skipping.")
            continue
        print(f"Calculating sloppy basis for {name} ...")
        # if module_name not in all_module_sloppy_basis:
        #     all_module_sloppy_basis[module_name] = {}
        
        r_key = get_pattern_key(config.rank_pattern.keys(), module_name)
        rank = config.rank_pattern.get(r_key, config.r)

        use_jac_as_fallback = False
        if eigen_path is not None:
            try:
                eig = torch.load(
                    f"{eigen_path}/{param_file_name}.pt", 
                    map_location=get_model_device(model)
                )
                # TODO: support this
                stiff_vecs = eig[:eig.shape[0] - rank, :].T.to(torch.float32)  # Shape: (num_params, num_params - rank)
            except FileNotFoundError:
                # If eigen_path is specified but file not found, fallback to using jacobian
                # It is expected that jac.shape[0] <= jac.shape[1] - rank
                use_jac_as_fallback = True
        
        if eigen_path is None or use_jac_as_fallback:
            try:
                jac_and_mask = torch.load(
                    f"{jacobian_path}/{param_file_name}.pt", 
                    map_location=get_model_device(model)
                )
                jac = jac_and_mask['jac']
                mask = jac_and_mask['mask']
                assert jac.shape[0] <= jac.shape[1] - rank
                stiff_vecs, _ = torch.linalg.qr(jac.T.to(torch.float32))  # normalization, shape: (num_params, num_samples)
            except FileNotFoundError:
                raise FileNotFoundError(f"Jacobian file for {name} not found in {jacobian_path}.")
        
        
        num_params = stiff_vecs.shape[0]
        use_random_vecs = True
        if target_jacobian_path is not None:
            use_random_vecs = False
            try:
                target_jac = torch.load(
                    f"{target_jacobian_path}/{param_file_name}.pt", 
                    map_location=get_model_device(model)
                )
                assert target_jac.shape[0] >= rank
                assert target_jac.shape[1] == num_params
                projection = stiff_vecs @ (stiff_vecs.T @ target_jac.T)
                sloppy_vecs = target_jac.T - projection
            except FileNotFoundError:
                use_random_vecs = True

        if use_random_vecs:
            num_candidates = rank + 10  # FIXME: why 10?
            random_vecs = torch.randn(
                num_params, num_candidates,
                dtype=torch.float32,
                device=get_model_device(model)
            )
            projection = stiff_vecs @ (stiff_vecs.T @ random_vecs)
            sloppy_vecs = random_vecs - projection

        sloppy_basis, _ = torch.linalg.qr(sloppy_vecs)
        torch.save(
            {
                param_name: sloppy_basis[:, :rank].clone().to(model.dtype),
                f"{param_name}_mask": mask
            },
            f"{sloppy_basis_path}/{module_file_name}.pt"
        )  #! FIXME: this only supports weight param
        # all_module_sloppy_basis[module_name][param_name] = sloppy_basis[:, :rank].clone().to(model.dtype)  # Shape: (num_params, rank)
        # del random_vecs, projection, sloppy_vecs, sloppy_basis, stiff_vecs  # free memory
    
    # return all_module_sloppy_basis