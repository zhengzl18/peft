
from collections.abc import Iterable
import json
import os
from typing import Optional
from peft.config import PeftConfig
import torch
import torch.nn as nn
from torch.nn import Linear
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft.peft_model import PeftModel
from peft.tuners.lora.model import LoraModel
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.config import LoraConfig


def target_modules(model: nn.Module, config: LoraConfig) -> Iterable[nn.Module]:
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


class Projector:
    def __init__(
        self,
        peft_adapter_path: str,
        jac_path: str,
        current_projected_model_path: Optional[str] = None,
        device_map: str = "auto"
    ):
        self.jac_path = jac_path
        self.config = PeftConfig.from_pretrained(peft_adapter_path)
        pissa_residual_model = AutoModelForCausalLM.from_pretrained(
            "/home/fit/lishbo/WORK/data/zhengzhilong/anticf/pissa_residual_model/Llama-2-7b-hf",
            dtype=torch.float32,
            device_map=device_map
        )
        self.device = pissa_residual_model.device
        self.final_peft_model = PeftModel.from_pretrained(
            pissa_residual_model,
            peft_adapter_path,
            dtype=torch.float32,
        ).merge_and_unload()
        del pissa_residual_model
        # self.config = PeftConfig.from_pretrained("/home/fit/lishbo/WORK/zzl/repo/peft/output/metamath-lora-llama-2-7b/checkpoint-782/")
        # self.final_peft_model = AutoModelForCausalLM.from_pretrained(
        #     peft_adapter_path,
        #     dtype=torch.float32,
        #     device_map=device_map
        # )
        # self.device = self.final_peft_model.device
        
        if current_projected_model_path is not None:
            self.current_projected_model = AutoModelForCausalLM.from_pretrained(
                current_projected_model_path,
                dtype=torch.float32,
                device_map=device_map
            )
        else:
            self.current_projected_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name_or_path,
                dtype=torch.float32,
                device_map=device_map
            )  # Load again in case that base_model is modified by self.get_merged_final_peft_model()

    @torch.no_grad()
    def project(self,):
        for name, module in target_modules(self.final_peft_model, self.config):
            # assert name.startswith("base_model.model.")
            # name = name[len("base_model.model."):]
            print(name)

            file_name = name.replace('.', '-')
            if not os.path.exists(f"{self.jac_path}/{file_name}.pt"):
                raise FileNotFoundError(f"Stiff basis file for {name} not found in {self.jac_path}, run preprocess.py to build stiff basis first.")
            jac = torch.load(
                f"{self.jac_path}/{file_name}.pt", 
                map_location=self.device
            )
            jac_w_a = jac['jac_w_a'].float()
            jac_w_b = jac['jac_w_b'].float()
            k_inv = jac['K_inv']
            # lora_a_init = jac['init_lora_a']
            # lora_b_init = jac['init_lora_b']

            # lora_ba = module.get_delta_weight(self.final_peft_model.active_adapter).data
            # pissa_lora_ba_init = lora_b_init @ lora_a_init * module.scaling[self.final_peft_model.active_adapter]
            # w0 = self.base_model.get_submodule(name).weight.data
            # pissa_w0 = w0 - pissa_lora_ba_init
            # delta_w = lora_ba - pissa_lora_ba_init
            w = module.weight.data.float()
            delta_w = (w - self.current_projected_model.get_submodule(name).weight.data).float()

            _, r, din = jac_w_a.shape
            _, dout, _ = jac_w_b.shape
            jac_w_a_flat = jac_w_a.reshape(-1, din)
            jac_w_b_T_flat = jac_w_b.transpose(1, 2).reshape(-1, dout)

            u_flat = ((jac_w_b_T_flat @ delta_w) * jac_w_a_flat).sum(dim=1)
            u = u_flat.view(-1, r).sum(dim=1)
            weights = (k_inv @ u).view(-1, 1, 1)
            B_weighted_flat = (jac_w_b * weights).permute(1, 0, 2).reshape(dout, -1)

            assert w.dtype == torch.float32
            # assert w0.dtype == torch.float32
            P = B_weighted_flat @ jac_w_a_flat
            updated_weight_fp32 = w - P
            # updated_weight_fp32 = w0 - 0.5*pissa_lora_ba_init - 0.5*lora_ba - 0.5*P
            # updated_weight_fp32 = pissa_w0 - P
            swallowed_ratio = ((updated_weight_fp32 == w) & (P.abs() > 1e-12)).float().mean().item()
            # swallowed_ratio = ((updated_weight_fp32 == pissa_w0) & (P.abs() > 1e-12)).float().mean().item()
            # pissa_w0_norm = pissa_w0.norm().item()
            delta_w_norm = delta_w.norm().item()
            p_norm = P.norm().item()
            
            # print(f"{name} pissa_w0_norm: {pissa_w0_norm}")
            print(f"{name} delta_w_norm: {delta_w_norm}")
            print(f"{name} p_norm: {p_norm}")
            print(f"{name} swallowed_ratio_fp32: {swallowed_ratio}")

            w.copy_(updated_weight_fp32)
            # w0.copy_(updated_weight_fp32)

        return self.final_peft_model
    

if __name__ == "__main__":
    base_model_path = "meta-llama/Llama-2-7b-hf"
    peft_adapter_path = "/home/fit/lishbo/WORK/zzl/repo/peft/output/code-pissa-llama-2-7b/checkpoint-782"
    # jac_path = "/home/fit/lishbo/WORK/data/zhengzhilong/anticf/jacobian/metamath_projected_half_tmp"
    jac_path = "/home/fit/lishbo/WORK/data/zhengzhilong/anticf/jacobian/nqopen_meta-llama_Llama-2-7b-hf_pissa_r_jac_approx32_256_233"
    # current_projected_model_path = "/home/fit/lishbo/WORK/zzl/repo/peft/output/metamath-pissa-llama-2-7b-r128/projected_half"
    current_projected_model_path = None
    save_path = f"{os.path.dirname(peft_adapter_path)}/projected"
    print(f"Saving projected model to {save_path}")
    projector = Projector(
        # base_model_path="meta-llama/Llama-2-7b-hf",
        peft_adapter_path=peft_adapter_path,
        jac_path=jac_path,
        current_projected_model_path=current_projected_model_path
    )
    projected_model = projector.project()
    projected_model = projected_model.to(torch.bfloat16)
    # projected_model = projected_model.merge_and_unload().to(torch.bfloat16)
    projected_model.save_pretrained(save_path)

    tokenizer_config = json.load(open(os.path.join(peft_adapter_path, "tokenizer_config.json"), "r"))
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        model_max_length=tokenizer_config["model_max_length"],
        padding_side=tokenizer_config["padding_side"],
        use_fast=True,
    )
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.save_pretrained(save_path)