
from collections.abc import Iterable
import os
from typing import Dict, List, Optional, Sequence
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from peft.config import PeftConfig
from peft.tuners.lora.config import LoraConfig
import torch
import torch.nn as nn
from torch.nn import Linear
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from peft import get_peft_model
from peft.peft_model import PeftModel
from peft.utils import PeftType
from peft.tuners.lora.model import LoraModel
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.xxx.config import XXXConfig
from peft.tuners.xxx.layer import XXXLayer

from datautils import get_knowledge_data


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


class Projector:
    def __init__(
        self,
        model_path: str,
        config: PeftConfig,
        knowledge_dataset: Optional[Sequence[str]] = None,
        n_knowledge_samples: Optional[int] = None,
        r_jac_approx: Optional[int] = None,
        seed: int = 42,
        device_map: str = "auto"
    ):
        self.config = config
        self.knowledge_dataset = knowledge_dataset
        self.n_knowledge_samples = n_knowledge_samples
        self.r_jac_approx = r_jac_approx
        self.seed = seed
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            device_map=device_map
        )

    def _calc_jac(
        self,
        model: nn.Module,
        config: LoraConfig,
        data_loader: List[Dict[str, torch.Tensor]],
    ):
        model.eval()
        save_path = "/home/fit/lishbo/WORK/data/zhengzhilong/anticf/jacobian/math_projected_half_tmp"
        os.makedirs(save_path, exist_ok=True)
        for name, module in target_modules(model, config):
            assert '-' not in name
            file_name = name.replace('.', '-')
            if 'q_proj' not in name and 'k_proj' not in name:
                continue

            if save_path and os.path.exists(f"{save_path}/{file_name}.pt"):
                print(f"Jacobian file for {save_path}/{name} already exists, loading.")
                
            print(f"Calculating jacobian for {save_path}/{name} ...")
            for param in model.parameters():
                param.requires_grad = False
            module.weight.requires_grad = True

            grads_w_b = []
            grads_w_a = []
            norms = []
            maxs = []
            ratios = []
            for data in tqdm(data_loader):
                data = {k: v.to(model.device) for k, v in data.items()}
                model.zero_grad()
                outputs = model(**data)
                outputs.loss.backward()
                assert module.weight.grad is not None

                U, S, V = torch.svd_lowrank(module.weight.grad, q=self.r_jac_approx)
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

            BB = torch.einsum('pkr, qks -> pqrs', jac_w_b, jac_w_b)
            AA = torch.einsum('prm, qsm -> pqrs', jac_w_a, jac_w_a)
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

    def calc_all_jac(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path)
        for dataset_name in self.knowledge_dataset:
            knowledge_data_loader = get_knowledge_data(
                name=dataset_name, 
                tokenizer=tokenizer, 
                model_id=self.config.base_model_name_or_path, 
                nsamples=self.n_knowledge_samples, 
                seed=self.seed
            )
            self._calc_jac(
                model=self.model, 
                config=self.config, 
                data_loader=knowledge_data_loader,
            )

    

if __name__ == "__main__":
    model_path = "/home/fit/lishbo/WORK/zzl/repo/peft/output/metamath-pissa-llama-2-7b-r128/projected_half"
    config_path = "/home/fit/lishbo/WORK/zzl/repo/peft/output/metamath-pissa-llama-2-7b-r128/checkpoint-782"
    config = PeftConfig.from_pretrained(config_path)
    projector = Projector(
        model_path=model_path,
        config=config,
        knowledge_dataset=["nqopen"],
        n_knowledge_samples=256,
        r_jac_approx=28,
        seed=233,
    )
    projector.calc_all_jac()
    # projected_model = projector.project()
    # projected_model = projected_model.merge_and_unload().to(torch.bfloat16)
    # projected_model.save_pretrained(f"{os.path.dirname(peft_adapter_path)}/projected")