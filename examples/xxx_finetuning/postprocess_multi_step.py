import os
import json
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from safetensors.torch import save_file, load_file
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
from torch.nn import Linear
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft.peft_model import PeftModel
from peft.tuners.lora.model import LoraModel
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.config import LoraConfig
from peft.config import PeftConfig
import transformers

from datautils import get_knowledge_data

IGNORE_INDEX = -100


def target_modules(model: nn.Module, config: LoraConfig) -> Iterable[nn.Module]:
    if isinstance(model, PeftModel):
        model = model.get_base_model()
    for name, module in model.named_modules():
        if LoraModel._check_target_module_exists(config, name) and isinstance(module, (Linear, LoraLinear)):
            yield name, module


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key].squeeze(0) for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
    

class GlobalJacobianFreeProjector:
    def __init__(
        self,
        peft_adapter_path: str,
        r_jac_approx: int = 32,
        delta_threshold: float = 0.95,
        beta: float = 0.7,
        chunk_size: int = 56
    ):
        self.config = PeftConfig.from_pretrained(peft_adapter_path)
        self.r_jac_approx = r_jac_approx
        self.chunk_size = chunk_size

        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device) 
        else:
            self.device = torch.device("cuda:0")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "/home/fit/lishbo/WORK/data/zhengzhilong/anticf/pissa_residual_model/Llama-2-7b-hf",
            dtype=torch.float16,
            device_map={"": self.device}
        )
        peft_model = PeftModel.from_pretrained(
            self.model,
            peft_adapter_path,
            dtype=torch.float32,
        )
        peft_model = peft_model.merge_and_unload()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name_or_path,
            dtype=torch.float16,
            device_map={"": self.device}
        )
        self.model.gradient_checkpointing_enable()
        
        self.delta_w_dict = {}
        for name, module in target_modules(peft_model, self.config):
            self.delta_w_dict[name] = (module.weight.data - self.model.get_submodule(name).weight.data).clone().cpu()

        torch.cuda.empty_cache()
        self.delta_threshold = delta_threshold
        self.beta = beta
        
        # 碎片化落盘目录准备
        self.jac_dir_old = f"{os.path.dirname(peft_adapter_path)}/jac_old"
        self.jac_dir_new = f"{os.path.dirname(peft_adapter_path)}/jac_new"

        self.Gram_old = None
        self.P_old = None
        self.Gram_new = None
        self.P_new = None
        self.alpha = 1.0

    @torch.no_grad()
    def _svd_compress_features(self, a, b, r=32):
        m, T, d_in = a.shape
        _, _, d_out = b.shape
        actual_r = min(r, T, d_in, d_out)  
        
        Q_b, R_b = torch.linalg.qr(b.transpose(1, 2))
        Q_a, R_a = torch.linalg.qr(a.transpose(1, 2))
        M = torch.bmm(R_b, R_a.transpose(1, 2))
        U_M, S, V_M_h = torch.linalg.svd(M)
        
        U_M_r = U_M[:, :, :actual_r]                
        S_r = S[:, :actual_r]                      
        V_M_r = V_M_h.transpose(1, 2)[:, :, :actual_r] 
        S_sqrt = torch.diag_embed(torch.sqrt(S_r))
        
        new_b_T = torch.bmm(Q_b, torch.bmm(U_M_r, S_sqrt)).transpose(1, 2)
        new_a_T = torch.bmm(Q_a, torch.bmm(V_M_r, S_sqrt)).transpose(1, 2)
        
        if actual_r < r:
            pad_a = torch.zeros(m, r - actual_r, d_in, device=a.device, dtype=a.dtype)
            pad_b = torch.zeros(m, r - actual_r, d_out, device=b.device, dtype=b.dtype)
            new_a_T = torch.cat([new_a_T, pad_a], dim=1)
            new_b_T = torch.cat([new_b_T, pad_b], dim=1)
            
        return new_a_T.to(torch.float16), new_b_T.to(torch.float16)

    def _get_projection_components(self, dataloader, target_jac_dir, for_landing_point=False, global_proj=False):
        print("Calculating Jacobian and projection components...")
        assert not global_proj, "Global projection is not currently supported in this implementation."
        m_total = len(dataloader.dataset)
        device = self.device

        if global_proj:
            #* G_global = sum_i [G_i]
            #* Jdw_global = sum_i [J_i@dw_i]
            Gram = torch.zeros((m_total, m_total), device=device, dtype=torch.float64)
            Jdw = torch.zeros((m_total,), device=device, dtype=torch.float64)
        else:
            Gram = {}
        #* Projection correction term P for each layer
        P = {}

        all_modules = list(target_modules(self.model, self.config))
        module_chunks = [all_modules[i:i + self.chunk_size] for i in range(0, len(all_modules), self.chunk_size)]
        
        for chunk_idx, chunk in enumerate(module_chunks):
            print(f"\n=== Processing Layer Chunk {chunk_idx + 1}/{len(module_chunks)} ===")
            
            jac_w_cache = {name: {'a': [], 'b': []} for name, _ in chunk}
            hooks = []
            
            for name, module in chunk:
                module.weight.requires_grad = True
                
                def make_hooks(layer_name):
                    fwd_activation_cache = []
                    def fwd_hook(mod, inp, out):
                        fwd_activation_cache.append(inp[0].detach())
                        
                    def bwd_hook(mod, grad_input, grad_output):
                        bwd_activation = grad_output[0].detach()  # Shape: (B, T, d_out)
                        fwd_activation = fwd_activation_cache[-1]  # Shape: (B, T, d_in)
                        fwd_activation_cache.clear()
                        
                        jac_w_a, jac_w_b = self._svd_compress_features(
                            fwd_activation.float(), bwd_activation.float(), r=self.r_jac_approx
                        )  
                        jac_w_a, jac_w_b = jac_w_a.contiguous(), jac_w_b.contiguous()
                        
                        if dist.is_initialized():
                            ws = dist.get_world_size()
                            jac_w_a_gather = [torch.zeros_like(jac_w_a) for _ in range(ws)]
                            jac_w_b_gather = [torch.zeros_like(jac_w_b) for _ in range(ws)]
                            dist.all_gather(jac_w_a_gather, jac_w_a)
                            dist.all_gather(jac_w_b_gather, jac_w_b)
                            jac_w_a = torch.cat(jac_w_a_gather, dim=0)
                            jac_w_b = torch.cat(jac_w_b_gather, dim=0)
                        
                        jac_w_cache[layer_name]['a'].append(jac_w_a.cpu())
                        jac_w_cache[layer_name]['b'].append(jac_w_b.cpu())
                        
                    return fwd_hook, bwd_hook
                    
                f_hook, b_hook = make_hooks(name)
                hooks.append(module.register_forward_hook(f_hook))
                hooks.append(module.register_full_backward_hook(b_hook))

            print("Calculating Jacobian ...")
            count = 0
            for batch_inputs in dataloader:
                count += 1
                if count % 10 == 0:
                    print(f"  Batch {count}/{len(dataloader)}...")
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                self.model.zero_grad(set_to_none=True)
                outputs = self.model(**batch_inputs)
                labels = batch_inputs["labels"]
                valid_token_num = (labels != IGNORE_INDEX).sum().item()
                loss = outputs.loss * valid_token_num 
                loss.backward()
                self.model.zero_grad(set_to_none=True)

            for h in hooks:
                h.remove()
            for name, module in chunk:
                module.weight.requires_grad = False

            for name, _ in chunk:
                jac_w_a = torch.cat(jac_w_cache[name]['a'], dim=0)
                jac_w_b = torch.cat(jac_w_cache[name]['b'], dim=0)
                
                file_path = f"{target_jac_dir}/{name.replace('.', '-')}.safetensors"
                save_file({"a": jac_w_a, "b": jac_w_b}, file_path)
                
                jac_w_a = jac_w_a.to(self.device)
                jac_w_b = jac_w_b.to(self.device)
                m, r = jac_w_a.shape[0], jac_w_a.shape[1]
                
                jac_w_a_flat = jac_w_a.view(m * r, -1).double()
                jac_w_b_flat = jac_w_b.view(m * r, -1).double()
                
                Ka = jac_w_a_flat @ jac_w_a_flat.T 
                Kb = jac_w_b_flat @ jac_w_b_flat.T
                
                #* G = J @ J^T
                G_local = (Ka * Kb).view(m, r, m, r).sum(dim=(1, 3))  # Shape: (m, m)
                if global_proj:
                    Gram += G_local
                else:
                    Gram[name] = G_local.cpu()
                
                #* y = J @ delta_w
                if for_landing_point:
                    delta_w = self.delta_w_dict[name].to(self.device, dtype=torch.float64, non_blocking=True)
                    P_old = self.P_old[name].to(self.device, dtype=torch.float64, non_blocking=True)
                    delta_w = (1 - self.alpha) * delta_w + self.alpha * P_old
                else:
                    delta_w = self.delta_w_dict[name].to(self.device, dtype=torch.float64, non_blocking=True)
                Jdw_local = ((jac_w_a_flat @ delta_w.T) * jac_w_b_flat).view(m, r, -1).sum(dim=(1, 2)).cpu()  # Shape: (m,)
                if global_proj:
                    Jdw += Jdw_local
                
                if not global_proj:
                    #* P = J^T @ G^-1 @ J @ delta_w
                    G_inv = torch.linalg.pinv(Gram[name].double(), rcond=1e-10)
                    v = (G_inv @ Jdw_local.double()).to(device=self.device)
                    P_local = torch.einsum(
                        'm, mto, mti -> oi', 
                        v, jac_w_b.double(), jac_w_a.double()
                    )  # Shape: (d_out, d_in)
                    P[name] = P_local.cpu()
                
                del jac_w_cache[name]
                del jac_w_a, jac_w_b, jac_w_a_flat, jac_w_b_flat, Ka, Kb
                torch.cuda.empty_cache()
        
        return Gram, P

    @torch.no_grad()
    def _compute_principal_angle(self, jac_dir_old, jac_dir_new, names, Gram_old, Gram_new, eps=1e-10):
        def compute_mean_cosine(Gram1, Gram2, cross_Gram, eps=1e-10):
            eig0, L0 = torch.linalg.eigh(Gram1.double())
            eig1, L1 = torch.linalg.eigh(Gram2.double())

            mask0 = eig0 > eps
            eig0_inv_sqrt = torch.zeros_like(eig0)
            eig0_inv_sqrt[mask0] = 1.0 / torch.sqrt(eig0[mask0])
            S0 = L0 * eig0_inv_sqrt.unsqueeze(0)  
            
            mask1 = eig1 > eps
            eig1_inv_sqrt = torch.zeros_like(eig1)
            eig1_inv_sqrt[mask1] = 1.0 / torch.sqrt(eig1[mask1])
            S1 = L1 * eig1_inv_sqrt.unsqueeze(0)

            M = S0.T @ cross_Gram @ S1
            cosines = torch.linalg.svdvals(M)
            cosines = torch.clamp(cosines, min=0.0, max=1.0)
            return cosines.mean().item()

        global_proj = isinstance(Gram_old, torch.Tensor)
        assert not global_proj, "Global projection is not currently supported in this implementation."
        if global_proj:
            m = Gram_old.shape[0]
            Cross_Gram_global = torch.zeros((m, m), device=self.device, dtype=torch.float64)
        else:
            cosines_dict = {}
        
        for name in names:
            print(f"Computing cross term for {name}...")
            
            file_old = f"{jac_dir_old}/{name.replace('.', '-')}.safetensors"
            file_new = f"{jac_dir_new}/{name.replace('.', '-')}.safetensors"
            
            tensors_0 = load_file(file_old)
            tensors_1 = load_file(file_new)
            
            jac_w_a_0 = tensors_0["a"].to(self.device)  # Shape: (m, r, i)
            jac_w_b_0 = tensors_0["b"].to(self.device)  # Shape: (m, r, o)
            jac_w_a_1 = tensors_1["a"].to(self.device)  # Shape: (m, r, i)
            jac_w_b_1 = tensors_1["b"].to(self.device)  # Shape: (m, r, o)
            
            #* G_{cross} = J1 @ J2^T
            AA_cross = torch.einsum('mti, nsi -> mnts', jac_w_a_0.double(), jac_w_a_1.double())  # Shape: (m, m, r, r)
            BB_cross = torch.einsum('mto, nso -> mnts', jac_w_b_0.double(), jac_w_b_1.double())  # Shape: (m, m, r, r)
            Cross_Gram_local = (AA_cross * BB_cross).sum(dim=(2, 3)).cpu()  # Shape: (m, m)
            
            del jac_w_a_0, jac_w_b_0, jac_w_a_1, jac_w_b_1, AA_cross, BB_cross, tensors_0, tensors_1
            torch.cuda.empty_cache()

            if global_proj:
                Cross_Gram_global += Cross_Gram_local
            else:
                cosines_dict[name] = compute_mean_cosine(Gram_old[name], Gram_new[name], Cross_Gram_local, eps=eps)
                print(f"  {name} mean cosine of principal angles={cosines_dict[name]:.4f}")

        if global_proj:
            mean_cosine = compute_mean_cosine(Gram_old, Gram_new, Cross_Gram_global, eps=eps)
        else:
            mean_cosine = sum(cosines_dict.values()) / len(cosines_dict)
        return mean_cosine

    def dynamic_global_manifold_projection_optimized(self, dataloader, global_proj=False):
        print("Clearing old Jacobian cache and preparing directories...")
        shutil.rmtree(self.jac_dir_old, ignore_errors=True)
        shutil.rmtree(self.jac_dir_new, ignore_errors=True)
        os.makedirs(self.jac_dir_old, exist_ok=True)
        os.makedirs(self.jac_dir_new, exist_ok=True)
        
        names = [n for n, _ in target_modules(self.model, self.config)]
        
        self.Gram_old, self.P_old = self._get_projection_components(dataloader, self.jac_dir_old, global_proj=global_proj)
        
        while True:
            #* Searching for proper alpha with rollback mechanism, starting with a full step (alpha=1.0)
            self.alpha = 1.0
            while True:
                #* Applying projected update with the step length of alpha to the model weights
                for name, module in target_modules(self.model, self.config):
                    delta_w = self.delta_w_dict[name].to(self.device)
                    P = self.P_old[name].to(self.device)
                    
                    delta_w_projed = self.alpha * (delta_w - P)
                    updated_weight = module.weight.data + delta_w_projed
                    swallowed_ratio = (((delta_w - P) == delta_w) & (P.abs() > 1e-12)).float().mean().item()
                    print(f"{name}, P norm={P.norm().item():.8f}")
                    print(f"{name} alpha={self.alpha:.4f}, swallowed_ratio={swallowed_ratio:.10f}")
                    module.weight.data.copy_(updated_weight)
                
                #* Calculating new global components with the updated model
                self.Gram_new, self.P_new = self._get_projection_components(dataloader, self.jac_dir_new, for_landing_point=True, global_proj=global_proj)
                
                mean_cos = self._compute_principal_angle(
                    self.jac_dir_old, 
                    self.jac_dir_new, 
                    names, 
                    self.Gram_old, 
                    self.Gram_new
                )
                print(f"  [Inner] alpha={self.alpha:.4f}, global_mean_cos={mean_cos:.4f}")
                
                if mean_cos > self.delta_threshold:
                    print(f"Global projection converged with alpha={self.alpha:.4f} and mean_cos={mean_cos:.4f}.")            
                    if dist.is_initialized():
                        dist.barrier()
                    break
                else:
                    print(f"  [Rollback] mean_cos 校验失败，纯显存回滚 alpha={self.alpha:.4f} 的权重...")
                    for name, module in target_modules(self.model, self.config):
                        delta_w = self.delta_w_dict[name].to(self.device)
                        P = self.P_old[name].to(self.device)
                        delta_w_projed = self.alpha * (delta_w - P)
                        module.weight.data.sub_(delta_w_projed)
                    self.alpha *= self.beta
                    
            if self.alpha == 1.0:
                print("全局投影收敛，流形追踪完成！")
                shutil.rmtree(self.jac_dir_old, ignore_errors=True)
                shutil.rmtree(self.jac_dir_new, ignore_errors=True)
                break
            else:
                print("状态切换，翻转硬盘文件指针...")
                shutil.rmtree(self.jac_dir_old, ignore_errors=True)
                os.rename(self.jac_dir_new, self.jac_dir_old)
                os.makedirs(self.jac_dir_new, exist_ok=True)

                for name in names:
                    delta_w = self.delta_w_dict[name]
                    P = self.P_old[name]
                    self.delta_w_dict[name] = (1 - self.alpha) * delta_w + self.alpha * P
                self.Gram_old = self.Gram_new
                self.P_old = self.P_new
                
        return self.model


if __name__ == "__main__":
    base_model_path = "meta-llama/Llama-2-7b-hf"
    peft_adapter_path = "/home/fit/lishbo/WORK/zzl/repo/peft/output/metamath-pissa-llama-2-7b/checkpoint-782"
    save_path = f"{os.path.dirname(peft_adapter_path)}/projected_multi_step_thres095_beta07"
    
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    tokenizer_config = json.load(open(os.path.join(peft_adapter_path, "tokenizer_config.json"), "r"))
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        model_max_length=tokenizer_config["model_max_length"],
        padding_side=tokenizer_config["padding_side"],
        use_fast=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_id = PeftConfig.from_pretrained(peft_adapter_path).base_model_name_or_path
    
    knowledge_dataset = get_knowledge_data(
        name="nqopen", 
        tokenizer=tokenizer, 
        model_id=model_id, 
        nsamples=256, 
        seed=233
    )
    data_collector = DataCollatorForSupervisedDataset(tokenizer)

    if dist.is_initialized():
        sampler = DistributedSampler(knowledge_dataset, shuffle=False)
    else:
        sampler = None
    
    dataloader = DataLoader(
        knowledge_dataset,
        batch_size=8,
        sampler=sampler,
        collate_fn=data_collector, 
        drop_last=True,      
        shuffle=False
    )

    print(f"Rank {os.environ.get('LOCAL_RANK', 0)} is processing {len(dataloader)} batches.")
    
    print("Initializing GlobalJacobianFreeProjector...")
    projector = GlobalJacobianFreeProjector(
        peft_adapter_path=peft_adapter_path,
    )
    print("Starting dynamic global manifold projection...")
    projected_model = projector.dynamic_global_manifold_projection_optimized(dataloader)
    
    projected_model = projected_model.to(torch.bfloat16)
    projected_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    if dist.is_initialized():
        dist.destroy_process_group()