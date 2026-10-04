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
        # input_ids, labels = tuple([instance[key].squeeze(0) for instance in instances] for key in ("input_ids", "labels"))
        # input_ids = [torch.tensor(x) for x in input_ids]
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        # )
        # labels = [torch.tensor(x) for x in labels]
        # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # return {
        #     "input_ids": input_ids,
        #     "labels": labels,
        #     "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        # }
    
        input_ids = [torch.tensor(instance["input_ids"].squeeze(0)) for instance in instances]
        labels = [torch.tensor(instance["labels"].squeeze(0)) for instance in instances]

        lengths = [len(seq) for seq in input_ids]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, l in enumerate(lengths):
            attention_mask[i, :l] = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
            

class GlobalJacobianFreeProjector:
    def __init__(
        self,
        original_model,
        finetuned_model,
        target_layers: Sequence[str],
        cache_dir: str,
        r_jac_approx: int = 32,
        delta_threshold: float = 0.95,
        beta: float = 0.7,
        chunk_size: int = 56
    ):
        self.r_jac_approx = r_jac_approx
        self.chunk_size = chunk_size

        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device) 
        else:
            self.device = torch.device("cuda:0")
        
        self.model = original_model.to(self.device)
        self.model.gradient_checkpointing_enable()
        finetuned_model = finetuned_model.to(self.device)
        self.target_layers = target_layers
        
        self.delta_w_dict = {}
        for name in target_layers:
            self.delta_w_dict[name] = (
                finetuned_model.get_submodule(name).weight.data - self.model.get_submodule(name).weight.data
            ).clone().cpu()

        torch.cuda.empty_cache()
        self.delta_threshold = delta_threshold
        self.beta = beta
        
        self.jac_dir_old = f"{cache_dir}/jac_old"
        self.jac_dir_new = f"{cache_dir}/jac_new"

        self.Gram_old = None
        self.P_old = None
        self.Gram_new = None
        self.P_new = None
        self.alpha = 1.0

    def _chunk_cache_file(self, target_dir: str, chunk_idx: int) -> str:
        return f"{target_dir}/chunk_{chunk_idx:04d}.safetensors"

    def _get_projection_components(self, dataloader, target_jac_dir, for_landing_point=False, global_proj=False):
        print("Calculating mean gradients and projection components...")
        assert not global_proj, "Global projection is not currently supported in this implementation."
        sample_count = len(dataloader.dataset)
        Gram = {}
        P = {}

        all_modules = [(name, self.model.get_submodule(name)) for name in self.target_layers]
        module_chunks = [all_modules[i:i + self.chunk_size] for i in range(0, len(all_modules), self.chunk_size)]

        for name, module in all_modules:
            module.weight.requires_grad = False

        for chunk_idx, chunk in enumerate(module_chunks):
            print(f"\n=== Processing Layer Chunk {chunk_idx + 1}/{len(module_chunks)} ===")
            grad_sum = {}
            chunk_mean_grads = {}
            for name, module in chunk:
                module.weight.requires_grad = True
                grad_sum[name] = torch.zeros_like(module.weight.data, dtype=torch.float32)

            print("Accumulating gradients ...")
            count = 0
            for batch_inputs in dataloader:
                count += 1
                if count % 8 == 0:
                    print(f"  Batch {count}/{len(dataloader)}...")
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                self.model.zero_grad(set_to_none=True)
                outputs = self.model(**batch_inputs)
                loss = outputs.loss
                loss.backward()

                for name, module in chunk:
                    if module.weight.grad is not None:
                        grad_sum[name].add_(module.weight.grad.detach().to(torch.float32))
                self.model.zero_grad(set_to_none=True)

            for name, module in chunk:
                module.weight.requires_grad = False

            for name, _ in chunk:
                grad_tensor = grad_sum[name].to(self.device)
                if dist.is_initialized():
                    dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM)
                mean_grad_cpu = (grad_tensor / sample_count).to(dtype=torch.float32, device="cpu")
                chunk_mean_grads[name] = mean_grad_cpu
                Gram[name] = float((mean_grad_cpu * mean_grad_cpu).sum().item())

                if for_landing_point:
                    delta_w = self.delta_w_dict[name].to(self.device, dtype=torch.float32, non_blocking=True)
                    P_old = self.P_old[name].to(self.device, dtype=torch.float32, non_blocking=True)
                    delta_w = (1 - self.alpha) * delta_w + self.alpha * P_old
                else:
                    delta_w = self.delta_w_dict[name].to(self.device, dtype=torch.float32, non_blocking=True)

                mean_grad = mean_grad_cpu.to(self.device, dtype=torch.float32, non_blocking=True)
                grad_norm_sq = torch.sum(mean_grad * mean_grad).clamp_min(1e-12)
                coeff = torch.sum(delta_w * mean_grad) / grad_norm_sq
                P[name] = (coeff * mean_grad).to(dtype=torch.float32).cpu()
                del grad_sum[name], mean_grad, mean_grad_cpu
                torch.cuda.empty_cache()
            save_file(chunk_mean_grads, self._chunk_cache_file(target_jac_dir, chunk_idx))
            del chunk_mean_grads

        return Gram, P

    @torch.no_grad()
    def _compute_principal_angle(self, jac_dir_old, jac_dir_new, names, eps=1e-10):
        cosines = {}
        name_chunks = [names[i:i + self.chunk_size] for i in range(0, len(names), self.chunk_size)]
        for chunk_idx, name_chunk in enumerate(name_chunks):
            file_old = self._chunk_cache_file(jac_dir_old, chunk_idx)
            file_new = self._chunk_cache_file(jac_dir_new, chunk_idx)
            grads_old = load_file(file_old)
            grads_new = load_file(file_new)
            for name in name_chunk:
                g0 = grads_old[name].to(dtype=torch.float32)
                g1 = grads_new[name].to(dtype=torch.float32)
                denom = (g0.norm() * g1.norm()).clamp_min(eps)
                cos = torch.abs(torch.sum(g0 * g1) / denom).item()
                cosines[name] = cos
                print(f"  {name} mean-grad cosine={cos:.4f}")
        return sum(cosines.values()) / len(cosines)

    def dynamic_global_manifold_projection_optimized(self, dataloader, global_proj=False):
        print("Clearing old gradient cache and preparing directories...")
        shutil.rmtree(self.jac_dir_old, ignore_errors=True)
        shutil.rmtree(self.jac_dir_new, ignore_errors=True)
        os.makedirs(self.jac_dir_old, exist_ok=True)
        os.makedirs(self.jac_dir_new, exist_ok=True)
        
        self.Gram_old, self.P_old = self._get_projection_components(dataloader, self.jac_dir_old, global_proj=global_proj)
        
        while True:
            #* Searching for proper alpha with rollback mechanism, starting with a full step (alpha=1.0)
            self.alpha = 1.0
            while True:
                #* Applying projected update with the step length of alpha to the model weights
                for name in self.target_layers:
                    module = self.model.get_submodule(name)
                    delta_w = self.delta_w_dict[name].to(self.device)
                    P = self.P_old[name].to(self.device)
                    
                    delta_w_projed = self.alpha * (delta_w - P)
                    updated_weight = module.weight.data + delta_w_projed
                    swallowed_ratio1 = (((delta_w - P) == delta_w) & (P.abs() > 1e-12)).float().mean().item()
                    swallowed_ratio2 = (module.weight.data == updated_weight).float().mean().item()
                    print(f"{name}, P norm={P.norm().item():.8f}")
                    print(f"{name}, delta W norm={delta_w.norm().item():.8f}")
                    print(f"{name}, delta W projed norm={delta_w_projed.norm().item():.8f}")
                    print(f"{name}, W norm={module.weight.data.norm().item():.8f}")
                    print(f"{name} alpha={self.alpha:.4f}, swallowed_ratio1={swallowed_ratio1:.10f}, swallowed_ratio2={swallowed_ratio2:.10f}")
                    module.weight.data.copy_(updated_weight)
                
                # # #* Calculating new global components with the updated model
                # self.Gram_new, self.P_new = self._get_projection_components(dataloader, self.jac_dir_new, for_landing_point=True, global_proj=global_proj)
                
                # mean_cos = self._compute_principal_angle(
                #     self.jac_dir_old,
                #     self.jac_dir_new,
                #     self.target_layers, 
                # )
                mean_cos = 1
                print(f"  [Inner] alpha={self.alpha:.4f}, global_mean_cos={mean_cos:.4f}")
                
                if mean_cos > self.delta_threshold:
                    print(f"Global projection converged with alpha={self.alpha:.4f} and mean_cos={mean_cos:.4f}.")            
                    if dist.is_initialized():
                        dist.barrier()
                    break
                else:
                    print(f"  [Rollback] mean_cos 校验失败，纯显存回滚 alpha={self.alpha:.4f} 的权重...")
                    for name in self.target_layers:
                        module = self.model.get_submodule(name)
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

                for name in self.target_layers:
                    delta_w = self.delta_w_dict[name]
                    P = self.P_old[name]
                    self.delta_w_dict[name] = (1 - self.alpha) * delta_w + self.alpha * P
                self.Gram_old = self.Gram_new
                self.P_old = self.P_new
                
        return self.model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str)
    parser.add_argument("--finetuned_model_path", type=str)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    print(f"Projected model will be saved to {args.save_path}")
    
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    tokenizer_config = json.load(open(os.path.join(args.finetuned_model_path, "tokenizer_config.json"), "r"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_id,
        model_max_length=tokenizer_config["model_max_length"],
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    knowledge_dataset = get_knowledge_data(
        name="nqopen", 
        tokenizer=tokenizer, 
        model_id=args.base_model_id, 
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
    
    print("Loading original and finetuned models...")
    # pissa
    pissa_residual_model = AutoModelForCausalLM.from_pretrained(
        # "/home/fit/lishbo/WORK/data/zhengzhilong/anticf/pissa_residual_model/Meta-Llama-3-8B",
        "/home/fit/lishbo/WORK/data/zhengzhilong/anticf/pissa_residual_model/Llama-2-7b-hf",
        dtype=torch.float16,
    )
    finetuned_model = PeftModel.from_pretrained(
        pissa_residual_model,
        args.finetuned_model_path,
        dtype=torch.float32,
    )
    finetuned_model = finetuned_model.merge_and_unload()

    # lora
    # origin_model = AutoModelForCausalLM.from_pretrained(
    #     args.base_model_id,
    #     dtype=torch.float16,
    #     # device_map={"": self.device}
    # )
    # finetuned_model = PeftModel.from_pretrained(
    #     origin_model,
    #     args.finetuned_model_path,
    #     dtype=torch.float32,
    # )
    # finetuned_model = finetuned_model.merge_and_unload()

    # full fine-tune
    # finetuned_model = AutoModelForCausalLM.from_pretrained(
    #     args.finetuned_model_path,
    #     dtype=torch.float16,
    #     # device_map={"": self.device}
    # )
    
    origin_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        dtype=torch.float16,
    )
    target_layers = [name for name, module in origin_model.named_modules() if isinstance(module, (Linear)) and "lm_head" not in name]
    print(f"Target layers for projection: {target_layers}")

    print("Initializing GlobalJacobianFreeProjector...")
    projector = GlobalJacobianFreeProjector(
        original_model=origin_model,
        finetuned_model=finetuned_model,
        target_layers=target_layers,
        cache_dir=f"{args.finetuned_model_path}/jac_cache",
        delta_threshold=args.threshold,
        beta=args.beta,
    )
    print("Starting dynamic global manifold projection...")
    projected_model = projector.dynamic_global_manifold_projection_optimized(dataloader)
    
    projected_model = projected_model.to(torch.bfloat16)
    projected_model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"Projected model and tokenizer saved to {args.save_path}.")

    if dist.is_initialized():
        dist.destroy_process_group()
