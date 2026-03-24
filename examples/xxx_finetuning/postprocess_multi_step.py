import os
import json
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import concurrent
import psutil
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
MAX_WORKERS = min(16, (os.cpu_count() or 4) + 4)

def monitor_gpu_vram(step=""):
    if not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    prefix = f"[{step}] " if step else ""
    print(f"{prefix}GPU {device} | 已分配: {allocated:.2f} GB | 峰值: {max_allocated:.2f} GB | 缓存池保留: {reserved:.2f} GB")


def monitor_cpu_ram(step=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    sys_mem = psutil.virtual_memory()
    sys_mem_percent = sys_mem.percent
    sys_mem_avail_gb = sys_mem.available / (1024 ** 3)
    cpu_percent = psutil.cpu_percent(interval=None)
    prefix = f"[{step}] " if step else ""
    print(f"{prefix}PID {process.pid} | 当前进程 RAM: {mem_mb:.2f} MB | 系统可用 RAM: {sys_mem_avail_gb:.2f} GB ({sys_mem_percent}% 已用) | CPU: {cpu_percent}%")


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

def target_modules(model: nn.Module, config: LoraConfig) -> Iterable[nn.Module]:
    if isinstance(model, PeftModel):
        model = model.get_base_model()
    for name, module in model.named_modules():
        if LoraModel._check_target_module_exists(config, name) and isinstance(module, (Linear, LoraLinear)):
            yield name, module

@torch.no_grad()
def svd_compress_features(a, b, r=32):
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
        
    return new_a_T, new_b_T
    

class GlobalJacobianFreeProjector:
    def __init__(
        self,
        peft_adapter_path: str,
        r_jac_approx: int = 32,
        delta_threshold: float = 0.,
        beta: float = 0.9,
    ):
        self.config = PeftConfig.from_pretrained(peft_adapter_path)
        self.r_jac_approx = r_jac_approx

        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device) 
        else:
            self.device = torch.device("cuda:0")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "/home/fit/lishbo/WORK/data/zhengzhilong/anticf/pissa_residual_model/Llama-2-7b-hf",
            dtype=torch.float32,
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
            dtype=torch.float32,
            device_map={"": self.device}
        )
        self.model.gradient_checkpointing_enable()
        
        # CPU 内存常驻：全量缓存 delta_w
        self.delta_w_dict = {}
        for name, module in target_modules(peft_model, self.config):
            self.delta_w_dict[name] = (module.weight.data - self.model.get_submodule(name).weight.data).clone().cpu()

        torch.cuda.empty_cache()
        self.delta_threshold = delta_threshold
        self.beta = beta
        
        # 碎片化落盘目录准备
        self.jac_dir_old = f"{os.path.dirname(peft_adapter_path)}/jac_old"
        self.jac_dir_new = f"{os.path.dirname(peft_adapter_path)}/jac_new"

    @staticmethod
    def _load_single_chunk(b_idx, jac_dir_old, base_name):
        f_a = f"{jac_dir_old}/{base_name}_a_{b_idx}.pt"
        f_b = f"{jac_dir_old}/{base_name}_b_{b_idx}.pt"
        # 直接返回元组
        return (
            torch.load(f_a, map_location="cpu", weights_only=True),
            torch.load(f_b, map_location="cpu", weights_only=True)
        )

    def _get_global_components(self, dataloader, target_jac_dir, calculate_y=True):
        m_total = len(dataloader.dataset)
        device = self.device
        
        G_global = torch.zeros((m_total, m_total), device=device, dtype=torch.float64)
        y_global = torch.zeros((m_total,), device=device, dtype=torch.float64) if calculate_y else None
        
        jac_w_cache = {
            name: {'a': [], 'b': []} 
            for name, _ in target_modules(self.model, self.config)
        }
        hooks = []
        
        for name, module in target_modules(self.model, self.config):
            module.weight.requires_grad = True
            
            def make_hooks(layer_name):
                fwd_activation_cache = []
                def fwd_hook(mod, inp, out):
                    fwd_activation_cache.append(inp[0].detach())
                    
                def bwd_hook(mod, grad_input, grad_output):
                    bwd_activation = grad_output[0].detach()  # Shape: (B, T, d_out)
                    fwd_activation = fwd_activation_cache[-1]  # Shape: (B, T, d_in)
                    
                    fwd_activation_cache.clear()
                    
                    jac_w_a, jac_w_b = svd_compress_features(
                        fwd_activation, 
                        bwd_activation, 
                        r=self.r_jac_approx
                    )  # Shapes: (B, r, d_in), (B, r, d_out)
                    jac_w_a, jac_w_b = jac_w_a.contiguous(), jac_w_b.contiguous()
                    
                    if dist.is_initialized():
                        ws = dist.get_world_size()
                        jac_w_a_gather = [torch.zeros_like(jac_w_a) for _ in range(ws)]
                        jac_w_b_gather = [torch.zeros_like(jac_w_b) for _ in range(ws)]
                        dist.all_gather(jac_w_a_gather, jac_w_a)
                        dist.all_gather(jac_w_b_gather, jac_w_b)
                        # Shape: (num_gpus * B, r, d_in) and (num_gpus * B, r, d_out).
                        jac_w_a = torch.cat(jac_w_a_gather, dim=0)
                        jac_w_b = torch.cat(jac_w_b_gather, dim=0)
                    
                    jac_w_cache[layer_name]['a'].append(jac_w_a.cpu())
                    jac_w_cache[layer_name]['b'].append(jac_w_b.cpu())
                    
                return fwd_hook, bwd_hook
                
            f_hook, b_hook = make_hooks(name)
            hooks.append(module.register_forward_hook(f_hook))
            hooks.append(module.register_full_backward_hook(b_hook))

        count = 0
        for batch_inputs in dataloader:
            count += 1
            print(f"Processing batch {count}/{len(dataloader)}...")
            monitor_gpu_vram()
            monitor_cpu_ram()
            
            # batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
            # self.model.zero_grad(set_to_none=True)
            # outputs = self.model(**batch_inputs)
            # labels = batch_inputs["labels"]
            # valid_token_num = (labels != IGNORE_INDEX).sum().item()
            # loss = outputs.loss * valid_token_num 
            # loss.backward()
            # # outputs.loss.backward()
            # self.model.zero_grad(set_to_none=True)
            
            # # --- 系统内存破局：批次分块落盘 ---
            # for name in jac_w_cache.keys():
            #     if len(jac_w_cache[name]['a']) > 0:
            #         a_batch = torch.cat(jac_w_cache[name]['a'], dim=0)
            #         b_batch = torch.cat(jac_w_cache[name]['b'], dim=0)
                    
            #         torch.save(a_batch, f"{target_jac_dir}/{name.replace('.', '-')}_a_{count}.pt")
            #         torch.save(b_batch, f"{target_jac_dir}/{name.replace('.', '-')}_b_{count}.pt")
                    
            #         # 立即清空缓存防止 OOM
            #         jac_w_cache[name]['a'].clear()
            #         jac_w_cache[name]['b'].clear()

        for h in hooks:
            h.remove()
        for name, module in target_modules(self.model, self.config):
            module.weight.requires_grad = False

        num_batches = count

        # 懒加载：逐层读取碎片拼装 G_global
        G_dict = {}
        y_dict = {}
        for name, _ in target_modules(self.model, self.config):
            print(f"Calculating G_global for {name}")
            monitor_gpu_vram()
            monitor_cpu_ram()
            

            a_chunks, b_chunks = [], []
            # for b_idx in range(1, num_batches + 1):
            #     f_a = f"{target_jac_dir}/{name.replace('.', '-')}_a_{b_idx}.pt"
            #     f_b = f"{target_jac_dir}/{name.replace('.', '-')}_b_{b_idx}.pt"
            #     a_chunks.append(torch.load(f_a, map_location="cpu", weights_only=True))
            #     b_chunks.append(torch.load(f_b, map_location="cpu", weights_only=True))
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # executor.map 会并发执行，并且保证返回的迭代器顺序与输入的 range 顺序完全一致
                results = executor.map(
                    lambda b_idx: GlobalJacobianFreeProjector._load_single_chunk(
                        b_idx, 
                        self.jac_dir_old, 
                        name.replace('.', '-')
                    ), 
                    range(1, num_batches + 1)
                )
                
                # 拆解解包结果
                for a, b in results:
                    a_chunks.append(a)
                    b_chunks.append(b)
                
            jac_w_a = torch.cat(a_chunks, dim=0).to(self.device)
            jac_w_b = torch.cat(b_chunks, dim=0).to(self.device)
            
            m, r = jac_w_a.shape[0], jac_w_a.shape[1]
            
            jac_w_a_flat = jac_w_a.view(m * r, -1).double()
            jac_w_b_flat = jac_w_b.view(m * r, -1).double()
            
            Ka = jac_w_a_flat @ jac_w_a_flat.T 
            Kb = jac_w_b_flat @ jac_w_b_flat.T
            
            #* G = J @ J^T
            G_local = (Ka * Kb).view(m, r, m, r).sum(dim=(1, 3))  # Gram matrix of size (m, m) for this layer
            # G_global += G_local
            G_dict[name] = G_local.cpu()  # 存入 CPU 内存字典
            
            if calculate_y:
                #* y = J @ delta_w
                delta_w = self.delta_w_dict[name].to(self.device, dtype=torch.float64, non_blocking=True)
                y_local = ((jac_w_a_flat @ delta_w.T) * jac_w_b_flat).view(m, r, -1).sum(dim=(1, 2))
                # y_global += y_local
                y_dict[name] = y_local.cpu()  # 存入 CPU 内存字典
                
            del jac_w_a, jac_w_b, jac_w_a_flat, jac_w_b_flat, Ka, Kb, a_chunks, b_chunks
            torch.cuda.empty_cache()
            
        # return num_batches, G_global, y_global
        return num_batches, G_dict, y_dict

    @staticmethod
    @torch.no_grad()
    def _compute_principal_angle(jac_dir_old, jac_dir_new, num_batches, names, G_global_0, G_global_1, eps=1e-10):
        m = G_global_0.shape[0]
        device = G_global_0.device
        C_global = torch.zeros((m, m), device=device, dtype=torch.float64)
        
        # --- 算子生命周期融合：流式计算并当场销毁 ---
        for name in names:
            print(f"Computing cross term for {name}...")
            a_chunks_0, b_chunks_0 = [], []
            a_chunks_1, b_chunks_1 = [], []
            # for b_idx in range(1, num_batches + 1):
            #     f_a_0 = f"{jac_dir_old}/{name.replace('.', '-')}_a_{b_idx}.pt"
            #     f_b_0 = f"{jac_dir_old}/{name.replace('.', '-')}_b_{b_idx}.pt"
            #     f_a_1 = f"{jac_dir_new}/{name.replace('.', '-')}_a_{b_idx}.pt"
            #     f_b_1 = f"{jac_dir_new}/{name.replace('.', '-')}_b_{b_idx}.pt"
                
            #     a_chunks_0.append(torch.load(f_a_0, map_location="cpu", weights_only=True))
            #     b_chunks_0.append(torch.load(f_b_0, map_location="cpu", weights_only=True))
            #     a_chunks_1.append(torch.load(f_a_1, map_location="cpu", weights_only=True))
            #     b_chunks_1.append(torch.load(f_b_1, map_location="cpu", weights_only=True))
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # executor.map 会并发执行，并且保证返回的迭代器顺序与输入的 range 顺序完全一致
                results = executor.map(
                    lambda b_idx: GlobalJacobianFreeProjector._load_single_chunk(
                        b_idx, 
                        jac_dir_old, 
                        name.replace('.', '-')
                    ), 
                    range(1, num_batches + 1)
                )
                
                # 拆解解包结果
                for a, b in results:
                    a_chunks_0.append(a)
                    b_chunks_0.append(b)
                
                results = executor.map(
                    lambda b_idx: GlobalJacobianFreeProjector._load_single_chunk(
                        b_idx, 
                        jac_dir_new, 
                        name.replace('.', '-')
                    ), 
                    range(1, num_batches + 1)
                )
                
                # 拆解解包结果
                for a, b in results:
                    a_chunks_1.append(a)
                    b_chunks_1.append(b)
                
            jac_w_a_0 = torch.cat(a_chunks_0, dim=0).to(device)
            jac_w_b_0 = torch.cat(b_chunks_0, dim=0).to(device)
            jac_w_a_1 = torch.cat(a_chunks_1, dim=0).to(device)
            jac_w_b_1 = torch.cat(b_chunks_1, dim=0).to(device)
            
            AA_cross = torch.einsum('mti, nsi -> mnts', jac_w_a_0.double(), jac_w_a_1.double())
            BB_cross = torch.einsum('mto, nso -> mnts', jac_w_b_0.double(), jac_w_b_1.double())
            C_global += (AA_cross * BB_cross).sum(dim=(2, 3))
            
            # 阅后即焚，节省显存
            del jac_w_a_0, jac_w_b_0, jac_w_a_1, jac_w_b_1, AA_cross, BB_cross
            torch.cuda.empty_cache()
        
        eig0, L0 = torch.linalg.eigh(G_global_0.double())
        eig1, L1 = torch.linalg.eigh(G_global_1.double())
        
        mask0 = eig0 > eps
        eig0_inv_sqrt = torch.zeros_like(eig0)
        eig0_inv_sqrt[mask0] = 1.0 / torch.sqrt(eig0[mask0])
        S0 = L0 * eig0_inv_sqrt.unsqueeze(0)  
        
        mask1 = eig1 > eps
        eig1_inv_sqrt = torch.zeros_like(eig1)
        eig1_inv_sqrt[mask1] = 1.0 / torch.sqrt(eig1[mask1])
        S1 = L1 * eig1_inv_sqrt.unsqueeze(0)

        M = S0.T @ C_global @ S1
        cosines = torch.linalg.svdvals(M)
        cosines = torch.clamp(cosines, min=0.0, max=1.0)
        return cosines.mean().item()

    def dynamic_global_manifold_projection_optimized(self, dataloader):
        # 初始化清空并建立临时文件夹
        # shutil.rmtree(self.jac_dir_old, ignore_errors=True)
        # shutil.rmtree(self.jac_dir_new, ignore_errors=True)
        os.makedirs(self.jac_dir_old, exist_ok=True)
        os.makedirs(self.jac_dir_new, exist_ok=True)
        
        names = [n for n, _ in target_modules(self.model, self.config)]
        
        # 首次计算基础流形状态
        # num_batches, G_global, y_global = self._get_global_components(dataloader, self.jac_dir_old, calculate_y=True)
        num_batches, G_dict, y_dict = self._get_global_components(dataloader, self.jac_dir_old, calculate_y=True)
        
        while True:
            # G_inv = torch.linalg.pinv(G_global.double(), rcond=1e-10)
            # v_global = (G_inv @ y_global.double()).to(G_global.dtype)
            
            # --- 算力复用：外层提取预计算全网所有的 P ---
            print("Precomputing P for all layers...")
            P_dict = {}
            for name in names:
                print(f"Precomputing P for {name}...")
                G_inv = torch.linalg.pinv(G_dict[name].double(), rcond=1e-10)
                v = (G_inv @ y_dict[name].double()).to(G_dict[name].dtype)
                a_chunks, b_chunks = [], []
                # for b_idx in range(1, num_batches + 1):
                #     f_a = f"{self.jac_dir_old}/{name.replace('.', '-')}_a_{b_idx}.pt"
                #     f_b = f"{self.jac_dir_old}/{name.replace('.', '-')}_b_{b_idx}.pt"
                #     a_chunks.append(torch.load(f_a, map_location="cpu", weights_only=True))
                #     b_chunks.append(torch.load(f_b, map_location="cpu", weights_only=True))
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # executor.map 会并发执行，并且保证返回的迭代器顺序与输入的 range 顺序完全一致
                    results = executor.map(
                        lambda b_idx: GlobalJacobianFreeProjector._load_single_chunk(
                            b_idx, 
                            self.jac_dir_old, 
                            name.replace('.', '-')
                        ), 
                        range(1, num_batches + 1)
                    )
                    
                    # 拆解解包结果
                    for a, b in results:
                        a_chunks.append(a)
                        b_chunks.append(b)
                    
                jac_w_a = torch.cat(a_chunks, dim=0).to(self.device)
                jac_w_b = torch.cat(b_chunks, dim=0).to(self.device)
                # v_local = v_global.to(device=self.device, dtype=jac_w_a.dtype)
                v = v.to(device=self.device, dtype=jac_w_a.dtype)
                
                # P = torch.einsum('m, mto, mti -> oi', v_local, jac_w_b, jac_w_a)
                P = torch.einsum('m, mto, mti -> oi', v, jac_w_b, jac_w_a)
                P_dict[name] = P.cpu()  # 存入 CPU 内存字典
                monitor_cpu_ram()
                
                torch.cuda.empty_cache()
            
            alpha = 1.0
            
            while True:
                # 使用内存中的 P_dict 和 delta_W_dict，完全 0 读盘
                for name, module in target_modules(self.model, self.config):
                    delta_w = self.delta_w_dict[name].to(self.device)
                    P = P_dict[name].to(self.device)
                    
                    delta_w_projed = alpha * (delta_w - P)
                    updated_weight = module.weight.data + delta_w_projed
                    swallowed_ratio = ((delta_w_projed == delta_w) & (P.abs() > 1e-12)).float().mean().item()
                    print(f"{name}, P norm={P.norm().item():.8f}")
                    print(f"{name} alpha={alpha:.4f}, swallowed_ratio={swallowed_ratio:.10f}")
                    module.weight.data.copy_(updated_weight)
                break
                
                # 重新计算新权重下的流形空间 (写入 jac_dir_new)
                _, G_global_new, y_global_new = self._get_global_components(dataloader, self.jac_dir_new, calculate_y=True)
                
                mean_cos = self._compute_principal_angle(
                    self.jac_dir_old, self.jac_dir_new, num_batches, names, G_global, G_global_new
                )
                print(f"  [Inner] alpha={alpha:.4f}, global_mean_cos={mean_cos:.4f}")
                monitor_gpu_vram()
                monitor_cpu_ram()
                
                if mean_cos > self.delta_threshold:
                    print(f"Global projection converged with alpha={alpha:.4f} and mean_cos={mean_cos:.4f}.")
                    
                    # 收敛后更新驻留在内存里的 delta_W
                    for name in names:
                        delta_w = self.delta_w_dict[name]
                        P = P_dict[name]
                        self.delta_w_dict[name] = delta_w.mul_(1 - alpha).add_(P, alpha=alpha)
                        
                    if dist.is_initialized():
                        dist.barrier()
                    break
                else:
                    print(f"  [Rollback] mean_cos 校验失败，纯显存回滚 alpha={alpha:.4f} 的权重...")
                    # --- 纯内存高速回滚 ---
                    for name, module in target_modules(self.model, self.config):
                        delta_w = self.delta_w_dict[name].to(self.device)
                        P = P_dict[name].to(self.device)
                        delta_w_projed = alpha * (delta_w - P)
                        module.weight.data.sub_(delta_w_projed)
                    alpha *= self.beta
                    
            if alpha == 1.0:
                print("全局投影收敛，流形追踪完成！")
                # shutil.rmtree(self.jac_dir_old, ignore_errors=True)
                # shutil.rmtree(self.jac_dir_new, ignore_errors=True)
                break
            else:
                print("状态切换，翻转硬盘文件指针...")
                shutil.rmtree(self.jac_dir_old, ignore_errors=True)
                os.rename(self.jac_dir_new, self.jac_dir_old)
                os.makedirs(self.jac_dir_new, exist_ok=True)
                
                # 直接继承状态，跳过下一次外层前向传播
                G_global = G_global_new
                y_global = y_global_new
                
        return self.model


if __name__ == "__main__":
    # 底部执行入口的逻辑与您提供的一致
    base_model_path = "meta-llama/Llama-2-7b-hf"
    peft_adapter_path = "/home/fit/lishbo/WORK/zzl/repo/peft/output/metamath-pissa-llama-2-7b/checkpoint-782"
    save_path = f"{os.path.dirname(peft_adapter_path)}/projected_multi_step"
    
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
    
    projector = GlobalJacobianFreeProjector(
        peft_adapter_path=peft_adapter_path,
    )
    projected_model = projector.dynamic_global_manifold_projection_optimized(dataloader)
    
    projected_model = projected_model.to(torch.bfloat16)
    projected_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    if dist.is_initialized():
        dist.destroy_process_group()