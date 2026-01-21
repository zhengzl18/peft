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

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from datautils import get_knowledge_data
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft.tuners.xxx.config import XXXConfig, XXXPreprocessConfig
from peft.tuners.xxx.utils import calculate_jacobian, calculate_stiff_basis, calculate_importance_score

CACHE_ROOT = "/Data2/zhengzhilong"  # peft/examples/corda_finetuning


def run_model(model, knowledge_loader):
    model.train()
    for data in tqdm(knowledge_loader):
        data = {k: v.to(model.device) for k, v in data.items()}
        model.zero_grad()
        outputs = model(**data)
        loss = outputs.loss
        
        if loss is None:
            print(f"Warning: Loss is None for sample {i}. Skipping.")
            continue

        loss.backward()

        current_sample_grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                current_sample_grads[name] = param.grad.clone().detach().cpu()


def main(args):
    # Setting random seed of numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    elif torch.xpu.is_available():
        torch.xpu.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    # Load model
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto"
    )

    mask_path = None
    jacobian_paths = []
    # Get the shared part of preprocess config and xxx config
    preprocess_config = XXXPreprocessConfig(
        n_param_downsample_rate=args.n_param_downsample_rate,
        quantize_stiff_basis=args.quantize_stiff_basis,
    )
    if args.fwd_importance_sampling:
        preprocess_config.fwd_importance_sampling = True
        preprocess_config.fwd_importance_score_path = f"{CACHE_ROOT}/fwd_importance_score/{path_name}"
    if args.bwd_importance_sampling:
        preprocess_config.bwd_importance_sampling = True
        preprocess_config.bwd_importance_score_path = f"{CACHE_ROOT}/bwd_importance_score/{path_name}"
    xxx_config = XXXConfig(target_modules=args.target_modules,)
    
    for dataset_name in args.knowledge_dataset:
        # Collect data
        knowledge_data_loader = get_knowledge_data(
            name=dataset_name, 
            tokenizer=tokenizer, 
            model_id=model_id, 
            nsamples=args.n_knowledge_samples, 
            seed=args.seed
        )
        task_data_loader = None
        if args.fwd_importance_sampling:
            task_data_loader = get_knowledge_data(
                name=args.task_dataset, 
                tokenizer=tokenizer, 
                model_id=model_id, 
                nsamples=args.n_task_samples, 
                seed=args.seed
            )

        dataset_name = dataset_name.replace("/", "_")
        path_name = f"{dataset_name}_{args.model_id.replace('/', '_')}_{args.n_knowledge_samples}_{args.seed}_down{int(1/args.n_param_downsample_rate)}"
        preprocess_config.jacobian_path = f"{CACHE_ROOT}/jacobian/{path_name}"
        xxx_config.preprocess_config = preprocess_config

        if args.fwd_importance_sampling:
            calculate_importance_score(
                model,
                task_data_loader,
                xxx_config,
                mode="abs",
                save_path=preprocess_config.fwd_importance_score_path,
            )
        if args.bwd_importance_sampling:
            calculate_importance_score(
                model,
                knowledge_data_loader,
                xxx_config,
                mode="abs",
                save_path=preprocess_config.bwd_importance_score_path,
            )

        # preprocess_xxx(
        #     model,
        #     xxx_config,
        #     knowledge_data_loader=knowledge_data_loader,
        #     task_data_loader=task_data_loader
        # )
        calculate_jacobian(
            model, 
            xxx_config, 
            data_loader=knowledge_data_loader, 
            mask_path=mask_path,
        )
        if mask_path is None:
            # Use the first dataset to generate the mask
            mask_path = preprocess_config.jacobian_path
        jacobian_paths.append(preprocess_config.jacobian_path)
    
    dataset_name = "_".join(sorted(args.knowledge_dataset)).replace("/", "_")
    if args.adaptive_r_stiff_basis:
        path_name = f"{dataset_name}_{args.model_id.replace('/', '_')}_adaptive_r{args.r_stiff_basis}_thres{args.cumulative_energy_threshold}_{args.seed}_down{int(1/args.n_param_downsample_rate)}"
    else:
        path_name = f"{dataset_name}_{args.model_id.replace('/', '_')}_r{args.r_stiff_basis}_{args.seed}_down{int(1/args.n_param_downsample_rate)}"
    preprocess_config.jacobian_path = jacobian_paths
    preprocess_config.stiff_basis_path = f"{CACHE_ROOT}/stiff_basis/{path_name}"
    preprocess_config.r_stiff_basis = args.r_stiff_basis
    preprocess_config.adaptive_r_stiff_basis = args.adaptive_r_stiff_basis
    preprocess_config.cumulative_energy_threshold = args.cumulative_energy_threshold
    xxx_config.preprocess_config = preprocess_config
    calculate_stiff_basis(
        model,
        xxx_config,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["k_proj","up_proj","v_proj","o_proj","q_proj","gate_proj","down_proj"],
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--n_knowledge_samples",
        type=int,
        default=256,
        help="number of samples used for covariance matrices",
    )
    parser.add_argument(
        "--knowledge_dataset",
        type=str,
        nargs="+",
        default=["nqopen",],
        choices=[],
        help="knowledge dataset",
    )
    parser.add_argument(
        "--n_param_downsample_rate",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--r_stiff_basis",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--adaptive_r_stiff_basis",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--cumulative_energy_threshold",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--min_r_stiff_basis",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--fwd_importance_sampling",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--bwd_importance_sampling",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--quantize_stiff_basis",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=233,
        help="random seed",
    )
    args = parser.parse_args()

    main(args)
