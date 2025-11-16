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
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from datautils import get_knowledge_data
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import get_peft_model
from peft.tuners.lora.config import LoraConfig
from peft.tuners.xxx.config import XXXConfig, XXXPreprocessConfig
from peft.tuners.xxx.utils import preprocess_xxx, calculate_importance_score

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

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    # Collect data
    knowledge_data_loader = get_knowledge_data(
        name=args.knowledge_dataset, 
        tokenizer=tokenizer, 
        model_id=model_id, 
        nsamples=args.n_knowledge_samples, 
        seed=args.seed
    )
    task_data_loader = None
    if args.fwd_importance_sampling or args.task_oriented_sloppy_basis:
        task_data_loader = get_knowledge_data(
            name=args.task_dataset, 
            tokenizer=tokenizer, 
            model_id=model_id, 
            nsamples=args.n_task_samples, 
            seed=args.seed
        )

    # Evaluate the original model
    print("\n---- model before svd ---\n")
    print(model)

    dataset_name = "_".join(sorted(args.knowledge_dataset)).replace("/", "_")
    if args.task_oriented_sloppy_basis:
        task_dataset_name = "_".join(sorted(args.task_dataset)).replace("/", "_")
    else:
        task_dataset_name = "none"
    n_knowledge_samples = args.n_knowledge_samples * len(args.knowledge_dataset)
    path_name = f"{dataset_name}_{args.model_id.replace('/', '_')}_{n_knowledge_samples}_{args.seed}_down{int(1/args.n_param_downsample_rate)}"
    sloppy_basis_path_name = f"kldg_{dataset_name}_task_{task_dataset_name}_{args.model_id.replace('/', '_')}_{n_knowledge_samples}_{args.seed}_down{int(1/args.n_param_downsample_rate)}_r{args.r}"
    preprocess_config = XXXPreprocessConfig(
        jacobian_path=f"{CACHE_ROOT}/jacobian/{path_name}",
        sloppy_basis_path=f"{CACHE_ROOT}/sloppy_basis/{sloppy_basis_path_name}",
        eigen_path=f"{CACHE_ROOT}/eigen/{path_name}",
        n_param_downsample_rate=args.n_param_downsample_rate,
    )
    if args.fwd_importance_sampling:
        preprocess_config.fwd_importance_sampling = True
        preprocess_config.fwd_importance_score_path = f"{CACHE_ROOT}/fwd_importance_score/{path_name}"
    if args.bwd_importance_sampling:
        preprocess_config.bwd_importance_sampling = True
        preprocess_config.bwd_importance_score_path = f"{CACHE_ROOT}/bwd_importance_score/{path_name}"
    if args.task_oriented_sloppy_basis:
        preprocess_config.task_oriented_sloppy_basis = True
        task_dataset_name = "_".join(sorted(args.task_dataset)).replace("/", "_")
        n_task_samples = args.n_task_samples * len(args.task_dataset)
        task_jacobian_path = f"{CACHE_ROOT}/task_jacobian/{task_dataset_name}_{args.model_id.replace('/', '_')}_{n_task_samples}_{args.seed}_down{int(1/args.n_param_downsample_rate)}"
        preprocess_config.task_jacobian_path = task_jacobian_path

    
    xxx_config = XXXConfig(
        r=args.r,
        target_modules=args.target_modules,
        preprocess_config=preprocess_config,
    )
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

    preprocess_xxx(
        model,
        xxx_config,
        knowledge_data_loader=knowledge_data_loader,
        task_data_loader=task_data_loader
    )
    model = get_peft_model(model, xxx_config)

    # Evaluate again to check if the model is consistent
    # Using `model.model` here because `get_peft_model` wraps a layer to the model
    print(model)

    # Save as hugging face model
    # if args.save_model:
    #     assert args.save_path is not None
    #     save_path = args.save_path

    #     # Save CorDA modules
    #     model.peft_config["default"].init_lora_weights = True
    #     model.save_pretrained(os.path.join(save_path, "corda_init"))

    #     # Save residual model
    #     model = model.unload()
    #     model.save_pretrained(save_path)

    #     # Save tokenizer
    #     tokenizer.save_pretrained(save_path)
    #     print(f"Done building CorDA huggingface model in {save_path}")


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
        default=["gate_proj",],
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
        default=0.01,
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
        "--task_oriented_sloppy_basis",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--task_dataset",
        type=str,
        nargs="+",
        default=["metamath"],
        choices=[],
        help="task dataset",
    )
    parser.add_argument(
        "--n_task_samples",
        type=int,
        default=256,
        help="number of samples used for covariance matrices",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=233,
        help="random seed",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=128,
    )
    # parser.add_argument(
    #     "--save_model",
    #     default=True,
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--save_path",
    #     type=str,
    #     default=f"{CACHE_ROOT}/opt_125m_corda_init",
    # )
    args = parser.parse_args()

    main(args)
