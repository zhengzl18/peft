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

from peft import get_peft_model
from peft.tuners.lora.config import LoraConfig
from peft.tuners.xxx.config import XXXConfig, XXXPreprocessConfig
from peft.tuners.xxx.utils import preprocess_xxx

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
    knowledge_loader = get_knowledge_data(
        name=args.knowledge_dataset, 
        tokenizer=tokenizer, 
        model_id=model_id, 
        nsamples=args.n_knowledge_samples, 
        seed=args.seed
    )

    # Evaluate the original model
    print("\n---- model before svd ---\n")
    print(model)

    # Perform decomposition
    dataset_name = "_".join(sorted(args.knowledge_dataset)).replace("/", "_")
    n_knowledge_samples = args.n_knowledge_samples * len(args.knowledge_dataset)
    path_name = f"{dataset_name}_{args.model_id.replace('/', '_')}_{n_knowledge_samples}_{args.seed}"
    preprocess_config = XXXPreprocessConfig(
        jacobian_path=f"{CACHE_ROOT}/jacobian/{path_name}",
        sloppy_basis_path=f"{CACHE_ROOT}/sloppy_basis/{path_name}",
        eigen_path=f"{CACHE_ROOT}/eigen/{path_name}",
    )
    xxx_config = XXXConfig(
        r=args.r,
        # target_modules=["q_proj", "v_proj"],
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        preprocess_config=preprocess_config,
    )
    preprocess_xxx(
        model,
        xxx_config,
        knowledge_loader,
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
        default="facebook/opt-125m",
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
        # default=["MetaMATH"],
        # default=["glue/cola",],
        default=["glue/sst2", "glue/mrpc", "glue/mnli", "glue/qqp"],
        choices=[
            "wikitext2",
            "c4",
            "ptb",
            "traivia_qa",
            "nqopen",
            "MetaMATH",
            "codefeedback",
            "WizLMinstruct",
            "alpaca",
        ],
        help="knowledge dataset",
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
