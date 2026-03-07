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

import copy
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional, Union

from peft.tuners.xxx.config import XXXConfig, XXXPreprocessConfig
from peft.tuners.xxx.utils import ProjectionCallback, preprocess_xxx
from peft.tuners.xxx.utils import ProjectionCallback, preprocess_xxx
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import Trainer

from peft import LoraConfig, get_peft_model


IGNORE_INDEX = -100

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

CACHE_ROOT = "/Data1/zhengzhilong"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: str = field(default="meta-llama/Llama-2-7b-hf")
    knowledge_dataset: list[str] = field(default_factory=lambda: ["nqopen",])
    lora_r: int = field(
        default=128,
        metadata={"help": "The rank of LoRA adapter."},
    )
    r_jac_approx: int = field(default=28)
    r_stiff_basis: int = field(default=256)
    n_knowledge_samples: int = field(default=256)
    proj_interval: int = field(default=1)
    init_lora_weights: Union[str, bool] = field(default=True)
    adaptive_r_stiff_basis: bool = field(default=False)
    cumulative_energy_threshold: float = field(default=0.9)
    quantize_stiff_basis: bool = field(default=True)
    seed: Optional[int] = field(default=233)
    data_path: str = field(default="fxmeng/pissa-dataset", metadata={"help": "Path to the training data."})
    dataset_split: str = field(default="train", metadata={"help": "(`['train', 'test', 'eval']`):"})
    sub_task: list[str] = field(default_factory=lambda: ["metamath:100"], metadata={"help": "(`['metamath', 'python', 'conversation']`)"})
    dataset_field: list[str] = field(default_factory=lambda: ["instruction", "output"], metadata={"help": "Fields of dataset input and output."})
    dataloader_num_proc: int = field(default=1, metadata={"help": "Number of processes to load dataset"})
    dataloader_batch_size: int = field(
        default=3000,
        metadata={
            "help": "batch size to load dataset. To set the batch size for training, you should pass --batch_size argument instead."
        },
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for AdamW."},
    )


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
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

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Preprocess the data by tokenizing."""

    def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_ids_lens": input_ids_lens,
            "labels_lens": labels_lens,
        }

    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = (_tokenize_fn(strings, tokenizer) for strings in (examples, sources))
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [
        PROMPT.format_map(
            {
                "instruction": instruction,
            }
        )
        for instruction in examples[query]
    ]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def get_nb_trainable_parameters(model) -> tuple[int, int]:
    """
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_bytes = param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/training_config.json", "w") as f:
        def set_default(obj):
            try:
                return str(obj)
            except Exception:
                return f"__Unserializable_Object_of_Type_{type(obj).__name__}__"
        json.dump(vars(args), f, indent=4, default=set_default)


    if args.r_stiff_basis is not None:
        print("Train in XXX mode")
        print("Loading base model...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        
        dataset_name = "_".join(sorted(args.knowledge_dataset)).replace("/", "_")
        if args.adaptive_r_stiff_basis:
            path_name = f"{dataset_name}_{args.model_name_or_path.replace('/', '_')}_adaptive_r{args.r_stiff_basis}_thres{args.cumulative_energy_threshold}_{args.seed}"
        else:
            # path_name = f"{dataset_name}_{args.model_name_or_path.replace('/', '_')}_pissa_r{args.r_stiff_basis}_{args.seed}"
            path_name = f"{dataset_name}_{args.model_name_or_path.replace('/', '_')}_pissa_r_jac_approx{args.r_jac_approx}_{args.n_knowledge_samples}_{args.seed}"
        preprocess_config = XXXPreprocessConfig(
            stiff_basis_path=f"{CACHE_ROOT}/jacobian/{path_name}",
            # stiff_basis_path=f"{CACHE_ROOT}/stiff_basis/{path_name}",
            quantize_stiff_basis=args.quantize_stiff_basis,
        )
        xxx_config = XXXConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r,
            target_modules=["0.self_attn.q_proj", ],
            init_lora_weights=args.init_lora_weights,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
            preprocess_config=preprocess_config,
        )
        print("Preprocessing for XXX...")
        preprocess_xxx(model, xxx_config)
        print("Getting PEFT model...")
        model = get_peft_model(model, xxx_config)
    elif args.lora_r is not None:
        print("Train in LoRA mode")
        print("Loading base model...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            dtype=torch.bfloat16,
        )
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r,
            init_lora_weights=args.init_lora_weights,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    else:
        print("Train in Full Finetuning mode")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            # device_map="auto",
        )

    if args.local_rank == 0:
        trainable_params, all_param = get_nb_trainable_parameters(model)
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        # trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    all_training_dataset = []
    for task in args.sub_task:
        if ":" in task: # e.g. math:500, gsm8k:100
            cur_task, num_split = task.split(":")
            cur_split = f"{args.dataset_split}[:{num_split}]"
        else:
            cur_task, cur_split = task, args.dataset_split

        if args.data_path == "glue":
            ds = load_dataset(args.data_path, name=cur_task, split=cur_split)
        else:
            ds = load_dataset(args.data_path, data_dir=cur_task, split=cur_split)
        all_training_dataset.append(ds)

    raw_train_datasets = concatenate_datasets(all_training_dataset)

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=args.dataloader_batch_size,
        num_proc=args.dataloader_num_proc,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer,
            "query": args.dataset_field[0],
            "response": args.dataset_field[1],
        },
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = {
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }
    trainer = Trainer(model=model, processing_class=tokenizer, args=args, **data_module)
    projection_callback = ProjectionCallback(
        local_rank=args.local_rank, 
        proj_interval=args.proj_interval
    )
    trainer.add_callback(projection_callback)
    if args.local_rank == 0:
        print("Start training...")
    trainer.train()
    trainer.save_state()
    model = model.merge_and_unload()
    model = model.to(torch.bfloat16)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    train()
