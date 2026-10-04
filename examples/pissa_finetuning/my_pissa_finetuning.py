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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import transformers
from datasets import load_dataset
from transformers import Trainer

from peft import LoraConfig, PeftModel, get_peft_model
from ella.core import (
    ELLAState,
    compute_ella_penalty_from_model,
    update_past_weights_from_model,
)

IGNORE_INDEX = -100

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def get_nb_trainable_parameters(model) -> tuple[int, int]:
    r"""
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


class ELLATrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        ella_lambda: float,
        ella_loss_type: str,
        ella_delta_mode: str,
        subtract_past_tensor: bool,
        ella_state: ELLAState,
        wandb_step_offset: int = 0,
        **kwargs: Any,
    ) -> None:
        self.wandb_step_offset = int(wandb_step_offset)
        super().__init__(*args, **kwargs)
        self.ella_lambda = ella_lambda
        self.ella_loss_type = ella_loss_type
        self.ella_delta_mode = ella_delta_mode
        self.subtract_past_tensor = subtract_past_tensor
        self.ella_state = ella_state

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        base_loss = outputs.loss
        penalty = compute_ella_penalty_from_model(
            model=model,
            state=self.ella_state,
            loss_type=self.ella_loss_type,
            delta_mode=self.ella_delta_mode,
            subtract_past_tensor=self.subtract_past_tensor,
        )
        ella_loss = self.ella_lambda * penalty
        loss = base_loss + ella_loss
        self._last_ce_loss = base_loss.detach()
        self._last_ella_loss = ella_loss.detach()
        self._last_penalty = penalty.detach()
        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Shift global_step only while logging so W&B stays monotonic across tasks."""
        if hasattr(self, "_last_ce_loss"):
            logs["train/ce_loss"] = self._last_ce_loss.item()
        if hasattr(self, "_last_ella_loss"):
            logs["train/ella_loss"] = self._last_ella_loss.item()
        if hasattr(self, "_last_penalty"):
            logs["train/penalty"] = self._last_penalty.item()
        o = self.wandb_step_offset
        if o:
            prev = int(self.state.global_step)
            self.state.global_step = prev + o
        try:
            super().log(logs, start_time=start_time)
        finally:
            if o:
                self.state.global_step = prev


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_split: str = field(default=None, metadata={"help": "(`['train', 'test', 'eval']`):"})
    sub_task: str = field(default=None, metadata={"help": "(`['metamath', 'python', 'conversation']`)"})
    dataset_field: list[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    dataloader_num_proc: int = field(default=16, metadata={"help": "Number of processes to load dataset"})
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
    lora_r: int = field(
        default=None,
        metadata={"help": "The rank of LoRA adapter. When passing `None`, CorDA or full fine-tuning is used."},
    )
    pissa_mode: bool = field(default=True, metadata={"help": "True for CorDA mode"})
    ella_lambda: float = field(
        default=0.0,
        metadata={"help": "Weight of the ELLA penalty. Set to 0 to disable ELLA."},
    )
    ella_loss_type: str = field(
        default="ella",
        metadata={"help": "Penalty variant used by ELLA."},
    )
    ella_delta_mode: str = field(
        default="layerwise",
        metadata={"help": "Delta computation mode: 'layerwise' or 'all'."},
    )


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


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Preprocess the data by tokenizing."""
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


def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    if script_args.pissa_mode:
        print("Train in PiSSA mode")
        res_model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            dtype=torch.bfloat16,
            # device_map="auto",
        )
        model = PeftModel.from_pretrained(
            res_model, script_args.model_name_or_path, subfolder="pissa_init", is_trainable=True
        )
    elif script_args.lora_r is not None:
        print("Train in LoRA mode")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            # device_map="auto",
        )
        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_r,
            init_lora_weights=True,  # script_args.init_lora_weights,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    else:
        print("Train in Full Finetuning mode")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            dtype=torch.bfloat16,
            # device_map="auto",
        )

    ella_state = None
    if script_args.ella_lambda != 0:
        if not script_args.pissa_mode:
            raise ValueError("ELLA training requires --pissa_mode True so the initial PiSSA deltaW is available.")
        # PiSSA starts with a non-zero low-rank update. Keep its initial factors
        # fixed as the historical deltaW used by the ELLA penalty.
        ella_state = ELLAState()
        update_past_weights_from_model(state=ella_state, model=model)
        if not ella_state.past:
            raise RuntimeError("Could not collect the initial PiSSA A/B factors for ELLA.")
        print(f"ELLA enabled: lambda={script_args.ella_lambda}, loss_type={script_args.ella_loss_type}")

    if script_args.local_rank == 0:
        trainable_params, all_param = get_nb_trainable_parameters(model)
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        # padding_side="right",
        # use_fast=True,
        # trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    task, num_split = script_args.sub_task.split(":")
    split = f"{script_args.dataset_split}[:{num_split}]"
    raw_train_datasets = load_dataset(script_args.data_path, data_dir=task, split=split)
    # raw_train_datasets = load_dataset(script_args.data_path, split=script_args.dataset_split)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=script_args.dataloader_batch_size,
        num_proc=script_args.dataloader_num_proc,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer,
            "query": script_args.dataset_field[0],
            "response": script_args.dataset_field[1],
        },
    )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = {
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }
    if ella_state is None:
        trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
    else:
        trainer = ELLATrainer(
            model=model,
            tokenizer=tokenizer,
            args=script_args,
            ella_lambda=script_args.ella_lambda,
            ella_loss_type=script_args.ella_loss_type,
            ella_delta_mode=script_args.ella_delta_mode,
            subtract_past_tensor=script_args.pissa_mode,
            ella_state=ella_state,
            **data_module,
        )
    trainer.train()
    trainer.save_state()
    model = model.merge_and_unload().to(torch.bfloat16)
    model.save_pretrained(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)


if __name__ == "__main__":
    train()
