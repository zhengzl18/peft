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

import os
import random

import numpy as np
import torch
from datasets import load_dataset

CACHE_ROOT = os.path.dirname(os.path.abspath(__file__))  # peft/examples/corda_finetuning

"""
doc https://huggingface.co/docs/datasets/loading
doc https://huggingface.co/docs/datasets/process
doc https://huggingface.co/blog/llama2#how-to-prompt-llama-2
"""


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def sample_train_loaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048):
    set_seed(seed)
    if "wikitext2" in name:
        traindata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
        )
        traindata = "\n\n".join(traindata["text"])
    elif "c4" in name:
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        traindata = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(traindata) - seqlen * 2 - 1)
        j = i + seqlen * 2
        # breakpoint()
        trainenc = tokenizer(traindata[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        trainloader.append(inp)
    return trainloader


def get_redpajama_train(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048):
    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    if percent != 100:
        split = f"train[:{int(850000 * percent / 100)}]"
    else:
        split = "train"
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split)

    processed_dataset = dataset.map(tokenization, batched=True, batch_size=batch_size, num_proc=os.cpu_count())
    return processed_dataset


def get_english_quote(dataset_name, tokenizer):
    data = load_dataset(dataset_name)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    return data["train"]


def get_qat_dataset(name, tokenizer, data_percent):
    if name == "red_pajama":
        data = get_redpajama_train(tokenizer, data_percent)

    elif name == "Abirate/english_quotes":
        data = get_english_quote(name, tokenizer)
    else:
        raise NotImplementedError
    data = data.shuffle()
    return data


llama_chat_format = """<s>[INST] <<SYS>>
"Below is an instruction that describes a task. Write a response that appropriately completes the request."
<</SYS>>

{instruction} [/INST] {response} </s>
"""

CACHE_ROOT = "/Data2/zhengzhilong" 

def get_knowledge_data(name, tokenizer, model_id, nsamples, seed=3):
    if isinstance(name, list):
        knowledge_dataset = []
        for subname in name:
            knowledge_dataset += get_knowledge_data(
                subname, tokenizer, model_id, nsamples, seed
            )
        return knowledge_dataset

    print(f" get_data_from: {name}, nsamples={nsamples}, seed={seed}")
    cache_file = f"{CACHE_ROOT}/knowledge_data/{name}_{model_id.replace('/', '_')}_{nsamples}_{seed}.pt"
    knowledge_dataset = []
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print(f"found data file: {cache_file}")
        knowledge_dataset = torch.load(cache_file)
        print("loaded ...")
        return knowledge_dataset

    if name == "trivia_qa":
        traindata = load_dataset("trivia_qa", "rc", split="train").shuffle(seed=seed).take(nsamples)
        PROMPT = "Answer these questions:\n\n Q: {question}?\nAnswer:{answer}"
        input_texts = [
            PROMPT.format(question=q, answer=a)
            for q, a in zip(traindata["question"], traindata["answer"])
        ]
    elif name == "nqopen":
        traindata = load_dataset("nq_open", split="train").shuffle(seed=seed).take(nsamples)
        PROMPT = "Answer these questions:\n\nQ: {question}?\nAnswer:{answer}"
        input_texts = [
            PROMPT.format(question=q, answer=a[0])
            for q, a in zip(traindata["question"], traindata["answer"])
        ]
    elif name == "metamath":
        traindata = load_dataset("fxmeng/pissa-dataset", data_dir="metamath", split="train").shuffle(seed=seed).take(nsamples)
        PROMPT = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:{output}"
        )
        input_texts = [
            PROMPT.format(instruction=i, output=o)
            for i, o in zip(traindata["instruction"], traindata["output"])
        ]
    elif name.startswith("glue/"):
        task_name = name.split("/")[1]
        # GLUE tasks often need 'validation' split for reasonable size, 
        # 'train' can be huge (like MNLI). Adjust split as needed.
        # Use 'validation' for sampling, 'train' if validation is too small or absent.
        split_to_use = 'train'
        # Some GLUE tasks might not have a standard 'train' split accessible easily
        # or might be very large. We default to train but might need adjustments.
        dataset = load_dataset("glue", task_name, split=split_to_use) 
        
        # Ensure nsamples is not larger than the dataset split
        actual_nsamples = min(nsamples, len(dataset))
        if actual_nsamples < nsamples:
            print(f"Warning: Requested {nsamples} samples, but split '{split_to_use}' only has {len(dataset)}. Using {actual_nsamples}.")
        
        # sampled_indices = np.random.RandomState(seed).choice(len(dataset), actual_nsamples, replace=False)
        traindata = dataset.shuffle(seed=seed).take(nsamples)

        # --- Task-specific formatting for GLUE ---
        if task_name == "sst2": # Sentiment (binary classification)
            label_map = {0: "negative", 1: "positive"}
            PROMPT = "{sentence}\nQuestion: Is this sentence positive or negative?\nAnswer:"
            input_texts = [
                PROMPT.format(sentence=s) + f" {label_map[l]}"
                for s, l in zip(traindata["sentence"], traindata["label"])
            ]
        elif task_name == "mrpc": # Paraphrase (binary classification)
            label_map = {0: "No", 1: "Yes"}
            PROMPT = "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nQuestion: Do both sentences mean the same thing?\nAnswer:"
            input_texts = [
                PROMPT.format(sentence1=s1, sentence2=s2) + f" {label_map[l]}"
                for s1, s2, l in zip(traindata["sentence1"], traindata["sentence2"], traindata["label"])
            ]
        elif task_name == "mnli": # Natural Language Inference (3-class classification)
            label_map = {0: "True", 1: "Neither", 2: "False"}
            PROMPT = "{premise}\nQuestion: {hypothesis} True, False or Neither?\nAnswer:"
            input_texts = [
                PROMPT.format(premise=p, hypothesis=h) + f" {label_map[l]}"
                for p, h, l in zip(traindata["premise"], traindata["hypothesis"], traindata["label"])
            ]
        elif task_name == "qqp": # Question Pairs (binary classification)
            label_map = {0: "No", 1: "Yes"}
            PROMPT = "Question 1: {question1}\nQuestion 2: {question2}\nQuestion: Do both questions ask the same thing?\nAnswer:"
            input_texts = [
                PROMPT.format(question1=q1, question2=q2) + f" {label_map[l]}"
                for q1, q2, l in zip(traindata["question1"], traindata["question2"], traindata["label"])
            ]
        elif task_name == "cola":
            label_map = {0: "No", 1: "Yes"}
            PROMPT = "{sentence}\nQuestion: Does this sentence make sense?\nAnswer:"
            input_texts = [
                PROMPT.format(sentence=s) + f" {label_map[l]}"
                for s, l in zip(traindata["sentence"], traindata["label"])
            ]
        # Add more elif blocks for other GLUE tasks (cola, rte, qnli, wnli, sts-b) 
        # following a similar pattern: define prompt, map labels to text.
        else:
            raise NotImplementedError(f"GLUE task '{task_name}' formatting not implemented.")
    else:
        raise NotImplementedError

    for text in input_texts:
        enc = tokenizer(text, return_tensors="pt")
        knowledge_dataset.append(
            {
                "input_ids": enc.input_ids, 
                "attention_mask": enc.attention_mask,
                "labels": enc.input_ids,
            }
        )
    torch.save(knowledge_dataset, cache_file)
    return knowledge_dataset


def get_eval_loaders(name, tokenizer):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    if "ptb" in name:
        valdata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    if "c4" in name:
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    raise NotImplementedError
