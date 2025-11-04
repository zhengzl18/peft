# Copyright 2023-present the HuggingFace Inc. team.
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

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from torch import nn

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class XXXPreprocessConfig:
    """
    This is the sub-configuration class to store the configuration of a [`LoraModel`].

    Args:
        cache_file (`Optional[str]`):
            File to store the SVD cache. The SVD cache is much smaller than the residual model (for example, residual
            model of Llama-3-8b is 15GB, while SVD cache is 1.4GB), but with SVD cache and original model weights,
            residual model weights can be built quickly. If you need to reuse residual model weights with limited
            storage, you can store the SVD cache instead.
        jacobian_file (`Optional[str]`):
            File to store the jacobian matrix. If you wish to train multiple models with different ranks, but they
            sample from the same dataset, you can store the jacobian matrix and reuse it for different ranks. Note
            that jacobian file is usually large (comparable to model size), so you will need sufficient storage.
        verbose (`bool`):
            If true, prints the progress of CorDA initialization. Defaults to `False`.
        use_float16_for_jacobian (`bool`):
            If true, uses float16 for the jacobian matrix. This can reduce the memory usage of the jacobian matrix
            by half, but may lead to numerical instability. Defaults to `False`.
        prune_temporary_fields (`bool`):
            If true, temporary fields generated in CorDA preprocessing will be pruned. Defaults to `True`.
    """
    jacobian_path: str = field(
        metadata={
            "help": (
                "File to store the jacobian matrix. If you wish to train multiple models with different ranks, but "
                "they sample from the same dataset, you can store the jacobian matrix and reuse it for different ranks. "
                "Note that jacobian file is usually large (comparable to model size), so you will need sufficient storage."
            )
        },
    )
    sloppy_basis_path: str = field(
        metadata={
            "help": (
                "File to store the SVD cache. The SVD cache is much smaller than the residual model (for example, "
                "residual model of Llama-3-8b is 15GB, while SVD cache is 1.4GB), but with SVD cache and original model "
                "weights, residual model weights can be built quickly. If you need to reuse residual model weights with "
                "limited storage, you can store the SVD cache instead."
            )
        },
    )
    eigen_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "File to store the jacobian matrix. If you wish to train multiple models with different ranks, but "
                "they sample from the same dataset, you can store the jacobian matrix and reuse it for different ranks. "
                "Note that jacobian file is usually large (comparable to model size), so you will need sufficient storage."
            )
        },
    )
    n_param_downsample_rate: float = field(
        default=1.0,
    )
    fwd_importance_sampling: bool = field(
        default=False,
    )
    fwd_importance_score_path: Optional[str] = field(
        default=None,
    )
    bwd_importance_sampling: bool = field(
        default=False,
    )
    bwd_importance_score_path: Optional[str] = field(
        default=None,
    )
    task_oriented_sloppy_basis: bool = field(
        default=False,
    )
    task_jacobian_path: Optional[str] = field(
        default=None,
    )
    verbose: bool = field(default=False, metadata={"help": "If true, prints the progress of CorDA initialization."})
    use_float16_for_jacobian: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, uses float16 for the jacobian matrix. This can reduce the memory usage of the jacobian matrix "
                "by half, but may lead to numerical instability."
            )
        },
    )
    prune_temporary_fields: bool = field(
        default=True, metadata={"help": "If true, temporary fields generated in CorDA preprocessing will be pruned."}
    )


@dataclass
class XXXConfig(PeftConfig):
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA. "
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D "
                "(if the model is a PreTrainedModel, the output layer excluded). "
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually. "
                "To avoid targeting any modules (because you want to apply `target_parameters`), set "
                "`target_modules=[]`."
            ),
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from Lora."},
    )
    scaling: float = field(default=1.0, metadata={"help": "Lora alpha"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
            "This only works when target_modules is a list of str. This should target the `nn.ModuleList` of the "
            "model, which is often called `'layers'` or `'h'`."
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`."
            )
        },
    )
    scaling_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`. "
                "For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`."
            )
        },
    )
    trainable_token_indices: Optional[Union[list[int], dict[str, list[int]]]] = field(
        default=None,
        metadata={
            "help": (
                "Lets you specify which token indices to selectively fine-tune without requiring to re-train the "
                "whole embedding matrix using the `peft.TrainableTokensModel` method. You can specify token indices "
                "in two ways. Either you specify a list of indices which will then target the model's input embedding "
                "layer (or, if not found, `embed_tokens`). Alternatively, you can specify a dictionary where the key "
                "is the name of the embedding module and the values are the list of token indices, e.g. "
                "`{'embed_tokens': [0, 1, ...]}`. Note that training with FSDP requires `use_orig_params=True` to "
                "avoid issues with non-uniform `requires_grad`."
            )
        },
    )
    target_parameters: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of parameter names or regex expression of the parameter names to replace with LoRA. "
                "This argument behaves similarly to `target_modules`, except that the parameter name should be passed. "
                "Generally, you should use `target_modules` to target the module (e.g. `nn.Linear`). However, in some "
                "circumstances, this is not possible. E.g., in many mixture of expert (MoE) layers in HF Transformers, "
                "instead of using `nn.Linear`, an `nn.Parameter` is used. PEFT normally overwrites the `forward` "
                "method for LoRA, but for `nn.Parameter`, there is none. Therefore, to apply LoRA to that parameter, "
                "it needs to be targeted with `target_parameters`. As an example, for Llama4, you can pass: "
                "`target_parameters=['feed_forward.experts.gate_up_proj', 'feed_forward.experts.down_proj]`. Passing a "
                "string for regex matching is not implemented yet."
            )
        },
    )
    preprocess_config: Optional[XXXPreprocessConfig] = field(
        default=None,
        metadata={"help": "The CorDA preprocessing configuration."},
    )

    def to_dict(self):
        """
        Returns the configuration for your adapter model as a dictionary. Removes runtime configurations.
        """
        rv = super().to_dict()
        # rv.pop("runtime_config")
        return rv

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.XXX
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        if isinstance(self.target_parameters, str):
            raise TypeError("`target_parameters` must be a list of strings or None.")

        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")