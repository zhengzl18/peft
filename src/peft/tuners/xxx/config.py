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

from dataclasses import dataclass, field
from typing import Optional, Union, List

from peft.tuners.lora.config import LoraConfig
from peft.utils import PeftType


@dataclass
class XXXPreprocessConfig:
    """
    This is the sub-configuration class to store the configuration of a [`LoraModel`].

    Args:
        jacobian_path (`Optional[str]`):
            File to store the jacobian matrix.
        quantize_jacobian (`bool`):
            If true, quantizes the jacobian matrix to int8. This can reduce the memory usage of the jacobian matrix by 4x.
    """
    jacobian_path: Optional[Union[str, List[str]]] = field(
        default=None,
        metadata={
            "help": (
                "File to store the jacobian matrix. If you wish to train multiple models with different ranks, but "
                "they sample from the same dataset, you can store the jacobian matrix and reuse it for different ranks. "
                "Note that jacobian file is usually large (comparable to model size), so you will need sufficient storage."
            )
        },
    )
    stiff_basis_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "File to store the stiff basis matrix. If you wish to train multiple models with different ranks, but "
                "they sample from the same dataset, you can store the stiff basis matrix and reuse it for different ranks. "
                "Note that stiff basis file is usually large (comparable to model size), so you will need sufficient storage."
            )
        },
    )
    r_jac_approx: Optional[int] = field(
        default=None,
    )
    r_stiff_basis: Optional[int] = field(
        default=None,
    )
    adaptive_r_stiff_basis: bool = field(
        default=False,
    )
    cumulative_energy_threshold: float = field(
        default=0.9,
    )
    min_r_stiff_basis: int = field(
        default=1,
    )
    verbose: bool = field(default=False, metadata={"help": "If true, prints the progress of CorDA initialization."})
    quantize_stiff_basis: bool = field(
        default=True,
        metadata={
            "help": (
                "If true, quantizes the stiff basis matrix to int8. This can reduce the memory usage of the stiff basis matrix by 4x."
            )
        },
    )


@dataclass
class XXXConfig(LoraConfig):
    preprocess_config: Optional[XXXPreprocessConfig] = field(
        default=None,
        metadata={"help": "The CorDA preprocessing configuration."},
    )
    
    # TODO: maybe override LoraConfig's init_lora_weights param

    # def to_dict(self):
    #     """
    #     Returns the configuration for your adapter model as a dictionary. Removes runtime configurations.
    #     """
    #     rv = super().to_dict()
    #     # rv.pop("runtime_config")
    #     return rv

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.XXX
        # assert self.init_lora_weights == "orthogonal", "XXX only supports 'orthogonal' initialization for LoRA weights."
        assert self.use_dora is False, "XXX does not support DoRA."
        assert self.use_rslora is False, "XXX does not support RS-LoRA."
        assert self.use_qalora is False, "XXX does not support QALoRA."
        assert self.arrow_config is None, "XXX does not support ArrowLinearVariant."