# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
from pydantic import Field, field_validator
from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from deepspeed.runtime.zero.config import DeepSpeedZeroConfig
from typing import Dict, Union, Optional
from enum import Enum


class DtypeEnum(Enum):
    fp16 = (torch.float16, "torch.float16", "fp16", "float16", "half")
    fp32 = (torch.float32, "torch.float32", "fp32", "float32", "float")
    bf16 = (torch.bfloat16, "torch.bfloat16", "bf16", "bfloat16", "bfloat")
    int8 = (torch.int8, "torch.int8", "int8")

    @classmethod
    def from_str(cls, value: str):
        for dtype in cls:
            if value in dtype.value:
                return dtype
        raise ValueError(f"'{value}' is not a valid DtypeEnum")


class MoETypeEnum(str, Enum):
    residual = "residual"
    standard = "standard"


class DeepSpeedTPConfig(DeepSpeedConfigModel):
    """ Configure tensor parallelism settings """

    enabled: bool = True
    """ Turn tensor parallelism on/off. """

    tp_size: int = 1
    """ Number of devices to split the model across using tensor parallelism. """

    mpu: object = None
    """
    A model parallelism unit object that implements
    ``get_{model,data}_parallel_{rank,group,world_size}()``.
    """

    tp_group: object = None


class DeepSpeedMoEConfig(DeepSpeedConfigModel):
    """ Sets parameters for MoE """

    enabled: bool = True
    ep_size: int = 1
    """
    The expert-parallelism size which is used for partitioning the experts
    across the GPUs in the expert-parallel group.
    """

    moe_experts: list = Field([1], alias="num_experts")
    """ The global number of experts used in an MoE layer. """

    type: MoETypeEnum = MoETypeEnum.standard
    """
    Specify the type of MoE layer. We have two types of MoE layer: 'Standard'
    and 'Residual'.
    """

    ep_mp_group: object = None
    ep_group: object = Field(None, alias="expert_group")


class QuantTypeEnum(str, Enum):
    asym = "asymmetric"
    sym = "symmetric"


class BaseQuantConfig(DeepSpeedConfigModel):
    enabled: bool = True
    num_bits: int = 8
    q_type: QuantTypeEnum = QuantTypeEnum.sym
    q_groups: int = 1


class WeightQuantConfig(BaseQuantConfig):
    enabled: bool = True
    quantized_initialization: Dict = {}
    post_init_quant: Dict = {}


class ActivationQuantConfig(BaseQuantConfig):
    enabled: bool = True


class QKVQuantConfig(DeepSpeedConfigModel):
    enabled: bool = True


class QuantizationConfig(DeepSpeedConfigModel):
    enabled: bool = True
    activation: ActivationQuantConfig = ActivationQuantConfig()
    weight: WeightQuantConfig = WeightQuantConfig()
    qkv: QKVQuantConfig = QKVQuantConfig()


# todo: brainstorm on how to do ckpt loading for DS inference
class InferenceCheckpointConfig(DeepSpeedConfigModel):
    checkpoint_dir: Optional[str] = None
    save_mp_checkpoint_path: Optional[str] = None
    base_dir: Optional[str] = None


class DeepSpeedInferenceConfig(DeepSpeedConfigModel):
    """ Sets parameters for DeepSpeed Inference Engine. """

    replace_with_kernel_inject: bool = Field(False, alias="kernel_inject")
    """
    Set to true to inject inference kernels for models such as, Bert, GPT2,
    GPT-Neo and GPT-J.  Otherwise, the injection_dict provides the names of two
    linear layers as a tuple:
    `(attention_output projection, transformer output projection)`
    """

    dtype: torch.dtype = torch.float16
    """
    Desired model data type, will convert model to this type.
    Supported target types: `torch.half`, `torch.int8`, `torch.float`
    """

    tensor_parallel: DeepSpeedTPConfig = Field({}, alias="tp")
    """
    Configuration for tensor parallelism used to split the model across several
    GPUs. Expects a dictionary containing values for :any:`DeepSpeedTPConfig`.
    """

    enable_cuda_graph: bool = False
    """
    Use this flag for capturing the CUDA-Graph of the inference ops, so that it
    can run faster using the graph replay method.
    """

    use_triton: bool = False
    """
    Use this flag to use triton kernels for inference ops.
    """

    triton_autotune: bool = False
    """
    Use this flag to enable triton autotuning.
    Turning it on is better for performance but increase the 1st runtime for
    autotuning.
    """

    zero: DeepSpeedZeroConfig = {}
    """
    ZeRO configuration to use with the Inference Engine. Expects a dictionary
    containing values for :any:`DeepSpeedZeroConfig`.
    """

    triangular_masking: bool = Field(True, alias="tm")
    """
    Controls the type of masking for attention scores in transformer layer.
    Note that the masking is application specific.
    """

    moe: Union[bool, DeepSpeedMoEConfig] = {}
    """
    Specify if the type of Transformer is MoE. Expects a dictionary containing
    values for :any:`DeepSpeedMoEConfig`.
    """

    quant: QuantizationConfig = {}
    """
    NOTE: only works for int8 dtype.
    Quantization settings used for quantizing your model using the MoQ.  The
    setting can be one element or a tuple. If one value is passed in, we
    consider it as the number of groups used in quantization. A tuple is passed
    in if we want to mention that there is extra-grouping for the MLP part of a
    Transformer layer (e.g. (True, 8) shows we quantize the model using 8
    groups for all the network except the MLP part that we use 8 extra
    grouping). Expects a dictionary containing values for
    :any:`QuantizationConfig`.
    """

    #todo: refactor the following 3 into the new checkpoint_config
    checkpoint: Optional[Union[str, Dict]] = None
    """
    Path to deepspeed compatible checkpoint or path to JSON with load policy.
    """

    base_dir: str = ""
    """
    This shows the root directory under which all the checkpoint files exists.
    This can be passed through the json config too.
    """

    set_empty_params: bool = False
    """
    specifying whether the inference-module is created with empty or real Tensor
    """

    save_mp_checkpoint_path: Optional[str] = None
    """
    The path for which we want to save the loaded model with a checkpoint. This
    feature is used for adjusting the parallelism degree to help alleviate the
    model loading overhead. It does not save any new checkpoint if no path is
    passed.
    """

    checkpoint_config: InferenceCheckpointConfig = Field({}, alias="ckpt_config")
    """
    TODO: Add docs. Expects a dictionary containing values for
    :any:`InferenceCheckpointConfig`.
    """

    return_tuple: bool = True
    """
    Specify whether or not the transformer layers need to return a tuple or a
    Tensor.
    """

    training_mp_size: int = 1
    """
    If loading a checkpoint this is the mp size that it was trained with, it
    may be different than what the mp size that you want to use during
    inference.
    """

    replace_method: str = Field(
        "auto",
        json_schema_extra={
            "deprecated": True,
            "deprecated_msg": "This parameter is no longer needed, please remove from your call to DeepSpeed-inference"
        })

    injection_policy: Optional[Dict] = Field(None, alias="injection_dict")
    """
    Dictionary mapping a client nn.Module to its corresponding injection
    policy. e.g., `{BertLayer : deepspeed.inference.HFBertLayerPolicy}`
    """

    injection_policy_tuple: Optional[tuple] = None
    """ TODO: Add docs """

    config: Optional[Dict] = Field(None, alias="args")  # todo: really no need for this field if we can refactor

    max_out_tokens: int = Field(1024, alias="max_tokens")
    """
    This argument shows the maximum number of tokens inference-engine can work
    with, including the input and output tokens. Please consider increasing it
    to the required token-length required for your use-case.
    """

    min_out_tokens: int = Field(1, alias="min_tokens")
    """
    This argument communicates to the runtime the minimum number of tokens you
    expect you will need to generate. This will cause the runtime to error
    if it unable to provide this and provide context on the memory pressure
    rather than seg-faulting or providing corrupted output.
    """

    transposed_mode: bool = Field(False, alias="transposed_mode")

    mp_size: int = Field(1, json_schema_extra={"deprecated": True, "new_param": "tensor_parallel.tp_size"})
    """
    Desired model parallel size, default is 1 meaning no model parallelism.
    Deprecated, please use the ``tensor_parallel` config to control model
    parallelism.
    """
    mpu: object = Field(None, json_schema_extra={"deprecated": True, "new_param": "tensor_parallel.mpu"})
    ep_size: int = Field(1, json_schema_extra={"deprecated": True, "new_param": "moe.ep_size"})
    ep_group: object = Field(None,
                             alias="expert_group",
                             json_schema_extra={
                                 "deprecated": True,
                                 "new_param": "moe.ep_group"
                             })
    ep_mp_group: object = Field(None,
                                alias="expert_mp_group",
                                json_schema_extra={
                                    "deprecated": True,
                                    "new_param": "moe.ep_mp_group"
                                })
    moe_experts: list = Field([1], json_schema_extra={"deprecated": True, "new_param": "moe.moe_experts"})
    moe_type: MoETypeEnum = Field(MoETypeEnum.standard,
                                  json_schema_extra={
                                      "deprecated": True,
                                      "new_param": "moe.type"
                                  })

    @field_validator("dtype", mode="before")
    def validate_dtype(cls, field_value, values):
        if isinstance(field_value, str):
            return DtypeEnum.from_str(field_value).value[0]
        if isinstance(field_value, torch.dtype):
            return field_value
        raise TypeError(f"Invalid type for dtype: {type(field_value)}")

    @field_validator("moe")
    def moe_backward_compat(cls, field_value, values):
        if isinstance(field_value, bool):
            return DeepSpeedMoEConfig(moe=field_value)
        return field_value

    @field_validator("use_triton")
    def has_triton(cls, field_value, values):
        if field_value and not deepspeed.HAS_TRITON:
            raise ValueError('Triton needs to be installed to use deepspeed with triton kernels')
        return field_value
