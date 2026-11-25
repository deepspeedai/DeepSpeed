# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Internal training quantizer selection and conversion lifecycle."""

from abc import ABC, abstractmethod


class TrainingQuantizer(ABC):
    """An adapter for an opt-in training quantization backend."""

    convert_after_device_placement: bool

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def validate_environment(self, engine, model):
        """Reject incompatible training options before mutating the model."""

    @abstractmethod
    def convert(self, engine, model):
        """Apply backend conversion after AutoEP and before optimizer setup."""


class SubchannelQuantizer(TrainingQuantizer):
    # The existing converter installs module overrides before device placement.
    convert_after_device_placement = False

    def validate_environment(self, engine, model):
        if engine.pipeline_parallelism:
            raise ValueError("quantization.subchannel does not support PipelineModule yet.")
        if engine.autotp_size() > 1:
            raise ValueError("quantization.subchannel does not support AutoTP yet: the tensor-parallel layers "
                             "replace nn.Linear and would run unquantized.")

    def convert(self, engine, model):
        from .subchannel import convert_to_subchannel_quantization

        convert_to_subchannel_quantization(model, self.config)


class TorchAOQuantizer(TrainingQuantizer):
    # AO selects and converts BF16 modules on their target device.
    convert_after_device_placement = True

    def validate_environment(self, engine, model):
        from deepspeed.moe.layer import MoE
        from deepspeed.runtime.zero.config import ZeroStageEnum

        if not engine.bfloat16_enabled():
            raise RuntimeError("HiFloat8 training requires BF16 model parameters")
        if engine.pipeline_parallelism:
            raise RuntimeError("HiFloat8 training has not been validated with DeepSpeed PipelineModule")
        if engine.autotp_size() > 1:
            raise RuntimeError("HiFloat8 training has not been validated with DeepSpeed AutoTP")
        if engine.mpu is not None:
            raise RuntimeError("HiFloat8 training has not been validated with a custom model-parallel unit")
        if any(isinstance(module, MoE) for module in model.modules()):
            raise RuntimeError("HiFloat8 experts require AutoEP, not the explicit DeepSpeed MoE layer")
        if engine.zero_optimization_stage() >= ZeroStageEnum.weights:
            raise RuntimeError("HiFloat8 phase-1 supports DeepSpeed ZeRO stages 0, 1, and 2 only")

    def convert(self, engine, model):
        from deepspeed.runtime.hifloat8 import configure_hifloat8

        engine.hifloat8_converted_module_names, engine.hifloat8_grouped_module_names = configure_hifloat8(
            model, self.config.hifloat8, engine.device)


_QUANTIZERS = {
    "torchao": TorchAOQuantizer,
    "subchannel": SubchannelQuantizer,
}


def create_training_quantizer(config):
    for name, quantizer in _QUANTIZERS.items():
        backend_config = getattr(config, name)
        if backend_config.enabled:
            return quantizer(backend_config)
    return None
