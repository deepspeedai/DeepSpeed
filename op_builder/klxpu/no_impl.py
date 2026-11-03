# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

try:
    from op_builder import __deepspeed__  # noqa: F401 # type: ignore
    from op_builder.builder import OpBuilder
except ImportError:
    from deepspeed.ops.op_builder.builder import OpBuilder


class NotImplementedBuilder(OpBuilder):
    NAME = "not_implemented"
    BUILD_VAR = "DS_BUILD_NOT_IMPLEMENTED"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.comm.{self.NAME}_op'

    def sources(self):
        return []

    def cxx_args(self):
        return []

    def extra_ldflags(self):
        return []

    def include_paths(self):
        return []

    def load(self, verbose=True):
        raise NotImplementedError(f"'{self.name}' is not supported on the KLXPU accelerator backend.")

    def is_compatible(self, verbose=False):
        return False


# Named stubs for ops KLXPU does not implement, so ``deepspeed.ops.op_builder.<Name>``
# resolves to a real class (with NAME) instead of None and dependent code can query
# ``compatible_ops[NAME]`` (always False) and skip cleanly.
class FusedLambBuilder(NotImplementedBuilder):
    NAME = "fused_lamb"
    BUILD_VAR = "DS_BUILD_FUSED_LAMB"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)


class FusedLionBuilder(NotImplementedBuilder):
    NAME = "fused_lion"
    BUILD_VAR = "DS_BUILD_FUSED_LION"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)


class InferenceBuilder(NotImplementedBuilder):
    NAME = "transformer_inference"
    BUILD_VAR = "DS_BUILD_TRANSFORMER_INFERENCE"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)


class TransformerBuilder(NotImplementedBuilder):
    NAME = "transformer"
    BUILD_VAR = "DS_BUILD_TRANSFORMER"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)


class QuantizerBuilder(NotImplementedBuilder):
    NAME = "quantizer"
    BUILD_VAR = "DS_BUILD_QUANTIZER"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)


class FPQuantizerBuilder(NotImplementedBuilder):
    NAME = "fp_quantizer"
    BUILD_VAR = "DS_BUILD_FP_QUANTIZER"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)


class GDSBuilder(NotImplementedBuilder):
    NAME = "gds"
    BUILD_VAR = "DS_BUILD_GDS"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)


class EvoformerAttnBuilder(NotImplementedBuilder):
    NAME = "evoformer_attn"
    BUILD_VAR = "DS_BUILD_EVOFORMER_ATTN"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)


class SpatialInferenceBuilder(NotImplementedBuilder):
    NAME = "spatial_inference"
    BUILD_VAR = "DS_BUILD_SPATIAL_INFERENCE"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)


class RaggedOpsBuilder(NotImplementedBuilder):
    NAME = "ragged_device_ops"
    BUILD_VAR = "DS_BUILD_RAGGED_DEVICE_OPS"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)


class RaggedUtilsBuilder(NotImplementedBuilder):
    NAME = "ragged_ops"
    BUILD_VAR = "DS_BUILD_RAGGED_OPS"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)


class RandomLTDBuilder(NotImplementedBuilder):
    NAME = "random_ltd"
    BUILD_VAR = "DS_BUILD_RANDOM_LTD"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)


class StochasticTransformerBuilder(NotImplementedBuilder):
    NAME = "stochastic_transformer"
    BUILD_VAR = "DS_BUILD_STOCHASTIC_TRANSFORMER"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)


class InferenceCoreBuilder(NotImplementedBuilder):
    NAME = "inference_core_ops"
    BUILD_VAR = "DS_BUILD_INFERENCE_CORE_OPS"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)


class InferenceCutlassBuilder(NotImplementedBuilder):
    NAME = "cutlass_ops"
    BUILD_VAR = "DS_BUILD_CUTLASS_OPS"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)


class DeepCompileBuilder(NotImplementedBuilder):
    NAME = "dc"
    BUILD_VAR = "DS_BUILD_DEEP_COMPILE"

    def __init__(self, name=None):
        super().__init__(name=name if name is not None else self.NAME)
