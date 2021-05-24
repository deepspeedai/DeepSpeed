"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .cpu_adam import CPUAdamBuilder
from .fused_adam import FusedAdamBuilder
from .fused_lamb import FusedLambBuilder
from .sparse_attn import SparseAttnBuilder
from .transformer import TransformerBuilder
from .stochastic_transformer import StochasticTransformerBuilder
from .utils import UtilsBuilder
from .async_io import AsyncIOBuilder
from .builder import get_default_compute_capatabilities
from .transformer_inference import InferenceBuilder
from .quantizer import QuantizerBuilder

# TODO: infer this list instead of hard coded
# List of all available ops
__op_builders__ = [
    CPUAdamBuilder(),
    FusedAdamBuilder(),
    FusedLambBuilder(),
    SparseAttnBuilder(),
    TransformerBuilder(),
    StochasticTransformerBuilder(),
    AsyncIOBuilder(),
    InferenceBuilder(),
    UtilsBuilder(),
    QuantizerBuilder()
]
ALL_OPS = {op.name: op for op in __op_builders__}
