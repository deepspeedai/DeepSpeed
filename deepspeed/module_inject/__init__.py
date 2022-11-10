from .replace_module import replace_transformer_layer, revert_transformer_layer, ReplaceWithTensorSlicing, GroupQuantizer
from .module_quantize import quantize_transformer_layer
from .replace_policy import DSPolicy, HFBertLayerPolicy
from .layers import LinearAllreduce, LinearLayer, EmbeddingLayer, Normalize
