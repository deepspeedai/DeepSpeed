# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .replace_module import replace_transformer_layer, revert_transformer_layer, ReplaceWithTensorSlicing, GroupQuantizer, generic_injection
from .module_quantize import quantize_transformer_layer
from .replace_policy import HFBertLayerPolicy
from .layers import LinearAllreduce, LinearLayer, EmbeddingLayer, Normalize
from .policy import DSPolicy
