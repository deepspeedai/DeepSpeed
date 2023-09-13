# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from __future__ import absolute_import, division, print_function, unicode_literals
# Copyright The Microsoft DeepSpeed Team
# DeepSpeed note, code taken from commit 3d59216cec89a363649b4fe3d15295ba936ced0f
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/modeling.py

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

import copy
import json
import logging
import math
from io import open

import torch
from torch import nn
from torch.utils import checkpoint

from torch.nn import Module
import torch.nn.functional as F
import torch.nn.init as init

#from numba import cuda

#from deepspeed_cuda import DeepSpeedSoftmaxConfig, DeepSpeedSoftmax
from deepspeed.accelerator import get_accelerator

logger = logging.getLogger(__name__)
"""
@torch.jit.script
def f_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))
@torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))
@torch.jit.script
def bias_tanh(bias, y):
    x = bias + y
    return torch.tanh(x)
 """


def f_gelu(x):
    x_type = x.dtype
    x = x.float()
    x = x * 0.5 * (1.0 + torch.erf(x / 1.41421))
    return x.to(x_type)


def bias_gelu(bias, y):
    y_type = y.dtype
    x = bias.float() + y.float()
    x = x * 0.5 * (1.0 + torch.erf(x / 1.41421))
    return x.to(y_type)


def bias_tanh(bias, y):
    y_type = y.dtype
    x = bias.float() + y.float()
    x = torch.tanh(x)
    return x.to(y_type)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return f_gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class GPUTimer:

    def __init__(self):
        super().__init__()
        self.start = get_accelerator().Event()  # noqa: F821
        self.stop = get_accelerator().Event()  # noqa: F821

    def record(self):
        self.start.record()

    def elapsed(self):
        self.stop.record()
        self.stop.synchronize()
        return self.start.elapsed_time(self.stop) / 1000.0


class LinearActivation(Module):
    r"""Fused Linear and activation Module.
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, weights, biases, act='gelu', bias=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fused_gelu = False
        self.fused_tanh = False
        if isinstance(act, str):
            if bias and act == 'gelu':
                self.fused_gelu = True
            elif bias and act == 'tanh':
                self.fused_tanh = True
            else:
                self.act_fn = ACT2FN[act]
        else:
            self.act_fn = act
        #self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight = weights[5]
        self.bias = biases[5]
        #if bias:
        #    self.bias = Parameter(torch.Tensor(out_features))
        #else:
        #    self.register_parameter('bias', None)
        #self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.fused_gelu:
            #timing = []
            #t1 = GPUTimer()
            #t1.record()
            y = F.linear(input, self.weight, None)
            #timing.append(t1.elapsed())
            #t1.record()
            bg = bias_gelu(self.bias, y)
            #timing.append(t1.elapsed())
            return bg
        elif self.fused_tanh:
            return bias_tanh(self.bias, F.linear(input, self.weight, None))
        else:
            return self.act_fn(F.linear(input, self.weight, self.bias))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias
                                                                 is not None)


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 batch_size=8,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 fp16=False):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            self.__dict__.update(json_config)
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.batch_size = batch_size
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.fp16 = fp16
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        config.__dict__.update(json_object)
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    import apex
    #apex.amp.register_half_function(apex.normalization.fused_layer_norm, 'FusedLayerNorm')
    import apex.normalization
    #apex.amp.register_float_function(apex.normalization.FusedLayerNorm, 'forward')
    BertLayerNorm = apex.normalization.FusedLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")

    class BertLayerNorm(nn.Module):

        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertSelfAttention(nn.Module):

    def __init__(self, i, config, weights, biases):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention "
                             "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.query.weight = weights[0]
        self.query.bias = biases[0]
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.key.weight = weights[1]
        self.key.bias = biases[1]
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.value.weight = weights[2]
        self.value.bias = biases[2]

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)
        #self.softmax_config = DeepSpeedSoftmaxConfig()
        #self.softmax_config.batch_size = config.batch_size
        #self.softmax_config.max_seq_length = config.max_position_embeddings
        #self.softmax_config.hidden_size = config.hidden_size
        #self.softmax_config.heads = config.num_attention_heads
        #self.softmax_config.softmax_id = i
        #self.softmax_config.fp16 = config.fp16
        #self.softmax_config.prob_drop_out = 0.0
        #self.softmax = DeepSpeedSoftmax(i, self.softmax_config)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 3, 1)

    def forward(self, hidden_states, attention_mask, grads=None):

        mixed_query_layer = self.query(hidden_states)

        mixed_key_layer = self.key(hidden_states)

        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        key_layer = self.transpose_key_for_scores(mixed_key_layer)

        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer1 = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer1.size()[:-2] + (self.all_head_size, )
        context_layer1 = context_layer1.view(*new_context_layer_shape)

        return context_layer1


class BertSelfOutput(nn.Module):

    def __init__(self, config, weights, biases):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense.weight = weights[3]
        self.dense.bias = biases[3]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def get_w(self):
        return self.dense.weight


class BertAttention(nn.Module):

    def __init__(self, i, config, weights, biases):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(i, config, weights, biases)
        self.output = BertSelfOutput(config, weights, biases)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

    def get_w(self):
        return self.output.get_w()


class BertIntermediate(nn.Module):

    def __init__(self, config, weights, biases):
        super(BertIntermediate, self).__init__()
        self.dense_act = LinearActivation(config.hidden_size,
                                          config.intermediate_size,
                                          weights,
                                          biases,
                                          act=config.hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config, weights, biases):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense.weight = weights[6]
        self.dense.bias = biases[6]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, i, config, weights, biases):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(i, config, weights, biases)
        self.intermediate = BertIntermediate(config, weights, biases)
        self.output = BertOutput(config, weights, biases)
        self.weight = weights
        self.biases = biases

    def forward(self, hidden_states, attention_mask, grads, collect_all_grads=False):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        if collect_all_grads:
            # self.weight[0].register_hook(lambda x, self=self: grads.append([x,"Q_W"]))
            # self.biases[0].register_hook(lambda x, self=self: grads.append([x,"Q_B"]))
            # self.weight[1].register_hook(lambda x, self=self: grads.append([x,"K_W"]))
            # self.biases[1].register_hook(lambda x, self=self: grads.append([x,"K_B"]))
            self.weight[2].register_hook(lambda x, self=self: grads.append([x, "V_W"]))
            self.biases[2].register_hook(lambda x, self=self: grads.append([x, "V_B"]))
            self.weight[3].register_hook(lambda x, self=self: grads.append([x, "O_W"]))
            self.biases[3].register_hook(lambda x, self=self: grads.append([x, "O_B"]))
            self.attention.output.LayerNorm.weight.register_hook(lambda x, self=self: grads.append([x, "N2_W"]))
            self.attention.output.LayerNorm.bias.register_hook(lambda x, self=self: grads.append([x, "N2_B"]))
            self.weight[5].register_hook(lambda x, self=self: grads.append([x, "int_W"]))
            self.biases[5].register_hook(lambda x, self=self: grads.append([x, "int_B"]))
            self.weight[6].register_hook(lambda x, self=self: grads.append([x, "out_W"]))
            self.biases[6].register_hook(lambda x, self=self: grads.append([x, "out_B"]))
            self.output.LayerNorm.weight.register_hook(lambda x, self=self: grads.append([x, "norm_W"]))
            self.output.LayerNorm.bias.register_hook(lambda x, self=self: grads.append([x, "norm_B"]))

        return layer_output

    def get_w(self):
        return self.attention.get_w()


class BertEncoder(nn.Module):

    def __init__(self, config, weights, biases):
        super(BertEncoder, self).__init__()
        #layer = BertLayer(config, weights, biases)
        self.FinalLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.layer = nn.ModuleList(
            [copy.deepcopy(BertLayer(i, config, weights, biases)) for i in range(config.num_hidden_layers)])
        self.grads = []
        self.graph = []

    def get_grads(self):
        return self.grads

    def get_modules(self, big_node, input):
        for mdl in big_node.named_children():
            self.graph.append(mdl)
            self.get_modules(self, mdl, input)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, checkpoint_activations=False):
        all_encoder_layers = []

        def custom(start, end):

            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_

            return custom_forward

        if checkpoint_activations:
            l = 0
            num_layers = len(self.layer)
            chunk_length = math.ceil(math.sqrt(num_layers))
            while l < num_layers:
                hidden_states = checkpoint.checkpoint(custom(l, l + chunk_length), hidden_states, attention_mask * 1)
                l += chunk_length
            # decoder layers
        else:
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(hidden_states, attention_mask, self.grads, collect_all_grads=True)
                hidden_states.register_hook(lambda x, i=i, self=self: self.grads.append([x, "hidden_state"]))
                #print("pytorch weight is: ", layer_module.get_w())

                if output_all_encoded_layers:
                    all_encoder_layers.append((hidden_states))

        if not output_all_encoded_layers or checkpoint_activations:
            all_encoder_layers.append((hidden_states))
        return all_encoder_layers
