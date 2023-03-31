# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from torch import nn
from deepspeed.ops.sparse_attention import SparseSelfAttention, FixedSparsityConfig


class BertSparseSelfAttention(nn.Module):
    """Implements Sparse Self Attention layer of Bert model based on https://github.com/microsoft/DeepSpeedExamples/blob/master/bing_bert/nvidia/modelingpreln.py#L373

    For more information please see, TODO DeepSpeed Sparse Transformer.

    For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial.
    """

    def __init__(
        self,
        config,
        # SparsityConfig parameters needs to be set accordingly
        sparsity_config=FixedSparsityConfig(num_heads=4)):
        """Initialize the bert sparse self attention layer.

        Note) you can use any of the provided sparsity configs or simply add yours!

        Arguments:
            config: required: Bert model config
            sparsity_config: optional: this parameter determines sparsity pattern configuration; it is based on FixedSparsityConfig class.
        """

        super(BertSparseSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention "
                             "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.sparse_self_attention = SparseSelfAttention(sparsity_config)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        """Applies forward phase of bert sparse self attention

        Arguments:
            hidden_states: required: hidden_states tensor of the bert model
            attn_mask: required: a mask tensor of size (SequenceLength X SequenceLength); currently only 2D is supported

        Return:
             context_layer: a dense tensor containing attention context
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        context_layer = self.sparse_self_attention(query_layer,
                                                   key_layer,
                                                   value_layer,
                                                   key_padding_mask=attention_mask)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
