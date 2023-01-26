import torch
from deepspeed import comm as dist
from torch import nn
from torch.nn import functional as F

from torch.nn.parameter import Parameter
from deepspeed.accelerator import get_accelerator


class LinearAllreduce(nn.Module):
    def __init__(self, weight, bias=None, mp_group=None):
        super(LinearAllreduce, self).__init__()
        self.weight = weight
        self.bias = bias
        self.mp_group = mp_group

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.mp_group is not None:
            dist.all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output += self.bias
        return output


class LinearLayer(nn.Module):
    def __init__(self, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        super(LinearLayer, self).__init__()
        if weight is not None:
            self.weight = weight
            self.bias = bias
        else:
            self.weight = Parameter(
                torch.empty(weight_shape,
                            dtype=dtype,
                            device=get_accelerator().current_device_name()))

            self.bias = Parameter(
                torch.empty(weight_shape[0],
                            dtype=dtype,
                            device=get_accelerator().current_device_name())) \
                if bias is not None else None

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.bias is not None:
            output += self.bias
        return output


class Normalize(nn.Module):
    def __init__(self, dim, dtype=torch.float, eps=1e-5):
        super(Normalize, self).__init__()
        self.norm = nn.LayerNorm(dim,
                                 eps=eps).to(dtype).to(
                                     get_accelerator().current_device_name())
        self.weight = self.norm.weight
        self.bias = self.norm.bias

    def forward(self, input):
        return self.norm(input)


class EmbeddingLayer(nn.Module):
    def __init__(self, weight_shape, dtype=torch.half):
        super(EmbeddingLayer, self).__init__()
        self.weight = Parameter(
            torch.empty(weight_shape[0],
                        weight_shape[1],
                        dtype=dtype,
                        device=get_accelerator().current_device_name()))

    def forward(self, input):
        return F.embedding(input, self.weight)


class OPTEmbedding(EmbeddingLayer):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, weight_shape):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(weight_shape)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask,
                                  dim=1).type_as(attention_mask) *
                     attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)
