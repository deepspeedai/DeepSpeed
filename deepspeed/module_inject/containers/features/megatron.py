# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from abc import ABC


class MegatronContainer(ABC):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.megatron_v2 = self.policy.is_megatron_v2

    def _align_qkv_transposed(self, x):
        attention_head_size = x.shape[-1] // self.num_attention_heads
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x_1 = x.view(*new_x_shape)
        (q, k, v) = torch.split(x_1, (x_1.shape[-1] // 3), dim=(x_1.dim() - 1))
        if len(q.shape) > 2:
            return torch.cat((q.reshape(q.shape[0], -1), k.reshape(q.shape[0], -1), v.reshape(q.shape[0], -1)),
                             dim=-1).reshape(x.shape)
        else:
            return torch.cat((q.reshape(-1), k.reshape(-1), v.reshape(-1)), dim=-1).reshape(x.shape)

    def _align_qkv(self, x):
        attention_head_size = x.shape[0] // self.num_attention_heads
        new_x_shape = (self.num_attention_heads, attention_head_size) + x.size()[1:]
        x_1 = x.view(*new_x_shape)
        div_dim = len(x_1.size()) - 2 if len(x.shape) == 2 else -1
        (q, k, v) = torch.split(x_1, (x_1.shape[div_dim] // 3), dim=div_dim)
        if len(q.shape) > 2:
            x.data.copy_(
                torch.cat((q.reshape(-1, q.shape[-1]), k.reshape(-1, q.shape[-1]), v.reshape(-1, q.shape[-1])),
                          dim=0).reshape(x.shape))
        else:
            x.data.copy_(torch.cat((q.reshape(-1), k.reshape(-1), v.reshape(-1)), dim=-1).reshape(x.shape))

    def _align_merged_qkv(self):
        if hasattr(self.qkvw, 'ds_id'):
            from deepspeed.runtime.zero import GatheredParameters
            from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
            param_list = [self.qkvw, self.qkvb]
            non_active_params = [param for param in param_list if (hasattr(param, 'ds_id') and \
                            param.ds_status == ZeroParamStatus.NOT_AVAILABLE)]
            with GatheredParameters(non_active_params):
                self._align_qkv(self.qkvw)
                self._align_qkv(self.qkvb)
        else:
            self._align_qkv(self.qkvw)
            self._align_qkv(self.qkvb)

    def _partition_qkv(self, x):
        q_k_v = torch.split(x, (x.shape[0] // 3), dim=0)
        attention_head_size = q_k_v[0].shape[0] // self.num_attention_heads
        new_x_shape = (self.num_attention_heads, attention_head_size) + x.size()[1:]
        q, k, v = [data.view(*new_x_shape) for data in q_k_v]
        if len(q.shape) > 2:
            x.data.copy_(torch.cat((q, k, v), dim=-2).reshape(-1, q.shape[-1]))
        else:
            x.data.copy_(torch.cat((q, k, v), dim=-1).reshape(-1))

    def _partition_merged_qkv(self):
        if hasattr(self.qkvw, 'ds_id'):
            from deepspeed.runtime.zero import GatheredParameters
            from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
            param_list = [self.qkvw, self.qkvb]
            non_active_params = [param for param in param_list if (hasattr(param, 'ds_id') and \
                            param.ds_status == ZeroParamStatus.NOT_AVAILABLE)]
            with GatheredParameters(non_active_params):
                self._partition_qkv(self.qkvw)
                self._partition_qkv(self.qkvb)
        else:
            self._partition_qkv(self.qkvw)
            self._partition_qkv(self.qkvb)

    def transpose(self):
        super().transpose()
        if self.megatron_v2:
            self.qkvw = torch.nn.parameter.Parameter(self._align_qkv_transposed(self.qkvw).contiguous())
            self.qkvb = torch.nn.parameter.Parameter(self._align_qkv_transposed(self.qkvb).contiguous())
