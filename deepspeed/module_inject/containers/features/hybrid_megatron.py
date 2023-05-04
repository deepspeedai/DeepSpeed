# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from .hybrid_engine import HybridEngineContainer
from .megatron import MegatronContainer


class HybridMegatronContainer(MegatronContainer, HybridEngineContainer):

    def _align_qkv(self, x: torch.Tensor):
        """
        Internal helper for accepting the head-contiguous weight matrix and chunking
        the query, key, and value components.
        """
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

    def transform_for_inference(self) -> None:
        """
        Overrides the HybridEngineContainer implementation.

        The alternative layout of the QKV matrix for Megatron is such that each head's Q, K, and V
        are sequential in memory. This is different from the default layout in which all of the Qs
        are sequential, followed by all of the Ks, and then all of the Vs. Here, we take the default
        layout and transform it to the inference layout.
        """
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

    def _partition_qkv(self, x: torch.Tensor):
        """
        Internal helper for taking contiguous QKV and partitioning it for contiguous
        heads.
        """
        q_k_v = torch.split(x, (x.shape[0] // 3), dim=0)
        attention_head_size = q_k_v[0].shape[0] // self.num_attention_heads
        new_x_shape = (self.num_attention_heads, attention_head_size) + x.size()[1:]
        q, k, v = [data.view(*new_x_shape) for data in q_k_v]
        if len(q.shape) > 2:
            x.data.copy_(torch.cat((q, k, v), dim=-2).reshape(-1, q.shape[-1]))
        else:
            x.data.copy_(torch.cat((q, k, v), dim=-1).reshape(-1))

    def transform_for_training(self):
        """
        Overrides the HybridEngineContainer implementation.

        The alternative layout of the QKV matrix for Megatron is such that each head's Q, K, and V
        are sequential in memory. This is different from the default layout in which all of the Qs
        are sequential, followed by all of the Ks, and then all of the Vs. This function takes the inference format and reverts it back to the default format.
        """
        # If parameter is distributed, handle gathering it
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
