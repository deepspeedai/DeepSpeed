# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch.nn.parameter import Parameter

from ..policy import DSPolicy
from ...model_implementations.diffusers.unet import DSUNet


class UNetPolicy(DSPolicy):

    def __init__(self):
        super().__init__()
        try:
            import diffusers
            self._orig_layer_class = diffusers.models.unet_2d_condition.UNet2DConditionModel
        except AttributeError:
            self._orig_layer_class = diffusers.models.unets.unet_2d_condition.UNet2DConditionModel
        except ImportError:
            self._orig_layer_class = None

    def match(self, module):
        return isinstance(module, self._orig_layer_class)

    def match_replaced(self, module):
        return isinstance(module, DSUNet)

    def apply(self, module, enable_cuda_graph=True):
        # TODO(cmikeh2): Enable cuda graph should be an inference configuration
        return DSUNet(module, enable_cuda_graph=enable_cuda_graph)

    def attention(self, client_module):
        qw = client_module.to_q.weight
        kw = client_module.to_k.weight
        vw = client_module.to_v.weight

        if qw.shape[1] == kw.shape[1]:
            qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)

            return qkvw, \
                   client_module.to_out[0].weight, \
                   client_module.to_out[0].bias, \
                   qw.shape[-1], \
                   client_module.heads
        else:
            #return None
            #kvw = Parameter(torch.cat((kw, vw), dim=0), requires_grad=False)
            return qw, \
                   kw, vw, \
                   client_module.to_out[0].weight, \
                   client_module.to_out[0].bias, \
                   qw.shape[-1], \
                   client_module.heads
