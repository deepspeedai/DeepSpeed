# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from .builder import SYCLOpBuilder


class FlashAttentionBuilderObject():

    def __init__(self):
        pass

    # general functions
    def flash_attn_func_v2(self, q, k, v, dropout_p, softmax_scale, is_causal):
        try:
            import torch
            import intel_extension_for_pytorch  # noqa
            return torch.nn.functional.scaled_dot_product_attention(q,
                                                                    k,
                                                                    v,
                                                                    dropout_p=dropout_p,
                                                                    is_causal=is_causal,
                                                                    scale=softmax_scale)
        except ImportError:
            raise ImportError(
                "Please install pytorch and intel_extension_for_pytorch to include scaled dot product attention.")


    def flash_attn_fwd(self, q, k, v, bias=None, dropout_p=0.0, is_causual=False, softmax_scale=None):
        try:
            import torch
            import intel_extension_for_pytorch  # noqa
            return torch.xpu.IpexSDP_forward(q, k, v, bias, dropout_p, is_causual, softmax_scale)
        except ImportError:
            raise ImportError(
                "Please install pytorch and intel_extension_for_pytorch to include scaled dot product attention.")

    def flash_attn_bwd(self, out, out_grad, q, k, v, bias, logsumexp, seed, offset, dropout_p, is_bais_grad, is_causal, softmax_scale):
        try:
            import torch
            import intel_extension_for_pytorch  # noqa
            return torch.xpu.IpexSDP_backward(out, out_grad, q, k, v, bias, logsumexp, seed, offset, dropout_p, is_bais_grad, is_causal, softmax_scale)
        except ImportError:
            raise ImportError(
                "Please install pytorch and intel_extension_for_pytorch to include scaled dot product attention.")


class FlashAttentionBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_FlashAttention"
    NAME = "flash_attn"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return

    def include_paths(self):
        return []

    def extra_ldflags(self):
        return []

    def cxx_args(self):
        return []

    def load(self):
        return FlashAttentionBuilderObject()
