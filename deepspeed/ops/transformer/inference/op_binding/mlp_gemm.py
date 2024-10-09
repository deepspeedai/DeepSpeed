# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import torch
import torch.nn.functional as F
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp
from deepspeed.utils.types import NormType
from .pre_rms_norm import PreRMSNormOp


class MLPGemmOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(MLPGemmOp, self).__init__(config)
        try:
            if self.config.norm_type == NormType.LayerNorm:
                if self.config.dtype in [
                        torch.float16, torch.int8
                ]:  # non-triton cuda kernel has a higher performance in MLP than mlp_gemm_func in triton.ops
                    self.mlp_gemm_func = self.inference_module.mlp_gemm_fp16  # type: ignore
                elif self.config.dtype == torch.bfloat16:
                    self.mlp_gemm_func = self.inference_module.mlp_gemm_bf16
                else:
                    self.mlp_gemm_func = self.inference_module.mlp_gemm_fp32  # type: ignore
            elif self.config.norm_type == NormType.RMSNorm:
                if self.config.dtype in [torch.float16, torch.int8]:
                    self.mlp_gemm_func = self.inference_module.rms_mlp_gemm_fp16  # type: ignore
                elif self.config.dtype == torch.bfloat16:
                    self.mlp_gemm_func = self.inference_module.rms_mlp_gemm_bf16
                else:
                    self.mlp_gemm_func = self.inference_module.rms_mlp_gemm_fp32  # type: ignore
        except AttributeError:
            if self.config.norm_type == NormType.LayerNorm:
                self.mlp_gemm_func = self.mlp_gemm_fallback
            elif self.config.norm_type == NormType.RMSNorm:
                self.mlp_gemm_func = self.rms_mlp_gemm_fallback
        self.pre_rms_norm = PreRMSNormOp()

    def mlp_gemm_fallback(self, input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps,
                          pre_layer_norm, mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type,
                          transpose):
        if mlp_after_attn:
            residual_add = F.layer_norm(input + residual + input_bias, (input.shape[2], ), gamma, beta, eps)
            tmp = torch.matmul(residual_add, weight_interm.t() if transpose else weight_interm)
            tmp = F.gelu(tmp + bias)
            output = torch.matmul(tmp, weight_out.t() if transpose else weight_out)

            return output, residual_add
        else:
            raise NotImplementedError

    def rms_mlp_gemm_fallback(self, input, residual, weight_interm, weight_out, gamma, eps, interm_scale, out_scale,
                              dtype, mlp_act_func_type, transpose):
        inp_norm, residual = self.pre_rms_norm(input, residual, gamma, eps)
        tmp = torch.matmul(inp_norm.view([-1, inp_norm.size(2)]), weight_interm.t() if transpose else weight_interm)
        up_proj, gate_proj = tmp.chunk(2, dim=1)

        from deepspeed.utils.types import ActivationFuncType
        if mlp_act_func_type == ActivationFuncType.GELU:
            intermediate = F.gelu(gate_proj)
        elif mlp_act_func_type == ActivationFuncType.ReLU:
            intermediate = F.relu(gate_proj)
        elif mlp_act_func_type == ActivationFuncType.GATED_GELU:
            intermediate = F.gelu(gate_proj)
        elif mlp_act_func_type == ActivationFuncType.GATED_SILU:
            intermediate = F.silu(gate_proj)
        else:
            raise f"rms_mlp_gemm_fallback not implemented for activation type {mlp_act_func_type}"

        intermediate = intermediate * up_proj

        output = torch.matmul(intermediate, weight_out.t() if transpose else weight_out)
        output = output.view([input.size(0), input.size(1), -1])

        return [output, residual]

    def forward(self,
                input: torch.Tensor,
                residual: torch.Tensor,
                weight_interm: torch.Tensor,
                weight_out: torch.Tensor,
                input_bias: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None,
                gamma: Optional[torch.Tensor] = None,
                beta: Optional[torch.Tensor] = None):
        if self.config.norm_type == NormType.LayerNorm:
            output, residual_add = self.mlp_gemm_func(
                input,
                residual,
                input_bias,
                weight_interm,
                weight_out,
                bias,
                gamma,
                beta,
                self.config.epsilon,
                self.config.pre_layer_norm,
                self.config.mlp_after_attn,
                weight_interm.scale if hasattr(weight_interm, 'scale') else torch.empty(1),  # type: ignore
                weight_out.scale if hasattr(weight_out, 'scale') else torch.empty(1),  # type: ignore
                self.config.dtype == torch.int8,
                self.config.mlp_act_func_type,
                self.config.transposed_mode)
        else:
            if input_bias is not None:
                input += input_bias
            output, residual_add = self.mlp_gemm_func(
                input,
                residual,
                weight_interm,
                weight_out,
                gamma,
                self.config.epsilon,
                weight_interm.scale if hasattr(weight_interm, 'scale') else torch.empty(1),  # type: ignore
                weight_out.scale if hasattr(weight_out, 'scale') else torch.empty(1),  # type: ignore
                self.config.dtype == torch.int8,
                self.config.mlp_act_func_type,
                self.config.transposed_mode)
        return output, residual_add
