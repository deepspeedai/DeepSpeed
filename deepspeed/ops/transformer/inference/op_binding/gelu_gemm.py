import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class GELUGemmOp(BaseOp):
    def __init__(self, config: DeepSpeedInferenceConfig):
        super(GELUGemmOp, self).__init__(config)
        if self.config.fp16:
            self.fused_gemm_gelu = self.inference_cuda_module.fused_gemm_gelu_fp16
        else:
            self.fused_gemm_gelu = self.inference_cuda_module.fused_gemm_gelu_fp32

    def forward(self,
                input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor,
                weight_out: torch.Tensor,
                async_op: bool = False):
        output = self.fused_gemm_gelu(input,
                                      weight,
                                      weight.scale,
                                      bias,
                                      weight_out,
                                      weight_out.scale,
                                      self.config.epsilon,
                                      self.config.pre_layer_norm,
                                      self.config.q_int8,
                                      async_op)
        return output
