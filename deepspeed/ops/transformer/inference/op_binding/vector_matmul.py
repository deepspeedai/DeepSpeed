import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class VectorMatMulOp(BaseOp):
    def __init__(self, config: DeepSpeedInferenceConfig):
        super(VectorMatMulOp, self).__init__(config)
        if self.config.fp16:
            self.vector_matmul_func = self.inference_cuda_module.vector_matmul_fp16
        else:
            self.vector_matmul_func = self.inference_cuda_module.vector_matmul_fp32

    def forward(self, input: torch.Tensor, weight: torch.Tensor, async_op: bool = False):
        q_scale = weight.scale
        q_int8 = self.config.q_int8
        output = self.vector_matmul_func(input, weight, async_op, q_scale, q_int8)
        return output
