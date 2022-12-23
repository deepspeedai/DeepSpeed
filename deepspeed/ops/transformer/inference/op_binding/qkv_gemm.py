import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp
from deepspeed import comm as dist


class QKVGemmOp(BaseOp):
    def __init__(self, config: DeepSpeedInferenceConfig):
        super(QKVGemmOp, self).__init__(config)
        if self.config.fp16:
            self.qkv_gemm_func = self.inference_cuda_module.qkv_gemm_fp16
        else:
            self.qkv_gemm_func = self.inference_cuda_module.qkv_gemm_fp32

    def forward(self,
                input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor,
                gamma: torch.Tensor,
                beta: torch.Tensor,
                add_bias: bool,
                num_layers: int,
                num_heads: int = None,
                max_out_tokens: int = None):
        q_scale = weight.scale
        external_cache = self.config.bigscience_bloom
        rank = dist.get_rank() if dist.is_initialized() else 0
        q_int8 = self.config.q_int8
        output = self.qkv_gemm_func(input,
                                    weight,
                                    q_scale,
                                    bias,
                                    gamma,
                                    beta,
                                    self.config.epsilon,
                                    add_bias,
                                    num_layers,
                                    external_cache,
                                    self.config.mp_size,
                                    rank,
                                    q_int8)
        return output
