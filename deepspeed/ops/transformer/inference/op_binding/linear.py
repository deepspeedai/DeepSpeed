import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class LinearOp(BaseOp):
    def __init__(self, config: DeepSpeedInferenceConfig):
        super(LinearOp, self).__init__(config)
        if self.config.fp16:
            self.linear_func = self.inference_cuda_module.linear_layer_fp16
        else:
            self.linear_func = self.inference_cuda_module.linear_layer_fp32

    def forward(self,
                input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor,
                add_bias: bool,
                do_flash_attn: bool,
                num_heads: int,
                external_cache: bool = None,
                num_layers: int = None):
        qkv_out = self.linear_func(input,
                                   weight,
                                   bias,
                                   add_bias,
                                   do_flash_attn,
                                   num_heads)
        return qkv_out
