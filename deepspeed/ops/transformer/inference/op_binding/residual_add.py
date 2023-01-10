import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class ResidualAddOp(BaseOp):
    def __init__(self, config: DeepSpeedInferenceConfig):
        super(ResidualAddOp, self).__init__(config)
        if self.config.fp16 or self.config.q_int8:
            self.residual_add_func = self.inference_cuda_module.residual_add_bias_fp16
        else:
            self.residual_add_func = self.inference_cuda_module.residual_add_bias_fp32

    def forward(self,
                hidden_state: torch.Tensor,
                residual: torch.Tensor,
                attention_output: torch.Tensor,
                attention_bias: torch.Tensor,
                final_bias: torch.Tensor,
                add_bias: bool,
                residual_add: torch.Tensor):

        if not self.config.pre_layer_norm and residual_add is not None:
            # only use residual add if its set and we are not pre layer norm
            residual = residual_add

        self.residual_add_func(hidden_state,
                               residual,
                               attention_output,
                               attention_bias,
                               final_bias,
                               self.config.mp_size,
                               self.config.mlp_after_attn,
                               add_bias,
                               self.config.pre_layer_norm)
        return residual
