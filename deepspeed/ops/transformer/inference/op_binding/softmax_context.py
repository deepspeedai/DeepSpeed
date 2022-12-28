import torch
from deepspeed import comm as dist
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class SoftmaxContextOp(BaseOp):
    def __init__(self, config: DeepSpeedInferenceConfig):
        super(SoftmaxContextOp, self).__init__(config)
        if self.config.fp16:
            self.softmax_context_func = self.inference_cuda_module.softmax_context_fp16
        else:
            self.softmax_context_func = self.inference_cuda_module.softmax_context_fp32

    def forward(self,
                query_key_value: torch.Tensor,
                attn_mask: torch.Tensor,
                heads: int,
                norm_factor: float,
                no_masking: bool,
                layer_id: int,
                num_layers: int,
                alibi: torch.Tensor):

        if alibi is not None:
            batch_heads = query_key_value.shape[0] * heads
            offset = dist.get_rank() * batch_heads if dist.is_initialized() else 0
            alibi = alibi[offset:batch_heads + offset, :, :]
        else:
            alibi = torch.empty(1)

        output = self.softmax_context_func(query_key_value,
                                           attn_mask,
                                           self.config.rotary_dim,
                                           self.config.rotate_half,
                                           self.config.rotate_every_two,
                                           heads,
                                           norm_factor,
                                           self.config.triangular_masking,
                                           self.config.local_attention,
                                           self.config.window_size,
                                           no_masking,
                                           layer_id,
                                           num_layers,
                                           alibi)
        return output
