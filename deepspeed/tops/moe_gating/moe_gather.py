
import torch

from typing import Tuple

from deepspeed.ops.op_builder import TopsBuilder

inf_module = None

class MoEGatherFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, 
                moe_output,
                scores,
                mapped_slots,
                is_grad_enabled,
                top_k):
        kernel = inf_module.moe_gather_fwd

        ctx.inp_shape = moe_output.shape
        moe_output = moe_output.reshape(-1, moe_output.shape[-1]).contiguous()
        top_k_tokens = scores.shape[0]
        _, hidden_size = moe_output.shape
        n_tokens = top_k_tokens // top_k
        layer_output = torch.empty(n_tokens, hidden_size, dtype=moe_output.dtype, device=torch.cuda.current_device())
        kernel(
            layer_output,
            moe_output,
            scores,
            mapped_slots,
            top_k
        )
        ctx.top_k = top_k
        if is_grad_enabled:
            ctx.save_for_backward(
                scores,
                mapped_slots,
                moe_output
            )

        return layer_output

    @staticmethod
    def backward(ctx, layer_output_grad):
        (scores,
         mapped_slots,
         moe_output) = ctx.saved_tensors
        layer_output_grad = layer_output_grad.contiguous()
        top_k = ctx.top_k
        n_tokens, hidden_size = layer_output_grad.shape
        kernel = inf_module.moe_gather_bwd

        moe_output_grad = torch.zeros(moe_output.shape, dtype=layer_output_grad.dtype, device=torch.cuda.current_device())
        scores_grad = torch.empty(n_tokens * top_k, dtype=scores.dtype, device=torch.cuda.current_device())

        kernel(
            layer_output_grad,
            scores_grad,
            moe_output_grad,
            moe_output,
            scores,
            mapped_slots,
            top_k,
        )
        return moe_output_grad.reshape(ctx.inp_shape), scores_grad, None, None, None

class MoEGather(torch.nn.Module):

    def __init__(self, logit_dtype=None, top_k=1, use_act_ckpting=False) -> None:
        super(MoEGather, self).__init__()
        global inf_module
        if inf_module is None:
            inf_module = TopsBuilder().load()
        self.top_k = top_k
        self.use_act_ckpting = use_act_ckpting
    def forward(self,
                moe_output: torch.Tensor,
                scores: torch.Tensor,
                mapped_slots: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        is_grad_enabled = self.use_act_ckpting and torch.is_grad_enabled()
        return MoEGatherFunction.apply(
            moe_output,
            scores,
            mapped_slots,
            is_grad_enabled,
            self.top_k
        )