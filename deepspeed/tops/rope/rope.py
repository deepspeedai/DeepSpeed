
import torch

from typing import Tuple

from deepspeed.ops.op_builder import TopsBuilder

inf_module = None

class RoPEFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, dim, theta):
        q = q.contiguous()
        k = k.contiguous()
        inf_module.rope_fwd(q, k, dim, theta)

        ctx.dim = dim
        ctx.theta = theta

        return q, k 

    @staticmethod
    def backward(ctx, q_grad, k_grad):
        q_grad = q_grad.contiguous()
        k_grad = k_grad.contiguous()

        inf_module.rope_bwd(q_grad, k_grad, ctx.dim, ctx.theta)

        return q_grad, k_grad, None, None

class RoPE(torch.nn.Module):

    def __init__(self, rotary_dim=None, rope_theta=10000.0) -> None:
        super(RoPE, self).__init__()
        global inf_module
        if inf_module is None:
            inf_module = TopsBuilder().load()
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta

    def forward(self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
    ) -> torch.Tensor:

        if self.rotary_dim is None:
            self.rotary_dim = query.shape[-1]

        return RoPEFunction.apply(
            query, key, self.rotary_dim, self.rope_theta
        )
