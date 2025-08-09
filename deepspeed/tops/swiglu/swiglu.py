
import torch

from typing import Tuple

from deepspeed.ops.op_builder import TopsBuilder

inf_module = None

def bwd(x, y, grad):
    x_float = x.float()
    y_float = y.float()
    g_float = grad.float()
    return (g_float * y_float * \
            torch.nn.functional.sigmoid(x_float) * (1.0 + x_float * (1.0 - torch.nn.functional.sigmoid(x_float)))).to(x.dtype)

class SwiGluFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, is_grad_enabled):
        out = torch.empty((inp.shape[:-1] + (inp.shape[-1] // 2,)), 
                          dtype=inp.dtype, device=inp.device)
                          
        inp = inp.contiguous()
        inf_module.swiglu_fwd(inp, out)

        if is_grad_enabled:
            ctx.save_for_backward(inp)
        return out 

    @staticmethod
    def backward(ctx, grad_out):
        (inp,) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        inp_grad = torch.empty_like(inp)
        inf_module.swiglu_bwd(inp, grad_out, inp_grad)
        return inp_grad, None

class SwiGlu(torch.nn.Module):

    def __init__(self, ) -> None:
        super(SwiGlu, self).__init__()
        global inf_module
        if inf_module is None:
            inf_module = TopsBuilder().load()

    def forward(self, 
                inp: torch.Tensor, 
                ) -> torch.Tensor:
        is_grad_enabled = torch.is_grad_enabled()
        return SwiGluFunction.apply(
            inp, is_grad_enabled
        )
