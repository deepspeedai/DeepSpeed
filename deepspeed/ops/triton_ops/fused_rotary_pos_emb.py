# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Fused Hugging Face rotary position embedding with the rounding of the eager split-half expression."""

from __future__ import annotations

import functools
import importlib
import sys

import torch
from torch.autograd.function import once_differentiable

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, triton, tl
from deepspeed.utils import logger

_IS_ROCM_PYTORCH = getattr(torch.version, "hip", None) is not None

SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
# Modeling modules whose ``apply_rotary_pos_emb`` computes ``q * cos + rotate_half(q) * sin`` with the split-half
# ``rotate_half``. Several architectures define a function of the same name that rotates only part of the head
# dimension or interleaves it, so the name alone is not enough; the installer also compares each function's code
# with the reference expression below.
SUPPORTED_ROTARY_MODULES = (
    "transformers.models.deepseek_v3.modeling_deepseek_v3",
    "transformers.models.llama.modeling_llama",
    "transformers.models.mistral.modeling_mistral",
    "transformers.models.mixtral.modeling_mixtral",
    "transformers.models.qwen2.modeling_qwen2",
    "transformers.models.qwen2_moe.modeling_qwen2_moe",
    "transformers.models.qwen3.modeling_qwen3",
    "transformers.models.qwen3_moe.modeling_qwen3_moe",
)
_MAX_HEAD_DIM = 512
_BLOCK_S = 16
_HEADS_PER_PROGRAM = 4
_ORIGINAL_ATTRIBUTE = "_deepspeed_fused_rotary_original"


def _reference_rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Both references keep a docstring, as the Hugging Face functions do, so their constants line up in the bytecode.
def _reference_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# The name the reference expression looks up, as the Hugging Face function does in its own module.
rotate_half = _reference_rotate_half

if _TRITON_AVAILABLE:

    @triton.jit
    def _rotary_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        n_heads,
        seq_len,
        stride_xb,
        stride_xh,
        stride_xs,
        stride_ob,
        stride_oh,
        stride_os,
        stride_cb,
        stride_cs,
        HALF: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
        BLOCK_S: tl.constexpr,
        HEADS_PER_PROGRAM: tl.constexpr,
        BACKWARD: tl.constexpr,
    ):
        s_block = tl.program_id(0).to(tl.int64)
        batch = tl.program_id(1).to(tl.int64)
        head_block = tl.program_id(2).to(tl.int64)
        positions = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
        columns = tl.arange(0, BLOCK_HALF)
        mask = (positions[:, None] < seq_len) & (columns[None, :] < HALF)
        dtype = out_ptr.dtype.element_ty

        # One table row serves every head of that position.
        table = batch * stride_cb + positions[:, None] * stride_cs + columns[None, :]
        cos_left = tl.load(cos_ptr + table, mask=mask, other=0.0).to(tl.float32)
        cos_right = tl.load(cos_ptr + table + HALF, mask=mask, other=0.0).to(tl.float32)
        sin_left = tl.load(sin_ptr + table, mask=mask, other=0.0).to(tl.float32)
        sin_right = tl.load(sin_ptr + table + HALF, mask=mask, other=0.0).to(tl.float32)

        for index in tl.static_range(HEADS_PER_PROGRAM):
            head = head_block * HEADS_PER_PROGRAM + index
            head_mask = mask & (head < n_heads)
            source = batch * stride_xb + head * stride_xh + positions[:, None] * stride_xs + columns[None, :]
            target = batch * stride_ob + head * stride_oh + positions[:, None] * stride_os + columns[None, :]
            left = tl.load(x_ptr + source, mask=head_mask, other=0.0).to(tl.float32)
            right = tl.load(x_ptr + source + HALF, mask=head_mask, other=0.0).to(tl.float32)
            # Every product and sum is rounded to the input dtype where the eager graph rounds it.
            if BACKWARD:
                # The gradient of x * cos + rotate_half(x) * sin: the rotation's transpose moves the sine term
                # to the other half with the opposite sign.
                out_left = (left * cos_left).to(dtype).to(tl.float32) + (right * sin_right).to(dtype).to(tl.float32)
                out_right = (right * cos_right).to(dtype).to(tl.float32) - (left * sin_left).to(dtype).to(tl.float32)
            else:
                out_left = (left * cos_left).to(dtype).to(tl.float32) - (right * sin_left).to(dtype).to(tl.float32)
                out_right = (right * cos_right).to(dtype).to(tl.float32) + (left * sin_right).to(dtype).to(tl.float32)
            tl.store(out_ptr + target, out_left.to(dtype), mask=head_mask)
            tl.store(out_ptr + target + HALF, out_right.to(dtype), mask=head_mask)


def is_available() -> bool:
    """Whether this build can run the fused rotary kernels."""
    return _TRITON_AVAILABLE and not _IS_ROCM_PYTORCH


def assert_supported(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                     unsqueeze_dim: int) -> None:
    """Reject inputs the kernels do not reproduce eager for, before the fused path runs."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("fused RoPE needs Triton, which is not installed in this environment.")
    if _IS_ROCM_PYTORCH:
        raise RuntimeError("fused RoPE is not yet supported on ROCm.")
    tensors = {"q": q, "k": k, "cos": cos, "sin": sin}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"fused RoPE needs tensors, but {name} is {type(tensor).__name__}.")
        if tensor.device.type != "cuda":
            raise RuntimeError(f'fused RoPE runs CUDA kernels but {name} is on device "{tensor.device.type}".')
        if tensor.dtype != q.dtype:
            # Eager type promotion would compute in the wider dtype.
            raise RuntimeError(f"fused RoPE needs q, k, cos and sin in one dtype, got {q.dtype} and {tensor.dtype}.")
        if tensor.device != q.device:
            raise RuntimeError(f"fused RoPE needs all tensors on one device, got {q.device} and {tensor.device}.")
    if q.dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(f"fused RoPE supports bfloat16 and float16 tensors, got {q.dtype}.")
    if unsqueeze_dim not in (1, 2):
        raise RuntimeError(f"fused RoPE supports unsqueeze_dim 1 or 2, got {unsqueeze_dim}.")
    if q.dim() != 4 or k.dim() != 4 or cos.dim() != 3 or sin.shape != cos.shape:
        raise RuntimeError("fused RoPE needs 4-D q and k and 3-D cos and sin of one shape, got "
                           f"{tuple(q.shape)}, {tuple(k.shape)}, {tuple(cos.shape)} and {tuple(sin.shape)}.")
    head_dim = q.shape[-1]
    if head_dim % 2 or not 0 < head_dim <= _MAX_HEAD_DIM:
        raise RuntimeError(f"fused RoPE needs an even head dimension up to {_MAX_HEAD_DIM}, got {head_dim}.")
    sequence_dim = 2 if unsqueeze_dim == 1 else 1
    for name, tensor in (("q", q), ("k", k)):
        expected = (cos.shape[0], cos.shape[1], head_dim)
        actual = (tensor.shape[0], tensor.shape[sequence_dim], tensor.shape[-1])
        if actual != expected and not (cos.shape[0] == 1 and actual[1:] == expected[1:]):
            raise RuntimeError(f"fused RoPE needs {name} to match cos in batch, sequence and head dimension, got "
                               f"{tuple(tensor.shape)} for cos {tuple(cos.shape)}.")
        if tensor.stride(-1) != 1:
            raise RuntimeError(f"fused RoPE needs a unit-stride head dimension in {name}.")
    if k.shape[0] != q.shape[0]:
        raise RuntimeError(f"fused RoPE needs q and k with one batch size, got {q.shape[0]} and {k.shape[0]}.")
    if cos.requires_grad or sin.requires_grad:
        raise RuntimeError("fused RoPE treats cos and sin as constants, but one of them requires grad.")


def _launch(x, cos, sin, out, backward):
    batch, heads, seq_len, head_dim = x.shape
    if x.numel() == 0:
        return out
    grid = (triton.cdiv(seq_len, _BLOCK_S), batch, triton.cdiv(heads, _HEADS_PER_PROGRAM))
    _rotary_kernel[grid](
        x,
        cos,
        sin,
        out,
        heads,
        seq_len,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        cos.stride(0) if cos.shape[0] > 1 else 0,
        cos.stride(1),
        HALF=head_dim // 2,
        BLOCK_HALF=max(16, triton.next_power_of_2(head_dim // 2)),
        BLOCK_S=_BLOCK_S,
        HEADS_PER_PROGRAM=_HEADS_PER_PROGRAM,
        BACKWARD=backward,
    )
    return out


def _unit_stride(tensor):
    return tensor if tensor.stride(-1) == 1 else tensor.contiguous()


class _FusedRotaryPosEmb(torch.autograd.Function):
    """Split-half RoPE on [batch, heads, sequence, head_dim] views; cos and sin are [batch, sequence, head_dim]."""

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        cos, sin = cos.contiguous(), sin.contiguous()
        # Outputs keep the strides of their inputs, as the eager expression's do.
        q_out = _launch(q, cos, sin, torch.empty_like(q), backward=False)
        k_out = _launch(k, cos, sin, torch.empty_like(k), backward=False)
        ctx.save_for_backward(cos, sin)
        # The gradients take the layout of q and k, which is what their producers wrote; only strides are kept.
        ctx.layouts = ((q_out.shape, q_out.stride()), (k_out.shape, k_out.stride()))
        return q_out, k_out

    # Autograd cannot see inside the Triton kernels, so differentiating this backward again would silently drop
    # the second derivative; once_differentiable makes that raise instead.
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_q, grad_k):
        cos, sin = ctx.saved_tensors
        grads = []
        for grad, (shape, stride) in zip((grad_q, grad_k), ctx.layouts):
            if grad is None:
                grads.append(None)
                continue
            out = torch.empty_strided(shape, stride, dtype=grad.dtype, device=grad.device)
            grads.append(_launch(_unit_stride(grad), cos, sin, out, backward=True))
        return grads[0], grads[1], None, None


def _run_kernels(q, k, cos, sin, unsqueeze_dim):
    if unsqueeze_dim == 2:
        q_out, k_out = _FusedRotaryPosEmb.apply(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
        return q_out.transpose(1, 2), k_out.transpose(1, 2)
    return _FusedRotaryPosEmb.apply(q, k, cos, sin)


def fused_apply_rotary_pos_emb(q: torch.Tensor,
                               k: torch.Tensor,
                               cos: torch.Tensor,
                               sin: torch.Tensor,
                               unsqueeze_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Hugging Face's split-half rotary position embedding with a fused Triton kernel.

    Computes ``q * cos + rotate_half(q) * sin`` and the same for ``k``, with the
    signature and broadcasting of the Hugging Face function: ``cos`` and ``sin``
    are ``[batch, sequence, head_dim]``, and ``unsqueeze_dim`` is 1 for
    ``[batch, heads, sequence, head_dim]`` queries and keys or 2 for
    ``[batch, sequence, heads, head_dim]``. Each product and sum is rounded to the
    input dtype where the eager expression rounds it, so outputs and gradients are
    bitwise identical to eager. Outputs keep the strides of ``q`` and ``k``, and
    so do the gradients produced for them.

    All four tensors must be bfloat16 or float16 CUDA tensors of one dtype, the
    head dimension must be even and at most 512 with unit stride, and ``cos`` and
    ``sin`` must not require grad. Gradients are first order only: differentiating
    them again raises. Unsupported inputs raise.
    """
    assert_supported(q, k, cos, sin, unsqueeze_dim)
    return _run_kernels(q, k, cos, sin, unsqueeze_dim)


def _same_code(function, reference):
    code, expected = getattr(function, "__code__", None), reference.__code__
    return (code is not None and code.co_code == expected.co_code and code.co_names == expected.co_names
            and code.co_varnames == expected.co_varnames and function.__defaults__ == reference.__defaults__)


def _make_fused(original):

    @functools.wraps(original)
    def fused(q, k, cos, sin, unsqueeze_dim=1):
        try:
            assert_supported(q, k, cos, sin, unsqueeze_dim)
        except RuntimeError as unsupported:
            # The replaced function is the eager computation the kernels stand in for, so the result is eager's.
            logger.warning_once(f"{unsupported} Running eager apply_rotary_pos_emb instead.")
            return original(q, k, cos, sin, unsqueeze_dim)
        return _run_kernels(q, k, cos, sin, unsqueeze_dim)

    setattr(fused, _ORIGINAL_ATTRIBUTE, original)
    return fused


def replace_rotary_pos_emb(model: torch.nn.Module) -> int:
    """Run the fused kernel in place of the Hugging Face ``apply_rotary_pos_emb`` used by ``model``.

    Attention modules look the function up in their modeling module, so the
    replacement is made there, and it applies to every model of that
    architecture in the process, not only to ``model``. A modeling module is
    patched if one of ``model``'s submodules is defined in it, it is listed in
    ``SUPPORTED_ROTARY_MODULES``, and its ``apply_rotary_pos_emb`` and
    ``rotate_half`` still have the code of the split-half expression. Modules
    already patched, by an earlier call or another installer, are left alone.
    ``restore_rotary_pos_emb`` undoes the replacement.

    The replacement runs the kernel when ``assert_supported`` accepts its inputs.
    For any other input it runs the original function, so the result is exactly
    eager's, and logs a warning once.

    Returns the number of modeling modules patched.
    """
    count = 0
    names = sorted({type(child).__module__ for child in model.modules()} & set(SUPPORTED_ROTARY_MODULES))
    for name in names:
        modeling = sys.modules.get(name) or importlib.import_module(name)
        original = getattr(modeling, "apply_rotary_pos_emb", None)
        rotate_half = getattr(modeling, "rotate_half", None)
        if not (_same_code(original, _reference_apply_rotary_pos_emb)
                and _same_code(rotate_half, _reference_rotate_half)):
            if not hasattr(original, _ORIGINAL_ATTRIBUTE):
                logger.warning(f"{name}.apply_rotary_pos_emb is not the split-half expression the fused RoPE "
                               f"kernel reproduces; leaving it unchanged.")
            continue
        modeling.apply_rotary_pos_emb = _make_fused(original)
        count += 1
    if count and not is_available():
        logger.warning(f"fused RoPE patched {count} modeling modules, but its kernel needs Triton on CUDA, not ROCm, "
                       f"so they will run the eager function.")
    return count


def restore_rotary_pos_emb() -> int:
    """Undo ``replace_rotary_pos_emb`` in every patched modeling module; returns how many were restored."""
    count = 0
    for name in SUPPORTED_ROTARY_MODULES:
        modeling = sys.modules.get(name)
        original = getattr(getattr(modeling, "apply_rotary_pos_emb", None), _ORIGINAL_ATTRIBUTE, None)
        if original is not None:
            modeling.apply_rotary_pos_emb = original
            count += 1
    return count
