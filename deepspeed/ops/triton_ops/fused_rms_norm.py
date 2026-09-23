# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Fused HF-style RMSNorm with the Qwen cast-before-gamma order."""

from __future__ import annotations

import types

import torch
from torch.autograd.function import once_differentiable

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, triton, tl

_IS_ROCM_PYTORCH = getattr(torch.version, "hip", None) is not None

SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
# Classes whose forward is confirmed to compute ``weight * normalized.to(input_dtype)`` with FP32 statistics, the
# expression the kernels reproduce. Hugging Face classes with the same name suffix and attributes, such as
# GptOssRMSNorm, multiply by the weight before that cast, so a class name and attributes are not enough.
SUPPORTED_RMS_NORM_CLASSES = ("transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRMSNorm", )
_MAX_FORWARD_BLOCK = 2048
_DWEIGHT_BLOCK_M = 16
_DWEIGHT_BLOCK_N = 256

if _TRITON_AVAILABLE:

    @triton.jit
    def _rms_norm_forward_kernel(
        hidden_ptr,
        weight_ptr,
        out_ptr,
        rstd_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < n_cols

        hidden = tl.load(hidden_ptr + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(hidden * hidden, axis=0) / n_cols
        rstd = tl.rsqrt(variance + eps)
        tl.store(rstd_ptr + row, rstd)

        normalized = (hidden * rstd).to(out_ptr.dtype.element_ty)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + row * n_cols + offsets, normalized * weight, mask=mask)

    @triton.jit
    def _rms_norm_dx_kernel(
        grad_out_ptr,
        hidden_ptr,
        weight_ptr,
        rstd_ptr,
        grad_hidden_ptr,
        n_cols: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < n_cols

        grad_out = tl.load(grad_out_ptr + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
        hidden = tl.load(hidden_ptr + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row).to(tl.float32)

        # Mirror the eager autograd graph step by step so values round where the eager backward rounds them.
        grad_norm = (grad_out * weight).to(grad_hidden_ptr.dtype.element_ty).to(tl.float32)
        grad_scaled = grad_norm * rstd
        grad_rstd = tl.sum(grad_norm * hidden, axis=0)
        grad_variance = (-0.5 * grad_rstd) * (rstd * rstd * rstd)
        grad_hidden = grad_scaled + (grad_variance / n_cols) * (2.0 * hidden)
        tl.store(grad_hidden_ptr + row * n_cols + offsets, grad_hidden, mask=mask)

    @triton.jit
    def _rms_norm_dweight_partial_kernel(
        grad_out_ptr,
        hidden_ptr,
        rstd_ptr,
        partial_ptr,
        n_rows,
        n_cols: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row_block = tl.program_id(0).to(tl.int64)
        col_block = tl.program_id(1).to(tl.int64)
        row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (row_offsets[:, None] < n_rows) & (col_offsets[None, :] < n_cols)

        hidden = tl.load(
            hidden_ptr + row_offsets[:, None] * n_cols + col_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_out = tl.load(
            grad_out_ptr + row_offsets[:, None] * n_cols + col_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        rstd = tl.load(rstd_ptr + row_offsets, mask=row_offsets < n_rows, other=0.0).to(tl.float32)
        normalized = (hidden * rstd[:, None]).to(hidden_ptr.dtype.element_ty).to(tl.float32)
        # Eager computes the gamma-gradient product in the input dtype before reducing it.
        product = (grad_out * normalized).to(hidden_ptr.dtype.element_ty).to(tl.float32)
        partial = tl.sum(product, axis=0)
        tl.store(partial_ptr + row_block * n_cols + col_offsets, partial, mask=col_offsets < n_cols)


def is_available() -> bool:
    """Whether this build can run the fused RMSNorm kernels."""
    return _TRITON_AVAILABLE and not _IS_ROCM_PYTORCH


def _assert_supported_device_and_dtype(tensor: torch.Tensor, *, name: str) -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError(f"fused RMSNorm needs Triton, which is not installed in this environment.")
    if _IS_ROCM_PYTORCH:
        raise RuntimeError("fused RMSNorm is not yet supported on ROCm.")
    if tensor.dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(
            f"fused RMSNorm supports bfloat16 and float16 tensors, but {name} has dtype {tensor.dtype}.")
    if tensor.device.type != "cuda":
        raise RuntimeError(f'fused RMSNorm runs CUDA kernels but {name} is on device "{tensor.device.type}".')


def assert_supported(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    """Reject unsupported inputs before the fused path runs."""
    _assert_supported_device_and_dtype(hidden, name="hidden")
    _assert_supported_device_and_dtype(weight, name="weight")
    if hidden.device != weight.device:
        raise RuntimeError(
            f"fused RMSNorm needs hidden and weight on the same device, got {hidden.device} and {weight.device}.")
    if hidden.dtype != weight.dtype:
        raise RuntimeError(
            f"fused RMSNorm needs hidden and weight to have the same dtype, got {hidden.dtype} and {weight.dtype}.")
    if hidden.dim() == 0:
        raise RuntimeError("fused RMSNorm needs at least one normalized dimension.")
    if weight.dim() != 1:
        raise RuntimeError(f"fused RMSNorm needs a 1-D weight Parameter, got shape {tuple(weight.shape)}.")
    if hidden.shape[-1] != weight.numel():
        raise RuntimeError(f"fused RMSNorm expected hidden last dimension {weight.numel()}, got {hidden.shape[-1]}.")
    if hidden.shape[-1] < 1:
        raise RuntimeError("fused RMSNorm needs a non-empty normalized dimension.")
    if hidden.shape[-1] > _MAX_FORWARD_BLOCK:
        raise RuntimeError(
            f"fused RMSNorm supports normalized dimensions up to {_MAX_FORWARD_BLOCK}, got {hidden.shape[-1]}.")
    if not isinstance(eps, float):
        raise RuntimeError(f"fused RMSNorm needs a float epsilon, got {type(eps).__name__}.")


def _block_n(n_cols: int) -> int:
    return max(16, triton.next_power_of_2(n_cols))


class _FusedRMSNorm(torch.autograd.Function):
    """RMSNorm autograd with FP32 reductions and a rounded pre-gamma value."""

    @staticmethod
    def forward(ctx, hidden, weight, eps):
        n_rows, n_cols = hidden.shape
        out = torch.empty_like(hidden)
        rstd = torch.empty((n_rows, ), dtype=torch.float32, device=hidden.device)

        ctx.save_for_backward(hidden, weight, rstd)
        ctx.n_cols = n_cols

        if n_rows > 0:
            _rms_norm_forward_kernel[(n_rows, )](
                hidden,
                weight,
                out,
                rstd,
                n_cols,
                eps,
                BLOCK_N=_block_n(n_cols),
            )
        return out

    # Autograd cannot see inside the Triton kernels, so differentiating this backward again would silently drop
    # the op's second derivative; once_differentiable makes that raise instead.
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        hidden, weight, rstd = ctx.saved_tensors
        n_rows, n_cols = hidden.shape
        grad_out = grad_out.contiguous()
        grad_hidden = torch.empty_like(hidden)

        if n_rows == 0:
            return grad_hidden, torch.zeros_like(weight), None

        _rms_norm_dx_kernel[(n_rows, )](
            grad_out,
            hidden,
            weight,
            rstd,
            grad_hidden,
            n_cols,
            BLOCK_N=_block_n(n_cols),
        )

        dweight_block_n = min(_DWEIGHT_BLOCK_N, _block_n(n_cols))
        n_row_blocks = triton.cdiv(n_rows, _DWEIGHT_BLOCK_M)
        n_col_blocks = triton.cdiv(n_cols, dweight_block_n)
        partial = torch.empty((n_row_blocks, n_cols), dtype=torch.float32, device=hidden.device)
        _rms_norm_dweight_partial_kernel[(n_row_blocks, n_col_blocks)](
            grad_out,
            hidden,
            rstd,
            partial,
            n_rows,
            n_cols,
            BLOCK_M=_DWEIGHT_BLOCK_M,
            BLOCK_N=dweight_block_n,
        )
        grad_weight = partial.sum(dim=0).to(weight.dtype)
        return grad_hidden, grad_weight, None


def fused_rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply HF Qwen-style RMSNorm with a fused CUDA/Triton implementation.

    The variance and normalization are computed in FP32, the normalized value is
    rounded once to the input dtype, and the module weight is multiplied after
    that rounding. ``hidden`` and ``weight`` must be bfloat16 or float16 CUDA
    tensors of the same dtype, and the normalized (last) dimension must be at
    most 2048.

    The kernels read hidden rows and the weight as dense arrays, so
    non-contiguous tensors of either kind are copied explicitly before they
    run, and arbitrary leading shapes are handled without relying on strided
    address reconstruction inside the kernels. The output and the gradient
    produced for ``hidden`` are contiguous whatever the input strides.
    Gradients are first order only: differentiating them again raises.
    """
    assert_supported(hidden, weight, eps)
    original_shape = hidden.shape
    if not hidden.is_contiguous():
        hidden = hidden.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    hidden_2d = hidden.reshape(-1, original_shape[-1])
    out = _FusedRMSNorm.apply(hidden_2d, weight, eps)
    return out.reshape(original_shape)


def _qualified_name(obj) -> str:
    return f"{getattr(obj, '__module__', None)}.{getattr(obj, '__qualname__', None)}"


def _runs_supported_rms_norm_forward(module: torch.nn.Module) -> bool:
    module_class = type(module)
    class_name = _qualified_name(module_class)
    if class_name not in SUPPORTED_RMS_NORM_CLASSES:
        return False
    # Kernel installers, including an earlier call of this one, patch forward on the class or on the instance.
    # Only the class's own forward is known to compute the expression the kernels reproduce.
    class_forward_patched = _qualified_name(module_class.forward) != f"{class_name}.forward"
    # functools.wraps copies the original's module and qualified name onto a wrapper, so a wrapper passes the check
    # above; the __wrapped__ attribute that functools.wraps also sets gives it away.
    class_forward_wrapped = hasattr(module_class.forward, "__wrapped__")
    instance_forward_patched = "forward" in vars(module)
    if class_forward_patched or class_forward_wrapped or instance_forward_patched:
        return False
    weight = getattr(module, "weight", None)
    eps = getattr(module, "variance_epsilon", None)
    return isinstance(weight, torch.nn.Parameter) and weight.dim() == 1 and isinstance(eps, float)


def _fused_module_forward(self, hidden_states):
    return fused_rms_norm(hidden_states, self.weight, self.variance_epsilon)


def replace_rms_norm(module: torch.nn.Module) -> int:
    """Run ``fused_rms_norm`` in place of each supported RMSNorm module's forward.

    A module is replaced only if its exact class is listed in
    ``SUPPORTED_RMS_NORM_CLASSES`` and it still runs that class's own forward.
    Subclasses and modules whose forward was patched or wrapped, by another
    installer or by an earlier call, are left untouched. Replaced modules keep
    their own weight Parameter and ``variance_epsilon``.

    Every supported module is checked before any is replaced: a weight that is
    not a bfloat16 or float16 CUDA tensor, or is wider than the kernels
    support, raises rather than silently falling back to eager RMSNorm.

    Returns the number of modules replaced.
    """
    supported = [child for child in module.modules() if _runs_supported_rms_norm_forward(child)]
    for child in supported:
        name = f"{child.__class__.__name__}.weight"
        _assert_supported_device_and_dtype(child.weight, name=name)
        if child.weight.numel() > _MAX_FORWARD_BLOCK:
            raise RuntimeError(f"fused RMSNorm supports normalized dimensions up to {_MAX_FORWARD_BLOCK}, "
                               f"but {name} has {child.weight.numel()} elements.")
    for child in supported:
        child.forward = types.MethodType(_fused_module_forward, child)
    return len(supported)
