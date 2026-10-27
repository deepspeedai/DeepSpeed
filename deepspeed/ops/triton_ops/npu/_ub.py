# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Ascend unified-buffer (UB) tile budgeting for the NPU triton_ops variants.

Ported from flash-linear-attention's ``fla/utils/ascend_ub_manager.py`` (itself
adapted from Liger-Kernel's ``ub_manager.py``). The unified buffer on an Ascend
NPU is much smaller than an NVIDIA SM's shared memory, so tile sizes that are
trivially safe on CUDA overflow it at compile time on Ascend; the kernels in
this package size their tiles from the helpers below instead of fixed
constants. The UB capacity is detected once per process (environment override
``ASCEND_UB_CAPACITY_BITS``, the ``tbe`` SOC spec, or a conservative 64 KiB
fallback) and every helper rounds to a power of two under a safety margin.
"""

import os
import warnings

import torch

from deepspeed.ops.triton_ops._triton import triton

# Per-axis launch limit on Ascend.
ASCEND_MAX_GRID_DIM = 65535

_FALLBACK_UB_CAPACITY_BITS = 65536 * 8
_UB_CAPACITY_BITS = None


def _ub_capacity_bits() -> int:
    """The device's UB capacity in bits, detected once per process."""
    global _UB_CAPACITY_BITS
    if _UB_CAPACITY_BITS is not None:
        return _UB_CAPACITY_BITS

    env_capacity = os.getenv("ASCEND_UB_CAPACITY_BITS")
    if env_capacity is not None:
        try:
            capacity_bits = int(env_capacity)
            if capacity_bits > 0:
                _UB_CAPACITY_BITS = capacity_bits
                return _UB_CAPACITY_BITS
        except ValueError:
            pass

    try:
        from tbe.common.platform import get_soc_spec, set_current_compile_soc_info

        device = torch.npu
        soc_info = device.get_device_name(device.current_device())
        set_current_compile_soc_info(soc_info)
        ub_size_bytes = get_soc_spec("UB_SIZE")
        if ub_size_bytes is not None and ub_size_bytes > 0:
            _UB_CAPACITY_BITS = ub_size_bytes * 8
            return _UB_CAPACITY_BITS
    except Exception as error:  # noqa: BLE001 - any failure falls back below
        warnings.warn(
            f"Unable to detect the Ascend UB capacity ({error}); using the 64 KiB fallback. "
            f"Set ASCEND_UB_CAPACITY_BITS to override.",
            stacklevel=2)

    _UB_CAPACITY_BITS = _FALLBACK_UB_CAPACITY_BITS
    return _UB_CAPACITY_BITS


def _safe_block_size(memory_multiplier: float, unit_param: float, safety_margin: float, dtype_size: int) -> int:
    """Largest power-of-two tile under the UB budget: ``mult * BLOCK * unit * bytes * 8`` bits."""
    safe_ub_bits = int(_ub_capacity_bits() * safety_margin)
    max_block_size = max(1, int(safe_ub_bits // (memory_multiplier * unit_param * dtype_size * 8)))
    return max(1, triton.next_power_of_2(max_block_size + 1) // 2)


def compute_ub_block_size(
    dim_size: int,
    memory_multiplier: float,
    *,
    safety_margin: float = 0.9,
    dtype_size: int = 4,
    fallback: int = 2048,
    min_block: int = 1,
    max_block: int | None = None,
    desired: int | None = None,
) -> int:
    """UB-safe power-of-two block size for one tilable dimension."""
    if desired is None:
        desired = triton.next_power_of_2(dim_size)
    block = min(desired, _safe_block_size(memory_multiplier, 1.0, safety_margin, dtype_size))
    block = max(min_block, min(block, fallback))
    if max_block is not None:
        block = min(block, max_block)
    return block


def compute_activation_block_size(
    total_elements: int,
    is_backward: bool,
    *,
    memory_multiplier: float | None = None,
    max_grid: int = ASCEND_MAX_GRID_DIM,
    max_core_dim: int = ASCEND_MAX_GRID_DIM,
    safety_margin: float = 0.9,
    max_block: int = 2048,
) -> int:
    """UB-safe block size for flattened elementwise (activation) kernels.

    The backward keeps more live temporaries than the forward, so it gets a
    larger memory multiplier and therefore a smaller tile - the same split the
    SwiGLU kernels need on Ascend (their forward is UB-safe at 2048, their
    backward is not).
    """
    if memory_multiplier is None:
        memory_multiplier = 6.0 if is_backward else 3.0
    block = compute_ub_block_size(
        dim_size=total_elements,
        memory_multiplier=memory_multiplier,
        safety_margin=safety_margin,
        fallback=2048,
        min_block=256,
        max_block=max_block,
    )
    block = min(block, max_core_dim // 8)
    if triton.cdiv(total_elements, block) > max_grid:
        block = max(triton.next_power_of_2(triton.cdiv(total_elements, max_grid)), 1)
    return block


def compute_row_tile_block_size(
    row_dim: int,
    fixed_dim: int,
    memory_multiplier: float,
    *,
    tiling_row: bool = True,
    safety_margin: float = 0.85,
    dtype_size: int = 4,
    fallback: int = 16,
    min_block: int = 1,
    max_block: int | None = None,
) -> int:
    """UB-safe tile along one axis of a 2-D [row, col] kernel tile."""
    unit_param = float(fixed_dim) if tiling_row else float(row_dim)
    desired = triton.next_power_of_2(row_dim if tiling_row else fixed_dim)
    block = min(desired, _safe_block_size(memory_multiplier, unit_param, safety_margin, dtype_size))
    block = max(min_block, min(block, fallback))
    if max_block is not None:
        block = min(block, max_block)
    return block


def compute_vocab_block_size(
    vocab_size: int,
    num_rows: int,
    memory_multiplier: float,
    *,
    safety_margin: float = 0.9,
    max_block: int = 8192,
    fallback: int = 2048,
) -> int:
    """UB-safe vocabulary tile size respecting the Ascend grid dim0 limit."""
    ub_block = compute_ub_block_size(
        dim_size=vocab_size,
        memory_multiplier=memory_multiplier,
        safety_margin=safety_margin,
        fallback=fallback,
    )
    max_splits = max(1, ASCEND_MAX_GRID_DIM // max(num_rows, 1))
    grid_min = triton.next_power_of_2(triton.cdiv(vocab_size, max_splits))
    return min(max(ub_block, grid_min), max_block)


def multiprocessor_count(device) -> int | None:
    """The device's processor count when the runtime exposes it, else ``None``."""
    try:
        props = torch.npu.get_device_properties(device.index)
        count = getattr(props, "multi_processor_count", None)
        return count if count and count > 0 else None
    except Exception:  # noqa: BLE001 - any failure means the count is unknown
        return None


def max_grid_axis_chunks(axis_size: int, other_grid_product: int, *, max_grid: int = ASCEND_MAX_GRID_DIM) -> int:
    """Largest launch chunk along one grid axis while the product stays within ``max_grid``."""
    return max(1, max_grid // max(other_grid_product, 1))


def iter_axis_launch_chunks(axis_size: int, other_grid_product: int, *, max_grid: int = ASCEND_MAX_GRID_DIM):
    """Yield ``(offset, chunk_len)`` pairs for host-side grid-axis chunking."""
    max_chunks = max_grid_axis_chunks(axis_size=axis_size, other_grid_product=other_grid_product, max_grid=max_grid)
    for offset in range(0, axis_size, max_chunks):
        yield offset, min(max_chunks, axis_size - offset)
