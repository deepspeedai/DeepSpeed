# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""fp32 master weights stored as the bf16 weights plus an int16 remainder (``bf16.store_param_remainders``).

The bf16 tensor holds the upper 16 bits of each fp32 value, rounded to nearest with ties away from zero: when the
lower 16 bits are 0x8000 or more, the upper half is incremented. The int16 remainder holds the lower 16 bits as a
signed value, so a negative remainder marks a rounded-up upper half. Rebuilding the fp32 value undoes the round-up and
joins the halves, which is exact. A NaN is stored as a quiet bf16 NaN with a zero remainder, so the weight the model
sees stays NaN.

The fused Adam kernel (``csrc/adam/multi_tensor_adam.cu``, ``AdamParamRemainderFunctor``) uses the same layout; so
does Transformer Engine's FusedAdam with ``store_param_remainders``, which Megatron-LM uses.
"""

import torch

# Elements processed at once, which bounds the temporaries of a conversion (a few int32 tensors of this size).
_CHUNK_NUMEL = 1 << 26
_QUIET_NAN_BF16_BITS = 0x7FC0


def _split_chunk(master):
    bits = master.view(torch.int32)
    lower = bits.to(torch.int16)
    round_up = (lower < 0).to(torch.int32)
    upper = ((bits >> 16) + round_up).to(torch.int16)
    nan = torch.isnan(master)
    upper = upper.masked_fill(nan, _QUIET_NAN_BF16_BITS)
    lower = lower.masked_fill(nan, 0)
    return upper, lower


def _rebuild_chunk(bf16, remainder):
    lower = remainder.to(torch.int32)
    upper = bf16.view(torch.int16).to(torch.int32) - (lower < 0).to(torch.int32)
    bits = ((upper & 0xFFFF) << 16) | (lower & 0xFFFF)
    return bits.view(torch.float32)


def split_fp32_master(master: torch.Tensor, bf16_out: torch.Tensor, remainder_out: torch.Tensor) -> None:
    """Store the fp32 values of ``master`` into ``bf16_out`` (bf16) and ``remainder_out`` (int16), in place.

    ``master`` may live on another device; it is converted in chunks on the device of ``bf16_out``.
    """
    if master.dtype != torch.float32 or bf16_out.dtype != torch.bfloat16 or remainder_out.dtype != torch.int16:
        raise TypeError(f"expected fp32 master, bf16 and int16 outputs; got {master.dtype}, {bf16_out.dtype}, "
                        f"{remainder_out.dtype}")
    if not (master.numel() == bf16_out.numel() == remainder_out.numel()):
        raise ValueError(f"size mismatch: {master.numel()}, {bf16_out.numel()}, {remainder_out.numel()}")
    master = master.reshape(-1)
    bf16_flat = bf16_out.view(-1)
    remainder_flat = remainder_out.view(-1)
    for start in range(0, master.numel(), _CHUNK_NUMEL):
        numel = min(_CHUNK_NUMEL, master.numel() - start)
        chunk = master.narrow(0, start, numel).to(bf16_out.device)
        upper, lower = _split_chunk(chunk)
        bf16_flat.narrow(0, start, numel).view(torch.int16).copy_(upper)
        remainder_flat.narrow(0, start, numel).copy_(lower)


def rebuild_fp32_master(bf16: torch.Tensor, remainder: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """The fp32 values stored in ``bf16`` and ``remainder``, written into ``out`` (any device) or a new tensor."""
    if bf16.dtype != torch.bfloat16 or remainder.dtype != torch.int16:
        raise TypeError(f"expected bf16 and int16 inputs; got {bf16.dtype}, {remainder.dtype}")
    if bf16.numel() != remainder.numel():
        raise ValueError(f"size mismatch: {bf16.numel()}, {remainder.numel()}")
    if out is None:
        out = torch.empty(bf16.shape, dtype=torch.float32, device=bf16.device)
    elif out.dtype != torch.float32 or out.numel() != bf16.numel():
        raise ValueError(f"out must be fp32 with {bf16.numel()} elements")
    bf16_flat = bf16.reshape(-1)
    remainder_flat = remainder.reshape(-1)
    out_flat = out.view(-1)
    for start in range(0, bf16_flat.numel(), _CHUNK_NUMEL):
        numel = min(_CHUNK_NUMEL, bf16_flat.numel() - start)
        chunk = _rebuild_chunk(bf16_flat.narrow(0, start, numel), remainder_flat.narrow(0, start, numel))
        out_flat.narrow(0, start, numel).copy_(chunk)
    return out
