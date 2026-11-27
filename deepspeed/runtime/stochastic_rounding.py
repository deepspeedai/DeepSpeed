# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import torch

# Bounds the temporaries of one rounding pass, however large a partition is.
_CHUNK_NUMEL = 1 << 24


class StochasticRoundingCopy:
    """Copy an optimizer partition, rounding fp32 masters to bf16 on the parameter device."""

    def __init__(self, dp_rank):
        self.seed = (torch.initial_seed() + dp_rank) % (1 << 64)
        self.generators = {}

    def __call__(self, dst, src):
        if dst.dtype != torch.bfloat16:
            dst.copy_(src)
            return
        generator = self.generators.get(dst.device)
        if generator is None:
            generator = self.generators[dst.device] = torch.Generator(device=dst.device).manual_seed(self.seed)
        for start in range(0, src.numel(), _CHUNK_NUMEL):
            _round_chunk(dst[start:start + _CHUNK_NUMEL], src[start:start + _CHUNK_NUMEL], generator)


def _round_chunk(dst, src, generator):
    src = src.to(dst.device)
    noise = torch.randint(1 << 16, src.shape, dtype=torch.int32, device=dst.device, generator=generator)
    # Carry into the retained magnitude with probability equal to the discarded fraction, for either sign.
    rounded = noise.add_(src.view(torch.int32)).bitwise_and_(-(1 << 16)).view(torch.float32)
    # A NaN payload can carry into infinity or wrap into zero. Infinity already has zero low bits.
    torch.where(src.isnan(), src, rounded, out=rounded)
    dst.copy_(rounded)
