# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import torch

# Bounds the temporaries of one rounding pass, however large a partition is.
_CHUNK_NUMEL = 1 << 24


def stochastic_round_to_bf16_(dst, src, generator):
    """Copy fp32 ``src`` into bf16 ``dst``, rounding each element to one of its two bf16 neighbours at random.

    An element rounds away from zero with probability equal to its distance from the neighbour toward zero, as a
    fraction of the spacing between the two, so ``E[dst] == src`` for values within the finite bf16 range.
    Non-finite values round to nearest; finite values beyond the bf16 range can round to infinity. The rounding runs
    on ``dst``'s device, where ``generator`` lives, so an offloaded master is copied there in fp32.
    """
    src = src.view(-1)
    dst = dst.view(-1)
    for start in range(0, src.numel(), _CHUNK_NUMEL):
        numel = min(_CHUNK_NUMEL, src.numel() - start)
        chunk = src.narrow(0, start, numel).to(dst.device)
        # Adding 16 random bits below bf16's last mantissa bit and truncating them rounds the magnitude up with the
        # probability of the dropped fraction; fp32 is sign-magnitude, so this holds for either sign.
        noise = torch.randint(0, 1 << 16, (numel, ), dtype=torch.int32, device=chunk.device, generator=generator)
        truncated = noise.add_(chunk.view(torch.int32)).bitwise_and_(-(1 << 16)).view(torch.float32)
        torch.where(torch.isfinite(chunk), truncated, chunk, out=truncated)
        dst.narrow(0, start, numel).copy_(truncated)


class StochasticRoundingCopy:
    """``copy_(dst, src)`` replacement for the fp32-master to bf16-parameter copy (``bf16.stochastic_round_params``).

    Each device's generator is seeded from ``torch.initial_seed()`` and the data-parallel rank. On like devices,
    replicas with the same seed, partition layout and copy order use the same draws. Generator state is not
    checkpointed. Destinations that are not bf16 (parameters kept in fp32) are copied exactly.
    """

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
        stochastic_round_to_bf16_(dst, src, generator)
