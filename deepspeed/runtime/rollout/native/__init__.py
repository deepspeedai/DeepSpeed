# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Native rollout: a tensor-parallel decoder whose weights stay at fixed addresses, CUDA-graph decoding, and a
rollout engine that syncs and reshards those weights in memory."""

from deepspeed.runtime.rollout.native.arena import WeightArena
from deepspeed.runtime.rollout.native.decoder import DecoderSpec, TPDecoder
from deepspeed.runtime.rollout.native.generator import NativeGenerator
from deepspeed.runtime.rollout.native.rollout import NativeRollout, NativeRolloutConfig, TrainerMPU

__all__ = [
    "WeightArena", "DecoderSpec", "TPDecoder", "NativeGenerator", "NativeRollout", "NativeRolloutConfig", "TrainerMPU"
]
