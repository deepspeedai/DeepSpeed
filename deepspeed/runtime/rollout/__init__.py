# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engines for on-policy generation during RL/distillation training.

Provides:
  - :class:`RolloutEngine` — abstract base class
  - :class:`RolloutRequest`, :class:`RolloutBatch`, :class:`SamplingConfig` — dataclasses
  - :class:`HybridEngineRollout` — concrete implementation using DeepSpeed hybrid engine
  - :class:`VLLMRollout` — concrete implementation using an external vLLM server
"""

from deepspeed.runtime.rollout.base import (
    RolloutBatch,
    RolloutEngine,
    RolloutRequest,
    SamplingConfig,
)
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout
from deepspeed.runtime.rollout.vllm_rollout import VLLMRollout, stitch_rollout

__all__ = [
    "HybridEngineRollout",
    "RolloutBatch",
    "RolloutEngine",
    "RolloutRequest",
    "SamplingConfig",
    "VLLMRollout",
    "stitch_rollout",
]
