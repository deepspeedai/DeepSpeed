# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engines for on-policy generation during RL/distillation training.

Provides:
  - :class:`RolloutEngine` — abstract base class
  - :class:`RolloutRequest`, :class:`RolloutBatch`, :class:`SamplingConfig` — dataclasses
  - :class:`HybridEngineRollout` — concrete implementation using DeepSpeed hybrid engine
  - :func:`build_rollout` — factory that selects the engine from config
"""

from deepspeed.runtime.rollout.base import (
    RolloutBatch,
    RolloutConfig,
    RolloutEngine,
    RolloutRequest,
    SamplingConfig,
)
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout

__all__ = [
    "HybridEngineRollout",
    "RolloutBatch",
    "RolloutConfig",
    "RolloutEngine",
    "RolloutRequest",
    "SamplingConfig",
    "build_rollout",
]


def build_rollout(rollout_cfg, student_engine=None, tokenizer=None, **kwargs):
    """Factory: construct the rollout engine specified by ``rollout_cfg.engine``.

    Args:
        rollout_cfg: :class:`RolloutConfig` (or any object with an ``engine``
            attribute set to ``"hybrid_engine"``).
        student_engine: DeepSpeed engine wrapping the student model.
        tokenizer: HuggingFace tokenizer.
    """
    engine_name = rollout_cfg.engine
    if engine_name == "hybrid_engine":
        if student_engine is None or tokenizer is None:
            raise ValueError("hybrid_engine rollout needs both student_engine and tokenizer")
        return HybridEngineRollout(engine=student_engine, tokenizer=tokenizer, cfg=rollout_cfg)

    raise ValueError(f"Unknown rollout engine {engine_name!r}; choose from 'hybrid_engine'")
