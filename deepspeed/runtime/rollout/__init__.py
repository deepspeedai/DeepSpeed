# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engines for on-policy generation during RL/distillation training.

Provides:
  - :class:`RolloutEngine` — abstract base class
  - :class:`RolloutRequest`, :class:`RolloutBatch`, :class:`SamplingConfig` — dataclasses
  - :class:`HybridEngineRollout` — concrete implementation using DeepSpeed hybrid engine
  - :class:`VLLMRollout` — concrete implementation using an external vLLM server
  - :func:`build_rollout` — factory that selects the engine from config
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
    "build_rollout",
    "stitch_rollout",
]


def build_rollout(rollout_cfg, student_engine=None, tokenizer=None, student_model_path=None):
    """Factory: construct the rollout engine specified by ``rollout_cfg.engine``.

    Imports of heavy backends are deferred so that selecting the hybrid-engine
    path doesn't transitively require vLLM (and vice versa).

    Args:
        rollout_cfg: :class:`RolloutConfig` (or any object with an ``engine``
            attribute set to ``"hybrid_engine"`` or ``"vllm"``).
        student_engine: DeepSpeed engine wrapping the student model.
        tokenizer: HuggingFace tokenizer.
        student_model_path: Model name/path for vLLM to load from disk.
    """
    engine_name = rollout_cfg.engine
    if engine_name == "hybrid_engine":
        if student_engine is None or tokenizer is None:
            raise ValueError("hybrid_engine rollout needs both student_engine and tokenizer")
        return HybridEngineRollout(engine=student_engine, tokenizer=tokenizer, cfg=rollout_cfg)

    if engine_name == "vllm":
        if tokenizer is None:
            raise ValueError("vllm rollout needs a tokenizer for length accounting")
        return VLLMRollout(
            cfg=rollout_cfg,
            tokenizer=tokenizer,
            student_engine=student_engine,
            student_model_path=student_model_path,
        )

    raise ValueError(f"Unknown rollout engine {engine_name!r}; choose from 'hybrid_engine' | 'vllm'")
