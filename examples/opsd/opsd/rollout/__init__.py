# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engines for OPSD: hybrid engine (built-in) or vLLM (disjoint GPUs)."""

from opsd.rollout.base import RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig

__all__ = ["RolloutBatch", "RolloutEngine", "RolloutRequest", "SamplingConfig", "build_rollout"]


def build_rollout(rollout_cfg, student_engine=None, tokenizer=None, student_model_path=None, arch=None):
    """Factory: construct the rollout engine specified by ``rollout_cfg.engine``.

    Imports of heavy backends are deferred to here so that selecting the
    hybrid-engine path doesn't transitively require vLLM (and vice versa).
    """
    engine_name = rollout_cfg.engine
    if engine_name == "hybrid_engine":
        from opsd.rollout.hybrid_engine import HybridEngineRollout

        if student_engine is None or tokenizer is None:
            raise ValueError("hybrid_engine rollout needs both student_engine and tokenizer")
        return HybridEngineRollout(engine=student_engine, tokenizer=tokenizer, cfg=rollout_cfg)

    if engine_name == "vllm":
        from opsd.rollout.vllm import VLLMRollout

        if tokenizer is None:
            raise ValueError("vllm rollout needs a tokenizer for length accounting")
        return VLLMRollout(
            cfg=rollout_cfg,
            tokenizer=tokenizer,
            student_engine=student_engine,
            student_model_path=student_model_path,
            arch=arch,
        )

    raise ValueError(f"Unknown rollout engine {engine_name!r}; choose from 'hybrid_engine' | 'vllm'")
