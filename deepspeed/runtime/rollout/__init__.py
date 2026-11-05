# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engines for on-policy generation during RL/distillation training.

Provides:
  - :class:`RolloutEngine` — abstract base class
  - :class:`RolloutRequest`, :class:`RolloutBatch`, :class:`SamplingConfig` — dataclasses
  - :class:`ResponseTokenBatch` — response-only causal training targets
  - :func:`group_advantages`, :func:`grpo_policy_loss` — GRPO advantages and PPO-clip policy loss
  - :class:`HybridEngineRollout` — concrete implementation using DeepSpeed hybrid engine
  - :class:`Tool`, :class:`ToolSet`, :func:`parse_hermes_tool_calls`, :func:`parse_qwen3_coder_tool_calls` — tools
    for multi-turn rollout
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
from deepspeed.runtime.rollout.grpo import (GRPOLossOutput, group_advantages, grpo_policy_loss,
                                            old_log_probs_from_rollout)
from deepspeed.runtime.rollout.opsd import JSDLossOutput, ResponseTokenBatch, generalized_jsd_loss
from deepspeed.runtime.rollout.tool import Tool, ToolCall, ToolSet, parse_hermes_tool_calls, parse_qwen3_coder_tool_calls

__all__ = [
    "GRPOLossOutput",
    "HybridEngineRollout",
    "JSDLossOutput",
    "RolloutBatch",
    "RolloutConfig",
    "RolloutEngine",
    "RolloutRequest",
    "ResponseTokenBatch",
    "SamplingConfig",
    "Tool",
    "ToolCall",
    "ToolSet",
    "build_rollout",
    "generalized_jsd_loss",
    "group_advantages",
    "grpo_policy_loss",
    "old_log_probs_from_rollout",
    "parse_hermes_tool_calls",
    "parse_qwen3_coder_tool_calls",
]


def build_rollout(rollout_cfg, generation_engine=None, tokenizer=None, tools=None, **kwargs):
    """Factory: construct the rollout engine specified by ``rollout_cfg.engine``.

    Args:
        rollout_cfg: :class:`RolloutConfig` (or any object with an ``engine``
            attribute set to ``"hybrid_engine"``).
        generation_engine: DeepSpeed engine wrapping the student model.
        tokenizer: HuggingFace tokenizer.
        tools: Tools the model may call in multi-turn rollout: a
            :class:`ToolSet` or a list of :class:`Tool`.
    """
    engine_name = rollout_cfg.engine
    if engine_name == "hybrid_engine":
        if generation_engine is None or tokenizer is None:
            raise ValueError("hybrid_engine rollout needs both generation_engine and tokenizer")
        if tools is not None and not isinstance(tools, ToolSet):
            tools = ToolSet(tools)
        return HybridEngineRollout(engine=generation_engine, tokenizer=tokenizer, cfg=rollout_cfg, tools=tools)

    raise ValueError(f"Unknown rollout engine {engine_name!r}; choose from 'hybrid_engine'")
