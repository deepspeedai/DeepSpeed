# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Thin wrapper that adapts the DeepSpeed rollout engine to the OPSD config.

The actual implementation lives in ``deepspeed.runtime.rollout``.
"""

from opsd.config import RolloutConfig
from opsd.rollout.base import RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig

from deepspeed.runtime.rollout import HybridEngineRollout as _HybridEngineRollout


class HybridEngineRollout(_HybridEngineRollout):
    """OPSD-specific wrapper that reads config from RolloutConfig."""

    def __init__(self, engine, tokenizer, cfg: RolloutConfig):
        if cfg.engine != "hybrid_engine":
            raise ValueError(f"RolloutConfig.engine must be 'hybrid_engine'; got {cfg.engine!r}")
        cb_size = getattr(cfg, "continuous_batching_size", 0)
        use_gc = getattr(cfg, "use_graph_capture", False)
        super().__init__(
            engine=engine,
            tokenizer=tokenizer,
            continuous_batching_size=cb_size,
            use_graph_capture=use_gc,
        )

