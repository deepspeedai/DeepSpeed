# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Engine-path DeepCompile validation for AutoEP + AutoTP folding."""

from types import SimpleNamespace

import pytest

from deepspeed.module_inject.auto_ep_config import AutoEPConfig
from deepspeed.runtime.engine import DeepSpeedEngine


def test_deepcompile_folded_rejected_before_autoep_process_groups(monkeypatch):
    engine = object.__new__(DeepSpeedEngine)
    engine.mpu = None
    engine._config = SimpleNamespace(
        expert_parallel_config=AutoEPConfig(enabled=True, autoep_size=2, expert_tensor_parallel_size=1),
        compile_config=SimpleNamespace(deepcompile=True),
        use_data_before_expert_parallel_=False,
        tensor_parallel_config=SimpleNamespace(preset_model=None),
    )
    engine.autotp_size = lambda: 2
    engine._autoep_sequence_parallel_world_size = lambda: 1
    engine.zero_optimization_stage = lambda: 0
    engine.zero_offload_optimizer = lambda: None
    engine.zero_offload_param = lambda: None

    group_creations = []
    monkeypatch.setattr("deepspeed.runtime.engine.dist.get_world_size", lambda: 4)
    monkeypatch.setattr("deepspeed.runtime.engine.dist.get_rank", lambda group=None: 0)
    monkeypatch.setattr("deepspeed.runtime.engine.groups._get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr("deepspeed.runtime.engine.groups._create_expert_and_data_parallel",
                        lambda **kwargs: group_creations.append(kwargs))
    monkeypatch.setattr("deepspeed.runtime.engine.groups._get_expert_parallel_group", lambda name: object())

    class _AutoEPNoop:

        def __init__(self, model, config):
            pass

        def ep_parser(self):
            return []

    monkeypatch.setattr("deepspeed.module_inject.auto_ep.AutoEP", _AutoEPNoop)

    with pytest.raises(ValueError, match="DeepCompile.*AutoEP\\+AutoTP folding"):
        engine._configure_expert_parallel(object())

    assert group_creations == []
