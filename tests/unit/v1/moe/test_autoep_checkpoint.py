# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Compact AutoEP checkpoint tests."""

import os

import pytest
import torch
import torch.nn as nn

from deepspeed.runtime.config import DeepSpeedConfig
from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import (
    UNSUPPORTED_LOAD_BALANCE_VALUES,
    assert_load_balance_coeff_rejection_message,
    init_autoep_engine,
)


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("include_key", [False, True])
def test_load_balance_coeff_disabled_values_accepted_by_deepspeed_config(enabled, include_key):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "expert_parallel": {
            "enabled": enabled,
            "autoep_size": 1,
            "preset_model": "mixtral",
        },
    }
    if include_key:
        config["expert_parallel"]["load_balance_coeff"] = None

    ds_config = DeepSpeedConfig(config)

    assert ds_config.expert_parallel_config.load_balance_coeff is None
    assert ds_config.expert_parallel_config._load_balance_coeff_explicit is include_key


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("value", UNSUPPORTED_LOAD_BALANCE_VALUES)
def test_load_balance_coeff_rejected_by_deepspeed_config(enabled, value):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "expert_parallel": {
            "enabled": enabled,
            "autoep_size": 1,
            "preset_model": "mixtral",
            "load_balance_coeff": value,
        },
    }

    with pytest.raises(ValueError) as exc_info:
        DeepSpeedConfig(config)
    assert_load_balance_coeff_rejection_message(exc_info.value, value)


class TestAutoEPCheckpointSaveLoad(DistributedTest):
    world_size = 1

    def test_save_load_same_ep_and_metadata(self, tmpdir):
        engine = init_autoep_engine(ep_size=1)
        params_before = {name: param.detach().clone() for name, param in engine.module.named_parameters()}
        save_dir = str(tmpdir)
        tag = "autoep"

        engine.save_checkpoint(save_dir, tag=tag)

        checkpoint = torch.load(os.path.join(save_dir, tag, "mp_rank_00_model_states.pt"),
                                map_location="cpu",
                                weights_only=False)
        metadata = checkpoint["ds_autoep_layers"]
        assert len(metadata) == 2
        for entry in metadata:
            assert {"moe_layer_id", "module_path", "num_experts", "num_local_experts", "ep_size"} <= entry.keys()
            assert entry["num_experts"] == entry["num_local_experts"] * entry["ep_size"]

        reloaded = init_autoep_engine(ep_size=1)
        reloaded.load_checkpoint(save_dir, tag=tag)
        for name, param in reloaded.module.named_parameters():
            assert torch.equal(param, params_before[name]), f"{name} changed after same-EP reload"

    def test_autoep_metadata_schema_validation(self):
        from deepspeed.runtime.engine import DeepSpeedEngine

        with pytest.raises(RuntimeError, match="malformed"):
            DeepSpeedEngine.load_moe_state_dict(checkpoint_path="/fake",
                                                tag="fake",
                                                state_dict={},
                                                old_moe_load=False,
                                                model=nn.Linear(1, 1),
                                                autoep_layers="not_a_list")

        with pytest.raises(RuntimeError, match="missing fields"):
            DeepSpeedEngine.load_moe_state_dict(checkpoint_path="/fake",
                                                tag="fake",
                                                state_dict={},
                                                old_moe_load=False,
                                                model=nn.Linear(1, 1),
                                                autoep_layers=[{
                                                    "moe_layer_id": 0
                                                }])
