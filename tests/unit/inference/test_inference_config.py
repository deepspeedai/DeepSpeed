# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.inference.config import DeepSpeedInferenceConfig
from deepspeed.inference.engine import _validate_keep_module_on_host
from unit.common import DistributedTest
from unit.simple_model import create_config_from_dict


@pytest.mark.inference
def test_keep_module_on_host_requires_injection_or_auto_tp():
    with pytest.raises(ValueError, match="requires an injection policy, kernel injection, or AutoTP"):
        deepspeed.init_inference(
            torch.nn.Module(),
            dtype=torch.float32,
            keep_module_on_host=True,
        )


@pytest.mark.inference
@pytest.mark.parametrize(
    "mode_config",
    [
        {
            "injection_policy": {
                torch.nn.Linear: ("output", )
            }
        },
        {
            "replace_with_kernel_inject": True
        },
        {
            "tensor_parallel": {
                "tp_size": 2
            }
        },
        {
            "mp_size": 2
        },
    ],
    ids=["injection-policy", "kernel-injection", "auto-tp", "deprecated-mp-size"],
)
def test_keep_module_on_host_accepts_supported_modes(mode_config):
    config = DeepSpeedInferenceConfig(keep_module_on_host=True, **mode_config)

    _validate_keep_module_on_host(config)
    assert config.keep_module_on_host
    if "mp_size" in mode_config:
        assert config.tensor_parallel.tp_size == mode_config["mp_size"]


@pytest.mark.inference
def test_keep_module_on_host_rejects_single_rank_mpu_mode():
    config = DeepSpeedInferenceConfig(
        keep_module_on_host=True,
        tensor_parallel={"mpu": object()},
    )

    with pytest.raises(ValueError, match="requires an injection policy, kernel injection, or AutoTP"):
        _validate_keep_module_on_host(config)


@pytest.mark.inference
class TestInferenceConfig(DistributedTest):
    world_size = 1

    def test_overlap_kwargs(self):
        config = {"replace_with_kernel_inject": True, "dtype": torch.float32}
        kwargs = {"replace_with_kernel_inject": True}

        engine = deepspeed.init_inference(torch.nn.Module(), config=config, **kwargs)
        assert engine._config.replace_with_kernel_inject

    def test_overlap_kwargs_conflict(self):
        config = {"replace_with_kernel_inject": True}
        kwargs = {"replace_with_kernel_inject": False}

        with pytest.raises(ValueError):
            engine = deepspeed.init_inference(torch.nn.Module(), config=config, **kwargs)

    def test_kwargs_and_config(self):
        config = {"replace_with_kernel_inject": True}
        kwargs = {"dtype": torch.float32}

        engine = deepspeed.init_inference(torch.nn.Module(), config=config, **kwargs)
        assert engine._config.replace_with_kernel_inject
        assert engine._config.dtype == kwargs["dtype"]

    def test_json_config(self, tmpdir):
        config = {"replace_with_kernel_inject": True, "dtype": "torch.float32"}
        config_json = create_config_from_dict(tmpdir, config)

        engine = deepspeed.init_inference(torch.nn.Module(), config=config_json)
        assert engine._config.replace_with_kernel_inject

    def test_moe_backward_compat_bool(self):
        # `moe` accepts a bool for backward compatibility (moe: Union[bool, DeepSpeedMoEConfig]);
        # it should build a DeepSpeedMoEConfig rather than raising a validation error.
        from deepspeed.inference.config import DeepSpeedInferenceConfig, DeepSpeedMoEConfig

        for value in (True, False):
            config = DeepSpeedInferenceConfig(moe=value)
            assert isinstance(config.moe, DeepSpeedMoEConfig)
            assert config.moe.enabled == value
