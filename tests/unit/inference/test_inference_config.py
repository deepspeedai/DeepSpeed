# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from pydantic import ValidationError
from deepspeed.inference.config import DeepSpeedInferenceConfig
from unit.common import DistributedTest
from unit.simple_model import create_config_from_dict


@pytest.mark.inference
@pytest.mark.parametrize("field", ["max_out_tokens", "max_tokens"])
@pytest.mark.parametrize("value", [-1, 0])
def test_max_out_tokens_must_be_positive(field, value):
    with pytest.raises(ValidationError):
        DeepSpeedInferenceConfig(**{field: value})


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
