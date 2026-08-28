# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest
from unit.simple_model import create_config_from_dict


@pytest.mark.inference
@pytest.mark.skipif(not get_accelerator().is_available(), reason="requires accelerator")
class TestInferenceCudaGraphConfig:

    def test_cuda_graph_with_kernel_inject_raises(self):
        # Regression test for https://github.com/deepspeedai/DeepSpeed/issues/8330
        from transformers import LlamaConfig, LlamaForCausalLM

        model_config = LlamaConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            torch_dtype=torch.bfloat16,
        )
        model = LlamaForCausalLM(model_config).to("cuda")

        with pytest.raises(ValueError, match="enable_cuda_graph is not supported"):
            deepspeed.init_inference(
                model,
                config={
                    "return_tuple": False,
                    "enable_cuda_graph": True,
                    "replace_with_kernel_inject": True,
                },
            )


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
