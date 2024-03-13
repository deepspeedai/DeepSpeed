# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from unit.simple_model import SimpleModel
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.utils import required_torch_version

from unit.common import DistributedTest

pytestmark = pytest.mark.skipif(not required_torch_version(min_version=2.1),
                                reason="Compile tests requires Pytorch version 2.1 or above")

custom_backend_called = False
custom_compler_fn_called = False

if deepspeed.is_compile_supported():
    # PyTorch v1 does not have torch.fx
    def custom_backend(gm: torch.fx.GraphModule, example_inputs):
        global custom_backend_called
        custom_backend_called = True
        return gm.forward

    def custom_compiler_fn(module: torch.nn.Module):
        global custom_compler_fn_called
        custom_compler_fn_called = True
        return torch.compile(module)


@pytest.fixture
def base_config():
    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        },
        "fp16": {
            "enabled": True
        },
        "compile": {
            "enabled": True,
            "backend": "inductor"
        }
    }

    if get_accelerator().device_name() == 'hpu':
        config_dict['compile']['backend'] = 'hpu_backend'
    return config_dict


class TestConfigLoad(DistributedTest):
    world_size = 1
    non_daemonic_procs = True

    def _init_engine(self, config):
        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        return engine

    def _run_model(self, engine):
        train_batch_size = 1
        device = torch.device(get_accelerator().current_device_name())
        dtype = engine.module.linears[0].weight.dtype
        hidden_dim = engine.module.linears[0].weight.shape[1]
        x = torch.rand(train_batch_size, hidden_dim, device=device, dtype=dtype)
        y = torch.randn_like(x)
        engine(x, y)

    @pytest.mark.skipif(not deepspeed.is_compile_supported(), reason="torch.compile is not supported")
    def test_compile(self, base_config):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not support this test yet.")
        engine = self._init_engine(base_config)
        self._run_model(engine)
        assert engine.is_compiled

    @pytest.mark.skipif(not deepspeed.is_compile_supported(), reason="torch.compile is not supported")
    def test_custom_backend(self, base_config):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not support this test yet.")
        global custom_backend_called
        custom_backend_called = False

        engine = self._init_engine(base_config)
        engine.set_backend(f"{__name__}.custom_backend")
        self._run_model(engine)
        assert custom_backend_called

    def test_compile_disabled(self, base_config):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not support this test yet.")
        base_config["compile"]["enabled"] = False
        engine = self._init_engine(base_config)
        self._run_model(engine)

    @pytest.mark.skipif(not deepspeed.is_compile_supported(), reason="torch.compile is not supported")
    def test_compile_kwargs(self, base_config):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not support this test yet.")
        base_config["compile"]["kwargs"] = {"mode": "default"}
        engine = self._init_engine(base_config)
        self._run_model(engine)
        assert "mode" in engine.torch_compile_kwargs

    @pytest.mark.skipif(not deepspeed.is_compile_supported(), reason="torch.compile is not supported")
    def test_set_compile_kwargs(self, base_config):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not support this test yet.")
        engine = self._init_engine(base_config)
        engine.set_torch_compile_kwargs({"mode": "default"})
        self._run_model(engine)
        assert "mode" in engine.torch_compile_kwargs

    @pytest.mark.skipif(not deepspeed.is_compile_supported(), reason="torch.compile is not supported")
    def test_set_compiler_fn(self, base_config):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not support this test yet.")
        global custom_compler_fn_called
        custom_compler_fn_called = False

        engine = self._init_engine(base_config)
        engine.set_compiler_fn(custom_compiler_fn)
        self._run_model(engine)
        assert custom_compler_fn_called
