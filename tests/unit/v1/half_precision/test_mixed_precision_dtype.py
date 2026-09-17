# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import types

import torch
import pytest

import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest


def _module_with_fp32_buffer(hidden_dim=8):
    """Linear layer plus an fp32 buffer that mimics the rotary inv_freq."""
    module = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim))
    module.register_buffer("inv_freq", torch.ones(hidden_dim, dtype=torch.float32))
    return module


class TestMixedPrecisionDtypeResolution:

    def _engine(self, param_dtype=None, buffer_dtype=None, fp16=False, bf16=False):
        cfg = types.SimpleNamespace(param_dtype=param_dtype, buffer_dtype=buffer_dtype)
        return types.SimpleNamespace(_config=cfg, fp16_enabled=lambda: fp16, bfloat16_enabled=lambda: bf16)

    def test_param_dtype_from_bf16_flag(self):
        param_dtype, buffer_dtype = DeepSpeedEngine._mixed_precision_dtypes(self._engine(bf16=True))
        assert param_dtype == torch.bfloat16
        assert buffer_dtype is None  # buffers preserved by default

    def test_param_dtype_from_fp16_flag(self):
        param_dtype, _ = DeepSpeedEngine._mixed_precision_dtypes(self._engine(fp16=True))
        assert param_dtype == torch.half

    def test_no_low_precision_when_disabled(self):
        param_dtype, buffer_dtype = DeepSpeedEngine._mixed_precision_dtypes(self._engine())
        assert param_dtype is None and buffer_dtype is None

    @pytest.mark.parametrize("cfg_str,expected", [("bf16", torch.bfloat16), ("fp16", torch.half),
                                                  ("fp32", torch.float32)])
    def test_buffer_dtype_string_resolution(self, cfg_str, expected):
        _, buffer_dtype = DeepSpeedEngine._mixed_precision_dtypes(self._engine(buffer_dtype=cfg_str, bf16=True))
        assert buffer_dtype == expected


class TestMixedPrecisionConfigValidation:

    def _engine(self, param_dtype=None, fp16=False, bf16=False):
        cfg = types.SimpleNamespace(param_dtype=param_dtype, buffer_dtype=None)
        return types.SimpleNamespace(_config=cfg, fp16_enabled=lambda: fp16, bfloat16_enabled=lambda: bf16)

    def test_unset_param_dtype_ok(self):
        DeepSpeedEngine._assert_valid_mixed_precision_config(self._engine(bf16=True))  # no raise

    def test_matching_param_dtype_ok(self):
        DeepSpeedEngine._assert_valid_mixed_precision_config(self._engine(param_dtype="bf16", bf16=True))  # no raise

    @pytest.mark.parametrize("param_dtype,fp16,bf16", [("fp16", False, True), ("fp32", False, True),
                                                       ("bf16", True, False), ("bf16", False, False)])
    def test_conflicting_param_dtype_rejected(self, param_dtype, fp16, bf16):
        with pytest.raises(AssertionError):
            DeepSpeedEngine._assert_valid_mixed_precision_config(
                self._engine(param_dtype=param_dtype, fp16=fp16, bf16=bf16))


class TestCastModuleMixedPrecision:

    def _engine(self, module):
        return types.SimpleNamespace(module=module)

    def test_params_cast_buffers_preserved_by_default(self):
        module = _module_with_fp32_buffer()
        DeepSpeedEngine._cast_module_mixed_precision(self._engine(module), torch.bfloat16, None, False)
        assert all(p.dtype == torch.bfloat16 for p in module.parameters())
        assert module.inv_freq.dtype == torch.float32  # the regression this PR fixes

    def test_buffer_dtype_forces_cast(self):
        module = _module_with_fp32_buffer()
        DeepSpeedEngine._cast_module_mixed_precision(self._engine(module), torch.bfloat16, torch.bfloat16, False)
        assert module.inv_freq.dtype == torch.bfloat16  # legacy blanket-cast parity

    def test_zero_init_skips_param_cast(self):
        module = _module_with_fp32_buffer()
        DeepSpeedEngine._cast_module_mixed_precision(self._engine(module), torch.bfloat16, None, True)
        assert all(p.dtype == torch.float32 for p in module.parameters())  # zero-init owns the param dtype

    def test_param_dtype_none_leaves_params(self):
        module = _module_with_fp32_buffer()
        DeepSpeedEngine._cast_module_mixed_precision(self._engine(module), None, torch.bfloat16, False)
        assert all(p.dtype == torch.float32 for p in module.parameters())
        assert module.inv_freq.dtype == torch.bfloat16


@pytest.mark.skipif(torch.bfloat16 not in get_accelerator().supported_dtypes(), reason="bf16 not supported")
@pytest.mark.parametrize("zero_stage", [0, 3])
class TestMixedPrecisionDtypeEndToEnd(DistributedTest):
    world_size = 1

    def _config(self, zero_stage, buffer_dtype=None):
        data_types = {}
        if buffer_dtype is not None:
            data_types["buffer_dtype"] = buffer_dtype
        return {
            "train_batch_size": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3,
                    "torch_adam": True
                }
            },
            "bf16": {
                "enabled": True
            },
            "data_types": data_types,
            "zero_optimization": {
                "stage": zero_stage
            }
        }

    def test_config_defaults(self, zero_stage):
        model = _module_with_fp32_buffer(1024)
        engine, _, _, _ = deepspeed.initialize(config=self._config(zero_stage),
                                               model=model,
                                               model_parameters=model.parameters())
        assert engine._config.param_dtype is None
        assert engine._config.buffer_dtype is None

    def test_buffer_preserved_by_default(self, zero_stage):
        model = _module_with_fp32_buffer(1024)
        engine, _, _, _ = deepspeed.initialize(config=self._config(zero_stage),
                                               model=model,
                                               model_parameters=model.parameters())
        assert all(p.dtype == torch.bfloat16 for p in engine.module.parameters())
        assert engine.module.inv_freq.dtype == torch.float32

    def test_buffer_dtype_opt_in(self, zero_stage):
        model = _module_with_fp32_buffer(1024)
        engine, _, _, _ = deepspeed.initialize(config=self._config(zero_stage, buffer_dtype="bf16"),
                                               model=model,
                                               model_parameters=model.parameters())
        assert engine.module.inv_freq.dtype == torch.bfloat16
