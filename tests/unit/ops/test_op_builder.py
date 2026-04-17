# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

BUILDER_PATH = Path(__file__).resolve().parents[3] / "op_builder" / "builder.py"
BUILDER_SPEC = importlib.util.spec_from_file_location("test_op_builder_module", BUILDER_PATH)
builder_module = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(builder_module)
CUDAOpBuilder = builder_module.CUDAOpBuilder

BUILDER_MODULE = builder_module
CUDA_API = BUILDER_MODULE.torch.cuda  #ignore-cuda


class _StubCUDAOpBuilder(CUDAOpBuilder):
    BUILD_VAR = "STUB_BUILDER"
    NAME = "stub"

    def __init__(self):
        super().__init__(name="stub")

    def absolute_name(self):
        return "deepspeed.ops.stub"

    def sources(self):
        return []

    def include_paths(self):
        return []


def make_builder(**overrides):
    builder = _StubCUDAOpBuilder()
    for key, value in overrides.items():
        setattr(builder, key, value)
    return builder


def assert_jit_uses_explicit_arch_list(builder, expected_arch_list, env_updates=None):
    env_updates = env_updates or {}

    with patch.dict(os.environ, env_updates, clear=False):
        if "TORCH_CUDA_ARCH_LIST" not in env_updates:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        with patch.object(CUDA_API, "device_count",
                          side_effect=AssertionError("probe should not be called")) as device_count:
            with patch.object(CUDA_API,
                              "get_device_capability",
                              side_effect=AssertionError("probe should not be called")) as get_device_capability:
                assert builder.compute_capability_args() == []
                assert os.environ["TORCH_CUDA_ARCH_LIST"] == expected_arch_list

    device_count.assert_not_called()
    get_device_capability.assert_not_called()


def test_jit_mode_prefers_explicit_arch_lists_before_cuda_probe():
    assert_jit_uses_explicit_arch_list(make_builder(jit_mode=True, _jit_arch_list="8.0;8.9"), "8.0;8.9+PTX")
    assert_jit_uses_explicit_arch_list(make_builder(jit_mode=True), "8.0;8.9+PTX", {"TORCH_CUDA_ARCH_LIST": "8.0 8.9"})


def test_bad_fork_jit_without_arch_list_raises_actionable_error():
    builder = make_builder(jit_mode=True)

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        with patch.object(CUDA_API, "_is_in_bad_fork", return_value=True):
            with patch.object(CUDA_API, "device_count",
                              side_effect=AssertionError("probe should not be called")) as device_count:
                with pytest.raises(RuntimeError, match="TORCH_CUDA_ARCH_LIST"):
                    builder.compute_capability_args()

    device_count.assert_not_called()


def test_jit_mode_probes_devices_when_safe_and_errors_without_visible_gpus():
    builder = make_builder(jit_mode=True)

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        with patch.object(CUDA_API, "_is_in_bad_fork", return_value=False):
            with patch.object(CUDA_API, "device_count", return_value=2) as device_count:
                with patch.object(CUDA_API, "get_device_capability", side_effect=[(7, 0),
                                                                                  (8, 9)]) as get_device_capability:
                    assert builder.compute_capability_args() == []
                    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "7.0;8.9+PTX"
                    assert builder.enable_bf16 is False

    device_count.assert_called_once_with()
    assert get_device_capability.call_count == 2

    builder = make_builder(jit_mode=True)
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        with patch.object(CUDA_API, "_is_in_bad_fork", return_value=False):
            with patch.object(CUDA_API, "device_count", return_value=0):
                with pytest.raises(RuntimeError, match="no CUDA devices"):
                    builder.compute_capability_args()


def test_jit_load_restores_env_and_state_after_failure():
    builder = make_builder()

    def fail_nvcc_args():
        assert getattr(builder, "_jit_arch_list", None) == "8.9"
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
        raise RuntimeError("build failed")

    with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "8.9"}, clear=False):
        with patch.object(builder, "is_compatible", return_value=True):
            with patch.object(CUDAOpBuilder, "is_rocm_pytorch", return_value=False):
                with patch.object(CUDA_API, "is_available", return_value=True):
                    with patch("torch.utils.cpp_extension.verify_ninja_availability", return_value=None):
                        with patch.object(builder, "nvcc_args", side_effect=fail_nvcc_args):
                            with pytest.raises(RuntimeError, match="build failed"):
                                builder.jit_load(verbose=False)

        assert getattr(builder, "_jit_arch_list", None) is None
        assert builder.jit_mode is False
        assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.9"


def test_jit_load_restores_state_after_success():
    builder = make_builder()
    op_module = MagicMock()

    def successful_nvcc_args():
        assert builder._jit_arch_list == "8.9"
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
        return []

    with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "8.9"}, clear=False):
        with patch.object(builder, "is_compatible", return_value=True):
            with patch.object(CUDAOpBuilder, "is_rocm_pytorch", return_value=False):
                with patch.object(CUDA_API, "is_available", return_value=True):
                    with patch("torch.utils.cpp_extension.verify_ninja_availability", return_value=None):
                        with patch.object(builder, "nvcc_args", side_effect=successful_nvcc_args):
                            with patch.object(builder, "cxx_args", return_value=[]):
                                with patch("torch.utils.cpp_extension.load", return_value=op_module):
                                    assert builder.jit_load(verbose=False) is op_module

        assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.9"
        assert getattr(builder, "_jit_arch_list", None) is None
        assert builder.jit_mode is False


def test_non_jit_branch_unchanged():
    builder = make_builder(jit_mode=False)

    with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "8.0;8.9+PTX"}, clear=False):
        args = builder.compute_capability_args()

    assert args == [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_89,code=compute_89",
    ]
