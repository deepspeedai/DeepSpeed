# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import sys
import subprocess
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


def test_non_jit_branch_sorts_and_dedupes_gencode_flags():
    builder = make_builder(jit_mode=False)

    with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "8.0;7.5;8.0;7.0"}, clear=False):
        args = builder.compute_capability_args()
        assert os.environ["TORCH_CUDA_ARCH_LIST"] == "7.0;7.5;8.0"

    assert args == [
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
    ]


def test_non_jit_branch_canonicalizes_mixed_ptx_variants_to_one_sm_and_one_ptx():
    # For mixed inputs such as "8.0;8.0+PTX" or "8.0+PTX;8.0", PyTorch
    # canonicalizes the architecture to one sm_80 entry plus one compute_80
    # PTX entry. Dedupe by the canonical numeric arch so we match.
    expected_arch_list = "7.5;8.0+PTX"
    expected_args = [
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_80,code=compute_80",
    ]

    for arch_input in ("8.0;8.0+PTX;7.5", "7.5;8.0+PTX;8.0", "8.0+PTX;7.5;8.0", "8.0;7.5;8.0+PTX"):
        builder = make_builder(jit_mode=False)
        with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": arch_input}, clear=False):
            args = builder.compute_capability_args()
            assert os.environ["TORCH_CUDA_ARCH_LIST"] == expected_arch_list, arch_input
        assert args == expected_args, arch_input


def test_non_jit_branch_canonical_dedupe_mixed_ptx_combinations():
    # Lock in the four mixed-PTX combinations for a single arch so the dedupe
    # behavior cannot regress on either ordering or duplication.
    builder = make_builder(jit_mode=False)
    cases = [
        ("8.0;8.0+PTX", "8.0+PTX", ["-gencode=arch=compute_80,code=sm_80",
                                    "-gencode=arch=compute_80,code=compute_80"]),
        ("8.0+PTX;8.0", "8.0+PTX", ["-gencode=arch=compute_80,code=sm_80",
                                    "-gencode=arch=compute_80,code=compute_80"]),
        ("8.0;8.0", "8.0", ["-gencode=arch=compute_80,code=sm_80"]),
        ("8.0+PTX;8.0+PTX", "8.0+PTX",
         ["-gencode=arch=compute_80,code=sm_80", "-gencode=arch=compute_80,code=compute_80"]),
    ]
    for arch_input, expected_arch_list, expected_args in cases:
        with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": arch_input}, clear=False):
            args = builder.compute_capability_args()
            assert os.environ["TORCH_CUDA_ARCH_LIST"] == expected_arch_list, arch_input
        assert args == expected_args, arch_input


def test_cuda_capability_major_skips_probe_when_context_not_initialized():
    # Probing device properties forces a lazy CUDA-context init, which creates a
    # CUDA context. Doing that while checking op compatibility at "import deepspeed"
    # time poisons fork()-based multiprocessing (issue #7918): a forked child cannot
    # reuse the parent's context. With no context yet, the probe must be skipped.
    builder = make_builder()
    with patch.object(CUDA_API, "is_initialized", return_value=False):
        with patch.object(
                CUDA_API, "get_device_properties",
                side_effect=AssertionError("must not initialize CUDA / poison fork")) as get_device_properties:
            assert builder.cuda_capability_major() is None
    get_device_properties.assert_not_called()


def test_cuda_capability_major_probes_when_context_already_initialized():
    # When a CUDA context already exists (e.g. at op load time), probing is safe
    # and must report the real compute-capability major.
    builder = make_builder()
    device_properties = MagicMock(major=8)
    with patch.object(CUDA_API, "is_initialized", return_value=True):
        with patch.object(CUDA_API, "_is_in_bad_fork", return_value=False):
            with patch.object(CUDA_API, "get_device_properties",
                              return_value=device_properties) as get_device_properties:
                assert builder.cuda_capability_major() == 8
    get_device_properties.assert_called_once_with(0)


def test_cuda_capability_major_skips_probe_in_bad_fork():
    # Inside a forked child that inherited an initialized-but-invalid context,
    # probing would raise "Cannot re-initialize CUDA in forked subprocess", so it
    # must be skipped there as well.
    builder = make_builder()
    with patch.object(CUDA_API, "is_initialized", return_value=True):
        with patch.object(CUDA_API, "_is_in_bad_fork", return_value=True):
            with patch.object(CUDA_API,
                              "get_device_properties",
                              side_effect=AssertionError("must not probe in a forked child")) as get_device_properties:
                assert builder.cuda_capability_major() is None
    get_device_properties.assert_not_called()


def test_forked_child_can_use_cuda_after_importing_deepspeed():
    # Core contract of issue #7918: after the parent process runs
    # ``import deepspeed``, a forked child must still be able to initialize and
    # use CUDA. If import created a CUDA context in the parent, the child fails
    # with "Cannot re-initialize CUDA in forked subprocess". Everything runs in a
    # dedicated subprocess so a poisoned parent cannot leak into the pytest worker
    # or other tests.
    program = "\n".join([
        "import os, sys",
        "import torch",
        "import deepspeed  # must not create a CUDA context in the parent",
        # device_count() is NVML-based and never initializes a context, so it is
        # a fork-safe way to check for a GPU before forking.
        "if torch.cuda.device_count() == 0:",  #ignore-cuda
        "    print('NO_CUDA'); sys.exit(0)",
        "pid = os.fork()",
        "if pid == 0:",
        "    try:",
        "        torch.ones(1, device='cuda')",
        "        os._exit(0)",
        "    except Exception as exc:",
        "        sys.stderr.write(repr(exc))",
        "        os._exit(1)",
        "_, status = os.waitpid(pid, 0)",
        "sys.exit(os.waitstatus_to_exitcode(status))",
    ])
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env=env, timeout=300)
    if result.returncode != 0 and ("No module named 'deepspeed'" in result.stderr
                                   or "No module named 'torch'" in result.stderr):
        pytest.skip("deepspeed/torch not importable in a subprocess in this environment")
    if result.stdout.strip() == "NO_CUDA":
        pytest.skip("no CUDA device available")
    assert result.returncode == 0, ("forked child could not use CUDA after 'import deepspeed' "
                                    "(a CUDA context was created during import, issue #7918):\n" + result.stderr)
