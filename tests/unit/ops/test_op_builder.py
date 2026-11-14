# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import platform
import struct
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
MissingCUDAException = builder_module.MissingCUDAException
installed_cuda_version = builder_module.installed_cuda_version
probe_is_compatible = builder_module.probe_is_compatible


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

    import torch.utils.cpp_extension  # noqa: F401
    builder = make_builder(jit_mode=True)

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        with patch.object(CUDA_API, "_is_in_bad_fork", return_value=False):
            with patch.object(CUDA_API, "device_count", return_value=2) as device_count:
                with patch.object(CUDA_API, "get_device_capability", side_effect=[(7, 0),
                                                                                  (8, 9)]) as get_device_capability:
                    assert builder.compute_capability_args() == []
                    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "7.0;8.9+PTX"

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
                with patch.object(builder_module, "assert_no_cuda_mismatch", return_value=True):
                    with patch("torch.utils.cpp_extension.verify_ninja_availability", return_value=None):
                        with patch.object(builder, "nvcc_args", side_effect=fail_nvcc_args):
                            with pytest.raises(RuntimeError, match="build failed"):
                                builder.jit_load(verbose=False)

        assert getattr(builder, "_jit_arch_list", None) is None
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
                with patch.object(builder_module, "assert_no_cuda_mismatch", return_value=True):
                    with patch("torch.utils.cpp_extension.verify_ninja_availability", return_value=None):
                        with patch.object(builder, "nvcc_args", side_effect=successful_nvcc_args):
                            with patch.object(builder, "cxx_args", return_value=[]):
                                with patch("torch.utils.cpp_extension.load", return_value=op_module):
                                    assert builder.jit_load(verbose=False) is op_module

        assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.9"
        assert getattr(builder, "_jit_arch_list", None) is None


def test_prebuilt_builder_targets_requested_archs_instead_of_local_gpus():
    # Builders default to JIT mode, which targets the local GPUs and leaves the -gencode flags to PyTorch.
    # Pre-built ops run on other machines, so builder() must emit explicit -gencode flags for the
    # requested arch list.
    builder = make_builder()
    with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "8.0;9.0"}, clear=False):
        with patch.object(CUDAOpBuilder, "is_rocm_pytorch", return_value=False):
            with patch.object(builder_module, "assert_no_cuda_mismatch", return_value=True):
                with patch.object(builder_module, "installed_cuda_version", return_value=(12, 8)):
                    with patch("torch.utils.cpp_extension.CUDAExtension") as cuda_extension:
                        builder.builder()

    nvcc_args = cuda_extension.call_args.kwargs["extra_compile_args"]["nvcc"]
    assert "-gencode=arch=compute_80,code=sm_80" in nvcc_args
    assert "-gencode=arch=compute_90,code=sm_90" in nvcc_args


@pytest.mark.parametrize(
    "arch_list, nvcc_version, expect_bf16",
    [
        # bf16 arithmetic is native from Ampere (8.x). nvcc 12.2+ also compiles it for older targets, but 12.1 fails
        # to compile the BF16 kernels for sm_70 and sm_75.
        ("7.0;8.0", (12, 1), False),
        ("7.0;8.0", (12, 2), True),
        ("8.0;9.0", (12, 1), True),
    ])
def test_bf16_kernels_are_built_for_pre_ampere_targets_only_from_nvcc_12_2(arch_list, nvcc_version, expect_bf16):
    builder = make_builder()
    with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": arch_list}, clear=False):
        with patch.object(CUDAOpBuilder, "is_rocm_pytorch", return_value=False):
            with patch.object(builder_module, "assert_no_cuda_mismatch", return_value=True):
                with patch("torch.utils.cpp_extension.CUDA_HOME", "/usr/local/cuda"):
                    with patch.object(builder_module, "installed_cuda_version", return_value=nvcc_version):
                        with patch("torch.utils.cpp_extension.CUDAExtension") as cuda_extension:
                            builder.builder()

    compile_args = cuda_extension.call_args.kwargs["extra_compile_args"]
    assert ("-DBF16_AVAILABLE" in compile_args["nvcc"]) is expect_bf16
    assert ("-DBF16_AVAILABLE" in compile_args["cxx"]) is expect_bf16


@pytest.mark.parametrize("build_order", [("jit", "prebuilt"), ("prebuilt", "jit")])
def test_prebuilt_and_jit_builds_compile_and_link_an_op_the_same_way(build_order):
    # A pre-built op must be compiled and linked like the same op built at load time. Only the targeted
    # archs differ by nature: JIT builds target the local GPUs and hand them to PyTorch through
    # TORCH_CUDA_ARCH_LIST, while pre-built ops pass -gencode flags for the requested archs.
    builder = make_builder(include_paths=lambda: ["csrc/includes"],
                           extra_ldflags=lambda: ["-lcublas"],
                           libraries_args=lambda: ["curand"])
    load = MagicMock()
    with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "9.0"}, clear=False):
        with patch.object(CUDAOpBuilder, "is_rocm_pytorch", return_value=False):
            with patch.object(builder_module, "assert_no_cuda_mismatch", return_value=True):
                with patch("torch.utils.cpp_extension.verify_ninja_availability", return_value=None):
                    with patch("torch.utils.cpp_extension.load", load):
                        with patch("torch.utils.cpp_extension.CUDAExtension") as cuda_extension:
                            for build_path in build_order:
                                BUILD_PATHS[build_path](builder)

    jit_kwargs = load.call_args.kwargs
    prebuilt_kwargs = cuda_extension.call_args.kwargs
    prebuilt_nvcc_args = [arg for arg in prebuilt_kwargs["extra_compile_args"]["nvcc"] if "-gencode" not in arg]
    # setuptools links each entry of libraries with -l<name>.
    prebuilt_link_args = prebuilt_kwargs["extra_link_args"] + [f"-l{name}" for name in prebuilt_kwargs["libraries"]]
    assert prebuilt_kwargs["extra_compile_args"]["cxx"] == jit_kwargs["extra_cflags"]
    assert prebuilt_nvcc_args == jit_kwargs["extra_cuda_cflags"]
    assert prebuilt_kwargs["include_dirs"] == jit_kwargs["extra_include_paths"]
    assert prebuilt_link_args == jit_kwargs["extra_ldflags"]


def test_prebuilt_op_that_drops_archs_does_not_drop_them_for_the_ops_built_after_it():
    # setup.py pre-builds every op in one process. An op that drops the archs it cannot run on, as
    # fp_quantizer drops those below 8.0, must leave them in place for the ops built after it.
    def keep_ampere_and_newer(ccs):
        return [cc for cc in ccs if cc.major >= 8]

    with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;9.0"}, clear=False):
        make_builder(jit_mode=False, filter_ccs=keep_ampere_and_newer).compute_capability_args()
        next_op_args = make_builder(jit_mode=False).compute_capability_args()

    assert next_op_args == [
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_90,code=compute_90",
    ]


BUILD_PATHS = {
    "prebuilt": lambda builder: builder.builder(),
    "jit": lambda builder: builder.jit_load(verbose=False),
}


def build_with_cuda_versions(build_path, installed_cuda, torch_cuda):
    with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "9.0"}, clear=False):
        with patch.object(CUDAOpBuilder, "is_rocm_pytorch", return_value=False):
            with patch.object(builder_module, "installed_cuda_version", return_value=installed_cuda):
                with patch.object(builder_module.torch.version, "cuda", torch_cuda):
                    with patch("torch.utils.cpp_extension.verify_ninja_availability", return_value=None):
                        with patch("torch.utils.cpp_extension.load"):
                            with patch("torch.utils.cpp_extension.CUDAExtension"):
                                BUILD_PATHS[build_path](make_builder())


@pytest.mark.parametrize("build_path", BUILD_PATHS)
def test_cuda_op_build_warns_instead_of_failing_on_cuda_minor_version_mismatch(build_path, capsys):
    # Like PyTorch's own extension build, and as docs/_tutorials/advanced-install.md says, only a CUDA
    # major version mismatch stops the build; a minor one gets a warning. A table of known compatible
    # minor versions used to reject every CUDA 13 minor mismatch.
    with patch.dict(os.environ, {"DS_SKIP_CUDA_CHECK": "0"}, clear=False):
        build_with_cuda_versions(build_path, installed_cuda=(13, 2), torch_cuda="13.0")

    assert "13.2" in capsys.readouterr().out


@pytest.mark.parametrize("build_path", BUILD_PATHS)
def test_cuda_op_build_fails_on_cuda_major_version_mismatch_unless_check_is_skipped(build_path):
    # PyTorch's load() does not check the toolkit version at all, so JIT builds need this check as much as
    # pre-built ones.
    with patch.dict(os.environ, {"DS_SKIP_CUDA_CHECK": "0"}, clear=False):
        with pytest.raises(builder_module.CUDAMismatchException):
            build_with_cuda_versions(build_path, installed_cuda=(13, 0), torch_cuda="12.8")

    with patch.dict(os.environ, {"DS_SKIP_CUDA_CHECK": "1"}, clear=False):
        build_with_cuda_versions(build_path, installed_cuda=(13, 0), torch_cuda="12.8")


@pytest.mark.parametrize(
    "arch_list, expected_targets",
    [
        # Sorted and deduplicated. No entry asks for PTX, so the highest arch gets it.
        ("8.0;7.5;8.0;7.0",
         ["compute_70,code=sm_70", "compute_75,code=sm_75", "compute_80,code=sm_80", "compute_80,code=compute_80"]),
        # +PTX on either duplicate of an arch survives deduplication (#7871) and replaces the automatic PTX on
        # the highest arch. Keeping only the first or only the last duplicate each fails one of these orders.
        ("7.5+PTX;8.0;7.5", ["compute_75,code=sm_75", "compute_75,code=compute_75", "compute_80,code=sm_80"]),
        ("7.5;8.0;7.5+PTX", ["compute_75,code=sm_75", "compute_75,code=compute_75", "compute_80,code=sm_80"]),
    ])
def test_prebuilt_gencode_flags_are_sorted_deduplicated_and_keep_requested_ptx(arch_list, expected_targets):
    builder = make_builder(jit_mode=False)
    with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": arch_list}, clear=False):
        args = builder.compute_capability_args()

    assert args == [f"-gencode=arch={target}" for target in expected_targets]


def arch_flags(arch_list, jit_mode):
    # JIT builds hand the arch list to PyTorch through TORCH_CUDA_ARCH_LIST, so use PyTorch's own parser to
    # turn it into the -gencode flags that load() compiles with.
    builder = make_builder(jit_mode=jit_mode)
    with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": arch_list}, clear=False):
        args = builder.compute_capability_args()
        if jit_mode:
            from torch.utils.cpp_extension import _get_cuda_arch_flags
            args = _get_cuda_arch_flags()
    return sorted(args)


@pytest.mark.parametrize("jit_mode", [False, True])
@pytest.mark.parametrize(
    "arch_list, expected_targets",
    [
        # An "a" target is kept, and PTX goes on the highest target without a suffix, because PTX for 9.0a
        # only runs on that exact GPU.
        ("8.0;9.0a", ["compute_80,code=compute_80", "compute_80,code=sm_80", "compute_90a,code=sm_90a"]),
        ("9.0a+PTX", ["compute_90a,code=compute_90a", "compute_90a,code=sm_90a"]),
        # Empty entries and extra whitespace are ignored. Without any +PTX entry, the highest target gets PTX
        # in both modes, so the op also runs on newer GPUs.
        ("8.0; 8.6;", ["compute_80,code=sm_80", "compute_86,code=compute_86", "compute_86,code=sm_86"]),
    ])
def test_prebuilt_and_jit_builds_target_the_same_archs(arch_list, expected_targets, jit_mode):
    assert arch_flags(arch_list, jit_mode) == [f"-gencode=arch={target}" for target in expected_targets]


def test_cuda_capability_major_skips_probe_when_context_not_initialized():
    # Probing device properties forces a lazy CUDA-context init, which creates a
    # CUDA context. Doing that while checking op compatibility for deepspeed.ops.__compatible_ops__
    # poisons fork()-based multiprocessing (issue #7918): a forked child cannot
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


@pytest.mark.parametrize("builder_name",
                         ["inference_cutlass_builder.InferenceCutlassBuilder", "ragged_ops.RaggedOpsBuilder"])
def test_jit_load_rejects_only_gpus_below_the_minimum_and_says_why(builder_name):
    # These ops link deepspeed-kernels libraries built only for compute capability 8.0 and newer, so
    # is_compatible() must reject a 7.x GPU rather than let the JIT build fail with no archs left, and the
    # JIT-load error must give the reason: load() checks with verbose=False, which used to leave the reason as
    # "None". An 8.0 GPU must still be accepted.
    module_name, class_name = builder_name.split(".")
    builder = getattr(importlib.import_module(f"deepspeed.ops.op_builder.{module_name}"), class_name)()
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DS_IGNORE_CUDA_DETECTION", None)
        # The real builders use deepspeed.ops.op_builder.builder, not this file's copy of that module.
        with patch("deepspeed.ops.op_builder.builder.installed_cuda_version", return_value=(12, 8)), \
                patch.object(CUDA_API, "is_available", return_value=True), \
                patch.object(builder, "is_rocm_pytorch", return_value=False):
            with patch.object(builder, "cuda_capability_major", return_value=8):
                assert builder.is_compatible() is True
            with patch.object(builder, "cuda_capability_major", return_value=7):
                assert builder.is_compatible() is False
                with pytest.raises(RuntimeError, match=r"compute capability 8\.0"):
                    builder.jit_load(verbose=False)


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


def test_import_deepspeed_defers_device_queries_compilation_and_transformers():
    # These used to run at "import deepspeed" and made it seconds slower: probing every op builder
    # and the module level availability checks query the GPUs (several seconds without GPU
    # persistence mode), @jit_script_compat called torch.compile, @compiler.disable imported
    # torch._dynamo, and transformers was imported eagerly. With the accelerator given explicitly,
    # the import must do none of them. torch._dynamo is only checked from torch 2.4, the first
    # version whose torch._disable_dynamo (used by compiler.lazy_disable) caches the disabled function.
    # deepspeed.compile.backend is checked because it imports torch._dynamo and torch._inductor.
    from deepspeed.accelerator import get_accelerator

    program = "\n".join([
        "import sys, torch",
        "calls = []",
        "def counting(name, fn):",
        "    def wrapper(*args, **kwargs):",
        "        calls.append(name)",
        "        return fn(*args, **kwargs)",
        "    return wrapper",
        "torch.compile = counting('torch.compile', torch.compile)",
        "torch.cuda.device_count = counting('device_count', torch.cuda.device_count)",  #ignore-cuda
        "torch.cuda.is_available = counting('is_available', torch.cuda.is_available)",  #ignore-cuda
        "import deepspeed",
        "assert not calls, calls",
        "assert 'transformers' not in sys.modules, 'transformers was imported'",
        "assert 'deepspeed.compile.backend' not in sys.modules, 'deepspeed.compile.backend was imported'",
        "from deepspeed.utils.torch import required_torch_version",
        "if required_torch_version(min_version=2.4):",
        "    assert 'torch._dynamo' not in sys.modules, 'torch._dynamo was imported'",
    ])
    env = os.environ.copy()
    env["DS_ACCELERATOR"] = get_accelerator().device_name()
    repo_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env=env, timeout=300)
    if result.returncode != 0 and ("No module named 'deepspeed'" in result.stderr
                                   or "No module named 'torch'" in result.stderr):
        pytest.skip("deepspeed/torch not importable in a subprocess in this environment")
    assert result.returncode == 0, "'import deepspeed' did work that should be deferred:\n" + result.stderr


def test_installed_cuda_version_reports_a_runtime_only_cuda_home_as_missing(tmp_path):
    # CUDA_HOME often points at a runtime only install (a pip torch wheel, a conda cudatoolkit) that has
    # no nvcc under bin/. That has to surface as MissingCUDAException, which says the CUDA op cannot be
    # compiled, rather than as an OSError from running the missing nvcc. See #7452.
    with patch("torch.utils.cpp_extension.CUDA_HOME", str(tmp_path)):
        with pytest.raises(MissingCUDAException, match="unable to compile CUDA op"):
            installed_cuda_version()


class _RaisingBuilder:
    name = "raising_stub"

    def is_compatible(self, verbose=False):
        raise MissingCUDAException("CUDA_HOME does not exist, unable to compile CUDA op(s)")


class _AnsweringBuilder:

    def __init__(self, answer):
        self.name = "answering_stub"
        self.answer = answer

    def is_compatible(self, verbose=False):
        return self.answer


def test_probe_is_compatible_does_not_propagate_a_failed_probe():
    # Every op is probed the first time deepspeed.ops.__compatible_ops__ is read, whether or not it will
    # ever be built, so one op builder that cannot answer must not break reading it.
    assert probe_is_compatible(_RaisingBuilder()) is False


@pytest.mark.parametrize("answer", [True, False])
def test_probe_is_compatible_passes_through_a_successful_probe(answer):
    assert probe_is_compatible(_AnsweringBuilder(answer)) is answer


_AT_HWCAP = 16  # <linux/auxvec.h>
_HWCAP_ISA_V = 1 << (ord("V") - ord("A"))  # COMPAT_HWCAP_ISA_V in <asm/hwcap.h>


def _kernel_hwcap_has_vector():
    # Read AT_HWCAP straight from the auxiliary vector, the same source getauxval(3)
    # serves; its V bit is the kernel's own userspace availability interface for the
    # vector extension, and the independent reference for the probe under test.
    with open("/proc/self/auxv", "rb") as f:
        auxv = f.read()
    for offset in range(0, len(auxv) - 15, 16):
        a_type, value = struct.unpack("=QQ", auxv[offset:offset + 16])
        if a_type == _AT_HWCAP:
            return value & _HWCAP_ISA_V != 0
    return False


@pytest.mark.skipif(platform.machine() != "riscv64", reason="requires a riscv64 host")
def test_riscv_has_vector_agrees_with_kernel_hwcap():
    # cpu_arch()/simd_width() gate -march=rv64gcv and __RVV__ on this probe. A wrong
    # True on draft-V hardware builds V 1.0 encodings the CPU cannot execute. On a
    # real host the probe must agree with the kernel's HWCAP V bit: the only input
    # where the probe overrides HWCAP is an explicitly versioned isa token, and a
    # kernel that accepted such a token also sets the HWCAP bit, so the two cannot
    # disagree on real hardware.
    assert make_builder()._riscv_has_vector() is _kernel_hwcap_has_vector()
