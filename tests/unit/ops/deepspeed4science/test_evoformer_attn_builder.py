# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from pathlib import Path
from unittest.mock import patch

import pytest

from deepspeed.ops.op_builder.builder import CUDAOpBuilder
# Import the concrete builder class instead of the accelerator-dispatched alias.
from deepspeed.ops.op_builder.evoformer_attn import EvoformerAttnBuilder


def make_cutlass_checkout(path):
    include_dir = path / "include" / "cutlass"
    include_dir.mkdir(parents=True)
    (include_dir / "cutlass.h").write_text("// cutlass marker\n")
    util_dir = path / "tools" / "util" / "include"
    util_dir.mkdir(parents=True)
    return path


def test_filter_ccs_removes_below_70_and_keeps_ptx_suffix():
    builder = EvoformerAttnBuilder()
    result = builder.filter_ccs(["6.0", "6.1", "7.0", "8.0+PTX"])

    majors = [int(cc[0]) for cc in result]
    assert 6 not in majors
    assert 7 in majors
    assert 8 in majors

    ptx_entries = [cc for cc in result if cc[1].endswith("+PTX")]
    assert len(ptx_entries) == 1
    assert ptx_entries[0] == ["8", "0+PTX"]


def test_nvcc_args_deprecates_env_and_omits_gpu_arch_define():
    builder = EvoformerAttnBuilder()
    with patch.dict("os.environ", {"DS_EVOFORMER_GPU_ARCH": "80"}, clear=False):
        with patch.object(builder, "warning") as warn:
            with patch.object(CUDAOpBuilder, "nvcc_args", return_value=["-O3", "-lineinfo"]):
                args = builder.nvcc_args()

    warning_messages = [call.args[0] for call in warn.call_args_list if call.args]
    assert any("DS_EVOFORMER_GPU_ARCH is deprecated and ignored" in msg for msg in warning_messages)
    assert all("-DGPU_ARCH=" not in arg for arg in args)


def test_no_cuda_arch_in_checkarch():
    header = Path(__file__).resolve().parents[4] / "csrc/deepspeed4science/evoformer_attn/gemm_kernel_utils.h"
    text = header.read_text()
    start = text.index("struct CheckArch")
    end = text.index("};", start) + 2
    block = text[start:end]
    assert "__CUDA_ARCH__" not in block


def test_include_paths_uses_cutlass_path_env(tmp_path):
    cutlass_path = make_cutlass_checkout(tmp_path / "cutlass")

    with patch.dict("os.environ", {"CUTLASS_PATH": str(cutlass_path)}, clear=False):
        builder = EvoformerAttnBuilder()

    assert builder.include_paths() == [
        str(cutlass_path / "include"),
        str(cutlass_path / "tools" / "util" / "include"),
    ]


def test_include_paths_finds_python_package_candidate_without_env(tmp_path):
    cutlass_path = make_cutlass_checkout(tmp_path / "python_package_cutlass")

    with patch.dict("os.environ", {}, clear=True):
        builder = EvoformerAttnBuilder()

    with patch.object(EvoformerAttnBuilder, "_python_package_cutlass_paths", return_value=[cutlass_path]):
        assert builder.include_paths()[0] == str(cutlass_path / "include")


def test_include_paths_finds_cutlass_from_cmake_prefix_path(tmp_path):
    cutlass_path = make_cutlass_checkout(tmp_path / "prefix")

    with patch.dict("os.environ", {"CMAKE_PREFIX_PATH": str(cutlass_path)}, clear=True):
        builder = EvoformerAttnBuilder()
        with patch.object(EvoformerAttnBuilder, "_python_package_cutlass_paths", return_value=[]):
            assert builder.include_paths()[0] == str(cutlass_path / "include")


def test_include_paths_finds_cutlass_from_compiler_include_path(tmp_path):
    cutlass_path = make_cutlass_checkout(tmp_path / "prefix")

    with patch.dict("os.environ", {"CPATH": str(cutlass_path / "include")}, clear=True):
        builder = EvoformerAttnBuilder()
        with patch.object(EvoformerAttnBuilder, "_python_package_cutlass_paths", return_value=[]):
            assert builder.include_paths()[0] == str(cutlass_path / "include")


def test_include_paths_accepts_cutlass_include_dir_directly(tmp_path):
    cutlass_path = make_cutlass_checkout(tmp_path / "cutlass")

    with patch.dict("os.environ", {"CUTLASS_PATH": str(cutlass_path / "include")}, clear=False):
        builder = EvoformerAttnBuilder()

    assert builder.include_paths() == [
        str(cutlass_path / "include"),
        str(cutlass_path / "tools" / "util" / "include"),
    ]


def test_include_paths_reports_missing_cutlass(tmp_path):
    with patch.dict("os.environ", {}, clear=True):
        builder = EvoformerAttnBuilder()

    with patch.object(builder, "_candidate_cutlass_paths", return_value=[tmp_path / "missing"]):
        with pytest.raises(RuntimeError, match="Unable to locate CUTLASS"):
            builder.include_paths()
