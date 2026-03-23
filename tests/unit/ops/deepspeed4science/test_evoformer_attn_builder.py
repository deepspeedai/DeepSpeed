# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from pathlib import Path
from unittest.mock import patch

from deepspeed.ops.op_builder.builder import CUDAOpBuilder
# Import the concrete builder class instead of the accelerator-dispatched alias.
from deepspeed.ops.op_builder.evoformer_attn import EvoformerAttnBuilder


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
