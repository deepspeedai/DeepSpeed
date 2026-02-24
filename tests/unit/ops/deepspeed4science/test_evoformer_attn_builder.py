# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from unittest import mock

from deepspeed.ops.op_builder.evoformer_attn import EvoformerAttnBuilder


def _cc_set(ccs):
    return {f"{major}.{minor}" for major, minor in ccs}


def test_normalize_gpu_arch_accepts_80_8dot0_sm80(monkeypatch):
    for value in ("80", "8.0", "sm80"):
        monkeypatch.setenv("DS_EVOFORMER_GPU_ARCH", value)
        builder = EvoformerAttnBuilder()
        assert builder._parse_gpu_arch(builder.gpu_arch) == 80


def test_normalize_gpu_arch_invalid_value_warns_and_falls_back(monkeypatch):
    monkeypatch.setenv("DS_EVOFORMER_GPU_ARCH", "invalid")
    builder = EvoformerAttnBuilder()
    warnings = []
    builder.warning = warnings.append

    with mock.patch.object(builder, "_detect_local_gpu_cc", return_value=90):
        raw_cc, floor_cc = builder._resolve_gpu_arch()

    assert (raw_cc, floor_cc) == (90, 80)
    assert any("Invalid DS_EVOFORMER_GPU_ARCH" in msg for msg in warnings)


def test_nvcc_args_uses_normalized_gpu_arch_macro(monkeypatch):
    monkeypatch.setenv("DS_EVOFORMER_GPU_ARCH", "8.0")
    builder = EvoformerAttnBuilder()

    with mock.patch("deepspeed.ops.op_builder.evoformer_attn.CUDAOpBuilder.nvcc_args", return_value=["-O3"]):
        args = builder.nvcc_args()

    assert "-DGPU_ARCH=80" in args
    assert "-DGPU_ARCH=8.0" not in args


def test_effective_floor_maps_90_to_80():
    assert EvoformerAttnBuilder._effective_floor_cc(90) == 80


def test_filter_ccs_prunes_below_70_even_without_floor(monkeypatch):
    monkeypatch.delenv("DS_EVOFORMER_GPU_ARCH", raising=False)
    builder = EvoformerAttnBuilder()

    with mock.patch.object(builder, "_resolve_gpu_arch", return_value=(None, None)):
        filtered = builder.filter_ccs(["6.1", "7.0", "8.0"])

    assert filtered == [["7", "0"], ["8", "0"]]


def test_filter_ccs_gpu_arch80_prunes_70(monkeypatch):
    monkeypatch.setenv("DS_EVOFORMER_GPU_ARCH", "80")
    builder = EvoformerAttnBuilder()

    with mock.patch.object(builder, "_resolve_gpu_arch", return_value=(80, 80)):
        filtered = builder.filter_ccs(["8.0", "7.0", "9.0"])

    assert filtered == [["8", "0"], ["9", "0"]]


def test_filter_ccs_preserves_ptx_suffix(monkeypatch):
    monkeypatch.setenv("DS_EVOFORMER_GPU_ARCH", "80")
    builder = EvoformerAttnBuilder()

    with mock.patch.object(builder, "_resolve_gpu_arch", return_value=(80, 80)):
        filtered = builder.filter_ccs(["9.0+PTX", "8.0", "7.5"])

    assert filtered == [["9", "0+PTX"], ["8", "0"]]


def test_filter_ccs_order_independent_result_set(monkeypatch):
    monkeypatch.setenv("DS_EVOFORMER_GPU_ARCH", "80")
    builder = EvoformerAttnBuilder()

    with mock.patch.object(builder, "_resolve_gpu_arch", return_value=(80, 80)):
        left = builder.filter_ccs(["8.0", "7.0", "9.0"])
        right = builder.filter_ccs(["7.0", "9.0", "8.0"])

    assert _cc_set(left) == _cc_set(right) == {"8.0", "9.0"}


def test_filter_ccs_empty_after_filter(monkeypatch):
    monkeypatch.setenv("DS_EVOFORMER_GPU_ARCH", "80")
    builder = EvoformerAttnBuilder()

    with mock.patch.object(builder, "_resolve_gpu_arch", return_value=(80, 80)):
        filtered = builder.filter_ccs(["6.0", "7.0", "7.5"])

    assert filtered == []


def test_filter_ccs_handles_whitespace_and_minor_variants(monkeypatch):
    monkeypatch.setenv("DS_EVOFORMER_GPU_ARCH", "80")
    builder = EvoformerAttnBuilder()

    with mock.patch.object(builder, "_resolve_gpu_arch", return_value=(80, 80)):
        filtered = builder.filter_ccs([" 8.6 ", " 7.5 ", " 9.0+PTX "])

    assert filtered == [["8", "6"], ["9", "0+PTX"]]
