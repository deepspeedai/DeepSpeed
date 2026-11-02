# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.runtime.zero.partition_parameters import (
    zero_wrapper_for_fp_tensor_constructor,
    get_new_tensor_fn_for_dtype,
)


class TestZeroWrapperDtypeRespected:
    """zero_wrapper_for_fp_tensor_constructor must not override an explicit dtype (#8948)."""

    @pytest.mark.parametrize("fn", [torch.empty, torch.zeros, torch.ones, torch.randn])
    def test_explicit_float32_kept(self, fn):
        wrapped = zero_wrapper_for_fp_tensor_constructor(fn, torch.bfloat16, torch.device("cpu"))
        tensor = wrapped(4, dtype=torch.float32)
        assert tensor.dtype == torch.float32

    @pytest.mark.parametrize("fn", [torch.empty, torch.zeros, torch.ones, torch.randn])
    def test_no_dtype_casts_to_target(self, fn):
        wrapped = zero_wrapper_for_fp_tensor_constructor(fn, torch.bfloat16, torch.device("cpu"))
        tensor = wrapped(4)
        assert tensor.dtype == torch.bfloat16

    def test_arange_explicit_float32(self):
        wrapped = zero_wrapper_for_fp_tensor_constructor(torch.arange, torch.bfloat16, torch.device("cpu"))
        tensor = wrapped(0, 8, 2, dtype=torch.float32)
        assert tensor.dtype == torch.float32

    def test_arange_no_dtype_int_unchanged(self):
        wrapped = zero_wrapper_for_fp_tensor_constructor(torch.arange, torch.bfloat16, torch.device("cpu"))
        tensor = wrapped(0, 8, 2)
        assert tensor.dtype == torch.int64

    def test_full_explicit_float64(self):
        wrapped = zero_wrapper_for_fp_tensor_constructor(torch.full, torch.bfloat16, torch.device("cpu"))
        tensor = wrapped((4,), 1.0, dtype=torch.float64)
        assert tensor.dtype == torch.float64

    def test_full_no_dtype_casts(self):
        wrapped = zero_wrapper_for_fp_tensor_constructor(torch.full, torch.bfloat16, torch.device("cpu"))
        tensor = wrapped((4,), 1.0)
        assert tensor.dtype == torch.bfloat16

    def test_rope_inv_freq_pattern(self):
        """The exact pattern from transformers RoPE: arange(dtype=float) then fp arithmetic."""
        wrapped_arange = zero_wrapper_for_fp_tensor_constructor(torch.arange, torch.bfloat16, torch.device("cpu"))
        t = wrapped_arange(0, 128, 2, dtype=torch.float)
        inv_freq = 1.0 / (10000.0 ** (t / 128))
        assert inv_freq.dtype == torch.float32


class TestNewTensorDtypeRespected:
    """get_new_tensor_fn_for_dtype must not override an explicit dtype."""

    def test_explicit_float32(self):
        new_fn = get_new_tensor_fn_for_dtype(torch.bfloat16, torch.device("cpu"))
        tensor = new_fn(None, 4, dtype=torch.float32)
        assert tensor.dtype == torch.float32

    def test_no_dtype_casts(self):
        new_fn = get_new_tensor_fn_for_dtype(torch.bfloat16, torch.device("cpu"))
        tensor = new_fn(None, 4)
        assert tensor.dtype == torch.bfloat16
