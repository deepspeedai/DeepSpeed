# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.runtime.zero.partition_parameters import zero_wrapper_for_fp_tensor_constructor


class TestZeroWrapperPreservesExplicitDtype:
    """Ensure zero_wrapper_for_fp_tensor_constructor respects an explicit dtype kwarg."""

    @pytest.fixture
    def wrapped_arange(self):
        return zero_wrapper_for_fp_tensor_constructor(torch.arange, torch.bfloat16, None)

    def test_no_dtype_kwarg_casts_to_target(self, wrapped_arange):
        t = wrapped_arange(0, 4)
        assert t.dtype == torch.bfloat16

    def test_explicit_float32_preserved(self, wrapped_arange):
        t = wrapped_arange(0, 4, dtype=torch.float32)
        assert t.dtype == torch.float32

    def test_explicit_float64_preserved(self, wrapped_arange):
        t = wrapped_arange(0, 4, dtype=torch.float64)
        assert t.dtype == torch.float64

    def test_explicit_none_casts_to_target(self, wrapped_arange):
        t = wrapped_arange(0, 4, dtype=None)
        assert t.dtype == torch.bfloat16

    def test_integer_dtype_not_cast(self, wrapped_arange):
        t = wrapped_arange(0, 4, dtype=torch.int64)
        assert t.dtype == torch.int64

    @pytest.mark.parametrize("fn", [torch.empty, torch.zeros, torch.ones, torch.randn])
    def test_explicit_float32_on_creation_fns(self, fn):
        wrapped = zero_wrapper_for_fp_tensor_constructor(fn, torch.bfloat16, None)
        t = wrapped(4, dtype=torch.float32)
        assert t.dtype == torch.float32

    @pytest.mark.parametrize("fn", [torch.empty, torch.zeros, torch.ones, torch.randn])
    def test_no_dtype_on_creation_fns_casts(self, fn):
        wrapped = zero_wrapper_for_fp_tensor_constructor(fn, torch.bfloat16, None)
        t = wrapped(4)
        assert t.dtype == torch.bfloat16
