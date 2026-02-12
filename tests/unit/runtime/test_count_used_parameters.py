# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
Tests for count_used_parameters_in_backward and its fallback behaviour
when PyTorch internal APIs are unavailable (issue #7756).
"""

import pytest
import torch
from unittest.mock import patch

from deepspeed.runtime.utils import (
    check_internal_apis_for_count_used_parameters,
    count_used_parameters_in_backward,
)


# ---------------------------------------------------------------------------
# Helper: build a small parameter list
# ---------------------------------------------------------------------------

def _make_params(n=4, requires_grad=True):
    """Return a list of n Parameters on CPU."""
    return [torch.nn.Parameter(torch.randn(3, 3), requires_grad=requires_grad) for _ in range(n)]


# ---------------------------------------------------------------------------
# Tests for check_internal_apis_for_count_used_parameters
# ---------------------------------------------------------------------------

class TestCheckInternalApis:
    """Verify the availability probe returns correct results."""

    def test_returns_bool(self):
        result = check_internal_apis_for_count_used_parameters()
        assert isinstance(result, bool)

    def test_false_when_get_grad_fn_missing(self):
        with patch.object(torch.autograd.graph, '_get_grad_fn_or_grad_acc', create=False, new=None):
            # Remove the attribute entirely
            saved = getattr(torch.autograd.graph, '_get_grad_fn_or_grad_acc', None)
            try:
                if hasattr(torch.autograd.graph, '_get_grad_fn_or_grad_acc'):
                    delattr(torch.autograd.graph, '_get_grad_fn_or_grad_acc')
                assert check_internal_apis_for_count_used_parameters() is False
            finally:
                if saved is not None:
                    torch.autograd.graph._get_grad_fn_or_grad_acc = saved

    def test_false_when_current_graph_task_id_missing(self):
        saved = getattr(torch._C, '_current_graph_task_id', None)
        try:
            if hasattr(torch._C, '_current_graph_task_id'):
                delattr(torch._C, '_current_graph_task_id')
            assert check_internal_apis_for_count_used_parameters() is False
        finally:
            if saved is not None:
                torch._C._current_graph_task_id = saved

    def test_false_when_will_engine_execute_node_missing(self):
        saved = getattr(torch._C, '_will_engine_execute_node', None)
        try:
            if hasattr(torch._C, '_will_engine_execute_node'):
                delattr(torch._C, '_will_engine_execute_node')
            assert check_internal_apis_for_count_used_parameters() is False
        finally:
            if saved is not None:
                torch._C._will_engine_execute_node = saved


# ---------------------------------------------------------------------------
# Tests for the fallback path of count_used_parameters_in_backward
# ---------------------------------------------------------------------------

class TestCountUsedParametersFallback:
    """When internal APIs are missing, the function must fall back to counting
    all parameters that require gradients instead of crashing."""

    def test_fallback_returns_total_grad_count(self):
        """With APIs unavailable, should return count of grad-requiring params."""
        params = _make_params(5, requires_grad=True)
        with patch('deepspeed.runtime.utils.check_internal_apis_for_count_used_parameters', return_value=False):
            result = count_used_parameters_in_backward(params)
        assert result == 5

    def test_fallback_excludes_no_grad_params(self):
        """Params with requires_grad=False should not be counted in fallback."""
        params = _make_params(3, requires_grad=True) + _make_params(2, requires_grad=False)
        with patch('deepspeed.runtime.utils.check_internal_apis_for_count_used_parameters', return_value=False):
            result = count_used_parameters_in_backward(params)
        assert result == 3

    def test_fallback_empty_list(self):
        """Empty parameter list should return 0."""
        with patch('deepspeed.runtime.utils.check_internal_apis_for_count_used_parameters', return_value=False):
            result = count_used_parameters_in_backward([])
        assert result == 0

    def test_fallback_all_no_grad(self):
        """All params with requires_grad=False should return 0."""
        params = _make_params(4, requires_grad=False)
        with patch('deepspeed.runtime.utils.check_internal_apis_for_count_used_parameters', return_value=False):
            result = count_used_parameters_in_backward(params)
        assert result == 0

    def test_fallback_mixed_tensors_and_non_tensors(self):
        """Non-tensor items in the parameter list should be skipped."""
        params = _make_params(2, requires_grad=True)
        mixed = params + [None, "not_a_tensor", 42]
        with patch('deepspeed.runtime.utils.check_internal_apis_for_count_used_parameters', return_value=False):
            result = count_used_parameters_in_backward(mixed)
        assert result == 2

    def test_fallback_does_not_raise(self):
        """The old code raised AssertionError; the fix must NOT raise."""
        params = _make_params(3, requires_grad=True)
        with patch('deepspeed.runtime.utils.check_internal_apis_for_count_used_parameters', return_value=False):
            # This should complete without any assertion or exception
            result = count_used_parameters_in_backward(params)
            assert isinstance(result, int)
            assert result >= 0

    def test_fallback_single_param(self):
        """Single parameter with grad should return 1."""
        params = _make_params(1, requires_grad=True)
        with patch('deepspeed.runtime.utils.check_internal_apis_for_count_used_parameters', return_value=False):
            result = count_used_parameters_in_backward(params)
        assert result == 1

    def test_fallback_large_param_list(self):
        """Ensure fallback scales correctly with many parameters."""
        params = _make_params(100, requires_grad=True)
        with patch('deepspeed.runtime.utils.check_internal_apis_for_count_used_parameters', return_value=False):
            result = count_used_parameters_in_backward(params)
        assert result == 100


# ---------------------------------------------------------------------------
# Tests for the native path (when APIs are available)
# ---------------------------------------------------------------------------

class TestCountUsedParametersNative:
    """When internal APIs are available, verify the native path works correctly."""

    @pytest.mark.skipif(
        not check_internal_apis_for_count_used_parameters(),
        reason="PyTorch internal APIs not available (likely PyTorch < 2.3)"
    )
    def test_native_path_during_backward(self):
        """Native path should work correctly when called inside a backward hook."""
        model = torch.nn.Linear(4, 2)
        params = list(model.parameters())
        results = []

        def hook_fn(grad):
            count = count_used_parameters_in_backward(params)
            results.append(count)
            return grad

        x = torch.randn(1, 4)
        out = model(x)
        loss = out.sum()

        # Register hook on one of the parameter's grad_fn
        params[0].register_hook(hook_fn)
        loss.backward()

        assert len(results) == 1
        assert results[0] > 0  # At least some params should participate

    @pytest.mark.skipif(
        not check_internal_apis_for_count_used_parameters(),
        reason="PyTorch internal APIs not available (likely PyTorch < 2.3)"
    )
    def test_native_path_raises_outside_backward(self):
        """Native path should raise RuntimeError when not inside backward."""
        params = _make_params(3, requires_grad=True)
        with pytest.raises(RuntimeError, match="must be called during backward execution"):
            count_used_parameters_in_backward(params)

    @pytest.mark.skipif(
        not check_internal_apis_for_count_used_parameters(),
        reason="PyTorch internal APIs not available (likely PyTorch < 2.3)"
    )
    def test_native_path_empty_list(self):
        """Native path should return 0 for empty list during backward."""
        model = torch.nn.Linear(4, 2)
        results = []

        def hook_fn(grad):
            count = count_used_parameters_in_backward([])
            results.append(count)
            return grad

        x = torch.randn(1, 4)
        out = model(x)
        loss = out.sum()
        list(model.parameters())[0].register_hook(hook_fn)
        loss.backward()

        assert len(results) == 1
        assert results[0] == 0
