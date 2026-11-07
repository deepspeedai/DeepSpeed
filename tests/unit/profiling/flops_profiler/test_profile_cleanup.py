# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed.profiling.flops_profiler.profiler as profiler_module
from deepspeed.profiling.flops_profiler import FlopsProfiler, get_model_profile


class ProfileModel(torch.nn.Module):

    def __init__(self, failure=None):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2, bias=False)
        self.failure = failure

    def forward(self, inputs):
        result = self.linear(inputs)
        if self.failure is not None:
            raise self.failure
        return result

    def generate(self, inputs):
        return self(inputs)


def _operations():
    return (torch.nn.functional.linear, torch.Tensor.__matmul__, torch.bmm)


def _hooks(model):
    return [(dict(module._forward_pre_hooks), dict(module._forward_hooks)) for module in model.modules()]


@pytest.mark.sequential
@pytest.mark.parametrize("stage, use_kwargs", [("forward", False), ("forward", True), ("generate", False),
                                               ("generate", True), ("report", False)])
def test_model_profile_cleans_up_after_exception(stage, use_kwargs, tmp_path):
    failure = RuntimeError("measured inference failed")
    model = ProfileModel(None if stage == "report" else failure)
    inputs = torch.ones(1, 4)
    args, kwargs = ([], {"inputs": inputs}) if use_kwargs else ([inputs], {})
    user_handles = (model.register_forward_pre_hook(lambda module, args: None),
                    model.linear.register_forward_hook(lambda module, args, result: None))
    original_operations = _operations()
    original_hooks = _hooks(model)
    original_depths = (len(profiler_module.module_flop_count), len(profiler_module.module_mac_count))

    try:
        expected_error = IsADirectoryError if stage == "report" else RuntimeError
        with pytest.raises(expected_error) as caught:
            get_model_profile(model,
                              args=args,
                              kwargs=kwargs,
                              mode="generate" if stage == "generate" else "forward",
                              warm_up=0,
                              print_profile=stage == "report",
                              output_file=str(tmp_path) if stage == "report" else None,
                              as_string=False)

        if stage == "report":
            assert caught.value.filename == str(tmp_path)
        else:
            assert caught.value is failure

        profiler_attrs = ("__flops__", "__macs__", "__params__", "__start_time__", "__duration__")
        restored_state = (
            _operations() == original_operations,
            _hooks(model) == original_hooks,
            not any(hasattr(module, attr) for module in model.modules() for attr in profiler_attrs),
            (len(profiler_module.module_flop_count), len(profiler_module.module_mac_count)) == original_depths,
        )
        assert restored_state == (True, True, True, True)

        model.failure = None
        assert get_model_profile(model, args=[inputs], warm_up=0, print_profile=False, as_string=False) == (16, 8, 8)
        assert _operations() == original_operations
        assert _hooks(model) == original_hooks
    finally:
        # Keep a pre-fix failure from contaminating the rest of the test process.
        if hasattr(model, "__flops__"):
            cleanup = FlopsProfiler(model)
            cleanup.started = True
            cleanup.func_patched = True
            cleanup.end_profile()
        del profiler_module.module_flop_count[original_depths[0]:]
        del profiler_module.module_mac_count[original_depths[1]:]
        for handle in user_handles:
            handle.remove()


@pytest.mark.sequential
@pytest.mark.parametrize("mode", ["forward", "generate"])
@pytest.mark.parametrize("use_kwargs", [False, True])
def test_model_profile_successful_counts_unchanged(mode, use_kwargs):
    model = ProfileModel()
    inputs = torch.ones(1, 4)
    args, kwargs = ([], {"inputs": inputs}) if use_kwargs else ([inputs], {})
    original_operations = _operations()

    for _ in range(3):
        assert get_model_profile(model,
                                 args=args,
                                 kwargs=kwargs,
                                 mode=mode,
                                 warm_up=1,
                                 print_profile=False,
                                 as_string=False) == (16, 8, 8)
        assert _operations() == original_operations


@pytest.mark.sequential
@pytest.mark.parametrize("mode", ["forward", "generate"])
def test_model_profile_string_return_unchanged(mode):
    result = get_model_profile(ProfileModel(),
                               args=[torch.ones(1, 4)],
                               mode=mode,
                               warm_up=0,
                               print_profile=False,
                               as_string=True)
    assert len(result) == 3
    assert all(isinstance(value, str) and value for value in result)
