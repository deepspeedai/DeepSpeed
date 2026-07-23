# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import FunctionType

import torch

from deepspeed.compile import z3_eager_fallback
from deepspeed.compile.z3_eager_fallback import DeepCompileZ3EagerFallback
from deepspeed.runtime.zero import parameter_offload
from deepspeed.runtime.zero.parameter_offload import ZeROOrderedDict
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def _zero_module_with_param():
    module = torch.nn.Module()
    param = torch.nn.Parameter(torch.empty(0))
    param.ds_id = 7
    param.ds_status = ZeroParamStatus.NOT_AVAILABLE

    params = ZeROOrderedDict(parent_module=module)
    params["weight"] = param
    params._in_forward = True
    module._parameters = params
    return module, param


def test_deepcompile_fallback_suppresses_guard_time_gather(monkeypatch):
    module, param = _zero_module_with_param()
    fallback = DeepCompileZ3EagerFallback(engine=None)
    gather_calls = []

    param.all_gather = lambda: gather_calls.append(param.ds_id)
    monkeypatch.setattr(z3_eager_fallback, "_ACTIVE_FALLBACK", fallback)
    monkeypatch.setattr(z3_eager_fallback, "is_dynamo_guard_evaluation", lambda: True)

    assert module._parameters["weight"] is param
    assert gather_calls == []
    assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE


def test_deepcompile_fallback_still_gathers_outside_guard(monkeypatch):
    module, param = _zero_module_with_param()
    fallback = DeepCompileZ3EagerFallback(engine=None)
    gather_calls = []

    def all_gather():
        gather_calls.append(param.ds_id)
        param.ds_status = ZeroParamStatus.AVAILABLE

    param.all_gather = all_gather
    monkeypatch.setattr(z3_eager_fallback, "_ACTIVE_FALLBACK", fallback)
    monkeypatch.setattr(z3_eager_fallback, "is_dynamo_guard_evaluation", lambda: False)
    monkeypatch.setattr(parameter_offload, "print_rank_0", lambda *args, **kwargs: None)

    assert module._parameters["weight"] is param
    assert gather_calls == [7]
    assert fallback.stats()["last_gathered_param_ids"] == [7]


def test_dynamo_guard_detection_is_false_outside_guard_stack():
    assert not z3_eager_fallback.is_dynamo_guard_evaluation()


def test_dynamo_guard_detection_is_true_inside_guard_module_stack():
    guard_fn = FunctionType(
        compile("result = is_dynamo_guard_evaluation()", "<guard-test>", "exec"),
        {
            "__name__": "torch._dynamo.guards",
            "is_dynamo_guard_evaluation": z3_eager_fallback.is_dynamo_guard_evaluation,
        },
    )
    guard_fn()
    assert guard_fn.__globals__["result"] is True
