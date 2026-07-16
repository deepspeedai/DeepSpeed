# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from types import FunctionType

from deepspeed.compile import z3_eager_fallback
from deepspeed.compile.z3_eager_fallback import DeepCompileZ3EagerFallback, deepcompile_z3_forward_context
from deepspeed.runtime.zero import parameter_offload
from deepspeed.runtime.zero.partition_parameters import GatheredParameters
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
    assert fallback.stats()["last_guard_suppressed_param_ids"] == [7]


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


def test_deepcompile_fallback_releases_leftover_gathered_params_before_forward():

    class FakeEngine:
        module = torch.nn.Linear(1, 1)

    param = next(FakeEngine.module.parameters())
    param.ds_id = 11
    param.ds_status = ZeroParamStatus.AVAILABLE
    param.ds_persist = False
    partition_calls = []

    def partition():
        partition_calls.append(param.ds_id)
        param.ds_status = ZeroParamStatus.NOT_AVAILABLE

    param.partition = partition
    fallback = DeepCompileZ3EagerFallback(FakeEngine())
    fallback.record_gathered_param(param)

    fallback.release_available_params_for_next_forward()

    assert partition_calls == [11]
    assert fallback.stats()["last_pre_forward_released_param_ids"] == [11]


def test_user_adopted_param_diagnostic_deduplicates_repeated_transfers():
    module, param = _zero_module_with_param()
    fallback = DeepCompileZ3EagerFallback(engine=None)

    for _ in range(3):
        fallback.record_gathered_param(param)
        fallback.transfer_gathered_param_to_user(param)

    assert fallback.stats()["last_user_adopted_param_ids"] == [7]


def test_deepcompile_forward_preserves_fallback_param_adopted_by_user_gathered_context():
    module, param = _zero_module_with_param()
    param.ds_persist = False
    partition_calls = []

    def all_gather(param_list):
        assert param_list == [param]
        param.ds_status = ZeroParamStatus.AVAILABLE

    def partition(param_list=None, has_been_updated=False):
        if param_list is not None:
            assert param_list == [param]
            assert has_been_updated is False
        partition_calls.append(param.ds_id)
        param.ds_status = ZeroParamStatus.NOT_AVAILABLE

    param.all_gather = all_gather
    param.partition = partition

    class FakeEngine:

        def __init__(self):
            self.module = module
            self._deepcompile_z3_eager_fallback = DeepCompileZ3EagerFallback(self)

        def is_deepcompile_active(self):
            return True

        def zero_optimization_partition_weights(self):
            return True

    engine = FakeEngine()
    param.ds_status = ZeroParamStatus.AVAILABLE
    engine._deepcompile_z3_eager_fallback.record_gathered_param(param)
    assert engine._deepcompile_z3_eager_fallback.stats()["tracked_param_ids"] == [7]

    with GatheredParameters([param]):
        assert param.ds_status == ZeroParamStatus.AVAILABLE
        assert engine._deepcompile_z3_eager_fallback.stats()["tracked_param_ids"] == []
        assert engine._deepcompile_z3_eager_fallback.stats()["last_user_adopted_param_ids"] == [7]
        with deepcompile_z3_forward_context(engine):
            assert param.ds_status == ZeroParamStatus.AVAILABLE
        assert param.ds_status == ZeroParamStatus.AVAILABLE

    assert partition_calls == [7]
    assert engine._deepcompile_z3_eager_fallback.stats()["last_user_adopted_param_ids"] == [7]
