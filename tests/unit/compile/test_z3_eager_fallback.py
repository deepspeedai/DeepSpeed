# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import inspect
import weakref
from types import FunctionType

import pytest
import torch

from deepspeed.compile import z3_eager_fallback
from deepspeed.compile.z3_eager_fallback import DeepCompileZ3EagerFallback, deepcompile_z3_forward_context
from deepspeed.runtime.zero import parameter_offload
from deepspeed.runtime.zero.parameter_offload import ZeROOrderedDict
from deepspeed.runtime.zero.partition_parameters import GatheredParameters, ZeroParamStatus
from deepspeed.runtime.utils import register_output_backward_hooks


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


def test_real_guard_builder_source_resolution_suppresses_gather(monkeypatch):
    from torch._dynamo.guards import GuardBuilder
    try:
        from torch._dynamo.source import AttrSource, LocalSource
    except ImportError:
        from torch._guards import AttrSource, LocalSource

    module, param = _zero_module_with_param()
    fallback = DeepCompileZ3EagerFallback(engine=None)
    gather_calls = []

    def all_gather():
        gather_calls.append(param.ds_id)
        param.ds_status = ZeroParamStatus.AVAILABLE

    param.all_gather = all_gather
    monkeypatch.setattr(z3_eager_fallback, "_ACTIVE_FALLBACK", fallback)
    monkeypatch.setattr(parameter_offload, "print_rank_0", lambda *args, **kwargs: None)
    before = (param.ds_status, param.data.data_ptr(), param.numel())

    builder = object.__new__(GuardBuilder)
    builder.scope = {"L": {"module": module}, "G": {}}
    builder.src_get_value_cache = weakref.WeakKeyDictionary()
    builder.source_get_cache = {}
    builder.save_guards = False
    source = AttrSource(LocalSource("module"), "weight")
    get_parameter = list(inspect.signature(GuardBuilder.get).parameters)[1]
    resolved = builder.get(source.name() if get_parameter == "name" else source)

    assert resolved is param
    assert gather_calls == []
    assert (param.ds_status, param.data.data_ptr(), param.numel()) == before
    assert fallback.stats()["last_guard_suppressed_param_ids"] == [7]

    assert module._parameters["weight"] is param
    assert gather_calls == [7]
    assert param.ds_status == ZeroParamStatus.AVAILABLE
    assert fallback.stats()["tracked_param_ids"] == [7]


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


def test_consecutive_deepcompile_inference_forwards_release_unowned_available_params():
    module, param = _zero_module_with_param()
    param.ds_persist = False
    partition_calls = []

    def partition():
        partition_calls.append(param.ds_id)
        param.ds_status = ZeroParamStatus.NOT_AVAILABLE

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
    for _ in range(2):
        param.ds_status = ZeroParamStatus.AVAILABLE
        with deepcompile_z3_forward_context(engine):
            assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE

    assert partition_calls == [7, 7]


def test_two_disjoint_fallback_forwards_remain_gathered_until_combined_backward(monkeypatch):
    module = torch.nn.Module()
    params = ZeROOrderedDict(parent_module=module)
    partition_calls = []

    for name, ds_id in (("first", 1), ("second", 2)):
        param = torch.nn.Parameter(torch.empty(0))
        param.ds_id = ds_id
        param.ds_status = ZeroParamStatus.NOT_AVAILABLE
        param.ds_persist = False
        gathered_data = torch.full((2, 3), float(ds_id))

        def all_gather(*, _param=param, _data=gathered_data):
            _param.data = _data.clone()
            _param.ds_status = ZeroParamStatus.AVAILABLE

        def partition(*, _param=param):
            partition_calls.append(int(_param.ds_id))
            _param.data = torch.empty(0)
            _param.ds_status = ZeroParamStatus.NOT_AVAILABLE

        param.all_gather = all_gather
        param.partition = partition
        params[name] = param

    module._parameters = params

    class FakeEngine:

        def __init__(self):
            self.module = module
            self._deepcompile_z3_eager_fallback = DeepCompileZ3EagerFallback(self)

        def is_deepcompile_active(self):
            return True

        def zero_optimization_partition_weights(self):
            return True

        def forward(self, name, input_tensor):
            with deepcompile_z3_forward_context(self) as fallback:
                output = torch.nn.functional.linear(input_tensor, self.module._parameters[name]).sum()

            graph_id = None

            def record_backward_start():
                fallback.record_backward_start(graph_id)

            hook_manager = register_output_backward_hooks(output, preprocess_once_fn=record_backward_start)
            if hook_manager.hook_handles:
                graph_id = fallback.record_forward_graph()
            return output

    monkeypatch.setattr(z3_eager_fallback, "is_dynamo_guard_evaluation", lambda: False)
    monkeypatch.setattr(parameter_offload, "print_rank_0", lambda *args, **kwargs: None)

    engine = FakeEngine()
    first_input = torch.ones((1, 3), requires_grad=True)
    second_input = torch.ones((1, 3), requires_grad=True)
    first_loss = engine.forward("first", first_input)
    second_loss = engine.forward("second", second_input)

    fallback = engine._deepcompile_z3_eager_fallback
    assert fallback.stats()["outstanding_forward_graph_ids"] == [0, 1]
    assert partition_calls == []
    assert all(param.ds_status == ZeroParamStatus.AVAILABLE for param in module.parameters())

    (first_loss + second_loss).backward()
    fallback.complete_backward()

    assert first_input.grad is not None
    assert second_input.grad is not None
    assert fallback.stats()["outstanding_forward_graph_ids"] == []
    assert partition_calls == [1, 2]
    assert all(param.ds_status == ZeroParamStatus.NOT_AVAILABLE for param in module.parameters())


def test_explicit_release_cleans_up_abandoned_forward_graph():
    module, param = _zero_module_with_param()
    param.ds_status = ZeroParamStatus.AVAILABLE
    param.ds_persist = False
    partition_calls = []

    def partition():
        partition_calls.append(param.ds_id)
        param.ds_status = ZeroParamStatus.NOT_AVAILABLE

    param.partition = partition

    class FakeEngine:
        pass

    engine = FakeEngine()
    engine.module = module
    fallback = DeepCompileZ3EagerFallback(engine)
    fallback.record_gathered_param(param)
    fallback.record_forward_graph()
    fallback.release_gathered_params()

    assert fallback.stats()["outstanding_forward_graph_ids"] == []
    assert fallback.stats()["tracked_param_ids"] == []
    assert partition_calls == [7]


def test_completed_backward_keeps_gathers_for_another_outstanding_forward():
    module, param = _zero_module_with_param()
    param.ds_status = ZeroParamStatus.AVAILABLE
    param.ds_persist = False
    partition_calls = []

    def partition():
        partition_calls.append(param.ds_id)
        param.ds_status = ZeroParamStatus.NOT_AVAILABLE

    param.partition = partition

    class FakeEngine:
        pass

    engine = FakeEngine()
    engine.module = module
    fallback = DeepCompileZ3EagerFallback(engine)
    fallback.record_gathered_param(param)
    first_graph = fallback.record_forward_graph()
    second_graph = fallback.record_forward_graph()

    fallback.record_backward_start(second_graph)
    fallback.complete_backward()

    assert fallback.stats()["outstanding_forward_graph_ids"] == [first_graph]
    assert fallback.stats()["tracked_param_ids"] == [7]
    assert partition_calls == []

    fallback.record_backward_start(first_graph)
    fallback.complete_backward()

    assert fallback.stats()["outstanding_forward_graph_ids"] == []
    assert fallback.stats()["tracked_param_ids"] == []
    assert partition_calls == [7]


def test_new_fallback_owner_drops_previous_owner():
    _, param = _zero_module_with_param()
    first = DeepCompileZ3EagerFallback(engine=None)
    second = DeepCompileZ3EagerFallback(engine=None)

    first.record_gathered_param(param)
    second.record_gathered_param(param)

    assert first.stats()["tracked_param_ids"] == []
    assert second.stats()["tracked_param_ids"] == [7]


def test_user_adopted_param_diagnostic_deduplicates_repeated_transfers():
    _, param = _zero_module_with_param()
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

    with GatheredParameters([param]):
        assert engine._deepcompile_z3_eager_fallback.stats()["tracked_param_ids"] == []
        with deepcompile_z3_forward_context(engine):
            assert param.ds_status == ZeroParamStatus.AVAILABLE

    assert partition_calls == [7]
    assert engine._deepcompile_z3_eager_fallback.stats()["last_user_adopted_param_ids"] == [7]


def test_context_exception_restores_fallback_and_user_ownership_depth(monkeypatch):
    module, param = _zero_module_with_param()
    param.ds_persist = False
    param.all_gather = lambda param_list: setattr(param, "ds_status", ZeroParamStatus.AVAILABLE)
    param.partition = lambda param_list=None, has_been_updated=False: setattr(param, "ds_status", ZeroParamStatus.
                                                                              NOT_AVAILABLE)

    class FakeEngine:

        def __init__(self):
            self.module = module
            self._deepcompile_z3_eager_fallback = DeepCompileZ3EagerFallback(self)

        def is_deepcompile_active(self):
            return True

        def zero_optimization_partition_weights(self):
            return True

    engine = FakeEngine()
    previous = object()
    monkeypatch.setattr(z3_eager_fallback, "_ACTIVE_FALLBACK", previous)

    with pytest.raises(RuntimeError, match="injected"):
        with GatheredParameters([param]):
            with deepcompile_z3_forward_context(engine):
                raise RuntimeError("injected")

    assert z3_eager_fallback.get_active_z3_eager_fallback() is previous
    assert not hasattr(param, "_deepspeed_gathered_param_context_depth")
    assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE


def _zero_params_with_collective_stubs(*ds_ids):
    params = []
    gather_calls = []
    partition_calls = []
    for ds_id in ds_ids:
        param = torch.nn.Parameter(torch.empty(0))
        param.ds_id = ds_id
        param.ds_status = ZeroParamStatus.NOT_AVAILABLE

        def all_gather(param_list, *, _calls=gather_calls):
            _calls.append([int(item.ds_id) for item in param_list])
            for item in param_list:
                item.ds_status = ZeroParamStatus.AVAILABLE

        def partition(param_list=None, has_been_updated=False, *, _param=param, _calls=partition_calls):
            selected = [_param] if param_list is None else param_list
            _calls.append([int(item.ds_id) for item in selected])
            for item in selected:
                item.ds_status = ZeroParamStatus.NOT_AVAILABLE

        param.all_gather = all_gather
        param.partition = partition
        params.append(param)
    return params, gather_calls, partition_calls


def test_gathered_parameters_rejects_partial_overlap_atomically():
    (first, shared, new), gather_calls, partition_calls = _zero_params_with_collective_stubs(1, 2, 3)

    with GatheredParameters([first, shared]):
        assert gather_calls == [[1, 2]]
        assert getattr(first, "_deepspeed_gathered_param_context_depth") == 1
        assert getattr(shared, "_deepspeed_gathered_param_context_depth") == 1

        with pytest.raises(RuntimeError, match=r"cannot overlap parameters.*\[2\]"):
            with GatheredParameters([shared, new]):
                pytest.fail("overlapping inner context must not enter")

        assert gather_calls == [[1, 2]]
        assert partition_calls == []
        assert first.ds_status == ZeroParamStatus.AVAILABLE
        assert shared.ds_status == ZeroParamStatus.AVAILABLE
        assert new.ds_status == ZeroParamStatus.NOT_AVAILABLE
        assert getattr(first, "_deepspeed_gathered_param_context_depth") == 1
        assert getattr(shared, "_deepspeed_gathered_param_context_depth") == 1
        assert not hasattr(new, "_deepspeed_gathered_param_context_depth")

    assert gather_calls == [[1, 2]]
    assert partition_calls == [[1, 2]]
    assert all(param.ds_status == ZeroParamStatus.NOT_AVAILABLE for param in (first, shared, new))
    assert all(not hasattr(param, "_deepspeed_gathered_param_context_depth") for param in (first, shared, new))


def test_gathered_parameters_preserves_disjoint_nesting():
    (first, second), gather_calls, partition_calls = _zero_params_with_collective_stubs(1, 2)

    with GatheredParameters([first]):
        with GatheredParameters([second]):
            assert gather_calls == [[1], [2]]
            assert getattr(first, "_deepspeed_gathered_param_context_depth") == 1
            assert getattr(second, "_deepspeed_gathered_param_context_depth") == 1

        assert first.ds_status == ZeroParamStatus.AVAILABLE
        assert second.ds_status == ZeroParamStatus.NOT_AVAILABLE
        assert getattr(first, "_deepspeed_gathered_param_context_depth") == 1
        assert not hasattr(second, "_deepspeed_gathered_param_context_depth")

    assert gather_calls == [[1], [2]]
    assert partition_calls == [[2], [1]]
    assert all(param.ds_status == ZeroParamStatus.NOT_AVAILABLE for param in (first, second))
    assert all(not hasattr(param, "_deepspeed_gathered_param_context_depth") for param in (first, second))
