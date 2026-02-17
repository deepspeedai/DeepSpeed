import types

from deepspeed.utils import torch as ds_torch_utils


def test_register_grad_hook_uses_post_accumulate_hook(monkeypatch):
    monkeypatch.setattr(ds_torch_utils, "required_torch_version", lambda **_kwargs: True)

    recorded = {}

    class DummyParam:

        def register_post_accumulate_grad_hook(self, hook):
            recorded["hook"] = hook
            return "post_acc_handle"

    handle = ds_torch_utils.register_grad_hook(DummyParam(), lambda *_args: None)

    assert handle == "post_acc_handle"
    assert "hook" in recorded


def test_register_grad_hook_uses_legacy_grad_accumulator(monkeypatch):
    monkeypatch.setattr(ds_torch_utils, "required_torch_version", lambda **_kwargs: False)

    recorded = {}

    class DummyGradAccumulator:

        def register_hook(self, hook):
            recorded["hook"] = hook
            return "grad_acc_handle"

    grad_acc = DummyGradAccumulator()

    class DummyParam:

        def expand_as(self, _param):
            return types.SimpleNamespace(
                grad_fn=types.SimpleNamespace(
                    next_functions=((grad_acc, None), ),
                ))

        def register_hook(self, _hook):
            raise AssertionError("legacy param hook fallback should not be used")

    handle = ds_torch_utils.register_grad_hook(DummyParam(), lambda *_args: None)

    assert handle == "grad_acc_handle"
    assert "hook" in recorded


def test_register_grad_hook_falls_back_when_grad_accumulator_missing(monkeypatch):
    monkeypatch.setattr(ds_torch_utils, "required_torch_version", lambda **_kwargs: False)
    monkeypatch.setattr(ds_torch_utils, "_legacy_fallback_logged", False)

    recorded = {}
    invoked = {}

    class DummyParam:

        def expand_as(self, _param):
            return types.SimpleNamespace(grad_fn=None)

        def register_hook(self, hook):
            recorded["hook"] = hook
            return "param_hook_handle"

    param = DummyParam()

    def on_grad(_param):
        invoked["called"] = True
        invoked["param"] = _param

    handle = ds_torch_utils.register_grad_hook(param, on_grad)

    assert handle == "param_hook_handle"
    assert "hook" in recorded

    grad = object()
    assert recorded["hook"](grad) is grad
    assert invoked["called"] is True
    assert invoked["param"] is param
