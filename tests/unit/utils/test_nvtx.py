# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import inspect
import pickle

import deepspeed.utils.nvtx as ds_nvtx
import accelerator.cuda_accelerator as cuda_accelerator
from accelerator.cuda_accelerator import CUDA_Accelerator


def _sample_nvtx_function():
    return "ok"


@ds_nvtx.instrument_w_nvtx
def _documented_nvtx_function(value, scale=2):
    """Return the value, scaled."""
    return value * scale


def test_instrument_w_nvtx_uses_deepspeed_domain(monkeypatch, capsys):
    calls = []

    class FakeAccelerator:
        supports_nvtx_domain = True

        def range_push(self, msg, domain=None, category=None):
            calls.append(("push", msg, domain, category))

        def range_pop(self, domain=None):
            calls.append(("pop", domain))

    monkeypatch.setattr(ds_nvtx, "enable_nvtx", True)
    monkeypatch.setattr(ds_nvtx, "is_compiling", lambda: False)
    monkeypatch.setattr(ds_nvtx, "get_accelerator", lambda: FakeAccelerator())

    wrapped_fn = ds_nvtx.instrument_w_nvtx(_sample_nvtx_function)

    assert wrapped_fn() == "ok"

    with capsys.disabled():
        print(f"\nNVTX instrumentation calls: {calls}")

    assert calls == [
        ("push", "_sample_nvtx_function", ds_nvtx.DEEPSPEED_NVTX_DOMAIN, None),
        ("pop", ds_nvtx.DEEPSPEED_NVTX_DOMAIN),
    ]


def test_instrument_w_nvtx_supports_legacy_accelerator_methods(monkeypatch, capsys):
    calls = []

    class LegacyAccelerator:

        def range_push(self, msg):
            calls.append(("push", msg))

        def range_pop(self):
            calls.append(("pop", ))

    monkeypatch.setattr(ds_nvtx, "enable_nvtx", True)
    monkeypatch.setattr(ds_nvtx, "is_compiling", lambda: False)
    monkeypatch.setattr(ds_nvtx, "get_accelerator", lambda: LegacyAccelerator())

    wrapped_fn = ds_nvtx.instrument_w_nvtx(_sample_nvtx_function)

    assert wrapped_fn() == "ok"

    with capsys.disabled():
        print(f"\nLegacy NVTX instrumentation calls: {calls}")

    assert calls == [
        ("push", "_sample_nvtx_function"),
        ("pop", ),
    ]


def test_cuda_accelerator_uses_nvtx_domain_when_available(monkeypatch, capsys):

    class FakeDomain:

        def __init__(self):
            self.calls = []

        def push_range(self, message=None, category=None):
            self.calls.append(("push", message, category))
            return "domain-push"

        def pop_range(self):
            self.calls.append(("pop", ))
            return "domain-pop"

    class FakeNvtx:

        def __init__(self):
            self.domains = {}

        def get_domain(self, name):
            self.domains.setdefault(name, FakeDomain())
            return self.domains[name]

    fake_nvtx = FakeNvtx()
    accelerator = CUDA_Accelerator.__new__(CUDA_Accelerator)
    accelerator._nvtx_domains = {}
    monkeypatch.setattr(cuda_accelerator, "nvtx", fake_nvtx)

    assert accelerator.range_push("my_range", domain="DeepSpeed", category="zero") == "domain-push"
    assert accelerator.range_pop(domain="DeepSpeed") == "domain-pop"

    with capsys.disabled():
        print(f"\nCUDA NVTX domain calls: {fake_nvtx.domains['DeepSpeed'].calls}")

    assert fake_nvtx.domains["DeepSpeed"].calls == [
        ("push", "my_range", "zero"),
        ("pop", ),
    ]


def test_cuda_accelerator_falls_back_to_torch_nvtx_without_nvtx_package(monkeypatch, capsys):
    calls = []

    class FakeTorchNvtx:

        def range_push(self, msg):
            calls.append(("push", msg))
            return "torch-push"

        def range_pop(self):
            calls.append(("pop", ))
            return "torch-pop"

    accelerator = CUDA_Accelerator.__new__(CUDA_Accelerator)
    monkeypatch.setattr(cuda_accelerator, "nvtx", None)
    monkeypatch.setattr(cuda_accelerator.torch.cuda, "nvtx", FakeTorchNvtx())  #ignore-cuda

    assert accelerator.range_push("my_range", domain="DeepSpeed", category="zero") == "torch-push"
    assert accelerator.range_pop(domain="DeepSpeed") == "torch-pop"

    with capsys.disabled():
        print(f"\nCUDA torch.nvtx fallback calls: {calls}")

    assert calls == [
        ("push", "my_range"),
        ("pop", ),
    ]


def test_instrument_w_nvtx_preserves_wrapped_function_metadata(capsys):
    # Without metadata copying the wrapper reports __name__ 'wrapped_fn', a None
    # docstring and a signature of (*args, **kwargs), so help(), IDE tooltips and
    # the autodoc build all describe the decorator instead of the function.
    decorated = _documented_nvtx_function

    with capsys.disabled():
        print(f"\nDecorated metadata: {decorated.__name__} {inspect.signature(decorated)}")

    assert decorated.__name__ == "_documented_nvtx_function"
    assert decorated.__qualname__ == "_documented_nvtx_function"
    assert decorated.__doc__ == "Return the value, scaled."
    assert decorated.__module__ == __name__
    assert str(inspect.signature(decorated)) == "(value, scale=2)"
    assert decorated.__wrapped__.__name__ == "_documented_nvtx_function"


def test_instrument_w_nvtx_keeps_a_module_level_function_picklable(capsys):
    # pickle resolves a function by __module__ and __qualname__ and then checks the
    # lookup returns the same object. A wrapper that keeps the decorator's own
    # qualname resolves to a local and raises, which breaks anything that sends a
    # decorated function to another process.
    blob = pickle.dumps(_documented_nvtx_function)

    with capsys.disabled():
        print(f"\nPickled qualname: {_documented_nvtx_function.__qualname__}")

    assert pickle.loads(blob) is _documented_nvtx_function
