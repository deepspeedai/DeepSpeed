# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Regression tests for issue #6961.

ZeRO-3 forward used to crash with ``AttributeError: 'dict' object has no
attribute '_in_forward'`` when a submodule's ``_parameters`` was a plain
``dict`` instead of a ``ZeROOrderedDict``. PyTorch 2.5+ defaults
``nn.Module._parameters`` to ``dict`` (pytorch/pytorch#129164), and any
module not converted at ``DeepSpeedZeRoOffload`` init time hits the crash.
The tests force the plain-dict condition explicitly so they exercise the
fix on every supported torch version, not only torch 2.5+.
"""

import torch

import deepspeed
from deepspeed.runtime.zero.parameter_offload import (ZeROOrderedDict, ensure_zero_ordered_dict)

from unit.common import DistributedTest, preferred_dtype


class _Tiny(torch.nn.Module):

    def __init__(self, hidden_dim=16):
        super().__init__()
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.fc(x)


def _zero3_config(dtype):
    return {
        "train_batch_size": 1,
        "fp16": {
            "enabled": dtype is torch.float16
        },
        "bf16": {
            "enabled": dtype is torch.bfloat16
        },
        "zero_optimization": {
            "stage": 3
        },
    }


class TestZero3LateModuleAttach(DistributedTest):
    world_size = 1

    def test_forward_after_late_submodule_attach(self):
        """Attaching a fresh ``nn.Linear`` after ``initialize`` must not crash."""
        hidden = 16
        dtype = preferred_dtype()
        model = _Tiny(hidden)
        engine, *_ = deepspeed.initialize(model=model,
                                          config=_zero3_config(dtype),
                                          model_parameters=list(model.parameters()))

        late = torch.nn.Linear(hidden, hidden, bias=False).to(device=engine.device, dtype=dtype)
        # Force the post-pytorch/pytorch#129164 condition deterministically so
        # the test exercises the fix regardless of the installed torch version.
        late._parameters = dict(late._parameters)
        engine.module.late = late

        x = torch.randn(2, hidden, dtype=dtype, device=engine.device)
        engine(x)

        # Prologue must have lazily converted the late submodule.
        assert isinstance(engine.module.late._parameters, ZeROOrderedDict)

    def test_idempotent_on_already_injected_modules(self):
        """Repeated forwards must not re-wrap an already-converted ``_parameters``."""
        hidden = 16
        dtype = preferred_dtype()
        model = _Tiny(hidden)
        engine, *_ = deepspeed.initialize(model=model,
                                          config=_zero3_config(dtype),
                                          model_parameters=list(model.parameters()))

        first_pdict = engine.module.fc._parameters
        assert isinstance(first_pdict, ZeROOrderedDict)

        x = torch.randn(2, hidden, dtype=dtype, device=engine.device)
        engine(x)
        engine(x)

        assert engine.module.fc._parameters is first_pdict


class TestEnsureZeroOrderedDict:
    """Direct unit tests for the helper. No distributed harness needed."""

    def test_skips_already_converted(self):
        m = torch.nn.Linear(4, 4, bias=False)
        m._parameters = ZeROOrderedDict(parent_module=m)
        before = m._parameters
        ensure_zero_ordered_dict(m)
        assert m._parameters is before

    def test_wraps_plain_dict(self):
        m = torch.nn.Linear(4, 4, bias=False)
        m._parameters = dict(m._parameters)
        ensure_zero_ordered_dict(m)
        assert isinstance(m._parameters, ZeROOrderedDict)
        assert "weight" in m._parameters
        assert m._original_parameters is not m._parameters

    def test_preserves_existing_original_parameters(self):
        """Subsequent wraps must not clobber the first-saved original.

        ``_inject_parameters`` at engine init records the true torch-native
        container in ``_original_parameters``; the deepcompile path in
        ``init_z3.py`` reads it back to un-inject. If the helper later runs
        after some intermediate replacement of ``_parameters``, it must not
        overwrite that saved reference.
        """
        m = torch.nn.Linear(4, 4, bias=False)
        sentinel = m._parameters
        m._original_parameters = sentinel
        m._parameters = dict(sentinel)  # different object, same contents
        ensure_zero_ordered_dict(m)
        assert m._original_parameters is sentinel

    def test_noop_when_parameters_missing(self):
        """Helper must not raise when ``_parameters`` is missing or None."""

        class Bare:
            pass

        m = Bare()
        ensure_zero_ordered_dict(m)  # no-op, no exception
        m._parameters = None
        ensure_zero_ordered_dict(m)  # no-op, no exception
        assert m._parameters is None
