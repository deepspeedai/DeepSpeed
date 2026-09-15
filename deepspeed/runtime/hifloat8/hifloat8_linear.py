# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Lazy bridge to torch_npu's optional HiFloat8 training implementation."""


def _load_implementation():
    from torch_npu.utils.hifloat8_train.hifloat8_linear import (
        HiFloat8Linear,
        assert_hifloat8_training_available,
        convert_to_hifloat8_training,
    )

    return HiFloat8Linear, assert_hifloat8_training_available, convert_to_hifloat8_training


def get_hifloat8_linear_class():
    return _load_implementation()[0]


def assert_hifloat8_training_available(*, probe_kernel=True, device=None):
    return _load_implementation()[1](probe_kernel=probe_kernel, device=device)


def convert_to_hifloat8_training(module, *, module_filter_fn=None):
    return _load_implementation()[2](module, module_filter_fn=module_filter_fn)


def is_hifloat8_available(*, probe_kernel=True, device=None):
    try:
        assert_hifloat8_training_available(probe_kernel=probe_kernel, device=device)
        return True
    except (ImportError, RuntimeError):
        return False


__all__ = [
    "assert_hifloat8_training_available",
    "convert_to_hifloat8_training",
    "get_hifloat8_linear_class",
    "is_hifloat8_available",
]
