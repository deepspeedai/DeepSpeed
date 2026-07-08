# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from contextlib import contextmanager

import torch

from deepspeed.runtime.zero.parameter_offload import ZeROOrderedDict, ensure_zero_ordered_dict
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

_ACTIVE_FALLBACK = None


def get_active_z3_eager_fallback():
    return _ACTIVE_FALLBACK


def record_z3_eager_fallback_param(param):
    fallback = get_active_z3_eager_fallback()
    if fallback is None:
        return False
    fallback.record_gathered_param(param)
    return True


@contextmanager
def deepcompile_z3_forward_context(engine):
    fallback = getattr(engine, "_deepcompile_z3_eager_fallback", None)
    if fallback is None or not engine.is_deepcompile_active() or not engine.zero_optimization_partition_weights():
        yield
        return

    with fallback.forward_context():
        yield


class DeepCompileZ3EagerFallback:

    def __init__(self, engine):
        self.engine = engine
        self._depth = 0
        self._tracked_params = {}
        self._last_gathered_param_ids = []
        self._last_released_param_ids = []
        self.total_gathered_params = 0

    @contextmanager
    def forward_context(self):
        global _ACTIVE_FALLBACK
        previous = _ACTIVE_FALLBACK
        self._depth += 1
        if self._depth == 1:
            self._last_gathered_param_ids = []
            self._enable_forward_fallback()
        _ACTIVE_FALLBACK = self
        try:
            yield
        finally:
            _ACTIVE_FALLBACK = previous
            self._depth -= 1
            if self._depth == 0:
                self._disable_forward_fallback()

    def _enable_forward_fallback(self):
        for module in self.engine.module.modules():
            ensure_zero_ordered_dict(module)
            module._parameters._in_forward = True

    def _disable_forward_fallback(self):
        for module in self.engine.module.modules():
            if isinstance(module._parameters, ZeROOrderedDict):
                module._parameters._in_forward = False

    def record_gathered_param(self, param):
        ds_id = int(param.ds_id)
        self._tracked_params[ds_id] = param
        self._last_gathered_param_ids.append(ds_id)
        self.total_gathered_params += 1

    @torch.no_grad()
    def release_gathered_params(self):
        released = []
        for ds_id, param in list(self._tracked_params.items()):
            if (hasattr(param, "ds_status") and param.ds_status == ZeroParamStatus.AVAILABLE
                    and not getattr(param, "ds_persist", False)):
                param.partition()
                released.append(ds_id)
        self._tracked_params.clear()
        self._last_released_param_ids = released

    def stats(self):
        return {
            "tracked_param_ids": sorted(self._tracked_params),
            "last_gathered_param_ids": list(self._last_gathered_param_ids),
            "last_released_param_ids": list(self._last_released_param_ids),
            "total_gathered_params": self.total_gathered_params,
        }
