# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

try:
    from torch._subclasses import FakeTensorMode
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch._dynamo.variables.builder import wrap_to_fake_tensor_and_record
except ImportError:
    # Unsupported torch version
    pass


def wrap_if_ds_param(t):
    if hasattr(t, 'ds_id'):
        data = torch.rand(t.ds_shape,
                          dtype=t.dtype,
                          layout=t.layout,
                          device=t.device,
                          pin_memory=t.is_pinned(),
                          requires_grad=t.requires_grad)
        if isinstance(t, torch.nn.Parameter):
            t = torch.nn.Parameter(data, requires_grad=t.requires_grad)
        else:
            t = data
    return t


def patch_fake_tensor():
    # dynamo tracer uses wrap_to_fake_tensor_and_record
    # Wrapping FakeTensorMode.from_tensor is not sufficient as dynamo generates SymbolicContext before calling from_tensor
    original_wrap_to_fake_tensor_and_record = wrap_to_fake_tensor_and_record

    def wrap_to_fake_tensor_and_record_wrapper(t, *args, **kwargs):
        dummy_tensor = wrap_if_ds_param(t)
        ret = original_wrap_to_fake_tensor_and_record(dummy_tensor, *args, **kwargs)
        if tracing_context := torch._guards.TracingContext.try_get():
            tracing_context.tensor_to_context[t] = tracing_context.tensor_to_context.pop(dummy_tensor)
        return ret

    torch._dynamo.variables.builder.wrap_to_fake_tensor_and_record = wrap_to_fake_tensor_and_record_wrapper

    # aot_module_simplified uses fake_mode.from_tensor to process inputs
    original_from_tensor = FakeTensorMode.from_tensor

    def from_tensor_wrapper(self, t, *args, **kwargs):
        with unset_fake_temporarily():
            return original_from_tensor(self, wrap_if_ds_param(t), *args, **kwargs)

    FakeTensorMode.from_tensor = from_tensor_wrapper
