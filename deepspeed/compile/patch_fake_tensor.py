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


def _get_guard_sizes_strides(t):
    if hasattr(t, "ds_id"):
        # ZeRO-3 may temporarily all-gather a parameter during tracing, but the
        # stable module state used by TorchDynamo guards is the released
        # partitioned form, where DeepSpeed resets param.data to empty(0).
        released = torch.empty(0, dtype=t.dtype, device=t.device)
        return released.size(), released.stride()

    return t.size(), t.stride()


def patch_fake_tensor():
    # dynamo tracer uses wrap_to_fake_tensor_and_record
    # Wrapping FakeTensorMode.from_tensor is not sufficient as dynamo generates SymbolicContext before calling from_tensor
    original_wrap_to_fake_tensor_and_record = wrap_to_fake_tensor_and_record

    def wrap_to_fake_tensor_and_record_wrapper(t, *args, **kwargs):
        dummy_tensor = wrap_if_ds_param(t)
        ret = original_wrap_to_fake_tensor_and_record(dummy_tensor, *args, **kwargs)
        tx = kwargs.get("tx") if "tx" in kwargs else args[0]
        source = kwargs.get("source")
        if tracing_context := torch._guards.TracingContext.try_get():
            tracing_context.tensor_to_context[t] = tracing_context.tensor_to_context.pop(dummy_tensor)
        if source is not None:
            # Keep the full ds_shape symbolic context from the dummy tensor, but
            # use the stable released ZeRO-3 parameter representation for
            # TorchDynamo's tensor-match guards. PyTorch 2.9 started enforcing
            # those guards for parameters during build_guards().
            size, stride = _get_guard_sizes_strides(t)
            tx.output.input_source_to_sizes_strides[source] = {
                "size": size,
                "stride": stride,
            }
        return ret

    torch._dynamo.variables.builder.wrap_to_fake_tensor_and_record = wrap_to_fake_tensor_and_record_wrapper

    # aot_module_simplified uses fake_mode.from_tensor to process inputs
    original_from_tensor = FakeTensorMode.from_tensor

    def from_tensor_wrapper(self, t, *args, **kwargs):
        with unset_fake_temporarily():
            return original_from_tensor(self, wrap_if_ds_param(t), *args, **kwargs)

    FakeTensorMode.from_tensor = from_tensor_wrapper
