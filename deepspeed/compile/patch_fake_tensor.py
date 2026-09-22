# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from functools import wraps

import torch

from deepspeed.accelerator import get_accelerator

try:
    from torch._subclasses import FakeTensorMode
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch._dynamo.guards import GuardBuilder, get_verbose_code_parts
    from torch._dynamo.variables.builder import wrap_to_fake_tensor_and_record
    from torch.utils.weak import TensorWeakRef
except ImportError:
    # Unsupported torch version
    pass


def wrap_if_ds_param(t):
    if hasattr(t, 'ds_id'):
        data = torch.rand(t.ds_shape,
                          dtype=t.dtype,
                          layout=t.layout,
                          device=t.device,
                          pin_memory=get_accelerator().is_pinned(t),
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


def _zero3_parameter_guard_metadata(param):
    return (type(param), torch._C._dispatch_keys(param), param.dtype, param.device, param.layout, param.requires_grad,
            param.ds_id, tuple(param.ds_shape))


def _matches_zero3_parameter_metadata(param, expected):
    return isinstance(param, torch.Tensor) and _zero3_parameter_guard_metadata(param) == expected


def _get_guard_source_value(builder, guard):
    try:
        return builder.get(guard)
    except TypeError:
        # Older PyTorch releases accept the guard's source expression only.
        return builder.get(guard.name)


def _resolve_zero3_guarded_value(builder, guard, value):
    guarded_value = value() if isinstance(value, TensorWeakRef) else value
    if not (hasattr(guarded_value, "ds_id") and hasattr(guarded_value, "ds_shape")):
        # Some Dynamo paths pass the full-shape dummy/FakeTensor produced
        # above instead of the module parameter. Resolve the source so all
        # guards for a ZeRO parameter use its stable semantic metadata.
        source_value = _get_guard_source_value(builder, guard)
        if hasattr(source_value, "ds_id") and hasattr(source_value, "ds_shape"):
            guarded_value = source_value
    return guarded_value if guarded_value is not None else _get_guard_source_value(builder, guard)


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

    original_tensor_match = GuardBuilder.TENSOR_MATCH

    @wraps(original_tensor_match)
    def tensor_match_wrapper(self, guard, value=None):
        guarded_value = _resolve_zero3_guarded_value(self, guard, value)
        if hasattr(guarded_value, "ds_id") and hasattr(guarded_value, "ds_shape"):
            expected = _zero3_parameter_guard_metadata(guarded_value)

            def semantic_parameter_guard(param):
                return _matches_zero3_parameter_metadata(param, expected)

            code = f"DeepCompile ZeRO-3 semantic parameter ds_id={expected[-2]} shape={expected[-1]}"
            guard_manager = self.get_guard_manager(guard)
            verbose_code = get_verbose_code_parts(code, guard)
            try:
                guard_manager.add_lambda_guard(semantic_parameter_guard, verbose_code, guard.user_stack)
            except TypeError:
                # PyTorch 2.10 does not yet accept the user stack argument.
                guard_manager.add_lambda_guard(semantic_parameter_guard, verbose_code)
            return
        return original_tensor_match(self, guard, value)

    GuardBuilder.TENSOR_MATCH = tensor_match_wrapper

    # aot_module_simplified uses fake_mode.from_tensor to process inputs
    original_from_tensor = FakeTensorMode.from_tensor

    def from_tensor_wrapper(self, t, *args, **kwargs):
        with unset_fake_temporarily():
            return original_from_tensor(self, wrap_if_ds_param(t), *args, **kwargs)

    FakeTensorMode.from_tensor = from_tensor_wrapper
