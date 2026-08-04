# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import deque

import torch
from deepspeed.utils.torch import required_torch_version

backward_inputs = {}
backward_frame_keys = deque()

enabled_patched_func = False
original_grad_fn = None
base_meta = type(torch.autograd.Function)

if required_torch_version(min_version=2.7):

    class FunctionMeta(base_meta):

        def __new__(cls, name, bases, dct):
            if name == "CompiledFunction":
                original_backward_impl = dct.get("_backward_impl")
                frame_key = backward_frame_keys.popleft() if backward_frame_keys else None

                def wrapped_backward_impl(ctx, all_args):
                    assert original_backward_impl is not None

                    if enabled_patched_func and frame_key is not None:
                        backward_inputs.setdefault(frame_key, []).append(all_args)
                        wrapped_backward_impl.owner_class.compiled_bw = None

                    return original_backward_impl(ctx, all_args)

                wrapped_backward_impl.owner_class = None
                dct["_backward_impl"] = staticmethod(wrapped_backward_impl)
                new_class = super().__new__(cls, name, bases, dct)
                wrapped_backward_impl.owner_class = new_class

                return new_class

            return super().__new__(cls, name, bases, dct)

elif required_torch_version(min_version=2.6):

    class FunctionMeta(base_meta):

        def __new__(cls, name, bases, dct):
            if name == "CompiledFunction":
                original_backward_prologue = dct.get("_backward_prologue")
                frame_key = backward_frame_keys.popleft() if backward_frame_keys else None

                def wrapped_backward_prologue(ctx, *grad_outputs):
                    assert original_backward_prologue is not None

                    all_args = original_backward_prologue(ctx, *grad_outputs)
                    if enabled_patched_func and frame_key is not None:
                        backward_inputs.setdefault(frame_key, []).append(all_args)
                        wrapped_backward_prologue.owner_class.compiled_bw = None

                    return all_args

                wrapped_backward_prologue.owner_class = None
                dct["_backward_prologue"] = staticmethod(wrapped_backward_prologue)
                new_class = super().__new__(cls, name, bases, dct)
                wrapped_backward_prologue.owner_class = new_class

                return new_class

            return super().__new__(cls, name, bases, dct)


def patch_compiled_func():

    global enabled_patched_func
    enabled_patched_func = True

    class PatchedFunction(torch.autograd.Function, metaclass=FunctionMeta):
        pass

    global original_grad_fn
    original_grad_fn = torch.autograd.Function
    torch.autograd.Function = PatchedFunction

    return backward_inputs


def register_backward_frame(frame_key):
    """Associate the next AOT compiled function with its DeepCompile frame."""
    backward_frame_keys.append(frame_key)


def unpatch_compiled_func():
    """Restore torch.autograd.Function and discard inputs captured for this compile cycle."""
    global enabled_patched_func
    enabled_patched_func = False

    global original_grad_fn
    if original_grad_fn is not None:
        torch.autograd.Function = original_grad_fn
        original_grad_fn = None
    clear_backward_inputs()


def get_backward_inputs(frame_key=None):
    if frame_key is None:
        return backward_inputs
    return backward_inputs.get(frame_key, [])


def pop_backward_input(frame_key):
    """Pop captured real inputs for one DeepCompile frame."""
    frame_inputs = backward_inputs.get(frame_key)
    if not frame_inputs:
        return None

    inputs = frame_inputs.pop()
    if not frame_inputs:
        backward_inputs.pop(frame_key)
    return inputs


def clear_backward_inputs(frame_keys=None):
    """Drop captured inputs and pending capture registrations for selected frames."""
    if frame_keys is None:
        backward_inputs.clear()
        backward_frame_keys.clear()
        return

    frame_keys = set(frame_keys)
    for frame_key in frame_keys:
        backward_inputs.pop(frame_key, None)

    retained_frame_keys = [frame_key for frame_key in backward_frame_keys if frame_key not in frame_keys]
    backward_frame_keys.clear()
    backward_frame_keys.extend(retained_frame_keys)
