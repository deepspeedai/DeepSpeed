# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Shared Triton availability check for the ``triton_ops`` package.
"""

from contextlib import contextmanager

try:
    import triton
    import triton.language as tl
    from triton.backends import backends

    # Same test triton.runtime.driver.active uses to pick a backend, without get_current_target(), which initializes
    # CUDA and breaks GPU use in fork()-ed children. For CUDA, is_active() asks torch whether a GPU is available, and
    # torch answers from NVML (deepspeed sets PYTORCH_NVML_BASED_CUDA_CHECK=1), so CPU-only machines get False.
    _TRITON_AVAILABLE = any(backend.driver.is_active() for backend in backends.values())
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    """Return True if Triton is importable and its kernels can be used."""
    return _TRITON_AVAILABLE


def is_accelerator_tensor(tensor) -> bool:
    """Whether the package's kernels can launch on ``tensor``'s device.

    Triton kernels run on the active accelerator's device and never on CPU, so
    this excludes CPU tensors before delegating to the accelerator's
    ``on_accelerator`` (the CPU accelerator accepts host tensors). The import
    is lazy to keep this module import-light.
    """
    from deepspeed.accelerator import get_accelerator

    return tensor.device.type != "cpu" and get_accelerator().on_accelerator(tensor)


@contextmanager
def current_device(tensor_or_index):
    """Run the block with the accelerator's current device switched to the tensor's device.

    Triton launches kernels on the accelerator's current device, which need not be the
    input tensor's device, so callers must switch before launching. ``torch.device`` cannot
    stand in here: it only sets the default allocation device, not the current device.
    """
    from deepspeed.accelerator import get_accelerator

    accelerator = get_accelerator()
    index = tensor_or_index if isinstance(tensor_or_index, int) else tensor_or_index.device.index
    previous = accelerator.current_device()
    accelerator.set_device(index)
    try:
        yield
    finally:
        accelerator.set_device(previous)
