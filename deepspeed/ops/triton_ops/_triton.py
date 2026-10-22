# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Shared Triton availability check and self-reporting kernel probes for the ``triton_ops`` package.
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


_TRITON_PROBE_CACHE: dict[str, bool] = {}


def probe_triton_kernel(key: str, run, reference=None, tolerance: float | None = None) -> bool:
    """Probe a Triton kernel once per process and cache the answer.

    The gate for a kernel becomes self-reporting: the current backend answers by
    actually running the kernel, so every platform - present or future - decides
    for itself, and a platform that gains or fixes support flips the gate
    automatically with no code change. ``run`` executes the smallest form of the
    kernel that reproduces its known-risky usage (a compile crash, a precision
    loss, ...); when ``reference`` is given the outputs are compared - bitwise
    when ``tolerance`` is None, else with ``torch.allclose``. Any exception
    answers False, which sends callers to their fallback.
    """
    if key in _TRITON_PROBE_CACHE:
        return _TRITON_PROBE_CACHE[key]
    import torch  # keep the module import-light; the probe path is cold

    supported = False
    try:
        actual = run()
        if reference is not None:
            expected = reference()
            if tolerance is None:
                supported = torch.equal(actual, expected)
            else:
                supported = torch.allclose(actual, expected, atol=tolerance, rtol=tolerance)
        else:
            supported = True
    except Exception:
        supported = False
    _TRITON_PROBE_CACHE[key] = supported
    return supported


def is_triton_kernel_supported(kernel: str, dtype, device, probe_fn, *probe_args) -> bool:
    """Ask the accelerator's Triton gate to validate ``kernel`` on this device and dtype.

    ``probe_fn(*probe_args)`` returns the kernel's minimal risky usage (see the
    ``_probes`` library) and is only invoked when the answer is not yet cached, so
    the probe runs at most once per process and every later call costs one
    dictionary lookup without even constructing the probe. No platform keeps a
    blacklist, and a backend that gains or fixes support flips the gate
    automatically. Platforms whose ``is_triton_supported`` predates the per-kernel
    extension keep their static backend-level answer.
    """
    from deepspeed.accelerator import get_accelerator

    accelerator = get_accelerator()
    if not accelerator.is_triton_supported():
        return False
    key = f"triton_kernel:{kernel}:{device.type}:{dtype}"
    if key in _TRITON_PROBE_CACHE:
        return _TRITON_PROBE_CACHE[key]
    try:
        supported = accelerator.is_triton_supported(kernel=f"{kernel}:{device.type}:{dtype}",
                                                    probe=probe_fn(*probe_args))
    except TypeError:
        supported = True
    _TRITON_PROBE_CACHE[key] = supported
    return supported


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
