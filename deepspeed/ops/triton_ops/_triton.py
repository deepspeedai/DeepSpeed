# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Shared Triton availability check and per-hardware implementation dispatch for the ``triton_ops`` package.
"""

import functools
import threading
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


_TRITON_KERNEL_PROBE_CACHE: dict[str, bool] = {}
_TRITON_KERNEL_PROBE_ERRORS: dict[str, Exception] = {}
_TRITON_KERNEL_PROBE_LOCK = threading.Lock()


def probe_once(key: str, run) -> bool:
    """Run a platform-owned kernel probe once per key and cache the answer.

    This is a building block for platform variants that choose to validate a
    kernel before using it; it is deliberately not a global gate. The caller
    owns the key, the probe input, and the policy on failure (fall back, raise,
    or warn), so each accelerator can decide how much validation and what
    failure behavior a given kernel gets. The policy is per probe/kernel, not
    per accelerator or per package. A platform with no variant of its own
    can add a probe-only variant that wraps the portable entry point with this
    helper; a platform that ships its own implementation can use it as a canary
    and raise on failure. The shared layer never probes on its own. Exceptions
    from ``run`` are contained and cached as a failed probe; ``probe_error``
    returns the original exception for logging or chaining. The lock serializes
    first-use probes of the same process, matching the one-time cost of a
    Triton compilation.
    """
    with _TRITON_KERNEL_PROBE_LOCK:
        if key in _TRITON_KERNEL_PROBE_CACHE:
            return _TRITON_KERNEL_PROBE_CACHE[key]
        try:
            run()
            supported = True
        except Exception as exc:
            _TRITON_KERNEL_PROBE_ERRORS[key] = exc
            supported = False
        _TRITON_KERNEL_PROBE_CACHE[key] = supported
        return supported


def probe_error(key: str) -> Exception | None:
    """The exception that closed a platform-owned probe, if any."""
    return _TRITON_KERNEL_PROBE_ERRORS.get(key)


_TRITON_OPS_IMPL_CACHE: dict[str, object] = {}


def resolve_triton_ops_impl(feature: str, portable_module: str | None = None):
    """Resolve the platform-specific implementation module for a triton_ops feature.

    The accelerator names its own triton_ops subpackage (op_builder-style) when
    it ships kernel variants; platforms without one resolve the portable module.
    A feature the platform does not override falls back to the portable module,
    so a subpackage may hold a single kernel. ``portable_module`` names the
    module holding the portable implementation, defaulting to the feature's
    module inside this package; Triton kernels living elsewhere in DeepSpeed
    (chunked cross entropy, fp8 GEMM, MoE permute) pass their own module, while
    every platform variant still lives under the accelerator's triton_ops
    subpackage. The accelerator is fixed per process, so the answer is cached
    per feature: the steady state is one dictionary lookup.
    """
    cached = _TRITON_OPS_IMPL_CACHE.get(feature)
    if cached is not None:
        return cached
    import importlib

    from deepspeed.accelerator import get_accelerator

    subpackage = get_accelerator().triton_ops_dir()
    if subpackage is not None:
        try:
            impl = importlib.import_module(f"{subpackage}.{feature}")
        except ModuleNotFoundError:
            # The platform ships a subpackage but does not override this
            # feature; the portable module stands.
            impl = None
        if impl is not None:
            _TRITON_OPS_IMPL_CACHE[feature] = impl
            return impl
    impl = importlib.import_module(portable_module or f"deepspeed.ops.triton_ops.{feature}")
    _TRITON_OPS_IMPL_CACHE[feature] = impl
    return impl


def dispatch(feature: str, portable_module: str | None = None):
    """Decorate a triton_ops public function to run the platform's variant when it ships one.

    The decorated function is the portable implementation and the fallback. The
    platform subpackage overrides it by defining a function with the same name;
    re-exporting the decorated portable function counts as "no override", so a
    variant module may re-export freely without recursing (the re-exported name
    is this very wrapper, recognized by identity). The resolution is cached per
    feature, so the steady state is one dictionary lookup plus one getattr per
    call. ``portable_module`` names the module holding the portable
    implementation when the feature does not live inside ``triton_ops``; the
    platform variant still lives under the accelerator's triton_ops subpackage.
    A variant overrides the whole function: shared steps the platform does not
    change (validation, the accelerator gate, the eager fallback) are factored
    into helpers of the portable module that the variant imports.
    """

    def decorator(fn):

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            impl = resolve_triton_ops_impl(feature, portable_module)
            impl_fn = getattr(impl, fn.__name__, None)
            if impl_fn is None or impl_fn is wrapper:
                # The platform module does not override this function, or only
                # re-exports the decorated portable one; the portable original stands.
                return fn(*args, **kwargs)
            return impl_fn(*args, **kwargs)

        return wrapper

    return decorator


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
