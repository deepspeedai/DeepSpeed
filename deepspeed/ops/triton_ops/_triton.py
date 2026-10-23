# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Shared Triton availability check and self-reporting kernel probes for the ``triton_ops`` package.
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


_TRITON_PROBE_CACHE: dict[str, bool] = {}
_TRITON_PROBE_LOCK = threading.Lock()


def probe_triton_kernel(key: str, run, reference=None, tolerance: float | None = None) -> bool:
    """Probe a Triton kernel once per process and cache the answer.

    The gate for a kernel becomes self-reporting: the current backend answers by
    actually running the kernel, so every platform - present or future - decides
    for itself, and a platform that gains or fixes support flips the gate
    automatically with no code change. ``run`` executes the smallest form of the
    kernel that reproduces its known-risky usage (a compile crash, for example)
    and is followed by a synchronization of the current stream so faults a
    backend only reports asynchronously fail the probe here, not on the first
    real workload. When ``reference`` is given the outputs are compared -
    bitwise when ``tolerance`` is None, else with ``torch.allclose`` - though
    none of the current probes needs a numerical check. Any exception answers
    False (with a one-time warning), which sends callers to their fallback.
    """
    with _TRITON_PROBE_LOCK:
        if key in _TRITON_PROBE_CACHE:
            return _TRITON_PROBE_CACHE[key]
        import torch  # keep the module import-light; the probe path is cold

        supported = False
        try:
            actual = run()
            from deepspeed.accelerator import get_accelerator

            # Synchronize the current stream only: a device-wide synchronize
            # would also wait on unrelated streams and could surface their
            # asynchronous errors here. The probe's run pinned the current
            # device, so this is the probe's own stream.
            get_accelerator().current_stream().synchronize()
            if reference is not None:
                if actual is None:
                    supported = False
                else:
                    expected = reference()
                    if tolerance is None:
                        supported = torch.equal(actual, expected)
                    else:
                        supported = torch.allclose(actual, expected, atol=tolerance, rtol=tolerance)
            else:
                supported = True
        except Exception:
            supported = False
        if not supported:
            from deepspeed.utils import logger

            logger.warning_once(f"Triton kernel probe {key!r} failed on this platform; "
                                "callers keep their non-Triton fallback.")
        _TRITON_PROBE_CACHE[key] = supported
        return supported


def is_triton_kernel_supported(kernel: str, dtype, device, probe_fn, *probe_args) -> bool:
    """Ask the accelerator's Triton gate to validate ``kernel`` on this device and dtype.

    ``probe_fn(*probe_args)`` returns the kernel's minimal risky usage (see the
    ``_probes`` library) and is only invoked when the answer is not yet cached, so
    the probe runs at most once per process and every later call costs one
    dictionary lookup without even constructing the probe. ``probe_fn`` may be
    None for a static per-kernel declaration: nothing runs, and the accelerator
    answers for the kernel name alone (used for probe-unsafe paths). No platform
    keeps a blacklist, and a backend that gains or fixes support flips the gate
    automatically. Platforms whose ``is_triton_supported`` predates the per-kernel
    extension keep their static backend-level answer.
    """
    key = f"triton_kernel:{kernel}:{device.type}:{device.index}"
    if key in _TRITON_PROBE_CACHE:
        # Steady state: one dictionary lookup per call; no probe construction, no backend query.
        return _TRITON_PROBE_CACHE[key]
    import os
    if os.environ.get("DS_TRITON_KERNEL_GATE") == "0":
        # Emergency escape hatch: skip per-kernel probing entirely and trust the
        # backend-level gate (the pre-probe behavior).
        from deepspeed.accelerator import get_accelerator

        return get_accelerator().is_triton_supported()
    from deepspeed.accelerator import get_accelerator

    accelerator = get_accelerator()
    if not accelerator.is_triton_supported():
        # The backend-level gate cannot open mid-process; cache its answer too.
        _TRITON_PROBE_CACHE[key] = False
        return False
    import inspect
    extended = "probe" in inspect.signature(accelerator.is_triton_supported).parameters
    if probe_fn is None:
        # A static per-kernel declaration with no probe: the accelerator
        # answers for the kernel name alone and nothing runs. Used for
        # probe-unsafe paths, whose answer must not depend on capture state.
        # Platforms whose gate predates the extension keep their backend-level
        # answer, already checked above.
        supported = accelerator.is_triton_supported(kernel) if extended else True
        _TRITON_PROBE_CACHE[key] = supported
        return supported
    if not extended:
        # Platform whose gate predates the per-kernel extension: its static
        # backend-level answer stands.
        _TRITON_PROBE_CACHE[key] = True
        return True
    if accelerator.is_current_stream_capturing():
        # Launching a probe while the stream is being captured would break the
        # capture, and a capture-time failure would poison the graph; defer the
        # probe to the first non-capturing call. Deliberately not cached: the
        # probe still has to run once outside capture.
        return True
    try:
        probe = probe_fn(*probe_args)
    except Exception:
        # The probe library cannot even build its minimal case here; fail closed
        # rather than open the gate on a broken probe.
        _TRITON_PROBE_CACHE[key] = False
        return False
    supported = accelerator.is_triton_supported(kernel=f"{kernel}:{device.type}:{device.index}", probe=probe)
    _TRITON_PROBE_CACHE[key] = supported
    return supported


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
    per feature: the steady state is one dictionary lookup. Kernel probes must
    import their classes through this resolver so the gate exercises the
    platform's variant, not the portable kernel.
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
    change (validation, the runtime gate, the eager fallback) are factored into
    helpers of the portable module that the variant imports.
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
