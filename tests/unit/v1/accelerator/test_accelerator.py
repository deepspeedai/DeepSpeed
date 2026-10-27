# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

import os
import sys
import importlib
import re

import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.accelerator.abstract_accelerator import DeepSpeedAccelerator

DS_ACCEL_PATH = "deepspeed.accelerator"
IGNORE_FILES = ["abstract_accelerator.py", "real_accelerator.py"]


@pytest.fixture
def accel_class_name(module_name):
    class_list = []
    mocked_modules = []

    # Get the accelerator class name for a given module
    while True:
        try:
            module = importlib.import_module(module_name)
            break
        except ModuleNotFoundError as e:
            # If the environment is missing a module, mock it so we can still
            # test importing the accelerator class
            missing_module = re.search(r"\'(.*)\'", e.msg).group().strip("'")
            sys.modules[missing_module] = lambda x: None
            mocked_modules.append(missing_module)
    for name in dir(module):
        if name.endswith("_Accelerator"):
            class_list.append(name)

    assert len(class_list) == 1, f"Multiple accelerator classes found in {module_name}"

    yield class_list[0]

    # Clean up mocked modules so as to not impact other tests
    for module in mocked_modules:
        del sys.modules[module]


@pytest.mark.parametrize(
    "module_name",
    [
        DS_ACCEL_PATH + "." + f.rstrip(".py") for f in os.listdir(deepspeed.accelerator.__path__[0])
        if f.endswith("_accelerator.py") and f not in IGNORE_FILES
    ],
)
def test_abstract_methods_defined(module_name, accel_class_name):
    module = importlib.import_module(module_name)
    accel_class = getattr(module, accel_class_name)
    # Stub __init__ so the class can be instantiated without a real device, but
    # restore it afterwards to avoid polluting the shared class object for other
    # tests that construct a fully-initialized accelerator.
    original_init = accel_class.__init__
    accel_class.__init__ = lambda self: None
    try:
        _ = accel_class()
    finally:
        accel_class.__init__ = original_init


def _require_native(monkeypatch):
    # Backend selection is env-driven and evaluated per call, so set native for
    # the whole test and skip if the pin_memory op cannot be built (constructing
    # the native manager raises in that case).
    from deepspeed.utils.pin_memory import get_active_native_pinned_memory
    monkeypatch.setenv("DS_PIN_MEMORY_BACKEND", "native")
    try:
        if get_active_native_pinned_memory() is None:
            pytest.skip("native backend not selected")
    except Exception:
        pytest.skip("pin_memory op could not be built; native pinning unavailable")


def test_pin_memory_torch_backend(monkeypatch):
    monkeypatch.delenv("DS_PIN_MEMORY_BACKEND", raising=False)
    accel = get_accelerator()

    tensor = torch.randn(4, 8)
    pinned = accel.pin_memory(tensor)
    # is_pinned must agree with torch for the torch backend.
    assert accel.is_pinned(pinned) == pinned.is_pinned()
    # Unpinning a torch-pinned tensor is a no-op (freed on garbage collection).
    assert accel.unpin_memory(pinned) is None


def test_pin_memory_torch_backend_make_copy_false_allocates_pinned(monkeypatch):
    """make_copy=False must page-lock a new buffer instead of pinning a copy."""
    monkeypatch.delenv("DS_PIN_MEMORY_BACKEND", raising=False)
    calls = []

    class _StubAccelerator:
        # Exercise the shared dispatch without needing a real pinning device.
        pin_memory = DeepSpeedAccelerator.pin_memory
        _pin_fresh = DeepSpeedAccelerator._pin_fresh
        _shape_numel = staticmethod(DeepSpeedAccelerator._shape_numel)
        _torch_pins_host_memory = True

        def _torch_pin_memory(self, tensor):
            calls.append(("pin_copy", tuple(tensor.shape)))
            return tensor

        def _torch_empty_pinned(self, tensor, shape):
            calls.append(("empty_pinned", tuple(shape)))
            return tensor.new_empty(shape)

    accel = _StubAccelerator()
    tensor = torch.randn(4, 8)

    accel.pin_memory(tensor)
    assert tuple(accel.pin_memory(tensor, make_copy=False).shape) == (4, 8)
    assert tuple(accel.pin_memory(tensor, make_copy=False, match_shape=False).shape) == (32, )
    assert calls == [("pin_copy", (4, 8)), ("empty_pinned", (4, 8)), ("empty_pinned", (32, ))]


def test_pin_empty_torch_backend_uses_zero_element_template(monkeypatch):
    """pin_empty must not build a full-size pageable tensor for the torch path."""
    monkeypatch.delenv("DS_PIN_MEMORY_BACKEND", raising=False)
    calls = []

    class _StubAccelerator:
        pin_empty = DeepSpeedAccelerator.pin_empty
        pin_empty_like = DeepSpeedAccelerator.pin_empty_like
        _pin_fresh = DeepSpeedAccelerator._pin_fresh
        _require_host_device = staticmethod(DeepSpeedAccelerator._require_host_device)
        _shape_numel = staticmethod(DeepSpeedAccelerator._shape_numel)
        _torch_pins_host_memory = True

        def _torch_empty_pinned(self, tensor, shape):
            calls.append((tuple(tensor.shape), tuple(shape), tensor.dtype, tensor.numel()))
            return tensor.new_empty(shape)

    accel = _StubAccelerator()
    buffer = accel.pin_empty(4, 8, dtype=torch.float32, device="cpu")
    assert tuple(buffer.shape) == (4, 8)
    src = torch.randn(2, 3, dtype=torch.float16)
    like = accel.pin_empty_like(src, device="cpu")
    assert tuple(like.shape) == (2, 3)
    assert calls == [((0, ), (4, 8), torch.float32, 0), ((0, ), (2, 3), torch.float16, 0)]


def test_pin_empty_requires_dtype_and_host_device(monkeypatch):
    monkeypatch.delenv("DS_PIN_MEMORY_BACKEND", raising=False)
    accel = get_accelerator()
    with pytest.raises(TypeError):
        accel.pin_empty(4, device="cpu")
    with pytest.raises(ValueError, match="cpu"):
        accel.pin_empty(4, dtype=torch.float32, device="cuda")


def test_pin_empty_cpu_torch_skips_accounting(monkeypatch):
    """CPU torch pinning is a no-op and must not inflate the pinned-byte total."""
    monkeypatch.delenv("DS_PIN_MEMORY_BACKEND", raising=False)
    from deepspeed.accelerator.cpu_accelerator import CPU_Accelerator
    from deepspeed.utils import pin_memory_tracker

    accel = CPU_Accelerator()
    before = pin_memory_tracker._tracker._bytes
    buffer = accel.pin_empty(8, dtype=torch.float32, device="cpu")
    assert tuple(buffer.shape) == (8, )
    assert accel.is_pinned(buffer) is False
    assert pin_memory_tracker._tracker._bytes == before


def test_pin_empty_torch_backend_is_pinned(monkeypatch):
    """pin_empty page-locks a host buffer on accelerators that can pin."""
    monkeypatch.delenv("DS_PIN_MEMORY_BACKEND", raising=False)
    accel = get_accelerator()
    if accel.device_name() == "cpu":
        pytest.skip("torch cannot pin CPU tensors on the CPU accelerator")

    buffer = accel.pin_empty(4, 8, dtype=torch.float32, device="cpu")
    assert buffer.is_pinned()
    assert tuple(buffer.shape) == (4, 8)
    src = torch.randn(2, 3)
    like = accel.pin_empty_like(src, device="cpu")
    assert like.is_pinned()
    assert tuple(like.shape) == (2, 3)


def test_pin_empty_native_backend(monkeypatch):
    _require_native(monkeypatch)
    accel = get_accelerator()
    buffer = accel.pin_empty(4, 8, dtype=torch.float32, device="cpu")
    assert tuple(buffer.shape) == (4, 8)
    assert accel.is_pinned(buffer)
    src = torch.randn(2, 3)
    like = accel.pin_empty_like(src, device="cpu")
    assert tuple(like.shape) == (2, 3)
    assert accel.is_pinned(like)
    assert accel.unpin_memory(buffer) is True
    assert accel.unpin_memory(like) is True


def test_pin_memory_torch_backend_no_copy_is_pinned(monkeypatch):
    """The buffer returned for make_copy=False is really page-locked."""
    monkeypatch.delenv("DS_PIN_MEMORY_BACKEND", raising=False)
    accel = get_accelerator()
    if accel.device_name() == "cpu":
        pytest.skip("torch cannot pin CPU tensors on the CPU accelerator")

    tensor = torch.randn(4, 8)
    buffer = accel.pin_memory(tensor, make_copy=False)
    assert buffer.is_pinned()
    assert tuple(buffer.shape) == (4, 8)
    assert buffer.data_ptr() != tensor.data_ptr()


def test_pin_memory_native_backend(monkeypatch):
    _require_native(monkeypatch)
    accel = get_accelerator()

    tensor = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    # Default: copy data and match the input shape, tag with .ds_pinned.
    pinned = accel.pin_memory(tensor)
    assert tuple(pinned.shape) == (4, 8)
    assert torch.equal(pinned, tensor)
    assert getattr(pinned, "ds_pinned", False) is True
    assert accel.is_pinned(pinned)

    # Pinned status propagates to slices/views via pointer-range tracking even
    # though the .ds_pinned attribute does not survive tensor ops.
    view = pinned.reshape(-1).narrow(0, 8, 8)
    assert getattr(view, "ds_pinned", False) is False
    assert accel.is_pinned(view)
    assert accel.is_pinned(pinned[1])

    # make_copy=False -> shaped allocation; match_shape=False -> flat buffer.
    alloc = accel.pin_memory(tensor, make_copy=False)
    assert tuple(alloc.shape) == (4, 8)
    assert accel.is_pinned(alloc)
    flat = accel.pin_memory(tensor, match_shape=False)
    assert tuple(flat.shape) == (tensor.numel(), )

    # Unpinning frees the native allocation and flips is_pinned for the buffer
    # and its derived views.
    assert accel.unpin_memory(pinned) is True
    assert not accel.is_pinned(pinned)
    assert not accel.is_pinned(view)


def test_pin_memory_native_raises_when_pin_memory_unavailable(monkeypatch):
    import deepspeed.ops.op_builder as op_builder
    import deepspeed.utils.pin_memory as pin_memory_util

    class _FailingBuilder:

        def load(self):
            raise RuntimeError("simulated pin_memory build failure")

    # Reset the shared manager so it is rebuilt with the failing builder below;
    # monkeypatch restores the original instance after the test.
    monkeypatch.setenv("DS_PIN_MEMORY_BACKEND", "native")
    monkeypatch.setattr(pin_memory_util, "_shared_native_pins", None)
    monkeypatch.setattr(op_builder, "PinMemoryBuilder", _FailingBuilder)

    # Native is selected but the pin_memory op cannot be built, so we fail early
    # (no silent torch fallback).
    with pytest.raises(RuntimeError, match="pin_memory"):
        pin_memory_util.get_active_native_pinned_memory()


def test_is_pinned_handles_storageless_tensors(monkeypatch):
    _require_native(monkeypatch)
    accel = get_accelerator()

    from torch._subclasses.fake_tensor import FakeTensorMode

    # FakeTensor/meta tensors have no real storage; is_pinned must not access
    # data_ptr() on them.
    with FakeTensorMode() as fake_mode:
        fake_tensor = fake_mode.from_tensor(torch.zeros(2, 8))
        assert accel.is_pinned(fake_tensor) is False

    meta_tensor = torch.zeros(2, 8, device="meta")
    assert accel.is_pinned(meta_tensor) is False


GiB = 1 << 30


@pytest.fixture
def cpu_cgroup(monkeypatch, tmp_path):
    """Fake cgroup tree and /proc/self/cgroup for the CPU accelerator's memory queries."""
    from deepspeed.accelerator import cpu_accelerator as cpu_module
    root = tmp_path / "cgroup"
    proc_self_cgroup = tmp_path / "proc_self_cgroup"
    monkeypatch.setattr(cpu_module, "_CGROUP_ROOT", str(root))
    monkeypatch.setattr(cpu_module, "_PROC_SELF_CGROUP", str(proc_self_cgroup))
    # No mountinfo file: the hierarchy is assumed at _CGROUP_ROOT, mounted at "/".
    monkeypatch.setattr(cpu_module, "_PROC_SELF_MOUNTINFO", str(tmp_path / "no_mountinfo"))
    monkeypatch.setattr(cpu_module.sys, "platform", "linux")
    monkeypatch.setenv("DS_CPU_RESPECT_CGROUP_MEMORY", "1")

    def write(cgroup_line, files):
        proc_self_cgroup.write_text(cgroup_line + "\n")
        for rel_path, content in files.items():
            (root / rel_path).parent.mkdir(parents=True, exist_ok=True)
            (root / rel_path).write_text(content + "\n")
        return cpu_module.CPU_Accelerator()

    return write


def test_cpu_memory_uses_limit_on_parent_cgroup(cpu_cgroup):
    """Kubernetes sets the pod limit on the parent cgroup; the container's own memory.max is 'max'."""
    import psutil
    accel = cpu_cgroup(
        "0::/pod/ctr", {
            "memory.max": "max",
            "pod/memory.max": str(GiB),
            "pod/memory.current": str(GiB // 2),
            "pod/memory.stat": f"anon {GiB // 4}\ninactive_file {GiB // 4}",
            "pod/ctr/memory.max": "max",
        })
    assert accel.total_memory() == min(GiB, psutil.virtual_memory().total)
    # 1 GiB limit - (512 MiB usage - 256 MiB reclaimable page cache)
    assert accel.available_memory() == min(3 * GiB // 4, psutil.virtual_memory().available)


def test_cpu_memory_smallest_limit_and_v1_sentinel(cpu_cgroup):
    accel = cpu_cgroup(
        "12:memory:/pod\n3:cpu,cpuacct:/pod", {
            "memory/memory.limit_in_bytes": "9223372036854771712",
            "memory/pod/memory.limit_in_bytes": str(GiB),
            "memory/pod/memory.usage_in_bytes": "0",
        })
    import psutil
    assert accel.total_memory() == min(GiB, psutil.virtual_memory().total)


@pytest.mark.parametrize("default, override, capped", [(False, None, False), (True, None, True), (False, "1", True),
                                                       (True, "0", False)])
def test_cpu_memory_cgroup_default_and_override(cpu_cgroup, monkeypatch, default, override, capped):
    """RESPECT_CGROUP_MEMORY_BY_DEFAULT is the one line that flips the behavior; the env var overrides it."""
    import psutil
    from deepspeed.accelerator import cpu_accelerator as cpu_module
    monkeypatch.setattr(cpu_module, "RESPECT_CGROUP_MEMORY_BY_DEFAULT", default)
    if override is None:
        monkeypatch.delenv("DS_CPU_RESPECT_CGROUP_MEMORY", raising=False)
    else:
        monkeypatch.setenv("DS_CPU_RESPECT_CGROUP_MEMORY", override)
    accel = cpu_cgroup("0::/", {"memory.max": str(GiB), "memory.current": "0"})
    expected = min(GiB, psutil.virtual_memory().total) if capped else psutil.virtual_memory().total
    assert accel.total_memory() == expected


def test_cpu_memory_checks_headroom_of_every_limited_ancestor(cpu_cgroup):
    """A parent limit also counts its other children: an empty 1 GiB child under a 2 GiB parent whose siblings use
    1.75 GiB can still take only 0.25 GiB."""
    import psutil
    accel = cpu_cgroup(
        "0::/pod/ctr", {
            "pod/memory.max": str(2 * GiB),
            "pod/memory.current": str(7 * GiB // 4),
            "pod/ctr/memory.max": str(GiB),
            "pod/ctr/memory.current": "0",
        })
    assert accel.total_memory() == min(GiB, psutil.virtual_memory().total)
    assert accel.available_memory() == min(GiB // 4, psutil.virtual_memory().available)


@pytest.mark.parametrize("mount_line, cgroup_line, limit_file", [
    ("30 25 0:27 /docker/abc {mount} rw - cgroup2 cgroup2 rw", "0::/docker/abc/job", "job/memory.max"),
    ("31 25 0:28 / {mount} rw - cgroup cgroup rw,cpu,memory", "5:cpu,memory:/pod", "pod/memory.limit_in_bytes"),
])
def test_cpu_memory_resolves_cgroup_mount_from_mountinfo(cpu_cgroup, monkeypatch, tmp_path, mount_line, cgroup_line,
                                                         limit_file):
    """The mount can expose a subtree of the hierarchy, and cgroup v1 can co-mount controllers as "cpu,memory"."""
    import psutil
    from deepspeed.accelerator import cpu_accelerator as cpu_module
    mount = tmp_path / "mnt"
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(mount_line.format(mount=mount) + "\n")
    monkeypatch.setattr(cpu_module, "_PROC_SELF_MOUNTINFO", str(mountinfo))
    (mount / limit_file).parent.mkdir(parents=True, exist_ok=True)
    (mount / limit_file).write_text(str(GiB) + "\n")
    accel = cpu_cgroup(cgroup_line, {})
    assert accel.total_memory() == min(GiB, psutil.virtual_memory().total)


def test_cpu_accelerator_exports_cgroup_memory_switch():
    """The launcher forwards env vars listed by export_envs() to every node."""
    from deepspeed.accelerator.cpu_accelerator import CPU_Accelerator
    assert "DS_CPU_RESPECT_CGROUP_MEMORY" in CPU_Accelerator().export_envs()
