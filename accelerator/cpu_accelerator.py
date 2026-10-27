# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .abstract_accelerator import DeepSpeedAccelerator

# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch
except ImportError as e:
    pass

try:
    import oneccl_bindings_for_pytorch  # noqa: F401 # type: ignore
    oneccl_imported_p = True
except ImportError as e:
    oneccl_imported_p = False

import os
import sys

# Default for capping total_memory() and available_memory() at the cgroup memory limit of the process.
# Flip this one line to change the default; DS_CPU_RESPECT_CGROUP_MEMORY=1 or =0 overrides it.
RESPECT_CGROUP_MEMORY_BY_DEFAULT = False

_CGROUP_ROOT = "/sys/fs/cgroup"
_PROC_SELF_CGROUP = "/proc/self/cgroup"
_PROC_SELF_MOUNTINFO = "/proc/self/mountinfo"
# cgroup v1 reports "no limit" as a huge sentinel instead of "max".
_CGROUP_UNLIMITED = 1 << 62


def _read_cgroup_int(path):
    try:
        with open(path) as f:
            value = f.read().strip()
    except OSError:
        return None
    return int(value) if value.isdigit() else None


def _read_memory_stat(path, key):
    try:
        with open(path) as f:
            for line in f:
                name, _, value = line.partition(" ")
                if name == key:
                    return int(value)
    except (OSError, ValueError):
        pass
    return 0


def _path_parts(path):
    return [part for part in path.split("/") if part]


def _cgroup_membership(controller):
    """The cgroup path of this process from /proc/self/cgroup, relative to the hierarchy root."""
    try:
        with open(_PROC_SELF_CGROUP) as f:
            for line in f:
                parts = line.strip().split(":", 2)
                if len(parts) != 3:
                    continue
                is_v2 = controller is None and parts[0] == "0" and parts[1] == ""
                is_v1 = controller is not None and controller in parts[1].split(",")
                if is_v2 or is_v1:
                    return parts[2]
    except OSError:
        pass
    return ""


def _cgroup_mount(controller, membership):
    """Returns (mount point, mount root) of the cgroup hierarchy from /proc/self/mountinfo.

    The mount can expose a subtree of the hierarchy (mount root other than "/"), cgroup v1 can co-mount controllers
    under a name such as "cpu,memory", and several mounts can match, e.g. a delegated subtree mounted before the main
    one. The mount whose root contains the membership is used; among those, the shortest root, because it
    exposes the most ancestors and an ancestor limit still applies.
    """
    candidates = []
    try:
        with open(_PROC_SELF_MOUNTINFO) as f:
            for line in f:
                before, separator, after = line.partition(" - ")
                mount_fields = before.split()
                fs_fields = after.split()
                if not separator or len(mount_fields) < 5 or len(fs_fields) < 3:
                    continue
                fs_type = fs_fields[0]
                super_options = fs_fields[2].split(",")
                is_v2 = controller is None and fs_type == "cgroup2"
                is_v1 = controller is not None and fs_type == "cgroup" and controller in super_options
                if is_v2 or is_v1:
                    # mountinfo escapes spaces as \040.
                    candidates.append((mount_fields[4].replace("\\040", " "), mount_fields[3].replace("\\040", " ")))
    except OSError:
        pass
    member_parts = _path_parts(membership)
    containing = [(mount_point, mount_root) for mount_point, mount_root in candidates
                  if member_parts[:len(_path_parts(mount_root))] == _path_parts(mount_root)]
    if containing:
        return min(containing, key=lambda mount: len(_path_parts(mount[1])))
    if candidates:
        return candidates[0]
    default_mount = _CGROUP_ROOT if controller is None else os.path.join(_CGROUP_ROOT, controller)
    return default_mount, "/"


def _cgroup_dirs(controller):
    """Directories of this process's cgroup, nearest first, mount point last."""
    membership = _cgroup_membership(controller)
    mount_point, mount_root = _cgroup_mount(controller, membership)
    components = _path_parts(membership)
    root_components = _path_parts(mount_root)
    if ".." in components:
        # A cgroup outside this cgroup namespace cannot be resolved from here.
        components = []
    elif components[:len(root_components)] == root_components:
        # /proc/self/cgroup is relative to the hierarchy root; the mount starts at mount_root.
        components = components[len(root_components):]
    dirs = [os.path.join(mount_point, *components[:depth]) for depth in range(len(components), 0, -1)]
    return dirs + [mount_point]


def _cgroup_memory():
    """Returns (limit, available) in bytes for the cgroups that limit this process, or None.

    The limit is the smallest one from the process's cgroup up to the root. Available memory is the smallest headroom
    over every limited level, because a parent's limit also counts the usage of its other children. Usage excludes the
    inactive page cache, which the kernel can reclaim. Available is None when no limited level exposes its usage.
    """
    override = os.environ.get("DS_CPU_RESPECT_CGROUP_MEMORY")
    enabled = RESPECT_CGROUP_MEMORY_BY_DEFAULT if override is None else override.strip().lower() in ("1", "true")
    if not enabled or not sys.platform.startswith("linux"):
        return None
    for controller, limit_file, usage_file, inactive_key in (
        (None, "memory.max", "memory.current", "inactive_file"),
        ("memory", "memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file"),
    ):
        limit = None
        available = None
        for cgroup_dir in _cgroup_dirs(controller):
            level_limit = _read_cgroup_int(os.path.join(cgroup_dir, limit_file))
            if level_limit is None or level_limit >= _CGROUP_UNLIMITED:
                continue
            limit = level_limit if limit is None else min(limit, level_limit)
            usage = _read_cgroup_int(os.path.join(cgroup_dir, usage_file))
            if usage is None:
                continue
            inactive_file = _read_memory_stat(os.path.join(cgroup_dir, "memory.stat"), inactive_key)
            headroom = max(0, level_limit - max(0, usage - inactive_file))
            available = headroom if available is None else min(available, headroom)
        if limit is not None:
            return limit, available
    return None


# accelerator for Intel CPU
class CPU_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'cpu'
        self._compile_backend = "inductor"
        if oneccl_imported_p:
            self._communication_backend_name = 'ccl'
        else:
            # fallback to gloo if oneccl_binding_for_pytorch is not installed
            self._communication_backend_name = 'gloo'
        try:
            import psutil
            mem = psutil.Process().memory_info().rss
            self.max_mem = mem
        except ImportError as e:
            self.max_mem = 0

    def is_synchronized_device(self):
        return True

    def use_host_timers(self):
        return self.is_synchronized_device()

    def resolves_data_dependency(self):
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None):
        return 'cpu'

    def device(self, device_index=None):
        return None

    def set_device(self, device_index):
        return

    def current_device(self):
        return os.environ.get('LOCAL_RANK', 0)

    def current_device_name(self):
        return 'cpu'

    def device_count(self):
        device_count = int(os.environ.get('LOCAL_SIZE', 0))
        if device_count > 0:
            return device_count
        else:
            from deepspeed.utils.numa import get_numa_cores
            # Count NUMA node for number of cpu accelerators. On machine with HBM
            # In flat mode, HBM is in separate NUMA node with no cores on this node.
            # Ignore these NUMA nodes with no cores.
            numa_core_lists = get_numa_cores()
            if not numa_core_lists:
                return 1
            numa_count = 0
            prev_core_list = []
            for core_list in numa_core_lists:
                if len(core_list) > 0 and core_list != prev_core_list:
                    numa_count += 1
                    prev_core_list = core_list
            return numa_count

    def synchronize(self, device_index=None):
        return

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.set_rng_state(new_state)
        return torch.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        return torch.get_rng_state()

    def manual_seed(self, seed):
        return torch.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.manual_seed(seed)

    def initial_seed(self):
        return torch.initial_seed()

    def default_generator(self, device_index):
        return torch.default_generator

    # Streams/Events
    @property
    def Stream(self):
        return None

    def stream(self, stream):
        from deepspeed.runtime.utils import noop_context
        return noop_context()

    def current_stream(self, device_index=None):
        return None

    def default_stream(self, device_index=None):
        return None

    @property
    def Event(self):
        return None

    # Memory management
    def empty_cache(self):
        return

    def get_rss(self):
        import psutil
        mem = psutil.Process().memory_info().rss
        if mem > self.max_mem:
            self.max_mem = mem
        return mem

    def reset_rss(self):
        import psutil
        mem = psutil.Process().memory_info().rss
        self.max_mem = mem
        return mem

    def memory_allocated(self, device_index=None):
        return self.get_rss()

    def max_memory_allocated(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def reset_max_memory_allocated(self, device_index=None):
        self.reset_rss()
        return

    def memory_cached(self, device_index=None):
        return self.get_rss()

    def max_memory_cached(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def reset_max_memory_cached(self, device_index=None):
        self.reset_rss()
        return

    def memory_stats(self, device_index=None):
        mem = self.get_rss()
        mem_stat = {}
        mem_stat['allocated_bytes.all.current'] = mem
        mem_stat['allocated_bytes.all.peak'] = self.max_mem
        return mem_stat

    def reset_peak_memory_stats(self, device_index=None):
        self.reset_rss()
        return

    def memory_reserved(self, device_index=None):
        return self.get_rss()

    def max_memory_reserved(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def total_memory(self, device_index=None):
        import psutil
        total = psutil.virtual_memory().total
        cgroup = _cgroup_memory()
        return total if cgroup is None else min(total, cgroup[0])

    def available_memory(self, device_index=None):
        import psutil
        available = psutil.virtual_memory().available
        cgroup = _cgroup_memory()
        if cgroup is None:
            return available
        limit, cgroup_available = cgroup
        return min(available, limit if cgroup_available is None else cgroup_available)

    # Misc
    def is_available(self):
        return True

    def range_push(self, msg, domain=None, category=None):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_push(msg)
        return

    def range_pop(self, domain=None):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_pop()
        return

    def lazy_call(self, callback):
        return callback()

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        return False

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        try:
            if torch.ops.mkldnn._is_mkldnn_fp16_supported():
                return True
        except Exception:
            return False

    def supported_dtypes(self):
        supported_dtypes = [torch.float, torch.bfloat16]
        if self.is_fp16_supported():
            supported_dtypes.append(torch.float16)
        return supported_dtypes

    # Graph operations
    def create_graph(self):
        return None

    def capture_to_graph(self, graph, pool=None, stream=None):
        from deepspeed.runtime.utils import noop_context
        return noop_context()

    def replay_graph(self, graph):
        return

    # Tensor operations
    @property
    def BFloat16Tensor(self):
        return torch.BFloat16Tensor

    @property
    def ByteTensor(self):
        return torch.ByteTensor

    @property
    def DoubleTensor(self):
        return torch.DoubleTensor

    @property
    def FloatTensor(self):
        return torch.FloatTensor

    @property
    def HalfTensor(self):
        return torch.HalfTensor

    @property
    def IntTensor(self):
        return torch.IntTensor

    @property
    def LongTensor(self):
        return torch.LongTensor

    _torch_pins_host_memory = False

    def pin_memory(self, tensor, make_copy=True, match_shape=True):
        # Torch cannot pin CPU tensors to a device; keep the historical no-op and
        # bypass ABC pinned-memory accounting (nothing is page-locked in that path).
        from deepspeed.utils.pin_memory import get_active_native_pinned_memory
        pins = get_active_native_pinned_memory()
        if pins is not None:
            from deepspeed.utils.pin_memory_tracker import track_pinned_memory
            track_pinned_memory(tensor.nbytes)
            return pins.pin(tensor, make_copy=make_copy, match_shape=match_shape)
        return tensor

    def _torch_pin_memory(self, tensor):
        # torch cannot pin CPU tensors to a device; keep the historical no-op.
        return tensor

    def _torch_empty_pinned(self, tensor, shape):
        # Same no-op as pin_memory: allocate host storage without page-locking.
        return tensor.new_empty(shape)

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder.cpu"
        except ImportError:
            return "deepspeed.ops.op_builder.cpu"

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('cpu'):
            return True
        else:
            return False

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, op_name):
        builder_class = self.get_op_builder(op_name)
        if builder_class is not None:
            return builder_class()
        return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            from op_builder.cpu import AsyncIOBuilder, CCLCommBuilder, ShareMemCommBuilder, FusedAdamBuilder, CPUAdamBuilder, CPUAdagradBuilder, CPULionBuilder, PinMemoryBuilder, NotImplementedBuilder
        except ImportError:
            from deepspeed.ops.op_builder.cpu import AsyncIOBuilder, CCLCommBuilder, ShareMemCommBuilder, FusedAdamBuilder, CPUAdamBuilder, CPUAdagradBuilder, CPULionBuilder, PinMemoryBuilder, NotImplementedBuilder

        if class_name == "CCLCommBuilder":
            return CCLCommBuilder
        elif class_name == "ShareMemCommBuilder":
            return ShareMemCommBuilder
        elif class_name == "FusedAdamBuilder":
            return FusedAdamBuilder
        elif class_name == "CPUAdamBuilder":
            return CPUAdamBuilder
        elif class_name == "CPUAdagradBuilder":
            return CPUAdagradBuilder
        elif class_name == "CPULionBuilder":
            return CPULionBuilder
        elif class_name == "AsyncIOBuilder":
            return AsyncIOBuilder
        elif class_name == "PinMemoryBuilder":
            return PinMemoryBuilder
        else:
            # return a NotImplementedBuilder to avoid get NoneType[Name] in unit tests
            return NotImplementedBuilder

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        # Forward the cgroup memory switch to every node started by the launcher.
        return ['DS_CPU_RESPECT_CGROUP_MEMORY']

    # TODO: cpu's visible envs is confirmed, keep as CUDA_VISIBLE_DEVICES
    def visible_devices_envs(self):
        return ['CUDA_VISIBLE_DEVICES']

    def set_visible_devices_envs(self, current_env, local_accelerator_ids):
        for env in self.visible_devices_envs():
            current_env[env] = ",".join(map(str, local_accelerator_ids))

    def get_compile_backend(self):
        return self._compile_backend

    def set_compile_backend(self, backend):
        supported_backends = torch._dynamo.list_backends(exclude_tags=())
        if backend in supported_backends:
            self._compile_backend = backend
        else:
            raise ValueError(
                f"{backend} not supported by {self.device_name()}. Supported Backends are {supported_backends}")
