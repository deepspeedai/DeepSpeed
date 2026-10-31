# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import ctypes
import torch
from deepspeed.accelerator.abstract_accelerator import DeepSpeedAccelerator
import functools
import importlib
import inspect

try:
    import oneccl_bindings_for_pytorch  # noqa: F401 # type: ignore
    oneccl_imported_p = True
except ImportError as e:
    oneccl_imported_p = False

# Host-memory registration on XPU goes through Level Zero's
# ZE_extension_external_memmap_sysmem extension: zeMemAllocHost with an
# ze_external_memmap_sysmem_ext_desc_t chained onto the host-alloc descriptor
# maps an existing page-locked host buffer into the device page tables (the
# returned pointer equals the input pointer), and zeMemFree releases the
# mapping without touching the host memory itself. This is the XPU equivalent
# of cudaHostRegister/cudaHostUnregister.
#
# Values from ze_api.h (stable ABI): structure types, result codes, and the
# descriptor layouts the loader dispatches on.
_ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0xD
_ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = 0x16
_ZE_STRUCTURE_TYPE_EXTERNAL_MEMMAP_SYSMEM_EXT_DESC = 0x00020037
_ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x4
_ZE_MAX_EXTENSION_NAME = 256
_ZE_RESULT_SUCCESS = 0x0
_ZE_INIT_FLAG_GPU_ONLY = 0x1


class _ZeContextDesc(ctypes.Structure):

    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32)]


class _ZeExternalMemmapSysmemDesc(ctypes.Structure):

    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("pSystemMemory", ctypes.c_void_p),
                ("size", ctypes.c_uint64)]


class _ZeHostMemAllocDesc(ctypes.Structure):

    _fields_ = [("stype", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32)]


class _ZeDriverExtensionProperties(ctypes.Structure):

    _fields_ = [("name", ctypes.c_char * _ZE_MAX_EXTENSION_NAME), ("version", ctypes.c_uint32)]


class _ZeDeviceProperties(ctypes.Structure):
    # The uuid sits at offset 96 and the driver writes the full 368-byte
    # struct, so keep the leading bytes as an opaque blob and pad the tail.
    _fields_ = [("stype", ctypes.c_uint32), ("opaque", ctypes.c_uint8 * 92), ("uuid", ctypes.c_uint8 * 16),
                ("pad", ctypes.c_uint8 * 256)]


@functools.lru_cache(maxsize=None)
def _l0_loader():
    """(loader, driver) with the registration argtypes wired, or None when unsupported."""
    try:
        ze = ctypes.CDLL("libze_loader.so.1")
    except OSError:
        return None

    ze.zeInit.argtypes = [ctypes.c_uint32]
    ze.zeInit.restype = ctypes.c_uint32
    if ze.zeInit(_ZE_INIT_FLAG_GPU_ONLY) != _ZE_RESULT_SUCCESS:
        return None

    ze.zeDriverGet.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
    ze.zeDriverGet.restype = ctypes.c_uint32
    count = ctypes.c_uint32(0)
    if ze.zeDriverGet(ctypes.byref(count), None) != _ZE_RESULT_SUCCESS or count.value == 0:
        return None
    drivers = (ctypes.c_void_p * count.value)()
    if ze.zeDriverGet(ctypes.byref(count), drivers) != _ZE_RESULT_SUCCESS:
        return None
    driver = drivers[0]

    # zeMemAllocHost-with-descriptor is an optional extension, so probe before
    # the first registration instead of failing per buffer.
    ze.zeDriverGetExtensionProperties.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(_ZeDriverExtensionProperties)
    ]
    ze.zeDriverGetExtensionProperties.restype = ctypes.c_uint32
    if ze.zeDriverGetExtensionProperties(driver, ctypes.byref(count), None) != _ZE_RESULT_SUCCESS:
        return None
    extensions = (_ZeDriverExtensionProperties * count.value)()
    if ze.zeDriverGetExtensionProperties(driver, ctypes.byref(count), extensions) != _ZE_RESULT_SUCCESS:
        return None
    names = [extensions[i].name.decode() for i in range(count.value)]
    if "ZE_extension_external_memmap_sysmem" not in names:
        return None

    ze.zeDeviceGet.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
    ze.zeDeviceGet.restype = ctypes.c_uint32
    ze.zeDeviceGetProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ZeDeviceProperties)]
    ze.zeDeviceGetProperties.restype = ctypes.c_uint32
    ze.zeContextCreateEx.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ZeContextDesc), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p)
    ]
    ze.zeContextCreateEx.restype = ctypes.c_uint32
    ze.zeMemAllocHost.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ZeHostMemAllocDesc), ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p)
    ]
    ze.zeMemAllocHost.restype = ctypes.c_uint32
    ze.zeMemFree.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    ze.zeMemFree.restype = ctypes.c_uint32
    return ze, driver


@functools.lru_cache(maxsize=None)
def _l0_device_context(device_index):
    """(loader, context) whose only device is the L0 device matching torch device ``device_index``.

    Registration must happen in a context restricted to the target device: in a
    multi-device context the driver only maps the range for the context's first
    device, and device-initiated copies from any other device of that context
    fail with ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY. Device page tables are
    shared across the driver's contexts, so the mapping is then usable by any
    context (including torch's runtime) on that device.
    """
    base = _l0_loader()
    if base is None:
        return None
    ze, driver = base

    count = ctypes.c_uint32(0)
    if ze.zeDeviceGet(driver, ctypes.byref(count), None) != _ZE_RESULT_SUCCESS:
        return None
    devices = (ctypes.c_void_p * count.value)()
    if ze.zeDeviceGet(driver, ctypes.byref(count), devices) != _ZE_RESULT_SUCCESS:
        return None

    # zeDeviceGet order is not guaranteed to match torch's device order, so
    # match on the device UUID instead of assuming positional identity.
    torch_uuid = str(torch.xpu.get_device_properties(device_index).uuid)
    for device in devices:
        props = _ZeDeviceProperties(stype=_ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES)
        if ze.zeDeviceGetProperties(device, ctypes.byref(props)) != _ZE_RESULT_SUCCESS:
            continue
        raw = bytes(props.uuid)
        l0_uuid = (f"{raw[0:4].hex()}-{raw[4:6].hex()}-{raw[6:8].hex()}-"
                   f"{raw[8:10].hex()}-{raw[10:16].hex()}")
        if l0_uuid != torch_uuid:
            continue
        device_ref = ctypes.c_void_p(device)
        context = ctypes.c_void_p()
        desc = _ZeContextDesc(stype=_ZE_STRUCTURE_TYPE_CONTEXT_DESC, pNext=None, flags=0)
        if ze.zeContextCreateEx(driver, ctypes.byref(desc), 1, ctypes.byref(device_ref),
                                ctypes.byref(context)) != _ZE_RESULT_SUCCESS:
            return None
        return ze, context
    return None


# Registrations are freed in the (per-device) context that created them.
_l0_registrations = {}


class XPU_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'xpu'
        if oneccl_imported_p:
            self._communication_backend_name = 'ccl'
        else:
            # changed to xccl if not using torch-CCL on XPU device
            self._communication_backend_name = 'xccl'
        self._compile_backend = "inductor"
        self.class_dict = None

    def is_synchronized_device(self):
        return False

    def use_host_timers(self):
        return self.is_synchronized_device()

    def resolves_data_dependency(self):
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None):
        if device_index == None:
            return 'xpu'
        return 'xpu:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.device('xpu', device_index)

    def set_device(self, device_index):
        torch.xpu.set_device(device_index)

    def current_device(self):
        return torch.xpu.current_device()

    def current_device_name(self):
        return 'xpu:{}'.format(torch.xpu.current_device())

    def device_count(self):
        return torch.xpu.device_count()

    def synchronize(self, device_index=None):
        return torch.xpu.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.xpu.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index == None:
            return torch.xpu.set_rng_state(new_state)
        return torch.xpu.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        if device_index == None:
            return torch.xpu.get_rng_state()
        return torch.xpu.get_rng_state(device_index)

    def manual_seed(self, seed):
        return torch.xpu.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.xpu.manual_seed_all(seed)

    def initial_seed(self):
        return torch.xpu.initial_seed()

    def default_generator(self, device_index):
        return torch.xpu.default_generators[device_index]

    # Streams/Events
    @property
    def Stream(self):
        return torch.xpu.Stream

    def stream(self, stream):
        return torch.xpu.stream(stream)

    def current_stream(self, device_index=None):
        return torch.xpu.current_stream(device_index)

    def default_stream(self, device_index=None):
        # torch.xpu does not support the sync behavior of default stream as cuda
        # use current_stream as workaround
        # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
        return torch.xpu.current_stream(device_index)

    @property
    def Event(self):
        return torch.xpu.Event

    # Memory management
    def empty_cache(self):
        return torch.xpu.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.xpu.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.xpu.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.xpu.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch.xpu.memory_reserved(device_index)

    def max_memory_cached(self, device_index=None):
        return torch.xpu.max_memory_reserved(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch.xpu.reset_max_memory_reserved(device_index)

    def memory_stats(self, device_index=None):
        return torch.xpu.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        return torch.xpu.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        return torch.xpu.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        return torch.xpu.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.xpu.get_device_properties(device_index).total_memory

    def available_memory(self, device_index=None):
        return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Misc
    def is_available(self):
        return torch.xpu.is_available()

    def range_push(self, msg, domain=None, category=None):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_push(msg)
        return

    def range_pop(self, domain=None):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_pop()
        return

    def lazy_call(self, callback):
        if hasattr(torch.xpu, "_lazy_call"):
            return torch.xpu._lazy_call(callback)
        else:
            return torch.xpu.lazy_init._lazy_call(callback)

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        return False

    # Graph operations
    def create_graph(self):
        return None

    def capture_to_graph(self, graph, pool=None, stream=None):
        from deepspeed.runtime.utils import noop_context
        return noop_context()

    def replay_graph(self, graph):
        return

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        return True

    def supported_dtypes(self):
        return [torch.float, torch.half, torch.bfloat16]

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return functools.partial(torch.tensor, dtype=torch.bfloat16, device=self._name)

    @property
    def ByteTensor(self):
        return functools.partial(torch.tensor, dtype=torch.uint8, device=self._name)

    @property
    def DoubleTensor(self):
        return functools.partial(torch.tensor, dtype=torch.double, device=self._name)

    @property
    def FloatTensor(self):
        return functools.partial(torch.tensor, dtype=torch.float, device=self._name)

    @property
    def HalfTensor(self):
        return functools.partial(torch.tensor, dtype=torch.half, device=self._name)

    @property
    def IntTensor(self):
        return functools.partial(torch.tensor, dtype=torch.int, device=self._name)

    @property
    def LongTensor(self):
        return functools.partial(torch.tensor, dtype=torch.long, device=self._name)

    def _torch_pin_memory(self, tensor):
        return tensor.pin_memory(device=self.current_device_name())

    def _torch_empty_pinned(self, tensor, shape):
        # Pinning on XPU needs an explicit device, which the allocation API
        # cannot express, so allocate and then pin.
        return self._torch_pin_memory(tensor.new_empty(shape))

    def _torch_is_pinned(self, tensor):
        return tensor.is_pinned(device=self.current_device_name())

    def register_host_memory(self, address, num_bytes):
        handles = _l0_device_context(self.current_device())
        if handles is None:
            from deepspeed.utils import logger
            logger.warning_once(
                "Level Zero host-memory registration is unavailable (libze_loader.so.1 or the "
                "ZE_extension_external_memmap_sysmem extension is missing); native pinned memory stays mlock-only.")
            return False
        ze, context = handles
        memmap_desc = _ZeExternalMemmapSysmemDesc(stype=_ZE_STRUCTURE_TYPE_EXTERNAL_MEMMAP_SYSMEM_EXT_DESC,
                                                  pNext=None,
                                                  pSystemMemory=address,
                                                  size=num_bytes)
        host_desc = _ZeHostMemAllocDesc(stype=_ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                        pNext=ctypes.cast(ctypes.byref(memmap_desc), ctypes.c_void_p),
                                        flags=0)
        mapped = ctypes.c_void_p()
        rc = ze.zeMemAllocHost(context, ctypes.byref(host_desc), num_bytes, 0, ctypes.byref(mapped))
        from deepspeed.utils import logger
        if rc != _ZE_RESULT_SUCCESS:
            logger.warning_once(
                "zeMemAllocHost failed with Level Zero result 0x%x; "
                "falling back to mlock-only pinned memory.", rc)
            return False
        if mapped.value != address:
            # The spec guarantees the mapping preserves the virtual address; a
            # non-identity mapping is unusable (callers hold the original
            # pointer), so release it and report failure so they fall back to mlock.
            ze.zeMemFree(context, mapped.value)
            logger.warning_once(
                "Level Zero returned a non-identity mapping for host registration "
                "(%#x != %#x); falling back to mlock-only pinned memory.", mapped.value, address)
            return False
        _l0_registrations[address] = (ze, context)
        return True

    def unregister_host_memory(self, address):
        record = _l0_registrations.pop(address, None)
        if record is None:
            return None
        ze, context = record
        rc = ze.zeMemFree(context, address)
        if rc != _ZE_RESULT_SUCCESS:
            # Raise (like the CUDA path's check_error) so NativePinnedMemory.unpin
            # keeps the CPU allocation alive instead of freeing pages the driver
            # may still have mapped.
            raise RuntimeError(
                f"zeMemFree failed with Level Zero result 0x{rc:x} while unregistering host memory at {address:#x}")

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder.xpu"
        except ImportError:
            return "deepspeed.ops.op_builder.xpu"

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('xpu:'):
            return True
        else:
            return False

    def _lazy_init_class_dict(self):
        if self.class_dict:
            return

        op_builder_module = importlib.import_module(self.op_builder_dir())

        # get op builder class from op_builder/xpu/__init__.py
        self.class_dict = {}
        for class_name, class_obj in inspect.getmembers(op_builder_module, inspect.isclass):
            self.class_dict[class_name] = class_obj

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, class_name):
        builder_class = self.get_op_builder(class_name)
        return builder_class()

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]
        else:
            return self.class_dict['NotImplementedBuilder']

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return []

    def visible_devices_envs(self):
        return ['ZE_AFFINITY_MASK']

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
