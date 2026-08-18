# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import pkgutil
import importlib

import torch

from .abstract_accelerator import DeepSpeedAccelerator

# During setup stage torch_xla may not be installed, so guard the import and let
# op-builder related APIs still be importable without a TPU runtime present.
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    torch_xla = None
    xm = None


class TPU_Accelerator(DeepSpeedAccelerator):
    """Google Cloud TPU backend, driven through PyTorch/XLA (torch_xla).

    TPUs are lazy, graph-compiled, asynchronous devices: XLA owns memory,
    collectives run through torch_xla, and device work is committed by
    xm.mark_step(). Several eager/CUDA-oriented hooks are therefore stubbed out
    the same way MPS/HPU stub theirs, and will be refined once validated on real
    TPU hardware.
    """

    def __init__(self):
        self._name = 'tpu'
        # XLA registers an 'xla' process-group backend with the torch distributed
        # layer, so the default TorchBackend path can drive TPU collectives.
        self._communication_backend_name = 'xla'
        # torch_xla's TorchDynamo backend.
        self._compile_backend = "openxla"
        if torch_xla is None:
            raise ValueError("TPU_Accelerator requires torch_xla, which is not installed on this system.")
        self.xm = xm

    def is_synchronized_device(self):
        # XLA execution is asynchronous/lazy.
        return False

    def use_host_timers(self):
        # No device-side event timers on XLA; fall back to host timers.
        return True

    def resolves_data_dependency(self):
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None):
        if device_index is None:
            return 'xla'
        return 'xla:{}'.format(device_index)

    def device(self, device_index=None):
        return self.xm.xla_device(device_index)

    def set_device(self, device_index):
        # XLA selects the device per-process; there is no global set_device.
        return

    def current_device(self):
        return self.xm.get_local_ordinal()

    def current_device_name(self):
        return str(self.xm.xla_device())

    def device_count(self):
        return len(self.xm.get_xla_supported_devices())

    def synchronize(self, device_index=None):
        # Block until all pending XLA device operations have completed.
        return self.xm.wait_device_ops()

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        return self.xm.set_rng_state(new_state, device=self.device(device_index))

    def get_rng_state(self, device_index=None):
        return self.xm.get_rng_state(device=self.device(device_index))

    def manual_seed(self, seed):
        return self.xm.set_rng_state(seed)

    def manual_seed_all(self, seed):
        return self.xm.set_rng_state(seed)

    def initial_seed(self):
        return

    def default_generator(self, device_index):
        return

    # Streams/Events
    @property
    def Stream(self):
        return None

    def stream(self, stream):
        return None

    def current_stream(self, device_index=None):
        return None

    def default_stream(self, device_index=None):
        return None

    @property
    def Event(self):
        return None

    # Memory management
    # XLA manages device memory itself; detailed accounting can be wired to
    # xm.get_memory_info() once confirmed on hardware. Stubbed like MPS for now.
    def empty_cache(self):
        return

    def memory_allocated(self, device_index=None):
        return

    def max_memory_allocated(self, device_index=None):
        return

    def reset_max_memory_allocated(self, device_index=None):
        return

    def memory_cached(self, device_index=None):
        return

    def max_memory_cached(self, device_index=None):
        return

    def reset_max_memory_cached(self, device_index=None):
        return

    def memory_stats(self, device_index=None):
        return

    def reset_peak_memory_stats(self, device_index=None):
        return

    def memory_reserved(self, device_index=None):
        return

    def max_memory_reserved(self, device_index=None):
        return

    def total_memory(self, device_index=None):
        return

    def available_memory(self, device_index=None):
        return

    # Data types
    def is_bf16_supported(self):
        # bfloat16 is the native TPU compute dtype.
        return True

    def is_fp16_supported(self):
        # TPUs have no native float16.
        return False

    def supported_dtypes(self):
        return [torch.float, torch.bfloat16]

    # Misc
    def is_available(self):
        if torch_xla is None:
            return False
        try:
            return len(self.xm.get_xla_supported_devices()) > 0
        except Exception:
            return False

    def range_push(self, msg, domain=None, category=None):
        return

    def range_pop(self, domain=None):
        return

    def lazy_call(self, callback):
        return callback()

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

    # Tensor operations
    @property
    def BFloat16Tensor(self):
        return

    @property
    def ByteTensor(self):
        return

    @property
    def DoubleTensor(self):
        return

    @property
    def FloatTensor(self):
        return

    @property
    def HalfTensor(self):
        return

    @property
    def IntTensor(self):
        return

    @property
    def LongTensor(self):
        return

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('xla'):
            return True
        else:
            return False

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder.tpu"
        except ImportError:
            return "deepspeed.ops.op_builder.tpu"

    # dict that holds class name <--> class type mapping i.e.
    # 'NotImplementedBuilder': <class 'op_builder.tpu.no_impl.NotImplementedBuilder'>
    # this dict will be filled at init stage
    class_dict = None

    def _lazy_init_class_dict(self):
        if self.class_dict is not None:
            return
        else:
            self.class_dict = {}
            # begin initialize for create_op_builder()
            # put all valid class name <--> class type mapping into class_dict
            op_builder_dir = self.op_builder_dir()
            op_builder_module = importlib.import_module(op_builder_dir)
            op_builder_absolute_path = os.path.dirname(op_builder_module.__file__)
            for _, module_name, _ in pkgutil.iter_modules([op_builder_absolute_path]):
                # avoid self references,
                # skip sub_directories which contains ops for other backend(cpu, npu, etc.).
                if module_name != 'all_ops' and module_name != 'builder' and not os.path.isdir(
                        os.path.join(op_builder_absolute_path, module_name)):
                    module = importlib.import_module("{}.{}".format(op_builder_dir, module_name))
                    for member_name in module.__dir__():
                        if member_name.endswith('Builder') and member_name != "OpBuilder" \
                                and member_name != "CPUOpBuilder" and member_name != "TPUOpBuilder":  # avoid abstract classes
                            if not member_name in self.class_dict:
                                self.class_dict[member_name] = getattr(module, member_name)
            # end initialize for create_op_builder()

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]()
        else:
            return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]
        else:
            return self.class_dict['NotImplementedBuilder'] if 'NotImplementedBuilder' in self.class_dict else None

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return []

    def visible_devices_envs(self):
        return ['TPU_VISIBLE_CHIPS']

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
