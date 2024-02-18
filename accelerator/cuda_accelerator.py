# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools
import os
import pkgutil
import importlib

from .abstract_accelerator import DeepSpeedAccelerator
# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch.cuda
except ImportError:
    pass

# Delay import pynvml to avoid import error when CUDA is not available
pynvml = None


class CUDA_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'cuda'
        self._communication_backend_name = 'nccl'
        if pynvml is None:
            self._init_pynvml()

    def _init_pynvml(self):
        global pynvml
        try:
            import pynvml
        except ImportError:
            return
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            pynvml = None
            return

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
        if device_index is None:
            return 'cuda'
        return 'cuda:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.cuda.device(device_index)

    def set_device(self, device_index):
        torch.cuda.set_device(device_index)

    def current_device(self):
        return torch.cuda.current_device()

    def current_device_name(self):
        return 'cuda:{}'.format(torch.cuda.current_device())

    def device_count(self):
        return torch.cuda.device_count()

    def synchronize(self, device_index=None):
        return torch.cuda.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.cuda.set_rng_state(new_state)

        return torch.cuda.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        if device_index is None:
            return torch.cuda.get_rng_state()

        return torch.cuda.get_rng_state(device_index)

    def manual_seed(self, seed):
        return torch.cuda.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.cuda.manual_seed_all(seed)

    def initial_seed(self, seed):
        return torch.cuda.initial_seed(seed)

    def default_generator(self, device_index):
        return torch.cuda.default_generators[device_index]

    # Streams/Events
    @property
    def Stream(self):
        return torch.cuda.Stream

    def stream(self, stream):
        return torch.cuda.stream(stream)

    def current_stream(self, device_index=None):
        return torch.cuda.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch.cuda.default_stream(device_index)

    @property
    def Event(self):
        return torch.cuda.Event

    # Memory management
    def empty_cache(self):
        return torch.cuda.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.cuda.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.cuda.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.cuda.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch.cuda.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        return torch.cuda.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch.cuda.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        if hasattr(torch.cuda, 'memory_stats'):
            return torch.cuda.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            return torch.cuda.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        if hasattr(torch.cuda, 'memory_reserved'):
            return torch.cuda.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch.cuda, 'max_memory_reserved'):
            return torch.cuda.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.cuda.get_device_properties(device_index).total_memory

    def _get_nvml_gpu_id(self, torch_gpu_id):
        """
        credit: https://discuss.pytorch.org/t/making-pynvml-match-torch-device-ids-cuda-visible-devices/103020

        Remap torch device id to nvml device id, respecting CUDA_VISIBLE_DEVICES.

        If the latter isn't set return the same id
        """
        # if CUDA_VISIBLE_DEVICES is used automagically remap the id since pynvml ignores this env var
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            ids = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
            return ids[torch_gpu_id]  # remap
        else:
            return torch_gpu_id

    def available_memory(self, device_index=None):
        if pynvml:
            if device_index is None:
                device_index = self.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._get_nvml_gpu_id(device_index))
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.free
        else:
            return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    def is_bf16_supported(self):
        if not torch.cuda.is_available():
            return True
        return torch.cuda.is_bf16_supported()

    def is_fp16_supported(self):
        if not torch.cuda.is_available():
            return True
        # See https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix
        # FP16 on compute capability 6.x is deprecated
        allow_deprecated_fp16 = os.environ.get('DS_ALLOW_DEPRECATED_FP16', '0') == '1'
        major, _ = torch.cuda.get_device_capability()
        if major >= 7:
            return True
        elif major == 6 and allow_deprecated_fp16:
            return True
        else:
            return False

    def supported_dtypes(self):
        supported_dtypes = [torch.float]
        if self.is_fp16_supported():
            supported_dtypes.append(torch.half)
        if self.is_bf16_supported():
            supported_dtypes.append(torch.bfloat16)
        return supported_dtypes

    # Misc
    def amp(self):
        if hasattr(torch.cuda, 'amp'):
            return torch.cuda.amp
        return None

    def is_available(self):
        return torch.cuda.is_available()

    def range_push(self, msg):
        if hasattr(torch.cuda.nvtx, 'range_push'):
            return torch.cuda.nvtx.range_push(msg)

    def range_pop(self):
        if hasattr(torch.cuda.nvtx, 'range_pop'):
            return torch.cuda.nvtx.range_pop()

    def lazy_call(self, callback):
        return torch.cuda._lazy_call(callback)

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return True
        else:
            return False

    # Graph operations
    def create_graph(self):
        return torch.cuda.CUDAGraph()

    def capture_to_graph(self, graph, pool=None, stream=None):
        return torch.cuda.graph(graph, pool, stream)

    def replay_graph(self, graph):
        graph.replay()
        return

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return functools.partial(torch.tensor, dtype=torch.bfloat16, device='cuda')

    @property
    def ByteTensor(self):
        return functools.partial(torch.tensor, dtype=torch.uint8, device='cuda')

    @property
    def DoubleTensor(self):
        return functools.partial(torch.tensor, dtype=torch.double, device='cuda')

    @property
    def FloatTensor(self):
        return functools.partial(torch.tensor, dtype=torch.float, device='cuda')

    @property
    def HalfTensor(self):
        return functools.partial(torch.tensor, dtype=torch.half, device='cuda')

    @property
    def IntTensor(self):
        return functools.partial(torch.tensor, dtype=torch.int, device='cuda')

    @property
    def LongTensor(self):
        return functools.partial(torch.tensor, dtype=torch.long, device='cuda')

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('cuda:'):
            return True
        else:
            return False

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder"
        except ImportError:
            return "deepspeed.ops.op_builder"

    # dict that holds class name <--> class type mapping i.e.
    # 'AsyncIOBuilder': <class 'op_builder.async_io.AsyncIOBuilder'>
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
                        if member_name.endswith(
                                'Builder'
                        ) and member_name != "OpBuilder" and member_name != "CUDAOpBuilder" and member_name != "TorchCPUOpBuilder":  # avoid abstract classes
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
            return None

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return ['NCCL']
