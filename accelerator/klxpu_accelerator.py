# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import pkgutil
import importlib

from .cuda_accelerator import CUDA_Accelerator as CudaAccelerator


class KLXPU_Accelerator(CudaAccelerator):
    # KUNLUNXIN XPU (torch_xmlir) presents itself through torch.cuda, so device  #ignore-cuda
    # strings, streams, RNG, and memory management are inherited from CUDA.
    supports_nvtx_domain = False

    def __init__(self):
        super().__init__()
        self._name = "klxpu"

    # KLXPU CUDA-event timers return 0.0 unless XPU_EVENT_KL3_ENABLE=1, so fall
    # back to host-side timing to keep throughput/latency measurements correct.
    def use_host_timers(self):
        if os.environ.get("XPU_EVENT_KL3_ENABLE", "0") == "1":
            return False
        return True

    def _get_nvtx_domain(self, domain):
        return None

    def is_triton_supported(self):
        return False

    def prefer_triton_grouped_mm(self):
        return False

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore

            return "op_builder.klxpu"
        except ImportError:
            return "deepspeed.ops.op_builder.klxpu"

    class_dict = None

    def _lazy_init_class_dict(self):
        if self.class_dict is not None:
            return
        self.class_dict = {}
        op_builder_dir = self.op_builder_dir()
        op_builder_module = importlib.import_module(op_builder_dir)
        op_builder_absolute_path = os.path.dirname(op_builder_module.__file__)
        for _, module_name, _ in pkgutil.iter_modules([op_builder_absolute_path]):
            if module_name in ("all_ops", "builder") or os.path.isdir(
                    os.path.join(op_builder_absolute_path, module_name)):
                continue
            module = importlib.import_module("{}.{}".format(op_builder_dir, module_name))
            for member_name in module.__dir__():
                if member_name.endswith("Builder") and member_name not in (
                        "OpBuilder",
                        "CUDAOpBuilder",
                        "TorchCPUOpBuilder",
                        "KLXPUOpBuilder",
                ):
                    if member_name not in self.class_dict:
                        self.class_dict[member_name] = getattr(module, member_name)

    def export_envs(self):
        # KUNLUNXIN env prefixes propagated to remote workers: BKCL_* (collective
        # comm) and XPU_*/XMLIR_* (torch_xmlir runtime), plus paths.
        return ["BKCL", "XPU", "XMLIR", "LD_LIBRARY", "PATH"]
