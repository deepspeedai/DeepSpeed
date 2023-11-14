# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import re
import os
try:
    import torch_npu
except ImportError as e:
    pass

try:
    # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
    # if successful this also means we're doing a local install and not JIT compile path
    from op_builder import __deepspeed__  # noqa: F401 # type: ignore
    from op_builder.builder import OpBuilder
except ImportError:
    from deepspeed.ops.op_builder.builder import OpBuilder


class NPUOpBuilder(OpBuilder):
    _ascend_path = None
    _torch_npu_path = None
    _cann_version = None

    def __init__(self, name):
        super().__init__(name)
        self._ascend_path = self.installed_cann_path()
        self._torch_npu_path = os.path.join(os.path.dirname(os.path.abspath(torch_npu.__file__)))
        try:
            self._cann_version = self.installed_cann_version(self.name)
        except BaseException:
            print(f"{self.name} ascend_cann is missing, npu ops cannot be compiled!")

    def cann_defs(self):
        if self._cann_version:
            return '-D__ENABLE_CANN__'
        return '-D__DISABLE_CANN__'

    def installed_cann_path(self):
        if "ASCEND_HOME_PATH" in os.environ or os.path.exists(os.environ["ASCEND_HOME_PATH"]):
            return os.environ["ASCEND_HOME_PATH"]
        return None

    def installed_cann_version(self, name=""):
        ascend_path = self.installed_cann_path()
        assert ascend_path is not None, "CANN_HOME does not exist, unable to compile NPU op(s)"
        cann_version = ""
        for dirpath, _, filenames in os.walk(os.path.realpath(ascend_path)):
            if cann_version:
                break
            install_files = [file for file in filenames if re.match(r"ascend_.*_install\.info", file)]
            if install_files:
                filepath = os.path.join(dirpath, install_files[0])
                with open(filepath, "r") as f:
                    for line in f:
                        if line.find("version") != -1:
                            cann_version = line.strip().split("=")[-1]
                            break
        return cann_version

    def include_paths(self):
        paths = super().include_paths()
        paths += [os.path.join(self._ascend_path, 'include'), os.path.join(self._torch_npu_path, 'include')]
        return paths

    def cxx_args(self):
        args = super().cxx_args()
        args += ['-O3', '-std=c++17', '-g', '-Wno-reorder', '-fopenmp']
        args += ['-fstack-protector-all', '-Wl,-z,relro,-z,now,-z,noexecstack', '-Wl,--disable-new-dtags,--rpath']
        args += [
            self.cann_defs(),
            self.cpu_arch(),
            self.simd_width(), '-L' + os.path.join(self._ascend_path, 'lib64'),
            '-L' + os.path.join(self._torch_npu_path, 'lib')
        ]
        return args

    def extra_ldflags(self):
        flags = super().extra_ldflags()
        flags += [
            '-L' + os.path.join(self._ascend_path, 'lib64'), '-lascendcl',
            '-L' + os.path.join(self._torch_npu_path, 'lib'), '-ltorch_npu'
        ]
        return flags
