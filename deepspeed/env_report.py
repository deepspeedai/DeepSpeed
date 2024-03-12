# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
import deepspeed
import subprocess
import argparse
from .ops.op_builder.all_ops import ALL_OPS
from .git_version_info import installed_ops, torch_info, accelerator_name
from deepspeed.accelerator import get_accelerator

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
END = '\033[0m'
SUCCESS = f"{GREEN} [SUCCESS] {END}"
OKAY = f"{GREEN}[OKAY]{END}"
WARNING = f"{YELLOW}[WARNING]{END}"
FAIL = f'{RED}[FAIL]{END}'
INFO = '[INFO]'

color_len = len(GREEN) + len(END)
okay = f"{GREEN}[OKAY]{END}"
warning = f"{YELLOW}[WARNING]{END}"


def op_report(verbose=True):
    max_dots = 23
    max_dots2 = 11
    h = ["op name", "installed", "compatible"]
    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))
    print("DeepSpeed C++/CUDA extension op report")
    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))

    print("NOTE: Ops not installed will be just-in-time (JIT) compiled at\n"
          "      runtime if needed. Op compatibility means that your system\n"
          "      meet the required dependencies to JIT install the op.")

    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))
    print("JIT compiled ops requires ninja")
    ninja_status = OKAY if ninja_installed() else FAIL
    print('ninja', "." * (max_dots - 5), ninja_status)
    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))
    print(h[0], "." * (max_dots - len(h[0])), h[1], "." * (max_dots2 - len(h[1])), h[2])
    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))
    installed = f"{GREEN}[YES]{END}"
    no = f"{YELLOW}[NO]{END}"
    for op_name, builder in ALL_OPS.items():
        dots = "." * (max_dots - len(op_name))
        is_compatible = OKAY if builder.is_compatible(verbose) else no
        is_installed = installed if installed_ops.get(op_name,
                                                      False) and accelerator_name == get_accelerator()._name else no
        dots2 = '.' * ((len(h[1]) + (max_dots2 - len(h[1]))) - (len(is_installed) - color_len))
        print(op_name, dots, is_installed, dots2, is_compatible)
    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))


def ninja_installed():
    try:
        import ninja  # noqa: F401 # type: ignore
    except ImportError:
        return False
    return True


def nvcc_version():
    import torch.utils.cpp_extension
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    if cuda_home is None:
        return f"{RED} [FAIL] cannot find CUDA_HOME via torch.utils.cpp_extension.CUDA_HOME={torch.utils.cpp_extension.CUDA_HOME} {END}"
    try:
        output = subprocess.check_output([cuda_home + "/bin/nvcc", "-V"], universal_newlines=True)
    except FileNotFoundError:
        return f"{RED} [FAIL] nvcc missing {END}"
    output_split = output.split()
    release_idx = output_split.index("release")
    release = output_split[release_idx + 1].replace(',', '').split(".")
    return ".".join(release)


def installed_cann_path():
    if "ASCEND_HOME_PATH" in os.environ or os.path.exists(os.environ["ASCEND_HOME_PATH"]):
        return os.environ["ASCEND_HOME_PATH"]
    return None


def installed_cann_version():
    import re
    ascend_path = installed_cann_path()
    if ascend_path is None:
        return f"CANN_HOME does not exist, unable to compile NPU op(s)"
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


def get_shm_size():
    try:
        shm_stats = os.statvfs('/dev/shm')
    except (OSError, FileNotFoundError, ValueError):
        return "UNKNOWN", None

    shm_size = shm_stats.f_frsize * shm_stats.f_blocks
    shm_hbytes = human_readable_size(shm_size)
    warn = []
    if shm_size < 512 * 1024**2:
        warn.append(
            f" {YELLOW} [WARNING] /dev/shm size might be too small, if running in docker increase to at least --shm-size='1gb' {END}"
        )
        if get_accelerator().communication_backend_name() == "nccl":
            warn.append(
                f" {YELLOW} [WARNING] see more details about NCCL requirements: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#sharing-data {END}"
            )
    return shm_hbytes, warn


def human_readable_size(size):
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    i = 0
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    return f'{size:.2f} {units[i]}'


def debug_report():
    max_dots = 33

    report = [("torch install path", torch.__path__), ("torch version", torch.__version__),
              ("deepspeed install path", deepspeed.__path__),
              ("deepspeed info", f"{deepspeed.__version__}, {deepspeed.__git_hash__}, {deepspeed.__git_branch__}")]
    if get_accelerator().device_name() == 'cuda':
        hip_version = getattr(torch.version, "hip", None)
        report.extend([("torch cuda version", torch.version.cuda), ("torch hip version", hip_version),
                       ("nvcc version", (None if hip_version else nvcc_version())),
                       ("deepspeed wheel compiled w.", f"torch {torch_info['version']}, " +
                        (f"hip {torch_info['hip_version']}" if hip_version else f"cuda {torch_info['cuda_version']}"))
                       ])
    elif get_accelerator().device_name() == 'npu':
        import torch_npu
        report.extend([("deepspeed wheel compiled w.", f"torch {torch_info['version']}"),
                       ("torch_npu install path", torch_npu.__path__), ("torch_npu version", torch_npu.__version__),
                       ("ascend_cann version", installed_cann_version())])
    else:
        report.extend([("deepspeed wheel compiled w.", f"torch {torch_info['version']} ")])

    report.append(("shared memory (/dev/shm) size", get_shm_size()))

    print("DeepSpeed general environment info:")
    for name, value in report:
        warns = []
        if isinstance(value, tuple):
            value, warns = value
        print(name, "." * (max_dots - len(name)), value)
        if warns:
            for warn in warns:
                print(warn)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hide_operator_status',
                        action='store_true',
                        help='Suppress display of installation and compatibility statuses of DeepSpeed operators. ')
    parser.add_argument('--hide_errors_and_warnings', action='store_true', help='Suppress warning and error messages.')
    args = parser.parse_args()
    return args


def main(hide_operator_status=False, hide_errors_and_warnings=False):
    if not hide_operator_status:
        op_report(verbose=not hide_errors_and_warnings)
    debug_report()


def cli_main():
    args = parse_arguments()
    main(hide_operator_status=args.hide_operator_status, hide_errors_and_warnings=args.hide_errors_and_warnings)


if __name__ == "__main__":
    main()
