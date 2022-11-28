import torch
from deepspeed.git_version_info import torch_info


def required_torch_version():
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])

    if TORCH_MAJOR >= 1 and TORCH_MINOR >= 8:
        return True
    else:
        return False


def bf16_required_version_check():
    split_version = lambda x: map(int, x.split('.')[:2])
    TORCH_MAJOR, TORCH_MINOR = split_version(torch_info['version'])
    NCCL_MAJOR, NCCL_MINOR = split_version(torch_info['nccl_version'])
    CUDA_MAJOR, CUDA_MINOR = split_version(torch_info['cuda_version'])

    if (TORCH_MAJOR > 1 or
        (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)) and (CUDA_MAJOR >= 11) and (
            NCCL_MAJOR > 2 or
            (NCCL_MAJOR == 2 and NCCL_MINOR >= 10)) and torch_info['bf16_support']:
        return True
    else:
        return False


def required_minimum_torch_version(major_version, minor_version):
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])

    if TORCH_MAJOR < major_version:
        return False

    return TORCH_MAJOR > major_version or TORCH_MINOR >= minor_version


def required_maximum_torch_version(major_version, minor_version):
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])

    if TORCH_MAJOR > major_version:
        return False

    return TORCH_MAJOR < major_version or TORCH_MINOR <= minor_version


def required_amp_check():
    from importlib.util import find_spec
    if find_spec('apex') is None:
        return False
    else:
        return True
