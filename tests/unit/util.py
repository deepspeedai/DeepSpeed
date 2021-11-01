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
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])

    if type(torch.cuda.nccl.version()) != tuple:
        return False
    else:
        NCCL_MAJOR = torch.cuda.nccl.version()[0]
        NCCL_MINOR = torch.cuda.nccl.version()[1]

    CUDA_MAJOR = int(torch_info['cuda_version'].split('.')[0])
    if (TORCH_MAJOR > 1 or
        (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)) and (CUDA_MAJOR >= 11) and (
            NCCL_MAJOR > 2 or
            (NCCL_MAJOR == 2 and NCCL_MINOR >= 10)) and torch.cuda.is_bf16_supported():
        return True
    else:
        return False
