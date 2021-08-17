import torch


def required_torch_version():
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])

    if TORCH_MAJOR >= 1 and TORCH_MINOR >= 8:
        return True
    else:
        return False
