# A test on its own
import torch

# A test on its own
import deepspeed


def test_cuda():
    assert (torch.cuda.is_available())


def test_check_version():
    assert hasattr(deepspeed, "__git_hash__")
    assert hasattr(deepspeed, "__git_branch__")
    assert hasattr(deepspeed, "__version__")
