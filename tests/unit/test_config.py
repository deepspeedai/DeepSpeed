# A test on its own
import torch

# A test on its own
import deepspeed


def test_cuda():
    assert (torch.cuda.is_available())
