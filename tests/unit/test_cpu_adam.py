import argparse
import torch
import time
import numpy as np
import pytest
import copy

import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.op_builder import CPUAdamBuilder

if not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
    pytest.skip("cpu-adam is not compatible")


def check_equal(first, second, atol=1e-2, verbose=False):
    x = first.detach().numpy()
    y = second.detach().numpy()
    if verbose:
        print("x = {}".format(x.flatten()))
        print("y = {}".format(y.flatten()))
        print('-' * 80)
    np.testing.assert_allclose(x, y, err_msg="param-update dismatch!", atol=atol)

@pytest.mark.parametrize('model_size',
                         [
                             (64),
                             (22),
                             (55),
                             (127),
                             (1024),
                             (1048576),
                         ]) # yapf: disable
def test_cpu_adam_opt(model_size):
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    device = 'cpu'
    rng_state = torch.get_rng_state()
    param = torch.nn.Parameter(torch.randn(model_size, device=device))
    torch.set_rng_state(rng_state)
    param1 = torch.nn.Parameter(torch.randn(model_size, device=device))
    torch.set_rng_state(rng_state)
    param2_data = torch.randn(model_size, device=device).cuda()
    param2 = torch.nn.Parameter(param2_data)

    optimizer1 = torch.optim.AdamW([param1])
    optimizer2 = FusedAdam([param2])
    optimizer = DeepSpeedCPUAdam([param])

    for i in range(10):
        rng_state = torch.get_rng_state()
        param.grad = torch.randn(model_size, device=device)
        torch.set_rng_state(rng_state)
        param1.grad = torch.randn(model_size, device=device)
        torch.set_rng_state(rng_state)
        param2.grad = torch.randn(model_size, device=device).cuda()

        optimizer.step()
        optimizer2.step()
        optimizer1.step()

    check_equal(param, param1, atol=1e-2, verbose=True)
    check_equal(param, param2.cpu(), atol=1e-2, verbose=True)
