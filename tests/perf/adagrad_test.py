# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
import time

NUM_ITERS = 100


def _test_perf(param, optimizer_func):
    optimizer = optimizer_func(param)
    avg = 0
    for i in range(NUM_ITERS):
        for i, p in enumerate(param):
            p.grad = torch.ones_like(p) * 2
        start = time.time()
        optimizer.step()
        stop = time.time()
        avg += (stop - start)

    return avg / NUM_ITERS


def _main():
    device = 'cpu'
    model_size = 1 * 1024**3
    group_size = [model_size, 274432]
    param = [torch.nn.Parameter(torch.ones(size, device=device)) for size in group_size]
    torch_time = _test_perf(param, torch.optim.Adagrad)
    ds_time = _test_perf(param, DeepSpeedCPUAdagrad)
    print(f"Step time: {torch_time=} {ds_time=}")


_main()
