# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import random
from deepspeed.runtime.sparse_tensor import SparseTensor


def test_csr_addition_self():
    row_count = 10
    random.seed(1234)

    x = torch.ones(1, 5)
    for i in range(row_count - 1):
        if random.random() > 0.75:
            x = torch.cat([x, torch.ones(1, 5)])
        else:
            x = torch.cat([x, torch.zeros(1, 5)])
    dense_x = x.clone()
    cx = SparseTensor(x)

    assert torch.all(dense_x == cx.to_dense())

    cx.add(cx)
    assert torch.all(dense_x + dense_x == cx.to_dense())


def test_csr_addition_different():
    row_count = 10
    random.seed(1234)

    x = torch.ones(1, 5)
    for i in range(row_count - 1):
        if random.random() > 0.75:
            x = torch.cat([x, torch.ones(1, 5)])
        else:
            x = torch.cat([x, torch.zeros(1, 5)])
    dense_x = x.clone()
    cx = SparseTensor(x)

    y = torch.ones(1, 5)
    for i in range(row_count - 1):
        if random.random() > 0.75:
            y = torch.cat([y, torch.ones(1, 5)])
        else:
            y = torch.cat([y, torch.zeros(1, 5)])
    dense_y = y.clone()
    cy = SparseTensor(y)

    dense_sum = dense_x + dense_y
    cx.add(cy)

    assert torch.all(dense_sum == cx.to_dense())
