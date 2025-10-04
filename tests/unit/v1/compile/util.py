# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from copy import deepcopy

import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero import GatheredParameters

from unit.simple_model import SimpleModel
from unit.common import enable_determinism, allclose_on_all_ranks


@enable_determinism(123)
def compare_loss(self, config, dtype, iteration=5, hidden_dim_override=None):
    hidden_dim = hidden_dim_override if hidden_dim_override is not None else 10

    # the default tolerances of torch.testing.assert_close are too small
    RTOL = 5e-1
    ATOL = 1e-2

    device = torch.device(get_accelerator().current_device_name())
    model = SimpleModel(hidden_dim)

    i = get_accelerator().current_device()
    baseline_model = deepcopy(model)
    baseline_config = deepcopy(config)
    baseline_config["zero_optimization"]["stage"] = 0
    baseline_config["zero_optimization"]["offload_optimizer"] = {}
    baseline_engine, baseline_optimizer, _, _ = deepspeed.initialize(config=baseline_config,
                                                                     model=baseline_model,
                                                                     model_parameters=baseline_model.parameters())

    if config["zero_optimization"]["stage"] == 3:
        with deepspeed.zero.Init(config_dict_or_path=config):
            target_model = SimpleModel(hidden_dim)
        with GatheredParameters(target_model.parameters(), modifier_rank=0):
            for p1, p2 in zip(target_model.parameters(), model.parameters()):
                p1.data.copy_(p2.data)
    else:
        target_model = deepcopy(model)

    target_engine, target_optimizer, _, _ = deepspeed.initialize(config=config,
                                                                 model=target_model,
                                                                 model_parameters=target_model.parameters())
    target_engine.compile()

    train_batch_size = config["train_micro_batch_size_per_gpu"]

    xs = [torch.randn(train_batch_size, hidden_dim, device=device, dtype=dtype) for _ in range(iteration)]
    ys = [torch.randn_like(x) for x in xs]

    for x, y in zip(xs, ys):
        baseline_loss = baseline_engine(x, y)
        target_loss = target_engine(x, y)

        allclose_on_all_ranks(baseline_loss, target_loss, "Loss values are not close.", rtol=RTOL, atol=ATOL)

        baseline_engine.backward(baseline_loss)
        target_engine.backward(target_loss)

        baseline_engine.step()
        target_engine.step()

        with GatheredParameters(target_engine.parameters()):
            for p1, p2 in zip(baseline_engine.parameters(), target_engine.parameters()):
                allclose_on_all_ranks(p1, p2, "Parameters are not equal.", rtol=RTOL, atol=ATOL)

    baseline_engine.destroy()
    target_engine.destroy()
