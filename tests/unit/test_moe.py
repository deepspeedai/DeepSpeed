import math
from deepspeed.utils import groups
import torch
import torch.distributed as dist
import deepspeed
import argparse
import pytest
import json
import os
from deepspeed.ops.adam import FusedAdam
from common import distributed_test
from deepspeed.ops.op_builder import CPUAdamBuilder
from simple_model import SimpleModel, SimpleOptimizer, random_dataloader, args_from_dict, create_deepspeed_args, SimpleMoEModel, sequence_dataloader
from util import required_torch_version

try:
    from apex import amp
    _amp_available = True
except ImportError:
    _amp_available = False
amp_available = pytest.mark.skip(_amp_available, reason="apex/amp is not installed")


@pytest.mark.parametrize("ep_size", [2, 4])
def test_moe(tmpdir, ep_size):
    if not required_torch_version():
        pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

    config_dict = {
        "train_batch_size": 8,
        "steps_per_print": 1,
        "fp16": {
            "enabled": True
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 16

    @distributed_test(world_size=[4])
    def _test_moe(args, hidden_dim, ep_size):
        # E+D -- ep_size = 2
        # E only -- ep_size = 4
        #groups.initialize_model_parallel(1)
        #groups.initialize_expert_parallel(2)
        groups.initialize(ep_size=ep_size)
        model = SimpleMoEModel(hidden_dim)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              optimizer=optimizer,
                                              dist_init_required=False)
        #dist_init_required=False -- parameterize to True/False?

        assert dist.get_world_size() == groups.get_data_parallel_world_size(), "incorrect data parallel world size"
        assert ep_size == groups.get_expert_parallel_world_size(), "incorrect expert parallel world size"

        data_loader = sequence_dataloader(model=model,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=model.device)

        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_moe(args=args, hidden_dim=hidden_dim, ep_size=ep_size)
