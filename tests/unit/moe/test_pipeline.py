import torch
import deepspeed
import pytest
from unit.common import DistributedTest
from deepspeed import get_accelerator
from deepspeed.moe.sharded_moe import _AllToAll
from deepspeed.moe.mappings import gather_tokens
from deepspeed.moe.layer import MoE


class MPU():

    def __init__(self, tp_world_size):
        self.rank = deepspeed.comm.get_rank()
        self.world_size = deepspeed.comm.get_world_size()
        self.tp_world_size = tp_world_size

        for i in range(0, self.world_size, tp_world_size):
            ranks = range(i, i + tp_world_size)
            group = deepspeed.comm.new_group(ranks)
            if self.rank in ranks:
                self.tp_group = group

        for i in range(0, tp_world_size):
            ranks = range(i, self.world_size, tp_world_size)
            group = deepspeed.comm.new_group(ranks)
            if self.rank in ranks:
                self.dp_group = group

    def get_model_parallel_rank(self):
        return self.rank % self.tp_world_size

    def get_model_parallel_world_size(self):
        return self.tp_world_size

    def get_data_parallel_rank(self):
        return self.rank // self.tp_world_size

    def get_data_parallel_world_size(self):
        return self.world_size // self.tp_world_size

    def get_data_parallel_group(self):
        return self.dp_group

    def get_model_parallel_group(self):
        return self.tp_group


@pytest.mark.parametrize("shard_num", [6, 10])
@pytest.mark.parametrize("C, M, scale", [(92, 32, 1),(209, 128, 5)])
class TestPipelineCommunication(DistributedTest):
    world_size = 8

    def test(self, shard_num, C, M, scale):
        tp_size = 2
        world_size = deepspeed.comm.get_world_size()
        E = world_size
        ep_size = 4
        config_dict = {"train_batch_size": 8, "steps_per_print": 1, "fp16": {"enabled": True}}
        hidden_dim = M
        device = get_accelerator().current_device_name()
        tensor_parallel_expert = torch.nn.Sequential(torch.nn.Linear(hidden_dim, 4 * hidden_dim // tp_size),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Linear(4 * hidden_dim // tp_size, hidden_dim))

        model = MoE(
            hidden_size=hidden_dim,
            expert=tensor_parallel_expert,
            num_experts=world_size * scale,
            ep_size=ep_size,
            use_residual=True,
            enable_expert_tensor_parallelism=True,
        )
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              optimizer=optimizer,
                                              dist_init_required=False,
                                              mpu=MPU(tp_size))
        model.deepspeed_moe.shard_num = shard_num
        input = torch.rand(E, C, M, device=device)

        # pipeline alltoall with allgather
        pipeline_output = model.deepspeed_moe.pipeline_alltoall_with_allgather(input)

        # first alltoall, then allgather
        alltoall_output = _AllToAll.apply(model.deepspeed_moe.ep_group, input)
        gather_output = gather_tokens(alltoall_output, dim=1)
        assert torch.allclose(pipeline_output, gather_output, atol=1e-07), f"pipeline_output {pipeline_output} is not equal to gather_output {gather_output}"