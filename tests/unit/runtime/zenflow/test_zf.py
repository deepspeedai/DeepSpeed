# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zenflow.zenflow_stage_1_and_2 import _num_selected_columns

from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader
import deepspeed
from deepspeed.ops.adam.zenflow_torch_adam import ZenFlowSelectiveAdamW


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("group_step", [False, True])
@pytest.mark.parametrize("cleared_gradient", [False, True])
def test_selective_optimizer_skips_unused_parameters(offload, group_step, cleared_gradient):
    model = torch.nn.Linear(3, 2, bias=False)
    unused = torch.nn.Parameter(torch.ones(2, 3))
    if cleared_gradient:
        unused.selected_grad = None
    param = model.weight
    initial = param.detach().clone()
    indices = torch.tensor([0])
    param.selected_indices = indices
    params = [unused, param]
    reference = torch.nn.Parameter(initial[:, indices].clone())
    optimizer = ZenFlowSelectiveAdamW(params, lr=0.01, weight_decay=0.1, offload=offload, bucket_size=1)
    reference_optimizer = torch.optim.AdamW([reference], lr=0.01, weight_decay=0.1)
    if offload:
        param.exp_avg_cpu_data = torch.zeros_like(reference)
        param.exp_avg_sq_cpu_data = torch.zeros_like(reference)

    def step():
        if group_step:
            optimizer.group_step({0: params})
        else:
            optimizer.step()

    for _ in range(3):
        optimizer.zero_grad()
        model(torch.tensor([[1.0, 2.0, 3.0]])).square().sum().backward()
        param.selected_grad = param.grad[:, indices].clone()
        optimizer.temp_copy_param({0: params})
        reference.grad = param.selected_grad.clone()
        reference_optimizer.step()
        step()
        expected = initial.clone()
        expected[:, indices] = reference.detach()
        torch.testing.assert_close(param, expected)
        torch.testing.assert_close(unused, torch.ones_like(unused))
        assert unused not in optimizer.state

    param.selected_grad = None
    optimizer.temp_copy_param({0: params})
    step()
    torch.testing.assert_close(param, expected)
    assert optimizer.state[param]["step"].item() == 3


class _ConditionalZenFlowModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.unused = torch.nn.Linear(4, 4)
        self.active = SimpleModel(4)

    def forward(self, inputs, labels, use_extra):
        if use_extra:
            inputs = self.unused(inputs)
        return self.active(inputs, labels)


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("offload", [False, True])
def test_zenflow_engine_unused_branch(tmp_path, monkeypatch, stage, offload):
    monkeypatch.setenv("LOCAL_RANK", "0")
    dist.init_distributed(get_accelerator().communication_backend_name(),
                          auto_mpi_discovery=False,
                          init_method=f"file://{tmp_path}/rendezvous",
                          rank=0,
                          world_size=1)
    engine = None
    try:
        torch.manual_seed(42)
        model = _ConditionalZenFlowModel()
        config = {
            "train_batch_size": 1,
            "steps_per_print": 10000,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.01,
                    "weight_decay": 0.0
                }
            },
            "zero_optimization": {
                "stage": stage,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": False
                },
                "overlap_comm": True,
                "reduce_bucket_size": 128,
                "allgather_bucket_size": 128,
                "ignore_unused_parameters": True,
                "zenflow": {
                    "topk_ratio": 0.5,
                    "select_strategy": "step",
                    "select_interval": 1,
                    "update_interval": 1,
                    "overlap_step": False,
                    "offload": offload,
                    "full_warm_up_rounds": 0,
                },
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=model, config=config)
        unused_before = model.unused.weight.detach().clone()
        active_before = model.active.linears[0].weight.detach().clone()
        inputs = torch.ones(1, 4, device=engine.device)
        labels = torch.zeros(1, dtype=torch.long, device=engine.device)
        # Exercise both never-used parameters and a branch that becomes unused again.
        for step, use_extra in enumerate([False, False, True, False, False]):
            loss = engine(inputs, labels, use_extra)
            assert torch.isfinite(loss)
            engine.backward(loss)
            engine.step()
            if step == 1:
                torch.testing.assert_close(model.unused.weight, unused_before, rtol=0, atol=0)
        assert not torch.equal(model.active.linears[0].weight, active_before)
    finally:
        if engine is not None:
            engine.destroy()
        dist.destroy_process_group()


@pytest.mark.parametrize("num_columns,topk_ratio,expected", [
    (0, 0.01, 0),
    (50, 0.01, 1),
    (200, 0.01, 2),
])
def test_num_selected_columns_has_nonzero_floor(num_columns, topk_ratio, expected):
    assert _num_selected_columns(num_columns, topk_ratio) == expected


class BaseZenFlowTest:
    hidden_dim = 10
    batch_size = 4
    grad_acc_steps = 1

    def get_config_dict(self,
                        stage,
                        offload_selective_optimizer,
                        select_strategy,
                        select_interval,
                        update_interval,
                        full_warm_up_rounds,
                        topk_ratio=0.2):
        config = {
            "train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.grad_acc_steps,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "zero_optimization": {
                "stage": stage,
                "offload_optimizer": {
                    "device": "cpu"
                },
                "overlap_comm": True,
                "zenflow": {
                    "topk_ratio": topk_ratio,
                    "select_strategy": select_strategy,
                    "select_interval": select_interval,
                    "update_interval": update_interval,
                    "overlap_step": False,
                    "offload": offload_selective_optimizer,
                    "auto_ratio": 0.99,
                    "full_warm_up_rounds": full_warm_up_rounds,
                }
            },
            "zero_allow_untested_optimizer": True,
        }

        if get_accelerator().is_bf16_supported():
            config["bf16"] = {"enabled": True}
        return config

    def run_training_distributed(self, config_dict):

        if get_accelerator().device_name() == "cpu":
            return

        model = SimpleModel(self.hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        train_dataloader = random_dataloader(model=model,
                                             total_samples=20,
                                             hidden_dim=self.hidden_dim,
                                             device=model.device)

        dist.barrier()

        for step, batch in enumerate(train_dataloader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
        model.destroy()


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("full_warm_up_rounds", [0, 3])
@pytest.mark.parametrize("offload_selective_optimizer", [True, False])
@pytest.mark.parametrize("select_strategy,select_interval,update_interval", [
    ("auto", "auto", "auto"),
    ("step", 10, 3),
    ("epoch", 1, 4),
])
class TestZenFlowSingleGPU(DistributedTest, BaseZenFlowTest):
    world_size = 1

    def test_zenflow_single_gpu(self, stage, offload_selective_optimizer, select_strategy, select_interval,
                                update_interval, full_warm_up_rounds):
        tester = BaseZenFlowTest()
        config_dict = tester.get_config_dict(stage, offload_selective_optimizer, select_strategy, select_interval,
                                             update_interval, full_warm_up_rounds)
        tester.run_training_distributed(config_dict)


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("full_warm_up_rounds", [0, 3])
@pytest.mark.parametrize("offload_selective_optimizer", [True, False])
@pytest.mark.parametrize("select_strategy,select_interval,update_interval", [
    ("auto", "auto", "auto"),
    ("step", 10, 3),
    ("epoch", 1, 4),
])
class TestZenFlowDistributed(DistributedTest, BaseZenFlowTest):
    world_size = 2

    def test_zenflow_distributed(self, stage, offload_selective_optimizer, select_strategy, select_interval,
                                 update_interval, full_warm_up_rounds):
        config_dict = self.get_config_dict(stage, offload_selective_optimizer, select_strategy, select_interval,
                                           update_interval, full_warm_up_rounds)
        self.run_training_distributed(config_dict)


@pytest.mark.parametrize("stage", [1, 2])
class TestZenFlowSmallTopKRatio(DistributedTest, BaseZenFlowTest):
    world_size = 2
    hidden_dim = 50

    def test_small_positive_topk_ratio(self, stage):
        config_dict = self.get_config_dict(stage, False, "step", 1, 1, 0, topk_ratio=0.01)
        self.run_training_distributed(config_dict)


@pytest.mark.parametrize(
    "cores,perc,expected_zf,expected_pt",
    [
        # Normal split: ceil(0.25 * 8) = 2 cores reserved for training.
        ([0, 1, 2, 3, 4, 5, 6, 7], 0.25, [2, 3, 4, 5, 6, 7], [0, 1]),
        # Rounds up: ceil(0.1 * 8) = 1.
        ([0, 1, 2, 3, 4, 5, 6, 7], 0.1, [1, 2, 3, 4, 5, 6, 7], [0]),
        # Two cores, half each.
        ([10, 11], 0.5, [11], [10]),
        # Reserve rounds to 0 -> both sides share the full set.
        ([0, 1, 2, 3], 0.0, [0, 1, 2, 3], [0, 1, 2, 3]),
        # Reserve rounds to every core -> both sides share the full set.
        ([0, 1, 2, 3], 1.0, [0, 1, 2, 3], [0, 1, 2, 3]),
    ])
def test_split_affinity(cores, perc, expected_zf, expected_pt):
    from deepspeed.runtime.zenflow.zenflow_utils import _split_affinity
    zf, pt = _split_affinity(cores, perc)
    assert zf == expected_zf
    assert pt == expected_pt
    # When the sides are actually isolated they must partition the cores exactly.
    if zf != pt:
        assert sorted(zf + pt) == sorted(cores)
        assert not (set(zf) & set(pt))
