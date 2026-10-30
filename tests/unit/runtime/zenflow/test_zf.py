# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import pytest
import torch
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zenflow.zenflow_stage_1_and_2 import _num_selected_columns
from deepspeed.ops.adam.zenflow_torch_adam import ZenFlowSelectiveAdamW, ZenFlowSelectiveAdamW_stage3

from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader
import deepspeed


@pytest.mark.parametrize("num_columns,topk_ratio,expected", [
    (0, 0.01, 0),
    (50, 0.01, 1),
    (200, 0.01, 2),
])
def test_num_selected_columns_has_nonzero_floor(num_columns, topk_ratio, expected):
    assert _num_selected_columns(num_columns, topk_ratio) == expected


@pytest.mark.parametrize("dtype,steps", [(torch.bfloat16, 300), (torch.float16, 2100), (torch.float32, 300)])
def test_selective_adamw_step_counter_keeps_advancing(dtype, steps):
    model = torch.nn.Linear(2, 1, bias=False, dtype=dtype)
    optimizer = ZenFlowSelectiveAdamW(model.parameters(), lr=0.01)
    inputs = torch.ones(1, 2, dtype=dtype)
    for _ in range(steps):
        optimizer.zero_grad()
        model(inputs).float().square().sum().backward()
        model.weight.selected_indices = torch.arange(2)
        model.weight.selected_grad = model.weight.grad
        optimizer.step()

    # BF16 cannot represent 257 and FP16 cannot represent 2049. The bias-correction
    # clock must keep advancing even when the model uses either of those dtypes.
    assert optimizer.state_dict()["state"][0]["step"].item() == steps


@pytest.mark.parametrize("optimizer_cls", [ZenFlowSelectiveAdamW, ZenFlowSelectiveAdamW_stage3])
@pytest.mark.parametrize("counter_dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_selective_adamw_loads_counter_without_losing_precision(optimizer_cls, counter_dtype):
    param = torch.nn.Parameter(torch.ones(2))
    original = torch.optim.AdamW([param])
    param.grad = torch.ones_like(param)
    original.step()
    checkpoint = original.state_dict()
    checkpoint["state"][0]["step"] = torch.tensor(256, dtype=counter_dtype)

    optimizer = optimizer_cls([param])
    optimizer.load_state_dict(checkpoint)
    restored_step = optimizer.state_dict()["state"][0]["step"]
    assert restored_step.item() == 256
    expected_dtype = torch.float64 if counter_dtype == torch.float64 else torch.float32
    assert restored_step.dtype == expected_dtype
    assert checkpoint["state"][0]["step"].dtype == counter_dtype


@pytest.mark.parametrize("dtype,boundary", [(torch.bfloat16, 256), (torch.float16, 2048)])
def test_selective_adamw_resumed_counter_advances_past_old_limit(dtype, boundary):
    param = torch.nn.Parameter(torch.ones(2, dtype=dtype))
    optimizer = ZenFlowSelectiveAdamW([param])
    param.selected_grad = torch.ones_like(param)
    optimizer.step()
    checkpoint = optimizer.state_dict()
    checkpoint["state"][0]["step"] = torch.tensor(boundary, dtype=dtype)

    optimizer.load_state_dict(checkpoint)
    param.selected_grad = torch.ones_like(param)
    optimizer.step()
    assert optimizer.state_dict()["state"][0]["step"].item() == boundary + 1


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


class _ExternalWeightModel(torch.nn.Module):

    def __init__(self, access_mode):
        super().__init__()
        self.projection = torch.nn.Linear(8, 4, bias=False)
        self.metadata = torch.nn.Linear(8, 4, bias=False)
        self.metadata.weight.requires_grad_(False)
        self.access_mode = access_mode

    def forward(self, value):
        if self.access_mode == "dtype_only":
            value = value.to(self.metadata.weight.dtype)
        elif self.access_mode == "nested_owner":
            value = value.to(self.projection.weight.dtype)
        return self.projection(value).square().mean()


class TestZenFlowExternalParameter(DistributedTest, BaseZenFlowTest):
    world_size = 1

    @pytest.mark.parametrize("param_persistence_threshold", [0, 32])
    @pytest.mark.parametrize("access_mode", ["none", "dtype_only", "nested_owner"])
    def test_external_weight_preserves_storage_layout(self, param_persistence_threshold, access_mode):
        """External discovery must not unbalance ZenFlow's compute/storage transposes."""
        if get_accelerator().device_name() == "cpu":
            pytest.skip("ZenFlow integration requires an accelerator")

        torch.manual_seed(42)
        reference = _ExternalWeightModel(access_mode).to(get_accelerator().current_device_name())
        model = copy.deepcopy(reference)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4, weight_decay=0)
        config = self.get_config_dict(3, False, "step", 10, 1, 10)
        config.pop("bf16", None)
        config["zero_optimization"]["stage3_param_persistence_threshold"] = param_persistence_threshold
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0
        config["zero_optimization"]["stage3_max_reuse_distance"] = 0
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)

        # Checkpoint restore can leave persistent weights resident before their first external access.
        for parameter in engine.module.parameters():
            if parameter.ds_persist:
                parameter.all_gather()

        for microstep in range(3):
            value = torch.arange(32, device=engine.device, dtype=torch.float32).reshape(4, 8) / 32
            value = value + microstep / 100
            expected_loss = reference(value)
            loss = engine(value)
            torch.testing.assert_close(loss, expected_loss)

            # ZenFlow stores each matrix transposed outside its active compute consumers.
            with deepspeed.zero.GatheredParameters(list(engine.module.parameters())):
                for parameter, expected in zip(engine.module.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter, expected.t())

            expected_loss.backward()
            engine.backward(loss)
            engine.step()
            reference_optimizer.step()
            reference_optimizer.zero_grad()
            with deepspeed.zero.GatheredParameters(list(engine.module.parameters())):
                for parameter, expected in zip(engine.module.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter, expected.t())

        assert engine.global_steps == 3
        engine.destroy()


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
