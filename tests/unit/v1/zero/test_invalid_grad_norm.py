# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from types import SimpleNamespace

import pytest
import torch
import deepspeed
import deepspeed.runtime.utils as ds_utils
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.superoffload.superoffload_stage3 import SuperOffloadOptimizer_Stage3
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.utils import safe_set_full_grad
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader


def has_inf_or_nan(x):
    float_x = x.float()
    return float_x.isnan().logical_or(float_x.isinf()).float().max()


def run_model_step(model, x_sample, y_label, grad_value):
    loss = model(x_sample, y_label)
    model.backward(loss)
    for p in model.parameters():
        grad = torch.empty_like(p, dtype=p.dtype)
        grad.fill_(grad_value)
        safe_set_full_grad(p, grad)
    model.step()


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_combine_grad_norm_groups_keeps_invalid_sentinels_invalid(bad):
    # vector_norm would turn the -1 DeepSpeed sentinel into 1.0 (and mix a
    # failed group into a finite hypot of the remaining groups).
    healthy = torch.tensor(3.0)
    combined = ds_utils.combine_grad_norm_groups([healthy, torch.tensor(bad)])
    assert not torch.isfinite(combined)


def test_combine_grad_norm_groups_matches_l2_for_finite_groups():
    combined = ds_utils.combine_grad_norm_groups([torch.tensor(3.0), torch.tensor(4.0)])
    torch.testing.assert_close(combined, torch.tensor(5.0))


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_get_global_norm_keeps_invalid_sentinels_invalid(bad):
    assert ds_utils.get_global_norm([3.0, bad]) == float("inf")


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_overflow_check_accepts_every_invalid_norm_representation(bad):
    checker = ds_utils.CheckOverflow()
    assert checker.check_using_norm([bad], reduce_overflow=False)


def test_invalid_clip_norm_zeros_gradients_for_direct_callers():
    # The -1 group-norm sentinel used to produce clip_coef=1 after clamp(min=1.0).
    optimizer = object.__new__(DeepSpeedZeroOptimizer)
    optimizer.clip_grad = 1.0
    optimizer.custom_loss_scaler = False
    optimizer.loss_scaler = SimpleNamespace(cur_scale=1.0)
    optimizer.device = "cpu"
    gradient = torch.ones(4)
    optimizer.unscale_and_clip_grads([gradient], total_norm=torch.tensor(-1.0))
    assert torch.count_nonzero(gradient) == 0


def test_scaled_global_norm_does_not_turn_cpu_offload_sentinel_into_one():
    optimizer = object.__new__(DeepSpeedZeroOptimizer)
    optimizer.cpu_offload = True
    optimizer.has_moe_layers = False
    optimizer.bit16_groups = [[None]]
    optimizer.params_in_partition = [[]]
    optimizer.complete_grad_norm_calculation_for_cpu_offload = lambda params: torch.tensor(-1.0)
    combined = optimizer.scaled_global_norm()
    assert not torch.isfinite(combined)


def test_stage3_rejects_invalid_group_norm_before_optimizer_step():
    optimizer = object.__new__(DeepSpeedZeroOptimizer_Stage3)
    optimizer._pre_step = lambda: None
    optimizer._partition_all_parameters = lambda: None
    optimizer.overflow = False
    optimizer._overflow_check_and_loss_scale_update = lambda update_scale: False
    optimizer._get_norm_groups = lambda: [torch.tensor(-1.0)]
    cleanup = []
    optimizer._loss_scale_update_and_overflow_cleanup = lambda: cleanup.append(optimizer.overflow
                                                                               ) or optimizer.overflow
    optimizer.swap_optimizer = False

    optimizer.step()

    assert cleanup == [True]
    assert optimizer.overflow


def test_stage3_overflow_state_is_reset_for_next_non_fp16_step():
    optimizer = object.__new__(DeepSpeedZeroOptimizer_Stage3)
    optimizer.dtype = torch.bfloat16
    optimizer.overflow = True

    assert not optimizer._overflow_check_and_loss_scale_update(update_scale=False)


def test_stage3_raw_overflow_replaces_stale_global_norm():
    optimizer = object.__new__(DeepSpeedZeroOptimizer_Stage3)
    optimizer.overflow = True
    optimizer._global_grad_norm = torch.tensor(3.0)
    optimizer.custom_loss_scaler = False
    optimizer.loss_scaler = SimpleNamespace(cur_scale=4.0)
    optimizer._update_scale = lambda overflow: setattr(optimizer.loss_scaler, "cur_scale", 2.0)
    optimizer._overflow_clean_up = lambda prev_scale: None
    optimizer._loco_err_buf_update = lambda overflow, scale: None

    assert optimizer._loss_scale_update_and_overflow_cleanup()
    assert optimizer._global_grad_norm == float("inf")


def test_bf16_optimizer_reports_invalid_step_and_recovers(monkeypatch):
    optimizer = object.__new__(BF16_Optimizer)
    optimizer.has_moe_layers = False
    optimizer.graph_harvesting = False
    optimizer.norm_type = 2
    optimizer.mpu = None
    optimizer.clip_grad = 0
    optimizer.grad_acc_dtype = torch.float32
    optimizer.fp32_groups_flat_partition = []
    optimizer.fp32_groups_gradient_flat_partition = []
    optimizer.get_grads_for_norm = lambda: ([], {})
    optimizer.clear_hp_grads = lambda: None
    optimizer.clear_lp_grads = lambda: None
    optimizer._lazy_init_hp_params_optimizer_state = lambda: None
    optimizer.update_lp_params = lambda: None
    steps = []
    optimizer.optimizer = SimpleNamespace(step=lambda: steps.append(True))
    norms = iter([torch.tensor(-1.0), torch.tensor(1.0)])
    monkeypatch.setattr("deepspeed.runtime.bf16_optimizer.get_global_norm_of_tensors", lambda **kwargs: next(norms))

    optimizer.step()
    assert optimizer.overflow
    assert not torch.isfinite(torch.tensor(optimizer._global_grad_norm))
    assert steps == []

    optimizer.step()
    assert not optimizer.overflow
    assert steps == [True]


def test_superoffload_rolls_back_only_subgroups_submitted_this_step():
    optimizer = object.__new__(SuperOffloadOptimizer_Stage3)
    optimizer._submitted_cpu_sub_groups = {1}
    optimizer.sub_group_to_group_id = {0: 10, 1: 11}
    parameter = SimpleNamespace(data=torch.tensor([1.0]), grad=SimpleNamespace(data=torch.tensor([2.0])))
    optimizer.fp32_partitioned_groups_flat = [parameter, parameter]
    rollbacks = []
    optimizer._sync_cpu_optimizer_step = lambda *args, **kwargs: rollbacks.append((args, kwargs))

    optimizer._handle_overflow_rollback()

    assert [args[1] for args, _ in rollbacks] == [1]
    assert rollbacks[0][1]["rollback"]
    assert optimizer._submitted_cpu_sub_groups == set()


class TestZeROBFloat16Stage3InvalidNormRecovery(DistributedTest):
    world_size = 2

    def test_valid_step_after_invalid_norm_is_applied(self):
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "bf16": {
                "enabled": True,
            },
            "zero_optimization": {
                "stage": 3,
                "stage3_param_persistence_threshold": 1e5,
            }
        }
        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=2,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        batches = list(data_loader)

        run_model_step(model, batches[0][0], batches[0][1], float("nan"))
        assert not model.was_step_applied()
        assert model.skipped_steps == 1

        run_model_step(model, batches[1][0], batches[1][1], 0.1)
        assert model.was_step_applied()
        assert model.skipped_steps == 1
        assert all([not has_inf_or_nan(p) for p in model.parameters()])
