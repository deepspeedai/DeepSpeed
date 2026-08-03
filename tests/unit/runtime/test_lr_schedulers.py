# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math

import torch
import deepspeed
import pytest
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader
from deepspeed.runtime.lr_schedules import LR_RANGE_TEST, LR_RANGE_TEST_MIN_LR, LR_RANGE_TEST_STEP_RATE, LR_RANGE_TEST_STEP_SIZE, LR_RANGE_TEST_STAIRCASE
from deepspeed.runtime.lr_schedules import WARMUP_LR, WARMUP_MIN_LR, WARMUP_MAX_LR, WARMUP_NUM_STEPS, WARMUP_TYPE, WARMUP_LOG_RATE, WARMUP_LINEAR_RATE
from deepspeed.runtime.lr_schedules import ONE_CYCLE, CYCLE_MIN_LR, CYCLE_MAX_LR, CYCLE_FIRST_STEP_SIZE, DECAY_LR_RATE, DECAY_STEP_SIZE
from deepspeed.runtime.lr_schedules import CYCLE_MIN_MOM, CYCLE_MAX_MOM, DECAY_MOM_RATE
from deepspeed.runtime.lr_schedules import WARMUP_DECAY_LR, TOTAL_NUM_STEPS
from deepspeed.runtime.lr_schedules import WARMUP_COSINE_LR, WARMUP_MIN_RATIO, COS_MIN_RATIO, WarmupCosineLR
from deepspeed.runtime.lr_schedules import WarmupLR, WarmupDecayLR, LRRangeTest, OneCycle


def _verify_continuous_decrease(values):
    for i in range(len(values) - 1):
        assert values[i] > values[i + 1]


def _verify_continuous_increase(values):
    for i in range(len(values) - 1):
        assert values[i] < values[i + 1]


def _verify_staircase_increase(values, step_size):
    num_values = len(values)
    for i in range(0, num_values, step_size):
        j = min(i + step_size, num_values)
        assert all([values[i] == v for v in values[i:j]])


@pytest.mark.parametrize("scheduler_type,params", [(WARMUP_LR, {}),
                                                   (WARMUP_DECAY_LR, {
                                                       WARMUP_NUM_STEPS: 10,
                                                       TOTAL_NUM_STEPS: 20
                                                   }), (WARMUP_COSINE_LR, {
                                                       WARMUP_NUM_STEPS: 10,
                                                       TOTAL_NUM_STEPS: 20
                                                   }), (ONE_CYCLE, {
                                                       CYCLE_MIN_LR: 0,
                                                       CYCLE_MAX_LR: 0.1
                                                   }), (LR_RANGE_TEST, {})])
class TestGetLrBeforeTrain(DistributedTest):
    world_size = 1

    def test(self, scheduler_type, params):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                },
            },
            "scheduler": {
                "type": scheduler_type,
                "params": params
            },
            "gradient_clipping": 1.0
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim, empty_grad=False)
        model, _, _, lr_scheduler = deepspeed.initialize(config=config_dict,
                                                         model=model,
                                                         model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)

        true_lrs = lr_scheduler.get_lr()
        for group, true_lr in zip(model.optimizer.param_groups, true_lrs):
            assert group['lr'] == true_lr, f"True lr {true_lr}, optimizer lr {group['lr']}"

        for n, batch in enumerate(data_loader):
            # get lr before training starts
            lr_scheduler.get_lr()
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize("warmup_num_steps", [10, 15, 19, 33])
@pytest.mark.parametrize("warmup_type", [WARMUP_LOG_RATE, WARMUP_LINEAR_RATE])
class TestLrSchedule(DistributedTest):
    world_size = 1

    def test_lr_warmup_schedule(self, warmup_num_steps, warmup_type):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                },
            },
            "scheduler": {
                "type": WARMUP_LR,
                "params": {
                    WARMUP_MIN_LR: 0.1,
                    WARMUP_MAX_LR: 0.2,
                    WARMUP_NUM_STEPS: warmup_num_steps,
                    WARMUP_TYPE: warmup_type,
                }
            },
            "gradient_clipping": 1.0
        }
        schedule_params = config_dict["scheduler"]["params"]
        total_num_steps = 2 * warmup_num_steps
        hidden_dim = 10

        model = SimpleModel(hidden_dim, empty_grad=False)
        model, _, _, lr_scheduler = deepspeed.initialize(config=config_dict,
                                                         model=model,
                                                         model_parameters=model.parameters())

        data_loader = random_dataloader(model=model,
                                        total_samples=total_num_steps * 2,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)
        step_lrs = []
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            step_lrs.append(lr_scheduler.get_lr())

        # Verify initial lr
        assert step_lrs[0] == [schedule_params[WARMUP_MIN_LR]]

        # Verify warmup completion
        warmup_num_steps = schedule_params[WARMUP_NUM_STEPS]
        warmup_max_lr = [schedule_params[WARMUP_MAX_LR]]
        assert step_lrs[warmup_num_steps] == warmup_max_lr

        # Verify post-warmup completion
        assert all([warmup_max_lr == lr for lr in step_lrs[warmup_num_steps:]])

    def test_lr_warmup_decay_schedule(self, warmup_num_steps, warmup_type):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                },
            },
            "scheduler": {
                "type": WARMUP_DECAY_LR,
                "params": {
                    WARMUP_MIN_LR: 0.1,
                    WARMUP_MAX_LR: 0.2,
                    WARMUP_NUM_STEPS: warmup_num_steps,
                    TOTAL_NUM_STEPS: warmup_num_steps * 2,
                    WARMUP_TYPE: warmup_type
                }
            },
            "gradient_clipping": 1.0
        }
        schedule_params = config_dict["scheduler"]["params"]
        total_num_steps = schedule_params[TOTAL_NUM_STEPS]
        hidden_dim = 10

        model = SimpleModel(hidden_dim, empty_grad=False)
        model, _, _, lr_scheduler = deepspeed.initialize(config=config_dict,
                                                         model=model,
                                                         model_parameters=model.parameters())

        data_loader = random_dataloader(model=model,
                                        total_samples=total_num_steps * 2,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)
        step_lrs = []
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            step_lrs.append(lr_scheduler.get_lr())

        # Verify initial lr
        assert step_lrs[0] == [schedule_params[WARMUP_MIN_LR]]

        # Verify lr at warmup completion
        warmup_num_steps = schedule_params[WARMUP_NUM_STEPS]
        warmup_max_lr = [schedule_params[WARMUP_MAX_LR]]
        assert step_lrs[warmup_num_steps] == warmup_max_lr

        # Verify decay phase
        previous_lr = warmup_max_lr
        for lr in step_lrs[warmup_num_steps + 1:]:
            assert lr < previous_lr
            previous_lr = lr


@pytest.mark.parametrize("scheduler_type,params", [(WARMUP_LR, {}),
                                                   (WARMUP_DECAY_LR, {
                                                       WARMUP_NUM_STEPS: 5,
                                                       TOTAL_NUM_STEPS: 10
                                                   }),
                                                   (ONE_CYCLE, {
                                                       CYCLE_MIN_LR: 0,
                                                       CYCLE_MAX_LR: 0.1,
                                                       CYCLE_FIRST_STEP_SIZE: 5,
                                                       DECAY_STEP_SIZE: 5
                                                   }),
                                                   (LR_RANGE_TEST, {
                                                       LR_RANGE_TEST_MIN_LR: 1e-4,
                                                       LR_RANGE_TEST_STEP_SIZE: 1
                                                   })])
class TestSchedulerOptimizerParity(DistributedTest):
    world_size = 1

    def test(self, scheduler_type, params):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                },
            },
            "scheduler": {
                "type": scheduler_type,
                "params": params
            },
            "gradient_clipping": 1.0
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim, empty_grad=False)
        model, _, _, lr_scheduler = deepspeed.initialize(config=config_dict,
                                                         model=model,
                                                         model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            assert lr_scheduler.get_lr() == model.get_lr()


@pytest.mark.parametrize("min_lr, step_rate, step_size, staircase",
                         [(1e-4, 1e-5, 1, True),
                          (1e-5, 1e-5, 1, False),
                          (1e-4, 1e-3, 10, True),
                          (1e-3, 1e-3, 10, False),
                          (1e-2, 1e-2, 19, True),
                          (1e-2, 1e-2, 19, False)
                           ])# yapf: disable
class TestLrRange(DistributedTest):
    world_size = 1

    def test(self, min_lr, step_rate, step_size, staircase):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                },
            },
            "scheduler": {
                "type": LR_RANGE_TEST,
                "params": {
                    LR_RANGE_TEST_MIN_LR: min_lr,
                    LR_RANGE_TEST_STEP_RATE: step_rate,
                    LR_RANGE_TEST_STEP_SIZE: step_size,
                    LR_RANGE_TEST_STAIRCASE: staircase
                }
            },
            "gradient_clipping": 1.0
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim, empty_grad=False)
        model, _, _, lr_scheduler = deepspeed.initialize(config=config_dict,
                                                         model=model,
                                                         model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=max(50, step_size * 2),
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)

        step_lrs = []
        for _, batch in enumerate(data_loader):
            step_lrs.extend(lr_scheduler.get_lr())
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        # Verify starting lr
        assert step_lrs[0] == min_lr

        if staircase:
            # Verify staircase increasing lr
            _verify_staircase_increase(step_lrs, step_size)
        else:
            # Verify continuous increasing lr
            _verify_continuous_increase(step_lrs)


class TestOneCycle(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("min_lr, max_lr, decay_rate, cycle_step_size, decay_step_size",
                             [
                                 (1e-5, 1e-2, 1e-3, 10, 10),
                                 (1e-3, 1e-1, 0, 21, 21),
                                 (1e-5, 1e-2, 1e-3, 10, 10),
                                 (1e-3, 1e-1, 1e-1, 21, 21),
                                 (1e-5, 1e-1, 0, 10, 0),
                             ])  # yapf: disable
    def test_lr(self, min_lr, max_lr, decay_rate, cycle_step_size, decay_step_size):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                },
            },
            "scheduler": {
                "type": ONE_CYCLE,
                "params": {
                    CYCLE_MIN_LR: min_lr,
                    CYCLE_MAX_LR: max_lr,
                    DECAY_LR_RATE: decay_rate,
                    CYCLE_FIRST_STEP_SIZE: cycle_step_size,
                    DECAY_STEP_SIZE: decay_step_size
                }
            },
            "gradient_clipping": 1.0
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim, empty_grad=False)
        model, _, _, lr_scheduler = deepspeed.initialize(config=config_dict,
                                                         model=model,
                                                         model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=max(50, cycle_step_size * 3),
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)

        step_lrs = []
        for _, batch in enumerate(data_loader):
            step_lrs.extend(lr_scheduler.get_lr())
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        # Verify starting lr
        assert step_lrs[0] == min_lr

        # Verify peak lr
        assert step_lrs[cycle_step_size] == max_lr

        # Verify increasing phase
        _verify_continuous_increase(step_lrs[:cycle_step_size])

        # Verify decreasing phase
        _verify_continuous_decrease(step_lrs[cycle_step_size:(cycle_step_size * 2)])

        # Verify decay phase
        if decay_rate > 0:
            _verify_continuous_decrease(step_lrs[(cycle_step_size * 2):])

    @pytest.mark.parametrize("min_mom, max_mom, decay_rate, step_size",
                             [
                                 (0.08, 0.09, 1e-3, 10),
                                 (0.08, 0.09, 0, 21),
                                 (0.08, 0.09, 1e-3, 10),
                                 (0.08, 0.09, 0, 21),
                             ]) # yapf: disable
    def test_mom(self, min_mom, max_mom, decay_rate, step_size):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                },
            },
            "scheduler": {
                "type": ONE_CYCLE,
                "params": {
                    CYCLE_MIN_LR: 1e-3,
                    CYCLE_MAX_LR: 1e-2,
                    CYCLE_MIN_MOM: min_mom,
                    CYCLE_MAX_MOM: max_mom,
                    DECAY_MOM_RATE: decay_rate,
                    CYCLE_FIRST_STEP_SIZE: step_size,
                    DECAY_STEP_SIZE: step_size
                }
            },
            "gradient_clipping": 1.0
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim, empty_grad=False)
        model, _, _, lr_scheduler = deepspeed.initialize(config=config_dict,
                                                         model=model,
                                                         model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=max(50, step_size * 3),
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)

        step_moms = []
        for _, batch in enumerate(data_loader):
            step_moms.append(lr_scheduler.get_mom())
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        # Verify starting lr
        assert step_moms[0][0][0] == max_mom

        # Verify peak lr
        assert step_moms[step_size][0][0] == min_mom

        # Verify decreasing phase
        _verify_continuous_decrease(step_moms[:step_size])

        # Verify increasing phase
        _verify_continuous_increase(step_moms[step_size:(step_size * 2)])

        # Verify decay phase
        if decay_rate > 0:
            _verify_continuous_increase(step_moms[(step_size * 2):])


class TestWarmupCosineLR(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("total_num_steps, warmup_num_steps, cos_min_ratio, warmup_min_ratio",
                             [
                                 (100, 10, 0.1, 0.2),
                                 (200, 20, 0.1, 0.2),
                                 (500, 30, 0.0, 0.2),
                                 (600, 300, 0.1, 0.0),
                                 (600, 550, 0.0, 0.0),
                             ])  # yapf: disable
    def test_lr(self, total_num_steps, warmup_num_steps, cos_min_ratio, warmup_min_ratio):
        opt_lr = 0.0015
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": opt_lr
                },
            },
            "scheduler": {
                "type": WARMUP_COSINE_LR,
                "params": {
                    TOTAL_NUM_STEPS: total_num_steps,
                    WARMUP_MIN_RATIO: warmup_min_ratio,
                    WARMUP_NUM_STEPS: warmup_num_steps,
                    COS_MIN_RATIO: cos_min_ratio,
                }
            },
            "gradient_clipping": 1.0
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim, empty_grad=False)
        model, _, _, lr_scheduler = deepspeed.initialize(config=config_dict,
                                                         model=model,
                                                         model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=max(50, total_num_steps * 3),
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)

        step_lrs = []
        for _, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            step_lrs.extend(lr_scheduler.get_lr())

        # Verify starting lr
        assert abs(step_lrs[0] - opt_lr * warmup_min_ratio) < 1e-7

        # Verify peak lr
        assert abs(step_lrs[warmup_num_steps - 1] - opt_lr) < 1e-7

        # Verify end lr
        assert abs(step_lrs[total_num_steps - 1] - opt_lr * cos_min_ratio) < 1e-7

        # Verify increasing phase
        _verify_continuous_increase(step_lrs[:warmup_num_steps])

        # Verify decreasing phase
        _verify_continuous_decrease(step_lrs[warmup_num_steps:total_num_steps])


def test_warmup_cosine_lr_initializes_all_param_groups():
    dense = torch.nn.Parameter(torch.zeros(1))
    expert = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([{"params": [dense], "lr": 0.0015}, {"params": [expert], "lr": 0.003}])

    scheduler = WarmupCosineLR(optimizer=optimizer, total_num_steps=100, warmup_num_steps=10, warmup_min_ratio=0.0)

    assert scheduler.get_lr_ratio() == 0.0
    assert scheduler.get_lr() == [0.0, 0.0]
    assert scheduler.get_last_lr() == [0.0, 0.0]
    assert [group["lr"] for group in optimizer.param_groups] == [0.0, 0.0]

    scheduler.step(1)

    expected_ratio = math.log(2) / math.log(10)
    expected_lrs = [0.0015 * expected_ratio, 0.003 * expected_ratio]

    assert scheduler.get_lr_ratio() == pytest.approx(expected_ratio)
    assert scheduler.get_lr() == pytest.approx(expected_lrs)
    assert scheduler.get_last_lr() == pytest.approx(expected_lrs)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(expected_lrs)


def test_warmup_lr_inherits_per_group_lr_when_max_unspecified():
    # With warmup_max_lr unspecified, WarmupLR inherits the optimizer's lr; on a
    # multi-group optimizer it must inherit EACH group's own lr, not group 0's for
    # all groups. Regression for the trailing [0] on the fallback, which reduced the
    # per-group list to group 0's scalar and _format_param then broadcast it, so the
    # other groups' configured LRs were silently discarded.
    dense = torch.nn.Parameter(torch.zeros(1))
    expert = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([{"params": [dense], "lr": 0.1}, {"params": [expert], "lr": 0.2}])

    scheduler = WarmupLR(optimizer=optimizer, warmup_num_steps=10)

    assert scheduler.max_lrs == [0.1, 0.2]

    # Step to the end of warmup (gamma == 1.0): each group reaches its own base lr.
    scheduler.step(10)
    assert scheduler.get_lr() == pytest.approx([0.1, 0.2])
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.1, 0.2])


@pytest.mark.parametrize("lr_shape", [(), (1, )])
def test_warmup_lr_preserves_tensor_lr(lr_shape):
    param = torch.nn.Parameter(torch.zeros(1))
    initial_lr = torch.full(lr_shape, 0.1, dtype=torch.float64)
    optimizer = torch.optim.SGD([param], lr=initial_lr)

    scheduler = WarmupLR(optimizer=optimizer, warmup_num_steps=10)

    assert optimizer.param_groups[0]["lr"] is initial_lr
    assert initial_lr.shape == lr_shape
    assert initial_lr.dtype == torch.float64
    assert initial_lr.item() == 0.0

    scheduler.step(1)

    assert optimizer.param_groups[0]["lr"] is initial_lr
    assert initial_lr.shape == lr_shape
    assert initial_lr.dtype == torch.float64
    assert initial_lr.item() == pytest.approx(0.1 * math.log(2) / math.log(10))


@pytest.mark.parametrize("lr_shape", [(), (1, )])
def test_warmup_cosine_lr_preserves_tensor_lr(lr_shape):
    param = torch.nn.Parameter(torch.zeros(1))
    initial_lr = torch.full(lr_shape, 0.1, dtype=torch.float64)
    optimizer = torch.optim.SGD([param], lr=initial_lr)

    scheduler = WarmupCosineLR(optimizer=optimizer, total_num_steps=100, warmup_num_steps=10)

    assert optimizer.param_groups[0]["lr"] is initial_lr
    assert initial_lr.shape == lr_shape
    assert initial_lr.dtype == torch.float64
    assert initial_lr.item() == 0.0

    scheduler.step(1)

    assert optimizer.param_groups[0]["lr"] is initial_lr
    assert initial_lr.shape == lr_shape
    assert initial_lr.dtype == torch.float64
    assert initial_lr.item() == pytest.approx(0.1 * math.log(2) / math.log(10))


@pytest.mark.parametrize("lr_shape", [(), (1, )])
def test_one_cycle_preserves_tensor_lr(lr_shape):
    param = torch.nn.Parameter(torch.zeros(1))
    initial_lr = torch.full(lr_shape, 0.1, dtype=torch.float64)
    optimizer = torch.optim.SGD([param], lr=initial_lr)

    scheduler = OneCycle(optimizer=optimizer,
                         cycle_min_lr=0.01,
                         cycle_max_lr=0.1,
                         cycle_first_step_size=10,
                         cycle_second_step_size=10,
                         cycle_momentum=False)

    assert optimizer.param_groups[0]["lr"] is initial_lr
    assert initial_lr.shape == lr_shape
    assert initial_lr.dtype == torch.float64
    assert initial_lr.item() == pytest.approx(0.01)

    scheduler.step(1)

    assert optimizer.param_groups[0]["lr"] is initial_lr
    assert initial_lr.shape == lr_shape
    assert initial_lr.dtype == torch.float64
    assert initial_lr.item() == pytest.approx(0.01 + (0.1 - 0.01) * 2 / 10)


def test_warmup_cosine_lr_total_num_steps_equals_warmup_num_steps():
    # total_num_steps == warmup_num_steps must not raise ZeroDivisionError, and because the
    # cosine decay window is empty, every step past warmup must stay at cos_min_ratio rather
    # than oscillating back up. The sibling WarmupDecayLR guards the same denominator with
    # max(1, ...); WarmupCosineLR must too, and additionally clamp the cosine progress.
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([{"params": [param], "lr": 0.01}])
    cos_min_ratio = 0.1

    scheduler = WarmupCosineLR(optimizer=optimizer,
                               total_num_steps=10,
                               warmup_num_steps=10,
                               cos_min_ratio=cos_min_ratio)

    for step in range(scheduler.warmup_num_steps, scheduler.warmup_num_steps + 5):
        scheduler.step(step)
        assert scheduler.get_lr_ratio() == pytest.approx(cos_min_ratio)


@pytest.mark.parametrize("scheduler_cls", [WarmupLR, WarmupDecayLR, WarmupCosineLR])
@pytest.mark.parametrize("bad_warmup_num_steps", [None, -5])
def test_warmup_schedulers_reject_invalid_warmup_num_steps(scheduler_cls, bad_warmup_num_steps):
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([param], lr=0.001)

    kwargs = {"optimizer": optimizer, "warmup_num_steps": bad_warmup_num_steps}
    if scheduler_cls in (WarmupDecayLR, WarmupCosineLR):
        kwargs["total_num_steps"] = 100

    with pytest.raises(ValueError):
        scheduler_cls(**kwargs)


def test_warmup_cosine_lr_unknown_warmup_type_falls_back_to_log():
    # WarmupLR warns and falls back to the log warmup curve for an unrecognized
    # warmup_type; WarmupCosineLR must do the same instead of crashing with an
    # UnboundLocalError in get_lr_ratio on the first step.
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([param], lr=0.001)

    scheduler = WarmupCosineLR(optimizer=optimizer,
                               total_num_steps=100,
                               warmup_num_steps=10,
                               warmup_type="not_a_warmup_type")

    assert scheduler.warmup_type == WARMUP_LOG_RATE

    param_ref = torch.nn.Parameter(torch.zeros(1))
    optimizer_ref = torch.optim.Adam([param_ref], lr=0.001)
    scheduler_ref = WarmupCosineLR(optimizer=optimizer_ref,
                                   total_num_steps=100,
                                   warmup_num_steps=10,
                                   warmup_type=WARMUP_LOG_RATE)

    for step in range(15):
        scheduler.step(step)
        scheduler_ref.step(step)
        assert scheduler.get_lr_ratio() == pytest.approx(scheduler_ref.get_lr_ratio())


def test_warmup_cosine_lr_linear_warmup_type_produces_linear_ratios():
    # No other test exercises WarmupCosineLR with warmup_type=WARMUP_LINEAR_RATE,
    # so a regression that silently routed 'linear' through the log curve would
    # keep the suite green. Pin the per-step warmup ratios: with
    # warmup_min_ratio=0.0 the linear curve is step / warmup_num_steps, which
    # clearly diverges from the log curve (e.g. 0.1 vs ~0.301 at step 1).
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([param], lr=0.001)
    warmup_num_steps = 10

    scheduler = WarmupCosineLR(optimizer=optimizer,
                               total_num_steps=100,
                               warmup_num_steps=warmup_num_steps,
                               warmup_min_ratio=0.0,
                               warmup_type=WARMUP_LINEAR_RATE)

    assert scheduler.warmup_type == WARMUP_LINEAR_RATE

    for step in range(warmup_num_steps):
        scheduler.step(step)
        assert scheduler.get_lr_ratio() == pytest.approx(step / warmup_num_steps)


def _one_cycle_lrs(first_stair_count, steps=16, first_step_size=8, second_step_size=8, second_stair_count=None):
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([{"params": [param], "lr": 0.001}], betas=(0.9, 0.99))
    scheduler = OneCycle(optimizer=optimizer,
                         cycle_min_lr=0.001,
                         cycle_max_lr=0.01,
                         cycle_first_step_size=first_step_size,
                         cycle_second_step_size=second_step_size,
                         cycle_first_stair_count=first_stair_count,
                         cycle_second_stair_count=second_stair_count)
    lrs = []
    for _ in range(steps):
        scheduler.step()
        lrs.append(scheduler.get_lr()[0])
    return lrs


def test_one_cycle_stair_count_holds_lr_flat():
    # cycle_first_stair_count / cycle_second_stair_count are documented, exposed as CLI
    # flags and plumbed through the config, but were read nowhere, so every value produced
    # the continuous schedule. A stair count must hold the lr flat across each step of the
    # half cycle, and 0 must stay continuous.
    continuous = _one_cycle_lrs(0)
    assert len(set(continuous[:8])) == 8

    stairs = _one_cycle_lrs(2)
    assert stairs != continuous
    # two stairs over an eight-batch half cycle: the lr changes far less often
    assert len(set(stairs[:8])) < len(set(continuous[:8]))
    # and it is flat within a stair rather than moving every batch
    assert stairs[0] == stairs[1]

    # a stair count matching the half-cycle length is the continuous schedule again
    assert _one_cycle_lrs(8) == continuous


def test_one_cycle_stair_count_handles_asymmetric_cycle():
    continuous = _one_cycle_lrs(0, steps=31, first_step_size=10, second_step_size=21, second_stair_count=0)

    # Floating point roundoff in the normalized scale must not floor away the peak or
    # shift an exact batch-aligned stair down by one.
    batch_aligned = _one_cycle_lrs(10, steps=31, first_step_size=10, second_step_size=21, second_stair_count=21)
    assert batch_aligned == pytest.approx(continuous)
    assert batch_aligned[9] == pytest.approx(0.01)

    # The first and second stair counts must apply only to their respective halves.
    first_only = _one_cycle_lrs(2, steps=31, first_step_size=10, second_step_size=21, second_stair_count=0)
    second_only = _one_cycle_lrs(0, steps=31, first_step_size=10, second_step_size=21, second_stair_count=3)
    assert first_only[:10] != continuous[:10]
    assert first_only[10:] == continuous[10:]
    assert second_only[:10] == continuous[:10]
    assert second_only[10:30] != continuous[10:30]


@pytest.mark.parametrize("bad_step_size", [0, -5])
def test_lr_range_test_rejects_nonpositive_step_size(bad_step_size):
    # lr_range_test_step_size divides the step index in _continuous_interval and
    # _staircase_interval, so the first step() with a value of 0 raises ZeroDivisionError.
    # Mirror the WarmupLR positive-integer guard and reject the misconfig at construction.
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.1)

    with pytest.raises(ValueError):
        LRRangeTest(optimizer, lr_range_test_step_size=bad_step_size)


@pytest.mark.parametrize("first, second", [(0, 0), (0, None), (-1, None), (0, 100), (100, -1)])
def test_one_cycle_rejects_nonpositive_step_sizes(first, second):
    # _initialize_cycle divides cycle_first_step_size by total_size, and _get_scale_factor
    # then divides by the resulting step_ratio. A total of 0 raises ZeroDivisionError at
    # construction; a zero first half keeps total_size positive but sets step_ratio to 0,
    # so the first get_lr() raises instead. Reject both shapes with a clear ValueError.
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.1)

    with pytest.raises(ValueError):
        OneCycle(optimizer,
                 cycle_min_lr=0.001,
                 cycle_max_lr=0.1,
                 cycle_first_step_size=first,
                 cycle_second_step_size=second)


def test_one_cycle_allows_zero_second_step_size():
    # The mirror case is not degenerate: a zero second half gives step_ratio 1.0, and x
    # stays below 1.0 in _get_scale_factor, so no division by zero is reachable. Pin it so
    # the guard above does not grow into rejecting a working configuration.
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.1)

    scheduler = OneCycle(optimizer,
                         cycle_min_lr=0.001,
                         cycle_max_lr=0.1,
                         cycle_first_step_size=100,
                         cycle_second_step_size=0)

    assert scheduler.step_ratio == 1.0
    assert scheduler.get_lr() == [pytest.approx(0.001)]
    for _ in range(3):
        scheduler.step()
    assert scheduler.get_lr()[0] > 0.001
