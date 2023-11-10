# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

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
from deepspeed.runtime.lr_schedules import WARMUP_COSINE_LR, WARMUP_MIN_RATIO, COS_MIN_RATIO


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
