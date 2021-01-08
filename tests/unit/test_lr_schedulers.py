import torch
import deepspeed
import argparse
import pytest
import json
import os
from common import distributed_test
from simple_model import SimpleModel, SimpleOptimizer, random_dataloader, args_from_dict
from deepspeed.runtime.lr_schedules import LR_RANGE_TEST, LR_RANGE_TEST_MIN_LR, LR_RANGE_TEST_STEP_RATE, LR_RANGE_TEST_STEP_SIZE, LR_RANGE_TEST_STAIRCASE
from deepspeed.runtime.lr_schedules import WARMUP_LR, WARMUP_MIN_LR, WARMUP_MAX_LR, WARMUP_NUM_STEPS
from deepspeed.runtime.lr_schedules import ONE_CYCLE, CYCLE_MIN_LR, CYCLE_MAX_LR, CYCLE_FIRST_STEP_SIZE, DECAY_LR_RATE, DECAY_STEP_SIZE
from deepspeed.runtime.lr_schedules import CYCLE_MIN_MOM, CYCLE_MAX_MOM, DECAY_MOM_RATE
from deepspeed.runtime.lr_schedules import WARMUP_DECAY_LR, TOTAL_NUM_STEPS


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


@pytest.mark.parametrize("scheduler_type,params",
                         [(WARMUP_LR,
                           {}),
                          (WARMUP_DECAY_LR,
                           {
                               WARMUP_NUM_STEPS: 10,
                               TOTAL_NUM_STEPS: 20
                           }),
                          (ONE_CYCLE,
                           {
                               CYCLE_MIN_LR: 0,
                               CYCLE_MAX_LR: 0.1
                           }),
                          (LR_RANGE_TEST,
                           {})])
def test_get_lr_before_train(tmpdir, scheduler_type, params):
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
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[1])
    def _test_get_lr_before_train(args, model, hidden_dim):
        model, _, _, lr_scheduler = deepspeed.initialize(args=args,
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

    _test_get_lr_before_train(args=args, model=model, hidden_dim=hidden_dim)


@pytest.mark.parametrize("warmup_num_steps", [10, 15, 19, 33])
def test_lr_warmup_schedule(tmpdir, warmup_num_steps):
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
                WARMUP_NUM_STEPS: warmup_num_steps
            }
        },
        "gradient_clipping": 1.0
    }

    total_num_steps = 2 * warmup_num_steps

    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[1])
    def _test_lr_warmup_schedule(args, model, hidden_dim, schedule_params, num_steps):
        model, _, _, lr_scheduler = deepspeed.initialize(args=args,
                                                         model=model,
                                                         model_parameters=model.parameters())

        data_loader = random_dataloader(model=model,
                                        total_samples=num_steps * 2,
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

    _test_lr_warmup_schedule(args=args,
                             model=model,
                             hidden_dim=hidden_dim,
                             schedule_params=config_dict["scheduler"]["params"],
                             num_steps=total_num_steps)


@pytest.mark.parametrize("warmup_num_steps", [10, 15, 19, 33])
def test_lr_warmup_decay_schedule(tmpdir, warmup_num_steps):
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
                TOTAL_NUM_STEPS: warmup_num_steps * 2
            }
        },
        "gradient_clipping": 1.0
    }

    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[1])
    def _test_lr_warmup_decay_schedule(args,
                                       model,
                                       hidden_dim,
                                       schedule_params,
                                       num_steps):
        model, _, _, lr_scheduler = deepspeed.initialize(args=args,
                                                         model=model,
                                                         model_parameters=model.parameters())

        data_loader = random_dataloader(model=model,
                                        total_samples=num_steps * 2,
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

    schedule_params = config_dict["scheduler"]["params"]

    total_num_steps = schedule_params[TOTAL_NUM_STEPS]

    _test_lr_warmup_decay_schedule(args=args,
                                   model=model,
                                   hidden_dim=hidden_dim,
                                   schedule_params=schedule_params,
                                   num_steps=total_num_steps)


@pytest.mark.parametrize("scheduler_type,params",
                         [(WARMUP_LR,
                           {}),
                          (WARMUP_DECAY_LR,
                           {
                               WARMUP_NUM_STEPS: 5,
                               TOTAL_NUM_STEPS: 10
                           }),
                          (ONE_CYCLE,
                           {
                               CYCLE_MIN_LR: 0,
                               CYCLE_MAX_LR: 0.1,
                               CYCLE_FIRST_STEP_SIZE: 5,
                               DECAY_STEP_SIZE: 5
                           }),
                          (LR_RANGE_TEST,
                           {
                               LR_RANGE_TEST_MIN_LR: 1e-4,
                               LR_RANGE_TEST_STEP_SIZE: 1
                           })])
def test_scheduler_optimizer_parity(tmpdir, scheduler_type, params):
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
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[1])
    def _test_scheduler_optimizer_parity(args, model, hidden_dim):
        model, _, _, lr_scheduler = deepspeed.initialize(args=args,
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

    _test_scheduler_optimizer_parity(args=args, model=model, hidden_dim=hidden_dim)


@pytest.mark.parametrize("min_lr, step_rate, step_size, staircase",
                         [(1e-4, 1e-5, 1, True),
                          (1e-5, 1e-5, 1, False),
                          (1e-4, 1e-3, 10, True),
                          (1e-3, 1e-3, 10, False),
                          (1e-2, 1e-2, 19, True),
                          (1e-2, 1e-2, 19, False)
                           ])# yapf: disable
def test_lr_range_test(tmpdir, min_lr, step_rate, step_size, staircase):
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
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[1])
    def _test_lr_range_test(args, model, hidden_dim, min_lr, step_size, staircase):
        model, _, _, lr_scheduler = deepspeed.initialize(args=args,
                                                         model=model,
                                                         model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=max(50,
                                                          step_size * 2),
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)

        step_lrs = []
        for _, batch in enumerate(data_loader):
            step_lrs.append(lr_scheduler.get_lr())
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

    _test_lr_range_test(args=args,
                        model=model,
                        hidden_dim=hidden_dim,
                        min_lr=[min_lr],
                        step_size=step_size,
                        staircase=staircase)


@pytest.mark.parametrize("min_lr, max_lr, decay_rate, step_size",
                         [
                             (1e-5, 1e-2, 1e-3, 10),
                             (1e-3, 1e-1, 0, 21),
                             (1e-5, 1e-2, 1e-3, 10),
                             (1e-3, 1e-1, 0, 21),
                         ])  # yapf: disable
def test_onecycle_lr(tmpdir, min_lr, max_lr, decay_rate, step_size):
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
                CYCLE_FIRST_STEP_SIZE: step_size,
                DECAY_STEP_SIZE: step_size
            }
        },
        "gradient_clipping": 1.0
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[1])
    def _test_onecycle_lr(args,
                          model,
                          hidden_dim,
                          min_lr,
                          max_lr,
                          step_size,
                          decay_rate):
        model, _, _, lr_scheduler = deepspeed.initialize(args=args,
                                                         model=model,
                                                         model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=max(50,
                                                          step_size * 3),
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)

        step_lrs = []
        for _, batch in enumerate(data_loader):
            step_lrs.append(lr_scheduler.get_lr())
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        # Verify starting lr
        assert step_lrs[0] == min_lr

        # Verify peak lr
        assert step_lrs[step_size] == max_lr

        # Verify increasing phase
        _verify_continuous_increase(step_lrs[:step_size])

        # Verify decreasing phase
        _verify_continuous_decrease(step_lrs[step_size:(step_size * 2)])

        # Verify decay phase
        if decay_rate > 0:
            _verify_continuous_decrease(step_lrs[(step_size * 2):])

    _test_onecycle_lr(args=args,
                      model=model,
                      hidden_dim=hidden_dim,
                      min_lr=[min_lr],
                      max_lr=[max_lr],
                      step_size=step_size,
                      decay_rate=decay_rate)


@pytest.mark.parametrize("min_mom, max_mom, decay_rate, step_size",
                         [
                             (0.08, 0.09, 1e-3, 10),
                             (0.08, 0.09, 0, 21),
                             (0.08, 0.09, 1e-3, 10),
                             (0.08, 0.09, 0, 21),
                         ]) # yapf: disable
def test_onecycle_mom(tmpdir, min_mom, max_mom, decay_rate, step_size):
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
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    @distributed_test(world_size=[1])
    def _test_onecycle_mom(args,
                           model,
                           hidden_dim,
                           min_mom,
                           max_mom,
                           step_size,
                           decay_rate):
        model, _, _, lr_scheduler = deepspeed.initialize(args=args,
                                                         model=model,
                                                         model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=max(50,
                                                          step_size * 3),
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

    _test_onecycle_mom(args=args,
                       model=model,
                       hidden_dim=hidden_dim,
                       min_mom=min_mom,
                       max_mom=max_mom,
                       step_size=step_size,
                       decay_rate=decay_rate)
