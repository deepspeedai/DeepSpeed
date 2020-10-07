import torch
import deepspeed
import argparse
import pytest
import json
import os
from common import distributed_test
from simple_model import SimpleModel, SimpleOptimizer, random_dataloader, args_from_dict
from deepspeed.runtime.lr_schedules import LR_RANGE_TEST, ONE_CYCLE, WARMUP_LR, WARMUP_DECAY_LR
from deepspeed.runtime.lr_schedules import WARMUP_MIN_LR, WARMUP_MAX_LR, WARMUP_NUM_STEPS, TOTAL_NUM_STEPS
from deepspeed.runtime.lr_schedules import CYCLE_MIN_LR, CYCLE_MAX_LR


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
                               CYCLE_MAX_LR: 0
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
