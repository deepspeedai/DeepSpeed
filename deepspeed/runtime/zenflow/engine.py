# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed import comm as dist
from typing import TYPE_CHECKING
from deepspeed.utils.torch import required_torch_version
from deepspeed.utils import logger

if TYPE_CHECKING:
    from deepspeed.runtime.engine import DeepSpeedEngine


def configure_zenflow(engine: "DeepSpeedEngine") -> None:
    """Configure ZenFlow-related scheduling parameters on the engine.

    This function initializes ZenFlow flags (e.g., `zenflow`, `auto_update`,
    `select_interval`, etc.) based on the `zenflow_config` object. It handles
    selection/update strategy resolution and performs basic validation.

    Args:
        engine (DeepSpeedEngine): The DeepSpeed engine to configure.
    """
    zenflow_config = engine.zenflow_config()
    if zenflow_config == None:
        engine.zenflow = False
        return
    if not required_torch_version(min_version=2.1):
        raise ValueError(
            "Please use PyTorch 2.1 or later to enable ZenFlow. Alternatively, omit `zenflow` config in the config file to fall back to the default ZeRO-Offload optimizer."
        )

    engine.zenflow = True
    select_strategy = zenflow_config.select_strategy

    if select_strategy == 'auto':
        select_strategy = "epoch"
        if isinstance(zenflow_config.select_interval, int):
            logger.warning(
                "ZenFlow: select_strategy is 'auto', so select_interval is one epoch and the "
                "configured value %s is ignored.", zenflow_config.select_interval)
        engine.select_interval = 1
    else:
        if isinstance(zenflow_config.select_interval, str):
            raise ValueError("If don't use auto select strategy, select_interval must be a number.")
        engine.select_interval = zenflow_config.select_interval

    if isinstance(zenflow_config.update_interval, str):
        engine.auto_update = True
        engine.update_interval = 0
    else:
        engine.auto_update = False
        engine.update_interval = int(zenflow_config.update_interval)

    if select_strategy == 'epoch':
        # `steps_per_epoch` may already be set by the user; otherwise it can only
        # come from a dataloader DeepSpeed owns.
        if not zenflow_config.steps_per_epoch and engine.training_dataloader is not None:
            # An empty dataloader would assign 0, which the config validator now
            # rejects on assignment -- and a crash is the wrong answer for a
            # degenerate-but-legal dataloader. Fall through to the warning.
            epoch_steps = len(engine.training_dataloader)
            if epoch_steps > 0:
                zenflow_config.steps_per_epoch = epoch_steps
        if zenflow_config.steps_per_epoch:
            engine.select_interval = engine.select_interval * zenflow_config.steps_per_epoch
        else:
            # is_zenflow_select_boundary() treats 0 as "never again", so this
            # leaves the columns chosen at the first step in place for the whole
            # run. Say so rather than degrading in silence.
            logger.warning("ZenFlow: select_strategy resolves to 'epoch', but the number of steps in an epoch "
                           "is unknown -- DeepSpeed reads it from a dataloader it owns, and none was given to "
                           "deepspeed.initialize(). Important columns will be selected once and never "
                           "re-selected. Set \"steps_per_epoch\" in the zenflow config, or pass training_data= "
                           "to deepspeed.initialize(), or use \"select_strategy\": \"step\" with an explicit "
                           "\"select_interval\".")
            engine.select_interval = 0

    if not engine.auto_update and engine.select_interval != 0 and engine.select_interval < engine.update_interval:
        raise ValueError("Select interval must be greater or equal to update interval")

    engine.overlap_step = zenflow_config.overlap_step

    engine.full_warm_up_rounds = zenflow_config.full_warm_up_rounds

    engine._config.gradient_accumulation_steps = engine.update_interval


def is_zenflow_update_boundary(engine: "DeepSpeedEngine"):
    """Determine whether the current step is an update boundary for ZenFlow.

    This function checks whether the engine should trigger an optimizer update
    based on gradient accumulation, warmup phase, and selection/update intervals.

    Returns:
        bool: True if this step is an update boundary, otherwise False.
    """
    if engine.auto_update:
        if (engine.micro_steps + 1) <= engine.full_warm_up_rounds:
            return True
        return (engine.optimizer.zenflow_need_update[engine.optimizer.zenflow_state ^ 1]
                or (engine.select_interval != 0 and (engine.micro_steps + 1) % engine.select_interval == 0))
    else:
        if (engine.micro_steps + 1) < engine.full_warm_up_rounds:
            return True
        return ((engine.micro_steps + 1 - engine.full_warm_up_rounds) % engine.gradient_accumulation_steps() == 0
                or (engine.select_interval != 0 and (engine.micro_steps + 1) % engine.select_interval == 0))


def zenflow_step(engine: "DeepSpeedEngine", lr_kwargs):
    """Main step logic for ZenFlow update scheduling.

    This function performs either:
    - a selective optimizer update (if at accumulation boundary),
    - or just a learning rate scheduler step and logging (if at accumulation iteration).

    Args:
        engine (DeepSpeedEngine): The engine managing training state.
        lr_kwargs (dict): Optional kwargs passed to the LR scheduler step.
    """
    if engine.is_gradient_accumulation_boundary():
        if engine.micro_steps + 1 >= engine.full_warm_up_rounds:
            _take_selective_parameter_step(engine)
        if engine.auto_update:
            if dist.get_rank() == 0:
                print(f"Zenflow: This is an update iter. update_interval: {engine.update_interval}")
            engine.update_interval = 0
    else:
        _take_lr_scheduler_step(engine, lr_kwargs)
        _log_selective_optimizer_timers(engine)


def _take_selective_parameter_step(engine: "DeepSpeedEngine"):
    """
    Trigger a step on the selective optimizer.
    """
    engine.optimizer.selective_optimizer_step()


def _take_lr_scheduler_step(engine: "DeepSpeedEngine", lr_kwargs):
    """
    Take a step on the learning rate scheduler.
    """
    if engine.lr_scheduler is not None:
        try:
            engine.lr_scheduler.step(**(lr_kwargs or {}))
        except TypeError:
            # XXX Hack to work with Megatron 2.0 and DeepSpeed pipelines.
            # We don't currently have a way to specify lr_kwargs from
            # pipe_engine.train_batch()
            engine.lr_scheduler.step(engine.train_batch_size())


def _log_selective_optimizer_timers(engine):
    """
    Log the selective optimizer timers.
    """
    engine.optimizer.log_selective_optimizer_timers()


def sync_zenflow_optimizer_lr(engine: "DeepSpeedEngine"):
    """
    Synchronize the learning rate of the selective optimizer.
    If auto_update is enabled, increment the update interval.
    """
    engine.optimizer._sync_selective_optimizer_lr()
    if engine.auto_update:
        engine.update_interval += 1
