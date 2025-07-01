# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepspeed.runtime.engine import DeepSpeedEngine


def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    transposed_tensors = [t.transpose(0, 1).contiguous() if t.dim() == 2 else t for t in tensors]
    return torch._C._nn.flatten_dense_tensors(transposed_tensors)


def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Args:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    transposed_tensors = [t.transpose(0, 1) if t.dim() == 2 else t for t in tensors]
    unflat = torch._C._nn.unflatten_dense_tensors(flat, transposed_tensors)
    return [t.transpose(0, 1) if t.dim() == 2 else t for t in unflat]


def configure_zenflow(engine: "DeepSpeedEngine") -> None:
    zenflow_config = engine.zenflow_config()
    if zenflow_config == None:
        engine.zenflow = False
        return

    engine.zenflow = True
    select_strategy = zenflow_config.select_strategy

    if select_strategy == 'auto':
        select_strategy = "epoch"
        if isinstance(zenflow_config.select_interval, int):
            raise Warning(
                "If use auto select strategy, select_interval will be set to 1 and select_strategy will be set to epoch, thus select_interval would be overwritten."
            )
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
        if engine.training_dataloader is not None:
            zenflow_config.steps_per_epoch = len(engine.training_dataloader)
            engine.select_interval = engine.select_interval * len(engine.training_dataloader)
        else:
            engine.select_interval = 0

    if not engine.auto_update and engine.select_interval != 0 and engine.select_interval < engine.update_interval:
        raise ValueError("Select interval must be greater or equal to update interval")

    engine.overlap_step = zenflow_config.overlap_step

    engine.full_warm_up_rounds = zenflow_config.full_warm_up_rounds

    engine._config.gradient_accumulation_steps = engine.update_interval
