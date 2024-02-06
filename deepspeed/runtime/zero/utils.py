# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from typing import List

import torch
from deepspeed import comm as dist
from deepspeed.utils import logger
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.lion import DeepSpeedCPULion, FusedLion
from deepspeed.utils.nvtx import instrument_w_nvtx
from deepspeed.accelerator import get_accelerator


def _initialize_parameter_parallel_groups(parameter_parallel_size=None):
    data_parallel_size = int(dist.get_world_size())
    parameter_parallel_size = parameter_parallel_size or data_parallel_size
    logger.info("data_parallel_size: %s, parameter_parallel_size: %s", data_parallel_size, parameter_parallel_size)
    assert data_parallel_size % parameter_parallel_size == 0, \
        'world size should be divisible by parameter parallel size'
    rank = dist.get_rank()
    my_group = None
    for i in range(data_parallel_size // parameter_parallel_size):
        ranks = range(i * parameter_parallel_size, (i + 1) * parameter_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            my_group = group
    return my_group


class ZeRORuntimeException(Exception):
    pass


ZERO_SUPPORTED_OPTIMIZERS = [
    torch.optim.Adam, torch.optim.AdamW, FusedAdam, DeepSpeedCPUAdam, torch.optim.Adagrad, DeepSpeedCPUAdagrad,
    DeepSpeedCPULion, FusedLion
]

# Add apex FusedAdam to supported list if apex is installed
try:
    import apex
    if hasattr(apex, 'optimizers') and hasattr(apex.optimizers, 'FusedAdam'):
        ZERO_SUPPORTED_OPTIMIZERS.append(apex.optimizers.FusedAdam)
except ImportError:
    pass


def is_zero_supported_optimizer(optimizer):
    if dist.get_rank() == 0:
        logger.info(f'Checking ZeRO support for optimizer={optimizer.__class__.__name__} type={type(optimizer)}')
    return type(optimizer) in ZERO_SUPPORTED_OPTIMIZERS


def get_lst_from_rank0(lst: List[int]) -> None:
    """
    NOTE: creates both communication and synchronization overhead so should be used
    sparingly
    """
    lst_tensor = torch.tensor(
        lst if dist.get_rank() == 0 else [-1] * len(lst),
        dtype=int,
        # device=get_accelerator().current_device_name(),
        device=torch.device(get_accelerator().device_name(os.environ["LOCAL_RANK"])),
        requires_grad=False,
    )
    dist.broadcast(lst_tensor, src=0, async_op=False)

    return list(lst_tensor.cpu().numpy())


@instrument_w_nvtx
def assert_ints_same_as_other_ranks(ints: List[int]) -> None:
    """
    NOTE: creates both communication and synchronization overhead so should be
    used sparingly

    takes a list of ints from each rank and ensures that they are the same
    across ranks, throwing an exception if they are not.
    """
    rank0_ints = get_lst_from_rank0(ints)
    if ints != rank0_ints:
        raise RuntimeError(f"disagreement between rank0 and rank{dist.get_rank()}: "
                           f"rank0: {rank0_ints}, rank{dist.get_rank()}: {ints}")


def is_builtin_type(obj):
    # https://stackoverflow.com/a/17795199
    return obj.__class__.__module__ == '__builtin__' or obj.__class__.__module__ == "builtins"


def isinstance_namedtuple(obj: object) -> bool:
    """
    Is this an instance of namedtuple/NamedTuple?
    From: https://stackoverflow.com/a/62692640

    Args:
        obj (object): An object.

    Returns:
        bool: True if namedtuple/NamedTuple else False.
    """
    return isinstance(obj, tuple) and hasattr(obj, '_asdict') and hasattr(obj, '_fields')


def is_zero_param(parameter):
    if not torch.is_tensor(parameter):
        return False
    return hasattr(parameter, 'ds_id')


def apply_to_tensors_only(function, value, warning_msg_fn=None):
    """
    Apply `function` to every Tensor in `value`.

    Args:
        functional: The function class to apply.
        value (Any): Target object to apply `function` to.

    Returns:
        Any: Output of `function`.
    """
    if isinstance(value, (tuple, list)):
        touched_outputs = []
        for elem in value:
            touched_output = apply_to_tensors_only(function, elem)
            touched_outputs.append(touched_output)

        if isinstance_namedtuple(value):
            # namedtuples require a slightly different syntax.
            return value.__class__(*touched_outputs)

        return value.__class__(touched_outputs)
    elif isinstance(value, dict):
        # apply inplace to avoid recreating dict inherited objects
        for key in value.keys():
            value[key] = apply_to_tensors_only(function, value[key])
        return value

    elif isinstance(value, torch.Tensor):
        # this also applies to torch.Tensor's subclasses like torch.nn.parameter.Parameter
        touched_output = function(value)

        # restore zero param attributes if those get stripped by `backward_function`
        if not is_zero_param(touched_output) and is_zero_param(value):
            touched_output.ds_param_alias = value

        return touched_output
    else:
        if not is_builtin_type(value):
            global warned
            if warning_msg_fn and not warned and dist.get_rank() == 0:
                logger.warning(warning_msg_fn(value))
                warned = True
        return value
