# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import inspect
from deepspeed.utils import get_caller_func


def get_local_rank_from_launcher():

    # DeepSpeed launcher will set it so get from there
    rank = os.environ.get('LOCAL_RANK')

    if rank is None:
        rank = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK')

    # Make it a single process job and set rank to 0
    if rank is None:
        rank = 0

    return int(rank)


def get_world_rank_from_launcher():

    # DeepSpeed launcher will set it so get from there
    rank = os.environ.get('RANK')

    if rank is None:
        rank = os.environ.get('OMPI_COMM_WORLD_RANK')

    # Make it a single process job and set rank to 0
    if rank is None:
        rank = 0

    return int(rank)


def get_world_size_from_launcher():
    # DeepSpeed launcher will set it so get from there
    size = os.environ.get('WORLD_SIZE')
    rank = os.environ.get('RANK')

    if size is None:
        size = os.environ.get('OMPI_COMM_WORLD_SIZE')

    # Make it a single process job and set size to 1
    if size is None:
        size = 1

    if rank == 0:
        print(f"set world size to {size}")

    return int(size)


def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


# We need this hacky function since torch doesn't consistently name or place the input tensor args
def get_tensor_position(func):
    sig_params = inspect.signature(func).parameters
    arg = None
    # most colls
    if 'tensor' in sig_params:
        arg = 'tensor'
    # all_reduce_coalesced coll
    elif 'tensors' in sig_params:
        arg = 'tensors'
    # reduce scatter coll
    elif 'input_list' in sig_params:
        arg = 'input_list'
    # all_to_all and torch multiGPU colls
    elif 'input_tensor_list' in sig_params:
        arg = 'input_tensor_list'
    if arg is None:
        return -1
    else:
        return list(sig_params).index(arg)


def get_tensor_kwarg(func, kwargs):
    func_args = get_default_args(func)
    func_args.update(kwargs)
    arg = None

    if 'tensor' in func_args:
        arg = func_args['tensor']
    elif 'tensors' in func_args:
        arg = func_args['tensors']
    elif 'input_list' in func_args:
        arg = func_args['input_list']
    elif 'input_tensor_list' in func_args:
        arg = func_args['input_tensor_list']
    return arg


def get_msg_size_from_args(func, *args, **kwargs):
    # 3 cases:
    #   - tensor arg is in args
    #   - tensor arg is in kwargs
    #   - tensor arg is not present (e.g. barrier)
    tensor_arg_position = -1
    tensor_arg = None
    # check if tensor arg is in args
    if len(args) > 0:
        tensor_arg_position = get_tensor_position(func)
        if tensor_arg_position > -1:
            tensor_arg = args[get_tensor_position(func)]
    # check if tensor arg is in kwargs
    if tensor_arg is None and len(kwargs) > 0:
        tensor_arg = get_tensor_kwarg(func, kwargs)
    # if tensor arg is not present, no data is being transmitted
    if tensor_arg is None:
        return 0
    else:
        # Sum of tensor sizes for list colls such as torch's all_to_all
        # NOTE: msg_size for list colls will not be the actual size transmitted by a given MPI/NCCL call within the coll op. Instead, it's the total amount of data transmitted.
        if type(tensor_arg) is list:
            return sum(x.element_size() * x.nelement() for x in tensor_arg)
        else:
            return tensor_arg.element_size() * tensor_arg.nelement()


def get_debug_log_name(func_args, debug):
    if debug:
        return func_args['log_name'] + ' | [Caller Func: ' + get_caller_func() + ']'
    else:
        return func_args['log_name']
