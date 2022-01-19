'''
Copyright 2021 The Microsoft DeepSpeed Team
'''

# The file has been adapted from https://github.com/NVIDIA/Megatron-LM and retains the following license from the original file

# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Support, expert, data, and model (only megatron-style) parallelism in DeepSpeed

Following are the possible scenarios:

Scenario 1 : There is no expert parallelism or model parallelism (D)
model = my_model(args)
engine = deepspeed.init(model) ---> initialize groups without mpu

Scenario 2 : There is expert parallelism but no model parallelism (E+D)
deepspeed.init_groups(args) --> groups will be initialized here
model = my_model(args)
engine = deepspeed.init(model) --> don't initialize groups

Scenario 3 : There is model parallelism but no expert parallelism (M)
mpu.init()
model = my_model(args)
engine = deepspeed.init(model, mpu = mpu)  --> initialize groups with mpu but expert_parallel_size = dp_world_size

Scenario 4 : There is model, data, and expert parallelism (E+D+M)
mpu.init()
deepspeed.init_groups(mpu, args)  ---> initialize groups with mpu
model = my_model(args)

#Valid but assert inside deepspeed to make sure mpu passed here is same as the one used to init the groups
engine = deepspeed.init(model, mpu = mpu)

#Also Valid
engine = deepspeed.init(model)

"""

import torch
from deepspeed.utils import logger, log_dist

# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Expert parallel group that the current rank belongs to.
_EXPERT_PARALLEL_GROUP = None  # {"32_expert": parallel_group}
# Expert data parallel group that the current rank belongs to.
_EXPERT_DATA_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# Max EP SIZE
_MAX_EP_SIZE = None
_MAX_EP_SIZE_NAME = None


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def initialize(ep_size=1, mpu=None, num_ep_list=None):
    """
    Process groups initialization supporting expert (E), data (D), and model (M) parallelism. DeepSpeed considers
    the following scenarios w.r.t. process group creation.

    * S1: There is no expert parallelism or model parallelism, only data (D)::

        model = my_model(args)
        engine = deepspeed.initialize(model) # initialize groups without mpu

    * S2: There is expert parallelism but no model parallelism (E+D)::

        deepspeed.utils.groups.initialize(ep_size) # groups will be initialized here
        model = my_model(args)
        engine = deepspeed.initialize(model)

    * S3: There is model parallelism but no expert parallelism (M)::

        mpu.init() # client initializes it's model parallel unit
        model = my_model(args)
        engine = deepspeed.initialize(model, mpu=mpu) # init w. mpu but ep_size = dp_world_size

    * S4: There is model, data, and expert parallelism (E+D+M)::

        mpu.init() # client initializes it's model parallel unit
        deepspeed.utils.groups.initialize(ep_size, mpu) # initialize expert groups wrt mpu
        model = my_model(args)
        engine = deepspeed.initialize(model, mpu=mpu) # passing mpu is optional in this case

    Arguments:
        ep_size (int, optional): default=1, maximum expert parallel size, which should be divisible/divided by the world size.
        by each element in num_ep_list.
        mpu (module, optional): default=None, model parallel unit (e.g., from Megatron)
            that describes model/data parallel ranks.
        num_ep_list (list, optional): default=None, list of number of expert parallel sizes in each MoE layer.

    """

    if num_ep_list is None:
        num_ep_list = [ep_size]

    assert max(num_ep_list) >= ep_size, f"ep_size={ep_size} is larger than the largest num_ep_list={max(num_ep_list)}, you should reduce expert parallel size"

    num_ep_list = list(set(num_ep_list))  # remove duplicates
    num_ep_list.sort()  # sort in ascending order
    for num_ep in num_ep_list:
        assert num_ep > 0, 'num_ep must be positive'
        assert num_ep % ep_size == 0 or ep_size % num_ep == 0, 'num_ep must be divisible/divided by ep_size'

    if mpu is not None:
        log_dist(message="initializing deepspeed groups using mpu", ranks=[0])
        initialize_model_and_expert_parallel(ep_size, mpu, num_ep_list)
    else:
        log_dist(message="initializing deepspeed groups", ranks=[0])
        initialize_model_parallel(1)
        initialize_expert_parallel(ep_size, num_ep_list)


def initialize_model_parallel(model_parallel_size_):
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel groups as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    log_dist(
        'initializing deepspeed model parallel group with size {}'.format(
            model_parallel_size_),
        [0])
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    model_parallel_size = min(model_parallel_size_, world_size)
    ensure_divisibility(world_size, model_parallel_size)
    rank = torch.distributed.get_rank()

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    for i in range(model_parallel_size):
        ranks = range(i, world_size, model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if i == (rank % model_parallel_size):
            _DATA_PARALLEL_GROUP = group

    # Build the model parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, \
        'model parallel group is already initialized'
    for i in range(world_size // model_parallel_size):
        ranks = range(i * model_parallel_size, (i + 1) * model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if i == (rank // model_parallel_size):
            _MODEL_PARALLEL_GROUP = group


def initialize_expert_parallel(expert_parallel_size_, num_ep_list_=None):
    """
        Initialize expert plus data parallel groups.

        Example - E + D parallel
        world_size = 16
        expert_parallel_size = 2 # number of experts in same group
        expert_data_parallel_group = [0,2,4,6,8,10,12,14], [1,3,5,7,9,11,13,15] - all reduce is only on MoE params
        expert_parallel_group = [0, 1], [2,3], [4,5], [6,7], [8,9] - no all reduce, but all to all
        data_parallel_group = [0,1,...,15] - all reduce is only on non-MoE
    """
    assert torch.distributed.is_initialized()

    global _MAX_EP_SIZE
    global _MAX_EP_SIZE_NAME
    _MAX_EP_SIZE = expert_parallel_size_
    _MAX_EP_SIZE_NAME = f"ep_size_{expert_parallel_size_}"

    if num_ep_list_ is None:
        num_ep_list_ = [expert_parallel_size_]

    log_dist(
        'initializing deepspeed expert parallel group with max size {} for number expert list {}'
        .format(expert_parallel_size_,
                num_ep_list_),
        [0])
    world_size = get_data_parallel_world_size()
    rank = get_data_parallel_rank()

    expert_parallel_size_ = min(expert_parallel_size_, world_size)
    ensure_divisibility(world_size, expert_parallel_size_)

    # Build the expert data parallel groups.
    global _EXPERT_DATA_PARALLEL_GROUP
    assert _EXPERT_DATA_PARALLEL_GROUP is None, \
        'expert data parallel group is already initialized'

    _EXPERT_DATA_PARALLEL_GROUP = {}

    for num_ep in num_ep_list_:
        # Build the data parallel groups for each num_ep
        # We will have two cases
        # 1. num_ep >= expert_parallel_size_, we can assign the same group to to num_ep from expert_parallel_size_ to num_ep
        # 2. num_ep < expert_parallel_size_, we will need to create the new group
        if num_ep >= expert_parallel_size_:
            if f"ep_size_{expert_parallel_size_}" not in _EXPERT_DATA_PARALLEL_GROUP:
                for i in range(expert_parallel_size_):
                    # generate all groups
                    ranks = range(i, world_size, expert_parallel_size_)
                    group = torch.distributed.new_group(ranks)
                    if i == (rank % expert_parallel_size_):
                        # get the correct group
                        _EXPERT_DATA_PARALLEL_GROUP[
                            f"ep_size_{expert_parallel_size_}"] = group
        else:
            for i in range(num_ep):
                ranks = range(i, world_size, num_ep)
                group = torch.distributed.new_group(ranks)
                if i == (rank % num_ep):
                    _EXPERT_DATA_PARALLEL_GROUP[f"ep_size_{num_ep}"] = group

    # Build the expert parallel groups.
    global _EXPERT_PARALLEL_GROUP
    assert _EXPERT_PARALLEL_GROUP is None, \
        'expert parallel group is already initialized'

    _EXPERT_PARALLEL_GROUP = {}

    for num_ep in num_ep_list_:
        # Similar as above we will need to think about two cases
        if num_ep >= expert_parallel_size_:
            if f"ep_size_{expert_parallel_size_}" not in _EXPERT_PARALLEL_GROUP:
                for i in range(world_size // expert_parallel_size_):
                    ranks = range(i * expert_parallel_size_,
                                  (i + 1) * expert_parallel_size_)
                    group = torch.distributed.new_group(ranks)
                    if i == (rank // expert_parallel_size_):
                        _EXPERT_PARALLEL_GROUP[
                            f"ep_size_{expert_parallel_size_}"] = group
        else:
            for i in range(world_size // num_ep):
                ranks = range(i * num_ep, (i + 1) * num_ep)
                group = torch.distributed.new_group(ranks)
                if i == (rank // num_ep):
                    _EXPERT_PARALLEL_GROUP[f"ep_size_{num_ep}"] = group


def initialize_model_and_expert_parallel(expert_parallel_size_, mpu, num_ep_list_=None):
    """
        Initialize Expert groups based on MPU groups.

        Example - E + M + D parallel
        world_size = 16
        model_degree = 2
        expert_degree = 4 # number of experts in same group
        mp_group = [0, 1], [2,3], [4,5] ...
        data_parallel_group =[0,2,4,6,8,10, 12,14],                 [1,3,5,7,9,11,13,15]
        expert_parallel_group = [0,2,4,6], [8,10,12,14]             [1,3,5,7], [9,11,13,15]
        expert_data_parallel_group = [0,8],[2,10],[4,12],[6,14],    [1,9],[3,11],[5,13],[]
    """
    assert torch.distributed.is_initialized(), "torch distributed is not initialized"
    assert mpu.model_parallel_is_initialized(), "model parallel group is not initialized"
    model_parallel_size_ = mpu.get_model_parallel_world_size()

    global _MAX_EP_SIZE
    global _MAX_EP_SIZE_NAME
    _MAX_EP_SIZE = expert_parallel_size_
    _MAX_EP_SIZE_NAME = f"ep_size_{expert_parallel_size_}"

    if num_ep_list_ is None:
        num_ep_list = [expert_parallel_size_]

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    dp_world_size = mpu.get_data_parallel_world_size()
    dp_rank = mpu.get_data_parallel_rank()

    log_dist(
        f"Initializing deepspeed groups with model parallel size {model_parallel_size_}, expert parallel size {expert_parallel_size_}, world size {world_size}, dp world size {dp_world_size}",
        [0])

    global _DATA_PARALLEL_GROUP, _MODEL_PARALLEL_GROUP
    global _EXPERT_PARALLEL_GROUP, _EXPERT_DATA_PARALLEL_GROUP

    # Get world size and rank. Ensure some consistencies.
    _DATA_PARALLEL_GROUP = mpu.get_data_parallel_group()
    _MODEL_PARALLEL_GROUP = mpu.get_model_parallel_group()

    expert_parallel_size_ = min(expert_parallel_size_, dp_world_size)
    ensure_divisibility(world_size, expert_parallel_size_)

    # Build the expert data parallel groups.
    assert _EXPERT_DATA_PARALLEL_GROUP is None, \
        'expert data parallel group is already initialized'
    # Build the expert parallel groups.
    assert _EXPERT_PARALLEL_GROUP is None, \
        'expert parallel group is already initialized'

    _EXPERT_DATA_PARALLEL_GROUP = {}
    _EXPERT_PARALLEL_GROUP = {}

    for num_ep in num_ep_list_:
        for j in range(model_parallel_size_):
            # For data parallel
            # Similar as initialize_expert_parallel we will need to think about two cases
            if num_ep >= expert_parallel_size_:
                #TODO: refactor this part of code to check condition in outer for-loop
                if True:  #f"ep_size_{expert_parallel_size_}" not in _EXPERT_DATA_PARALLEL_GROUP:
                    for i in range(expert_parallel_size_):
                        ranks = range(i * model_parallel_size_ + j,
                                      world_size,
                                      expert_parallel_size_ * model_parallel_size_)
                        group = torch.distributed.new_group(ranks)
                        if rank in list(ranks):
                            _EXPERT_DATA_PARALLEL_GROUP[
                                f"ep_size_{expert_parallel_size_}"] = group
            else:
                for i in range(num_ep):
                    ranks = range(i * model_parallel_size_ + j,
                                  world_size,
                                  num_ep * model_parallel_size_)
                    group = torch.distributed.new_group(ranks)
                    if rank in list(ranks):
                        _EXPERT_DATA_PARALLEL_GROUP[f"ep_size_{num_ep}"] = group

            # For expert parallel
            if num_ep >= expert_parallel_size_:
                #TODO: refactor this part of code to check condition in outer for-loop
                if True:  #f"ep_size_{expert_parallel_size_}" not in _EXPERT_PARALLEL_GROUP:
                    for i in range(dp_world_size // expert_parallel_size_):
                        ranks = range(
                            i * expert_parallel_size_ * model_parallel_size_ + j,
                            (i + 1) * expert_parallel_size_ * model_parallel_size_,
                            model_parallel_size_)
                        group = torch.distributed.new_group(ranks)
                        if rank in list(ranks):
                            _EXPERT_PARALLEL_GROUP[
                                f"ep_size_{expert_parallel_size_}"] = group
            else:
                for i in range(dp_world_size // num_ep):
                    ranks = range(i * num_ep * model_parallel_size_ + j,
                                  (i + 1) * num_ep * model_parallel_size_,
                                  model_parallel_size_)
                    group = torch.distributed.new_group(ranks)
                    if rank in list(ranks):
                        _EXPERT_PARALLEL_GROUP[f"ep_size_{num_ep}"] = group


def is_initialized():
    """Check if deepspeed groups have been initialized."""
    if _MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None or _EXPERT_PARALLEL_GROUP is None or _EXPERT_DATA_PARALLEL_GROUP is None:
        return False
    return True


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def expert_parallel_is_initialized():
    """Check if expert and expert data parallel groups are initialized."""
    if _EXPERT_PARALLEL_GROUP is None or _EXPERT_DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_max_expert_parallel_group():
    """Get the max expert parallel size."""
    return get_expert_parallel_group(get_max_expert_size_name())


def get_max_expert_size_name():
    """Get the maximum experts group size name in all group."""
    assert _MAX_EP_SIZE_NAME is not None, \
        'max expert parallel size is not initialized'
    return _MAX_EP_SIZE_NAME


def get_max_expert_size():
    """Get the maximum experts group size in all group."""
    assert _MAX_EP_SIZE is not None, \
        'max expert parallel size is not initialized'
    return _MAX_EP_SIZE


def get_expert_parallel_group(group_name):
    """Get the expert parallel group the caller rank belongs to."""
    assert _EXPERT_PARALLEL_GROUP is not None, \
        'expert parallel group is not initialized'
    return _EXPERT_PARALLEL_GROUP[group_name]


def get_expert_parallel_group_dict():
    """Get the expert parallel group dict."""
    assert _EXPERT_PARALLEL_GROUP is not None, \
        'expert parallel group is not initialized'
    return _EXPERT_PARALLEL_GROUP


def get_expert_data_parallel_group(group_name):
    """Get the expert data parallel group the caller rank belongs to."""
    assert _EXPERT_DATA_PARALLEL_GROUP is not None, \
        'expert data parallel group is not initialized'
    return _EXPERT_DATA_PARALLEL_GROUP[group_name]


def get_expert_data_parallel_group_dict():
    """Get the expert data parallel group dict."""
    assert _EXPERT_DATA_PARALLEL_GROUP is not None, \
        'expert data parallel group is not initialized'
    return _EXPERT_DATA_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def get_expert_parallel_world_size(group_name):
    """Return world size for the expert parallel group."""
    return torch.distributed.get_world_size(group=get_expert_parallel_group(group_name))


def get_expert_data_parallel_world_size(group_name):
    """Return world size for the expert data parallel group."""
    return torch.distributed.get_world_size(
        group=get_expert_data_parallel_group(group_name))


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    return torch.distributed.get_rank(group=get_model_parallel_group())


def get_expert_parallel_rank(group_name):
    """Return my rank for the expert parallel group."""
    return torch.distributed.get_rank(group=get_expert_parallel_group(group_name))


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_expert_parallel_src_rank(group_name):
    """Calculate the global rank corresponding to a local rank zero
    in the expert parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_expert_parallel_world_size(group_name)
    return (global_rank // local_world_size) * local_world_size


def get_expert_data_parallel_rank(group_name):
    """Return my rank for the expert data parallel group."""
    return torch.distributed.get_rank(group=get_expert_data_parallel_group(group_name))


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
