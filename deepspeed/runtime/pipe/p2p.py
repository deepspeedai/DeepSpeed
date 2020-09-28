'''
Copyright 2019 The Microsoft DeepSpeed Team
'''

import torch.distributed as dist

_groups = None
_grid = None


#initializes adjacent process groups
#run this only after torch.distributed.init_process_group() has been called
def init_process_groups(grid):
    global _groups, _grid
    _grid = grid

    assert _grid.pipe_parallel_size > 1, "There is no pipeline parallelism"

    _groups = [dist.new_group(ranks=group) for group in _grid.p2p_groups]


def _is_valid_send_recv(src_stage, dest_stage):
    first_stage = 0
    last_stage = _grid.pipe_parallel_size - 1
    assert abs(src_stage-dest_stage) == 1 or \
        (src_stage == first_stage and dest_stage == last_stage) or \
        (src_stage == last_stage and dest_stage == first_stage), \
    "Functionality currently limited to send and receive between adjacent ranks only"


def send(tensor, dest_stage, async_op=False):
    global _groups

    async_op = False
    src_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    group = _get_send_recv_group(src_stage, dest_stage)
    src_rank = _grid.stage_to_global(stage_id=src_stage)

    return dist.broadcast(tensor, src_rank, group=group, async_op=async_op)


def recv(tensor, src_stage, async_op=False):

    global _groups

    async_op = False
    dest_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    group = _get_send_recv_group(src_stage, dest_stage)
    src_rank = _grid.stage_to_global(stage_id=src_stage)

    return dist.broadcast(tensor, src_rank, group=group, async_op=async_op)


def barrier(stage_id):
    global _groups, _grid
    group_id = _grid.stage_to_global(stage_id=stage_id)
    if (dist.get_rank() >= 0):
        print("Barrier Group ID", group_id)
        print("Barrier Group", _grid.p2p_groups[group_id])
    dist.barrier(group=_groups[group_id])
    if (dist.get_rank() >= 0):
        print("Exiting Barrier ", group_id)


def _get_send_recv_group(src_stage, dest_stage):
    '''the group id is always the smaller rank unless its a wrap around'''

    stage_id = None

    first_stage = 0
    last_stage = _grid.pipe_parallel_size - 1

    if (src_stage == first_stage and dest_stage == last_stage
            or dest_stage == first_stage and src_stage == last_stage):
        stage_id = last_stage
    elif src_stage > dest_stage:
        stage_id = dest_stage
    else:
        stage_id = src_stage
    '''group_id corresponds to group of [group_id, group_id+1]
     unless group_id is the rank of the last stage
     in which case group_id correspods to group[group_id-num_stages+1, group_id]
     '''
    group_id = _grid.stage_to_global(stage_id=stage_id)

    return _groups[group_id]
