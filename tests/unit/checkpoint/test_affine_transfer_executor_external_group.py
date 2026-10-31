# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""execute_transfer over a weight-update group built outside the default world.

Trainer and rollout processes each own a default process group (here of world size 1, as a TP=1 engine or trainer
has) and join a separate two-process group created with ``_new_process_group_helper`` on its own store -- the way
RL frameworks build their trainer <-> engine weight-update group. In that group torch's rank bookkeeping maps every
member's default-world rank 0 to group rank 0, so the executor must take the rank from the group itself.
"""
import datetime
import os
import tempfile

import pytest
import torch
import torch.multiprocessing as mp

from deepspeed.accelerator import get_accelerator


def _worker(rank, port, tmp):
    # Using torch.distributed directly is intended: RL frameworks build this group outside deepspeed.comm.
    import torch.distributed as tdist
    from torch.distributed.distributed_c10d import PrefixStore, _new_process_group_helper, _world

    tdist.init_process_group('gloo', init_method=f'file://{tmp}/default-{rank}', rank=0, world_size=1)
    import deepspeed
    deepspeed.init_distributed(dist_backend='gloo')
    store = tdist.TCPStore('127.0.0.1', port, 2, rank == 0, timeout=datetime.timedelta(seconds=60))
    group, _ = _new_process_group_helper(2,
                                         rank, [],
                                         tdist.Backend('gloo'),
                                         PrefixStore('weight-update', store),
                                         group_name='weight-update',
                                         backend_options=None,
                                         timeout=datetime.timedelta(seconds=60))
    _world.pg_group_ranks[group] = {0: 0, 1: 1}

    from deepspeed.checkpoint.affine import contiguous_split_map
    from deepspeed.checkpoint.affine_transfer import plan_transfer
    from deepspeed.checkpoint.affine_transfer_executor import TransferEndpoints, execute_transfer

    full = torch.arange(96, dtype=torch.float32).reshape(8, 12)
    source = contiguous_split_map((8, 12), [4, 4], 0)
    target = contiguous_split_map((8, 12), [6, 6], 1)
    plan = plan_transfer(target, source)
    # Group rank 0 reads every source shard, group rank 1 writes every target shard.
    endpoints = TransferEndpoints(source={loc: 0
                                          for loc in source.shard_shapes},
                                  target={loc: 1
                                          for loc in target.shard_shapes})
    sources = {loc: source.extract(full, loc).contiguous() for loc in source.shard_shapes} if rank == 0 else {}
    targets = {loc: torch.full(shape, -1.0) for loc, shape in target.shard_shapes.items()} if rank == 1 else {}
    execute_transfer(plan, sources, targets, endpoints, torch.float32, group=group)
    for loc, shard in targets.items():
        assert torch.equal(shard, target.extract(full, loc)), loc


@pytest.mark.skipif(get_accelerator().device_name() != 'cpu',
                    reason='gloo groups over host buffers; CPU accelerator only')
def test_transfer_over_a_weight_update_group_outside_the_default_world():
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(29000 + os.getpid() % 1000, tmp), nprocs=2, join=True)
