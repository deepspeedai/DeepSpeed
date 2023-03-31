# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

import torch
import deepspeed.comm as dist

from deepspeed.runtime.pipe.topology import PipelineParallelGrid as Grid
from deepspeed.runtime.pipe.topology import ProcessTopology as Topo
from deepspeed.runtime.pipe.topology import _prime_factors

from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest


def test_topology_2d():
    topo = Topo(axes=['row', 'col'], dims=[2, 2])

    assert topo.world_size() == 4

    assert topo.get_rank(row=0, col=0) == 0
    assert topo.get_rank(row=0, col=1) == 1
    assert topo.get_rank(row=1, col=0) == 2
    assert topo.get_rank(row=1, col=1) == 3

    assert topo.get_axis_list(axis='row', idx=0) == [0, 1]
    assert topo.get_axis_list(axis='row', idx=1) == [2, 3]
    assert topo.get_axis_list(axis='col', idx=0) == [0, 2]
    assert topo.get_axis_list(axis='col', idx=1) == [1, 3]


def test_topology_dims():
    topo = Topo(axes=['a', 'b', 'c'], dims=[2, 3, 4])
    assert topo.world_size() == 24
    assert topo.get_dim('a') == 2
    assert topo.get_dim('b') == 3
    assert topo.get_dim('c') == 4


def test_topology_match():
    topo = Topo(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
    print(topo.filter_match(pipe=0, data=1))
    assert topo.filter_match(pipe=0, data=1) == [2, 3]
    print([topo.get_coord(r) for r in topo.filter_match(pipe=0, data=1)])


def test_topology_rank_repr():
    topo = Topo(axes=['a', 'b'], dims=[2, 2])
    assert topo.get_rank_repr(rank=0) == 'a_00-b_00'
    assert topo.get_rank_repr(rank=1) == 'a_00-b_01'
    assert topo.get_rank_repr(rank=2) == 'a_01-b_00'
    assert topo.get_rank_repr(rank=3) == 'a_01-b_01'

    assert topo.get_rank_repr(rank=3, inner_sep='+') == 'a+01-b+01'
    assert topo.get_rank_repr(rank=3, inner_sep='🤗', outer_sep='_JEFF_') == 'a🤗01_JEFF_b🤗01'

    topo = Topo(axes=['pipe', 'data'], dims=[2, 2])
    assert topo.get_rank_repr(rank=0) == ''
    assert topo.get_rank_repr(rank=1) == ''
    assert topo.get_rank_repr(rank=2) == ''
    assert topo.get_rank_repr(rank=3) == ''

    assert topo.get_rank_repr(rank=0, omit_axes=['pipe']) == 'data_00'
    assert topo.get_rank_repr(rank=1, omit_axes=['pipe']) == 'data_01'
    assert topo.get_rank_repr(rank=2, omit_axes=['pipe']) == 'data_00'
    assert topo.get_rank_repr(rank=3, omit_axes=['pipe']) == 'data_01'

    assert topo.get_rank_repr(rank=0, omit_axes=[]) == 'pipe_00-data_00'
    assert topo.get_rank_repr(rank=1, omit_axes=[]) == 'pipe_00-data_01'
    assert topo.get_rank_repr(rank=2, omit_axes=[]) == 'pipe_01-data_00'
    assert topo.get_rank_repr(rank=3, omit_axes=[]) == 'pipe_01-data_01'

    topo = Topo(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
    assert topo.get_rank_repr(rank=0) == 'model_00'
    assert topo.get_rank_repr(rank=1) == 'model_01'
    assert topo.get_rank_repr(rank=2) == 'model_00'
    assert topo.get_rank_repr(rank=3) == 'model_01'
    assert topo.get_rank_repr(rank=4) == 'model_00'
    assert topo.get_rank_repr(rank=5) == 'model_01'
    assert topo.get_rank_repr(rank=6) == 'model_00'
    assert topo.get_rank_repr(rank=7) == 'model_01'


def test_topology_3d():
    topo = Topo(axes=['a', 'b', 'c'], dims=[2, 2, 2])

    assert topo.get_rank(a=0, b=0, c=0) == 0
    assert topo.get_rank(a=0, b=0, c=1) == 1
    assert topo.get_rank(a=0, b=1, c=0) == 2
    assert topo.get_rank(a=0, b=1, c=1) == 3
    assert topo.get_rank(a=1, b=0, c=0) == 4
    assert topo.get_rank(a=1, b=0, c=1) == 5
    assert topo.get_rank(a=1, b=1, c=0) == 6
    assert topo.get_rank(a=1, b=1, c=1) == 7

    assert topo.get_axis_list('a', 0) == [0, 1, 2, 3]
    assert topo.get_axis_list('a', 1) == [4, 5, 6, 7]
    assert topo.get_axis_list('b', 0) == [0, 1, 4, 5]
    assert topo.get_axis_list('b', 1) == [2, 3, 6, 7]
    assert topo.get_axis_list('c', 0) == [0, 2, 4, 6]
    assert topo.get_axis_list('c', 1) == [1, 3, 5, 7]

    assert topo.get_coord(0) == topo.ProcessCoord(0, 0, 0)
    assert topo.get_coord(1) == topo.ProcessCoord(0, 0, 1)
    assert topo.get_coord(2) == topo.ProcessCoord(0, 1, 0)
    assert topo.get_coord(3) == topo.ProcessCoord(0, 1, 1)
    assert topo.get_coord(4) == topo.ProcessCoord(1, 0, 0)
    assert topo.get_coord(5) == topo.ProcessCoord(1, 0, 1)
    assert topo.get_coord(6) == topo.ProcessCoord(1, 1, 0)
    assert topo.get_coord(7) == topo.ProcessCoord(1, 1, 1)

    assert topo.filter_match(a=0) == [0, 1, 2, 3]
    assert topo.filter_match(b=1, c=1) == [3, 7]
    assert topo.filter_match(a=1, b=1, c=1) == [7]

    # Easy access method
    assert topo.get_coord(0).a == 0


def test_topology_comm_list():
    topo = Topo(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])

    assert topo.get_rank(pipe=0, data=0, model=0) == 0
    assert topo.get_rank(pipe=0, data=0, model=1) == 1
    assert topo.get_rank(pipe=0, data=1, model=0) == 2
    assert topo.get_rank(pipe=0, data=1, model=1) == 3
    assert topo.get_rank(pipe=1, data=0, model=0) == 4
    assert topo.get_rank(pipe=1, data=0, model=1) == 5
    assert topo.get_rank(pipe=1, data=1, model=0) == 6
    assert topo.get_rank(pipe=1, data=1, model=1) == 7

    pipe_list = [
        [0, 4],  # data=0, model=0
        [1, 5],  # data=0, model=1
        [2, 6],  # data=1, model=0
        [3, 7],  # data=1, model=1
    ]
    assert topo.get_axis_comm_lists('pipe') == pipe_list

    data_list = [
        [0, 2],  # pipe=0, model=0
        [1, 3],  # pipe=0, model=1
        [4, 6],  # pipe=1, model=0
        [5, 7],  # pipe=1, model=1
    ]
    assert topo.get_axis_comm_lists('data') == data_list

    model_list = [
        [0, 1],  # pipe=0, data=0
        [2, 3],  # pipe=0, data=1
        [4, 5],  # pipe=1, data=0
        [6, 7],  # pipe=1, data=1
    ]
    assert topo.get_axis_comm_lists('model') == model_list

    # Handle nonsense. We don't want to RuntimeError because it allows us to write more
    # generalized code for data/model/pipe parallelism
    assert topo.get_axis_comm_lists('jeff') == []


class TestDistributedTopology(DistributedTest):
    world_size = 4

    def test_grid_pipe_data(self):
        topo = Topo(axes=['pipe', 'data'], dims=[2, 2])
        grid = Grid(topology=topo)

        assert grid._is_grid_valid()

        rank = dist.get_rank()

        assert grid.is_first_stage == (grid.get_stage_id() == 0)
        assert grid.is_last_stage == (grid.get_stage_id() == grid.get_pipe_parallel_world_size() - 1)

        # Test collectives along the pipeline parallel process groups
        rank_tensor = torch.LongTensor(data=[rank]).to(get_accelerator().device_name())
        dist.all_reduce(rank_tensor, group=grid.get_pipe_parallel_group())
        pipe_group = grid.pp_group
        assert torch.all(rank_tensor == sum(pipe_group))

        # Test collectives along the data parallel process groups
        rank_tensor = torch.LongTensor(data=[rank]).to(get_accelerator().device_name())
        dist.all_reduce(rank_tensor, group=grid.get_data_parallel_group())
        data_group = grid.dp_group
        assert torch.all(rank_tensor == sum(data_group))

    def test_stage_to_global(self):
        topo = Topo(axes=['pipe', 'data'], dims=[2, 2])
        grid = Grid(topology=topo)

        assert grid._is_grid_valid()

        assert grid.stage_to_global(stage_id=0, data=0) == 0
        assert grid.stage_to_global(stage_id=0, data=1) == 1
        assert grid.stage_to_global(stage_id=1, data=0) == 2
        assert grid.stage_to_global(stage_id=1, data=1) == 3

        me = topo.get_coord(rank=dist.get_rank())
        if me.data == 0:
            assert grid.stage_to_global(stage_id=0) == 0
            assert grid.stage_to_global(stage_id=1) == 2
        else:
            assert grid.stage_to_global(stage_id=0) == 1
            assert grid.stage_to_global(stage_id=1) == 3


def test_primes():
    """ Test prime factorizations. """

    def _product(ps):
        p = 1
        for num in ps:
            p *= num
        return p

    with pytest.raises(ValueError):
        _prime_factors(0)

    for x in range(1, 30):
        primes = _prime_factors(x)
        assert _product(primes) == x
        for p in primes:
            assert _prime_factors(p) == [p]
