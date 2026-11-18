# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys
import weakref

import deepspeed.comm as dist
import pytest
import torch

from deepspeed.comm import comm
from deepspeed.runtime.sequence_parallel import parallel_state_sp
from deepspeed.utils import groups
from deepspeed.utils.groups import _get_expert_parallel_ranks
from unit.common import DistributedTest


def test_get_expert_parallel_ranks():
    """
    Example - E + M + D parallel
    world_size = 16
    model_degree = 2
    expert_degree = 4 # number of experts in same group
    mp_group = [0, 1], [2,3], [4,5] ...
    data_parallel_group =[0,2,4,6,8,10, 12,14],                 [1,3,5,7,9,11,13,15]
    expert_parallel_group = [0,2,4,6], [8,10,12,14]             [1,3,5,7], [9,11,13,15]
    expert_data_parallel_group = [0,8],[2,10],[4,12],[6,14],    [1,9],[3,11],[5,13],[7,15]
    """
    expert_parallel_groups, expert_data_parallel_groups = _get_expert_parallel_ranks(world_size=16,
                                                                                     tensor_parallel_size_=2,
                                                                                     expert_parallel_size_=4)
    assert expert_parallel_groups == [
        [0, 2, 4, 6],
        [8, 10, 12, 14],
        [1, 3, 5, 7],
        [9, 11, 13, 15],
    ]
    assert expert_data_parallel_groups == [
        [0, 8],
        [2, 10],
        [4, 12],
        [6, 14],
        [1, 9],
        [3, 11],
        [5, 13],
        [7, 15],
    ]


def test_destroy_process_group_failure_preserves_references(monkeypatch):
    world_group = object()
    sequence_parallel_group = object()
    sequence_data_parallel_group = object()

    class Backend:

        def destroy_process_group(self, group=None):
            raise RuntimeError("destroy failed")

    monkeypatch.setattr(comm, "cdb", Backend())
    monkeypatch.setattr(groups, "_WORLD_GROUP", world_group)
    monkeypatch.setattr(parallel_state_sp, "_SEQUENCE_PARALLEL_GROUP", sequence_parallel_group)
    monkeypatch.setattr(parallel_state_sp, "_SEQUENCE_DATA_PARALLEL_GROUP", sequence_data_parallel_group)

    with pytest.raises(RuntimeError, match="destroy failed"):
        comm.destroy_process_group()
    assert groups._WORLD_GROUP is world_group
    assert parallel_state_sp._SEQUENCE_PARALLEL_GROUP is sequence_parallel_group
    assert parallel_state_sp._SEQUENCE_DATA_PARALLEL_GROUP is sequence_data_parallel_group


def test_destroy_process_group_does_not_import_sequence_parallel(monkeypatch):
    module_name = "deepspeed.runtime.sequence_parallel.parallel_state_sp"

    class Backend:

        def destroy_process_group(self, group=None):
            pass

    monkeypatch.setattr(comm, "cdb", Backend())
    monkeypatch.delitem(sys.modules, module_name)

    comm.destroy_process_group()

    assert module_name not in sys.modules


class TestGroupCacheTeardown(DistributedTest):
    world_size = 2
    backend = "gloo"
    requires_cuda_env = False

    def test_world_destroy_releases_sequence_parallel_groups(self, tmpdir):
        rank = dist.get_rank()
        parallel_state_sp.initialize_sequence_parallel(sequence_parallel_size=self.world_size)

        dist.destroy_process_group()

        assert parallel_state_sp._SEQUENCE_PARALLEL_GROUP is None
        assert parallel_state_sp._SEQUENCE_DATA_PARALLEL_GROUP is None

        second_store = tmpdir.join("sequence-parallel-reinit")
        dist.init_distributed(dist_backend="gloo",
                              auto_mpi_discovery=False,
                              init_method=f"file://{second_store}",
                              rank=rank,
                              world_size=self.world_size)
        parallel_state_sp.initialize_sequence_parallel(sequence_parallel_size=self.world_size)
        value = torch.tensor(float(rank + 1))
        dist.all_reduce(value, group=parallel_state_sp.get_sequence_parallel_group())
        assert value.item() == 3

    @pytest.mark.world_size(1)
    @pytest.mark.parametrize("explicit_world", [False, True])
    def test_default_destroy_releases_cached_groups(self, explicit_world):

        class MPU:
            pass

        groups._clone_world_group()
        groups._create_expert_and_data_parallel(expert_parallel_size_=1)
        groups._init_tp_groups_with_new_group(tensor_model_parallel_size=1)
        ranks = list(range(dist.get_world_size()))
        groups._ZERO_PARAM_INTRA_PARALLEL_GROUP = dist.new_group(ranks=ranks)
        groups._ALL_TO_ALL_GROUP["test"] = dist.new_group(ranks=ranks)
        groups._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = dist.get_world_size()
        groups._MPU_TENSOR_MODEL_PARALLEL_RANK = dist.get_rank()
        groups.expert_tensor_parallel_world_size = dist.get_world_size()
        groups.mesh_device = object()
        ds_only_mpu = MPU()
        ds_only_mpu.process_group = groups._TENSOR_MODEL_PARALLEL_GROUP
        group_ref = weakref.ref(ds_only_mpu.process_group)
        mpu_ref = weakref.ref(ds_only_mpu)
        groups.mpu = ds_only_mpu
        del ds_only_mpu

        world_group = dist.get_world_group() if explicit_world else None
        dist.destroy_process_group(group=world_group)

        assert group_ref() is None
        assert mpu_ref() is None
        assert all(not cache for cache in (groups._EXPERT_PARALLEL_GROUP, groups._EXPERT_PARALLEL_GROUP_RANKS,
                                           groups._EXPERT_DATA_PARALLEL_GROUP,
                                           groups._EXPERT_DATA_PARALLEL_GROUP_RANKS, groups._ALL_TO_ALL_GROUP))
        assert all(group is None for group in (groups._WORLD_GROUP, groups._ZERO_PARAM_INTRA_PARALLEL_GROUP,
                                               groups._TENSOR_MODEL_PARALLEL_GROUP, groups._MODEL_PARALLEL_GROUP,
                                               groups._DATA_PARALLEL_GROUP, groups.mesh_device))
        assert (groups._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE, groups._MPU_TENSOR_MODEL_PARALLEL_RANK,
                groups.expert_tensor_parallel_world_size) == (None, None, 1)
        assert groups.mpu is None

        dist.init_distributed(dist_backend="gloo")
        groups._init_tp_groups_with_new_group(tensor_model_parallel_size=1, data_parallel_size=1)
        fresh_group = groups._TENSOR_MODEL_PARALLEL_GROUP
        assert fresh_group is not None
        value = torch.ones(1)
        dist.all_reduce(value, group=fresh_group)
        assert value.item() == 1

        caller_owned_mpu = MPU()
        caller_owned_mpu.process_group = fresh_group
        groups.mpu = caller_owned_mpu
        dist.destroy_process_group(group=dist.get_world_group())
        assert groups.mpu is None
        assert caller_owned_mpu.process_group is fresh_group

    def test_subgroup_destroy_preserves_unrelated_groups(self):
        ranks = list(range(self.world_size))
        target_group = dist.new_group(ranks=ranks)
        unrelated_group = dist.new_group(ranks=ranks)
        groups._TENSOR_MODEL_PARALLEL_GROUP = target_group
        groups._MODEL_PARALLEL_GROUP = target_group
        groups._DATA_PARALLEL_GROUP = unrelated_group
        groups._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = self.world_size
        groups._MPU_TENSOR_MODEL_PARALLEL_RANK = dist.get_rank()
        groups._EXPERT_PARALLEL_GROUP["target"] = target_group
        groups._EXPERT_PARALLEL_GROUP_RANKS["target"] = ranks
        groups._EXPERT_PARALLEL_GROUP["unrelated"] = unrelated_group
        groups._EXPERT_PARALLEL_GROUP_RANKS["unrelated"] = ranks
        caller_mpu = object()
        groups.mpu = caller_mpu

        dist.destroy_process_group(group=target_group)

        assert dist.is_initialized()
        assert groups._TENSOR_MODEL_PARALLEL_GROUP is None
        assert groups._MODEL_PARALLEL_GROUP is None
        assert groups._DATA_PARALLEL_GROUP is unrelated_group
        assert groups._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is None
        assert groups._MPU_TENSOR_MODEL_PARALLEL_RANK is None
        assert groups._EXPERT_PARALLEL_GROUP == {"unrelated": unrelated_group}
        assert groups._EXPERT_PARALLEL_GROUP_RANKS == {"unrelated": ranks}
        assert groups.mpu is caller_mpu
        dist.barrier(group=unrelated_group)
        dist.barrier()

    def test_nonmember_destroy_preserves_cached_groups(self):
        for rank in range(self.world_size):
            groups._ALL_TO_ALL_GROUP[f"rank_{rank}"] = dist.new_group(ranks=[rank])
        cached_groups = dict(groups._ALL_TO_ALL_GROUP)

        # A nonmember sentinel represents no destroyed group; unrelated cache
        # entries must survive just as they do in the backend's no-op contract.
        dist.destroy_process_group(group=cached_groups[f"rank_{1 - dist.get_rank()}"])

        assert groups._ALL_TO_ALL_GROUP == cached_groups
        value = torch.ones(1)
        dist.all_reduce(value, group=groups._ALL_TO_ALL_GROUP[f"rank_{dist.get_rank()}"])
        assert value.item() == 1
        dist.all_reduce(value)
        assert value.item() == self.world_size

    def test_subgroup_destroy_keeps_expert_data_parallel_table(self):
        groups._create_expert_and_data_parallel(expert_parallel_size_=self.world_size)
        (group_name, own_group), = groups._EXPERT_DATA_PARALLEL_GROUP.items()
        expected_table = tuple((rank, ) for rank in range(self.world_size))
        assert groups._get_expert_data_parallel_group_table(group_name) == expected_table

        # Only the members of a subgroup release it, so a table describing every subgroup must
        # outlive it on those ranks too, or ranks would disagree about the topology.
        dist.destroy_process_group(group=own_group)

        assert group_name not in groups._EXPERT_DATA_PARALLEL_GROUP
        assert groups._get_expert_data_parallel_group_table(group_name) == expected_table
        dist.barrier()
