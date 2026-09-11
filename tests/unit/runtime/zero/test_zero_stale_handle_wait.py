# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
import deepspeed.comm as dist

from unit.common import DistributedTest, preferred_dtype

from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

HIDDEN = 64


def _released(param):
    return param.ds_status == ZeroParamStatus.NOT_AVAILABLE and param.data.numel() == 0


def _stale_wait_cycle(victim, other):
    """Drive the coordinator's registration pattern by hand.

    __all_gather_params_ stores ONE handle object under EVERY parameter of a gather group, and
    _fetch_sub_module_impl pops and waits those keys independently. So a group's handle is waited
    once per parameter. If the victim is released between two of those waits, the second wait must
    not mark it available again.
    """
    assert _released(victim) and _released(other)
    handle = victim.all_gather_coalesced([victim, other])
    handle.wait()  # popped through the victim's own key
    assert victim.ds_status == ZeroParamStatus.AVAILABLE
    victim.partition()  # the victim's module finished; the coordinator released it
    assert _released(victim)
    handle.wait()  # popped LATER through the other parameter's key
    return victim


class TestStaleHandleWait(DistributedTest):
    world_size = 2

    def test_all_gather_handle_wait_is_idempotent(self):
        # stage3_allgather_sequential gives every parameter its own AllGatherHandle.
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_allgather_sequential": True
            },
        }
        with deepspeed.zero.Init(config_dict_or_path=config, dtype=preferred_dtype()):
            a = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
            b = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)

        victim = _stale_wait_cycle(a.weight, b.weight)
        assert _released(victim), (
            "a second wait() on a released parameter marked it "
            f"{victim.ds_status.name} over {victim.data.numel()} elements; the next fetch would "
            "skip the all-gather and compute on an empty weight")

    def test_no_gather_handle_wait_is_idempotent(self):
        # A parameter whose partition group holds one rank takes the no-gather path and gets a
        # NoGatherHandle. Expert-parallel placements produce exactly this alongside ordinary
        # parameters, which makes all_gather_coalesced split the group by process group.
        rank, world = dist.get_rank(), dist.get_world_size()
        solo_group = None
        for r in range(world):
            group = dist.new_group([r])
            if r == rank:
                solo_group = group

        class SoloGroupParam(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.empty(HIDDEN, HIDDEN))
                self.w.ds_zero_partition_process_group = solo_group

        with deepspeed.zero.Init(dtype=preferred_dtype()):
            dense = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
            solo = SoloGroupParam()

        assert solo.w.ds_zero_partition_world_size == 1
        victim = _stale_wait_cycle(solo.w, dense.weight)
        assert _released(victim), (
            "a second wait() on a released parameter marked it "
            f"{victim.ds_status.name} over {victim.data.numel()} elements")
