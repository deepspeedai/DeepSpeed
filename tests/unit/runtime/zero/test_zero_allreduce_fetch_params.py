# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
import deepspeed.runtime.zero.partition_parameters as partition_parameters
from unit.common import DistributedTest

# Odd numels need a slot of partition padding under world_size=2, so ds_numel_aligned
# differs from ds_numel. Even numels make the two equal, which is the case the flat-buffer
# stride already handled; both are checked so the padded fix does not regress it.
PADDED_NUMELS = [5, 7]
ALIGNED_NUMELS = [4, 6]
POISON = 7777.0


class ParamHolder(torch.nn.Module):

    def __init__(self, numels):
        super().__init__()
        for i, numel in enumerate(numels):
            self.register_parameter(f"p{i}", torch.nn.Parameter(torch.arange(1, numel + 1, dtype=torch.float32)))


class TestAllReduceFetchParamsPadded(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("numels", [PADDED_NUMELS, ALIGNED_NUMELS], ids=["padded", "aligned"])
    def test_params_reconstruct_exactly(self, numels):
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_use_all_reduce_for_fetch_params": True,
            },
        }

        # The tail of the last rank's partition is never written (torch.empty, and the
        # partial-copy branch only copies the elements that exist), so its contents are
        # whatever the allocator returns. Fill new allocations with a sentinel so the test
        # is deterministic instead of depending on whether the allocator hands back a
        # freshly zeroed page.
        real_empty = partition_parameters._orig_torch_empty

        def poisoned_empty(*args, **kwargs):
            tensor = real_empty(*args, **kwargs)
            if tensor.dtype.is_floating_point:
                tensor.fill_(POISON)
            return tensor

        expected = [torch.arange(1, numel + 1, dtype=torch.float32) for numel in numels]

        partition_parameters._orig_torch_empty = poisoned_empty
        try:
            with deepspeed.zero.Init(config_dict_or_path=config, mem_efficient_linear=False, enabled=True):
                module = ParamHolder(numels)
        finally:
            partition_parameters._orig_torch_empty = real_empty

        params = [getattr(module, f"p{i}") for i in range(len(numels))]
        params[0].all_gather_coalesced(params).wait()

        for i, param in enumerate(params):
            gathered = param.data.detach().reshape(-1).cpu()
            assert torch.equal(
                gathered, expected[i]), (f"param p{i} was not reconstructed exactly: expected {expected[i].tolist()}, "
                                         f"got {gathered.tolist()}")
