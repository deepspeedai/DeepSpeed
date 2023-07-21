# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
import pytest
import gc
from unit.common import DistributedTest
from unit.simple_model import SimplePRMoEModel, SimpleMoEModel, sequence_dataloader
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer, is_moe_param
from deepspeed.runtime.utils import required_torch_version


@pytest.mark.parametrize("ep_size", [2, 4])
@pytest.mark.parametrize("zero_stage", [0, 1, 2])
@pytest.mark.parametrize("use_residual", [True, False])
class TestMoE(DistributedTest):
    world_size = 4

    def test(self, ep_size, zero_stage, use_residual):
        if not required_torch_version(min_version=1.8):
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage
            }
        }
        hidden_dim = 16

        # E+D -- ep_size = 2
        # E only -- ep_size = 4
        model = SimpleMoEModel(hidden_dim, ep_size=ep_size, use_residual=use_residual)
        param_group = {'params': [p for p in model.parameters()], 'name': 'random-unique-name'}
        params = split_params_into_different_moe_groups_for_optimizer(param_group)
        optimizer = torch.optim.AdamW(params=params)
        model, optimizer, _, _ = deepspeed.initialize(config=config_dict,
                                                      model=model,
                                                      optimizer=optimizer,
                                                      dist_init_required=False)
        #dist_init_required=False -- parameterize to True/False?

        data_loader = sequence_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)

        def strict_average_tensor(tensor):
            process_group = optimizer.dp_process_group
            curr_size = 0
            pg_offsets = []
            for i, param, param_id in optimizer.params_in_ipg_bucket:
                process_group = optimizer.dp_process_group
                if optimizer.ipg_bucket_has_moe_params:
                    process_group = optimizer.expert_dp_process_group[param.group_name] if is_moe_param(
                        param) else optimizer.dp_process_group
                partition_ids = optimizer.param_to_partition_ids[i][param_id]
                # Get all partition ids + their offsets
                partition_offsets = []
                for partition_id in partition_ids:
                    offset = optimizer.grad_start_offset[i][partition_id][param_id]
                    partition_offsets.append(offset)
                partition_offsets.sort()
                # Calculate rank and offsets for grad slices
                for idx, offset in enumerate(partition_offsets):
                    # Calculate numel for grad slice depending on partition location
                    if idx == len(partition_offsets) - 1:
                        # Last partition_id uses its own offset
                        numel = param.numel() - offset
                    else:
                        # Set numel to next partition's offset
                        numel = partition_offsets[idx + 1] - offset
                    pg_offsets.append((curr_size, process_group))
                    curr_size += numel

            def strict_narrow(dim, start, length):
                lo, hi = 0, len(pg_offsets) - 1
                while lo < hi:
                    mi = lo + (hi - lo) // 2
                    if pg_offsets[mi][0] >= start:
                        hi = mi
                    else:
                        lo = mi + 1
                curr_slice, reduce_process_group = lo, pg_offsets[lo][1]
                while curr_slice < len(pg_offsets) and start + length > pg_offsets[curr_slice][0]:
                    assert reduce_process_group == pg_offsets[curr_slice][
                        1], "reduce process_group does not match the parameter's process_group"
                    curr_slice += 1
                return orig_narrow(dim, start, length)  # real call

            orig_narrow, tensor.narrow = tensor.narrow, strict_narrow
            type(optimizer).average_tensor(optimizer, tensor)  # real call
            tensor.narrow = orig_narrow

        if "average_tensor" in dir(optimizer):
            optimizer.average_tensor = strict_average_tensor

        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            gc.collect()  # Must do this or we get a memory leak in this test


@pytest.mark.parametrize("ep_size, use_residual", [(2, True), (2, False)])
class TestPRMoE(DistributedTest):
    world_size = 4

    def test(self, ep_size, use_residual):
        if not required_torch_version(min_version=1.8):
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {"train_batch_size": 8, "steps_per_print": 1, "fp16": {"enabled": True}}
        hidden_dim = 16

        # E+D -- ep_size = 2
        # E only -- ep_size = 4
        model = SimplePRMoEModel(hidden_dim, ep_size=ep_size, use_residual=use_residual)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              optimizer=optimizer,
                                              dist_init_required=False)

        data_loader = sequence_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)

        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
