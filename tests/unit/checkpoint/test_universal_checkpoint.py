# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import math

import deepspeed
from types import SimpleNamespace
from torch.utils._pytree import tree_map

from deepspeed.utils.torch import required_torch_version
from deepspeed.checkpoint import UNIVERSAL_CHECKPOINT_INFO
from deepspeed.checkpoint.ds_to_universal import main as convert_to_universal

from unit.common import DistributedTest, DistributedFixture
from unit.simple_model import *
from unit.util import bf16_required_version_check

from unit.checkpoint.common import compare_opt_state_dicts, compare_state_dicts

import pytest
import deepspeed.comm as dist


def get_expected_mismatch_keys():
    # torch 1.2.* stores raw tensor id numbers in checkpoint state which leads to
    # false positive mismatches in checkpoint state comparisons.
    # Newer torch versions store tensor ids as 0, 1, 2, ...
    return [] if required_torch_version(min_version=1.4) else ['params']


def maybe_step(t):
    return not torch.is_tensor(t) or (t.device.type == 'cpu' and t.numel() == 1)


def gather_opt_state(optimizer_state):

    def gather_tensor(t):

        if maybe_step(t):
            return t
        else:
            buffer = [torch.zeros_like(t.flatten()) for _ in range(dist.get_world_size())]
            dist.all_gather(buffer, t.flatten())
            return torch.cat(buffer)

    return tree_map(gather_tensor, optimizer_state)


def remove_pad_in_opt_state(optimizer_state, num_params):

    def remove_pad(t):
        if maybe_step(t):
            return t
        else:
            return t[:num_params]

    return tree_map(remove_pad, optimizer_state)


CP_TAG = "test_tag"


def init_ds_engine(model, ds_config, use_torch_adam):

    if use_torch_adam:
        ds_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        del ds_config["optimizer"]
        model, _, _, _ = deepspeed.initialize(config=ds_config, model=model, optimizer=ds_optimizer)
    else:
        model, _, _, _ = deepspeed.initialize(config=ds_config, model=model, model_parameters=model.parameters())

    return model


def train_save_convert(ds_config, hidden_dim, load_optim, use_torch_adam, dtype, tmpdir, world_size):
    if dtype == torch.bfloat16 and not bf16_required_version_check():
        return

    test_step = 8

    model = SimpleModel(hidden_dim, nlayers=2)
    model = init_ds_engine(model, ds_config, use_torch_adam)
    data_loader = random_dataloader(model=model,
                                    total_samples=test_step,
                                    hidden_dim=hidden_dim,
                                    device=model.device,
                                    dtype=dtype)
    for batch in data_loader:
        loss = model(batch[0], batch[1])
        model.backward(loss)
        model.step()

    if ds_config["zero_optimization"]["stage"] == 3:
        model.optimizer._set_fp32_optimizer_param_groups()
        sd = model.optimizer.optimizer.state_dict() if load_optim else None
        model.optimizer._clear_fp32_optimizer_param_groups()
    else:
        sd = model.optimizer.optimizer.state_dict() if load_optim else None

    client_state = {}
    client_state[UNIVERSAL_CHECKPOINT_INFO] = {}
    client_state['iteration'] = test_step
    model.save_checkpoint(tmpdir, tag=CP_TAG, client_state=client_state)

    cp_dir = os.path.join(tmpdir, CP_TAG)
    univ_cp_dir = f"{cp_dir}_universal"

    args = SimpleNamespace(input_folder=cp_dir,
                           output_folder=univ_cp_dir,
                           num_extract_workers=1,
                           num_merge_workers=1,
                           keep_temp_folder=False,
                           strict=True,
                           inject_missing_state=False)

    dist.barrier()
    if dist.get_rank() == 0:
        convert_to_universal(args)

    model_state = model.state_dict()
    optimizer_state = None
    if load_optim:
        if ds_config["zero_optimization"]["stage"] == 3:
            model.optimizer._set_fp32_optimizer_param_groups()
            optimizer_state = gather_opt_state(model.optimizer.optimizer.state_dict())
            model.optimizer._clear_fp32_optimizer_param_groups()
            update_gathered_stage3_optimizer(optimizer_state, model._get_zero_param_shapes(), world_size)
        else:
            optimizer_state = gather_opt_state(model.optimizer.optimizer.state_dict())

    if dist.get_rank() == 0:
        torch.save((model_state, optimizer_state), os.path.join(tmpdir, "baseline_state.pt"))

    dist.barrier()
    model.destroy()


@pytest.fixture
def ds_config(zero_stage, dtype, sub_group_size):
    ds_config = {
        "train_batch_size": 8,
        "optimizer": {
            "type": 'Adam'
        },
        "zero_optimization": {
            "stage": zero_stage,
        }
    }
    if dtype == torch.float16:
        ds_config["fp16"] = {"enabled": True, "initial_scale_power": 8}
    elif dtype == torch.bfloat16:
        ds_config["bf16"] = {"enabled": True}
    if sub_group_size > 0:
        ds_config["zero_optimization"]["sub_group_size"] = sub_group_size
    return ds_config


class _baseline(DistributedFixture):
    world_size = None

    def run(self, tmpdir, ds_config, zero_stage, dtype, load_optim, use_torch_adam):
        hidden_dim = 10
        train_save_convert(ds_config, hidden_dim, load_optim, use_torch_adam, dtype, tmpdir, self.world_size)


class baseline_ws2(_baseline):
    world_size = 2


class baseline_ws4(_baseline):
    world_size = 4


# Stage3 use shard parameter, need to reorganize the optimizer parameters.
def update_gathered_stage3_optimizer(optimizer_state, param_shapes, world_size):
    for sub_group_id, group in enumerate(optimizer_state["param_groups"]):
        group["params"] = None

    new_state = {}
    for sub_group_id, sub_group_param_shape in enumerate(param_shapes):
        total_numel = optimizer_state['state'][sub_group_id]['exp_avg'].numel()
        assert total_numel % world_size == 0
        numel_per_rank = total_numel // world_size
        param_offset_in_current_rank = 0
        for param_name, param_shape in sub_group_param_shape.items():
            param_numel = param_shape.numel()
            param_partition_numel = math.ceil(param_numel / world_size)
            param_optimizer_tensor = {
                "exp_avg": torch.zeros(param_numel),
                "exp_avg_sq": torch.zeros(param_numel),
                "step": optimizer_state['state'][sub_group_id]['step'],
            }
            for key in ["exp_avg", "exp_avg_sq"]:
                write_offset = 0
                for rank in range(world_size):
                    offset = param_offset_in_current_rank + rank * numel_per_rank
                    length = min(param_partition_numel, param_numel - rank * param_partition_numel)
                    tmp = optimizer_state['state'][sub_group_id][key].narrow(0, offset, length)
                    param_optimizer_tensor[key].narrow(0, write_offset, length).copy_(tmp)
                    write_offset += length
            param_offset_in_current_rank += param_partition_numel
            new_state[param_name] = param_optimizer_tensor
    optimizer_state["state"] = new_state


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("zero_stage", [1, 3])
@pytest.mark.parametrize("use_torch_adam", [False, True])
@pytest.mark.parametrize("load_optim", [False, True])
@pytest.mark.parametrize("sub_group_size", [-1, 100])
@pytest.mark.parametrize('compile_mode', [True, False])
class TestZeROUniversalCheckpointDP(DistributedTest):

    def _run_test(self, tmpdir, dtype, ds_config, load_optim, use_torch_adam, world_size, compile_mode):
        if dtype == torch.bfloat16 and not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        hidden_dim = 10
        loaded_model_state, loaded_optimizer_state = torch.load(f"{tmpdir}/baseline_state.pt", weights_only=False)

        ds_config["checkpoint"] = {"load_universal": True}
        univ_model = SimpleModel(hidden_dim, nlayers=2)
        univ_model = init_ds_engine(univ_model, ds_config, use_torch_adam)
        if compile_mode:
            univ_model.compile()

        univ_model.load_checkpoint(tmpdir, tag=f"{CP_TAG}_universal", load_optimizer_states=load_optim)

        model_state = univ_model.state_dict()
        compare_state_dicts(model_state, loaded_model_state)

        if load_optim:
            if ds_config["zero_optimization"]["stage"] == 3:
                univ_model.optimizer._set_fp32_optimizer_param_groups()
                optimizer_state = gather_opt_state(univ_model.optimizer.optimizer.state_dict())
                univ_model.optimizer._clear_fp32_optimizer_param_groups()
                update_gathered_stage3_optimizer(optimizer_state, univ_model._get_zero_param_shapes(), world_size)
            else:
                optimizer_state = gather_opt_state(univ_model.optimizer.optimizer.state_dict())
            # padding sizes may differ when dp sizes are different
            param_count = sum(p.numel() for p in univ_model.parameters())
            optimizer_state = remove_pad_in_opt_state(optimizer_state, param_count)
            loaded_optimizer_state = remove_pad_in_opt_state(loaded_optimizer_state, param_count)

            compare_opt_state_dicts(optimizer_state, loaded_optimizer_state, get_expected_mismatch_keys())

        # Run training again to verify that the optimizer has necessary states
        test_step = 8
        data_loader = random_dataloader(model=univ_model,
                                        total_samples=test_step,
                                        hidden_dim=hidden_dim,
                                        device=univ_model.device,
                                        dtype=dtype)
        for batch in data_loader:
            loss = univ_model(batch[0], batch[1])
            univ_model.backward(loss)
            univ_model.step()

        univ_model.destroy()

    @pytest.mark.world_size(2)
    def test_dp_world_size_2to2(self, baseline_ws2, tmpdir, dtype, ds_config, load_optim, use_torch_adam,
                                compile_mode):
        self._run_test(tmpdir, dtype, ds_config, load_optim, use_torch_adam, compile_mode)

    @pytest.mark.world_size(2)
    def test_dp_world_size_4to2(self, baseline_ws4, tmpdir, dtype, ds_config, load_optim, use_torch_adam,
                                compile_mode):
        self._run_test(tmpdir, dtype, ds_config, load_optim, use_torch_adam, compile_mode)

    @pytest.mark.world_size(4)
    def test_dp_world_size_2to4(self, baseline_ws2, tmpdir, dtype, ds_config, load_optim, use_torch_adam,
                                compile_mode):
        self._run_test(tmpdir, dtype, ds_config, load_optim, use_torch_adam, 4, compile_mode)
