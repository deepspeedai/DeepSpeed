# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
import numbers

import deepspeed
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from unit.common import preferred_dtype
from unit.simple_model import *
from unittest.mock import MagicMock, patch


def compare_deepspeed_states(saved_model, loaded_model):
    # These are compared in more depth in other places
    assert hasattr(loaded_model, 'module')

    assert saved_model.sparse_tensor_module_names == loaded_model.sparse_tensor_module_names
    assert saved_model.skipped_steps == loaded_model.skipped_steps
    assert saved_model.global_steps == loaded_model.global_steps


def zero3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]


def compare_model_states(saved_model, loaded_model, compare_optimizer=True, load_module_only=False):
    if not load_module_only:
        compare_deepspeed_states(saved_model, loaded_model)

    params_to_fetch = zero3_params_to_fetch(
        list(saved_model.module.named_parameters()) + list(loaded_model.module.named_parameters()))
    enable_gather = len(params_to_fetch) > 0
    with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=enable_gather):
        for p0, p1 in zip(saved_model.module.named_parameters(), loaded_model.module.named_parameters()):
            np0, p0 = p0
            np1, p1 = p1
            if 'deepspeed_moe.gate.wg' in np0:
                # these params are converted to float at runtime, cast to half for comparison
                p1 = p1.half()
                p0 = p0.half()
            assert id(p0) != id(p1), f'Comparing fp16 model state tensor against itself : {id(p0)} <====> {id(p1)}'
            try:
                assert torch.allclose(p0, p1,
                                      atol=1e-07), f"FP16 model state {p0} is not equal to {p1}, names:{np0}, {np1}"
            except RuntimeError as err:
                print(f"FP16 model state {p0} is not equal to {p1}, names:{np0}, {np1}")
                raise err

    if not compare_optimizer:
        return

    if DeepSpeedZeroOptimizer_Stage3 is not None and isinstance(saved_model.optimizer, DeepSpeedZeroOptimizer_Stage3):
        for p0, p1 in zip(saved_model.optimizer.fp32_partitioned_groups_flat,
                          loaded_model.optimizer.fp32_partitioned_groups_flat):
            assert torch.allclose(p0, p1, atol=1e-07), f"Fp32 model states {p0} is not equal to {p1}"

    elif isinstance(saved_model.optimizer, DeepSpeedZeroOptimizer):
        for p0, p1 in zip(saved_model.optimizer.single_partition_of_fp32_groups,
                          loaded_model.optimizer.single_partition_of_fp32_groups):
            assert id(p0) != id(p1), f'Comparing fp32 model state tensor against itself: {id(p0)} <====> {id(p1)}'
            assert torch.allclose(p0, p1, atol=1e-07), f"Fp32 model states {p0} is not equal to {p1}"

    elif isinstance(saved_model.optimizer, FP16_Optimizer):
        for p0, p1 in zip(saved_model.optimizer.fp32_groups_flat, loaded_model.optimizer.fp32_groups_flat):
            assert id(p0) != id(p1), f'Comparing fp32 model state tensor against itself: {id(p0)} <====> {id(p1)}'
            assert torch.allclose(p0, p1, atol=1e-07), f"FP32 model states {p0} is not equal to {p1}"

    elif isinstance(saved_model.optimizer, FP16_UnfusedOptimizer):
        for params0, params1 in zip(saved_model.optimizer.fp32_groups, loaded_model.optimizer.fp32_groups):
            for p0, p1 in zip(params0, params1):
                assert id(p0) != id(p1), f'Comparing fp32 model state tensor against itself: {id(p0)} <====> {id(p1)}'
                assert torch.allclose(p0, p1, atol=1e-07), f"FP32 model states {p0} is not equal to {p1}"
    elif isinstance(saved_model.optimizer, torch.optim.Optimizer):
        pass
    else:
        assert False, f'Unexpected Optimizer Type: {saved_model.optimizer}'


def compare_state_dicts(state0, state1, expected_mismatch_keys=[]):
    for (k0, s0), (k1, s1) in zip(state0.items(), state1.items()):
        assert k0 == k1, f'failure due to key mismatch {k0} != {k1}'
        if k0 in expected_mismatch_keys:
            continue
        if isinstance(s0, torch.Tensor) and isinstance(s1, torch.Tensor):
            assert id(s0) != id(s1), f'Comparing optimizer state tensor against itself: {id(s0)} <====> {id(s1)}'
            assert torch.equal(s0.to('cpu'), s1.to('cpu'))
        else:
            assert s0 == s1, f'failures with keys = {k0}, {k1}, values = {type(s0[0])} and {type(s1[0])}'


def compare_opt_state_dicts(state0, state1, expected_mismatch_keys=[]):
    for param_group0, saved_param_group1 in zip(state0['param_groups'], state1['param_groups']):
        compare_state_dicts(param_group0, saved_param_group1, expected_mismatch_keys)

    assert "state" in state0
    assert "state" in state1
    assert len([state0["state"].keys()]) == len([state1["state"].keys()])

    for (k0, s0), (k1, s1) in zip(state0["state"].items(), state1["state"].items()):
        assert k0 == k1, f'failure due to key mismatch {k0} != {k1}'
        compare_state_dicts(s0, s1, expected_mismatch_keys)


def compare_optimizer_states(saved_model, loaded_model, hidden_dim, fp16=True):
    saved_optimizer = saved_model.optimizer.optimizer if fp16 else saved_model.optimizer
    loaded_optimizer = loaded_model.optimizer.optimizer if fp16 else loaded_model.optimizer

    for state0, state1 in zip(saved_optimizer.state.values(), loaded_optimizer.state.values()):
        compare_state_dicts(state0, state1)


def compare_lr_scheduler_states(saved_model, loaded_model):
    assert hasattr(saved_model, 'lr_scheduler')
    assert hasattr(loaded_model, 'lr_scheduler')

    saved_scheduler = saved_model.lr_scheduler
    loaded_scheduler = loaded_model.lr_scheduler

    assert hasattr(saved_scheduler, 'state_dict')
    assert hasattr(loaded_scheduler, 'state_dict')

    saved_sd = saved_scheduler.state_dict()
    loaded_sd = loaded_scheduler.state_dict()

    print(f"saved_sd = {saved_sd}")
    print(f"loaded_sd = {loaded_sd}")

    assert saved_sd.keys() == loaded_sd.keys()

    for state0, state1 in zip(saved_sd.values(), loaded_sd.values()):
        if isinstance(state0, numbers.Number) and isinstance(state1, numbers.Number):
            assert state0 == state1


# following mixture-of-experts.md
def create_moe_param_groups(model):
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

    parameters = {'params': [p for p in model.parameters()], 'name': 'parameters'}
    return split_params_into_different_moe_groups_for_optimizer(parameters)


def create_deepspeed_model(config_dict, model, base_optimizer):
    ds_model, _, _, _ = deepspeed.initialize(config=config_dict,
                                             model=model,
                                             model_parameters=create_moe_param_groups(model),
                                             optimizer=base_optimizer)
    ds_model.empty_partition_cache()
    return ds_model


def checkpoint_correctness_verification(config_dict,
                                        models,
                                        hidden_dim,
                                        tmpdir,
                                        load_optimizer_states=False,
                                        load_lr_scheduler_states=False,
                                        train_batch=False,
                                        base_optimizers=[None, None],
                                        empty_tag=False,
                                        seq_dataloader=False,
                                        load_module_only=False,
                                        dtype=None):
    if dtype == None:
        dtype = preferred_dtype()

    ds_model = create_deepspeed_model(config_dict=config_dict, model=models[0], base_optimizer=base_optimizers[0])

    if seq_dataloader:
        data_loader = sequence_dataloader(model=ds_model,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=ds_model.device,
                                          dtype=dtype)
    else:
        data_loader = random_dataloader(model=ds_model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=ds_model.device,
                                        dtype=dtype)

    if train_batch:
        ds_model.set_dataloader(data_loader)
        for _, batch in enumerate(data_loader):
            loss = ds_model.train_batch()
    else:
        for _, batch in enumerate(data_loader):
            loss = ds_model(batch[0], batch[1])
            ds_model.backward(loss)
            ds_model.step()

    # Flush zero stage 3 cache
    ds_model.empty_partition_cache()

    trained_model = ds_model

    save_folder = os.path.join(tmpdir, 'saved_checkpoint')
    save_tag = None if empty_tag else '1'

    trained_model.save_checkpoint(save_folder, tag=save_tag)

    dist.barrier()

    for root, _, files in os.walk(save_folder):
        for f in files:
            if "_expert_" in f and "_model_states" in f:
                expert = torch.load(os.path.join(root, f))
                needed, storages = 0, {}
                for name, tensor in expert.items():
                    needed += tensor.size().numel()
                    storage = tensor.storage()
                    # some storage can be shared within an expert's checkpoint
                    storages[storage.data_ptr()] = storage.size()
                stored = sum(v for _, v in storages.items())
                assert needed == stored, f"MoE expert checkpoint uses more storage than required: {f}"

    loaded_model = create_deepspeed_model(config_dict=config_dict, model=models[1], base_optimizer=base_optimizers[1])
    assert list(trained_model.parameters())[0].dtype == list(loaded_model.parameters())[0].dtype

    context = patch.object(loaded_model, "_get_optimizer_ckpt_name",
                           wraps=loaded_model._get_optimizer_ckpt_name) if not load_optimizer_states else MagicMock()
    with context as optim_load_state_dict_mock:
        loaded_model.load_checkpoint(save_folder,
                                     tag=save_tag,
                                     load_optimizer_states=load_optimizer_states,
                                     load_lr_scheduler_states=load_lr_scheduler_states,
                                     load_module_only=load_module_only)
        if not load_optimizer_states:
            # should not attempt to get the file name to load it
            optim_load_state_dict_mock.assert_not_called()

    compare_model_states(trained_model,
                         loaded_model,
                         compare_optimizer=load_optimizer_states,
                         load_module_only=load_module_only)

    if load_optimizer_states:
        compare_optimizer_states(trained_model, loaded_model, hidden_dim, dtype == torch.float16)

    if load_lr_scheduler_states:
        compare_lr_scheduler_states(trained_model, loaded_model)
