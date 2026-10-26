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
    return not torch.is_tensor(t) or t.numel() == 1


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
        # fp16 configs crash deepspeed.initialize's sanity check on accelerators
        # without fp16 support, surfacing as a setup error for every dependent
        # test instead of a skip.
        if dtype == torch.float16 and not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported on this accelerator")
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
class TestZeROUniversalCheckpointDP(DistributedTest):

    def _run_test(self, tmpdir, dtype, ds_config, load_optim, use_torch_adam, world_size):
        if dtype == torch.bfloat16 and not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        hidden_dim = 10
        loaded_model_state, loaded_optimizer_state = torch.load(f"{tmpdir}/baseline_state.pt", weights_only=False)

        ds_config["checkpoint"] = {"load_universal": True}
        univ_model = SimpleModel(hidden_dim, nlayers=2)
        univ_model = init_ds_engine(univ_model, ds_config, use_torch_adam)
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
    def test_dp_world_size_2to2(self, baseline_ws2, tmpdir, dtype, ds_config, load_optim, use_torch_adam):
        self._run_test(tmpdir, dtype, ds_config, load_optim, use_torch_adam, 2)

    @pytest.mark.world_size(2)
    def test_dp_world_size_4to2(self, baseline_ws4, tmpdir, dtype, ds_config, load_optim, use_torch_adam):
        self._run_test(tmpdir, dtype, ds_config, load_optim, use_torch_adam, 2)

    @pytest.mark.world_size(4)
    def test_dp_world_size_2to4(self, baseline_ws2, tmpdir, dtype, ds_config, load_optim, use_torch_adam):
        self._run_test(tmpdir, dtype, ds_config, load_optim, use_torch_adam, 4)


@pytest.mark.parametrize("fused,capturable", [(True, False), (False, True), (False, False)])
def test_zero_optimizer_normalize_step_tensor(fused, capturable):
    from deepspeed.runtime.base_optimizer import ZeROOptimizer
    from deepspeed.accelerator import get_accelerator

    acc = get_accelerator()
    target_device = torch.device(acc.device_name(0) if acc.is_available() else 'cpu')
    param = torch.nn.Parameter(torch.randn(2, 2, device=target_device))
    group = {"fused": fused, "capturable": capturable}

    # Step on CPU as a float tensor
    cpu_step = torch.tensor(3.0, device="cpu")
    normalized = ZeROOptimizer._normalize_step_tensor(cpu_step, param, group)
    if fused or capturable:
        assert normalized.device == param.device
        assert normalized.dtype == torch.float32
        assert normalized.item() == 3.0
    else:
        assert normalized.device == torch.device("cpu")
        assert normalized.item() == 3.0

    # Step on accelerator as a float tensor
    if acc.is_available():
        acc_step = torch.tensor(4.0, device=acc.device_name(0))
        normalized = ZeROOptimizer._normalize_step_tensor(acc_step, param, group)
        if fused or capturable:
            assert normalized.device == param.device
            assert normalized.dtype == torch.float32
            assert normalized.item() == 4.0
        else:
            assert normalized.device == torch.device("cpu")
            assert normalized.item() == 4.0

    # Integer scalar and integer tensor steps
    for raw_step in (5, torch.tensor(5, dtype=torch.int64)):
        normalized = ZeROOptimizer._normalize_step_tensor(raw_step, param, group)
        if fused or capturable:
            assert normalized.device == param.device
            assert normalized.dtype == torch.float32
            assert normalized.item() == 5.0
        else:
            if isinstance(raw_step, int):
                assert normalized == 5 and not torch.is_tensor(normalized)
            else:
                assert normalized.item() == 5 and normalized.device == torch.device("cpu")


@pytest.mark.parametrize("fused,capturable", [(True, False), (False, True), (False, False)])
def test_universal_checkpoint_load_hp_step_device_policy(tmp_path, fused, capturable):
    from deepspeed.runtime.base_optimizer import ZeROOptimizer
    from deepspeed.accelerator import get_accelerator

    acc = get_accelerator()
    if (fused or capturable) and not acc.is_available():
        pytest.skip("Accelerator required for fused/capturable optimizer test")

    # Guard against capturable=True on CPU device
    if capturable and (not torch.cuda.is_available()  #ignore-cuda
                       or acc.is_available() and acc.device_name() == "cpu"):
        pytest.skip("PyTorch capturable optimizers require a CUDA/GPU accelerator device")

    device = torch.device(acc.device_name(0) if acc.is_available() else "cpu")

    zero_dir = tmp_path / "zero"
    zero_dir.mkdir(parents=True)

    optim_sd = {
        "param_groups": [{
            "fused": fused,
            "capturable": capturable,
            "lr": 0.001
        }],
        "ds_version": "0.14.0",
    }
    torch.save(optim_sd, zero_dir / "optimizer_state.pt")

    param = torch.nn.Parameter(torch.randn(4, 4, device=device))
    client_optimizer = torch.optim.AdamW([param], lr=0.001, fused=fused, capturable=capturable)

    class DummyZeROOptimizer(ZeROOptimizer):

        def __init__(self, optimizer):
            super().__init__()
            self.optimizer = optimizer
            self.mpu = None
            self.param_names = {param: "param_0"}
            self.dummy_groups = [[param]]

        def _load_global_state(self, sd):
            pass

    param._hp_mapping = SimpleNamespace(get_optim_state_keys=lambda: [])
    # Universal checkpoint restores steps from disk on CPU
    param.load_hp_checkpoint_state = lambda *args, **kwargs: torch.tensor(7.0, device="cpu")

    zero_opt = DummyZeROOptimizer(client_optimizer)
    zero_opt.load_hp_checkpoint_state_from_checkpoint_dir("dummy_groups", str(tmp_path))

    restored_step = client_optimizer.state[param]["step"]
    if fused or capturable:
        assert restored_step.device == param.device
        assert restored_step.dtype == torch.float32
    else:
        assert restored_step.device == torch.device("cpu")

    # Verify that the restored optimizer state allows step() to execute without device errors
    client_optimizer.state[param]["exp_avg"] = torch.zeros_like(param)
    client_optimizer.state[param]["exp_avg_sq"] = torch.zeros_like(param)
    param.grad = torch.randn_like(param)
    client_optimizer.step()
    assert client_optimizer.state[param]["step"].item() == 8.0


@pytest.mark.parametrize("fused,capturable", [(True, False), (False, True), (False, False)])
def test_universal_checkpoint_stage3_load_hp_step_device_policy(tmp_path, fused, capturable):
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
    from deepspeed.checkpoint.constants import OPTIMIZER_STATE_DICT
    from deepspeed.accelerator import get_accelerator

    acc = get_accelerator()
    if (fused or capturable) and not acc.is_available():
        pytest.skip("Accelerator required for fused/capturable optimizer test")

    # Guard against capturable=True on CPU device
    if capturable and (not torch.cuda.is_available()  #ignore-cuda
                       or acc.is_available() and acc.device_name() == "cpu"):
        pytest.skip("PyTorch capturable optimizers require a CUDA/GPU accelerator device")

    device = torch.device(acc.device_name(0) if acc.is_available() else "cpu")

    zero_dir = tmp_path / "zero"
    zero_dir.mkdir(parents=True)

    fp32_param = torch.nn.Parameter(torch.randn(16, device=device))
    client_optimizer = torch.optim.AdamW([fp32_param], lr=0.001, fused=fused, capturable=capturable)

    optim_sd = {
        OPTIMIZER_STATE_DICT: {
            "state": [{
                "step": torch.tensor(12.0, device="cpu")
            }],
            "param_groups": [{
                "fused": fused,
                "capturable": capturable,
                "lr": 0.001
            }],
        },
        "ds_version": "0.14.0",
    }
    torch.save(optim_sd, zero_dir / "optimizer_state.pt")

    class DummyStage3Optimizer(DeepSpeedZeroOptimizer_Stage3):

        def __init__(self, optimizer):
            self._backward_hook_state = SimpleNamespace()
            self._is_gradient_accumulation_boundary = True
            self.optimizer = optimizer
            self.fp32_partitioned_groups_flat = [fp32_param]
            self.fp16_partitioned_groups_flat = [fp32_param]
            self.fp16_groups = [[fp32_param]]
            self.fp16_partitioned_groups = [[fp32_param]]
            self.param_names = {fp32_param: "param_0"}
            self.sub_group_to_group_id = {0: 0}
            self.swap_optimizer = False

        def _load_global_state_stage3(self, sd):
            pass

        def load_hp_checkpoint_state(self, path, key, param=None):
            return torch.zeros_like(param)

        def unflatten(self, flat, params):
            return [flat]

    stage3_opt = DummyStage3Optimizer(client_optimizer)
    stage3_opt.load_hp_checkpoint_state_from_checkpoint_dir_stage3(str(tmp_path))

    restored_step = client_optimizer.state[fp32_param]["step"]
    if fused or capturable:
        assert restored_step.device == fp32_param.device
        assert restored_step.dtype == torch.float32
    else:
        assert restored_step.device == torch.device("cpu")

    client_optimizer.state[fp32_param]["exp_avg"] = torch.zeros_like(fp32_param)
    client_optimizer.state[fp32_param]["exp_avg_sq"] = torch.zeros_like(fp32_param)
    client_optimizer.param_groups[0]["params"] = [fp32_param]
    fp32_param.grad = torch.randn_like(fp32_param)
    client_optimizer.step()
    assert client_optimizer.state[fp32_param]["step"].item() == 13.0
