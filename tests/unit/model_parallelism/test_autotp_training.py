# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import deepspeed.comm as dist
import torch
import math
from copy import deepcopy

from unit.common import DistributedTest, preferred_dtype
import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.simple_model import SimpleModel, random_dataloader
from deepspeed.utils import groups
from contextlib import contextmanager
from torch import nn
from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer, set_autotp_mode, is_autotp_training_mode
from unit.checkpoint.common import compare_lr_scheduler_states, compare_optimizer_states
import os
from deepspeed.runtime.utils import is_model_parallel_parameter


def skip_on_device():
    if get_accelerator().device_name() == 'xpu':
        pytest.skip("XPU requires a higher version for test")


def reset_tp_model_init_state():
    deepspeed._TP_MODEL_INIT_ARGS = None
    set_autotp_mode(training=False)


class DummyMPU:

    def __init__(self, tp_world_size=1):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.tp_world_size = tp_world_size
        self.dp_group = dist.get_world_group()
        self.tp_group = dist.get_world_group()

    def get_model_parallel_rank(self):
        return self.rank % self.tp_world_size

    def get_model_parallel_world_size(self):
        return self.tp_world_size

    def get_data_parallel_rank(self):
        return self.rank // self.tp_world_size

    def get_data_parallel_world_size(self):
        return self.world_size // self.tp_world_size

    def get_data_parallel_group(self):
        return self.dp_group

    def get_model_parallel_group(self):
        return self.tp_group


class SequentialLinearModel(torch.nn.Module):

    def __init__(self, hidden_dim, empty_grad=False, nlayers=1):
        super(SequentialLinearModel, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(nlayers)])
        if empty_grad:
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.empty_grad = empty_grad

    def forward(self, x, y):
        if len(self.linears) == 1:
            x = self.linears[0](x)
        else:
            for i, l in enumerate(self.linears):
                x = self.linears[i](x)
        return self.cross_entropy_loss(x, y)


@contextmanager
def should_assert_with_msg(expected_message):
    try:
        yield
    except AssertionError as e:
        if dist.get_rank() == 0:
            print(expected_message)
            print(str(e))
        if str(e) == expected_message:
            pass
        else:
            raise e
    else:
        raise AssertionError(f"Expected AssertionError with message '{expected_message}' "
                             "but no exception was raised.")


@pytest.mark.parametrize("tp_size", [2, 4])
class TestTpParallelStates(DistributedTest):
    world_size = 4

    def test(self, tp_size: int):
        skip_on_device()
        dp_size = 4 / tp_size
        hidden_dim = 128
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": tp_size
            },
            "zero_optimization": {
                "stage": 0
            }
        }
        model = SimpleModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        assert groups.get_tensor_model_parallel_world_size() == tp_size
        assert groups.get_data_parallel_world_size() == dp_size


class TestTpModelInitCompatibility(DistributedTest):
    world_size = 4
    reuse_dist_env = False

    def test_tp_model_init_merges_config(self):
        skip_on_device()
        reset_tp_model_init_state()
        model = SimpleModel(hidden_dim=8)
        deepspeed.tp_model_init(model, tp_size=1, dtype=preferred_dtype())
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 0,
            }
        }
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=config_dict,
                                               mpu=DummyMPU())
        assert engine.autotp_size() == 1
        assert is_autotp_training_mode()

    def test_tp_model_init_config_autotp_size_mismatch(self):
        skip_on_device()
        reset_tp_model_init_state()
        model = SimpleModel(hidden_dim=8)
        deepspeed.tp_model_init(model, tp_size=1, dtype=preferred_dtype())
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2,
            },
            "zero_optimization": {
                "stage": 0,
            }
        }
        with pytest.raises(ValueError, match="tensor_parallel.autotp_size"):
            deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict, mpu=DummyMPU())

    def test_tp_model_init_requires_mpu_or_mesh_param(self):
        skip_on_device()
        reset_tp_model_init_state()
        model = SimpleModel(hidden_dim=8)
        deepspeed.tp_model_init(model, tp_size=1, dtype=preferred_dtype())
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 0,
            }
        }
        with pytest.raises(ValueError, match="requires mpu or mesh_param"):
            deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

    def test_tp_model_init_tp_group_rejects_mpu(self):
        skip_on_device()
        reset_tp_model_init_state()
        model = SimpleModel(hidden_dim=8)
        tp_group = dist.new_group(ranks=[0])
        deepspeed.tp_model_init(model, tp_size=1, dtype=preferred_dtype(), tp_group=tp_group)
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 0,
            }
        }
        with pytest.raises(ValueError, match="tp_model_init provided tp_group"):
            deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict, mpu=DummyMPU())

    def test_tp_model_init_dtype_mismatch(self):
        skip_on_device()
        reset_tp_model_init_state()
        model = SimpleModel(hidden_dim=8)
        deepspeed.tp_model_init(model, tp_size=1, dtype=torch.float16)
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "bf16": {
                "enabled": True,
            },
            "zero_optimization": {
                "stage": 0,
            }
        }
        with pytest.raises(ValueError, match="Conflicting dtype"):
            deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict, mpu=DummyMPU())

    @pytest.mark.sequential
    @pytest.mark.parametrize("tp_size", [2, 4])
    @pytest.mark.parametrize("tp_overlap_comm", [True, False])
    def test_tp_model_init_row_parallel(self, tp_size: int, tp_overlap_comm: bool):
        run_tp_layer_fwd_bwd(tp_size, tp_overlap_comm, column_parallel=False, use_tp_model_init=True)

    @pytest.mark.sequential
    @pytest.mark.parametrize("tp_size", [2, 4])
    @pytest.mark.parametrize("tp_overlap_comm", [True, False])
    def test_tp_model_init_column_parallel(self, tp_size: int, tp_overlap_comm: bool):
        run_tp_layer_fwd_bwd(tp_size, tp_overlap_comm, column_parallel=True, use_tp_model_init=True)


@pytest.mark.parametrize("tp_size", [2, 4])
class TestTpDataloaderCorrectness(DistributedTest):
    world_size = 4
    reuse_dist_env = False

    def test(self, tp_size: int):
        skip_on_device()
        hidden_dim = 128
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "tensor_parallel": {
                "autotp_size": tp_size
            },
            "zero_optimization": {
                "stage": 0,
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        model = SimpleModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        torch.manual_seed(42)

        data_loader = random_dataloader(model=model,
                                        total_samples=3,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=preferred_dtype())
        dist.barrier()
        with should_assert_with_msg(
                "Data inconsistency within the TP group. Please check the Dataloader implementation to ensure consistency."
        ):
            for batch in data_loader:
                # batch[0].requires_grad = requires_grad
                batch[0] += dist.get_rank()
                model(batch[0], batch[1])

        model = SimpleModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        data_loader = random_dataloader(model=model,
                                        total_samples=3,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=preferred_dtype())
        for batch in data_loader:
            dist.broadcast(batch[0],
                           src=groups.get_tensor_model_parallel_src_rank(),
                           group=groups.get_tensor_model_parallel_group())
            dist.broadcast(batch[1],
                           src=groups.get_tensor_model_parallel_src_rank(),
                           group=groups.get_tensor_model_parallel_group())
            model(batch[0], batch[1])


def process_linear_layer(hidden_dim, input):
    torch.manual_seed(42)
    torch_linear = nn.Linear(hidden_dim,
                             hidden_dim,
                             dtype=preferred_dtype(),
                             device=get_accelerator().current_device())
    torch_out = torch_linear(input)
    torch_loss = torch_out.sum()
    torch_loss.backward()
    return torch_linear, torch_out


def run_tp_layer_fwd_bwd(tp_size, tp_overlap_comm, column_parallel, use_tp_model_init=False):
    skip_on_device()
    hidden_dim = 128
    batch_size_per_device = 1
    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-6
            }
        },
        "tensor_parallel": {
            "autotp_size": tp_size,
            "tp_overlap_comm": tp_overlap_comm
        },
        "zero_optimization": {
            "stage": 0,
        }
    }
    if preferred_dtype() is torch.float16:
        config_dict["fp16"] = {"enabled": True}
    elif preferred_dtype() is torch.bfloat16:
        config_dict["bf16"] = {"enabled": True}

    model = SequentialLinearModel(hidden_dim=hidden_dim)
    if use_tp_model_init:
        reset_tp_model_init_state()
        deepspeed.tp_model_init(model, tp_size=tp_size, dtype=preferred_dtype())
        mpu = DummyMPU(tp_world_size=tp_size)
        model, _, _, _ = deepspeed.initialize(model=model,
                                              model_parameters=model.parameters(),
                                              config=config_dict,
                                              mpu=mpu)
    else:
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

    input = torch.randn(batch_size_per_device,
                        hidden_dim,
                        dtype=preferred_dtype(),
                        requires_grad=True,
                        device=get_accelerator().current_device())
    dist.broadcast(input, groups.get_tensor_model_parallel_src_rank(), group=groups.get_tensor_model_parallel_group())

    torch_linear, torch_out = process_linear_layer(hidden_dim, input)
    if column_parallel:
        linear = LinearLayer(deepcopy(torch_linear), groups.get_tensor_model_parallel_group())
        out = linear(input.to(get_accelerator().current_device()))
        loss = out.sum()
        loss.backward()

        cur_device_out = torch.chunk(torch_out, tp_size, dim=-1)[groups.get_tensor_model_parallel_rank()]
        torch_grad = torch.chunk(torch_linear.weight.grad, tp_size, dim=0)[groups.get_tensor_model_parallel_rank()]
        torch_bias_grad = torch.chunk(torch_linear.bias.grad, tp_size, dim=0)[groups.get_tensor_model_parallel_rank()]

        torch.testing.assert_close(linear.bias.grad,
                                   torch_bias_grad.to(get_accelerator().current_device()),
                                   atol=1e-3,
                                   rtol=1e-3)
        torch.testing.assert_close(linear.weight.grad,
                                   torch_grad.to(get_accelerator().current_device()),
                                   atol=1e-3,
                                   rtol=1e-3)
        torch.testing.assert_close(cur_device_out.to(get_accelerator().current_device()).contiguous(),
                                   out.contiguous(),
                                   atol=1e-2,
                                   rtol=1e-2)
    else:
        linear = LinearAllreduce(deepcopy(torch_linear), groups.get_tensor_model_parallel_group())
        input_ = torch.chunk(input, tp_size, dim=-1)[groups.get_tensor_model_parallel_rank()]
        out = linear(input_.to(get_accelerator().current_device()))
        loss = out.sum()
        loss.backward()

        torch_grad = torch.chunk(torch_linear.weight.grad, tp_size, dim=1)[groups.get_tensor_model_parallel_rank()]
        torch_bias_grad = torch_linear.bias.grad
        torch.testing.assert_close(linear.bias.grad,
                                   torch_bias_grad.to(get_accelerator().current_device()),
                                   atol=1e-3,
                                   rtol=1e-3)
        torch.testing.assert_close(linear.weight.grad,
                                   torch_grad.to(get_accelerator().current_device()),
                                   atol=1e-3,
                                   rtol=1e-3)
        torch.testing.assert_close(out, torch_out.to(get_accelerator().current_device()), atol=1e-2, rtol=1e-2)


@pytest.mark.sequential
@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("tp_overlap_comm", [True, False])
class TestTpLayerFwdBwd(DistributedTest):
    world_size = 4
    reuse_dist_env = False

    def testRowParallel(self, tp_size: int, tp_overlap_comm: bool):
        run_tp_layer_fwd_bwd(tp_size, tp_overlap_comm, column_parallel=False)

    def testColumnParallel(self, tp_size: int, tp_overlap_comm: bool):
        run_tp_layer_fwd_bwd(tp_size, tp_overlap_comm, column_parallel=True)


# @pytest.mark.sequential
class TestParamsGather(DistributedTest):
    world_size = 4
    reuse_dist_env = False

    @pytest.mark.parametrize("layer_type", ["linear", "linearallreduce"])
    def test(self, layer_type):
        skip_on_device()
        tp_size = 4
        hidden_dim = 128
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "tensor_parallel": {
                "autotp_size": tp_size
            },
            "zero_optimization": {
                "stage": 0,
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        torch.manual_seed(42)
        model = SequentialLinearModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        torch_linear = nn.Linear(hidden_dim, hidden_dim, dtype=preferred_dtype(), device="cpu")
        total_params = sum(p.numel() for p in torch_linear.parameters())
        tp_layer = None
        if layer_type == "linear":
            tp_layer = LinearLayer(deepcopy(torch_linear), groups.get_tensor_model_parallel_group())
        elif layer_type == "linearallreduce":
            tp_layer = LinearAllreduce(deepcopy(torch_linear), groups.get_tensor_model_parallel_group())
        else:
            raise ValueError(f"Invalid linear type: {config_dict['linear_type']}")

        tp_params = sum(p.numel() for p in tp_layer.parameters())

        expected_tp_params = 0
        # compute expected TP params:
        # - column-parallel (LinearLayer): weight & bias both split => total // tp_size
        # - row-parallel    (LinearAllreduce): weight split, bias (1d tensors) replicated
        if layer_type == "linearallreduce":
            weight_params = torch_linear.weight.numel()
            bias_params = torch_linear.bias.numel()
            expected_tp_params = weight_params // tp_size + bias_params
        else:
            expected_tp_params = total_params // tp_size
        assert expected_tp_params == tp_params, (
            f"{layer_type}: expected {expected_tp_params} tp params, got {tp_params}")

        for name, param in tp_layer.named_parameters(recurse=False):
            if is_model_parallel_parameter(param):
                param.gather_params([param])

        torch_linear = torch_linear.to(get_accelerator().current_device())
        is_same_weights = all(
            torch.equal(param1, param2) for param1, param2 in zip(tp_layer.parameters(), torch_linear.parameters()))

        assert is_same_weights

        params1 = sum(p.numel() for p in tp_layer.parameters())
        assert total_params == params1

        for name, param in tp_layer.named_parameters(recurse=False):
            if is_model_parallel_parameter(param):
                param._tp_partition([param])

        tp_params2 = sum(p.numel() for p in tp_layer.parameters())

        assert expected_tp_params == tp_params2


def dummy_init_engine(config):
    # This is a dummy initialization function for the DeepSpeed engine.
    # We only need to use the config to initialize the distributed settings for the test.
    model = SequentialLinearModel(hidden_dim=8)
    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)


def prepare_tp_model(hidden_dim, nlayers, linear_indices, allreduce_indices, group, return_global_copy=False):
    model = SequentialLinearModel(hidden_dim=hidden_dim, nlayers=nlayers).to(preferred_dtype())
    base_model = None
    if return_global_copy:
        base_model = deepcopy(model)
    for i in linear_indices:
        layer = LinearLayer(model.linears[i], group)
        model.linears[i] = layer

    for i in allreduce_indices:
        layer = LinearAllreduce(model.linears[i], group)
        model.linears[i] = layer

    return model, base_model


@pytest.mark.parametrize("zero_stage", [0, 1, 2])
@pytest.mark.parametrize("tp_size", [2, 4])
class TestSave(DistributedTest):

    world_size = 4
    reuse_dist_env = False

    def test_save_original_weight(self, tp_size: int, zero_stage: int):
        skip_on_device()
        hidden_dim = 64
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "tensor_parallel": {
                "autotp_size": tp_size
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}
        dummy_init_engine(config_dict)
        torch.manual_seed(42)

        model, base_model = prepare_tp_model(hidden_dim,
                                             8, [2, 5], [3, 6],
                                             groups.get_tensor_model_parallel_group(),
                                             return_global_copy=True)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        cur_params_numel = sum(p.numel() for p in model.parameters())
        base_params_numel = sum(p.numel() for p in base_model.parameters())
        assert cur_params_numel < base_params_numel

        tp_state_dict = model._consolidated_16bit_state_dict()

        def compare_state_dicts(state_dict1, state_dict2):
            if state_dict1.keys() != state_dict2.keys():
                print("The state_dicts have different keys!")
                return False

            for key in state_dict1:
                if not torch.allclose(state_dict1[key], state_dict2[key], atol=1e-3):
                    assert state_dict1[key].device == "cpu"
                    print(f"Parameters for {key} are different!")
                    return False

            return True

        base_state_dict = base_model.state_dict()
        if dist.get_rank() == 0:
            # we should consider the case when zero3 is used in the future.
            assert compare_state_dicts(base_state_dict, tp_state_dict), "State_dict is not the same!"
        else:
            assert tp_state_dict is None, "noly rank0 should have the state_dict"

    def test_ckpt_save(self, tmpdir, tp_size: int, zero_stage: int):
        skip_on_device()
        hidden_dim = 64
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
            },
            "tensor_parallel": {
                "autotp_size": tp_size
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 0.001,
                    "warmup_num_steps": 1000
                }
            }
        }

        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        dummy_init_engine(config_dict)

        trained_model, _ = prepare_tp_model(hidden_dim, 8, [2, 5], [3, 6], groups.get_tensor_model_parallel_group())
        loaded_model, _ = prepare_tp_model(hidden_dim, 8, [2, 5], [3, 6], groups.get_tensor_model_parallel_group())

        trained_model, _, _, _ = deepspeed.initialize(model=trained_model,
                                                      model_parameters=trained_model.parameters(),
                                                      config=config_dict)
        torch.manual_seed(42)

        data_loader = random_dataloader(model=trained_model,
                                        total_samples=3,
                                        hidden_dim=hidden_dim,
                                        device=trained_model.device,
                                        dtype=preferred_dtype())
        ckpt_path = os.path.join(tmpdir, 'tp_saved_checkpoint')
        for i, batch in enumerate(data_loader):
            batch[0].requires_grad = True
            loss = trained_model(batch[0], batch[1])
            loss = loss
            trained_model.backward(loss)
            trained_model.step()
        trained_model.save_checkpoint(ckpt_path)

        loaded_model, _, _, _ = deepspeed.initialize(model=loaded_model,
                                                     model_parameters=loaded_model.parameters(),
                                                     config=config_dict)
        loaded_model.load_checkpoint(ckpt_path, load_optimizer_states=True, load_lr_scheduler_states=True)
        compare_optimizer_states(trained_model, loaded_model, hidden_dim, fp16=(preferred_dtype() == torch.float16))
        compare_lr_scheduler_states(trained_model, loaded_model)


@pytest.mark.parametrize("zero_stage", [0, 1, 2])
@pytest.mark.parametrize("tp_size", [2, 4])
class TestTpGradNorm(DistributedTest):

    world_size = 4
    reuse_dist_env = False

    def test(self, tp_size: int, zero_stage: int):
        skip_on_device()
        hidden_dim = 64
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "tensor_parallel": {
                "autotp_size": tp_size
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            if zero_stage == 0:
                pytest.skip(
                    "This test has an overflow data and needs to implement an overflow skip mechanism in BF16_optimizer"
                )
            config_dict["bf16"] = {"enabled": True}

        torch.manual_seed(42)

        dummy_init_engine(config=config_dict)
        tp_model, base_model = prepare_tp_model(hidden_dim,
                                                8, [2, 5], [3, 6],
                                                groups.get_tensor_model_parallel_group(),
                                                return_global_copy=True)

        base_model, base_optimizer, _, _ = deepspeed.initialize(model=base_model,
                                                                model_parameters=base_model.parameters(),
                                                                config=config_dict)
        data_loader = random_dataloader(model=base_model,
                                        total_samples=20,
                                        hidden_dim=hidden_dim,
                                        device=base_model.device,
                                        dtype=preferred_dtype())

        for i, batch in enumerate(data_loader):
            batch[0].requires_grad = True
            loss = base_model(batch[0], batch[1])
            loss = loss
            base_model.backward(loss)
            base_model.step()

        base_norm = base_optimizer._global_grad_norm

        base_model.destroy()

        tp_model, tp_optimizer, _, _ = deepspeed.initialize(model=tp_model,
                                                            model_parameters=tp_model.parameters(),
                                                            config=config_dict)
        for i, batch in enumerate(data_loader):
            batch[0].requires_grad = True
            loss = tp_model(batch[0], batch[1])
            loss = loss
            tp_model.backward(loss)
            tp_model.step()

        tp_norm = tp_optimizer._global_grad_norm

        assert math.isclose(base_norm, tp_norm, abs_tol=1e-3), f"base_norm: {base_norm}, tp_norm: {tp_norm}"
        tp_params_numel = sum(p.numel() for p in tp_model.parameters())
        base_params_numel = sum(p.numel() for p in base_model.parameters())
        assert tp_params_numel < base_params_numel, f"tp_params_numel: {tp_params_numel}, base_params_numel: {base_params_numel}"
