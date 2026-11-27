# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
import pytest
from deepspeed.ops.adam import FusedAdam
from unit.common import DistributedTest
from deepspeed.ops.op_builder import CPUAdamBuilder
from unit.simple_model import SimpleModel, SimpleOptimizer, random_dataloader, random_dataset
from unit.util import bf16_required_version_check
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad, safe_set_full_fp32_param
from unit.v1.zero.test_zero_user_backward import (initialize_distributed, create_ddp_model, collect_ddp_gradients,
                                                  collect_gradients_safe, compare_gradients)


class TestAdamBF16ZeroOneCycleCompatibility(DistributedTest):
    world_size = 1

    def test(self, zero_stage=2, use_cpu_offload=False):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "scheduler": {
                "type": "OneCycle",
                "params": {
                    "cycle_first_step_size": 16000,
                    "cycle_first_stair_count": 8000,
                    "decay_step_size": 16000,
                    "cycle_min_lr": 1e-06,
                    "cycle_max_lr": 3e-05,
                    "decay_lr_rate": 1e-07,
                    "cycle_min_mom": 0.85,
                    "cycle_max_mom": 0.99,
                    "decay_mom_rate": 0.0
                }
            },
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            }
        }

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestZeroAllowUntestedOptimizer(DistributedTest):
    world_size = 1

    def test(self, zero_stage=2, use_cpu_offload=False):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_micro_batch_size_per_gpu": 4,
            "steps_per_print": 1,
            "fp16": {
                "enabled": False,
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            },
            "zero_allow_untested_optimizer": False
        }

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        optimizer = SimpleOptimizer(model.parameters())
        with pytest.raises(AssertionError):
            model, optim, _, _ = deepspeed.initialize(config=config_dict,
                                                      model=model,
                                                      optimizer=optimizer,
                                                      model_parameters=model.parameters())


class TestZeroEmptyPartition(DistributedTest):
    world_size = 3

    def test(self, zero_stage=2, use_cpu_offload=False):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        if zero_stage == 3:
            pytest.skip("skip for now")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload,
                "reduce_bucket_size": 100,
                "allgather_bucket_size": 100
            }
        }

        hidden_dim = 1
        model = SimpleModel(hidden_dim)

        # Ensure model has 2 parameters, to cause empty partition with DP=3
        assert len(list(model.parameters())) == 2
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        # Now make sure things work..
        data_loader = random_dataloader(model=model,
                                        total_samples=1,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize("optimizer_constructor", [torch.optim.Adam, FusedAdam])
class TestZeroSupportedClientOptimizer(DistributedTest):
    world_size = 1

    def test(self, optimizer_constructor, zero_stage=2):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "steps_per_print": 1,
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        client_optimizer = optimizer_constructor(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, optimizer=client_optimizer)


class TestZero2ReduceScatterOff(DistributedTest):
    world_size = 2

    def test(self):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "allgather_bucket_size": 2000000000,
                "reduce_bucket_size": 200000000,
                "overlap_comm": False,
                "reduce_scatter": False
            },
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestZeroEmptyGrad(DistributedTest):
    world_size = 1

    def test(self, stage=2):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": stage
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.Adam(model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize("comp_type", [torch.float16, torch.bfloat16, torch.float], ids=["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("comm_type", [torch.float16, torch.bfloat16, None], ids=["fp16", "bf16", "default"])
class TestZeroDtypeCocktail(DistributedTest):
    world_size = 2

    def test(self, comp_type, comm_type):
        if comp_type == torch.bfloat16 or comm_type == torch.bfloat16:
            if not bf16_required_version_check():
                pytest.skip(
                    " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
                )

        if comp_type == torch.float16 or comm_type == torch.float16:
            if not get_accelerator().is_fp16_supported():
                pytest.skip("fp16 is not supported")

        type_str = {torch.float16: "fp16", torch.bfloat16: "bf16"}

        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "steps_per_print": 1,
            "fp16": {
                "enabled": comp_type == torch.float16
            },
            "bf16": {
                "enabled": comp_type == torch.bfloat16
            },
            "zero_optimization": {
                "stage": 2
            },
        }
        if comm_type is not None:
            config_dict["communication_data_type"] = type_str[comm_type]
        else:
            comm_type = comp_type
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.Adam(model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=2,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=comp_type)

        def custom_reduce(tensor, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
            assert tensor.dtype == comm_type
            return orig_torch_reduce(tensor, dst, op, group, async_op)

        orig_torch_reduce = dist.reduce
        dist.reduce = custom_reduce
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
        dist.reduce = orig_torch_reduce


@pytest.mark.parametrize("bf16_optimizer_states,use_cpu_offload,zero_stage", [
    pytest.param(False, True, 1, id="zero_stage_1_cpu_offload"),
    pytest.param(True, False, 1, id="zero_stage_1_bf16_opt_states_True"),
    pytest.param(True, True, 1, id="zero_stage_1_bf16_opt_states_cpu_offload"),
    pytest.param(False, True, 2, id="zero_stage_2_cpu_offload"),
    pytest.param(True, False, 2, id="zero_stage_2_bf16_opt_states_True"),
    pytest.param(True, True, 2, id="zero_stage_2_bf16_opt_states_cpu_offload"),
    pytest.param(False, True, 3, id="zero_stage_3_cpu_offload"),
    pytest.param(True, False, 3, id="zero_stage_3_bf16_opt_states_True"),
    pytest.param(True, True, 3, id="zero_stage_3_bf16_opt_states_cpu_offload"),
])
class TestBF16MasterWeightsGradients(DistributedTest):
    world_size = 2

    def test_gradients_match_ddp(self, bf16_optimizer_states, use_cpu_offload, zero_stage):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        hidden_dim = 6
        lr = 1e-3
        seed = 123

        device, rank, dtype = initialize_distributed()

        model_ddp, optimizer_ddp = create_ddp_model(SimpleModel,
                                                    device,
                                                    rank,
                                                    dtype,
                                                    seed=seed,
                                                    lr=lr,
                                                    hidden_dim=hidden_dim,
                                                    nlayers=2)

        torch.manual_seed(seed)
        ds_model = SimpleModel(hidden_dim, nlayers=2)

        bf16_config = {
            "enabled": True,
            "bf16_master_weights_and_grads": True,
        }
        if bf16_optimizer_states:
            bf16_config["bf16_optimizer_states"] = True

        zero_config = {"stage": zero_stage}
        if use_cpu_offload:
            zero_config["cpu_offload"] = True

        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": lr
                }
            },
            "bf16": bf16_config,
            "zero_optimization": zero_config
        }

        engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                               model=ds_model,
                                               model_parameters=ds_model.parameters())

        data_loader = random_dataloader(model=engine,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=device,
                                        dtype=torch.bfloat16)
        batch = next(iter(data_loader))

        optimizer_ddp.zero_grad()
        loss_ddp = model_ddp(batch[0], batch[1])
        loss_ddp.backward()
        grads_ddp = collect_ddp_gradients(model_ddp)

        loss_ds = engine(batch[0], batch[1])
        loss_ds.backward()
        grads_ds = collect_gradients_safe(engine)

        compare_gradients(
            grads_ddp,
            grads_ds,
            step_info=
            f"bf16_optimizer_states={bf16_optimizer_states}, cpu_offload={use_cpu_offload}, zero_stage={zero_stage}")

        optimizer_ddp.step()
        optimizer_ddp.zero_grad()
        engine.step()
        engine.zero_grad()

        if bf16_optimizer_states and use_cpu_offload:
            # With CPU offload the Adam moments must be allocated in bf16 on the host so the
            # offloaded optimizer-state footprint is smaller than with fp32 moments.
            cpu_adam_state = engine.optimizer.optimizer.state
            moment_tensors = []
            for param_state in cpu_adam_state.values():
                for moment_key in ("exp_avg", "exp_avg_sq"):
                    if moment_key in param_state:
                        moment_tensors.append(param_state[moment_key])
            assert moment_tensors, "expected Adam moment tensors to be allocated after a step"
            for moment in moment_tensors:
                assert moment.dtype == torch.bfloat16, f"expected bf16 moment, got {moment.dtype}"
                assert moment.device.type == "cpu", f"expected moment on cpu, got {moment.device}"

        engine.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestBF16OptimizerStatesOffloadValidation(DistributedTest):
    world_size = 1

    def test_user_cpu_adam_must_enable_bf16_states(self, zero_stage):
        """A user-provided DeepSpeedCPUAdam must be built with fp32_optimizer_states=False
        to combine bf16_optimizer_states with ZeRO-Offload, otherwise the moments would
        silently stay fp32 and the memory benefit would be lost."""
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )
        if not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        from deepspeed.ops.adam import DeepSpeedCPUAdam

        hidden_dim = 6
        model = SimpleModel(hidden_dim, nlayers=2)
        # fp32_optimizer_states defaults to True, which keeps fp32 moments and is
        # incompatible with bf16_optimizer_states under ZeRO-Offload.
        optimizer = DeepSpeedCPUAdam(model.parameters())

        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "steps_per_print": 1,
            "bf16": {
                "enabled": True,
                "bf16_master_weights_and_grads": True,
                "bf16_optimizer_states": True,
            },
            # offload_optimizer is the current config key for ZeRO optimizer offload
            # (TestBF16MasterWeightsGradients above still uses the legacy cpu_offload alias).
            "zero_optimization": {
                "stage": zero_stage,
                "offload_optimizer": {
                    "device": "cpu"
                },
            },
        }

        with pytest.raises(AssertionError, match="fp32_optimizer_states=False"):
            deepspeed.initialize(config=config_dict, model=model, optimizer=optimizer)


class TestBF16ImmediateGradUpdateReleasesLowPrecisionGrads(DistributedTest):
    """Accumulating each gradient as it is produced must train identically and not keep a BF16 copy."""
    world_size = 2

    def _train(self, immediate_grad_update, batches, hidden_dim, seed):
        torch.manual_seed(seed)
        model = SimpleModel(hidden_dim, nlayers=2)
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": len(batches),
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-2
                }
            },
            "bf16": {
                "enabled": True,
                "immediate_grad_update": immediate_grad_update
            },
            "data_types": {
                "grad_accum_dtype": "fp32"
            },
            # BF16 with FP32 gradient accumulation selects BF16_Optimizer at ZeRO stage 1.
            "zero_optimization": {
                "stage": 1
            },
        }
        engine, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        lp_grads_after_backward = []
        fp32_grads = None
        for index, (inputs, labels) in enumerate(batches):
            engine.backward(engine(inputs, labels))
            lp_grads_after_backward.append([param.grad for param in engine.module.parameters()])
            if index == len(batches) - 1:
                # The FP32 accumulation buffer after the last micro-batch, read the same way in both modes;
                # safe_get_full_grad would return the BF16 .grad whenever one is still alive.
                fp32_grads = {
                    name: param.get_full_hp_grad().detach().cpu().clone()
                    for name, param in engine.module.named_parameters()
                }
            # DeepSpeed advances the accumulation boundary in step(); only the last call updates parameters.
            engine.step()
        parameters = {name: param.detach().float().cpu().clone() for name, param in engine.module.named_parameters()}
        engine.destroy()
        return lp_grads_after_backward, fp32_grads, parameters

    def test_matches_backward_epilogue_accumulation(self):
        if not bf16_required_version_check():
            pytest.skip("DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA >= 11.0 and HW support")
        hidden_dim = 8
        device, _, _ = initialize_distributed()
        data_loader = torch.utils.data.DataLoader(random_dataset(8, hidden_dim, device, dtype=torch.bfloat16),
                                                  batch_size=2)
        batches = [batch for batch, _ in zip(data_loader, range(2))]

        epilogue_lp, epilogue_grads, epilogue_parameters = self._train(False, batches, hidden_dim, seed=7)
        immediate_lp, immediate_grads, immediate_parameters = self._train(True, batches, hidden_dim, seed=7)

        assert all(grad is not None for grads in epilogue_lp for grad in grads)
        assert all(grad is None for grads in immediate_lp for grad in grads), "BF16 gradients outlived accumulation"
        assert epilogue_grads.keys() == immediate_grads.keys() and epilogue_grads
        for name in epilogue_grads:
            assert torch.equal(immediate_grads[name], epilogue_grads[name]), name
        for name in epilogue_parameters:
            assert torch.equal(immediate_parameters[name], epilogue_parameters[name]), name


class _GradientFragmentModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # The small parameter fits one partition; the following four-element parameter crosses it.
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.weight = torch.nn.Parameter(torch.ones(4))

    def forward(self, x):
        return (x * self.weight).float().sum() + self.bias.float().sum()


class TestBF16FullGradientReconstruction(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    @pytest.mark.parametrize("zero_stage", [1, 2])
    @pytest.mark.parametrize("micro_batches", [1, 2])
    def test_split_and_unsplit_gradients_match_the_global_reference(self, zero_stage, micro_batches):
        if not bf16_required_version_check():
            pytest.skip("DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA >= 11.0 and HW support")
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": micro_batches,
            "gradient_clipping": 0.25,
            "optimizer": {
                "type": "SGD",
                "params": {
                    "lr": 0.1
                }
            },
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": zero_stage
            },
            "bf16": {
                "enabled": True,
                "immediate_grad_update": True
            },
            "data_types": {
                "grad_accum_dtype": "fp32"
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=_GradientFragmentModel(), config=config)
        if zero_stage == 1:
            assert isinstance(engine.optimizer, BF16_Optimizer)
        rank_scale = dist.get_rank() + 1
        mean_rank_scale = (dist.get_world_size() + 1) / 2
        before = {
            name: safe_get_full_fp32_param(param).detach().cpu().clone()
            for name, param in engine.module.named_parameters()
        }
        for step in range(2):
            for micro_batch in range(micro_batches):
                x = torch.arange(1, 5, dtype=torch.bfloat16, device=engine.device)
                x = x * rank_scale + step + micro_batch
                engine.backward(engine(x.reshape(1, 4)))
                if micro_batch + 1 < micro_batches:
                    engine.step()

            expected = {
                "bias": torch.ones(1),
                "weight": torch.arange(1, 5, dtype=torch.float32) * mean_rank_scale + step + (micro_batches - 1) / 2,
            }
            for name, param in engine.module.named_parameters():
                gradient = safe_get_full_grad(param)
                assert gradient is not None and gradient.dtype == torch.float32
                torch.testing.assert_close(gradient.cpu(),
                                           expected[name],
                                           rtol=0,
                                           atol=0,
                                           msg=f"{name} at step {step}")

            expected_norm = torch.cat([gradient.flatten() for gradient in expected.values()]).double().norm()
            assert expected_norm > config["gradient_clipping"]
            clip_scale = config["gradient_clipping"] / (expected_norm.item() + 1e-6)
            engine.step()
            torch.testing.assert_close(float(engine.get_global_grad_norm()),
                                       expected_norm.item(),
                                       rtol=1e-6,
                                       atol=1e-6)
            for name, param in engine.module.named_parameters():
                after = safe_get_full_fp32_param(param).detach().cpu()
                expected_parameter = before[name] - config["optimizer"]["params"]["lr"] * expected[name] * clip_scale
                torch.testing.assert_close(after,
                                           expected_parameter,
                                           rtol=1e-6,
                                           atol=1e-7,
                                           msg=f"{name} at step {step}")
                before[name] = after.clone()
        engine.destroy()


_SR_NUMEL = (1 << 16) + 1  # odd, so the last rank's partition ends in padding
_SR_STEPS = 8
_SR_LR = 1e-4  # 8 steps move a master by 8e-4, less than half the bf16 spacing of any |w| in [0.3, 1]


class _ConstantGradientModel(torch.nn.Module):
    """Every element's gradient is 1, so each Adam step moves every fp32 master by approximately -lr."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        magnitude = torch.rand(_SR_NUMEL, generator=generator) * 0.7 + 0.3
        sign = torch.randint(0, 2, (_SR_NUMEL, ), generator=generator) * 2 - 1
        self.weight = torch.nn.Parameter((sign * magnitude).to(torch.bfloat16).float())

    def forward(self):
        return self.weight.float().sum()


def _bf16_neighbours(master):
    """The bf16 values just below and just above each fp32 ``master`` (equal when it is a bf16 value)."""
    nearest = master.to(torch.bfloat16)
    up = torch.nextafter(nearest, torch.full_like(nearest, float("inf")))
    down = torch.nextafter(nearest, torch.full_like(nearest, float("-inf")))
    master = master.double()
    low = torch.where(nearest.double() <= master, nearest, down).double()
    high = torch.where(nearest.double() >= master, nearest, up).double()
    return low, high


def _rounding_config(stage):
    return dict(train_micro_batch_size_per_gpu=1,
                bf16=dict(enabled=True, stochastic_round_params=True),
                zero_optimization=dict(stage=stage))


class TestBF16StochasticRoundRefresh(DistributedTest):
    world_size = 1

    def test_master_neighbours(self):
        if not bf16_required_version_check():
            pytest.skip("bf16 is required")
        tiny = 2.0**-133
        values = torch.tensor([1 + 2.0**-9, -(1 + 3 * 2.0**-9), 0.3, 2 - 2.0**-9, 2.5 * tiny, -1.25 * tiny])
        exact = torch.tensor([0.0, -0.0, 1.0, -0.5, tiny, -tiny, torch.finfo(torch.bfloat16).max])
        specials = torch.tensor([0x7F800000, -0x00800000, 0x7FFFFFFF, -1], dtype=torch.int32).view(torch.float32)
        draws = 1 << 16
        fixed = torch.cat([exact, specials]).repeat(64)
        master = torch.cat([values.repeat_interleave(draws), fixed])

        def round_with_seed(seed):
            torch.manual_seed(seed)
            model = torch.nn.Linear(master.numel(), 1, bias=False)
            config = _rounding_config(1)
            config["data_types"] = dict(grad_accum_dtype="fp32")
            engine, _, _, _ = deepspeed.initialize(model=model,
                                                   optimizer=torch.optim.Adam(model.parameters()),
                                                   config=config)
            weight = engine.module.weight
            safe_set_full_fp32_param(weight, master.to(weight.device).view_as(weight))
            engine.optimizer.update_lp_params()
            result = weight.detach().flatten().cpu().clone()
            engine.destroy()
            return result

        rounded = round_with_seed(0)
        assert torch.equal(rounded.view(torch.int16), round_with_seed(0).view(torch.int16))
        assert not torch.equal(rounded[:draws], round_with_seed(1)[:draws])
        low, high = _bf16_neighbours(values)
        fraction = (values.double() - low) / (high - low)
        samples = rounded[:values.numel() * draws].double().view(-1, draws)
        assert ((samples == low[:, None]) | (samples == high[:, None])).all()
        rounded_up = (samples == high[:, None]).double().mean(dim=1)
        assert ((rounded_up - fraction).abs() < 5 * (fraction * (1 - fraction) / draws).sqrt()).all()
        nearest = fixed.to(get_accelerator().current_device_name()).to(torch.bfloat16).cpu()
        assert torch.equal(rounded[-fixed.numel():].view(torch.int16), nearest.view(torch.int16))


@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("zero_stage, offload, grad_accum_dtype, optimizer_type", [
    (1, False, "fp32", BF16_Optimizer),
    (2, False, "bf16", DeepSpeedZeroOptimizer),
    (2, True, "bf16", DeepSpeedZeroOptimizer),
])
class TestBF16StochasticRoundParams(DistributedTest):
    world_size = 2

    def test_bf16_params_track_fp32_masters(self, stochastic, zero_stage, offload, grad_accum_dtype, optimizer_type):
        if not bf16_required_version_check():
            pytest.skip("bf16 is required")
        torch.manual_seed(0)
        config_dict = _rounding_config(zero_stage)
        config_dict["data_types"] = dict(grad_accum_dtype=grad_accum_dtype)
        if offload:
            config_dict["zero_optimization"]["offload_optimizer"] = dict(device="cpu")
            config_dict["zero_force_ds_cpu_optimizer"] = False
        if not stochastic:
            del config_dict["bf16"]["stochastic_round_params"]
        model = _ConstantGradientModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=_SR_LR)
        engine, _, _, _ = deepspeed.initialize(config=config_dict, model=model, optimizer=optimizer)
        assert type(engine.optimizer) is optimizer_type
        assert getattr(engine.optimizer, "cpu_offload", False) == offload

        weight = engine.module.weight
        initial = weight.detach().cpu().clone()
        for _ in range(_SR_STEPS):
            engine.backward(engine())
            engine.step()
        served = weight.detach().cpu()
        master = safe_get_full_fp32_param(weight).detach().cpu()

        gathered = [torch.empty(_SR_NUMEL, device=weight.device) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, weight.detach().float())
        assert all(torch.equal(g, gathered[0]) for g in gathered)
        expected_master = initial.double() - _SR_STEPS * _SR_LR
        # Allow one fp32 epsilon per step for the update near unit magnitude.
        master_atol = _SR_STEPS * torch.finfo(torch.float32).eps
        torch.testing.assert_close(master.double(), expected_master, rtol=0, atol=master_atol)

        if not stochastic:
            # Round-to-nearest: no master moved half a bf16 spacing, so no bf16 parameter moved.
            assert torch.equal(served, master.to(torch.bfloat16))
            assert torch.equal(served, initial)
        else:
            low, high = _bf16_neighbours(master)
            served = served.double()
            assert ((served == low) | (served == high)).all()
            between = high > low
            spacing = (high - low)[between]
            fraction = (master.double() - low)[between] / spacing
            # Unbiased: the summed error is within 5 standard errors of 0.
            error = ((served - master.double())[between] / spacing).sum()
            standard_error = (fraction * (1 - fraction)).sum().sqrt()
            assert error.abs() < 5 * standard_error, (error, standard_error)
        engine.destroy()


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_stochastic_round_requires_fp32_masters(precision):
    config = _rounding_config(2)
    config["zero_optimization"]["offload_optimizer"] = dict(device="cpu")
    config.setdefault(precision, {})[f"{precision}_master_weights_and_grads"] = True
    with pytest.raises(ValueError, match="requires fp32 master weights"):
        DeepSpeedConfig(config)


class TestBF16StochasticRoundParamsUnsupported(DistributedTest):
    world_size = 1

    def test_zero3_refused(self):
        if not bf16_required_version_check():
            pytest.skip("bf16 is required")
        model = _ConstantGradientModel()
        with pytest.raises(ValueError, match="supports BF16_Optimizer"):
            engine, _, _, _ = deepspeed.initialize(config=_rounding_config(3),
                                                   model=model,
                                                   optimizer=torch.optim.Adam(model.parameters()))
            engine.destroy()
            raise AssertionError("ZeRO-3 accepted stochastic rounding")


class TestStochasticRoundParallelism(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("tensor_parallel", [False, True])
    def test_replicated_weights(self, tensor_parallel):
        from deepspeed.runtime.pipe.topology import PipelineParallelGrid, PipeModelDataParallelTopology
        from deepspeed.runtime.sequence_parallel import parallel_state_sp

        if not bf16_required_version_check():
            pytest.skip("bf16 is required")
        initialize_distributed()
        config = _rounding_config(2)
        model = _ConstantGradientModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=_SR_LR)
        if tensor_parallel:
            mpu = PipelineParallelGrid(PipeModelDataParallelTopology(num_pp=1, num_mp=2, num_dp=1))
            with pytest.raises(ValueError, match="replicated tensor-parallel weights"):
                engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config, mpu=mpu)
                engine.destroy()
                raise AssertionError("TP accepted independent rounding of replicas")
        else:
            parallel_state_sp.initialize_sequence_parallel(2)
            config["sequence_parallel_size"] = 2
            engine, _, _, _ = deepspeed.initialize(model=model,
                                                   optimizer=optimizer,
                                                   config=config,
                                                   mpu=parallel_state_sp)
            before = model.weight.detach().clone()
            engine.backward(engine())
            engine.step()
            assert (model.weight != before).any()
            engine.destroy()


class TestPipelineTiedStochasticRound(DistributedTest):
    world_size = 2

    def test_step_and_universal_warm_start_keep_copies_equal(self, tmpdir):
        from deepspeed.checkpoint.constants import PIPELINE_REPLICATED_PARAMETER_PATTERNS, UNIVERSAL_CHECKPOINT_INFO
        from unit.checkpoint.test_autotp_uc_checkpoint import _convert_to_universal
        from deepspeed.pipe import PipelineModule, TiedLayerSpec

        if not bf16_required_version_check():
            pytest.skip("bf16 is required")
        initialize_distributed()
        torch.manual_seed(42)
        tied = TiedLayerSpec("shared", torch.nn.Linear, 32, 32, bias=False)
        layers = [tied, torch.nn.Identity(), torch.nn.Linear(32, 32, bias=False), tied]
        model = PipelineModule(layers, num_stages=2, partition_method="uniform", loss_fn=lambda x, y: x.float().mean())
        for param in model.parameters():
            param.data.fill_(1.0)
        # The second stage's extra matrix precedes the tied weight in its optimizer partition.
        optimizer = torch.optim.Adam(reversed(list(model.parameters())), lr=0.001)
        config = _rounding_config(1)
        config["checkpoint"] = dict(load_universal=True)
        config["data_types"] = dict(grad_accum_dtype="fp32")
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
        batch = (torch.ones(1, 32, dtype=torch.bfloat16), torch.zeros(1))
        engine.train_batch(data_iter=iter([batch]))
        weight = model.tied_modules["shared"].weight

        def check_tied():
            copies = [torch.empty_like(weight) for _ in range(2)]
            dist.all_gather(copies, weight.detach())
            assert weight.ne(1).any()
            assert torch.equal(*copies)

        check_tied()
        info = {PIPELINE_REPLICATED_PARAMETER_PATTERNS: [r"^tied_modules\.shared\.weight$"]}
        engine.save_checkpoint(str(tmpdir), tag="native", client_state={UNIVERSAL_CHECKPOINT_INFO: info})
        if dist.get_rank() == 0:
            _convert_to_universal(str(tmpdir / "native"), str(tmpdir / "universal"))
        dist.barrier()
        path, _ = engine.load_checkpoint(str(tmpdir), tag="universal", load_module_only=True)
        assert path
        check_tied()
        assert torch.equal(safe_get_full_fp32_param(weight), weight.float())
        engine.destroy()
