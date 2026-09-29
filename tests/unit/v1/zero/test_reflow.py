# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Contract tests for Reflow, the asynchronous CPU-offload optimizer for ZeRO-3.

Reflow is meant to be a bit-exact drop-in for ZeRO-Offload (ZeRO-3 + CPU optimizer offload), so every training
test runs the same model and batches through plain ZeRO-Offload as the oracle and compares exactly.
"""

import copy

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.constants import FP32_FLAT_GROUPS
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.op_builder import CPUAdamBuilder
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.reflow.reflow_cpu_adam import ReflowCPUAdam
from deepspeed.runtime.reflow.reflow_cpu_lion import ReflowCPULion
from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam
from deepspeed.runtime.zero.utils import ZeRORuntimeException
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_optimizer_state
from unit.common import DistributedTest
from unit.simple_model import SimpleModel


def minimal_reflow_config(reflow=None, **zero_overrides):
    """The smallest config that enables Reflow, for the checks that run when the config is parsed."""
    zero_optimization = {"stage": 3, "offload_optimizer": {"device": "cpu"}, "reflow": reflow or {}}
    zero_optimization.update(zero_overrides)
    return {"train_batch_size": 1, "zero_optimization": zero_optimization}


def test_reflow_block_configures_the_optimizer():
    config = DeepSpeedConfig(minimal_reflow_config({"state_update_cores": 4}))
    assert config.zero_config.reflow.state_update_cores == 4

    with pytest.raises(ValueError):
        DeepSpeedConfig(minimal_reflow_config({"state_update_cores": 0}))


def test_requires_stage3_and_cpu_optimizer_offload():
    # Reflow only extends the stage-3 CPU-offload path; anything else would silently train without it.
    with pytest.raises(ValueError, match="Reflow requires ZeRO stage 3"):
        DeepSpeedConfig(minimal_reflow_config(stage=2))

    with pytest.raises(ValueError, match="Reflow requires optimizer offload"):
        DeepSpeedConfig(minimal_reflow_config(offload_optimizer=None))


def test_rejects_deepcompile():
    config = minimal_reflow_config()
    config["compile"] = {"deepcompile": True}
    with pytest.raises(ValueError, match="not supported with DeepCompile"):
        DeepSpeedConfig(config)


def test_rejects_muon_config():
    # Reflow's CPU step would replace Muon's orthogonalized update with Adam, so the combination must fail loudly.
    config = minimal_reflow_config()
    config["optimizer"] = {"type": "Muon", "params": {"lr": 1e-3}}
    with pytest.raises(ValueError, match="does not support the Muon optimizer"):
        DeepSpeedConfig(config)


# The oracle runs are bf16 ZeRO-3 on a GPU accelerator; Reflow is not validated on the CPU accelerator.
# Skipping here, rather than inside the test, keeps a CPU-only run from starting the distributed workers.
needs_gpu_accelerator = pytest.mark.skipif(get_accelerator().device_name() == "cpu",
                                           reason="Reflow training tests need a GPU accelerator.")


def skip_if_reflow_training_unsupported():
    if not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
        pytest.skip("CPUAdamBuilder is not compatible on this system.")
    if not get_accelerator().is_bf16_supported():
        pytest.skip("bf16 is not supported on this accelerator.")


def make_batches(hidden_dim, count, dtype=torch.bfloat16):
    device = get_accelerator().current_device_name()
    batches = []
    for _ in range(count):
        inputs = torch.randn(2, hidden_dim, device=device, dtype=dtype)
        labels = torch.randint(0, hidden_dim, (2, ), device=device)
        batches.append((inputs, labels))
    return batches


def get_offload_clip_config(reflow, gradient_clipping, sub_group_size=None):
    offload_optimizer = {"device": "cpu", "pin_memory": True}
    config = {
        "train_micro_batch_size_per_gpu": 2,
        "bf16": {
            "enabled": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 0.1
            }
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": offload_optimizer,
        },
        "gradient_clipping": gradient_clipping,
    }
    if sub_group_size is not None:
        config["zero_optimization"]["sub_group_size"] = sub_group_size
    if reflow:
        config["zero_optimization"]["reflow"] = {}
    return config


def train_with_offload(model, config, batches):
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    losses = []
    grad_norms = []
    for inputs, labels in batches:
        loss = engine(inputs, labels)
        # Shrink the grads so the 1e-6 term in the clip factor (norm + 1e-6) / clip is not negligible and
        # per-element grads sit near Adam's eps, where the update size depends on the grad scale.
        engine.backward(loss * 1e-7)
        engine.step()
        losses.append(loss.detach().clone())
        grad_norms.append(float(engine.get_global_grad_norm()))
    engine.destroy()
    return losses, grad_norms


def assert_same_training(expected, actual):
    expected_losses, expected_norms = expected
    actual_losses, actual_norms = actual
    assert actual_norms == expected_norms
    for step, (want, got) in enumerate(zip(expected_losses, actual_losses)):
        assert torch.equal(got, want), f"loss differs at step {step}: {got} vs {want}"


@pytest.mark.parametrize("clip_ratio", [0.5, 1.0, 100.0])
@needs_gpu_accelerator
class TestReflowGradClipMatchesZeroOffload(DistributedTest):
    world_size = [1, 2]

    def test_losses_and_grad_norms_match(self, clip_ratio):
        # A clip threshold equal to the grad norm is the boundary case: ZeRO-Offload still clips there
        # because it clips whenever (norm + 1e-6) / clip > 1.
        skip_if_reflow_training_unsupported()

        hidden_dim = 16
        torch.manual_seed(1234)
        initial_model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        batches = make_batches(hidden_dim, count=4)

        # The first step's grad norm does not depend on the clip threshold, so measure it unclipped.
        _, unclipped_norms = train_with_offload(copy.deepcopy(initial_model),
                                                get_offload_clip_config(reflow=False, gradient_clipping=0.0),
                                                batches[:1])
        gradient_clipping = unclipped_norms[0] * clip_ratio

        expected = train_with_offload(copy.deepcopy(initial_model),
                                      get_offload_clip_config(reflow=False, gradient_clipping=gradient_clipping),
                                      batches)
        actual = train_with_offload(copy.deepcopy(initial_model),
                                    get_offload_clip_config(reflow=True, gradient_clipping=gradient_clipping), batches)
        assert_same_training(expected, actual)


@needs_gpu_accelerator
class TestReflowGradNormWithPerParamPartitionGroups(DistributedTest):
    world_size = [1, 2]

    def test_losses_and_grad_norms_match(self):
        # Some params partition over a process group of their own (as AutoEP expert params do), so the grad norm
        # must be reduced per subgroup group. The groups have the same members here, which keeps plain ZeRO-Offload a
        # valid oracle while still exercising the per-group reduction.
        skip_if_reflow_training_unsupported()

        hidden_dim = 16
        torch.manual_seed(1234)
        initial_model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        batches = make_batches(hidden_dim, count=4)

        def model_with_own_partition_group():
            model = copy.deepcopy(initial_model)
            own_group = dist.new_group(ranks=list(range(dist.get_world_size())))
            for param in model.linears[0].parameters():
                param.ds_zero_partition_process_group = own_group
            return model

        # One param per subgroup, so no subgroup mixes partition groups.
        _, unclipped_norms = train_with_offload(
            model_with_own_partition_group(),
            get_offload_clip_config(reflow=False, gradient_clipping=0.0, sub_group_size=1), batches[:1])
        gradient_clipping = unclipped_norms[0] * 0.5

        expected = train_with_offload(
            model_with_own_partition_group(),
            get_offload_clip_config(reflow=False, gradient_clipping=gradient_clipping, sub_group_size=1), batches)
        actual = train_with_offload(
            model_with_own_partition_group(),
            get_offload_clip_config(reflow=True, gradient_clipping=gradient_clipping, sub_group_size=1), batches)
        assert_same_training(expected, actual)


def get_variant_config(reflow, variant):
    offload_optimizer = {"device": "cpu", "pin_memory": True}
    config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_clipping": 0.0,
        "bf16": {
            "enabled": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-3
            }
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": offload_optimizer,
        },
    }
    if variant == "lion":
        config["optimizer"] = {"type": "Lion", "params": {"lr": 1e-4}}
    elif variant == "offload_param":
        config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}
    elif variant == "fp16_overflow":
        del config["bf16"]
        # With this tiny model a 2**16 loss scale overflows the first two of eight steps (2**24 overflows all eight).
        # Reflow must skip exactly the same steps as ZeRO-Offload and then train identically.
        config["fp16"] = {"enabled": True, "initial_scale_power": 16, "hysteresis": 1}
    elif variant == "client_optimizer":
        # The client passes a DeepSpeedCPUAdam instead, which Reflow remaps to ReflowCPUAdam.
        del config["optimizer"]
    if reflow:
        config["zero_optimization"]["reflow"] = {}
    return config


def train_and_collect(model, config, batches, client_optimizer=None):
    engine, _, _, _ = deepspeed.initialize(model=model,
                                           model_parameters=model.parameters(),
                                           optimizer=client_optimizer,
                                           config=config)
    losses = []
    for inputs, labels in batches:
        loss = engine(inputs, labels)
        engine.backward(loss)
        engine.step()
        losses.append(loss.detach().clone())
    result = {
        "losses": losses,
        "fp32_params": [safe_get_full_fp32_param(param).clone() for param in engine.module.parameters()],
        "skipped_steps": engine.skipped_steps,
        "cpu_optimizer": type(engine.optimizer.optimizer),
    }
    engine.destroy()
    return result


@pytest.mark.parametrize("variant", ["lion", "offload_param", "fp16_overflow", "client_optimizer"])
@needs_gpu_accelerator
class TestReflowVariantsMatchZeroOffload(DistributedTest):
    world_size = [1, 2]

    def test_matches_zero_offload(self, variant):
        skip_if_reflow_training_unsupported()
        if variant == "fp16_overflow" and not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported on this accelerator.")

        hidden_dim = 16
        dtype = torch.bfloat16
        num_steps = 4
        if variant == "fp16_overflow":
            dtype = torch.half
            num_steps = 8
        torch.manual_seed(1234)
        initial_model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        batches = make_batches(hidden_dim, count=num_steps, dtype=dtype)

        def run(reflow):
            model = copy.deepcopy(initial_model)
            client_optimizer = None
            if variant == "client_optimizer":
                client_optimizer = DeepSpeedCPUAdam(model.parameters(), lr=1e-3)
            return train_and_collect(model, get_variant_config(reflow, variant), batches, client_optimizer)

        expected = run(reflow=False)
        actual = run(reflow=True)

        # Without this check, a run that silently fell back to plain ZeRO-Offload would pass every comparison below.
        expected_cpu_optimizer = ReflowCPULion if variant == "lion" else ReflowCPUAdam
        assert issubclass(actual["cpu_optimizer"], expected_cpu_optimizer)
        assert actual["skipped_steps"] == expected["skipped_steps"]
        if variant == "fp16_overflow":
            assert 0 < actual["skipped_steps"] < num_steps, "the loss scale should overflow some steps, not all"
        for step, (want, got) in enumerate(zip(expected["losses"], actual["losses"])):
            assert torch.equal(got, want), f"loss differs at step {step}: {got} vs {want}"
        for index, (want, got) in enumerate(zip(expected["fp32_params"], actual["fp32_params"])):
            assert torch.equal(got, want), f"fp32 param {index} differs"


@needs_gpu_accelerator
class TestReflowRejectsClientMuon(DistributedTest):
    world_size = 1

    def test_raises_instead_of_training_with_adam(self):
        # MuonWithAuxAdam's class name contains "Adam", so it used to be rebuilt as ReflowCPUAdam and silently train
        # every parameter with Adam.
        skip_if_reflow_training_unsupported()
        model = SimpleModel(hidden_dim=16, nlayers=2)
        muon_params = [param for param in model.parameters() if param.ndim >= 2]
        adam_params = [param for param in model.parameters() if param.ndim < 2]
        for param in muon_params:
            param.use_muon = True
        for param in adam_params:
            param.use_muon = False
        client_optimizer = MuonWithAuxAdam([
            dict(params=muon_params, use_muon=True, lr=1e-3),
            dict(params=adam_params, use_muon=False, lr=1e-3),
        ])
        config = get_variant_config(reflow=True, variant="client_optimizer")
        # Without this, ZeRO-Offload rejects every non-DeepSpeed CPU optimizer before Reflow sees it.
        config["zero_force_ds_cpu_optimizer"] = False
        with pytest.raises(ZeRORuntimeException, match="does not support the Muon optimizer"):
            deepspeed.initialize(model=model, optimizer=client_optimizer, config=config)


def get_state_read_config(reflow, gradient_accumulation_steps, warmup_lr):
    offload_optimizer = {"device": "cpu", "pin_memory": True}
    config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 0.0,
        "bf16": {
            "enabled": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-3
            }
        },
        "zero_optimization": {
            "stage": 3,
            # Split every weight into its own subgroup, so the chained async commits keep running after step()
            # returns and overlap the next micro-steps and the LR scheduler step.
            "sub_group_size": 2_000_000,
            "offload_optimizer": offload_optimizer,
        },
    }
    if warmup_lr:
        # The LR changes after every optimizer step, so a commit that reads it late would use the next step's LR.
        config["scheduler"] = {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0.0,
                "warmup_max_lr": 1e-3,
                "warmup_num_steps": 10
            }
        }
    if reflow:
        config["zero_optimization"]["reflow"] = {}
    return config


def read_state_right_after_each_step(model, config, batches):
    # Read as soon as an optimizer step returns, which is when a training script would save a checkpoint.
    engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    snapshots = []
    for inputs, labels in batches:
        loss = engine(inputs, labels)
        engine.backward(loss)
        at_boundary = engine.is_gradient_accumulation_boundary()
        engine.step()
        if not at_boundary:
            continue
        flat_groups = [group.clone() for group in optimizer.state_dict()[FP32_FLAT_GROUPS]]
        fp32_params = [safe_get_full_fp32_param(lp).clone() for lp in engine.module.parameters()]
        exp_avgs = [safe_get_full_optimizer_state(lp, "exp_avg").clone() for lp in engine.module.parameters()]
        snapshots.append((flat_groups, fp32_params, exp_avgs))
    engine.destroy()
    return snapshots


@pytest.mark.parametrize("gradient_accumulation_steps, warmup_lr", [(1, False), (3, False), (1, True)])
@needs_gpu_accelerator
class TestReflowStateReadAfterStep(DistributedTest):
    world_size = [1, 2]

    def test_matches_zero_offload(self, gradient_accumulation_steps, warmup_lr):
        # Reflow commits the FP32 master and optimizer state in background workers after step() returns, while the
        # next micro-steps run. Checkpoints and safe_get_* reads right after step() must see the committed values, and
        # the commit must not be disturbed by the next micro-steps' grads or by an LR scheduler step. The model is
        # large enough that the commit outlasts the read.
        skip_if_reflow_training_unsupported()

        hidden_dim = 2048
        torch.manual_seed(1234)
        batches = make_batches(hidden_dim, count=2 * gradient_accumulation_steps)

        expected = read_state_right_after_each_step(
            SimpleModel(hidden_dim=hidden_dim, nlayers=4),
            get_state_read_config(reflow=False,
                                  gradient_accumulation_steps=gradient_accumulation_steps,
                                  warmup_lr=warmup_lr), batches)
        torch.manual_seed(1234)
        actual = read_state_right_after_each_step(
            SimpleModel(hidden_dim=hidden_dim, nlayers=4),
            get_state_read_config(reflow=True,
                                  gradient_accumulation_steps=gradient_accumulation_steps,
                                  warmup_lr=warmup_lr), batches)

        for step, (expected_step, actual_step) in enumerate(zip(expected, actual)):
            for name, expected_tensors, actual_tensors in zip(("state_dict fp32 groups", "fp32 params", "exp_avg"),
                                                              expected_step, actual_step):
                for index, (want, got) in enumerate(zip(expected_tensors, actual_tensors)):
                    assert torch.equal(got, want), f"{name}[{index}] differs after step {step}"
