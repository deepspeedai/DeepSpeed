# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""ZeRO-3 with offload_optimizer.ratio < 1, where the sub-groups kept on the device are stepped by a torch Adam."""

import glob
import os

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import CPUAdamBuilder
from deepspeed.utils import safe_get_full_optimizer_state
from unit.common import DistributedTest

if not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
    pytest.skip("cpu-adam is not compatible", allow_module_level=True)

HIDDEN = 64


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(HIDDEN, HIDDEN) for _ in range(3))

    def forward(self, x):
        for layer in self.layers:
            x = torch.nn.functional.gelu(layer(x))
        return x.float().pow(2).mean()


def _config(ratio, super_offload=False, **optimizer_params):
    optimizer_params.setdefault("lr", 3e-3)
    optimizer_params.setdefault("weight_decay", 0.01)
    # 3 sub-groups: the first two weights, the last weight, the biases. Ratio 0.5 keeps the first on the device.
    return {
        "train_micro_batch_size_per_gpu": 4,
        "steps_per_print": 1000,
        "optimizer": {
            "type": "AdamW",
            "params": optimizer_params
        },
        "zero_optimization": {
            "stage": 3,
            "sub_group_size": 4500,
            "offload_optimizer": {
                "device": "cpu",
                "ratio": ratio,
                "super_offload": super_offload
            },
        },
        "bf16": {
            "enabled": True
        },
    }


def _engine(config):
    torch.manual_seed(0)
    model = Net()
    named = list(model.named_parameters())
    # Weights with weight decay and biases without, as most training scripts set them up.
    params = [
        dict(params=[p for n, p in named if n.endswith("weight")]),
        dict(params=[p for n, p in named if n.endswith("bias")], weight_decay=0.0)
    ]
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=params, config=config)
    return engine


def _train(engine, steps, first_step=0):
    generator = torch.Generator().manual_seed(1 + first_step + 100 * dist.get_rank())
    dtype = torch.bfloat16 if engine.bfloat16_enabled() else torch.float32
    masters = []
    for _ in range(steps):
        x = torch.randn(4, HIDDEN, generator=generator).to(engine.device, dtype)
        loss = engine(x)
        engine.backward(loss)
        engine.step()
        masters.append(
            torch.cat([t.detach().float().flatten().cpu() for t in engine.optimizer.fp32_partitioned_groups_flat]))
    return masters


def _close(engine):
    if hasattr(engine.optimizer, "cpu_adam"):
        engine.optimizer.cpu_adam.close()


def _device_params(engine):
    optimizer = engine.optimizer
    return [optimizer.fp32_partitioned_groups_flat[i] for i, d in optimizer.subgroup_to_device.items() if d != "cpu"]


def _adam_state(engine):
    return {
        (name, key): safe_get_full_optimizer_state(p, key).cpu()
        for name, p in engine.module.named_parameters()
        for key in ("exp_avg", "exp_avg_sq")
    }


class TestPartialOffloadHyperparameters(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("super_offload", [False, True])
    def test_first_step_at_zero_lr(self, super_offload):
        if super_offload and get_accelerator().device_name() != "cuda":
            pytest.skip("SuperOffload supports only CUDA")
        config = _config(ratio=0.5, super_offload=super_offload)
        config["scheduler"] = {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0.0,
                "warmup_max_lr": 3e-3,
                "warmup_num_steps": 4,
                "warmup_type": "linear"
            }
        }
        engine = _engine(config)
        assert _device_params(engine) and "cpu" in engine.optimizer.subgroup_to_device.values()
        before = torch.cat([t.detach().float().flatten().cpu() for t in engine.optimizer.fp32_partitioned_groups_flat])
        after = _train(engine, 1)[0]
        assert torch.equal(after, before)
        # A momentum or weight decay schedule changes the other hyperparameters of the param groups.
        for group in engine.optimizer.param_groups:
            group["betas"], group["eps"], group["weight_decay"] = (0.8, 0.99), 1e-6, 0.2
        _train(engine, 1)
        optimizer = engine.optimizer
        # (The scheduler has already set the lr of the next step.)
        for i in {optimizer.sub_group_to_group_id[i] for i, d in optimizer.subgroup_to_device.items() if d != "cpu"}:
            for key in ("betas", "eps", "weight_decay"):
                assert optimizer.backup_optimizer.param_groups[i][key] == optimizer.param_groups[i][key], key
        _close(engine)


class TestPartialOffloadCheckpoint(DistributedTest):
    world_size = 2

    def test_resume(self, tmpdir):
        """Betas of 0.5 make beta**t exact, so the resume is bit-exact (CPU Adam recomputes beta**t on resume)."""
        config = _config(ratio=0.5, betas=[0.5, 0.5])
        engine = _engine(config)
        _train(engine, 4)
        engine.save_checkpoint(str(tmpdir), tag="t")
        uninterrupted = _train(engine, 4, first_step=4)
        engine = _engine(config)
        engine.load_checkpoint(str(tmpdir), tag="t")
        resumed = _train(engine, 4, first_step=4)
        for step, (r, u) in enumerate(zip(resumed, uninterrupted)):
            assert torch.equal(r, u), f"step {step} after the resume: max diff {(r - u).abs().max().item()}"


class TestPartialOffloadCheckpointPortability(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("load_ratio", [0.25, 0.75, 1.0])
    def test_load_with_another_ratio(self, tmpdir, load_ratio):
        engine = _engine(_config(ratio=0.5))
        _train(engine, 3)
        saved = _adam_state(engine)
        engine.save_checkpoint(str(tmpdir), tag="t")
        engine = _engine(_config(ratio=load_ratio))
        engine.load_checkpoint(str(tmpdir), tag="t")
        loaded = _adam_state(engine)
        for key, value in saved.items():
            assert torch.equal(loaded[key], value), key
        _train(engine, 1)
        engine.save_checkpoint(str(tmpdir), tag="again")

    def test_resume_with_bf16_master_weights(self, tmpdir):
        """torch Adam keeps the moments of the device sub-groups in the param dtype, also after a load that returns
        fp32 moments for bf16 params (DeepSpeedCPUAdam with fp32_optimizer_states)."""
        config = _config(ratio=0.5)
        config["bf16"]["bf16_master_weights_and_grads"] = True
        engine = _engine(config)
        _train(engine, 2)
        engine.save_checkpoint(str(tmpdir), tag="t")
        engine = _engine(config)
        engine.load_checkpoint(str(tmpdir), tag="t")
        assert torch.isfinite(_train(engine, 1)[0]).all()
        for fp32_param in _device_params(engine):
            state = engine.optimizer.backup_optimizer.state[fp32_param]
            assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == fp32_param.dtype

    def test_load_checkpoint_without_device_state(self, tmpdir):
        """Checkpoints saved before the state of the sub-groups kept on the device was saved have none for them."""
        engine = _engine(_config(ratio=0.5))
        _train(engine, 3)
        engine.save_checkpoint(str(tmpdir), tag="t")
        device_ids = [i for i, d in engine.optimizer.subgroup_to_device.items() if d != "cpu"]
        path, = glob.glob(os.path.join(str(tmpdir), "t", "*zero_pp_rank_0_mp_rank_00_optim_states.pt"))
        checkpoint = torch.load(path, weights_only=False)
        for i in device_ids:
            checkpoint["optimizer_state_dict"]["optimizer_state_dict"]["state"].pop(i, None)
        torch.save(checkpoint, path)

        engine = _engine(_config(ratio=0.5))
        engine.load_checkpoint(str(tmpdir), tag="t")
        _train(engine, 1)
        # Only the sub-groups kept on the device are stepped by the backup optimizer.
        assert set(engine.optimizer.backup_optimizer.state) == set(_device_params(engine))
        engine.save_checkpoint(str(tmpdir), tag="again")


class TestPartialOffloadAdamSettings(DistributedTest):
    world_size = 1

    def test_l2_weight_decay(self):
        """With adam_w_mode off, the sub-groups kept on the device apply L2 weight decay, as DeepSpeedCPUAdam does."""
        results = []
        for ratio in (1.0, 0.5):
            config = _config(ratio=ratio, weight_decay=0.1)
            config["optimizer"] = {"type": "Adam", "params": {**config["optimizer"]["params"], "adam_w_mode": False}}
            del config["bf16"]
            engine = _engine(config)
            results.append(_train(engine, 4))
        for step, (partial, cpu) in enumerate(zip(results[1], results[0])):
            # torch Adam and DeepSpeedCPUAdam round differently.
            torch.testing.assert_close(partial, cpu, rtol=0, atol=1e-6, msg=f"step {step}")

    def test_bias_correction_off_is_rejected(self):
        config = _config(ratio=0.5)
        config["optimizer"]["params"]["bias_correction"] = False
        with pytest.raises(ValueError, match="bias_correction"):
            _engine(config)
