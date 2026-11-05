# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""A ZeRO-3 run with NVMe optimizer offload continues from a checkpoint exactly as the saved run did."""

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import AsyncIOBuilder, CPUAdamBuilder
from deepspeed.runtime.zero.partition_parameters import Init
from unit.common import DistributedTest

# With one sub-group per param, the weights of `a` and `b` are large enough to be swapped to NVMe; the biases and `c`
# stay in memory.
HIDDEN = 1024
SAVE_STEP, STEPS = 3, 6


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(HIDDEN, HIDDEN)
        self.b = torch.nn.Linear(HIDDEN, HIDDEN)
        self.c = torch.nn.Linear(HIDDEN, 1)

    def forward(self, x):
        return self.c(torch.tanh(self.b(torch.tanh(self.a(x))))).float().pow(2).mean()


def _config(nvme_path, precision, pipeline=False, offload_param=False):
    config = {
        "train_micro_batch_size_per_gpu": 2,
        "steps_per_print": 1000,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-2
            }
        },
        # The lr changes in every step.
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-2,
                "warmup_num_steps": 10,
                "warmup_type": "linear"
            }
        },
        "zero_optimization": {
            "stage": 3,
            "sub_group_size": 1,
            "offload_optimizer": {
                "device": "nvme",
                "nvme_path": nvme_path,
                "pipeline_read": pipeline,
                "pipeline_write": pipeline,
                # 4 states per sub-group (param, gradient, exp_avg, exp_avg_sq) for the previous, current and next
                # sub-group when pipelined.
                "buffer_count": 12 if pipeline else 4
            },
        },
    }
    if offload_param:
        config["zero_optimization"]["offload_param"] = {"device": "nvme", "nvme_path": nvme_path}
    if precision == "fp16":
        # The loss scale doubles after every 2 steps, so the saved scale is not the initial one.
        config["fp16"] = {"enabled": True, "initial_scale_power": 8, "loss_scale_window": 2}
    else:
        config["bf16"] = {"enabled": True}
    return config


def _engine(config, seed):
    # Swap files are named after ds_id, which a new process counts from 0 again.
    Init.param_id = 0
    torch.manual_seed(seed)
    model = Model()
    weights = [p for name, p in model.named_parameters() if name.endswith("weight")]
    biases = [p for name, p in model.named_parameters() if name.endswith("bias")]
    param_groups = [{"params": weights, "weight_decay": 0.1}, {"params": biases, "weight_decay": 0.0}]
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=param_groups, config=config)
    swapped = [engine.optimizer._swappable_optimizer_subgroup(i) for i in range(len(engine.optimizer.fp16_groups))]
    assert any(swapped) and not all(swapped), swapped
    return engine


def _weights(engine):
    params = list(engine.module.parameters())
    with deepspeed.zero.GatheredParameters(params):
        return torch.cat([p.detach().flatten().float().cpu() for p in params])


def _state(engine):
    return {"weights": _weights(engine), "lr": engine.get_lr(), "loss_scale": engine.optimizer.loss_scale}


def _batches(engine):
    dtype = torch.float16 if engine.fp16_enabled() else torch.bfloat16
    generator = torch.Generator().manual_seed(1 + dist.get_rank())
    return [torch.randn(2, HIDDEN, generator=generator).to(engine.device, dtype) for _ in range(STEPS)]


def _train(engine, batches):
    weights = []
    for x in batches:
        engine.backward(engine(x))
        engine.step()
        weights.append(_weights(engine))
    return weights


def _assert_same_state(actual, expected):
    assert torch.equal(actual["weights"], expected["weights"]), \
        f"weights differ by up to {(actual['weights'] - expected['weights']).abs().max().item()}"
    assert actual["lr"] == expected["lr"]
    assert actual["loss_scale"] == expected["loss_scale"]


def _skip_if_unsupported(precision):
    if not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
        pytest.skip("cpu-adam is not compatible")
    if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
        pytest.skip("async-io is not compatible")
    if precision == "fp16" and not get_accelerator().is_fp16_supported():
        pytest.skip("fp16 is not supported")
    if precision == "bf16" and not get_accelerator().is_bf16_supported():
        pytest.skip("bf16 is not supported")


class TestZero3NVMeResume(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("offload_param", [False, True])
    @pytest.mark.parametrize("pipeline", [False, True])
    @pytest.mark.parametrize("precision", ["fp16", "bf16"])
    def test(self, tmpdir, precision, pipeline, offload_param):
        """The resumed run has the saved weights, lr, loss scale and Adam state, so it continues bit for bit."""
        _skip_if_unsupported(precision)
        self._check_resume(tmpdir, _config(str(tmpdir.join("nvme")), precision, pipeline, offload_param))

    def test_params_on_nvme(self, tmpdir):
        """With max_in_cpu=0, the swapped parameters have no flat buffer in memory, so loading writes their weights
        to the parameter swap files."""
        _skip_if_unsupported("bf16")
        config = _config(str(tmpdir.join("nvme")), "bf16", offload_param=True)
        config["zero_optimization"]["offload_param"]["max_in_cpu"] = 0
        # The NVMe parameter prefetch can swap in a parameter twice (#4072).
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0
        self._check_resume(tmpdir, config)

    def test_pipeline_loading_checkpoint(self, tmpdir):
        """The ranks of a node load one after another, so loading must not wait for the other ranks, e.g. to gather
        the persistent parameters (the biases)."""
        _skip_if_unsupported("bf16")
        config = _config(str(tmpdir.join("nvme")), "bf16")
        config["zero_optimization"]["pipeline_loading_checkpoint"] = True
        self._check_resume(tmpdir, config)

    def _check_resume(self, tmpdir, config):
        ckpt_dir = str(tmpdir.join("ckpt"))

        engine = _engine(config, seed=0)
        batches = _batches(engine)
        _train(engine, batches[:SAVE_STEP])
        engine.save_checkpoint(ckpt_dir)
        saved = _state(engine)
        expected = _train(engine, batches[SAVE_STEP:])
        engine.destroy()

        engine = _engine(config, seed=1)
        engine.load_checkpoint(ckpt_dir)
        _assert_same_state(_state(engine), saved)
        actual = _train(engine, batches[SAVE_STEP:])
        engine.destroy()
        for step, (a, e) in enumerate(zip(actual, expected), start=SAVE_STEP):
            assert torch.equal(a, e), f"step {step}: weights differ by up to {(a - e).abs().max().item()}"

    def test_without_optimizer_states(self, tmpdir):
        """Without optimizer states, the resumed run still starts from the saved weights."""
        _skip_if_unsupported("bf16")
        config = _config(str(tmpdir.join("nvme")), "bf16")
        ckpt_dir = str(tmpdir.join("ckpt"))

        engine = _engine(config, seed=0)
        _train(engine, _batches(engine)[:SAVE_STEP])
        engine.save_checkpoint(ckpt_dir)
        saved = _weights(engine)
        engine.destroy()

        engine = _engine(config, seed=1)
        engine.load_checkpoint(ckpt_dir, load_optimizer_states=False)
        loaded = _weights(engine)
        assert torch.equal(loaded, saved), f"weights differ by up to {(loaded - saved).abs().max().item()}"
        _train(engine, _batches(engine)[:1])
        engine.destroy()
