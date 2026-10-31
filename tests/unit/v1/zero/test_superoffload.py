# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""SuperOffload against ZeRO-3 CPU offload, which applies the same DeepSpeedCPUAdam update in the main process."""

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import CPUAdamBuilder
from unit.common import DistributedTest

if not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
    pytest.skip("cpu-adam is not compatible", allow_module_level=True)
if get_accelerator().device_name() != "cuda":
    pytest.skip("SuperOffload supports only CUDA", allow_module_level=True)

HIDDEN = 64


class PartlyUsedModel(torch.nn.Module):
    """`a` is always used. `b` and `c` are used, skipped, or multiplied by 0, which gives them a zero gradient."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(HIDDEN, HIDDEN)
        self.b = torch.nn.Linear(HIDDEN, HIDDEN)
        self.c = torch.nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x, uses):
        h = torch.nn.functional.gelu(self.a(x))
        for name in ("b", "c"):
            use = uses.get(name, "use")
            if use == "use":
                h = h + getattr(self, name)(x)
            elif use == "zero":
                h = h + 0 * getattr(self, name)(x)
        return h.float().pow(2).mean()


def _config(super_offload, clip=0.0, ga=1, precision="bf16", ratio=1.0, sub_group_size=1e9, **optimizer_params):
    optimizer_params.setdefault("lr", 3e-3)
    optimizer_params.setdefault("weight_decay", 0.01)
    optimizer_type = optimizer_params.pop("type", "AdamW")
    config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": ga,
        "gradient_clipping": clip,
        "steps_per_print": 1000,
        "optimizer": {
            "type": optimizer_type,
            "params": optimizer_params
        },
        "zero_optimization": {
            "stage": 3,
            "sub_group_size": sub_group_size,
            "offload_optimizer": {
                "device": "cpu",
                "ratio": ratio,
                "super_offload": super_offload,
            },
        },
    }
    if precision == "fp16":
        config["fp16"] = {"enabled": True, "initial_scale_power": 24}
    elif precision != "fp32":
        config["bf16"] = {"enabled": True, "bf16_master_weights_and_grads": precision == "bf16_master"}
    return config


def _engine(config):
    torch.manual_seed(0)
    model = PartlyUsedModel()
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    return engine


def _masters(engine):
    return torch.cat([t.detach().float().flatten().cpu() for t in engine.optimizer.fp32_partitioned_groups_flat])


def _close(engine):
    if hasattr(engine.optimizer, "superoffload_cpu_optimizer"):
        engine.optimizer.superoffload_cpu_optimizer.close()


def _train(engine, steps, uses_for, first_step=0):
    """Train `steps` optimizer steps and return the master weights after each one.

    uses_for(step, micro_step) gives the `uses` of PartlyUsedModel.forward.
    """
    dtype = torch.float32
    if engine.fp16_enabled():
        dtype = torch.float16
    elif engine.bfloat16_enabled():
        dtype = torch.bfloat16
    generator = torch.Generator().manual_seed(1 + first_step + 100 * dist.get_rank())
    masters = []
    for step in range(first_step, first_step + steps):
        for micro_step in range(engine.gradient_accumulation_steps()):
            x = torch.randn(4, HIDDEN, generator=generator).to(engine.device, dtype)
            loss = engine(x, uses_for(step, micro_step))
            engine.backward(loss)
            engine.step()
        masters.append(_masters(engine))
    return masters


def _all_used(step, micro_step):
    return {}


def _b_unused_on_even_steps(unused):
    return lambda step, micro_step: {"b": "use" if step % 2 else unused}


def _b_and_c_in_one_micro_step(unused):
    return lambda step, micro_step: {"b": unused, "c": "use"} if micro_step else {"b": "use", "c": unused}


# name: (config overrides, uses_for(unused), compare exactly)
CASES = {
    "all_used": (dict(), lambda unused: _all_used, True),
    "unused_param": (dict(), _b_unused_on_even_steps, True),
    # Clipping is on but never triggers, so the CPU steps still run during backward and must stay exact.
    "unused_param_clip": (dict(clip=100.0), _b_unused_on_even_steps, True),
    "grad_accumulation": (dict(ga=2), _b_and_c_in_one_micro_step, True),
    "adam": (dict(type="Adam", adam_w_mode=False), lambda unused: _all_used, True),
    "no_bias_correction": (dict(bias_correction=False), lambda unused: _all_used, True),
    "bf16_master_weights": (dict(precision="bf16_master"), lambda unused: _all_used, True),
    # Clipping undoes the steps taken during backward and redoes them, which is exact up to rounding. In fp32 the
    # rounding stays small; bf16 weights would amplify it.
    "clipping": (dict(clip=0.05, precision="fp32"), lambda unused: _all_used, False),
    # Overflow steps are skipped; the steps taken during backward are undone, exact up to rounding.
    "fp16_overflow": (dict(precision="fp16", sub_group_size=4500), lambda unused: _all_used, False),
}


class TestSuperOffloadMatchesZeRO3(DistributedTest):
    world_size = 1
    # SuperOffload starts its CPU worker process, which a daemonic test process cannot do.
    non_daemonic_procs = True

    @pytest.mark.parametrize("case", list(CASES))
    def test(self, case):
        overrides, uses_for, exact = CASES[case]
        # The reference gives an unused param an exactly zero gradient, which SuperOffload must give a param that
        # gets no gradient (as ZeRO-1/2 do).
        engine = _engine(_config(super_offload=False, **overrides))
        expected = _train(engine, 8, uses_for("zero"))
        _close(engine)
        engine = _engine(_config(super_offload=True, **overrides))
        actual = _train(engine, 8, uses_for("skip"))
        _close(engine)

        for step, (a, e) in enumerate(zip(actual, expected)):
            assert torch.isfinite(a).all(), f"step {step}"
            if exact:
                assert torch.equal(a, e), f"step {step}: max diff {(a - e).abs().max().item()}"
            else:
                torch.testing.assert_close(a, e, rtol=0, atol=1e-5, msg=f"step {step}")


class InfInjectionModel(torch.nn.Module):
    """With sub_group_size 1000 and ratio 0.5, `lin` is a sub-group kept on the GPU and `emb` a CPU sub-group."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(32, 32)
        self.emb = torch.nn.Embedding(512, 32)

    def forward(self, tokens, inject):
        loss = self.lin(self.emb(tokens)).float().pow(2).mean()
        # inf * weight puts an inf gradient into `lin` only, so `emb` gets a finite gradient in the skipped step.
        return loss + self.lin.weight[0, 0].float() * inject


class TestSuperOffloadSkippedStep(DistributedTest):
    world_size = 1
    non_daemonic_procs = True

    def test_sub_group_with_finite_grads(self):
        """A step skipped for an inf gradient elsewhere undoes the CPU sub-group's step; its Adam state must stay
        valid. Each step looks up new embedding rows, which have zero Adam state before their first gradient."""
        masters = {}
        for super_offload in (False, True):
            config = _config(super_offload=super_offload, ratio=0.5, sub_group_size=1000, lr=1e-3, weight_decay=0.0)
            torch.manual_seed(0)
            model = InfInjectionModel()
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            masters[super_offload] = []
            for step in range(6):
                tokens = (torch.arange(step * 64, step * 64 + 64).view(4, 16) % 512).to(engine.device)
                loss = engine(tokens, float("inf") if step == 1 else 0.0)
                engine.backward(loss)
                engine.step()
                masters[super_offload].append(_masters(engine))
            assert engine.skipped_steps == 1
            _close(engine)
        for step, (a, e) in enumerate(zip(masters[True], masters[False])):
            assert torch.isfinite(a).all(), f"step {step}"
            # Undoing a step is exact up to rounding, which Adam amplifies for the rows whose state was zero.
            torch.testing.assert_close(a, e, rtol=0, atol=1e-4, msg=f"step {step}")


class TestSuperOffloadTwoRanks(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    def test_unused_param(self):
        engine = _engine(_config(super_offload=False))
        expected = _train(engine, 6, _b_unused_on_even_steps("zero"))
        _close(engine)
        engine = _engine(_config(super_offload=True))
        actual = _train(engine, 6, _b_unused_on_even_steps("skip"))
        _close(engine)
        for step, (a, e) in enumerate(zip(actual, expected)):
            assert torch.equal(a, e), f"step {step}: max diff {(a - e).abs().max().item()}"


def _resume_matches_uninterrupted(config, checkpoint_dir):
    """Train 4 steps, save, train 4 more; a new engine loads the checkpoint and trains the same 4 steps."""
    engine = _engine(config)
    _train(engine, 4, _all_used)
    engine.save_checkpoint(checkpoint_dir, tag="t")
    uninterrupted = _train(engine, 4, _all_used, first_step=4)
    _close(engine)
    engine = _engine(config)
    engine.load_checkpoint(checkpoint_dir, tag="t")
    resumed = _train(engine, 4, _all_used, first_step=4)
    _close(engine)
    for step, (r, u) in enumerate(zip(resumed, uninterrupted)):
        assert torch.equal(r, u), f"step {step} after the resume: max diff {(r - u).abs().max().item()}"


class TestSuperOffloadCheckpoint(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    # Betas of 0.5 make beta**t exact, so the resume must be bit-exact (CPU Adam recomputes beta**t on resume).
    @pytest.mark.parametrize("super_offload,ratio", [(True, 1.0), (True, 0.5), (False, 0.5)])
    def test_resume(self, tmpdir, super_offload, ratio):
        config = _config(super_offload=super_offload, ratio=ratio, sub_group_size=4500, betas=[0.5, 0.5])
        _resume_matches_uninterrupted(config, str(tmpdir))


class TestPartialOffloadLRSchedule(DistributedTest):
    world_size = 1
    non_daemonic_procs = True

    @pytest.mark.parametrize("super_offload", [True, False])
    def test_first_step_at_zero_lr(self, super_offload):
        # With ratio < 1 the sub-groups kept on the device have their own optimizer; it must follow the scheduler.
        config = _config(super_offload=super_offload, ratio=0.5, sub_group_size=4500)
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
        assert "cpu" in engine.optimizer.subgroup_to_device.values()
        assert set(engine.optimizer.subgroup_to_device.values()) != {"cpu"}
        before = _masters(engine)
        after = _train(engine, 1, _all_used)[0]
        _close(engine)
        assert torch.equal(after, before)
