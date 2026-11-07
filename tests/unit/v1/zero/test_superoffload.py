# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""SuperOffload against ZeRO-3 CPU offload, which applies the same DeepSpeedCPUAdam update in the main process."""

import time

import pytest
import torch
import torch.utils.checkpoint

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
    """`a` is always used. `b` and `c` are used, skipped, multiplied by 0, which gives them a zero gradient, or used
    twice, which gives them two gradients in one step."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.hidden = hidden
        self.a = torch.nn.Linear(hidden, hidden)
        self.b = torch.nn.Linear(hidden, hidden)
        self.c = torch.nn.Linear(hidden, hidden)

    def forward(self, x, uses):
        h = torch.nn.functional.gelu(self.a(x))
        for name in ("b", "c"):
            use = uses.get(name, "use")
            if use == "use":
                h = h + getattr(self, name)(x)
            elif use == "zero":
                h = h + 0 * getattr(self, name)(x)
            elif use in ("twice", "twice_in_one_backward"):
                layer = getattr(self, name)
                if use == "twice":
                    # The backward of a reentrant checkpoint runs on its own, so the param's gradient hook fires twice.
                    again = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=True)
                else:
                    again = layer(h)
                h = h + layer(x) + again
        return h.float().pow(2).mean()


def _config(super_offload,
            clip=0.0,
            ga=1,
            precision="bf16",
            ratio=1.0,
            sub_group_size=1e9,
            reduce_bucket_size=None,
            **optimizer_params):
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
    if reduce_bucket_size is not None:
        config["zero_optimization"]["reduce_bucket_size"] = reduce_bucket_size
    if precision == "fp16":
        config["fp16"] = {"enabled": True, "initial_scale_power": 24}
    elif precision != "fp32":
        config["bf16"] = {
            "enabled": True,
            "bf16_master_weights_and_grads": precision in ("bf16_master", "bf16_states"),
            "bf16_optimizer_states": precision == "bf16_states",
        }
    return config


def _engine(config, two_groups=False, hidden=HIDDEN):
    torch.manual_seed(0)
    model = PartlyUsedModel(hidden)
    params = model.parameters()
    if two_groups:
        # Weights with weight decay and biases without, as most training scripts set them up.
        named = list(model.named_parameters())
        params = [
            dict(params=[p for n, p in named if n.endswith("weight")]),
            dict(params=[p for n, p in named if n.endswith("bias")], weight_decay=0.0)
        ]
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=params, config=config)
    return engine


def _masters(engine):
    return torch.cat([t.detach().float().flatten().cpu() for t in engine.optimizer.fp32_partitioned_groups_flat])


def _close(engine):
    if hasattr(engine.optimizer, "cpu_adam"):
        engine.optimizer.cpu_adam.close()


def _train(engine, steps, uses_for, first_step=0, before_step=None):
    """Train `steps` optimizer steps and return the master weights after each one.

    uses_for(step, micro_step) gives the `uses` of PartlyUsedModel.forward. before_step(engine, step), if given, runs
    before each step, as an LR or momentum schedule would.
    """
    dtype = torch.float32
    if engine.fp16_enabled():
        dtype = torch.float16
    elif engine.bfloat16_enabled():
        dtype = torch.bfloat16
    generator = torch.Generator().manual_seed(1 + first_step + 100 * dist.get_rank())
    masters = []
    for step in range(first_step, first_step + steps):
        if before_step is not None:
            before_step(engine, step)
        for micro_step in range(engine.gradient_accumulation_steps()):
            x = torch.randn(4, engine.module.hidden, generator=generator).to(engine.device, dtype)
            loss = engine(x, uses_for(step, micro_step))
            engine.backward(loss)
            engine.step()
        masters.append(_masters(engine))
    return masters


def _all_used(step, micro_step):
    return {}


def _b_unused_on_even_steps(unused):
    return lambda step, micro_step: {"b": "use" if step % 2 else unused}


def _b_used_twice(unused):
    # ZeRO-3 keeps only the last gradient when the hook of a param fires twice in a step, so the reference uses the
    # param twice in one backward, where autograd sums the two gradients.
    use = "twice_in_one_backward" if unused == "zero" else "twice"
    return lambda step, micro_step: {"b": use}


def _b_and_c_in_one_micro_step(unused):
    return lambda step, micro_step: {"b": unused, "c": "use"} if micro_step else {"b": "use", "c": unused}


# name: (config overrides, uses_for(unused), compare exactly)
CASES = {
    "all_used": (dict(), lambda unused: _all_used, True),
    "unused_param": (dict(), _b_unused_on_even_steps, True),
    # Clipping is on but never triggers, so the CPU steps still run during backward and must stay exact.
    "unused_param_clip": (dict(clip=100.0), _b_unused_on_even_steps, True),
    "grad_accumulation": (dict(ga=2), _b_and_c_in_one_micro_step, True),
    # The second gradient adds to the first in the gradient accumulation dtype, bf16 or fp32. ZeRO-3 takes a param
    # once per reduce bucket, so a small bucket keeps the two gradients apart.
    "param_used_twice": (dict(reduce_bucket_size=1000), _b_used_twice, True),
    "param_used_twice_fp32": (dict(reduce_bucket_size=1000, precision="fp32"), _b_used_twice, True),
    "adam": (dict(type="Adam", adam_w_mode=False), lambda unused: _all_used, True),
    "no_bias_correction": (dict(bias_correction=False), lambda unused: _all_used, True),
    "bf16_master_weights": (dict(precision="bf16_master"), lambda unused: _all_used, True),
    "bf16_optimizer_states": (dict(precision="bf16_states"), lambda unused: _all_used, True),
    "two_param_groups": (dict(two_groups=True, sub_group_size=4500), _b_unused_on_even_steps, True),
    # Clipping undoes the steps taken during backward and redoes them, which is exact up to rounding. In fp32 the
    # rounding stays small; bf16 weights would amplify it.
    "clipping": (dict(clip=0.05, precision="fp32"), lambda unused: _all_used, False),
    # A step cannot be undone with a beta of 0 (the rollback divides by the betas), so those CPU steps wait for
    # step() and stay exact.
    "clipping_beta1_zero": (dict(clip=0.05, precision="fp32", betas=[0.0, 0.999]), lambda unused: _all_used, True),
    "clipping_beta2_zero": (dict(clip=0.05, precision="fp32", betas=[0.9, 0.0]), lambda unused: _all_used, True),
    # Overflow steps are skipped; the steps taken during backward are undone, exact up to rounding.
    "fp16_overflow": (dict(precision="fp16", sub_group_size=4500), lambda unused: _all_used, False),
}


class TestSuperOffloadMatchesZeRO3(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("case", list(CASES))
    def test(self, case):
        overrides, uses_for, exact = CASES[case]
        overrides = dict(overrides)
        two_groups = overrides.pop("two_groups", False)
        # The reference gives an unused param an exactly zero gradient, which SuperOffload must give a param that
        # gets no gradient (as ZeRO-1/2 do).
        engine = _engine(_config(super_offload=False, **overrides), two_groups)
        expected = _train(engine, 8, uses_for("zero"))
        _close(engine)
        engine = _engine(_config(super_offload=True, **overrides), two_groups)
        actual = _train(engine, 8, uses_for("skip"))
        _close(engine)
        _assert_same(actual, expected, exact)


def _assert_same(actual, expected, exact, atol=1e-5):
    for step, (a, e) in enumerate(zip(actual, expected)):
        assert torch.isfinite(a).all(), f"step {step}"
        if exact:
            assert torch.equal(a, e), f"step {step}: max diff {(a - e).abs().max().item()}"
        else:
            torch.testing.assert_close(a, e, rtol=0, atol=atol, msg=f"step {step}")


def _change_hyperparams(engine, step):
    for group in engine.optimizer.param_groups:
        group["betas"] = (0.9 - 0.05 * step, 0.999 - 0.01 * step)
        group["eps"] = 1e-8 * 10**step
        group["weight_decay"] = 0.01 + 0.1 * step


class TestSuperOffloadFollowsZeRO3(DistributedTest):
    world_size = 1

    def test_hyperparameter_changes(self):
        """Momentum and weight decay schedules change betas and weight_decay of the param groups between steps."""
        results = []
        for super_offload in (False, True):
            engine = _engine(_config(super_offload=super_offload), two_groups=True)
            results.append(_train(engine, 6, _all_used, before_step=_change_hyperparams))
            _close(engine)
        _assert_same(results[1], results[0], exact=True)

    def test_large_grads(self):
        """The bf16 gradient of a 4090 x 4090 weight, 31.9 MiB, goes to the CPU in two pieces of at most 16 MiB. A CPU
        step that did not wait for the second piece would read part of the previous gradient. Only to pinned memory
        are the copies asynchronous."""
        results = []
        for super_offload in (False, True):
            config = _config(super_offload=super_offload)
            config["zero_optimization"]["offload_optimizer"]["pin_memory"] = True
            engine = _engine(config, hidden=4090)
            results.append(_train(engine, 3, _all_used))
            _close(engine)
        _assert_same(results[1], results[0], exact=True)

    def test_fp16_loss_scale_increase(self):
        """Dynamic loss scaling raises the scale in step(), after the CPU steps taken during backward used the old
        one; those steps are undone and redone, exact up to rounding, which fp16 weights amplify (1.4e-5 after 6
        steps; 3.9e-3 when the steps are kept)."""
        results = []
        for super_offload in (False, True):
            config = _config(super_offload=super_offload, precision="fp16")
            config["fp16"] = {"enabled": True, "initial_scale_power": 8, "loss_scale_window": 2, "hysteresis": 1}
            engine = _engine(config)
            results.append(_train(engine, 6, _all_used))
            assert engine.optimizer.loss_scale > 2**8
            _close(engine)
        _assert_same(results[1], results[0], exact=False, atol=1e-4)

    def test_cpu_steps_run_during_backward(self):
        engine = _engine(_config(super_offload=True))
        cpu_adam = engine.optimizer.cpu_adam
        calls = []

        def recording(name):
            update = getattr(cpu_adam, name)

            def record(state, weights, grad, *args):
                calls.append((name, grad.dtype))
                update(state, weights, grad, *args)

            return record

        cpu_adam.step, cpu_adam.rollback = recording("step"), recording("rollback")
        x = torch.randn(4, HIDDEN).to(engine.device, torch.bfloat16)
        engine.backward(engine(x, {}))
        cpu_sub_groups = list(engine.optimizer.subgroup_to_device.values()).count("cpu")
        # The sub-groups step once their gradients reach the CPU, without waiting for step(), on the bf16 gradients
        # as they went to the CPU.
        deadline = time.monotonic() + 30
        while len(calls) < cpu_sub_groups and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls == [("step", torch.bfloat16)] * cpu_sub_groups
        engine.step()
        assert calls == [("step", torch.bfloat16)] * cpu_sub_groups
        # CPU Adam steps the weights where they are, without moving them to shared memory.
        assert not any(t.is_shared() for t in engine.optimizer.fp32_partitioned_groups_flat)
        _close(engine)

    def test_cpu_buffers_pinned(self):
        config = _config(super_offload=True)
        config["zero_optimization"]["offload_optimizer"]["pin_memory"] = True
        engine = _engine(config)
        # The gradient copies of backward and the weight copies of step() stay asynchronous only from page-locked
        # memory; ZeRO-3 pins the gradients but not the fp32 weights.
        assert all(t.is_pinned() and t.grad.is_pinned() for t in engine.optimizer.fp32_partitioned_groups_flat)
        assert engine.optimizer._lp_staging_stream is not None
        _close(engine)


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

    def test_unused_param(self):
        engine = _engine(_config(super_offload=False))
        expected = _train(engine, 6, _b_unused_on_even_steps("zero"))
        _close(engine)
        engine = _engine(_config(super_offload=True))
        actual = _train(engine, 6, _b_unused_on_even_steps("skip"))
        _close(engine)
        for step, (a, e) in enumerate(zip(actual, expected)):
            assert torch.equal(a, e), f"step {step}: max diff {(a - e).abs().max().item()}"


def _resume_matches_uninterrupted(config, checkpoint_dir, two_groups):
    """Train 4 steps, save, train 4 more; a new engine loads the checkpoint and trains the same 4 steps."""
    engine = _engine(config, two_groups)
    _train(engine, 4, _all_used)
    engine.save_checkpoint(checkpoint_dir, tag="t")
    uninterrupted = _train(engine, 4, _all_used, first_step=4)
    _close(engine)
    engine = _engine(config, two_groups)
    engine.load_checkpoint(checkpoint_dir, tag="t")
    resumed = _train(engine, 4, _all_used, first_step=4)
    _close(engine)
    for step, (r, u) in enumerate(zip(resumed, uninterrupted)):
        assert torch.equal(r, u), f"step {step} after the resume: max diff {(r - u).abs().max().item()}"


class TestSuperOffloadCheckpoint(DistributedTest):
    world_size = 2

    # Betas of 0.5 make beta**t exact, so the resume must be bit-exact (CPU Adam recomputes beta**t on resume).
    @pytest.mark.parametrize("two_groups", [False, True])
    def test_resume(self, tmpdir, two_groups):
        config = _config(super_offload=True, sub_group_size=4500, betas=[0.5, 0.5])
        _resume_matches_uninterrupted(config, str(tmpdir), two_groups)


def test_cpu_adam_steps_and_rolls_back_on_bf16_grads():
    """A step on a bf16 gradient matches DeepSpeedCPUAdam on the same gradient in fp32, and the rollback undoes it."""
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    from deepspeed.runtime.superoffload.superoffload_utils import SuperOffloadCPUAdam

    hyperparams = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, bias_correction=True)
    torch.manual_seed(0)
    weights, grad = torch.randn(4099), torch.randn(4099).to(torch.bfloat16)
    expected = torch.nn.Parameter(weights.clone())
    expected.grad = grad.float()
    DeepSpeedCPUAdam([expected], **hyperparams).step()
    original = weights.clone()
    # All the cores go to CPU Adam, which leaves the affinity of this thread as it is.
    cpu_adam = SuperOffloadCPUAdam(hyperparams, adamw_mode=True, fp32_optimizer_states=True, cpuadam_cores_perc=1.0)
    try:
        state = {}
        cpu_adam.submit(cpu_adam.step, state, weights, grad, hyperparams).result()
        assert torch.equal(weights, expected.data)
        cpu_adam.submit(cpu_adam.rollback, state, weights, grad, hyperparams).result()
        torch.testing.assert_close(weights, original, rtol=0, atol=1e-6)
    finally:
        cpu_adam.close()
