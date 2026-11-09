# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""ZeRO-3 steps a param that gets no gradient in an accumulation window as if its gradient were zero, and with
optimizer offload passes on the gradients of micro-steps before the boundary micro-step."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import AsyncIOBuilder, CPUAdamBuilder
from deepspeed.utils import safe_get_local_fp32_param, safe_get_local_grad, safe_set_full_grad, safe_set_local_grad
from deepspeed.utils.torch import required_torch_version
from unit.common import DistributedTest

# Large enough for the NVMe swapper to swap the sub-group; c.bias has one element, so the second rank's partition of
# it is only padding.
HIDDEN = 1024


class PartlyUsedModel(torch.nn.Module):
    """`a` and `c` are always used. `b` is used, skipped, or multiplied by 0, which gives it a zero gradient."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(HIDDEN, HIDDEN)
        self.b = torch.nn.Linear(HIDDEN, HIDDEN)
        self.c = torch.nn.Linear(HIDDEN, 1)

    def forward(self, x, use_b):
        h = torch.tanh(self.a(x))
        if use_b == "use":
            h = h + torch.tanh(self.b(h))
        elif use_b == "zero":
            h = h + 0 * torch.tanh(self.b(h))
        return self.c(h).float().pow(2).mean()


class ScaleModel(torch.nn.Module):
    """`b` is used, skipped, or multiplied by 0. `scale` has one element, so when `b` is skipped, the second rank
    reduces no gradient into a partition of its own."""

    def __init__(self):
        super().__init__()
        self.b = torch.nn.Linear(HIDDEN, HIDDEN)
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, use_b):
        h = x
        if use_b == "use":
            h = h + torch.tanh(self.b(h))
        elif use_b == "zero":
            h = h + 0 * torch.tanh(self.b(h))
        return (self.scale * h).float().pow(2).mean()


# The use of `b` in each micro-step of each step: its last gradient of a window comes before the boundary micro-step,
# it gets none in a window, and its first gradient of a window comes in the boundary micro-step.
SCHEDULES = {
    1: [["use"], ["skip"], ["skip"], ["use"], ["skip"]],
    2: [["use", "use"], ["use", "skip"], ["skip", "skip"], ["skip", "use"], ["use", "use"]],
}


def _skip_if_unsupported(offload):
    if offload != "none" and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
        pytest.skip("cpu-adam is not compatible")
    if offload == "nvme" and not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
        pytest.skip("async-io is not compatible")


def _config(offload, gas, nvme_path=None, managed=True, precision="fp32"):
    config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": gas,
        "steps_per_print": 1000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-2
            }
        },
        "zero_optimization": {
            "stage": 3
        },
    }
    if not managed:
        config["managed_gradient_accumulation"] = False
    if precision == "bf16":
        config["bf16"] = {"enabled": True}
    if offload == "cpu":
        config["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
    elif offload == "nvme":
        config["zero_optimization"]["offload_optimizer"] = {"device": "nvme", "nvme_path": nvme_path}
        # Smaller than the model, so that the gradients go to NVMe in more than one piece at the boundary as well.
        config["zero_optimization"]["reduce_bucket_size"] = HIDDEN * HIDDEN
    return config


def _engine(config, model_class=PartlyUsedModel):
    torch.manual_seed(0)
    model = model_class()
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    return engine


def _weights(engine, params=None):
    params = list(engine.module.parameters()) if params is None else params
    with deepspeed.zero.GatheredParameters(params):
        return torch.cat([p.detach().flatten().float().cpu() for p in params])


def _train(engine, schedule, managed=True):
    """Train one step per entry of `schedule` and return the weights after each step."""
    dtype = torch.float32
    if engine.fp16_enabled():
        dtype = torch.float16
    elif engine.bfloat16_enabled():
        dtype = torch.bfloat16
    generator = torch.Generator().manual_seed(1 + dist.get_rank())
    weights = []
    for uses in schedule:
        for use_b in uses:
            x = torch.randn(2, HIDDEN, generator=generator).to(engine.device, dtype)
            engine.backward(engine(x, use_b))
            if managed:
                engine.step()
        if not managed:
            engine.step()
        weights.append(_weights(engine))
    return weights


class TestZero3UnusedParamGradient(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("precision", ["fp32", "bf16"])
    @pytest.mark.parametrize("gas", [1, 2])
    @pytest.mark.parametrize("offload", ["none", "cpu", "nvme"])
    def test(self, tmpdir, offload, gas, precision):
        self._compare(tmpdir, offload, gas, managed=True, precision=precision)

    @pytest.mark.parametrize("offload", ["none", "cpu", "nvme"])
    def test_unmanaged(self, tmpdir, offload):
        """With managed_gradient_accumulation false, step() ends the window."""
        self._compare(tmpdir, offload, gas=2, managed=False)

    @pytest.mark.parametrize("offload", ["none", "cpu", "nvme"])
    def test_padding_rank(self, tmpdir, offload):
        """When `b` is skipped, the second rank reduces no gradient of its own; its partition of `b` still gets a zero
        gradient."""
        self._compare(tmpdir, offload, gas=1, managed=True, model_class=ScaleModel)

    @pytest.mark.parametrize("offload", ["none", "cpu", "nvme"])
    def test_unmanaged_set_full_grad(self, tmpdir, offload):
        """With managed_gradient_accumulation false, a gradient set with safe_set_full_grad after the last backward is
        the one step() applies, whether the param got a gradient, a zero gradient, or none."""
        _skip_if_unsupported(offload)
        results = {}
        for use_b in ("use", "zero", "skip"):
            config = _config(offload, gas=1, nvme_path=str(tmpdir.join(use_b)), managed=False)
            # Clipping would scale b's gradient by the norm of all gradients, which differs between the runs.
            config["gradient_clipping"] = 0.0
            engine = _engine(config)
            value = torch.full((HIDDEN, HIDDEN), 1e-3, device=engine.device)
            generator = torch.Generator().manual_seed(1 + dist.get_rank())
            # Step 1 uses b in all runs; step 2 sets its gradient after the backward.
            for step, step_use_b in enumerate(("use", use_b)):
                x = torch.randn(2, HIDDEN, generator=generator).to(engine.device)
                engine.backward(engine(x, step_use_b))
                if step == 1:
                    safe_set_full_grad(engine.module.b.weight, value)
                engine.step()
            results[use_b] = (_weights(engine), _weights(engine, [engine.module.b.weight]))
            engine.destroy()
        assert torch.equal(results["skip"][0], results["zero"][0])
        # b.weight gets the same update in all three runs: the set gradient.
        for use_b in ("zero", "skip"):
            actual, expected = results[use_b][1], results["use"][1]
            assert torch.equal(actual, expected), f"{use_b}: max diff {(actual - expected).abs().max().item()}"

    @pytest.mark.parametrize("managed", [True, False])
    def test_set_full_grad_norm(self, managed):
        """The gradient norm of step(), which clipping uses, has a gradient set with safe_set_full_grad in it, with
        optimizer offload as without."""
        _skip_if_unsupported("cpu")
        norms = {}
        for offload in ("none", "cpu"):
            engine = _engine(_config(offload, gas=1, managed=managed))
            generator = torch.Generator().manual_seed(1 + dist.get_rank())
            x = torch.randn(2, HIDDEN, generator=generator).to(engine.device)
            engine.backward(engine(x, "use"))
            safe_set_full_grad(engine.module.b.weight, torch.full((HIDDEN, HIDDEN), 1e-3, device=engine.device))
            engine.step()
            norms[offload] = float(engine.get_global_grad_norm())
            engine.destroy()
        assert abs(norms["cpu"] - norms["none"]) <= 1e-5 * norms["none"], norms

    def _compare(self, tmpdir, offload, gas, managed, precision="fp32", model_class=PartlyUsedModel):
        _skip_if_unsupported(offload)
        if precision == "bf16" and not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported")
        results = {}
        for unused in ("zero", "skip"):
            engine = _engine(_config(offload, gas, str(tmpdir.join(unused)), managed, precision), model_class)
            schedule = [[unused if use_b == "skip" else use_b for use_b in uses] for uses in SCHEDULES[gas]]
            results[unused] = _train(engine, schedule, managed)
            engine.destroy()
        # Both runs do the same float operations: `0 *` gives an exactly zero gradient, and with two ranks each
        # reduced element is the sum of the same two values.
        for step, (actual, expected) in enumerate(zip(results["skip"], results["zero"])):
            assert torch.equal(actual, expected), f"step {step}: max diff {(actual - expected).abs().max().item()}"


class TestZero3UnusedParamOverflow(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("offload", ["none", "cpu"])
    def test(self, offload):
        """An inf gradient in a param that is then unused skips only its own step."""
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        _skip_if_unsupported(offload)
        config = _config(offload, gas=1)
        config["fp16"] = {"enabled": True, "initial_scale_power": 8, "loss_scale_window": 1000, "hysteresis": 1}
        engine = _engine(config)
        inject_inf = [True]
        engine.module.b.weight.register_hook(lambda grad: grad * float("inf") if inject_inf[0] else grad)
        _train(engine, [["use"]])
        inject_inf[0] = False
        _train(engine, [["skip"]] * 3)
        assert engine.skipped_steps == 1
        engine.destroy()


class SharedLayerModel(torch.nn.Module):
    """`shared` is used in two segments, which are checkpointed when `reentrant` is set."""

    def __init__(self, reentrant=False):
        super().__init__()
        self.reentrant = reentrant
        self.inp = torch.nn.Linear(HIDDEN, HIDDEN)
        self.shared = torch.nn.Linear(HIDDEN, HIDDEN)
        self.out = torch.nn.Linear(HIDDEN, 1)

    def segment(self, h):
        return torch.tanh(self.shared(h))

    def forward(self, x, use_b=None):
        h = torch.tanh(self.inp(x))
        for _ in range(2):
            h = checkpoint(self.segment, h, use_reentrant=True) if self.reentrant else self.segment(h)
        return self.out(h).pow(2).mean()


class TestZero3SharedParamReentrantCheckpoint(DistributedTest):
    world_size = 1

    def test(self):
        """Each checkpointed segment's backward reduces the gradient of `shared`; ZeRO-3 must add both."""
        config = _config("none", gas=1)
        # Smaller than a layer, so that the two gradients of `shared` are reduced in different buckets.
        config["zero_optimization"]["reduce_bucket_size"] = HIDDEN * HIDDEN
        results = []
        for reentrant in (False, True):
            engine = _engine(config, lambda: SharedLayerModel(reentrant))
            results.append(_train(engine, [["use"]] * 2))
            engine.destroy()
        for step, (actual, expected) in enumerate(zip(results[1], results[0])):
            assert torch.equal(actual, expected), f"step {step}: max diff {(actual - expected).abs().max().item()}"


class EvenModel(PartlyUsedModel):
    """No param has a partition of only padding: DeepCompile updates that padding, eager ZeRO-3 does not."""

    def __init__(self):
        super().__init__()
        self.c = torch.nn.Linear(HIDDEN, 2)


class TestZero3DeepCompileGradients(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.parametrize("set_grad", [False, True])
    def test(self, set_grad):
        """DeepCompile reduces into the gradient partitions without partition_grads; step() must apply them, also after
        a safe_set_local_grad call."""
        if not required_torch_version(min_version=2.6):
            pytest.skip("DeepCompile requires PyTorch >= v2.6")
        if get_accelerator().device_name() == "cpu":
            pytest.skip("DeepCompile does not support the CPU accelerator")
        results = []
        for deepcompile in (False, True):
            config = _config("none", gas=1)
            del config["optimizer"]
            config["zero_allow_untested_optimizer"] = True
            config["compile"] = {"deepcompile": deepcompile}
            torch.manual_seed(0)
            model = EvenModel()
            # SGD: an update proportional to the gradient, unlike Adam's first step, which turns rounding
            # differences of near-zero gradients into updates of the size of the lr.
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
            if deepcompile:
                engine.compile()
            results.append(self._updates(engine, set_grad))
            engine.destroy()
        for step, (actual, expected) in enumerate(zip(results[1], results[0])):
            # Compiled and eager kernels round differently, by far less than the update.
            atol = 1e-3 * expected.abs().max().item()
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=atol, msg=f"step {step}")

    def _updates(self, engine, set_grad):
        """The change of this rank's fp32 partitions after each of 3 steps. Gathering the params instead would
        interleave with DeepCompile's own gathers."""

        def masters():
            return torch.cat([safe_get_local_fp32_param(p).flatten().cpu() for p in engine.parameters()])

        initial = masters()
        generator = torch.Generator().manual_seed(1 + dist.get_rank())
        updates = []
        for _ in range(3):
            x = torch.randn(2, HIDDEN, generator=generator).to(engine.device)
            engine.backward(engine(x, "use"))
            if set_grad:
                # Writes back the same gradient.
                safe_set_local_grad(engine.module.a.weight, safe_get_local_grad(engine.module.a.weight))
            engine.step()
            updates.append(masters() - initial)
        return updates
