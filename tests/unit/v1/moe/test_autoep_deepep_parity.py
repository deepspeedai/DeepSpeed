# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Numerical parity between the DeepEP transport and the collective one.

Both backends move the same tokens to the same experts and reduce the same
weighted sum, so a step run through either must produce the same activations
and the same gradients. Only the route differs.

This is the test that catches a transport which silently drops something. The
routing weights were once handed to DeepEP's combine, which transports and
reduces them but does not multiply the rows by them, so expert outputs came
back summed but unweighted. Nothing raised, the loss still fell, and every
mock-level test still passed; only comparing the two paths' numbers exposes it.

Requires GPUs and a DeepEP build, so it is opt-in.
"""

import pytest
import torch

import deepspeed

from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import (
    MockMoETransformer,
    engine_input_dtype,
    make_autoep_config,
    seed_everything,
    skip_unless_h100_tests_enabled,
)

# DeepEP combine vectorizes one 16-byte element per warp lane.
HIDDEN_SIZE = 256
SEQ_LEN = 8


def _deepep_available() -> bool:
    try:
        import deep_ep  # noqa: F401
    except Exception:
        return False
    return True


def _run_one_step(backend, ep_size, seed):
    """Build a model on ``backend``, run one step, return its output and grads."""
    seed_everything(seed)

    config = make_autoep_config(ep_size=ep_size)
    # Pinned, because make_autoep_config prefers fp16 wherever it is available
    # and DeepEP dispatches bfloat16 only. Both backends have to run the same
    # dtype anyway for the comparison to mean anything.
    config.pop("fp16", None)
    config["bf16"] = {"enabled": True}
    config["expert_parallel"]["comm_backend"] = backend
    if backend == "deepep":
        # Sized explicitly rather than from the first batch, so both backends
        # see identical shapes whatever that batch turns out to be.
        config["expert_parallel"]["comm_max_tokens_per_rank"] = 512

    model = MockMoETransformer(hidden_size=HIDDEN_SIZE)
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)

    # Reseeded so the input is identical on every rank and across backends: the
    # comparison is of the transport, so nothing else may differ.
    seed_everything(seed)
    hidden = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, device=engine.device, dtype=engine_input_dtype(engine))

    output = engine(hidden)
    loss = output.float().pow(2).mean()
    engine.backward(loss)

    gradients = {
        name: parameter.grad.detach().float().clone()
        for name, parameter in engine.module.named_parameters() if parameter.grad is not None
    }
    return output.detach().float().clone(), gradients


@pytest.mark.skipif(not _deepep_available(), reason="deep_ep is not installed")
class TestDeepEPMatchesCollective(DistributedTest):
    """One step through each transport must agree, forwards and backwards."""

    world_size = 4
    reuse_dist_env = False

    def test_forward_and_backward_match_the_collective_path(self):
        skip_unless_h100_tests_enabled("DeepEP parity needs H100s and a DeepEP build")

        collective_output, collective_grads = _run_one_step("comm", self.world_size, seed=1234)
        deepep_output, deepep_grads = _run_one_step("deepep", self.world_size, seed=1234)

        # bfloat16 with a different reduction order, so exact equality is not
        # the bar. A dropped weight or a missing expert is orders of magnitude
        # larger than a reordered sum.
        torch.testing.assert_close(deepep_output, collective_output, rtol=2e-2, atol=2e-2)

        assert set(deepep_grads) == set(collective_grads), "the two paths produced gradients for different parameters"
        for name, expected in collective_grads.items():
            torch.testing.assert_close(deepep_grads[name], expected, rtol=5e-2, atol=5e-2, msg=f"gradient for {name}")

    def test_the_router_gate_receives_gradients(self):
        """The gate silently never learning is what a dropped weight costs.

        A transport that returns the routing weights outside its autograd graph
        trains without complaint and never updates the gate, so this asserts the
        gradient exists and is not uniformly zero rather than only comparing it.
        """
        skip_unless_h100_tests_enabled("DeepEP parity needs H100s and a DeepEP build")

        _, gradients = _run_one_step("deepep", self.world_size, seed=99)

        gate_grads = [value for name, value in gradients.items() if "gate" in name]
        assert gate_grads, "the router gate received no gradient at all"
        assert any(value.abs().sum() > 0 for value in gate_grads), "the router gate's gradient was entirely zero"
