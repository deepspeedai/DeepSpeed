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

import functools
from unittest import mock

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import deepspeed
import deepspeed.comm as dist
from deepspeed.module_inject import auto_ep_layer
from deepspeed.module_inject.auto_ep_comm import destroy_exchanges
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.utils import safe_get_full_fp32_param

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
INTERMEDIATE_SIZE = 128
SEQ_LEN = 8


def _deepep_available() -> bool:
    try:
        import deep_ep  # noqa: F401
    except Exception:
        return False
    return True


def _install_legacy_deepep_prep(engine):
    """Reproduce the pre-cleanup work without changing DeepEP mathematics."""
    for module in engine.module.modules():
        if not isinstance(module, AutoEPMoELayer):
            continue
        deepep_route = module._deepep_route

        @functools.wraps(deepep_route)
        def legacy_deepep_route(tokens, ro, *, _module=module, _deepep_route=deepep_route):
            token_indices_sorted = torch.argsort(ro.selected_experts.view(-1), stable=True)
            top_scores_sorted = ro.top_scores.view(-1)[token_indices_sorted]
            ro.selected_experts.reshape(-1).index_select(0, token_indices_sorted)
            routed_input = tokens[token_indices_sorted // _module.top_k]
            auto_ep_layer.apply_scores_before_experts_if_enabled(routed_input,
                                                                 top_scores_sorted,
                                                                 score_apply=_module.score_apply)
            auto_ep_layer.compute_split_plan(
                selected_experts=ro.selected_experts,
                num_experts=_module.num_experts,
                ep_size=_module.ep_size,
                num_local_experts=_module.num_local_experts,
                ep_group=_module.ep_group,
                num_tokens_per_expert=ro.num_tokens_per_expert,
            )
            return _deepep_route(tokens, ro)

        module._deepep_route = legacy_deepep_route


def _install_skewed_routing(engine):
    for module in engine.module.modules():
        if not isinstance(module, AutoEPMoELayer):
            continue
        router = module.router

        def skewed_forward(hidden_states, _expert_bias, *, _router=router):
            logits = _router.gate(hidden_states)
            scores = torch.softmax(logits.float(), dim=-1).to(hidden_states.dtype)
            pattern = torch.tensor([[1, 0], [1, 0], [2, 0], [1, 2]], dtype=torch.long, device=hidden_states.device)
            selected_experts = pattern.repeat((hidden_states.shape[0] + pattern.shape[0] - 1) // pattern.shape[0],
                                              1)[:hidden_states.shape[0]]
            top_scores = scores.gather(1, selected_experts)
            top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True)
            counts = torch.bincount(selected_experts.flatten(), minlength=_router.num_experts).to(torch.int32)
            return top_scores, selected_experts, counts

        router.forward = skewed_forward


def _checkpoint_autoep_layers(engine):
    for module in engine.module.modules():
        if isinstance(module, AutoEPMoELayer):
            module.forward = functools.partial(checkpoint, module.forward, use_reentrant=False)


def _snapshot_fp32_parameters(engine):
    optimizer = engine.optimizer
    master_parameters = {}
    # Stage-0 low-precision wrappers do not expose the safe_get parameter mapping.
    if hasattr(optimizer, "fp16_groups"):
        state = optimizer.state_dict()
        for index, parameters in enumerate(optimizer.fp16_groups):
            if "fp32_groups_flat" in state:
                master_group = engine.unflatten(state["fp32_groups_flat"][index], parameters)
            else:
                assert "fp32_groups" in state, "Expected FP32 master parameters in optimizer state_dict"
                master_group = state["fp32_groups"][index]
            assert len(master_group) == len(parameters), "FP32 master parameter group does not match model parameters"
            master_parameters.update(zip(parameters, master_group))

    snapshot = {}
    for name, parameter in engine.module.named_parameters():
        full_parameter = master_parameters.get(parameter)
        if full_parameter is None:
            full_parameter = safe_get_full_fp32_param(parameter)
        assert full_parameter is not None, f"Expected FP32 master parameter for {name}"
        assert full_parameter.dtype == torch.float32, f"Expected FP32 master parameter dtype for {name}"
        snapshot[name] = full_parameter.detach().cpu().clone()
    return snapshot


def _run_one_step(backend, ep_size, seed, *, cleanup=True, activation_checkpointing=False, skewed_routing=False):
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

    model = MockMoETransformer(hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE)
    # Mock experts start from unscaled N(0, 1) tensors, unlike the linear
    # layers around them. Scale each projection by its fan-in so two MoE layers
    # do not amplify BF16 reduction-order differences into outputs in the
    # thousands.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("experts.gate_up_proj"):
                parameter.mul_(HIDDEN_SIZE**-0.5)
            elif name.endswith("experts.down_proj"):
                parameter.mul_(INTERMEDIATE_SIZE**-0.5)
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    if backend == "deepep" and not cleanup:
        _install_legacy_deepep_prep(engine)
    if skewed_routing:
        _install_skewed_routing(engine)
    if activation_checkpointing:
        _checkpoint_autoep_layers(engine)

    # Reseeded so the input is identical on every rank and across backends: the
    # comparison is of the transport, so nothing else may differ.
    seed_everything(seed)
    hidden = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, device=engine.device,
                         dtype=engine_input_dtype(engine)).requires_grad_(True)
    parameters_before = _snapshot_fp32_parameters(engine)
    routes = []
    score_tensors = []
    hooks = []
    for name, module in engine.module.named_modules():
        if not isinstance(module, AutoEPMoELayer):
            continue

        def capture_route(_module, _inputs, output, *, _name=name):
            scores, selected_experts = output[:2]
            if scores.requires_grad:
                scores.retain_grad()
                score_tensors.append((_name, scores))
            routes.append((_name, selected_experts.detach().cpu().clone()))

        hooks.append(module.router.register_forward_hook(capture_route))

    output = engine(hidden)
    loss = output.float().pow(2).mean()
    engine.backward(loss)

    gradients = {
        name: parameter.grad.detach().float().clone()
        for name, parameter in engine.module.named_parameters() if parameter.grad is not None
    }
    score_gradient_parts = {}
    for name, scores in score_tensors:
        if scores.grad is not None:
            score_gradient_parts.setdefault(name, []).append(scores.grad.detach().float())
    score_gradients = {name: torch.stack(parts).sum(dim=0) for name, parts in score_gradient_parts.items()}
    input_gradient = hidden.grad.detach().float().clone()
    engine.step()
    parameters_after = _snapshot_fp32_parameters(engine)
    parameter_deltas = {name: parameters_after[name] - parameters_before[name] for name in parameters_before}
    for hook in hooks:
        hook.remove()
    result = {
        "output": output.detach().float().clone(),
        "loss": loss.detach().float().clone(),
        "routes": routes,
        "input_gradient": input_gradient,
        "score_gradients": score_gradients,
        "gradients": gradients,
        "parameter_deltas": parameter_deltas,
    }
    if backend == "deepep":
        destroy_exchanges(engine.module)
    return result


def _assert_cleanup_results_close(actual, expected, *, compare_score_gradients):
    for name, rtol, atol in (
        ("output", 2e-3, 2e-3),
        ("loss", 2e-3, 2e-3),
        ("input_gradient", 1e-2, 1e-2),
    ):
        difference = (actual[name] - expected[name]).abs()
        torch.testing.assert_close(actual[name],
                                   expected[name],
                                   rtol=rtol,
                                   atol=atol,
                                   msg=(f"{name} mismatch; max_diff={difference.max().item()}, "
                                        f"actual_norm={actual[name].norm().item()}, "
                                        f"expected_norm={expected[name].norm().item()}"))
    # DeepEP atomics can change small gradient elements between equivalent runs.
    # The checks below retain exact routes and compare the stable training invariants.
    assert len(actual["routes"]) == len(expected["routes"])
    for (actual_name, actual_route), (expected_name, expected_route) in zip(actual["routes"], expected["routes"]):
        assert actual_name == expected_name
        assert torch.equal(actual_route, expected_route)
    assert actual["score_gradients"].keys() == expected["score_gradients"].keys()
    for name, actual_grad in actual["score_gradients"].items():
        expected_grad = expected["score_gradients"][name]
        actual_norm = actual_grad.norm()
        expected_norm = expected_grad.norm()
        assert torch.isfinite(actual_grad).all() and torch.isfinite(expected_grad).all(), (
            f"routing-score gradient is non-finite for {name}")
        assert actual_norm > 0 and expected_norm > 0, f"routing-score gradient is zero for {name}"
        if not compare_score_gradients:
            continue
        # DeepEP's cross-rank reduction order can change retained score-gradient
        # elements between equivalent runs. The norm is stable, while the
        # downstream router parameter gradient is compared elementwise below.
        torch.testing.assert_close(actual_norm,
                                   expected_norm,
                                   rtol=2e-1,
                                   atol=2e-2,
                                   msg=f"routing-score gradient norm for {name}")
    assert actual["gradients"].keys() == expected["gradients"].keys()
    assert actual["parameter_deltas"].keys() == expected["parameter_deltas"].keys()
    for name in actual["gradients"]:
        torch.testing.assert_close(
            actual["gradients"][name],
            expected["gradients"][name],
            rtol=5e-2,
            atol=5e-2,
            msg=(f"gradient for {name}; max_diff="
                 f"{(actual['gradients'][name] - expected['gradients'][name]).abs().max().item()}"))
        torch.testing.assert_close(
            actual["parameter_deltas"][name],
            expected["parameter_deltas"][name],
            rtol=5e-3,
            atol=5e-4,
            msg=(f"optimizer delta for {name}; max_diff="
                 f"{(actual['parameter_deltas'][name] - expected['parameter_deltas'][name]).abs().max().item()}"))


@pytest.mark.skipif(not _deepep_available(), reason="deep_ep is not installed")
class TestDeepEPMatchesCollective(DistributedTest):
    """One step through each transport must agree, forwards and backwards."""

    world_size = 4
    reuse_dist_env = False

    def test_forward_and_backward_match_the_collective_path(self):
        skip_unless_h100_tests_enabled("DeepEP parity needs H100s and a DeepEP build")

        collective = _run_one_step("comm", self.world_size, seed=1234)
        deepep = _run_one_step("deepep", self.world_size, seed=1234)

        # bfloat16 with a different reduction order, so exact equality is not
        # the bar. A dropped weight or a missing expert is orders of magnitude
        # larger than a reordered sum.
        torch.testing.assert_close(deepep["output"], collective["output"], rtol=2e-2, atol=2e-2)

        assert set(deepep["gradients"]) == set(
            collective["gradients"]), "the two paths produced gradients for different parameters"
        for name, expected in collective["gradients"].items():
            torch.testing.assert_close(deepep["gradients"][name],
                                       expected,
                                       rtol=5e-2,
                                       atol=5e-2,
                                       msg=f"gradient for {name}")

    def test_the_router_gate_receives_gradients(self):
        """The gate silently never learning is what a dropped weight costs.

        A transport that returns the routing weights outside its autograd graph
        trains without complaint and never updates the gate, so this asserts the
        gradient exists and is not uniformly zero rather than only comparing it.
        """
        skip_unless_h100_tests_enabled("DeepEP parity needs H100s and a DeepEP build")

        result = _run_one_step("deepep", self.world_size, seed=99)

        gate_grads = [value for name, value in result["gradients"].items() if "gate" in name]
        assert gate_grads, "the router gate received no gradient at all"
        assert any(value.abs().sum() > 0 for value in gate_grads), "the router gate's gradient was entirely zero"

    @pytest.mark.parametrize(
        "activation_checkpointing, skewed_routing",
        [
            (True, False),
            (False, True),
        ],
    )
    def test_cleanup_matches_legacy_preparation(self, activation_checkpointing, skewed_routing):
        skip_unless_h100_tests_enabled("DeepEP cleanup parity needs H100s and a DeepEP build")
        seed = 5678

        legacy = _run_one_step(
            "deepep",
            self.world_size,
            seed,
            cleanup=False,
            activation_checkpointing=activation_checkpointing,
            skewed_routing=skewed_routing,
        )
        cleanup = _run_one_step(
            "deepep",
            self.world_size,
            seed,
            cleanup=True,
            activation_checkpointing=activation_checkpointing,
            skewed_routing=skewed_routing,
        )

        _assert_cleanup_results_close(cleanup, legacy, compare_score_gradients=not activation_checkpointing)
        if skewed_routing:
            all_routes = torch.cat([route.flatten() for _, route in cleanup["routes"]])
            assert torch.count_nonzero(all_routes == 3) == 0
            assert torch.count_nonzero(all_routes == 1) > torch.count_nonzero(all_routes == 2)


@pytest.mark.skipif(not _deepep_available(), reason="deep_ep is not installed")
class TestDeepEPColdStart(DistributedTest):
    world_size = 4
    init_distributed = False
    reuse_dist_env = False

    def test_cleanup_initializes_a_lazy_ep_communicator(self):
        skip_unless_h100_tests_enabled("DeepEP cold start needs H100s and a DeepEP build")

        # Exercise an unbound process group, as when a caller initializes
        # distributed without device_id before handing the model to DeepSpeed.
        with mock.patch("deepspeed.comm.torch.known_world_size", return_value=1):
            deepspeed.init_distributed(dist_backend="nccl")
        assert dist.get_world_group().bound_device_id is None

        cleanup = _run_one_step("deepep", self.world_size, seed=1234)
        collective = _run_one_step("comm", self.world_size, seed=1234)

        torch.testing.assert_close(cleanup["output"], collective["output"], rtol=2e-2, atol=2e-2)
        assert cleanup["gradients"].keys() == collective["gradients"].keys()
        for name, expected in collective["gradients"].items():
            torch.testing.assert_close(cleanup["gradients"][name],
                                       expected,
                                       rtol=5e-2,
                                       atol=5e-2,
                                       msg=f"cold-start gradient for {name}")
