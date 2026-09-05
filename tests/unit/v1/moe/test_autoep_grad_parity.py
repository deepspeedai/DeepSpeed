# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP gradient parity paths."""

import copy

import deepspeed
import deepspeed.comm as dist
import pytest
import torch
import torch.nn as nn
from deepspeed.runtime.compiler import is_compiling
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad
from torch.utils.checkpoint import checkpoint
from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import (
    MockHFConfig,
    MockMoEBlock,
    MockMoETransformer,
    engine_input_dtype as _engine_input_dtype,
    h100_tests_enabled,
    make_autoep_config,
    mixed_precision_config as _mixed_precision_config,
    seed_everything as _seed_everything,
)


def _make_model():
    return MockMoETransformer(num_layers=1, num_experts=4, hidden_size=128, intermediate_size=256)


def _make_zero2_config():
    return {
        **_mixed_precision_config(),
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "gradient_clipping": 0.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-3,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
        },
    }


def _make_autoep_zero2_config(ep_size):
    config = _make_zero2_config()
    config["expert_parallel"] = {
        "enabled": True,
        "autoep_size": ep_size,
        "preset_model": "mixtral",
        "load_balance_coeff": None,
        "use_grouped_mm": False,
    }
    return config


def _make_autoep_zero3_config(ep_size):
    config = _make_autoep_zero2_config(ep_size)
    config["zero_optimization"] = {
        "stage": 3,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
    }
    return config


def _make_local_batches(*, logical_dp_world_size, logical_dp_rank, grad_accum, seed, seq_len, micro_batch_size,
                        hidden_size, device, dtype):
    batches = []
    for accum_idx in range(grad_accum):
        batch_idx = accum_idx * logical_dp_world_size + logical_dp_rank
        generator = torch.Generator().manual_seed(seed + batch_idx)
        batches.append(
            torch.randn((micro_batch_size, seq_len, hidden_size), generator=generator, dtype=dtype).to(device))
    return batches


def _run_until_boundary(engine, *, logical_dp_world_size, logical_dp_rank, grad_accum, seed):
    batches = _make_local_batches(
        logical_dp_world_size=logical_dp_world_size,
        logical_dp_rank=logical_dp_rank,
        grad_accum=grad_accum,
        seed=seed,
        seq_len=16,
        micro_batch_size=1,
        hidden_size=128,
        device=engine.device,
        dtype=_engine_input_dtype(engine),
    )
    for batch_idx, batch in enumerate(batches):
        loss = engine(batch).mean()
        engine.backward(loss)
        if batch_idx + 1 < len(batches):
            engine.step()


def _gather_autoep_expert_grad(param, group):
    grad = safe_get_full_grad(param)
    assert grad is not None, "Expected full expert grad"
    group_size = dist.get_world_size(group=group)
    shards = [torch.zeros_like(grad) for _ in range(group_size)]
    dist.all_gather(shards, grad.detach(), group=group)
    # The gather reconstructs expert shards; gradient reduction has already
    # applied the data-parallel normalization, so do not average by EP size.
    return torch.cat([shard.float().cpu() for shard in shards], dim=0)


def _collect_autoep_expert_grads(engine):
    from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

    grads = {}
    for module_name, module in engine.module.named_modules():
        if not isinstance(module, AutoEPMoELayer):
            continue
        prefix = f"{module_name}.experts"
        w1 = _gather_autoep_expert_grad(module.experts.w1, module.ep_group)
        w2 = _gather_autoep_expert_grad(module.experts.w2, module.ep_group)
        w3 = _gather_autoep_expert_grad(module.experts.w3, module.ep_group)
        grads[f"{prefix}.gate_up_proj"] = torch.cat([w1, w3], dim=1)
        grads[f"{prefix}.down_proj"] = w2
    return grads


def _collect_zero2_expert_grads(engine):
    grads = {}
    for name, param in engine.module.named_parameters():
        if name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj"):
            grad = safe_get_full_grad(param)
            assert grad is not None, f"Expected full grad for {name}"
            grads[name] = grad.detach().float().cpu().clone()
    return grads


def _assert_grad_maps_close(actual, expected, *, lhs_name, rhs_name):
    for name in sorted(expected):
        assert name in actual, f"Missing {lhs_name} param snapshot for {name}"
        diff = (actual[name] - expected[name]).abs()
        torch.testing.assert_close(actual[name],
                                   expected[name],
                                   atol=1e-1,
                                   rtol=5e-3,
                                   msg=(f"Gradient mismatch for {name} between {lhs_name} and {rhs_name}; "
                                        f"max_diff={diff.max().item()} "
                                        f"actual_norm={actual[name].norm().item()} "
                                        f"expected_norm={expected[name].norm().item()}"))


class _CompiledDecoderLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(128)
        self.dense = nn.Linear(128, 128, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(128)
        self.mlp = MockMoEBlock(num_experts=4, ffn_hidden=256, hidden_size=128)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.dense(hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class _CompiledAutoEPModel(nn.Module):

    def __init__(self, checkpoint_enabled):
        super().__init__()
        self.config = copy.copy(MockHFConfig())
        self.config.hidden_size = 128
        self.config.intermediate_size = 256
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_CompiledDecoderLayer() for _ in range(2)])
        self.output = nn.Linear(128, 64, bias=False)
        self.checkpoint_enabled = checkpoint_enabled
        with torch.no_grad():
            for name, param in self.named_parameters():
                if "layernorm" in name and param.ndim == 1:
                    param.fill_(1.0)
                else:
                    param.normal_(mean=0.0, std=0.02)

    def forward(self, hidden_states):
        for layer in self.model.layers:
            if self.checkpoint_enabled and self.training:
                hidden_states = checkpoint(layer, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states)
        return self.output(hidden_states)


def _make_compile_config():
    config = make_autoep_config(zero_stage=1, ep_size=2)
    config.pop("fp16", None)
    config["bf16"] = {"enabled": True}
    config["optimizer"] = {
        "type": "SGD",
        "params": {
            "lr": 1e-2
        },
    }
    config["zero_allow_untested_optimizer"] = True
    return config


def _snapshot_dynamo_stats():
    return dict(torch._dynamo.utils.counters["stats"])


def _dynamo_stat_delta(before, after, key):
    return after.get(key, 0) - before.get(key, 0)


def _snapshot_parameter_data(engine):
    snapshot = {}
    for name, param in engine.module.named_parameters():
        full_param = safe_get_full_fp32_param(param)
        assert full_param is not None, f"Expected FP32 master parameter for {name}"
        assert full_param.dtype == torch.float32, f"Expected FP32 master parameter dtype for {name}"
        snapshot[name] = full_param.detach().float().cpu().clone()
    return snapshot


def _run_compile_step(engine, batch):
    input_tensor = batch.detach().clone().requires_grad_(True)
    params_before = _snapshot_parameter_data(engine)
    output = engine(input_tensor)
    loss = output.float().square().mean()
    engine.backward(loss)

    grads = {}
    for name, param in engine.module.named_parameters():
        grad = safe_get_full_grad(param)
        assert grad is not None, f"Expected gradient for {name}"
        grads[name] = grad.detach().float().cpu().clone()
    input_grad = input_tensor.grad.detach().float().cpu().clone()

    engine.step()
    params_after = _snapshot_parameter_data(engine)
    deltas = {name: params_after[name] - params_before[name] for name in params_before}
    return {
        "output": output.detach().float().cpu(),
        "loss": loss.detach().float().cpu(),
        "input_grad": input_grad,
        "grads": grads,
        "deltas": deltas,
    }


def _warm_compile_step(engine, batch):
    input_tensor = batch.detach().clone().requires_grad_(True)
    output = engine(input_tensor)
    output.float().square().mean().backward()
    engine.zero_grad()
    engine.optimizer.zero_grad()


def _assert_relative_tensor_error(actual, expected, name):
    reference_norm = expected.double().norm().item()
    error_norm = (actual.double() - expected.double()).norm().item()
    # Elementwise absolute tolerances alone can accept dropping small gradients or updates.
    allowed_error = 5e-2 * reference_norm
    assert error_norm <= allowed_error, (
        f"{name} relative L2 error exceeds 5%; error_norm={error_norm}, reference_norm={reference_norm}")


def _assert_compile_step_close(actual, expected):
    for name, rtol, atol in (
        ("output", 5e-3, 2e-2),
        ("loss", 5e-3, 2e-3),
        ("input_grad", 5e-3, 5e-3),
    ):
        difference = (actual[name] - expected[name]).abs()
        torch.testing.assert_close(actual[name],
                                   expected[name],
                                   rtol=rtol,
                                   atol=atol,
                                   msg=(f"{name} mismatch; max_diff={difference.max().item()}, "
                                        f"actual_norm={actual[name].norm().item()}, "
                                        f"expected_norm={expected[name].norm().item()}"))
    _assert_relative_tensor_error(actual["input_grad"], expected["input_grad"], "input_grad")
    assert actual["grads"].keys() == expected["grads"].keys(), "Gradient parameter sets differ"
    assert actual["deltas"].keys() == expected["deltas"].keys(), "Optimizer parameter sets differ"
    for name in actual["grads"]:
        torch.testing.assert_close(actual["grads"][name],
                                   expected["grads"][name],
                                   rtol=5e-3,
                                   atol=5e-3,
                                   msg=f"Gradient mismatch for {name}")
        torch.testing.assert_close(actual["deltas"][name],
                                   expected["deltas"][name],
                                   rtol=5e-3,
                                   atol=5e-5,
                                   msg=(f"Optimizer delta max_diff="
                                        f"{(actual['deltas'][name] - expected['deltas'][name]).abs().max().item()}, "
                                        f"actual_norm={actual['deltas'][name].norm().item()}, "
                                        f"expected_norm={expected['deltas'][name].norm().item()}, name={name}"))
        _assert_relative_tensor_error(actual["grads"][name], expected["grads"][name], f"grads[{name}]")
        _assert_relative_tensor_error(actual["deltas"][name], expected["deltas"][name], f"deltas[{name}]")


def _register_autoep_observers(engine):
    from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

    eager_calls = []
    routes = []
    handles = []
    for name, module in engine.module.named_modules():
        if not isinstance(module, AutoEPMoELayer):
            continue

        def observe_router(_module, _inputs, output, name=name):
            # Observe the production eager boundary without installing one in the test.
            assert not is_compiling(), f"AutoEP router was traced for {name}"
            eager_calls.append(name)
            routes.append((name, output[1].detach().cpu()))

        handles.append(module.router.register_forward_hook(observe_router))
    return eager_calls, routes, handles


class TestAutoEPCompileParityAssertions:

    @pytest.mark.parametrize("field", ["grads", "deltas"])
    @pytest.mark.parametrize("multiplier", [0.0, -1.0, 1.01])
    def test_small_gradient_and_update_relative_error(self, field, multiplier):
        expected = {
            "output": torch.ones(2),
            "loss": torch.ones(()),
            "input_grad": torch.ones(2),
            "grads": {
                "router.weight": torch.tensor([1e-5, -2e-5])
            },
            "deltas": {
                "router.weight": torch.tensor([-1e-7, 2e-7])
            },
        }
        actual = copy.deepcopy(expected)
        actual[field]["router.weight"].mul_(multiplier)

        if multiplier <= 0:
            with pytest.raises(AssertionError, match=f"{field}.*relative L2 error"):
                _assert_compile_step_close(actual, expected)
        else:
            _assert_compile_step_close(actual, expected)

    def test_zero_reference_requires_zero_actual(self):
        reference = torch.zeros(2)
        _assert_relative_tensor_error(reference.clone(), reference, "zero")
        with pytest.raises(AssertionError, match="zero relative L2 error"):
            _assert_relative_tensor_error(torch.tensor([1e-10, 0.0]), reference, "zero")


class TestAutoEPGradParity(DistributedTest):
    world_size = 4

    def test_zero2_autoep_matches_zero2_after_one_update(self):
        ep_size = 2
        seed = 1234

        _seed_everything(seed)
        reference_state = _make_model().state_dict()

        autoep_model = _make_model()
        zero2_model = _make_model()
        autoep_model.load_state_dict(reference_state)
        zero2_model.load_state_dict(reference_state)

        autoep_engine, _, _, _ = deepspeed.initialize(model=autoep_model, config=_make_autoep_zero2_config(ep_size))
        zero2_engine, _, _, _ = deepspeed.initialize(model=zero2_model, config=_make_zero2_config())

        autoep_rank = dist.get_rank() // ep_size
        _run_until_boundary(autoep_engine,
                            logical_dp_world_size=self.world_size // ep_size,
                            logical_dp_rank=autoep_rank,
                            grad_accum=2,
                            seed=seed)
        _run_until_boundary(zero2_engine,
                            logical_dp_world_size=self.world_size // ep_size,
                            logical_dp_rank=autoep_rank,
                            grad_accum=2,
                            seed=seed)

        autoep_expert = _collect_autoep_expert_grads(autoep_engine)
        zero2_expert = _collect_zero2_expert_grads(zero2_engine)

        dist.barrier()
        if dist.get_rank() != 0:
            return

        _assert_grad_maps_close(autoep_expert, zero2_expert, lhs_name="AutoEP expert", rhs_name="ZeRO-2 expert")

    def test_zero3_autoep_expert_grads_match_zero2_autoep(self):
        ep_size = 2
        seed = 2345

        _seed_everything(seed)
        reference_state = _make_model().state_dict()

        zero2_model = _make_model()
        zero3_model = _make_model()
        zero2_model.load_state_dict(reference_state)
        zero3_model.load_state_dict(reference_state)

        zero2_engine, _, _, _ = deepspeed.initialize(model=zero2_model, config=_make_autoep_zero2_config(ep_size))
        zero3_engine, _, _, _ = deepspeed.initialize(model=zero3_model, config=_make_autoep_zero3_config(ep_size))

        logical_rank = dist.get_rank() // ep_size
        logical_world_size = self.world_size // ep_size
        _run_until_boundary(zero2_engine,
                            logical_dp_world_size=logical_world_size,
                            logical_dp_rank=logical_rank,
                            grad_accum=2,
                            seed=seed)
        _run_until_boundary(zero3_engine,
                            logical_dp_world_size=logical_world_size,
                            logical_dp_rank=logical_rank,
                            grad_accum=2,
                            seed=seed)

        zero2_expert = _collect_autoep_expert_grads(zero2_engine)
        zero3_expert = _collect_autoep_expert_grads(zero3_engine)

        dist.barrier()
        if dist.get_rank() != 0:
            return

        _assert_grad_maps_close(zero3_expert,
                                zero2_expert,
                                lhs_name="ZeRO-3 AutoEP expert",
                                rhs_name="ZeRO-2 AutoEP expert")


@pytest.mark.skipif(not h100_tests_enabled(), reason="AutoEP regional compile parity requires an H100 test run")
class TestAutoEPRegionalCompileParity(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("checkpoint_enabled", [True, False])
    def test_regional_compile_matches_eager(self, checkpoint_enabled):
        seed = 3456
        _seed_everything(seed)
        reference_model = _CompiledAutoEPModel(checkpoint_enabled)
        reference_state = copy.deepcopy(reference_model.state_dict())

        eager_model = _CompiledAutoEPModel(checkpoint_enabled)
        compiled_model = _CompiledAutoEPModel(checkpoint_enabled)
        eager_model.load_state_dict(reference_state)
        compiled_model.load_state_dict(reference_state)

        eager_engine, _, _, _ = deepspeed.initialize(model=eager_model, config=_make_compile_config())
        compiled_engine, _, _, _ = deepspeed.initialize(model=compiled_model, config=_make_compile_config())
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        compiled_engine.compile(compile_mode="autoep_non_moe")

        eager_calls, eager_routes, eager_handles = _register_autoep_observers(eager_engine)
        compiled_calls, compiled_routes, compiled_handles = _register_autoep_observers(compiled_engine)
        generator = torch.Generator().manual_seed(seed + dist.get_rank())
        dtype = _engine_input_dtype(eager_engine)
        warmup_batch = torch.randn((1, 16, 128), generator=generator, dtype=dtype).to(eager_engine.device)
        measured_batch = torch.randn((1, 16, 128), generator=generator, dtype=dtype).to(eager_engine.device)

        _warm_compile_step(eager_engine, warmup_batch)
        _warm_compile_step(compiled_engine, warmup_batch)

        eager_call_start = len(eager_calls)
        compiled_call_start = len(compiled_calls)
        eager_route_start = len(eager_routes)
        compiled_route_start = len(compiled_routes)
        dynamo_start = _snapshot_dynamo_stats()
        assert dynamo_start.get("unique_graphs", 0) > 0, f"Warmup did not capture graphs: {dynamo_start}"
        assert dynamo_start.get("calls_captured", 0) > 0, f"Warmup did not capture calls: {dynamo_start}"

        measured_eager = _run_compile_step(eager_engine, measured_batch)
        measured_compiled = _run_compile_step(compiled_engine, measured_batch)
        _assert_compile_step_close(measured_compiled, measured_eager)

        dynamo_end = _snapshot_dynamo_stats()
        expected_calls = len(compiled_engine._compiled_regions) * (2 if checkpoint_enabled else 1)
        eager_call_delta = len(eager_calls) - eager_call_start
        compiled_call_delta = len(compiled_calls) - compiled_call_start
        assert eager_call_delta == expected_calls, f"Eager AutoEP calls: expected={expected_calls}, got={eager_call_delta}"
        assert compiled_call_delta == expected_calls, (
            f"Compiled AutoEP calls: expected={expected_calls}, got={compiled_call_delta}")
        unique_graph_delta = _dynamo_stat_delta(dynamo_start, dynamo_end, "unique_graphs")
        captured_call_delta = _dynamo_stat_delta(dynamo_start, dynamo_end, "calls_captured")
        assert unique_graph_delta == 0, f"Measured unique_graphs delta={unique_graph_delta}"
        assert captured_call_delta == 0, f"Measured calls_captured delta={captured_call_delta}"

        measured_eager_routes = eager_routes[eager_route_start:]
        measured_compiled_routes = compiled_routes[compiled_route_start:]
        assert len(measured_eager_routes) == expected_calls, (
            f"Eager routes: expected={expected_calls}, got={len(measured_eager_routes)}")
        assert len(measured_compiled_routes) == expected_calls, (
            f"Compiled routes: expected={expected_calls}, got={len(measured_compiled_routes)}")
        for (eager_name, eager_route), (compiled_name, compiled_route) in zip(measured_eager_routes,
                                                                              measured_compiled_routes):
            assert eager_name == compiled_name, f"Route layer mismatch: {eager_name} != {compiled_name}"
            assert torch.equal(eager_route, compiled_route), f"Route assignment mismatch for {eager_name}"

        grad_names = measured_compiled["grads"]
        assert any(".experts.w1" in name for name in grad_names), "Expert gradients were not checked"
        assert any(".router.gate.weight" in name for name in grad_names), "Router gradients were not checked"
        assert any(".dense.weight" in name for name in grad_names), "Non-MoE gradients were not checked"

        for handle in eager_handles + compiled_handles:
            handle.remove()
