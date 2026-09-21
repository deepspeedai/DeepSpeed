# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP gradient parity paths."""

import copy
import functools
from types import SimpleNamespace

import deepspeed
import deepspeed.comm as dist
import pytest
import torch
import torch.nn as nn
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
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


def _make_async_split_model():
    model = _make_model()
    # The mock experts use unscaled N(0, 1) weights. Keep the multi-step squared-loss
    # parity test finite so it compares planner behavior rather than overflow.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("experts.gate_up_proj"):
                parameter.mul_(128**-0.5)
            elif name.endswith("experts.down_proj"):
                parameter.mul_(256**-0.5)
    return model


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

    def __init__(self, checkpoint_enabled, model_layout="nested"):
        super().__init__()
        self.config = copy.copy(MockHFConfig())
        self.config.hidden_size = 128
        self.config.intermediate_size = 256
        self.model_layout = model_layout
        if model_layout == "root":
            for name, module in _CompiledDecoderLayer().named_children():
                self.add_module(name, module)
        else:
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
        if self.model_layout == "root":
            hidden_states = _CompiledDecoderLayer.forward(self, hidden_states)
        else:
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


def _run_compile_step(engine, batch, checkpoint_root=False):
    input_tensor = batch.detach().clone().requires_grad_(True)
    params_before = _snapshot_parameter_data(engine)
    if checkpoint_root:
        output = checkpoint(engine, input_tensor, use_reentrant=False)
    else:
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


def _warm_compile_step(engine, batch, checkpoint_root=False):
    input_tensor = batch.detach().clone().requires_grad_(True)
    if checkpoint_root:
        output = checkpoint(engine, input_tensor, use_reentrant=False)
    else:
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

    @pytest.mark.parametrize("async_split_plan", [False, True])
    @pytest.mark.parametrize("model_layout", ["nested", "root"])
    @pytest.mark.parametrize("checkpoint_enabled", [True, False])
    def test_regional_compile_matches_eager(self, checkpoint_enabled, model_layout, async_split_plan):
        seed = 3456
        _seed_everything(seed)
        reference_model = _CompiledAutoEPModel(checkpoint_enabled, model_layout)
        reference_state = copy.deepcopy(reference_model.state_dict())

        eager_model = _CompiledAutoEPModel(checkpoint_enabled, model_layout)
        compiled_model = _CompiledAutoEPModel(checkpoint_enabled, model_layout)
        eager_model.load_state_dict(reference_state)
        compiled_model.load_state_dict(reference_state)

        eager_config = _make_compile_config()
        eager_config["expert_parallel"]["async_split_plan"] = async_split_plan
        if model_layout == "root":
            eager_config["expert_parallel"]["moe_layer_pattern"] = "mlp"
        compiled_config = copy.deepcopy(eager_config)
        compiled_config["compile"] = {"autoep_non_moe": True}
        eager_engine, _, _, _ = deepspeed.initialize(model=eager_model, config=eager_config)
        compiled_engine, _, _, _ = deepspeed.initialize(model=compiled_model, config=compiled_config)
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        compiled_engine.compile()

        eager_calls, eager_routes, eager_handles = _register_autoep_observers(eager_engine)
        compiled_calls, compiled_routes, compiled_handles = _register_autoep_observers(compiled_engine)
        generator = torch.Generator().manual_seed(seed + dist.get_rank())
        dtype = _engine_input_dtype(eager_engine)
        warmup_batch = torch.randn((1, 16, 128), generator=generator, dtype=dtype).to(eager_engine.device)
        measured_batch = torch.randn((1, 16, 128), generator=generator, dtype=dtype).to(eager_engine.device)

        # A root region has no parent module to own activation checkpointing.
        checkpoint_root = model_layout == "root" and checkpoint_enabled
        _warm_compile_step(eager_engine, warmup_batch, checkpoint_root)
        _warm_compile_step(compiled_engine, warmup_batch, checkpoint_root)

        eager_call_start = len(eager_calls)
        compiled_call_start = len(compiled_calls)
        eager_route_start = len(eager_routes)
        compiled_route_start = len(compiled_routes)
        dynamo_start = _snapshot_dynamo_stats()
        assert dynamo_start.get("unique_graphs", 0) > 0, f"Warmup did not capture graphs: {dynamo_start}"
        assert dynamo_start.get("calls_captured", 0) > 0, f"Warmup did not capture calls: {dynamo_start}"

        measured_eager = _run_compile_step(eager_engine, measured_batch, checkpoint_root)
        measured_compiled = _run_compile_step(compiled_engine, measured_batch, checkpoint_root)
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
        assert any(name.endswith("dense.weight") for name in grad_names), "Non-MoE gradients were not checked"

        for handle in eager_handles + compiled_handles:
            handle.remove()


def _async_split_config(enabled):
    return {
        **_mixed_precision_config(),
        "train_micro_batch_size_per_gpu": 1,
        "gradient_clipping": 0.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-3,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        },
        "expert_parallel": {
            "enabled": True,
            "autoep_size": 2,
            "preset_model": "mixtral",
            "load_balance_coeff": None,
            "use_grouped_mm": False,
            "async_split_plan": enabled,
        },
    }


def _checkpoint_autoep_layers(engine):
    for module in engine.module.modules():
        if isinstance(module, AutoEPMoELayer):
            module.forward = functools.partial(torch.utils.checkpoint.checkpoint, module.forward, use_reentrant=False)


def _trainable_parameters(engine):
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
        if not parameter.requires_grad:
            continue
        full_parameter = master_parameters.get(parameter)
        if full_parameter is None:
            full_parameter = safe_get_full_fp32_param(parameter)
        assert full_parameter is not None, f"Expected FP32 master parameter for {name}"
        assert full_parameter.dtype == torch.float32, f"Expected FP32 master parameter dtype for {name}"
        snapshot[name] = full_parameter.detach().cpu().clone()
    return snapshot


def _trainable_gradients(engine):
    gradients = {}
    for name, param in engine.module.named_parameters():
        if not param.requires_grad:
            continue
        grad = safe_get_full_grad(param)
        if grad is not None:
            gradients[name] = grad.detach().float().cpu().clone()
    return gradients


def _async_split_step(engine, seed, seq_len):
    generator = torch.Generator().manual_seed(seed + dist.get_rank())
    batch = torch.randn((1, seq_len, 128), generator=generator, dtype=torch.float32)
    batch = batch.to(engine.device, dtype=_engine_input_dtype(engine)).requires_grad_(True)
    before = _trainable_parameters(engine)

    output = engine(batch)
    loss = output.float().square().mean()
    engine.backward(loss)
    gradients = _trainable_gradients(engine)
    input_grad = batch.grad.detach().float().cpu().clone()
    engine.step()
    after = _trainable_parameters(engine)

    return {
        "loss": loss.detach().float().cpu(),
        "output": output.detach().float().cpu(),
        "input_grad": input_grad,
        "gradients": gradients,
        "delta": {
            name: after[name] - value
            for name, value in before.items()
        },
    }


def _assert_async_split_step_matches(actual, expected):
    tolerance = {"rtol": 1e-2, "atol": 1e-3}
    for name in ("loss", "output", "input_grad"):
        difference = (actual[name] - expected[name]).abs()
        torch.testing.assert_close(actual[name],
                                   expected[name],
                                   msg=f"{name} mismatch; max_diff={difference.max().item()}",
                                   **tolerance)
    _assert_relative_l2_close(actual["input_grad"], expected["input_grad"], "input gradient")

    assert actual["gradients"]
    assert set(actual["gradients"]) == set(expected["gradients"])
    assert any(".router." in name for name in actual["gradients"])
    assert any(".experts.w" in name for name in actual["gradients"])
    for name in sorted(expected["gradients"]):
        max_difference = (actual["gradients"][name] - expected["gradients"][name]).abs().max().item()
        torch.testing.assert_close(actual["gradients"][name],
                                   expected["gradients"][name],
                                   msg=f"gradient mismatch for {name}; max_diff={max_difference}",
                                   **tolerance)
        _assert_relative_l2_close(actual["gradients"][name], expected["gradients"][name], f"gradient {name}")
    assert set(actual["delta"]) == set(expected["delta"])
    for name in sorted(expected["delta"]):
        max_difference = (actual["delta"][name] - expected["delta"][name]).abs().max().item()
        torch.testing.assert_close(actual["delta"][name],
                                   expected["delta"][name],
                                   msg=f"parameter update mismatch for {name}; max_diff={max_difference}",
                                   **tolerance)
        _assert_relative_l2_close(actual["delta"][name], expected["delta"][name], f"parameter update {name}")


def _assert_relative_l2_close(actual, expected, name):
    # An absolute tolerance can hide a missing gradient or a small Adam update.
    assert torch.isfinite(actual).all(), f"{name} contains non-finite values"
    assert torch.isfinite(expected).all(), f"reference {name} contains non-finite values"
    difference_norm = torch.linalg.vector_norm((actual.double() - expected.double()).flatten())
    reference_norm = torch.linalg.vector_norm(expected.double().flatten())
    assert difference_norm <= 0.05 * reference_norm, (
        f"{name} relative L2 error exceeds 5%: error_norm={difference_norm.item()}, "
        f"reference_norm={reference_norm.item()}")


def _small_async_split_result(scale=1.0):
    values = torch.tensor([1e-6, -2e-6]) * scale
    names = ("layer.router.gate.weight", "layer.experts.w1")
    return {
        "loss": torch.tensor(1.0),
        "output": torch.ones(2),
        "input_grad": values.clone(),
        "gradients": {
            name: values.clone()
            for name in names
        },
        "delta": {
            name: values.clone()
            for name in names
        },
    }


@pytest.mark.parametrize("field", ["input_grad", "gradients", "delta"])
@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_async_parity_rejects_small_missing_or_reversed_signal(field, scale):
    expected = _small_async_split_result()
    actual = _small_async_split_result()
    if field == "input_grad":
        actual[field].mul_(scale)
    else:
        actual[field]["layer.router.gate.weight"].mul_(scale)

    with pytest.raises(AssertionError, match="relative L2 error"):
        _assert_async_split_step_matches(actual, expected)


def test_async_parity_accepts_one_percent_relative_error():
    _assert_async_split_step_matches(_small_async_split_result(1.01), _small_async_split_result())


@pytest.mark.parametrize("master_key", ["fp32_groups_flat", "fp32_groups"])
def test_async_master_snapshot_preserves_updates_below_bf16_resolution(master_key):
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1)
    master = torch.ones_like(model.weight, dtype=torch.float32)
    groups = [master.flatten()] if master_key == "fp32_groups_flat" else [[master]]
    optimizer = SimpleNamespace(fp16_groups=[[model.weight]], state_dict=lambda: {master_key: groups})
    engine = SimpleNamespace(module=model,
                             optimizer=optimizer,
                             unflatten=lambda flat, parameters: [flat.reshape_as(parameters[0])])

    before = _trainable_parameters(engine)
    update = 2**-16
    master.add_(update)
    after = _trainable_parameters(engine)

    torch.testing.assert_close(model.weight.float(), torch.ones_like(master), rtol=0, atol=0)
    torch.testing.assert_close(before["weight"], torch.ones_like(master), rtol=0, atol=0)
    torch.testing.assert_close(after["weight"] - before["weight"], torch.full_like(master, update), rtol=0, atol=0)


def test_async_master_snapshot_requires_real_fp32_parameters():
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    engine = SimpleNamespace(module=model, optimizer=object())
    with pytest.raises(AssertionError, match="Expected FP32 master parameter"):
        _trainable_parameters(engine)


class TestAutoEPAsyncSplitPlanParity(DistributedTest):
    """Training after inference warmup across checkpoint and caller-stream modes."""

    world_size = 2

    @pytest.mark.parametrize("checkpoint_activations", [False, True])
    @pytest.mark.parametrize("non_default_stream", [False, True])
    def test_async_split_plan_matches_synchronous_step(self, checkpoint_activations, non_default_stream):
        accelerator = get_accelerator()
        if not accelerator.is_available() or not accelerator.device_name().startswith("cuda"):
            pytest.skip("async split-plan parity requires CUDA")
        seed = 9753
        _seed_everything(seed)
        reference_state = _make_async_split_model().state_dict()

        sync_model = _make_async_split_model()
        sync_model.load_state_dict(reference_state)
        sync_engine, _, _, _ = deepspeed.initialize(model=sync_model, config=_async_split_config(False))
        if checkpoint_activations:
            _checkpoint_autoep_layers(sync_engine)
        sequence_lengths = (16, 7, 23)
        caller_stream = accelerator.current_stream(sync_engine.device)
        training_stream = accelerator.Stream(device=sync_engine.device) if non_default_stream else caller_stream
        # Initialization writes parameters on the caller stream. Keep them alive
        # until training has finished before handing their storage back to it.
        training_stream.wait_stream(caller_stream)
        try:
            with accelerator.stream(training_stream):
                with torch.inference_mode():
                    sync_engine(
                        torch.zeros((1, 5, 128), device=sync_engine.device, dtype=_engine_input_dtype(sync_engine)))
                expected = [
                    _async_split_step(sync_engine, seed + step, seq_len)
                    for step, seq_len in enumerate(sequence_lengths)
                ]
            caller_stream.wait_stream(training_stream)

            async_model = _make_async_split_model()
            async_model.load_state_dict(reference_state)
            async_engine, _, _, _ = deepspeed.initialize(model=async_model, config=_async_split_config(True))
            assert all(module.async_split_plan for module in async_engine.module.modules()
                       if isinstance(module, AutoEPMoELayer))
            if checkpoint_activations:
                _checkpoint_autoep_layers(async_engine)
            training_stream.wait_stream(caller_stream)
            with accelerator.stream(training_stream):
                with torch.inference_mode():
                    async_engine(
                        torch.zeros((1, 5, 128), device=async_engine.device, dtype=_engine_input_dtype(async_engine)))
                for step, seq_len in enumerate(sequence_lengths):
                    actual = _async_split_step(async_engine, seed + step, seq_len)
                    _assert_async_split_step_matches(actual, expected[step])
                    assert all(module._async_split_plan_pending is None for module in async_engine.module.modules()
                               if isinstance(module, AutoEPMoELayer))
        finally:
            caller_stream.wait_stream(training_stream)
            training_stream.synchronize()
