# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP expert activation registry: the built-in forms, registration, and the detection-time check.

Everything here runs on CPU in one process, like test_autoep_unit.py.
"""

import copy
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.auto_ep import AutoEP
from deepspeed.module_inject.auto_ep_config import (
    AutoEPConfig,
    PRESET_MODELS,
    parse_autoep_config,
    validate_autoep_config,
)
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.ops.triton_ops import is_triton_available
from deepspeed.module_inject.auto_ep_presets.registry import resolve_preset_candidates
from deepspeed.moe.ep_experts import (
    EXPERT_ACTIVATIONS,
    ExpertWeightGradSlot,
    GroupedExperts,
    apply_expert_activation,
    apply_packed_expert_activation,
    register_expert_activation,
)
from unit.v1.moe.autoep_test_utils import MockMoETransformer

MOE_PATTERN = r"model\.layers\.\d+\.mlp"


def _runtime_config(**kwargs):
    kwargs.setdefault("enabled", True)
    kwargs.setdefault("autoep_size", 1)
    kwargs.setdefault("use_grouped_mm", False)
    return AutoEPConfig(**kwargs)


def _custom_pattern_model(**expert_attrs):
    """A mock MoE model whose model_type has no preset, so AutoEP takes the custom-pattern path."""
    model = MockMoETransformer(num_layers=1, num_experts=4, moe_every_n=1)
    model.config.model_type = "no_preset_for_this_type"
    for name, value in expert_attrs.items():
        setattr(model.model.layers[0].mlp.experts, name, value)
    return model


def _written_out(gate, up, activation, alpha, limit):
    """Each built-in form as the model code writes it, independent of the registry."""
    if activation == "swiglu":
        return F.silu(gate) * up
    if activation == "geglu_tanh":
        return F.gelu(gate, approximate="tanh") * up
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    if activation == "swiglu_clamped":
        return F.silu(gate) * up
    if activation == "swiglu_oai":
        return (up + 1.0) * gate * torch.sigmoid(alpha * gate)
    raise AssertionError(f"{activation}: add it here when adding a built-in form")


class TestApplyExpertActivation:

    def test_swiglu_oai_known_values(self):
        # gate=10 is clamped to 7, up=-9 is clamped to -7:  (-7 + 1) * 7 * sigmoid(7 * 1.702)
        out = apply_expert_activation(torch.tensor([10.0]), torch.tensor([-9.0]), "swiglu_oai")
        assert out.item() == pytest.approx(-42.0 / (1.0 + math.exp(-7.0 * 1.702)), rel=1e-6)

        # gate is clamped from above only: gate=-10 stays -10, up=0.5 is inside the limit
        out = apply_expert_activation(torch.tensor([-10.0]), torch.tensor([0.5]), "swiglu_oai")
        assert out.item() == pytest.approx(1.5 * -10.0 / (1.0 + math.exp(10.0 * 1.702)), rel=1e-6)

        # alpha and limit are honoured
        out = apply_expert_activation(torch.tensor([3.0]), torch.tensor([3.0]), "swiglu_oai", alpha=1.0, limit=2.0)
        assert out.item() == pytest.approx(3.0 * 2.0 / (1.0 + math.exp(-2.0)), rel=1e-6)

    def test_swiglu_clamped_known_values(self):
        # gate=12 is clamped to 10, up=-11 is clamped to -10:  silu(10) * -10
        out = apply_expert_activation(torch.tensor([12.0]), torch.tensor([-11.0]), "swiglu_clamped", limit=10.0)
        assert out.item() == pytest.approx(-100.0 / (1.0 + math.exp(-10.0)), rel=1e-6)

        # inside the limit it is plain SwiGLU
        gate, up = torch.randn(256), torch.randn(256)
        torch.testing.assert_close(apply_expert_activation(gate, up, "swiglu_clamped", limit=10.0), F.silu(gate) * up)

    def test_geglu_tanh_known_values(self):
        # tanh-approximated GELU: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))), times up
        x = 1.5
        gelu = 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
        out = apply_expert_activation(torch.tensor([x]), torch.tensor([2.0]), "geglu_tanh")
        assert out.item() == pytest.approx(2.0 * gelu, rel=1e-6)

    def test_the_forms_are_different_functions(self):
        gate, up = 4 * torch.randn(4096), 4 * torch.randn(4096)
        outputs = {name: apply_expert_activation(gate, up, name, fused=False) for name in EXPERT_ACTIVATIONS}
        torch.testing.assert_close(outputs["swiglu"], F.silu(gate) * up)
        names = sorted(outputs)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                assert (outputs[first] - outputs[second]).abs().mean() > 0.1, (first, second)

    def test_matches_the_transformers_experts(self):
        # Against the model code itself, where the installed Transformers has it.
        gate_up = 6 * torch.randn(64, 2 * 32)
        gate, up = gate_up.chunk(2, dim=-1)
        checked = 0
        for module_name, class_name, activation in (
            ("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl", "MiniMaxM3VLExperts", "swiglu_oai"),
            ("transformers.models.deepseek_v4.modeling_deepseek_v4", "DeepseekV4Experts", "swiglu_clamped"),
        ):
            try:
                experts_class = getattr(__import__(module_name, fromlist=[class_name]), class_name)
            except (ImportError, AttributeError):
                continue
            stub = SimpleNamespace(swiglu_alpha=1.702, swiglu_limit=5.0, limit=5.0, act_fn=F.silu)
            want = experts_class._apply_gate(stub, gate_up)
            got = apply_expert_activation(gate, up, activation, alpha=1.702, limit=5.0, fused=False)
            torch.testing.assert_close(got, want)
            checked += 1

        # The plain gated families go through transformers' default gate with the config's act_fn.
        try:
            from transformers.activations import ACT2FN
            from transformers.integrations.moe import _default_apply_gate
        except ImportError:
            _default_apply_gate = None
        if _default_apply_gate is not None:
            for hidden_act, activation in (("silu", "swiglu"), ("gelu_pytorch_tanh", "geglu_tanh")):
                want = _default_apply_gate(SimpleNamespace(act_fn=ACT2FN[hidden_act]), gate_up)
                torch.testing.assert_close(apply_expert_activation(gate, up, activation, fused=False), want)
                checked += 1
        if not checked:
            pytest.skip("installed Transformers has none of the expert classes this test checks against")

    def test_unknown_activation_rejected(self):
        with pytest.raises(ValueError, match="unknown expert activation"):
            apply_expert_activation(torch.zeros(1), torch.zeros(1), "gelu")
        with pytest.raises(ValueError, match="unknown expert activation"):
            GroupedExperts(dim=8, hidden_dim=16, num_experts=2, use_grouped_mm=False, activation="gelu")
        with pytest.raises(ValueError, match="expert_activation must be one of"):
            validate_autoep_config(AutoEPConfig(enabled=True, autoep_size=1, expert_activation="gelu"), 1, 1, 1, 1)

    def test_registered_form_is_selectable_everywhere(self):
        # A form registered by a user (or a future preset module) must work wherever a built-in does:
        # the activation call, the expert layer, config validation, and a custom preset.
        name = "test_reglu"
        register_expert_activation(name, lambda gate, up, alpha, limit: F.relu(gate) * up, gate_fn=F.relu)
        try:
            gate, up = torch.randn(32), torch.randn(32)
            torch.testing.assert_close(apply_expert_activation(gate, up, name), F.relu(gate) * up)
            experts = GroupedExperts(dim=8, hidden_dim=16, num_experts=2, use_grouped_mm=False, activation=name)
            assert experts.activation == name
            validate_autoep_config(AutoEPConfig(enabled=True, autoep_size=1, expert_activation=name), 1, 1, 1, 1)
            config = parse_autoep_config({
                "enabled": True,
                "autoep_size": 1,
                "moe_layer_pattern": MOE_PATTERN,
                "expert_activation": name
            })
            [(_, preset)] = resolve_preset_candidates(config, None)
            assert preset.expert_activation == name
            with pytest.raises(ValueError, match="already registered"):
                register_expert_activation(name, lambda gate, up, alpha, limit: gate * up)
        finally:
            EXPERT_ACTIVATIONS.pop(name)


class TestGroupedExpertsActivation:

    @pytest.mark.parametrize("activation", tuple(EXPERT_ACTIVATIONS))
    def test_for_loop_matches_dense_reference(self, activation):
        torch.manual_seed(0)
        experts = GroupedExperts(dim=16,
                                 hidden_dim=32,
                                 num_experts=4,
                                 use_grouped_mm=False,
                                 activation=activation,
                                 activation_alpha=1.5,
                                 activation_limit=1.0)
        for weight in (experts.w1, experts.w2, experts.w3):
            nn.init.normal_(weight, std=0.5)
        x = torch.randn(10, 16, requires_grad=True)
        counts = torch.tensor([1, 4, 0, 5])
        out = experts(x, counts)

        x_ref = x.detach().clone().requires_grad_(True)
        chunks = []
        for expert_idx, tokens in enumerate(torch.split(x_ref, counts.tolist())):
            gate, up = tokens @ experts.w1[expert_idx].T, tokens @ experts.w3[expert_idx].T
            chunks.append(_written_out(gate, up, activation, 1.5, 1.0) @ experts.w2[expert_idx].T)
        ref = torch.cat(chunks)

        torch.testing.assert_close(out, ref)
        out.sum().backward()
        ref.sum().backward()
        torch.testing.assert_close(x.grad, x_ref.grad)

    @pytest.mark.parametrize("path", ["grouped_mm", "triton_grouped_mm", "fused_gate_up"])
    @pytest.mark.parametrize("activation", tuple(EXPERT_ACTIVATIONS))
    def test_grouped_gemm_paths_match_the_for_loop(self, activation, path):
        # The grouped-GEMM paths hand the activation the packed [rows, ffn] tensors and, for swiglu,
        # the fused Triton kernel; the for-loop path hands it per-expert slices in plain PyTorch. The
        # three paths must compute the same function, forward and backward. Runs on the accelerator.
        if get_accelerator().device_name() == "cpu":
            pytest.skip("needs an accelerator")
        if path in ("grouped_mm", "fused_gate_up") and not hasattr(torch, "_grouped_mm"):
            pytest.skip("this PyTorch build has no torch._grouped_mm")
        device = get_accelerator().current_device_name()
        torch.manual_seed(0)
        shape = dict(dim=64, hidden_dim=128, num_experts=4)
        act = dict(activation=activation, activation_alpha=1.5, activation_limit=1.0)
        loop = GroupedExperts(use_grouped_mm=False, **shape, **act)
        for weight in (loop.w1, loop.w2, loop.w3):
            nn.init.normal_(weight, std=0.1)  # pre-activations of order one: the clamps engage on ~1/5 of them
        grouped = GroupedExperts(use_grouped_mm=True,
                                 disable_triton_grouped_mm=True,
                                 gate_up_impl="fused" if path == "fused_gate_up" else "separate",
                                 **shape,
                                 **act)
        grouped.load_state_dict(loop.state_dict())
        grouped.use_triton_grouped_mm = path == "triton_grouped_mm"
        loop.to(device=device, dtype=torch.bfloat16)
        grouped.to(device=device, dtype=torch.bfloat16)

        counts = torch.tensor([16, 32, 16, 64], device=device)  # aligned rows, no empty expert
        x = torch.randn(int(counts.sum()), 64, device=device, dtype=torch.bfloat16)
        x_loop = x.clone().requires_grad_(True)
        x_grouped = x.clone().requires_grad_(True)
        out_loop = loop(x_loop, counts)
        out_grouped = grouped(x_grouped, counts)
        grad_out = torch.randn_like(out_loop)  # a dense upstream gradient, as training produces
        out_loop.backward(grad_out)
        out_grouped.backward(grad_out)

        # The paths differ only in bf16 rounding: the fused kernel rounds once after fp32 math, the
        # eager ops once per op. A matmul then sums the rounded values, so the absolute error of an
        # output or gradient scales with the largest magnitude in that tensor, not with each element.
        def assert_same(got, want):
            torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2 * want.abs().max().item())

        assert_same(out_grouped, out_loop)
        assert_same(x_grouped.grad, x_loop.grad)
        for name in ("w1", "w2", "w3"):
            assert_same(getattr(grouped, name).grad, getattr(loop, name).grad)

    def test_for_loop_path_does_not_need_the_triton_kernel(self, monkeypatch):
        # The for-loop path is the reference path and runs on CPU tensors. It must not reach the
        # Triton kernel, which exists whenever Triton is installed.
        import deepspeed.ops.triton_ops.swiglu_triton as swiglu_triton

        def _fail(*args, **kwargs):
            raise AssertionError("for-loop path called the Triton swiglu kernel")

        monkeypatch.setattr(swiglu_triton, "swiglu", _fail)
        experts = GroupedExperts(dim=8, hidden_dim=16, num_experts=2, use_grouped_mm=False)
        for weight in (experts.w1, experts.w2, experts.w3):
            nn.init.normal_(weight, std=0.02)
        assert experts(torch.randn(4, 8), torch.tensor([2, 2])).shape == (4, 8)


class TestFusedGateUp:
    """gate_up_impl="fused": one grouped GEMM over the concatenated gate and up weights."""

    @pytest.mark.parametrize("activation", tuple(EXPERT_ACTIVATIONS))
    def test_packed_activation_matches_the_separate_halves(self, activation):
        # Same elementwise math on views of the two halves. On CPU, a strided view can take a different
        # vectorization path than a contiguous tensor, so agreement is to rounding rather than bitwise.
        torch.manual_seed(0)
        gate = torch.randn(6, 5, requires_grad=True)
        up = torch.randn(6, 5, requires_grad=True)
        gate_up = torch.cat([gate, up], dim=-1).detach().requires_grad_(True)
        expected = apply_expert_activation(gate, up, activation, 1.5, 1.0, fused=False)
        packed = apply_packed_expert_activation(gate_up, activation, 1.5, 1.0, fused=False)

        torch.testing.assert_close(packed, expected)
        grad = torch.randn_like(expected)
        expected.backward(grad)
        packed.backward(grad)
        torch.testing.assert_close(gate_up.grad, torch.cat([gate.grad, up.grad], dim=-1))

    def test_only_swiglu_has_a_packed_kernel(self):
        # The other forms run their plain expression on views of the two halves, so none of them needs one.
        assert [name for name, entry in EXPERT_ACTIVATIONS.items() if entry.packed_fused_fn] == ["swiglu"]

    def test_default_keeps_the_separate_projections(self):
        assert parse_autoep_config({"enabled": True}).gate_up_impl == "separate"
        assert GroupedExperts(dim=8, hidden_dim=16, num_experts=2, use_grouped_mm=False).gate_up_impl == "separate"
        fused = parse_autoep_config({"enabled": True, "gate_up_impl": "fused"})
        assert fused.gate_up_impl == "fused"

    def test_unsupported_settings_are_rejected(self, monkeypatch):

        def validate(**settings):
            validate_autoep_config(parse_autoep_config({"enabled": True, **settings}), 1, 1, 1, 1)

        validate(gate_up_impl="fused")
        with pytest.raises(ValueError, match="gate_up_impl must be one of"):
            validate(gate_up_impl="concat")
        with pytest.raises(ValueError, match="sequential expert loop"):
            validate(gate_up_impl="fused", use_grouped_mm=False)
        with pytest.raises(ValueError, match="gate_up_impl must be"):
            GroupedExperts(dim=8, hidden_dim=16, num_experts=2, use_grouped_mm=False, gate_up_impl="concat")
        with pytest.raises(ValueError, match="sequential expert loop"):
            GroupedExperts(dim=8, hidden_dim=16, num_experts=2, use_grouped_mm=False, gate_up_impl="fused")
        # Devices below sm90 auto-select the Triton grouped GEMM, which has no fused form.
        monkeypatch.setattr(type(get_accelerator()), "prefer_triton_grouped_mm", lambda self: True)
        with pytest.raises(ValueError, match="disable_triton_grouped_mm"):
            GroupedExperts(dim=8, hidden_dim=16, num_experts=2, use_grouped_mm=True, gate_up_impl="fused")
        assert GroupedExperts(dim=8,
                              hidden_dim=16,
                              num_experts=2,
                              use_grouped_mm=True,
                              disable_triton_grouped_mm=True,
                              gate_up_impl="fused").gate_up_impl == "fused"

    def test_overlap_weight_grad_settings_are_validated(self, monkeypatch):

        def validate(zero_stage=0, **settings):
            base = {"enabled": True, "autoep_size": 2, "comm_backend": "deepep", "comm_max_tokens_per_rank": 64}
            config = parse_autoep_config({**base, **settings})
            validate_autoep_config(config, 2, 1, 1, 1, zero_stage=zero_stage)

        assert parse_autoep_config({"enabled": True}).overlap_weight_grad is False
        validate(overlap_weight_grad=True)
        with pytest.raises(ValueError, match="must be true or false"):
            validate(overlap_weight_grad="yes")
        with pytest.raises(ValueError, match='Set comm_backend="deepep"'):
            validate(overlap_weight_grad=True, comm_backend="comm")
        with pytest.raises(ValueError, match="sequential expert loop"):
            validate(overlap_weight_grad=True, use_grouped_mm=False)
        with pytest.raises(ValueError, match="ZeRO stage 3"):
            validate(overlap_weight_grad=True, zero_stage=3)
        monkeypatch.delattr(torch._C, "_set_sm_carveout_experimental")
        with pytest.raises(ValueError, match="lacks"):
            validate(overlap_weight_grad=True)

    def test_overlap_weight_grad_rejects_the_triton_grouped_gemm(self, monkeypatch):
        monkeypatch.setattr(type(get_accelerator()), "prefer_triton_grouped_mm", lambda self: True)
        model = _custom_pattern_model()
        auto_ep = AutoEP(model,
                         _runtime_config(moe_layer_pattern=MOE_PATTERN, use_grouped_mm=True, overlap_weight_grad=True))
        [spec] = auto_ep.ep_parser()
        with pytest.raises(ValueError, match="disable_triton_grouped_mm=true"):
            auto_ep.replace_moe_layer(spec, ep_size=1, ep_rank=0)

    def test_config_reaches_the_replaced_experts(self):
        if not hasattr(torch, "_grouped_mm"):
            pytest.skip("this PyTorch build has no torch._grouped_mm")
        model = _custom_pattern_model()
        config = _runtime_config(moe_layer_pattern=MOE_PATTERN,
                                 use_grouped_mm=True,
                                 disable_triton_grouped_mm=True,
                                 gate_up_impl="fused")
        auto_ep = AutoEP(model, config)
        [spec] = auto_ep.ep_parser()
        auto_ep.replace_moe_layer(spec, ep_size=1, ep_rank=0)
        experts = model.model.layers[0].mlp.experts
        assert experts.gate_up_impl == "fused"
        # Parameters, and so checkpoints and optimizer state, keep the separate layout.
        assert [name for name, _ in experts.named_parameters()] == ["w1", "w2", "w3"]

    @pytest.mark.parametrize("activation", tuple(EXPERT_ACTIVATIONS))
    def test_matches_separate_with_skewed_and_empty_experts(self, activation):
        # Forward values and weight gradients are the separate path's; the input gradient sums both projections
        # inside one FP32 GEMM accumulation, so it is held to FP64 instead and must be no less accurate.
        if get_accelerator().device_name() != "cuda" or not hasattr(torch, "_grouped_mm"):
            pytest.skip("needs CUDA with torch._grouped_mm and FP64")
        device = get_accelerator().current_device_name()
        torch.manual_seed(0)
        shape = dict(dim=256, hidden_dim=384, num_experts=4)
        act = dict(activation=activation, activation_alpha=1.5, activation_limit=1.0)
        separate = GroupedExperts(use_grouped_mm=True, disable_triton_grouped_mm=True, **shape, **act)
        for weight in (separate.w1, separate.w2, separate.w3):
            nn.init.normal_(weight, std=0.06)
        fused = GroupedExperts(use_grouped_mm=True,
                               disable_triton_grouped_mm=True,
                               gate_up_impl="fused",
                               **shape,
                               **act)
        fused.load_state_dict(separate.state_dict())
        reference = copy.deepcopy(separate).to(device=device, dtype=torch.float64)
        reference.use_grouped_mm = False
        separate.to(device=device, dtype=torch.bfloat16)
        fused.to(device=device, dtype=torch.bfloat16)

        counts = torch.tensor([37, 0, 291, 72], device=device)
        x = torch.randn(int(counts.sum()), 256, device=device, dtype=torch.bfloat16)
        grad_out = torch.randn(x.shape, device=device, dtype=torch.bfloat16)
        inputs = {name: x.clone().requires_grad_(True) for name in ("separate", "fused")}
        out_separate = separate(inputs["separate"], counts)
        out_fused = fused(inputs["fused"], counts)
        out_separate.backward(grad_out)
        out_fused.backward(grad_out)
        x_reference = x.double().requires_grad_(True)
        reference(x_reference, counts).backward(grad_out.double())

        torch.testing.assert_close(out_fused, out_separate, rtol=0, atol=0)
        for name in ("w1", "w2", "w3"):
            torch.testing.assert_close(getattr(fused, name).grad, getattr(separate, name).grad, rtol=0, atol=0)

        def relative_error(got):
            return ((got.double() - x_reference.grad).norm() / x_reference.grad.norm()).item()

        assert relative_error(inputs["fused"].grad) <= relative_error(inputs["separate"].grad) * 1.05


def _grouped_mm_runs_on(device):
    try:
        torch._grouped_mm(torch.zeros(4, 16, device=device),
                          torch.zeros(1, 16, 8, device=device),
                          offs=torch.tensor([4], dtype=torch.int32, device=device))
    except (AttributeError, RuntimeError, NotImplementedError):
        return False
    return True


class TestDeferredExpertWeightGrad:
    """Weight gradients left in a slot are the plain path's, bitwise; only when they are computed changes."""

    def _compare(self, device, dtype, activation, gate_up_impl, deferred):
        torch.manual_seed(0)
        kwargs = dict(dim=16,
                      hidden_dim=8,
                      num_experts=4,
                      use_grouped_mm=True,
                      disable_triton_grouped_mm=True,
                      activation=activation,
                      activation_alpha=1.5,
                      activation_limit=1.0,
                      gate_up_impl=gate_up_impl)
        plain = GroupedExperts(**kwargs)
        for weight in plain.parameters():
            nn.init.normal_(weight, std=0.4)
        overlapped = GroupedExperts(**kwargs)
        overlapped.load_state_dict(plain.state_dict())
        plain.to(device=device, dtype=dtype)
        overlapped.to(device=device, dtype=dtype)
        counts = torch.tensor([5, 0, 7, 12], device=device)
        x = torch.randn(24, 16, device=device, dtype=dtype)
        grad_out = torch.randn(24, 16, device=device, dtype=dtype)
        x_plain, x_overlapped = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        out_plain = plain(x_plain, counts)
        out_plain.backward(grad_out)
        slot = ExpertWeightGradSlot() if deferred else None
        out_overlapped = overlapped(x_overlapped, counts, slot)
        out_overlapped.backward(grad_out)
        if deferred:
            assert all(weight.grad is None for weight in overlapped.parameters())
            for weight, grad in zip((overlapped.w1, overlapped.w2, overlapped.w3), slot.run()):
                weight.grad = grad

        torch.testing.assert_close(out_overlapped, out_plain, rtol=0, atol=0)
        torch.testing.assert_close(x_overlapped.grad, x_plain.grad, rtol=0, atol=0)
        for name in ("w1", "w2", "w3"):
            torch.testing.assert_close(getattr(overlapped, name).grad, getattr(plain, name).grad, rtol=0, atol=0)

    @pytest.mark.parametrize("deferred", [False, True])
    @pytest.mark.parametrize("gate_up_impl", ["separate", "fused"])
    @pytest.mark.parametrize("activation", tuple(EXPERT_ACTIVATIONS))
    def test_matches_the_plain_path_bitwise_on_cpu(self, activation, gate_up_impl, deferred):
        if not _grouped_mm_runs_on("cpu"):
            pytest.skip("this PyTorch build has no CPU torch._grouped_mm")
        if EXPERT_ACTIVATIONS[activation].fused_fn is not None and is_triton_available():
            # The grouped path's fused kernel is Triton, which reads CUDA tensors only; the CUDA test covers it.
            pytest.skip("with Triton installed, this form's grouped path needs CUDA tensors")
        for dtype in (torch.float32, torch.bfloat16):
            self._compare("cpu", dtype, activation, gate_up_impl, deferred)

    @pytest.mark.parametrize("gate_up_impl", ["separate", "fused"])
    @pytest.mark.parametrize("activation", tuple(EXPERT_ACTIVATIONS))
    def test_matches_the_plain_path_bitwise_with_the_fused_kernels(self, activation, gate_up_impl):
        # On CUDA the swiglu forms run the Triton kernels, whose backward the deferred path calls directly.
        if get_accelerator().device_name() != "cuda" or not hasattr(torch, "_grouped_mm"):
            pytest.skip("needs CUDA with torch._grouped_mm")
        self._compare(get_accelerator().current_device_name(), torch.bfloat16, activation, gate_up_impl, True)

    def test_slot_runs_once_after_the_expert_backward(self):
        slot = ExpertWeightGradSlot()
        with pytest.raises(RuntimeError, match="did not run before"):
            slot.run()
        slot.defer(lambda: (1, 2, 3))
        with pytest.raises(RuntimeError, match="deferred twice"):
            slot.defer(lambda: (1, 2, 3))
        assert slot.run() == (1, 2, 3)
        with pytest.raises(RuntimeError, match="did not run before"):
            slot.run()

    def test_deferring_needs_the_grouped_mm_path(self):
        experts = GroupedExperts(dim=8, hidden_dim=16, num_experts=2, use_grouped_mm=False)
        with pytest.raises(RuntimeError, match="torch._grouped_mm path"):
            experts(torch.randn(4, 8), torch.tensor([2, 2]), ExpertWeightGradSlot())


class TestPresetAndConfig:

    def test_builtin_presets_name_their_form(self):
        # MiniMax-M3 experts use the clamped GPT-OSS form; every other family with a preset gates
        # with silu (transformers hidden_act="silu").
        for name, preset in PRESET_MODELS.items():
            assert preset.expert_activation == ("swiglu_oai" if name == "minimax_m3" else "swiglu"), name

    def test_config_key_reaches_custom_preset_and_overrides_builtin(self):
        config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "moe_layer_pattern": MOE_PATTERN,
            "expert_activation": "swiglu_oai"
        })
        [(name, preset)] = resolve_preset_candidates(config, None)
        assert name == "custom"
        assert preset.expert_activation == "swiglu_oai"

        default = parse_autoep_config({"enabled": True, "autoep_size": 1, "moe_layer_pattern": MOE_PATTERN})
        assert resolve_preset_candidates(default, None)[0][1].expert_activation == "swiglu"

        override = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "preset_model": "mixtral",
            "expert_activation": "swiglu_oai"
        })
        assert resolve_preset_candidates(override, None)[0][1].expert_activation == "swiglu_oai"
        assert PRESET_MODELS["mixtral"].expert_activation == "swiglu"  # the registered preset is untouched


class TestExpertActivationDetection:

    def test_plain_model_is_unchanged(self):
        model = _custom_pattern_model()
        auto_ep = AutoEP(model, _runtime_config(moe_layer_pattern=MOE_PATTERN))
        [spec] = auto_ep.ep_parser()
        assert spec.expert_activation == "swiglu"
        auto_ep.replace_moe_layer(spec, ep_size=1, ep_rank=0)
        assert model.model.layers[0].mlp.experts.activation == "swiglu"

    @pytest.mark.parametrize("attrs", [{"swiglu_limit": 7.0, "swiglu_alpha": 1.702}, {"limit": 7.0, "alpha": 1.702}])
    def test_clamped_model_with_plain_preset_is_rejected(self, attrs):
        auto_ep = AutoEP(_custom_pattern_model(**attrs), _runtime_config(moe_layer_pattern=MOE_PATTERN))
        with pytest.raises(ValueError, match="clamp their SwiGLU"):
            auto_ep.ep_parser()

    def test_limit_on_the_model_config_is_detected(self):
        model = _custom_pattern_model()
        model.config.swiglu_limit = 7.0
        with pytest.raises(ValueError, match="clamp their SwiGLU"):
            AutoEP(model, _runtime_config(moe_layer_pattern=MOE_PATTERN)).ep_parser()

    def test_explicit_choice_is_kept(self):
        model = _custom_pattern_model(swiglu_limit=7.0)
        auto_ep = AutoEP(model, _runtime_config(moe_layer_pattern=MOE_PATTERN, expert_activation="swiglu"))
        [spec] = auto_ep.ep_parser()
        assert spec.expert_activation == "swiglu"

    def test_alpha_and_limit_come_from_the_model(self):
        model = _custom_pattern_model(swiglu_limit=5.0, swiglu_alpha=1.25)
        auto_ep = AutoEP(model, _runtime_config(moe_layer_pattern=MOE_PATTERN, expert_activation="swiglu_oai"))
        [spec] = auto_ep.ep_parser()
        assert (spec.expert_activation, spec.expert_activation_alpha, spec.expert_activation_limit) == ("swiglu_oai",
                                                                                                        1.25, 5.0)
        auto_ep.replace_moe_layer(spec, ep_size=1, ep_rank=0)
        experts = model.model.layers[0].mlp.experts
        assert (experts.activation, experts.activation_alpha, experts.activation_limit) == ("swiglu_oai", 1.25, 5.0)

    def test_limit_without_alpha_as_in_deepseek_v4(self):
        # DeepSeek-V4: `limit` on the experts module, `swiglu_limit` on the config, no alpha anywhere.
        def build():
            model = _custom_pattern_model(limit=10.0)
            model.config.swiglu_limit = 10.0
            return model

        with pytest.raises(ValueError, match="swiglu_clamped"):
            AutoEP(build(), _runtime_config(moe_layer_pattern=MOE_PATTERN)).ep_parser()

        auto_ep = AutoEP(build(), _runtime_config(moe_layer_pattern=MOE_PATTERN, expert_activation="swiglu_clamped"))
        [spec] = auto_ep.ep_parser()
        assert (spec.expert_activation, spec.expert_activation_limit) == ("swiglu_clamped", 10.0)

    def test_gelu_gated_experts_with_plain_preset_are_rejected(self):
        # Gemma-4 experts carry act_fn = gelu_pytorch_tanh, and no clamp. With the plain preset the
        # replacement would compute silu(gate) * up.
        model = _custom_pattern_model(act_fn=nn.GELU(approximate="tanh"))
        with pytest.raises(ValueError, match="matches geglu_tanh"):
            AutoEP(model, _runtime_config(moe_layer_pattern=MOE_PATTERN)).ep_parser()

        auto_ep = AutoEP(model, _runtime_config(moe_layer_pattern=MOE_PATTERN, expert_activation="geglu_tanh"))
        [spec] = auto_ep.ep_parser()
        assert spec.expert_activation == "geglu_tanh"
        auto_ep.replace_moe_layer(spec, ep_size=1, ep_rank=0)
        assert model.model.layers[0].mlp.experts.activation == "geglu_tanh"

    def test_silu_gated_experts_pass_the_check(self):
        model = _custom_pattern_model(act_fn=nn.SiLU())
        [spec] = AutoEP(model, _runtime_config(moe_layer_pattern=MOE_PATTERN)).ep_parser()
        assert spec.expert_activation == "swiglu"

    def test_unregistered_act_fn_is_rejected_unless_chosen(self):
        model = _custom_pattern_model(act_fn=nn.Tanh())
        with pytest.raises(ValueError, match="none of the registered forms"):
            AutoEP(model, _runtime_config(moe_layer_pattern=MOE_PATTERN)).ep_parser()

        auto_ep = AutoEP(model, _runtime_config(moe_layer_pattern=MOE_PATTERN, expert_activation="swiglu"))
        [spec] = auto_ep.ep_parser()
        assert spec.expert_activation == "swiglu"


class TestMiniMaxM3Block:
    """A real MiniMaxM3VLSparseMoeBlock through the preset (found from the model type, no config keys)
    and through the custom pattern, which is how a clamped-form model runs before it has a preset."""

    def _tiny_model(self, std):
        modeling = pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl",
                                       reason="MiniMax-M3 needs Transformers >= 5.15")
        from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig

        torch.manual_seed(0)
        config = MiniMaxM3VLTextConfig(hidden_size=64,
                                       intermediate_size=128,
                                       shared_intermediate_size=128,
                                       dense_intermediate_size=128,
                                       num_hidden_layers=1,
                                       num_attention_heads=4,
                                       num_key_value_heads=2,
                                       head_dim=16,
                                       num_local_experts=8,
                                       num_experts_per_tok=2,
                                       vocab_size=128,
                                       bos_token_id=1,
                                       eos_token_id=2)

        class Layer(nn.Module):

            def __init__(self):
                super().__init__()
                self.mlp = modeling.MiniMaxM3VLSparseMoeBlock(config)

        class Tiny(nn.Module):

            def __init__(self):
                super().__init__()
                self.config = config
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([Layer()])

        model = Tiny()
        for param in model.parameters():  # the experts are created with torch.empty
            nn.init.normal_(param, std=std)
        return model

    def _model_for(self, route, std):
        model = self._tiny_model(std)
        if route == "custom_pattern":
            # A model type with a preset is matched to it before the custom pattern is considered, so
            # the custom route is only reachable for a model type AutoEP does not know yet.
            model.config.model_type = "no_preset_for_this_type"
        return model

    def _config(self, route, **kwargs):
        if route == "preset":
            return _runtime_config(**kwargs)
        return _runtime_config(moe_layer_pattern=MOE_PATTERN,
                               score_func="sigmoid",
                               score_apply="post",
                               route_norm=True,
                               has_shared_experts=True,
                               shared_experts_pattern="shared_experts",
                               **kwargs)

    def test_custom_pattern_without_the_key_is_refused(self):
        auto_ep = AutoEP(self._model_for("custom_pattern", 0.02), self._config("custom_pattern"))
        with pytest.raises(ValueError, match="clamp their SwiGLU"):
            auto_ep.ep_parser()

    @pytest.mark.parametrize("std", [0.02, 0.5])  # 0.5 pushes pre-activations past the clamp limit
    @pytest.mark.parametrize("route", ["preset", "custom_pattern"])
    def test_replaced_block_matches_hf_block(self, route, std):
        model = self._model_for(route, std)
        reference = copy.deepcopy(model.model.layers[0].mlp)

        keys = {} if route == "preset" else {"expert_activation": "swiglu_oai"}
        auto_ep = AutoEP(model, self._config(route, **keys))
        [spec] = auto_ep.ep_parser()
        assert spec.model_family == ("minimax_m3" if route == "preset" else "custom")
        assert (spec.expert_activation, spec.expert_activation_alpha, spec.expert_activation_limit) == ("swiglu_oai",
                                                                                                        1.702, 7.0)
        auto_ep.replace_moe_layer(spec, ep_size=1, ep_rank=0)
        replaced = model.model.layers[0].mlp
        assert isinstance(replaced, AutoEPMoELayer)

        x = torch.randn(2, 16, model.config.hidden_size, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        out, out_ref = replaced(x), reference(x_ref)
        out = out[0] if isinstance(out, tuple) else out
        out_ref = out_ref[0] if isinstance(out_ref, tuple) else out_ref

        torch.testing.assert_close(out, out_ref, rtol=1e-4, atol=1e-5 * out_ref.abs().mean().item())
        out.sum().backward()
        out_ref.sum().backward()
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-4, atol=1e-5 * x_ref.grad.abs().mean().item())
