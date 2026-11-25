# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Subchannel (block-scaled) FP8/INT8 training: numerics, module conversion and engine integration.

Accuracy is measured against FP64 matmuls; the Triton kernels (``cuda``) and the PyTorch reference (``cpu``)
must both meet the same bounds.
"""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.ep_experts import GroupedExperts
from deepspeed.runtime.quantization.config import (GemmQuantizationConfig, SubchannelQuantizedMatmulConfig,
                                                   get_quantization_config)
from deepspeed.runtime.quantization.subchannel import (SubchannelQuantizedLinear, convert_to_subchannel_quantization,
                                                       subchannel_grouped_matmul, subchannel_linear)
from unit.common import DistributedTest


def _require_fp8_kernels():
    if "e4m3" not in get_accelerator().supported_subchannel_quantization_formats():
        pytest.skip("needs the FP8 Triton kernels (CUDA compute capability 8.9+)")


@pytest.fixture(params=["cpu", "accelerator"])
def device(request):
    """CPU tensors run the PyTorch reference; the accelerator runs the fused kernels.

    Capability is probed here rather than at import so that collecting this module does not initialize the device.
    """
    if request.param == "cpu":
        return "cpu"
    _require_fp8_kernels()
    return get_accelerator().device_name()


# Relative Frobenius error of a quantized GEMM against FP64, for Gaussian operands. The rounding step relative to
# a value is 2^-3 (e4m3), 2^-2 (e5m2) and about amax/127 (int8); measured errors sit near 60-75% of these bounds.
ERROR_BOUND = {"e4m3": 0.05, "e5m2": 0.10, "int8": 0.015, "none": 0.005}


def _recipe(fmt="e4m3", hadamard_size=0, stochastic=False, block_size=128, **overrides):
    gemm = dict(format=fmt, hadamard_size=hadamard_size, stochastic_rounding=stochastic, block_size=block_size)
    return SubchannelQuantizedMatmulConfig(enabled=True, forward=gemm, dx=gemm, dw=gemm, **overrides)


def _rel_error(actual, expected):
    return float((actual.double() - expected).norm() / expected.norm())


def _group_ranges(sizes):
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    return [(start, start + size) for start, size in zip(starts, sizes)]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gemm", [
    {
        "block_size": 48
    },
    {
        "hadamard_size": 24
    },
    {
        "block_size": 64,
        "hadamard_size": 128
    },
    {
        "format": "fp4"
    },
    {
        "block_size": 0,
        "format": "int8"
    },
    {
        "block_size": 0,
        "hadamard_size": 64
    },
    {
        "stochastic": True
    },
])
def test_invalid_gemm_config_is_rejected(gemm):
    with pytest.raises(ValueError):
        GemmQuantizationConfig(**gemm)


def test_quantization_section_defaults_to_disabled():
    assert not get_quantization_config({}).subchannel.enabled
    config = get_quantization_config({"quantization": {"subchannel": {"enabled": True, "dx": {"format": "int8"}}}})
    assert config.subchannel.enabled
    assert config.subchannel.dx.format == "int8"
    # Unconfigured GEMMs keep their defaults: nearest rounding forward, stochastic rounding for gradients.
    assert not config.subchannel.forward.stochastic_rounding
    assert config.subchannel.dw.stochastic_rounding


@pytest.mark.parametrize("name,gemm_format", [("int8", "int8"), ("fp8", "e4m3")])
def test_named_subchannel_recipe_matches_flat_config(name, gemm_format):
    named = {"enabled": True, "modules": ["*.experts"], "dx": {"block_size": 32}}
    flat = {
        "enabled": True,
        "modules": ["*.experts"],
        "forward": {
            "format": gemm_format
        },
        "dx": {
            "format": gemm_format,
            "block_size": 32,
            "stochastic_rounding": True
        },
        "dw": {
            "format": gemm_format,
            "stochastic_rounding": True
        },
    }
    assert get_quantization_config({
        "quantization": {
            "subchannel": {
                name: named
            }
        }
    }).subchannel == get_quantization_config({
        "quantization": {
            "subchannel": flat
        }
    }).subchannel


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt,hadamard_size,block_size", [("e4m3", 0, 128), ("e4m3", 64, 128), ("e5m2", 0, 128),
                                                          ("int8", 0, 128), ("int8", 128, 128), ("none", 0, 128),
                                                          ("e4m3", 0, 0), ("e5m2", 0, 0)])
def test_grouped_matmul_matches_fp64(device, fmt, hadamard_size, block_size):
    """Forward, input and weight gradients of uneven expert groups, including an empty expert and padding rows.

    The experts' token spans start and end off every kernel tile boundary; one fits inside a tile and one covers
    several.
    """
    sizes = [37, 0, 419, 70]
    padding_rows = 5
    m, n, k = sum(sizes) + padding_rows, 192, 320
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(m, k, generator=generator).to(device, torch.bfloat16).requires_grad_()
    w = torch.randn(len(sizes), n, k, generator=generator).to(device, torch.bfloat16).requires_grad_()
    grad_out = torch.randn(m, n, generator=generator).to(device, torch.bfloat16)
    group_ends = torch.cumsum(torch.tensor(sizes), 0).to(device, torch.int32)

    y = subchannel_grouped_matmul(x, w, group_ends, _recipe(fmt, hadamard_size, block_size=block_size))
    y.backward(grad_out)

    xd, wd, gd = x.detach().double(), w.detach().double(), grad_out.double()
    y_ref, dx_ref, dw_ref = torch.zeros_like(y, dtype=torch.float64), torch.zeros_like(xd), torch.zeros_like(wd)
    for expert, (start, end) in enumerate(_group_ranges(sizes)):
        y_ref[start:end] = xd[start:end] @ wd[expert].T
        dx_ref[start:end] = gd[start:end] @ wd[expert]
        dw_ref[expert] = gd[start:end].T @ xd[start:end]

    bound = ERROR_BOUND[fmt]
    assert _rel_error(y, y_ref) < bound
    assert _rel_error(x.grad, dx_ref) < bound
    assert _rel_error(w.grad, dw_ref) < bound
    # No expert owns the padding rows, and an expert without tokens has no weight gradient.
    assert torch.all(y[-padding_rows:] == 0)
    assert torch.all(x.grad[-padding_rows:] == 0)
    assert torch.all(w.grad[1] == 0)


@pytest.mark.parametrize("block_size", [128, 0])
def test_linear_matches_fp64(device, block_size):
    """A 3D input whose feature sizes are not multiples of the block size, with a bias."""
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(3, 50, 200, generator=generator).to(device).requires_grad_()
    w = torch.randn(72, 200, generator=generator).to(device).requires_grad_()
    b = torch.randn(72, generator=generator).to(device).requires_grad_()
    grad_out = torch.randn(3, 50, 72, generator=generator).to(device)

    subchannel_linear(x, w, b, _recipe("e4m3", block_size=block_size)).backward(grad_out)
    grads = [x.grad, w.grad, b.grad]

    refs = [t.detach().double().requires_grad_() for t in (x, w, b)]
    F.linear(*refs).backward(grad_out.double())
    for grad, ref in zip(grads, refs):
        assert _rel_error(grad, ref.grad) < ERROR_BOUND["e4m3"]


@pytest.mark.parametrize("dw", [{"format": "none"}, {"block_size": 0}, {"block_size": 128}])
def test_single_expert_ignores_alignment_padding(dw):
    """One local expert whose group ends before the rows AutoEP pads the buffer with.

    No expert owns those rows, so nothing they hold may reach an output or a gradient; NaN stands in for
    whatever they hold. The expert MLP must give the same result as on the buffer cut off at the group's end.
    """
    _require_fp8_kernels()
    device = get_accelerator().current_device_name()
    torch.manual_seed(6)
    experts = GroupedExperts(64, 128, 1, use_grouped_mm=False)
    for weight in experts.parameters():
        nn.init.normal_(weight, std=0.1)
    layer = nn.Module()
    layer.experts = experts.to(device)
    forward = {"stochastic_rounding": False}
    dw = dict(dw, stochastic_rounding=False)
    recipe = SubchannelQuantizedMatmulConfig(enabled=True, modules=["*"], forward=forward, dx=forward, dw=dw)
    convert_to_subchannel_quantization(layer, recipe)

    group_end, padding_rows = 16, 8
    counts = torch.tensor([group_end], device=device)
    rows = torch.randn(group_end, 64, device=device)
    grad_out = torch.randn(group_end, 64, device=device)
    padding = torch.full((padding_rows, 64), float("nan"), device=device)

    def output_and_grads(x, grad):
        experts.zero_grad()
        x = x.clone().requires_grad_()
        y = experts(x, counts)
        y.backward(grad)
        return [y.detach()[:group_end], x.grad[:group_end]] + [weight.grad.clone() for weight in experts.parameters()]

    expected = output_and_grads(rows, grad_out)
    padded = output_and_grads(torch.cat([rows, padding]), torch.cat([grad_out, padding]))
    for actual, reference in zip(padded, expected):
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("fmt,block_size", [("e4m3", 128), ("e4m3", 0), ("int8", 128)])
def test_nan_reaches_outputs_and_gradients(device, fmt, block_size):
    """A NaN operand must come out as NaN: stored as a finite value it would hide a diverged run."""
    x = torch.ones(4, 128, device=device)
    x[1, 5] = float("nan")
    x.requires_grad_()
    w = torch.ones(8, 128, device=device, requires_grad=True)
    grad_out = torch.ones(4, 8, device=device)
    grad_out[2, 3] = float("nan")

    y = subchannel_linear(x, w, None, _recipe(fmt, block_size=block_size))
    y.backward(grad_out)

    clean_rows = [0, 2, 3]
    assert torch.isnan(y[1]).all()
    assert not torch.isnan(y[clean_rows]).any()
    # dx = dy W: only token 2 has a NaN output gradient.
    assert torch.isnan(x.grad[2]).all()
    assert not torch.isnan(x.grad[[0, 1, 3]]).any()
    # dw = dy^T x: the NaN of dy sits in output feature 3, the NaN of x in input feature 5.
    assert torch.isnan(w.grad[3]).all()
    assert torch.isnan(w.grad[:, 5]).all()
    clean_inputs = [k for k in range(128) if k != 5]
    assert not torch.isnan(w.grad[[0, 1, 2, 4, 5, 6, 7]][:, clean_inputs]).any()


def test_hadamard_reduces_int8_outlier_error(device):
    """Activation outlier channels inflate an INT8 block scale; rotating the block spreads them out."""
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(256, 512, generator=generator)
    x[:, ::97] *= 100.0
    w = torch.randn(1, 256, 512, generator=generator)
    x, w = x.to(device), w.to(device)
    group_ends = torch.tensor([256], dtype=torch.int32, device=device)
    exact = x.double() @ w[0].double().T

    plain = _rel_error(subchannel_grouped_matmul(x, w, group_ends, _recipe("int8")), exact)
    rotated = _rel_error(subchannel_grouped_matmul(x, w, group_ends, _recipe("int8", hadamard_size=128)), exact)
    assert rotated < plain / 2


@pytest.mark.parametrize("fmt,block_size,qmax,value,nearest", [("e4m3", 128, 448.0, 1.1, 1.125),
                                                               ("e4m3", 0, 448.0, 1.1, 1.125),
                                                               ("int8", 128, 127.0, 10.3, 10.0)])
def test_stochastic_rounding_is_unbiased(device, fmt, block_size, qmax, value, nearest):
    """``value`` lies between two grid points of ``fmt``; nearest rounding is biased, stochastic rounding is not.

    Each row's first element pins the block scale to 1, and an identity weight (exact in every format) makes
    the product return the dequantized activations.
    """
    x = torch.full((256, 128), value, device=device)
    x[:, 0] = qmax
    identity = torch.eye(128, device=device).unsqueeze(0)
    group_ends = torch.tensor([256], dtype=torch.int32, device=device)

    def mean_of_quantized(stochastic):
        gemm = dict(format=fmt, block_size=block_size, stochastic_rounding=stochastic)
        recipe = SubchannelQuantizedMatmulConfig(forward=gemm)
        return torch.stack([subchannel_grouped_matmul(x, identity, group_ends, recipe)[:, 1:]
                            for _ in range(8)]).mean().item()

    assert mean_of_quantized(False) == pytest.approx(nearest)
    assert mean_of_quantized(True) == pytest.approx(value, abs=abs(nearest - value) / 10)


@pytest.mark.parametrize("block_size", [128, 0])
def test_stochastic_rounding_errors_average_out_in_weight_gradient(device, block_size):
    """Averaging independent SR weight gradients converges to the exact one; repeated nearest rounding cannot."""
    sizes = [100, 0, 156]
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(256, 192, generator=generator).to(device).requires_grad_()
    w = torch.randn(len(sizes), 128, 192, generator=generator).to(device).requires_grad_()
    grad_out = torch.randn(256, 128, generator=generator).to(device)
    group_ends = torch.cumsum(torch.tensor(sizes), 0).to(device, torch.int32)
    exact = torch.stack([grad_out[s:e].double().T @ x[s:e].detach().double() for s, e in _group_ranges(sizes)])

    def mean_error(stochastic, trials=32):
        recipe = SubchannelQuantizedMatmulConfig(dw=dict(block_size=block_size, stochastic_rounding=stochastic))
        grads = []
        for _ in range(trials):
            w.grad = None
            subchannel_grouped_matmul(x, w, group_ends, recipe).backward(grad_out)
            grads.append(w.grad.clone())
        return _rel_error(grads[0], exact), _rel_error(torch.stack(grads).mean(0), exact)

    single, averaged = mean_error(stochastic=True)
    assert averaged < single / 3
    single, averaged = mean_error(stochastic=False)
    assert averaged == pytest.approx(single)


# ---------------------------------------------------------------------------
# Module conversion
# ---------------------------------------------------------------------------


class _Block(nn.Module):

    def __init__(self, hidden=64):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, 2, batch_first=True)
        self.up_proj = nn.Linear(hidden, 2 * hidden)
        self.down_proj = nn.Linear(2 * hidden, hidden)
        self.gate = nn.Linear(hidden, 4, bias=False)
        self.lm_head = nn.Linear(hidden, 10)


def test_conversion_keeps_parameters_and_state_dict():
    model = nn.Module()
    model.mlp = _Block()
    state_before = model.state_dict()
    params_before = dict(model.named_parameters())

    converted = convert_to_subchannel_quantization(model, _recipe())

    assert sorted(converted) == ["mlp.down_proj", "mlp.up_proj"]
    assert isinstance(model.mlp.up_proj, SubchannelQuantizedLinear)
    # Router gates and the output head are excluded by default; attention's out_proj never calls forward().
    assert type(model.mlp.gate) is nn.Linear
    assert type(model.mlp.lm_head) is nn.Linear
    assert type(model.mlp.attn.out_proj) is not SubchannelQuantizedLinear
    assert dict(model.named_parameters()) == params_before
    assert model.state_dict().keys() == state_before.keys()


def test_converted_linear_forward_matches_unconverted():
    torch.manual_seed(4)
    model = _Block()
    reference = copy.deepcopy(model)
    convert_to_subchannel_quantization(model, _recipe("none"))
    x = torch.randn(5, 64)
    torch.testing.assert_close(model.down_proj(model.up_proj(x)), reference.down_proj(reference.up_proj(x)))


@pytest.mark.parametrize("block_size", [128, 0])
def test_torch_compile_matches_eager(block_size):
    _require_fp8_kernels()
    from deepspeed.runtime.compiler import is_compile_supported
    if not is_compile_supported():
        pytest.skip("torch.compile is not supported in this environment")
    torch.manual_seed(5)
    device = get_accelerator().current_device_name()
    model = _Block(200).to(device, torch.bfloat16)
    convert_to_subchannel_quantization(model, _recipe(block_size=block_size))
    x = torch.randn(16, 200, device=device, dtype=torch.bfloat16)

    def output_and_weight_grads(run):
        model.zero_grad()
        y = run(x)
        y.float().square().mean().backward()
        return y.detach().clone(), model.up_proj.weight.grad.clone(), model.down_proj.weight.grad.clone()

    def forward(x):
        return model.down_proj(F.gelu(model.up_proj(x)))

    expected = output_and_weight_grads(forward)
    compiled = output_and_weight_grads(torch.compile(forward))
    # Nearest rounding is deterministic, so compiling may not change a single value.
    for actual, reference in zip(compiled, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_conversion_without_matches_fails():
    with pytest.raises(ValueError, match="no nn.Linear"):
        convert_to_subchannel_quantization(_Block(), _recipe(modules=["*does_not_exist*"]))


def test_conversion_rejects_fused_gate_up_experts():
    experts = nn.Module()
    experts.experts = GroupedExperts(64, 128, 4, use_grouped_mm=False)
    experts.experts.gate_up_impl = "fused"
    with pytest.raises(ValueError, match="gate_up_impl"):
        convert_to_subchannel_quantization(experts, _recipe())


def test_conversion_rejects_staged_expert_weight_grads():
    # Staging hands the weight-gradient GEMMs to torch._grouped_mm, which would silently skip quantization.
    layer = nn.Module()
    layer.experts = GroupedExperts(64, 128, 4, use_grouped_mm=False)
    layer.dw_staging = object()
    with pytest.raises(ValueError, match="deferred_dw_flush_period"):
        convert_to_subchannel_quantization(layer, _recipe())


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


class _Regressor(nn.Module):
    """Fits a fixed linear map; feature sizes deliberately not multiples of the block size."""

    def __init__(self, hidden=200, layers=3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])

    def forward(self, x, target):
        for layer in self.layers[:-1]:
            x = F.gelu(layer(x))
        return F.mse_loss(self.layers[-1](x), target)


def _train_regressor(config, steps=100, hidden=200):
    torch.manual_seed(0)
    model = _Regressor(hidden)
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    teacher = torch.randn(hidden, hidden, generator=torch.Generator().manual_seed(1)) / hidden**0.5
    data = torch.Generator().manual_seed(engine.global_rank)
    losses = []
    for _ in range(steps):
        x = torch.randn(16, hidden, generator=data)
        loss = engine(x.to(engine.device, torch.bfloat16), (x @ teacher.T).to(engine.device, torch.bfloat16))
        engine.backward(loss)
        engine.step()
        losses.append(loss.item())
    return engine, losses


def _mean(values):
    return sum(values) / len(values)


class TestSubchannelQuantizedTraining(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("zero_stage,block_size", [(1, 128), (2, 128), (3, 128), (2, 0)])
    def test_dense_training_tracks_bf16(self, zero_stage, block_size):
        _require_fp8_kernels()
        config = {
            "train_micro_batch_size_per_gpu": 16,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 3e-3,
                    "torch_adam": True
                }
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage
            },
        }
        _, baseline = _train_regressor(config)

        gemm = {"block_size": block_size}
        config["quantization"] = {"subchannel": {"enabled": True, "forward": gemm, "dx": gemm, "dw": gemm}}
        engine, quantized = _train_regressor(config)

        assert all(isinstance(layer, SubchannelQuantizedLinear) for layer in engine.module.layers)
        assert _mean(quantized[-5:]) < 0.5 * _mean(quantized[:5])
        assert _mean(quantized[-5:]) == pytest.approx(_mean(baseline[-5:]), rel=0.1)

    def test_autoep_experts(self):
        _require_fp8_kernels()
        from unit.v1.moe.autoep_test_utils import (MockMoETransformer, make_autoep_integration_config,
                                                   run_training_steps, seed_everything)

        def build(gemm=None):
            seed_everything(1234)
            config = make_autoep_integration_config(zero_stage=2, ep_size=2)
            config["optimizer"]["params"]["torch_adam"] = True
            if gemm is not None:
                recipe = {"enabled": True, "modules": ["*experts*"], "forward": gemm, "dx": gemm, "dw": gemm}
                config["quantization"] = {"subchannel": recipe}
            engine, _, _, _ = deepspeed.initialize(model=MockMoETransformer(), config=config)
            return engine

        seed_everything(99)
        x = torch.randn(1, 8, 64, device=get_accelerator().current_device_name())
        with torch.no_grad():
            expected = build()(x).double()
            # With quantization off the expert GEMMs must reproduce the native path: same tokens reach the same
            # experts, and the alignment padding rows stay out of the result.
            unquantized = build({"format": "none"})(x)
            engine = build({})
            quantized = engine(x)
            row_scaled = build({"block_size": 0})(x)
        assert _rel_error(unquantized, expected) < 1e-5
        assert 0 < _rel_error(quantized, expected) < 1
        assert 0 < _rel_error(row_scaled, expected) < 1

        experts = [module for module in engine.module.modules() if isinstance(module, GroupedExperts)]
        assert len(experts) == 2
        before = [param.detach().clone() for module in experts for param in module.parameters()]
        losses, _ = run_training_steps(engine, num_steps=3)
        after = [param.detach() for module in experts for param in module.parameters()]
        assert all(torch.isfinite(torch.tensor(losses)))
        assert all(not torch.equal(old, new) for old, new in zip(before, after))
