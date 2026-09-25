# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Compare fused RoPE with the Hugging Face eager split-half expression."""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.triton_ops import fused_rotary_pos_emb as fused_rope

eager_rope = fused_rope._reference_apply_rotary_pos_emb


def _fused_available():
    accelerator = get_accelerator()
    return accelerator.is_available() and accelerator.device_name().startswith("cuda") and fused_rope.is_available()


pytestmark = pytest.mark.skipif(not _fused_available(), reason="fused RoPE needs CUDA and Triton")


def _device():
    return get_accelerator().current_device_name()


def _tables(batch, seq_len, head_dim, dtype, base=10000.0):
    # The Hugging Face rotary embedding: duplicated FP32 frequencies, cast to the model dtype.
    inv_freq = 1.0 / (base**(torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)[None, :].expand(batch, -1)
    freqs = positions[:, :, None] * inv_freq[None, None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(_device(), dtype), emb.sin().to(_device(), dtype)


def _projection_layout(batch, seq_len, heads, head_dim, dtype, generator):
    # Attention builds q and k as a [batch, sequence, heads, head_dim] projection transposed to heads-first.
    value = torch.randn(batch, seq_len, heads, head_dim, generator=generator).to(_device(), dtype)
    return value.transpose(1, 2).detach().requires_grad_(True)


def _contiguous_layout(batch, seq_len, heads, head_dim, dtype, generator):
    value = torch.randn(batch, heads, seq_len, head_dim, generator=generator).to(_device(), dtype)
    return value.requires_grad_(True)


def _run(function, q, k, cos, sin, grad_q, grad_k, unsqueeze_dim=1):
    q, k = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True)
    q_out, k_out = function(q, k, cos, sin, unsqueeze_dim)
    torch.autograd.backward((q_out, k_out), (grad_q, grad_k))
    return q_out, k_out, q.grad, k.grad


def _layout(tensor):
    # A dimension of size one has no meaningful stride.
    return [stride for size, stride in zip(tensor.shape, tensor.stride()) if size > 1]


def _assert_equal(actual, expected, label):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, label
    # torch.equal treats -0.0 and +0.0 as equal, which is the only difference the order of an exact sum can make.
    assert torch.equal(actual, expected), f"{label}: max |diff| {(actual.float() - expected.float()).abs().max()}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch, seq_len, q_heads, k_heads, head_dim, layout",
    [
        (1, 1024, 32, 4, 128, "projection"),
        (2, 333, 8, 2, 128, "projection"),
        (1, 17, 5, 3, 64, "contiguous"),
        (2, 64, 4, 4, 80, "contiguous"),
        (1, 1, 2, 1, 128, "projection"),
    ],
)
def test_matches_eager(dtype, batch, seq_len, q_heads, k_heads, head_dim, layout):
    generator = torch.Generator().manual_seed(seq_len * 131 + head_dim)
    make = _projection_layout if layout == "projection" else _contiguous_layout
    q = make(batch, seq_len, q_heads, head_dim, dtype, generator)
    k = make(batch, seq_len, k_heads, head_dim, dtype, generator)
    cos, sin = _tables(batch, seq_len, head_dim, dtype)
    grad_q = torch.randn(q.shape, generator=generator).to(_device(), dtype)
    grad_k = torch.randn(k.shape, generator=generator).to(_device(), dtype)

    expected = _run(eager_rope, q, k, cos, sin, grad_q, grad_k)
    actual = _run(fused_rope.fused_apply_rotary_pos_emb, q, k, cos, sin, grad_q, grad_k)
    for name, a, e in zip(("q_embed", "k_embed", "q.grad", "k.grad"), actual, expected):
        _assert_equal(a, e, name)
    # The outputs keep the strides of their inputs, as eager's do; the gradients take the layout of q and k.
    assert _layout(actual[0]) == _layout(expected[0])
    assert _layout(actual[1]) == _layout(expected[1])
    assert _layout(actual[2]) == _layout(q)


def test_heads_second_layout_with_unsqueeze_dim_2():
    generator = torch.Generator().manual_seed(2)
    dtype = torch.bfloat16
    q = torch.randn(2, 96, 6, 128, generator=generator).to(_device(), dtype)
    k = torch.randn(2, 96, 2, 128, generator=generator).to(_device(), dtype)
    cos, sin = _tables(2, 96, 128, dtype)
    grad_q, grad_k = torch.randn_like(q), torch.randn_like(k)
    expected = _run(eager_rope, q, k, cos, sin, grad_q, grad_k, unsqueeze_dim=2)
    actual = _run(fused_rope.fused_apply_rotary_pos_emb, q, k, cos, sin, grad_q, grad_k, unsqueeze_dim=2)
    for name, a, e in zip(("q_embed", "k_embed", "q.grad", "k.grad"), actual, expected):
        _assert_equal(a, e, name)


def test_one_table_broadcast_over_the_batch():
    generator = torch.Generator().manual_seed(3)
    dtype = torch.bfloat16
    q = _projection_layout(3, 50, 4, 128, dtype, generator)
    k = _projection_layout(3, 50, 2, 128, dtype, generator)
    cos, sin = _tables(1, 50, 128, dtype)
    grad_q, grad_k = torch.randn_like(q), torch.randn_like(k)
    expected = _run(eager_rope, q, k, cos, sin, grad_q, grad_k)
    actual = _run(fused_rope.fused_apply_rotary_pos_emb, q, k, cos, sin, grad_q, grad_k)
    for name, a, e in zip(("q_embed", "k_embed", "q.grad", "k.grad"), actual, expected):
        _assert_equal(a, e, name)


def test_non_contiguous_incoming_gradients():
    generator = torch.Generator().manual_seed(4)
    dtype = torch.bfloat16
    q = _projection_layout(1, 256, 8, 128, dtype, generator)
    k = _projection_layout(1, 256, 2, 128, dtype, generator)
    cos, sin = _tables(1, 256, 128, dtype)
    # Heads-first contiguous gradients, as attention backward may return, and one with a padded head dimension.
    grad_q = torch.randn(1, 8, 256, 128, generator=generator).to(_device(), dtype)
    grad_k = torch.randn(1, 2, 256, 256, generator=generator).to(_device(), dtype)[..., :128]
    expected = _run(eager_rope, q, k, cos, sin, grad_q, grad_k)
    actual = _run(fused_rope.fused_apply_rotary_pos_emb, q, k, cos, sin, grad_q, grad_k)
    for name, a, e in zip(("q_embed", "k_embed", "q.grad", "k.grad"), actual, expected):
        _assert_equal(a, e, name)


def test_unused_key_output_leaves_no_key_gradient():
    generator = torch.Generator().manual_seed(5)
    dtype = torch.bfloat16
    q = _projection_layout(1, 32, 4, 128, dtype, generator)
    k = _projection_layout(1, 32, 2, 128, dtype, generator)
    cos, sin = _tables(1, 32, 128, dtype)
    q_out, _ = fused_rope.fused_apply_rotary_pos_emb(q, k, cos, sin)
    q_out.float().sum().backward()
    assert k.grad is None
    assert q.grad is not None


def test_empty_sequence():
    dtype = torch.bfloat16
    q = torch.empty(1, 4, 0, 128, device=_device(), dtype=dtype, requires_grad=True)
    k = torch.empty(1, 2, 0, 128, device=_device(), dtype=dtype, requires_grad=True)
    cos, sin = _tables(1, 0, 128, dtype)
    q_out, k_out = fused_rope.fused_apply_rotary_pos_emb(q, k, cos, sin)
    (q_out.sum() + k_out.sum()).backward()
    assert q_out.shape == q.shape and q.grad.shape == q.shape


def test_double_backward_raises():
    generator = torch.Generator().manual_seed(6)
    dtype = torch.bfloat16
    q = _projection_layout(1, 16, 2, 128, dtype, generator)
    k = _projection_layout(1, 16, 2, 128, dtype, generator)
    cos, sin = _tables(1, 16, 128, dtype)
    q_out, k_out = fused_rope.fused_apply_rotary_pos_emb(q, k, cos, sin)
    (grad_q, ) = torch.autograd.grad(q_out.float().pow(2).sum(), q, create_graph=True)
    with pytest.raises(RuntimeError):
        grad_q.float().sum().backward()


# Parametrized by name, so collection creates no tensors on any device; each test builds its own inputs.
_UNSUPPORTED_CASES = ("float32", "dtype mismatch", "cpu", "odd head dimension", "strided head dimension",
                      "sequence mismatch", "table requires grad", "unsqueeze_dim")


def _unsupported_cases():
    dtype = torch.bfloat16
    q = torch.randn(1, 4, 8, 128, device=_device(), dtype=dtype)
    k = torch.randn(1, 2, 8, 128, device=_device(), dtype=dtype)
    cos, sin = _tables(1, 8, 128, dtype)
    return {
        "float32": (q.float(), k.float(), cos.float(), sin.float(), 1),
        "dtype mismatch": (q, k, cos.float(), sin.float(), 1),
        "cpu": (q.cpu(), k.cpu(), cos.cpu(), sin.cpu(), 1),
        "odd head dimension": (q[..., :127], k[..., :127], cos[..., :127], sin[..., :127], 1),
        "strided head dimension": (q[..., ::2], k[..., ::2], cos[..., :64], sin[..., :64], 1),
        "sequence mismatch": (q, k, cos[:, :4], sin[:, :4], 1),
        "table requires grad": (q, k, cos.clone().requires_grad_(True), sin, 1),
        "unsqueeze_dim": (q, k, cos, sin, 0),
    }


@pytest.mark.parametrize("case", _UNSUPPORTED_CASES)
def test_direct_call_rejects_unsupported_inputs(case):
    with pytest.raises(RuntimeError, match="fused RoPE"):
        fused_rope.fused_apply_rotary_pos_emb(*_unsupported_cases()[case])


def _tiny_qwen3():
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3 import modeling_qwen3

    config = Qwen3Config(hidden_size=256,
                         num_attention_heads=4,
                         num_key_value_heads=2,
                         head_dim=128,
                         num_hidden_layers=2,
                         intermediate_size=128,
                         vocab_size=97)
    # Eager attention and a dense MLP keep the rest of the model deterministic, so any difference is RoPE's.
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    model = modeling_qwen3.Qwen3ForCausalLM(config).to(_device(), torch.bfloat16)
    return transformers, modeling_qwen3, model


@pytest.fixture
def restore_after():
    yield
    fused_rope.restore_rotary_pos_emb()


def test_installer_matches_eager_model(restore_after):
    _, modeling, model = _tiny_qwen3()
    tokens = torch.randint(0, 97, (2, 40), device=_device())

    def loss_and_grads():
        model.zero_grad(set_to_none=True)
        output = model(tokens, labels=tokens)
        output.loss.backward()
        return output.logits.detach(), {name: p.grad.detach().clone() for name, p in model.named_parameters()}

    eager_logits, eager_grads = loss_and_grads()
    repeat_logits, repeat_grads = loss_and_grads()
    assert torch.equal(repeat_logits, eager_logits) and all(
        torch.equal(repeat_grads[name], grad)
        for name, grad in eager_grads.items()), "eager model is not deterministic"
    original = modeling.apply_rotary_pos_emb
    assert fused_rope.replace_rotary_pos_emb(model) == 1
    assert modeling.apply_rotary_pos_emb is not original
    calls = []
    run_kernels = fused_rope._run_kernels
    fused_rope._run_kernels = lambda *args: calls.append(1) or run_kernels(*args)
    try:
        fused_logits, fused_grads = loss_and_grads()
    finally:
        fused_rope._run_kernels = run_kernels
    assert len(calls) == 2, "each decoder layer must run the kernel"
    _assert_equal(fused_logits, eager_logits, "logits")
    for name, grad in eager_grads.items():
        _assert_equal(fused_grads[name], grad, name)


def test_installer_is_idempotent_and_restorable(restore_after):
    _, modeling, model = _tiny_qwen3()
    original = modeling.apply_rotary_pos_emb
    assert fused_rope.replace_rotary_pos_emb(model) == 1
    patched = modeling.apply_rotary_pos_emb
    assert fused_rope.replace_rotary_pos_emb(model) == 0
    assert modeling.apply_rotary_pos_emb is patched
    assert fused_rope.restore_rotary_pos_emb() == 1
    assert modeling.apply_rotary_pos_emb is original


def test_installer_skips_models_it_does_not_list(restore_after):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    assert fused_rope.replace_rotary_pos_emb(model) == 0


def test_installer_leaves_a_changed_expression_alone(restore_after, monkeypatch):
    _, modeling, model = _tiny_qwen3()

    def rope_with_cast(q, k, cos, sin, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim).float()
        sin = sin.unsqueeze(unsqueeze_dim).float()
        return (q * cos + modeling.rotate_half(q) * sin).to(q.dtype), (k * cos + modeling.rotate_half(k) * sin).to(
            k.dtype)

    monkeypatch.setattr(modeling, "apply_rotary_pos_emb", rope_with_cast)
    assert fused_rope.replace_rotary_pos_emb(model) == 0
    assert modeling.apply_rotary_pos_emb is rope_with_cast


def _rotate_a_quarter(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 4]
    x2 = x[..., x.shape[-1] // 4:]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_along_the_sequence(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-2] // 2]
    x2 = x[..., x.shape[-2] // 2:]
    return torch.cat((-x2, x1), dim=-2)


# Each differs from the split-half rotate_half only in constants, which its bytecode refers to by index. From
# Python 3.14 small integers are written into the bytecode instead, so only the second stays bytecode-identical there.
@pytest.mark.parametrize("changed_rotate_half", [_rotate_a_quarter, _rotate_along_the_sequence],
                         ids=["quarter", "sequence-axis"])
def test_installer_leaves_a_rotation_with_changed_constants_alone(restore_after, monkeypatch, changed_rotate_half):
    _, modeling, model = _tiny_qwen3()
    original = modeling.apply_rotary_pos_emb
    monkeypatch.setattr(modeling, "rotate_half", changed_rotate_half)
    assert fused_rope.replace_rotary_pos_emb(model) == 0
    assert modeling.apply_rotary_pos_emb is original


def _rotate_half_documented_differently(x):
    """The split-half rotation, documented differently."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# The expression below looks the rotation up by this name, as the Hugging Face function does in its module.
rotate_half = _rotate_half_documented_differently


def _apply_documented_differently(q, k, cos, sin, unsqueeze_dim=1):
    """The split-half expression, documented differently."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def test_installer_accepts_functions_that_differ_only_in_docstrings(restore_after, monkeypatch):
    _, modeling, model = _tiny_qwen3()
    monkeypatch.setattr(modeling, "apply_rotary_pos_emb", _apply_documented_differently)
    monkeypatch.setattr(modeling, "rotate_half", _rotate_half_documented_differently)
    assert fused_rope.replace_rotary_pos_emb(model) == 1
    assert modeling.apply_rotary_pos_emb is not _apply_documented_differently


def test_installed_function_runs_eager_for_unsupported_inputs(restore_after):
    _, modeling, model = _tiny_qwen3()
    original = modeling.apply_rotary_pos_emb
    fused_rope.replace_rotary_pos_emb(model)
    q, k, cos, sin, _ = _unsupported_cases()["float32"]
    actual = modeling.apply_rotary_pos_emb(q, k, cos, sin)
    expected = original(q, k, cos, sin)
    for a, e in zip(actual, expected):
        _assert_equal(a, e, "float32 fallback")
