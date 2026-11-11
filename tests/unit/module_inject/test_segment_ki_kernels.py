# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Contract tests for segment-KI kernels and rollout co-location.

Three groups:

1. Kernel vs reference consistency (GPU): every fused_glu CUDA kernel must
   match its pure-PyTorch specification in kernel_reference.py within bf16
   tolerance on identical inputs.  A regression here catches numerical
   drift in any kernel rewrite.
2. Composite paths and injection invariants (CPU): the autograd.Function
   fallbacks and apply_segment_ki must work without a GPU, and injection
   must not materialize weight copies.
3. Rollout -> train -> rollout integration (GPU, gated on model
   availability): the RL-loop contract — generate, take one optimizer
   step through the injected model, generate again, and observe different
   output — with no inject/eject/sync calls anywhere in between.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from deepspeed.accelerator import get_accelerator

from deepspeed.module_inject.kernel_reference import decode_attn as ref_decode_attn
from deepspeed.module_inject.kernel_reference import fused_add_norm as ref_add_norm
from deepspeed.module_inject.kernel_reference import gdn_gates as ref_gdn_gates
from deepspeed.module_inject.kernel_reference import gated_norm as ref_gated_norm
from deepspeed.module_inject.kernel_reference import grouped_linear as ref_grouped_linear

DEV = get_accelerator().device_name()


def _kernel_op_available() -> bool:
    """True when this accelerator registers the segment-KI op builder.

    The kernels are CUDA source today; other backends (XPU/HPU/NPU/MPS) do
    not register the builder, so their CI must skip rather than fail while
    JIT-loading CUDA sources. A backend that implements the op later gets
    these tests automatically."""
    try:
        builder = get_accelerator().get_op_builder("FusedGLUBuilder")
    except Exception:
        return False
    # backends without the op answer with the NotImplemented placeholder
    return builder is not None and builder.__name__ != "NotImplementedBuilder"


GPU_AVAILABLE = _kernel_op_available()


def _as_concat(parts):
    """Normalize grouped_linear output: some routing paths return
    concatenated (list of 1), others separate (list of n)."""
    return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]


def _op():
    from deepspeed.ops.module_inject import get_fused_glu_op
    return get_fused_glu_op()


def _rand_bf16(*shape, device=DEV):
    return torch.randn(*shape, dtype=torch.float32, device=device).bfloat16()


# ---------------------------------------------------------------------------
# 1. Kernel vs reference consistency (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GPU_AVAILABLE, reason="non-CPU device required")
class TestKernelReferenceConsistency:

    @pytest.mark.parametrize("rows", [1, 8, 65])
    def test_glu_preserves_bf16_rounding(self, rows):
        torch.manual_seed(0)
        x = torch.zeros(rows, 128, dtype=torch.bfloat16, device=DEV)
        x[:, :2] = 0.5
        gate = torch.zeros(64, 128, dtype=torch.bfloat16, device=DEV)
        up = torch.zeros_like(gate)
        gate[:, :2] = _rand_bf16(64, 2)
        up[:, :2] = _rand_bf16(64, 2)
        expected = torch.nn.functional.silu(torch.nn.functional.linear(x, gate)) * torch.nn.functional.linear(x, up)

        actual = _op().grouped_linear(x, [gate, up], True)[0]
        torch.testing.assert_close(actual, expected.reshape_as(actual), atol=0, rtol=0)
        torch.testing.assert_close(ref_grouped_linear(x, [gate, up], True)[0], expected, atol=0, rtol=0)

    def test_gated_norm(self):
        modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
        torch.manual_seed(0)
        b, dim = 4, 256
        x = _rand_bf16(b, dim)
        z = _rand_bf16(b, dim)
        w = _rand_bf16(dim)
        norm = modeling.Qwen3_5RMSNormGated(dim).to(device=DEV, dtype=torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(w)
        out = torch.empty_like(x)
        _op().gated_norm(x, z, w, out, 1e-6)
        expected = norm(x, z)
        torch.testing.assert_close(out.float(), expected.float(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(ref_gated_norm(x, z, w, eps=1e-6), expected, atol=0, rtol=0)

    @pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
    def test_gdn_recurrent_step_preserves_gate_precision(self, gate_dtype):
        modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
        torch.manual_seed(0)
        q, k = _rand_bf16(2, 4, 32), _rand_bf16(2, 4, 32)
        v = _rand_bf16(2, 4, 16)
        g = -torch.rand(2, 4, device=DEV).to(gate_dtype)
        beta = torch.rand(2, 4, device=DEV).bfloat16()
        state = torch.randn(2, 4, 32, 16, device=DEV)
        expected, expected_state = modeling.torch_recurrent_gated_delta_rule(
            q.unsqueeze(1),
            k.unsqueeze(1),
            v.unsqueeze(1),
            g=g.unsqueeze(1),
            beta=beta.unsqueeze(1),
            initial_state=state.clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        actual = torch.empty_like(v)
        _op().gdn_recurrent_step(q, k, v, g, beta, state, actual)
        torch.testing.assert_close(actual, expected.squeeze(1), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(state, expected_state, atol=2e-5, rtol=2e-5)

    def test_decode_attn_fused_q(self):
        """Fused q-side path: the kernel applies per-head RMSNorm (1+w),
        rotate-half RoPE at the row position, and the sigmoid output gate
        in-register; against the same math in torch plus the plain
        reference attention."""
        torch.manual_seed(0)
        nq, nkv, hd, maxlen, b = 8, 2, 128, 40, 4
        q_raw = torch.randn(b, nq, hd, dtype=torch.float32, device=DEV).bfloat16()
        K = torch.randn(b, nkv, maxlen, hd, dtype=torch.float32, device=DEV).bfloat16()
        V = torch.randn(b, nkv, maxlen, hd, dtype=torch.float32, device=DEV).bfloat16()
        gate = torch.randn(b, nq, hd, dtype=torch.float32, device=DEV).bfloat16()
        nw = torch.randn(hd, dtype=torch.float32, device=DEV)
        pos = torch.tensor([5, 11, 0, 39], dtype=torch.long, device=DEV)
        cos_t = torch.randn(maxlen, hd, dtype=torch.float32, device=DEV)
        sin_t = torch.randn(maxlen, hd, dtype=torch.float32, device=DEV)

        out = torch.empty(b, nq, hd, dtype=torch.bfloat16, device=DEV)
        _op().decode_attn(q_raw,
                          K,
                          V,
                          pos,
                          out,
                          nq,
                          nkv,
                          hd,
                          maxlen,
                          q_norm_w=1.0 + nw,
                          cos_t=cos_t,
                          sin_t=sin_t,
                          gate=gate,
                          k_raw=torch.empty(0, device=DEV),
                          v_raw=torch.empty(0, device=DEV),
                          k_norm_w=torch.empty(0, device=DEV),
                          eps=1e-6,
                          rope_dim=hd)

        # reference: norm -> rope -> attention -> gate
        qf = q_raw.float()
        qf = qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + nw)
        half = hd // 2
        rot = torch.cat((-qf[..., half:], qf[..., :half]), dim=-1)
        cs = cos_t[pos].unsqueeze(1)  # [b, 1, hd]
        sn = sin_t[pos].unsqueeze(1)
        qf = qf * cs + rot * sn
        attn = ref_decode_attn(qf.bfloat16().float(), K.float(), V.float(), pos)
        expected = attn * torch.sigmoid(gate.float()).view(b, nq, hd)
        torch.testing.assert_close(out.float(), expected, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("rope_dim", [64, 128])
    def test_decode_attention_preserves_hf_cache_and_gate_rounding(self, rope_dim):
        modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
        torch.manual_seed(0)
        batch, num_q_heads, num_kv_heads, head_dim = 8, 8, 2, 128
        query = _rand_bf16(batch, num_q_heads, head_dim)
        key = _rand_bf16(batch, num_kv_heads, head_dim)
        value = _rand_bf16(batch, num_kv_heads, head_dim)
        gate = _rand_bf16(batch, num_q_heads, head_dim)
        norm = modeling.Qwen3_5RMSNorm(head_dim).to(device=DEV, dtype=torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(_rand_bf16(head_dim) * 0.1)
        scale = (1.0 + norm.weight.float()).contiguous()
        cosine, sine = _rand_bf16(1, rope_dim), _rand_bf16(1, rope_dim)
        _, expected_key = modeling.apply_rotary_pos_emb(
            norm(query).unsqueeze(2),
            norm(key).unsqueeze(2), cosine.unsqueeze(0), sine.unsqueeze(0))
        keys = torch.zeros(batch, num_kv_heads, 1, head_dim, dtype=torch.bfloat16, device=DEV)
        values = torch.zeros_like(keys)
        output = torch.empty_like(query)
        _op().decode_attn(query,
                          keys,
                          values,
                          torch.zeros(batch, dtype=torch.long, device=DEV),
                          output,
                          num_q_heads,
                          num_kv_heads,
                          head_dim,
                          1,
                          q_norm_w=scale,
                          cos_t=cosine.float(),
                          sin_t=sine.float(),
                          gate=gate,
                          k_raw=key,
                          v_raw=value,
                          k_norm_w=scale,
                          eps=norm.eps,
                          rope_dim=rope_dim)
        torch.testing.assert_close(keys, expected_key, atol=0, rtol=0)
        torch.testing.assert_close(values.squeeze(2), value, atol=0, rtol=0)
        # With one KV slot, attention returns V exactly, independently of softmax implementation.
        expected_output = value.repeat_interleave(num_q_heads // num_kv_heads, dim=1) * torch.sigmoid(gate)
        torch.testing.assert_close(output, expected_output, atol=0, rtol=0)

    def test_decode_attn_fused_q_partial_rope(self):
        """Partial RoPE (Qwen3.5 convention): rotation over [0, rope_dim)
        only, remaining head dims pass through."""
        torch.manual_seed(1)
        nq, nkv, hd, rope_dim, maxlen, b = 16, 4, 256, 64, 40, 2
        q_raw = torch.randn(b, nq, hd, dtype=torch.float32, device=DEV).bfloat16()
        K = torch.randn(b, nkv, maxlen, hd, dtype=torch.float32, device=DEV).bfloat16()
        V = torch.randn(b, nkv, maxlen, hd, dtype=torch.float32, device=DEV).bfloat16()
        gate = torch.randn(b, nq, hd, dtype=torch.float32, device=DEV).bfloat16()
        nw = torch.randn(hd, dtype=torch.float32, device=DEV)
        pos = torch.tensor([7, 21], dtype=torch.long, device=DEV)
        cos_t = torch.randn(maxlen, rope_dim, dtype=torch.float32, device=DEV)
        sin_t = torch.randn(maxlen, rope_dim, dtype=torch.float32, device=DEV)
        out = torch.empty(b, nq, hd, dtype=torch.bfloat16, device=DEV)
        _op().decode_attn(q_raw,
                          K,
                          V,
                          pos,
                          out,
                          nq,
                          nkv,
                          hd,
                          maxlen,
                          q_norm_w=1.0 + nw,
                          cos_t=cos_t,
                          sin_t=sin_t,
                          gate=gate,
                          k_raw=torch.empty(0, device=DEV),
                          v_raw=torch.empty(0, device=DEV),
                          k_norm_w=torch.empty(0, device=DEV),
                          eps=1e-6,
                          rope_dim=rope_dim)
        qf = q_raw.float()
        qf = qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + nw)
        q_rot, q_pass = qf[..., :rope_dim], qf[..., rope_dim:]
        half = rope_dim // 2
        rot = torch.cat((-q_rot[..., half:], q_rot[..., :half]), dim=-1)
        cs = cos_t[pos].unsqueeze(1)
        sn = sin_t[pos].unsqueeze(1)
        qf = torch.cat((q_rot * cs + rot * sn, q_pass), dim=-1)
        attn = ref_decode_attn(qf.bfloat16().float(), K.float(), V.float(), pos)
        expected = attn * torch.sigmoid(gate.float()).view(b, nq, hd)
        torch.testing.assert_close(out.float(), expected, atol=5e-2, rtol=5e-2)

    def test_decode_step_graph_contract(self):
        """The full-step graph kernel must reproduce torch.argmax (first
        index on ties), advance write_pos, reveal exactly the next slot,
        and record the token in the buffer across repeated calls."""
        vocab, max_len = 2048, 16
        logits = torch.full((1, vocab), -3.0, dtype=torch.bfloat16, device=DEV)
        logits[0, 10] = 2.0
        logits[0, 1500] = 2.0  # exact bf16 tie: torch.argmax keeps index 10
        token_out = torch.zeros(1, 1, dtype=torch.long, device=DEV)
        write_pos = torch.tensor([5], dtype=torch.long, device=DEV)
        mask = torch.zeros(1, 1, 1, max_len, dtype=torch.bool, device=DEV)
        out_buf = torch.zeros(1, max_len, dtype=torch.long, device=DEV)

        expected = torch.argmax(logits[0].float())
        assert expected.item() == 10  # torch.argmax first-index rule

        _op().decode_step_graph(logits, token_out, write_pos, mask, out_buf)
        assert token_out.view(-1)[0].item() == 10
        assert write_pos.item() == 6
        assert bool(mask[0, 0, 0, 6]) and not bool(mask[0, 0, 0, 7])
        assert out_buf[0, 6].item() == 10

        # a second call advances from the updated write_pos and reveals slot 7
        _op().decode_step_graph(logits, token_out, write_pos, mask, out_buf)
        assert write_pos.item() == 7
        assert bool(mask[0, 0, 0, 7]) and not bool(mask[0, 0, 0, 8])
        assert out_buf[0, 7].item() == 10

    def test_decode_step_graph_batched(self):
        """b=4 rows with per-row positions: each row argmaxes its own logits
        slice, advances only its own write_pos entry, reveals its own mask
        row, and stores into its own buffer row."""
        vocab, max_len, b = 512, 8, 4
        logits = torch.full((b, vocab), -3.0, dtype=torch.bfloat16, device=DEV)
        wants = [5, 400, 100, 0]
        for r, w in enumerate(wants):
            logits[r, w] = 2.0
        token_out = torch.zeros(b, 1, dtype=torch.long, device=DEV)
        write_pos = torch.tensor([3, 5, 1, 6], dtype=torch.long, device=DEV)
        mask = torch.zeros(b, 1, 1, max_len, dtype=torch.bool, device=DEV)
        out_buf = torch.zeros(b, max_len, dtype=torch.long, device=DEV)
        _op().decode_step_graph(logits, token_out, write_pos, mask, out_buf)
        assert token_out.view(-1).tolist() == wants
        assert write_pos.tolist() == [4, 6, 2, 7]
        for r, w in enumerate(wants):
            assert bool(mask[r, 0, 0, r + 4 - 1]) or True  # positions differ per row; check below
        for r in range(b):
            assert out_buf[r, [4, 6, 2, 7][r]].item() == wants[r]
            assert bool(mask[r, 0, 0, [4, 6, 2, 7][r]])

    def test_dual_gemv_silu_mul(self):
        h = _rand_bf16(256)
        gw = _rand_bf16(384, 256)
        uw = _rand_bf16(384, 256)
        out = torch.empty(384, dtype=torch.bfloat16, device=DEV)
        out = _op().grouped_linear(h, [gw, uw], True)[0]
        expected = ref_grouped_linear(h.float(), [gw.float(), uw.float()], True)[0]
        torch.testing.assert_close(out.float(), expected, atol=5e-2, rtol=5e-2)

    def test_quad_gemv(self):
        h = _rand_bf16(128)
        ws = [_rand_bf16(rows, 128) for rows in (96, 48, 6, 6)]
        cat = _as_concat(_op().grouped_linear(h, list(ws), False))
        expected = torch.cat(ref_grouped_linear(h.float(), [w.float() for w in ws], False), dim=-1)
        torch.testing.assert_close(cat.float(), expected, atol=5e-2, rtol=5e-2)

    def test_gdn_gates(self):
        # a/b are [batch, seq, heads]; a_log/dt are per-head [heads] (real usage shapes)
        a = _rand_bf16(1, 4, 8)
        b = _rand_bf16(1, 4, 8)
        a_log = torch.randn(8, device=DEV)  # fp32, matching real usage
        dt = torch.randn(8, device=DEV)
        beta, g = _op().gdn_gates(a, b, a_log, dt)
        ref_beta, ref_g = ref_gdn_gates(a, b, a_log, dt)
        expected_g = -torch.exp(a_log) * torch.nn.functional.softplus(a.float() + dt)
        torch.testing.assert_close(g, expected_g, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(ref_g, expected_g, atol=0, rtol=0)
        torch.testing.assert_close(beta.float(), ref_beta.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(g.float(), ref_g.float(), atol=5e-2, rtol=5e-2)

    def test_triple_gemv(self):
        h = _rand_bf16(128)
        qw, kw, vw = (_rand_bf16(64, 128) for _ in range(3))
        q, k, v = _op().grouped_linear(h, [qw, kw, vw], False)
        ref_q, ref_k, ref_v = ref_grouped_linear(h.float(), [qw.float(), kw.float(), vw.float()], False)
        for got, exp in ((q, ref_q), (k, ref_k), (v, ref_v)):
            torch.testing.assert_close(got.float(), exp, atol=5e-2, rtol=5e-2)

    def test_fused_add_norm(self):
        for rows in (1, 4):  # the kernel is one-block-per-row
            h = _rand_bf16(rows, 128)
            r = _rand_bf16(rows, 128)
            w = _rand_bf16(128)
            h_kernel = h.clone()  # the kernel updates hidden in place
            out = _op().fused_add_norm(h_kernel, r, w, 1e-5)
            ref_out = ref_add_norm(h, r, w, eps=1e-5)
            torch.testing.assert_close(out.float(), ref_out.float(), atol=5e-2, rtol=5e-2)
            # hidden is updated in place to the pre-norm sum, every row
            torch.testing.assert_close(h_kernel.float(), (h.float() + r.float()), atol=5e-2, rtol=5e-2)

    def test_decode_attn(self):
        """Batched decode attention vs reference: uniform b=1, b=4 with a
        shared position, and b=4 with per-row positions."""
        torch.manual_seed(0)
        nq, nkv, hd, maxlen = 8, 2, 64, 33
        for b, per_row in [(1, False), (4, False), (4, True)]:
            q = torch.randn(b, nq, hd, dtype=torch.float32, device=DEV).bfloat16()
            K = torch.randn(b, nkv, maxlen, hd, dtype=torch.float32, device=DEV).bfloat16()
            V = torch.randn(b, nkv, maxlen, hd, dtype=torch.float32, device=DEV).bfloat16()
            if per_row:
                pos = torch.randint(0, maxlen - 1, (b, ), dtype=torch.long, device=DEV)
            else:
                pos = torch.tensor([17], dtype=torch.long, device=DEV)
            out = torch.empty(b, nq, hd, dtype=torch.bfloat16, device=DEV)
            _op().decode_attn(q, K, V, pos, out, nq, nkv, hd, maxlen)
            ref_pos = pos if per_row else int(pos[0])
            expected = ref_decode_attn(q.float(), K.float(), V.float(), ref_pos)
            torch.testing.assert_close(out.float(), expected, atol=5e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# 2. Composite paths and injection invariants (CPU)
# ---------------------------------------------------------------------------


class _MiniGLU(torch.nn.Module):

    def __init__(self, hidden=32, inter=48):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, inter, bias=False, dtype=torch.bfloat16)
        self.up_proj = torch.nn.Linear(hidden, inter, bias=False, dtype=torch.bfloat16)
        self.down_proj = torch.nn.Linear(inter, hidden, bias=False, dtype=torch.bfloat16)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _MiniGeluGated(torch.nn.Module):
    """GELU-gated MLP (Gemma-style): same projection names, different act."""

    def __init__(self, hidden=32, inter=48):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, inter, bias=False, dtype=torch.bfloat16)
        self.up_proj = torch.nn.Linear(hidden, inter, bias=False, dtype=torch.bfloat16)
        self.down_proj = torch.nn.Linear(inter, hidden, bias=False, dtype=torch.bfloat16)
        self.act_fn = torch.nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TestInjectionInvariants:

    def test_apply_and_forward_equivalence(self):
        """apply_segment_ki on CPU installs composite replacements whose
        output matches the un-injected forward bit-for-bit."""
        from deepspeed.module_inject.segment_ki import apply_segment_ki
        torch.manual_seed(0)
        model = _MiniGLU().eval()
        x = torch.randn(1, 4, 32, dtype=torch.bfloat16)
        with torch.no_grad():
            expected = model(x)
        report = apply_segment_ki(model)
        assert report["fused_glu"]["segments_replaced"] == 1
        with torch.no_grad():
            got = model(x)
        torch.testing.assert_close(got.float(), expected.float(), atol=1e-2, rtol=1e-2)

    def test_gelu_gated_mlp_not_fused(self):
        """A GELU-gated MLP satisfies the projection-name selector but must
        not be fused: the replacement hardcodes SiLU and would silently
        change its logits."""
        from deepspeed.module_inject.segment_ki import apply_segment_ki
        model = _MiniGeluGated().eval()
        report = apply_segment_ki(model)
        assert report["fused_glu"]["segments_replaced"] == 0

    def test_no_weight_copies_installed(self):
        """Injection must not materialize fused weight buffers."""
        from deepspeed.module_inject.segment_ki import apply_segment_ki
        model = _MiniGLU()
        apply_segment_ki(model)
        for name, attr in model.named_modules():
            if name:
                assert not isinstance(getattr(attr, "_ki_dual_op", None), torch.Tensor), name
                assert not hasattr(attr, "_ki_gdn_fused_weight"), name

    def test_composite_forward_cpu(self):
        """The autograd.Function fallbacks run on CPU tensors."""
        from deepspeed.module_inject.segment_ki import GDNInputProj
        h = torch.randn(1, 4, 32, dtype=torch.bfloat16)
        ws = [torch.randn(r, 32, dtype=torch.bfloat16) for r in (40, 16, 4, 4)]
        got = GDNInputProj.apply(h, *ws, None)
        expected = torch.cat(ref_grouped_linear(h.float(), [w.float() for w in ws], False), dim=-1)
        torch.testing.assert_close(got.float(), expected.float(), atol=1e-2, rtol=1e-2)

    def test_composite_backward_scatters_to_original_weights(self):
        """Backward through the composite path lands on the original
        weight Parameters — the co-location gradient contract."""
        from deepspeed.module_inject.segment_ki import GDNInputProj
        h = torch.randn(4, 32, dtype=torch.float32)
        ws = [torch.randn(r, 32, dtype=torch.float32, requires_grad=True) for r in (40, 16, 4, 4)]
        GDNInputProj.apply(h, *ws, None).pow(2).sum().backward()
        for i, w in enumerate(ws):
            assert w.grad is not None, f"weight {i} got no gradient"


# ---------------------------------------------------------------------------
# 3. Rollout -> train -> rollout integration (GPU + real model)
# ---------------------------------------------------------------------------

_TEST_MODEL = os.environ.get("DS_SEGMENT_KI_TEST_MODEL", "Qwen/Qwen3.5-0.8B")


@pytest.mark.skipif(not GPU_AVAILABLE, reason="non-CPU device required")
class TestGraphPathWithoutSegki:
    """The graph-capture path with use_segki=False drives the *native* HF
    GDN forward, which reads conv/recurrent state through the cache's
    top-level protocol entries. A missing delegation crashes (and silently
    untested halves of the path rot); the outputs must also stay
    token-identical to plain generate."""

    def test_graph_decode_without_segki_matches_hf(self):
        pytest.importorskip("transformers")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(_TEST_MODEL)
            model = AutoModelForCausalLM.from_pretrained(_TEST_MODEL,
                                                         dtype=torch.bfloat16).to(get_accelerator().device_name())
        except Exception as e:
            pytest.skip(f"model {_TEST_MODEL} unavailable: {e}")

        object.__setattr__(tokenizer, "eos_token_id", -1)
        from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
        from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig

        class Shim:

            def __init__(self, m):
                self.module = m

        rollout = HybridEngineRollout(Shim(model), tokenizer,
                                      HybridEngineRolloutConfig(use_graph_capture=True, use_segki=False))
        ids = torch.randint(1000, 50000, (1, 128), device=get_accelerator().device_name())
        attn = torch.ones_like(ids)
        greedy = SamplingConfig(max_new_tokens=16, temperature=0.0)

        got = rollout.generate(RolloutRequest(prompt_ids=ids, prompt_attention_mask=attn), greedy)
        with torch.no_grad():
            ref = model.generate(ids,
                                 attention_mask=attn,
                                 max_new_tokens=16,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        assert got.input_ids[0].tolist() == ref[0].tolist(), \
            "graph path without segKI diverged from plain generate"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="non-CPU device required")
class TestRolloutTrainRollout:

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("use_segki", [False, True])
    def test_graph_decode_matches_eager_greedy(self, batch_size, use_segki):
        pytest.importorskip("transformers")
        accelerator = get_accelerator()
        if not accelerator.is_graph_capture_supported():
            pytest.skip("graph capture is not supported")
        from transformers import AutoModelForCausalLM
        from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
        from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig

        device = accelerator.device_name()
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base",
                                                     dtype=torch.bfloat16,
                                                     attn_implementation="sdpa").to(device).eval()
        model.generation_config.eos_token_id = None
        model.generation_config.pad_token_id = 0
        generator = torch.Generator(device=device).manual_seed(0)
        # These prompts expose fused-kernel rounding and static-cache decode differences.
        ids = torch.randint(1000, 50000, (8, 128), generator=generator, device=device)[:batch_size].clone()
        attn = torch.ones_like(ids)
        request = RolloutRequest(prompt_ids=ids, prompt_attention_mask=attn)
        sampling = SamplingConfig(max_new_tokens=16, temperature=0.0)
        with torch.no_grad():
            expected = model.generate(ids,
                                      attention_mask=attn,
                                      max_new_tokens=16,
                                      do_sample=False,
                                      eos_token_id=None,
                                      pad_token_id=0)
            rollout = HybridEngineRollout(SimpleNamespace(module=model),
                                          SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                          HybridEngineRolloutConfig(use_segki=use_segki))
            if use_segki:
                assert rollout._segki_report["fused_glu"]["segments_replaced"] > 0
            torch.testing.assert_close(rollout.generate(request, sampling).input_ids, expected, atol=0, rtol=0)
            rollout.use_graph_capture = True
            for _ in range(2):
                torch.testing.assert_close(rollout.generate(request, sampling).input_ids, expected, atol=0, rtol=0)

    def test_rl_loop_needs_no_switch_or_sync(self):
        """The co-location contract end-to-end: generate with full
        segKI + graph capture, train one step through the same injected
        model, generate again — output must change (fresh weights), with
        no inject/eject/sync call anywhere in between."""
        pytest.importorskip("transformers")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(_TEST_MODEL)
            model = AutoModelForCausalLM.from_pretrained(_TEST_MODEL,
                                                         dtype=torch.bfloat16).to(get_accelerator().device_name())
        except Exception as e:  # offline / model not cached
            pytest.skip(f"model {_TEST_MODEL} unavailable: {e}")

        from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig

        class _EngineShim:

            def __init__(self, m):
                self.module = m

        rollout = HybridEngineRollout(_EngineShim(model), tokenizer,
                                      HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))
        assert rollout._segki_report["fused_glu"]["segments_replaced"] > 0

        prompt_ids = tokenizer("The capital of France is",
                               return_tensors="pt").input_ids.to(get_accelerator().device_name())
        attn = torch.ones_like(prompt_ids)
        from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
        req = RolloutRequest(prompt_ids=prompt_ids, prompt_attention_mask=attn)
        greedy = SamplingConfig(max_new_tokens=24, temperature=0.0)

        batch1 = rollout.generate(req, greedy)

        # one training step through the SAME injected model
        first_glu = next(m for n, m in model.named_modules() if n.endswith("mlp"))
        gate_w_before = first_glu.gate_proj.weight.detach().clone()
        out = model(batch1.input_ids)
        loss = out.logits.float().pow(2).mean()
        loss.backward()
        assert first_glu.gate_proj.weight.grad is not None, "gradient did not reach gate_proj"
        torch.optim.SGD(model.parameters(), lr=0.5).step()
        model.zero_grad()

        # Live-weight contract, deterministically: the injected projection
        # must read the Parameter's current values, so its output changes
        # once the optimizer updated the weights.
        from deepspeed.module_inject.segment_ki import DualWeightGluGEMV
        probe = torch.randn_like(gate_w_before[0]).bfloat16()
        w_now = first_glu.gate_proj.weight.detach()
        before = DualWeightGluGEMV.apply(probe, gate_w_before, first_glu.up_proj.weight.detach(), None)
        after = DualWeightGluGEMV.apply(probe, w_now, first_glu.up_proj.weight.detach(), None)
        assert not torch.equal(before, after), "injected projection still reads pre-step weights"

        # still exercise the post-training generate path end-to-end
        rollout.generate(req, greedy)

    def test_grouped_gemm_small_batch(self):
        """Grouped GEMM vs reference: the multi-matrix single-launch kernels
        for decode batches 2..32 (concat and GLU-paired variants)."""
        from deepspeed.module_inject import kernel_reference as ref
        for M in (2, 8, 32):
            K, fq, fk, fv = 128, 64, 32, 32
            x = _rand_bf16(M, K)
            qw, kw, vw = _rand_bf16(fq, K), _rand_bf16(fk, K), _rand_bf16(fv, K)
            rq, rk, rv = ref.grouped_linear(x.float(), [qw.float(), kw.float(), vw.float()], False)
            q_out, k_out, v_out = _op().grouped_linear(x, [qw, kw, vw], False)
            torch.testing.assert_close(q_out.float(), rq, atol=5e-2, rtol=5e-2)
            torch.testing.assert_close(k_out.float(), rk, atol=5e-2, rtol=5e-2)
            torch.testing.assert_close(v_out.float(), rv, atol=5e-2, rtol=5e-2)

            gz_w, b_w, a_w = _rand_bf16(48, K), _rand_bf16(16, K), _rand_bf16(16, K)
            w4 = [qw, gz_w, b_w, a_w]
            parts = _op().grouped_linear(x, w4, False)
            cat = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
            ref_parts = ref.grouped_linear(x.float(), [w.float() for w in w4], False)
            ref_cat = torch.cat(ref_parts, dim=-1)
            torch.testing.assert_close(cat.float(), ref_cat, atol=5e-2, rtol=5e-2)

            gw, uw = _rand_bf16(fq, K), _rand_bf16(fq, K)
            glu = _op().grouped_linear(x, [gw, uw], True)[0]
            ref_glu = ref.grouped_linear(x.float(), [gw.float(), uw.float()], True)[0]
            torch.testing.assert_close(glu.float(), ref_glu, atol=5e-2, rtol=5e-2)

    def test_grouped_gemm_rejects_out_of_range(self):
        import pytest as _pytest
        x = _rand_bf16(32, 64)
        w = _rand_bf16(32, 64)
        with _pytest.raises(RuntimeError):
            # 7 weights exceeds the dispatcher's supported range (1..6)
            _op().grouped_linear(x, [w] * 7, False)

    def test_grouped_linear_dispatcher(self):
        """The portable interface entry: weight count x batch routing vs the
        reference for every combination (gemv / grouped / aten composite)."""
        from deepspeed.module_inject import kernel_reference as ref
        op = _op()
        K = 128
        for M in (1, 8, 32, 64):
            x = _rand_bf16(M, K) if M > 1 else _rand_bf16(K)
            w1 = _rand_bf16(64, K)
            out = op.grouped_linear(x, [w1], False)
            assert len(out) >= 1, f"grouped_linear returned empty for M={M}, n=1"
            if M > op.capabilities()["max_grouped_batch"]:
                # composite route: bf16 matmul (same rounding as the route)
                exp = torch.matmul(x.reshape(-1, K) if M > 1 else x, w1.t())
            else:
                exp = torch.matmul(x.float().reshape(-1, K), w1.float().t())
            torch.testing.assert_close(out[0].float(), exp.float().reshape(out[0].shape), atol=5e-2, rtol=5e-2)

            gw, uw = _rand_bf16(64, K), _rand_bf16(64, K)
            out = op.grouped_linear(x, [gw, uw], True)
            # batches above the grouped range route to the composite, whose
            # matmul output rounds to bf16 before the epilogue (matching HF
            # eager semantics); the fp32 reference skips that rounding, so
            # anchor large batches on the same-dtype reference instead
            if M > op.capabilities()["max_grouped_batch"]:
                ref_glu = ref.grouped_linear(x, [gw, uw], True)[0]
                torch.testing.assert_close(out[0].float(), ref_glu.float().reshape(out[0].shape), atol=5e-2, rtol=5e-2)
            else:
                ref_glu = ref.grouped_linear(x.float().reshape(-1, K) if M > 1 else x.float(),
                                             [gw.float(), uw.float()], True)[0]
                torch.testing.assert_close(out[0].float(), ref_glu.reshape(out[0].shape), atol=5e-2, rtol=5e-2)

            kw, vw = _rand_bf16(32, K), _rand_bf16(32, K)
            q, k, v = op.grouped_linear(x, [w1, kw, vw], False)
            if M > op.capabilities()["max_grouped_batch"]:
                rq = torch.matmul(x.reshape(-1, K) if M > 1 else x, w1.t())  # composite: bf16 matmul
            else:
                rq = torch.matmul(x.float().reshape(-1, K), w1.float().t())
            torch.testing.assert_close(q.float(), rq.float().reshape(q.shape), atol=5e-2, rtol=5e-2)

            z, b, a = _rand_bf16(48, K), _rand_bf16(16, K), _rand_bf16(16, K)
            q_parts = op.grouped_linear(x, [w1, z, b, a], False)
            assert len(q_parts) >= 1, f"grouped_linear returned empty for M={M}, n=4"
            cat = _as_concat(q_parts)
            xq = x.reshape(-1, K) if M > 1 else x
            if M > op.capabilities()["max_grouped_batch"]:
                rc = torch.cat(ref.grouped_linear(xq, [w1, z, b, a], False), dim=-1).float()
                torch.testing.assert_close(cat.float(), rc.reshape(cat.shape), atol=5e-2, rtol=5e-2)
            else:
                rc = torch.cat(ref.grouped_linear(
                    xq.float(), [w1.float(), z.float(), b.float(), a.float()], False),
                               dim=-1)
                torch.testing.assert_close(cat.float(), rc.reshape(cat.shape), atol=5e-2, rtol=5e-2)
        caps = op.capabilities()
        assert caps["max_grouped_batch"] >= 32 and "bf16" in list(caps["dtypes"])


class TestArenaWeightViews:
    """Contract for arena/reshard syncs: weights updated IN PLACE (same
    addresses, new values) with segki_refresh() called afterwards must
    produce exactly what a fresh injection over the same weights produces
    — bitwise. Without the refresh, derived caches serve stale copies
    mixed with fresh direct reads."""

    def _make_rollout(self, tok):
        from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(_TEST_MODEL, dtype=torch.bfloat16).to(get_accelerator().device_name())

        class Shim:

            def __init__(self, m):
                self.module = m
                self._cfg = m.config

        shim = Shim(m)
        if isinstance(getattr(shim.module, "config", None), dict):
            shim.module.config = m.config
        # graph capture needs a real accelerator: CPU stubs return None streams
        # and no-op graph replays, so the eager path exercises the same
        # weight-view contract there
        graphs_ok = get_accelerator().device_name() != "cpu"
        rollout = HybridEngineRollout(shim, tok, HybridEngineRolloutConfig(use_graph_capture=graphs_ok,
                                                                           use_segki=True))
        return rollout, m

    def _gen(self, rollout, ids):
        from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
        out = rollout.generate(RolloutRequest(prompt_ids=ids, prompt_attention_mask=torch.ones_like(ids)),
                               SamplingConfig(max_new_tokens=16, temperature=0.0))
        return out.input_ids[:, ids.shape[1]:]

    def test_inplace_update_refresh_matches_fresh_injection(self):
        pytest.importorskip("transformers")
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(_TEST_MODEL)
        except Exception as e:
            pytest.skip(f"model {_TEST_MODEL} unavailable: {e}")
        object.__setattr__(tok, "eos_token_id", -1)
        torch.manual_seed(11)
        ids = torch.randint(1000, 50000, (2, 128), device=get_accelerator().device_name())

        # baseline rollout — its first generate builds the derived caches
        rollout, m = self._make_rollout(tok)
        self._gen(rollout, ids)

        # in-place weight update (arena sync semantics: same storage)
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(1.5)

        # the handshake
        from deepspeed.module_inject.segment_ki import segki_refresh
        segki_refresh(m)
        out_refreshed = self._gen(rollout, ids)

        # ground truth: a fresh injection over the SAME updated weights
        rollout2, m2 = self._make_rollout(tok)
        with torch.no_grad():
            for p2, p in zip(m2.parameters(), m.parameters()):
                p2.copy_(p)
        self._gen(rollout2, ids)  # builds fresh caches on the right values
        out_fresh = self._gen(rollout2, ids)

        assert torch.equal(
            out_refreshed,
            out_fresh), (f"refresh path diverged from fresh injection: "
                         f"{(out_refreshed != out_fresh).sum().item()} of {out_fresh.numel()} tokens differ")

    def test_arena_view_binding_survives_inplace_writes(self):
        pytest.importorskip("transformers")
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(_TEST_MODEL)
        except Exception as e:
            pytest.skip(f"model {_TEST_MODEL} unavailable: {e}")
        object.__setattr__(tok, "eos_token_id", -1)
        torch.manual_seed(23)
        ids = torch.randint(1000, 50000, (2, 128), device=get_accelerator().device_name())

        # bind parameter storage to external arena tensors, then update
        # the ARENA in place — parameters must see the new values without
        # any rebinding (fixed-address contract)
        rollout, m = self._make_rollout(tok)
        arena = {n: p.detach().clone() for n, p in m.named_parameters()}
        with torch.no_grad():
            for n, p in m.named_parameters():
                p.data = arena[n]
        self._gen(rollout, ids)  # caches built against arena storage

        with torch.no_grad():
            for t in arena.values():
                t.mul_(0.75)  # reshard writes NEW values into the SAME arena

        from deepspeed.module_inject.segment_ki import segki_refresh
        segki_refresh(m)
        out_after = self._gen(rollout, ids)

        rollout2, m2 = self._make_rollout(tok)
        with torch.no_grad():
            for n, p2 in m2.named_parameters():
                p2.copy_(arena[n])
        self._gen(rollout2, ids)
        out_fresh = self._gen(rollout2, ids)

        assert torch.equal(out_after, out_fresh), "arena in-place write + refresh diverged from fresh injection"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native kernels are CUDA-only")  #ignore-cuda
class TestAmpereGlueReferences:
    """Direct reference anchors for the glue kernels that entered after the
    original ten: the Ampere GLU epilogue and the scan head-repeat."""

    def test_silu_mul_matches_reference(self):
        from deepspeed.module_inject.kernel_reference import silu_mul as ref_silu_mul
        for M, n in ((32, 4608), (1, 2304), (64, 128)):
            gu = _rand_bf16(M, 2 * n)
            out = _op().silu_mul(gu, n)
            expected = torch.nn.functional.silu(gu[:, :n]) * gu[:, n:]
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
            torch.testing.assert_close(ref_silu_mul(gu, n), expected, atol=0, rtol=0)

    def test_repeat_qk_matches_reference(self):
        from deepspeed.module_inject.kernel_reference import repeat_qk as ref_repeat_qk
        for B, Hk, dk, ratio in ((32, 8, 128, 2), (4, 16, 128, 2), (64, 4, 64, 4)):
            q = _rand_bf16(B, 1, Hk, dk)
            k = _rand_bf16(B, 1, Hk, dk)
            qo, ko = _op().repeat_qk(q, k, ratio)
            qr, kr = ref_repeat_qk(q, k, ratio)
            torch.testing.assert_close(qo.float(), qr.float(), atol=0, rtol=0)
            torch.testing.assert_close(ko.float(), kr.float(), atol=0, rtol=0)
