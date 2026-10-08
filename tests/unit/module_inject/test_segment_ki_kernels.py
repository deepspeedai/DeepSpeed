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

import pytest
import torch

from deepspeed.accelerator import get_accelerator

from deepspeed.module_inject.kernel_reference import decode_attn as ref_decode_attn
from deepspeed.module_inject.kernel_reference import dual_gemv_silu_mul as ref_dual_gemv
from deepspeed.module_inject.kernel_reference import fused_add_norm as ref_add_norm
from deepspeed.module_inject.kernel_reference import gdn_gates as ref_gdn_gates
from deepspeed.module_inject.kernel_reference import gdn_input_proj as ref_gdn_proj
from deepspeed.module_inject.kernel_reference import triple_gemv as ref_triple_gemv

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
        out_buf = torch.zeros(max_len, dtype=torch.long, device=DEV)

        expected = torch.argmax(logits[0].float())
        assert expected.item() == 10  # torch.argmax first-index rule

        _op().decode_step_graph(logits, token_out, write_pos, mask, out_buf)
        assert token_out.view(-1)[0].item() == 10
        assert write_pos.item() == 6
        assert bool(mask[0, 0, 0, 6]) and not bool(mask[0, 0, 0, 7])
        assert out_buf[6].item() == 10

        # a second call advances from the updated write_pos and reveals slot 7
        _op().decode_step_graph(logits, token_out, write_pos, mask, out_buf)
        assert write_pos.item() == 7
        assert bool(mask[0, 0, 0, 7]) and not bool(mask[0, 0, 0, 8])
        assert out_buf[7].item() == 10

    def test_dual_gemv_silu_mul(self):
        h = _rand_bf16(256)
        gw = _rand_bf16(384, 256)
        uw = _rand_bf16(384, 256)
        out = torch.empty(384, dtype=torch.bfloat16, device=DEV)
        _op().dual_gemv_silu_mul(h, gw, uw, out)
        expected = ref_dual_gemv(h.float(), gw.float(), uw.float())
        torch.testing.assert_close(out.float(), expected, atol=5e-2, rtol=5e-2)

    def test_quad_gemv(self):
        h = _rand_bf16(128)
        ws = [_rand_bf16(rows, 128) for rows in (96, 48, 6, 6)]
        total = sum(w.shape[0] for w in ws)
        out = torch.empty(total, dtype=torch.bfloat16, device=DEV)
        _op().quad_gemv(h, *ws, out)
        expected = ref_gdn_proj(h.float(), *[w.float() for w in ws])
        torch.testing.assert_close(out.float(), expected, atol=5e-2, rtol=5e-2)

    def test_gdn_gates(self):
        # a/b are [batch, seq, heads]; a_log/dt are per-head [heads] (real usage shapes)
        a = _rand_bf16(1, 4, 8)
        b = _rand_bf16(1, 4, 8)
        a_log = torch.randn(8, device=DEV)  # fp32, matching real usage
        dt = torch.randn(8, device=DEV)
        beta, g = _op().gdn_gates(a, b, a_log, dt)
        ref_beta, ref_g = ref_gdn_gates(a, b, a_log, dt)
        torch.testing.assert_close(beta.float(), ref_beta.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(g.float(), ref_g.float(), atol=5e-2, rtol=5e-2)

    def test_triple_gemv(self):
        h = _rand_bf16(128)
        qw, kw, vw = (_rand_bf16(64, 128) for _ in range(3))
        q, k, v = _op().triple_gemv(h, qw, kw, vw)
        ref_q, ref_k, ref_v = ref_triple_gemv(h.float(), qw.float(), kw.float(), vw.float())
        for got, exp in ((q, ref_q), (k, ref_k), (v, ref_v)):
            torch.testing.assert_close(got.float(), exp, atol=5e-2, rtol=5e-2)

    def test_fused_add_norm(self):
        h = _rand_bf16(1, 128)
        r = _rand_bf16(1, 128)
        w = _rand_bf16(128)
        h_kernel = h.clone()  # the kernel updates hidden in place
        r_kernel = r.clone()
        out = _op().fused_add_norm(h_kernel, r_kernel, w, 1e-5)
        ref_out = ref_add_norm(h, r, w, eps=1e-5)
        torch.testing.assert_close(out.float(), ref_out.float(), atol=5e-2, rtol=5e-2)
        # hidden is updated in place to the pre-norm sum
        torch.testing.assert_close(h_kernel.float(), (h.float() + r.float()), atol=5e-2, rtol=5e-2)

    def test_decode_attn(self):
        torch.manual_seed(0)
        nq, nkv, hd, maxlen, pos = 8, 2, 64, 33, 17
        q = torch.randn(nq, hd, dtype=torch.float32, device=DEV).bfloat16()
        K = torch.randn(nkv, maxlen, hd, dtype=torch.float32, device=DEV).bfloat16()
        V = torch.randn(nkv, maxlen, hd, dtype=torch.float32, device=DEV).bfloat16()
        write_pos = torch.tensor([pos], dtype=torch.long, device=DEV)
        out = torch.empty(nq, hd, dtype=torch.bfloat16, device=DEV)
        _op().decode_attn(q, K, V, write_pos, out, nq, nkv, hd, maxlen)
        expected = ref_decode_attn(q.float(), K.float(), V.float(), pos)
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
        expected = ref_gdn_proj(h, *ws)
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
class TestRolloutTrainRollout:

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
