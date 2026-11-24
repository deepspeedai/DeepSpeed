# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Direct parity against the original HF forward (outputs AND gradients).

The kernel-consistency tests compare kernels against our own reference
implementations; both sides could share a wrong formula. These tests
snapshot the pristine Qwen3.5 layers BEFORE injection, then compare
forward outputs and input gradients with explicit bf16 tolerances.
Known rounding-order differences (fused paths keep fp32 intermediates
where HF rounds to bf16 mid-way) are expected within these tolerances.
"""

import pytest
import torch

from deepspeed.accelerator import get_accelerator

_TEST_MODEL = "Qwen/Qwen3.5-0.8B-Base"

# bf16 forward parity: fused paths differ from HF only in intermediate
# rounding order; empirically these bound the drift of one decoder layer
ATOL, RTOL = 5e-2, 5e-2
# gradient parity tolerates the same drift propagated through backward
GATOL, GRTOL = 8e-2, 8e-2


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("transformers")
    try:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(_TEST_MODEL, dtype=torch.bfloat16).to("cuda")
    except Exception as e:
        pytest.skip(f"model {_TEST_MODEL} unavailable: {e}")
    return m


def _pick_layer(model, want_block):
    """Return the sub-block (self_attn / linear_attn), not the decoder
    layer — the layer requires position_embeddings, the blocks do not."""
    types = getattr(model.config, "layer_types", None) or []
    for i, layer in enumerate(model.model.layers):
        if types[i:i + 1] == [want_block]:
            return layer.linear_attn if want_block == "linear_attention" else layer
    pytest.skip(f"no {want_block} layer found")


@pytest.mark.skipif(get_accelerator().device_name() == "cpu", reason="non-CPU device required")
class TestFullAttentionLayerParity:

    def test_forward_and_backward(self, model):
        """Full-attn layer pre/post injection: output + d(loss)/dx."""
        from deepspeed.module_inject.segment_ki import apply_segment_ki
        layer = _pick_layer(model, "full_attention")
        torch.manual_seed(0)
        hidden = torch.randn(2, 5, model.config.hidden_size, dtype=torch.bfloat16, device="cuda")
        pos = torch.arange(5, device="cuda")
        pos_ids = pos.unsqueeze(0).expand(2, -1)
        cos_sin = model.model.rotary_emb(hidden, pos_ids)

        def run():
            h = hidden.clone().requires_grad_(True)
            out = layer(h, position_embeddings=cos_sin, position_ids=pos_ids)
            logits = out[0].float().pow(2).sum()
            logits.backward()
            return out[0].float(), h.grad.float()

        out_p, grad_p = run()  # pristine, before injection
        apply_segment_ki(model)
        out_i, grad_i = run()  # same module object, injected forward
        torch.testing.assert_close(out_i, out_p, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(grad_i, grad_p, atol=GATOL, rtol=GRTOL)


@pytest.mark.skipif(get_accelerator().device_name() == "cpu", reason="non-CPU device required")
class TestGDNLayerParity:

    def test_forward_and_backward(self, model):
        """GDN layer pre/post injection: output + input gradient."""
        from deepspeed.module_inject.segment_ki import apply_segment_ki
        layer = _pick_layer(model, "linear_attention")
        torch.manual_seed(0)
        hidden = torch.randn(2, 5, model.config.hidden_size, dtype=torch.bfloat16, device="cuda")

        def run():
            h = hidden.clone().requires_grad_(True)
            out = layer(h)  # GDN block: no position/rotary inputs
            loss = out.float().pow(2).sum()
            loss.backward()
            return out.float(), h.grad.float()

        out_p, grad_p = run()
        apply_segment_ki(model)
        out_i, grad_i = run()
        torch.testing.assert_close(out_i, out_p, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(grad_i, grad_p, atol=GATOL, rtol=GRTOL)


@pytest.mark.skipif(get_accelerator().device_name() == "cpu", reason="non-CPU device required")
class TestMLPParity:
    """Injected GLU replacement vs the HF formula on the same weights."""

    def test_forward_and_weight_grads(self, model):
        from deepspeed.module_inject.segment_ki import apply_segment_ki
        apply_segment_ki(model)
        mlp = next(m for n, m in model.named_modules() if n.endswith("mlp"))
        torch.manual_seed(0)
        x = torch.randn(2, 5, mlp.gate_proj.in_features, dtype=torch.bfloat16, device="cuda")

        x1 = x.clone().requires_grad_(True)
        out1 = mlp(x1)  # injected forward (composite path, grad-enabled)
        out1.float().pow(2).sum().backward()

        x2 = x.clone().requires_grad_(True)
        ref = mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(x2)) * mlp.up_proj(x2))
        ref.float().pow(2).sum().backward()

        torch.testing.assert_close(out1.float(), ref.float(), atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(x1.grad.float(), x2.grad.float(), atol=GATOL, rtol=GRTOL)


@pytest.mark.skipif(get_accelerator().device_name() == "cpu", reason="non-CPU device required")
class TestFusedAddNormVsHFFormula:
    """The reviewer's rounding-order case: HF materializes the residual sum
    in bf16 before the norm; the kernel keeps the fp32 sum for the output
    and rounds only the residual write-back. Pin both sides' magnitude."""

    def test_output_and_residual_vs_hf(self, model):
        from deepspeed.ops.module_inject import get_fused_glu_op
        torch.manual_seed(0)
        dim = 256
        h = (torch.randn(4, dim, device="cuda") * 0.5).bfloat16()
        r = (torch.randn(4, dim, device="cuda") * 0.5).bfloat16()
        w = (torch.randn(dim, device="cuda") * 0.1).bfloat16()
        eps = 1e-6

        h_kernel = h.clone()
        # hold the scaled weight in a variable: the kernel launch is async,
        # and an unreferenced temporary's memory can be handed to the next
        # allocation (the ref below) before the kernel reads it
        w_scaled = (1.0 + w.float()).bfloat16()
        out_kernel = get_fused_glu_op().fused_add_norm(h_kernel, r, w_scaled, eps)

        # HF formula verbatim: bf16 sum materialized, fp32 norm, (1+w) scale
        combined_bf16 = h + r  # bf16 rounding happens here in HF
        xf = combined_bf16.float()
        ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).bfloat16()

        torch.testing.assert_close(out_kernel.float(), ref.float(), atol=ATOL, rtol=RTOL)
        # the residual stream the kernel writes back is the bf16 sum, as in HF
        torch.testing.assert_close(h_kernel.float(), combined_bf16.float(), atol=ATOL, rtol=RTOL)
