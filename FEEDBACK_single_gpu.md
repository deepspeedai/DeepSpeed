# seg-KI (#8602) on Qwen3.5-4B: bench feedback

Measured on PR head `6647fb300` (Oct 16) plus the local fixes below.

## 1. Setup

How this differs from your test (same prefix, greedy, bs=8):

- **A different prompt in every row** instead of a shared prefix.
- **Sampling at temperature 1** instead of greedy.
- **Batch sizes 1, 4, 32 and 128.**
- **No padding.** Every prompt is exactly 256 tokens, so all rows in a batch have the same length. These numbers do not
  cover batches with mixed prompt lengths.

Bench: 1x A100-SXM4-40GB, torch 2.14.1+cu130, transformers 5.19.0. Random prompts of 256 tokens, different per row,
256 generated tokens, temperature 1, no EOS stop. Reference: SGLang 0.5.23 (CUDA graphs on) with the same model, prompts
and sampling.

## 2. Results

Decode step time (ms), graph capture + seg-KI vs SGLang:

| bs | seg-KI | SGLang | seg-KI as % of SGLang speed |
|---|---|---|---|
| 1 | 9.9 | 8.0 | 81% |
| 4 | 11.5 | 8.7 | 76% |
| 32 | 17.9 | 11.9 | 66% |

## 3. Kernel coverage at bs>1

| Kernel | bs>1 |
|---|---|
| `decode_attn` (8 full-attention layers) | on |
| `fused_add_norm` (32 layers) | on |
| `gdn_gates`, `gated_norm` (24 GDN layers) | on |
| `dual_gemv_silu_mul` (MLP) | **off**: falls back to 2 GEMMs + silu*mul |
| `triple_gemv` (q/k/v) | **off**: falls back to 3 GEMMs |
| `quad_gemv` (GDN input projections) | **off**: falls back to 4 GEMMs + `cat` |
| `decode_step_graph` | **off** (bs=1 greedy only) |

- **The weight-bound projections run as separate GEMMs at bs>1.** Only the light kernels remain fused, which fits the
  growing gap above. SGLang and vLLM use merged gate|up and qkv weights.
- **Idea:** a merged-weight GEMM, or a small-batch GEMV, for bs ≤ 32 would probably close much of the gap.

## 4. Fixes the bench needed (both still in `deafce7`)

- **`decode_attn` receives a strided query.** Qwen3.5's `q_proj` emits `[query | gate]` per head, so
  `query_states.reshape(batch, heads, head_dim)` is a strided view. The kernel indexes it as packed, so heads > 0 read
  the wrong values.
  - In our tests, greedy output diverged from HF within 1 to 8 tokens at every batch size.
  - My fix: call `.contiguous()` before the kernel, and reject non-contiguous `q` in the kernel wrapper.
- **The GDN scan fails on transformers 5.19.** It no longer binds `chunk_/recurrent_gated_delta_rule` onto each layer,
  so the fused GDN forward raises `AttributeError`.
  - My fix: fall back to the module-level `torch_*_gated_delta_rule` functions.
