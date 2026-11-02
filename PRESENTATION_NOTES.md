# PR #8602 — Segment-Based Kernel Injection: Talk Notes

> 4 pages + appendix (FAQ & methodology). English. RTX 4080 SUPER, Qwen3.5-4B bf16, b=1 greedy unless noted.

---

# Page 1 — Performance Summary

## Throughput (Qwen3.5-4B, b=1 greedy — all four engines, same session, slope method¹)

**Short context (5-token prompt):**

| Engine | Decode-only tok/s | End-to-end tok/s |
|---|---|---|
| HF eager generate | 18.1–18.7 | 17.7–19.4 |
| Graph capture only | 63.9 | 61.8 |
| **Graph + segKI** | **68.2** | **64.4–65.0** |
| vLLM 0.31 | 76.4 | 75.9 |

**Long context (2,048-token prompt):**

| Engine | Decode-only tok/s | End-to-end tok/s |
|---|---|---|
| HF eager generate | 20.3–22.7 | 16.7–20.1 |
| Graph capture only | 59.3 | 56.0–56.1 |
| **Graph + segKI** | **65.1–65.2** | **59.5–59.6** |
| vLLM 0.31 | 74.3 | 72.9 |

¹ decode-only = slope of two generation lengths (T₁₀₂₄−T₂₅₆)/768 — removes every
per-call fixed cost (prefill, graph capture, cache setup); identical method for
all four engines. Two repeats per config; ranges shown. Absolute numbers this
session ran ~3–5% below the previous boot (host variance) — cross-engine ratios
are same-session and fair. The earlier "author-reported vLLM 74.9" is confirmed
by our re-measurement (75.9 short e2e).

## Gain attribution (decode-only basis)

| Step | Short ctx | 2K ctx |
|---|---|---|
| eager → +graph capture | 18.4 → 63.9 (**3.5×**) | 21.5 → 59.3 (**2.8×**) |
| +graph → +segKI | 63.9 → 68.2 (**+6.7%**) | 59.3 → 65.2 (**+9.9%**) |

Graph capture is the big lever; segKI adds a consistent gain at both contexts
(the earlier "+0.8% at short context" was an artifact of mixed measurement bases).

**Why decode is weight-bandwidth-bound**: at b=1 the arithmetic intensity is
~1 FLOP/byte (≈8 GFLOP vs ≈8 GB per token) — memory time exceeds compute time
~50×. Signature: decode barely slows as context grows (68.2 → 65.2 tok/s from
5 → 2K KV). Effective bandwidth ≈ 546 GB/s = ~74% of the 4080 SUPER's 736 GB/s
spec (typical for memory-bound kernels); the absolute floor is ~91 tok/s.

## Versus vLLM

| | short | 2K |
|---|---|---|
| segKI decode-only ÷ vLLM decode-only | 89% | 88% |
| segKI e2e ÷ vLLM e2e | 85% | 82% |

vLLM is nearly flat across contexts (76→74); our gap is stable ~10–15%, and
unlike vLLM this path co-locates with training on the same live weights.

## How graph capture is enabled

```python
rollout = HybridEngineRollout(
    engine, tokenizer,
    HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))
```

- **Both flags default to `False`** — the PR is fully opt-in; the default path
  is HF `module.generate` unchanged.
- `use_graph_capture=True` engages the static-cache + CUDA-graph path **only
  for greedy sampling** (`temperature=0`); sampling requests and padded
  prompts fall back to `module.generate` automatically.
- `use_segki=True` layers kernel injection on top (works with or without
  graph capture; its b=1 fused decode step needs the graph path).

## One-line takeaway

> Graph capture makes decode fast; segKI keeps it fast as context grows — decode runs weight-bandwidth-bound at ~74% of the GPU's memory spec.

---

# Page 2 — What segKI Injects (structural, zero per-model code)

Detection is by attribute names + safety gates (SiLU-verified MLPs, bias-free
projections, no TP collectives). Anything unmatched passes through silently.

| Kernel | Replaces | What it fuses |
|---|---|---|
| `dual_gemv_silu_mul` | GLU MLP gate+up | two GEMVs + SiLU-gate in one launch, reads both weights directly |
| `quad_gemv` | GDN input projections | qkv/z/b/a four GEMVs in one launch (b=1) |
| `gdn_gates` | GDN gating math | sigmoid/softplus gates, one launch |
| `triple_gemv` | Q/K/V projections | three GEMVs in one launch (bias-free only) |
| `decode_attn` | SDPA at decode | warp-per-head, KV split-K, online softmax, valid-length only |
| `fused_add_norm` | residual-add + RMSNorm | in-place residual stream + norm in one launch |
| `decode_step_graph` | Python argmax + buffer updates | argmax, write_pos advance, mask reveal, token store — captured inside the graph |

Layered decode: full-step graph (b=1) → graph + Python argmax (b>1) → Python loop.

## Per-kernel ablation (2K context, decode-only, leave-one-out¹)

| Kernel group | decode-only tok/s | Contribution vs full |
|---|---|---|
| **full segKI** | **67.3** | — |
| − attention (`triple_gemv` + `decode_attn`) | 61.0 | **−9.4% (dominant)** |
| − norm (`fused_add_norm`) | 65.9 | −2.1% |
| − GDN (`quad_gemv` + `gdn_gates`) | 66.0 | −1.9% |
| − step (`decode_step_graph`) | 66.8 | −0.7% |
| − GLU (`dual_gemv_silu_mul`) | 67.5 | ~0 (noise) |

¹ Temporary measurement-only env gates on the GPU box (not in the PR);
one fresh process + fresh model per case; single run each (±1% noise).

Reading: at 2K the win is almost entirely the **custom decode attention** —
it replaces SDPA's slow path against the full-width static cache. The GEMV
fusions are marginal because decode is already weight-bandwidth-bound;
`decode_step_graph` is small because graph replay already amortizes launches.

Also: one op (`fused_glu`), one builder in `op_builder/` (install-time precompile),
dispatches gated on bf16/contiguity/bias/SiLU — every unsupported case falls back, never crashes.

---

# Page 3 — The Python Reference: one spec, three roles

Every CUDA kernel in this PR has a pure-PyTorch executable specification
(`deepspeed/module_inject/kernel_reference.py`) — readable math, no CUDA:

```python
def dual_gemv_silu_mul(hidden, gate_w, up_w):
    return F.silu(hidden @ gate_w.t()) * (hidden @ up_w.t())
```

**Role 1 — test oracle.** Kernel-consistency tests run each CUDA kernel and
its reference on identical inputs and assert closeness within bf16 tolerance.
Any kernel rewrite that drifts numerically fails CI — the contract is the
math, not the implementation.

**Role 2 — runtime fallback.** Every dispatch gate (bf16? contiguous?
bias-free? SiLU? grad-enabled?) falls back to the reference composite instead
of crashing. fp16 models, CPU tensors, training forwards, packed shapes —
all run through it, which is why "unsupported" means *slower*, never *broken*.

**Role 3 — porting spec.** A backend that wants to implement the op (e.g. an
XPU/SYCL port) gets an unambiguous behavioral specification for free — the
same file the tests use.

Bonus: the composite path carries exact hand-written autograd backward, so
the fallback also preserves the co-location gradient contract (Page 4).

---

# Page 4 — Co-Location: train and generate share one model

The RL-loop contract: **generate → one optimizer step → generate again, with zero inject/eject/sync in between.**

- Replacements read the original weight `Parameter` objects directly — **no weight copies**, so weights are fresh by construction.
- GLU / GDN projections run through `autograd.Function` with **exact hand-written backward** — gradients land on the original Parameters (verified: `gate_proj.weight.grad` populated through the injected path).
- Kernels without autograd (`gdn_gates`, `fused_add_norm`) are **no-grad-path only** — grad-enabled forwards route to native/composite ops, so no gradient is ever silently dropped.
- Deterministic live-weight probe (replaced a flaky assertion): the injected projection's output on fixed input **changes after an SGD step** — proves kernels read live weights, not a stale snapshot.
- Graph-replay freshness: scale/weights are read from Parameter memory inside the captured graph — replays see post-training values.

Verified end-to-end on GPU: rollout-train-rollout integration test green (generate, train through the same injected model, generate again).

---

# Page 5 — Future Work

- **batch > 1**: today b>1 rides graph + Python argmax; needs batched step kernel, batched decode attention, per-row write positions.
- **TP > 1**: collective-bearing projections are never fused today (correct but unaccelerated); needs shard-aware weight slicing in the GEMV kernels.
- **Continuous batching**: hybrid models currently rejected up front (clear error); needs batched GDN state — per-row conv/recurrent buffers, scatter on admission, compaction on retirement — integrated with the scheduler (align-decode-fronts, cache trimming).
- Smaller items: vLLM head-to-head re-measurement; upstream varlen cross-sequence bleed to report to transformers; install-time numerical probe to convert detection from name-based to computation-verified.

---
---

# Appendix (backup slides / FAQ)

## A1. Full verified-results table

| Check | Result |
|---|---|
| Unit + kernel-consistency suite | 18 passed (incl. direct `decode_step_graph` contract test: tied-logits argmax, write_pos, mask reveal, buffer updates) |
| Token parity: graph-only vs HF eager | bit-identical over 32 tokens |
| segKI vs eager at near-ties | can diverge where top-2 logits differ <1% — expected bf16 behavior, same class as vLLM-vs-HF |
| fp16 model | generates correctly (bf16-gated dispatches fall back to composite) |
| Left-padded prompt | correct continuation via HF fallback (right-pad echo is decoder-only-inherent, matches HF) |
| Varlen (packed sequences) | injected forward matches native forward: argmax identical at every position |
| Rollout→train→rollout | gradients reach original Parameters; deterministic live-weight probe passes |

## A2. Measurement method (why the old numbers needed a redo)

- Earlier sessions mixed bases: 2K decode-only (prefill subtracted) vs short
  end-to-end (capture cost amortized) — that produced artifacts like "2K is
  faster than short" and "segKI +0.8%".
- Final method: **slope of two lengths** — T(256) and T(1024) per config,
  decode-only = (T₁₀₂₄−T₂₅₆)/768; every fixed cost cancels; same formula for
  HF eager, graph-only, segKI, and vLLM.
- Roofline at b=1: ≈8 GFLOP vs ≈8 GB per token → ~1 FLOP/byte, memory-bound
  by ~50×. Effective bandwidth measured ≈ 546 GB/s (~74% of the 736 GB/s
  spec); earlier "99% of floor" claims came from superseded subtraction
  numbers (the subtraction method inflated the 2K decode rate).

## A3. Fair comparison methodology (incl. vLLM decode-only)

vLLM's offline API exposes `RequestOutput.metrics` per request:

```python
decode_tps = (n_new - 1) / (m.finished_time - m.first_token_time)
ttft       = m.first_token_time - m.arrival_time
e2e_tps    = n_new / (m.finished_time - m.arrival_time)
```

One run yields TTFT + decode-only + e2e for vLLM. Fairness: same GPU, dtype,
b=1, greedy, warmup excluded; vLLM in a separate conda env.

## A4. Why does the hand-written triple_gemv beat cuBLAS by ~5%?

It is not a faster kernel — it is **one launch instead of three**:

| Variant | μs/iter (b=1, DRAM-regime) |
|---|---|
| pre-concat + single cuBLAS call | 77.5 |
| **triple_gemv (3 original weights)** | 78.1 |
| 3× F.linear (cuBLAS) | 82.4 |

cuBLAS itself gains the same ~6% when handed the pre-concatenated matrix
(77.5 vs 82.4) — the loss lives in the three-launch structure: tail waves
that leave DRAM idle at each kernel's end ×3, and memory-pipeline bubbles
at kernel boundaries. `triple_gemv` reaches single-call performance while
still reading the three original live weights — variant A *is* the fused
concat+GEMM, with the concat as address arithmetic instead of a data copy.
A pre-concat-once variant saves only 0.8% per step here (~0.15% end-to-end,
noise) while costing a ~375 MB copy and refresh logic. cuBLAS's tuning focus is
training-shape GEMM; b=1 GEMV is served "well enough" (86% of spec) rather
than optimally, and it cannot merge three user-side calls into one.
Micro-benchmark note: weight sets were cycled so the 3.3 GB working set
exceeds L2 — a first L2-resident run measured L2 bandwidth (2.2 TB/s) and
was discarded.

## A5. Investigation 1 (CLOSED): why in-place weight reading stays

The concat-amortization argument — "pre-reshape+concat the weights once per
rollout, decode many times, recover the one-time cost" — is logically sound
but bounded by physics at b=1:

- **Amortization works, the prize doesn't**: one-time concat ≈ 70 μs; per-step
  saving 0.6 μs × 36 layers; net ≈ 0.14% end-to-end (noise).
- **Why so small**: the op is DRAM-bound — time = weight bytes ÷ bandwidth,
  independent of layout. Measured: in-place kernel 671 GB/s (91% of spec),
  pre-concat single GEMM 676 (92%). The ceiling leaves ≤1% to layout.
- **Where the argument regains teeth**: b>1 — tail-wave (SM wave-quantization)
  losses don't shrink as ops grow, and freshness is recoverable by re-concat
  at each generate's re-capture. Folded into Investigation 3's test matrix
  (pre-concat + single GEMM vs 3× GEMM at b ∈ {1, 4, 16}).
- Related survey: no other concat-reuse opportunity exists — all weight-side
  kernels already read original weights via address mapping (the fused
  concat+GEMM); the only one-time weight concat ever in the code was removed
  for staleness; the remaining concat is activation-side (GDN b>1), which
  changes every step by nature.

## A6. Review campaign (context slide)

- 24 bot comments (Codex 9 + Copilot 15) addressed, ~30 commits; plus finds beyond the bots (`gdn_gates` fp16 crash, bare op-load escaping rollout construction, prefill padding corruption).
- Removed the unreachable C++ decode-loop layer (−239 lines): zero perf change measured.
- All fixes GPU-verified; 3 rounds of on-instance validation (correctness, varlen, throughput).
