# Post-meeting investigations (2026-10-08 discussion) — PLANNING ONLY, no code yet

## 1. triple_gemv: in-place kernel vs pre-concat weights? — CLOSED (no change)

We read the three original weight matrices directly in one custom kernel.
Question: would concatenating weights first (then one big GEMV / cuBLAS) be faster?

**Micro-benchmark** (Qwen3.5-4B q/k/v shapes, b=1 bf16, 64 weight sets
cycling so the 3.3 GB working set forces DRAM traffic like a real decode
step; RTX 4080 SUPER spec 736 GB/s):

| Variant | μs/iter | eff. GB/s | verdict |
|---|---|---|---|
| **A. triple_gemv (3 orig weights)** | 78.1 | **671 (91%)** | current — near DRAM spec |
| D. 3× F.linear (cuBLAS) | 82.4 | 636 (86%) | 5% slower than our kernel |
| B. pre-concat copy + 1 linear | 77.5 | 676 (92%) | −0.8% vs A per step (noise-level) |

**Conclusion**: keep triple_gemv as-is. It already runs at 91% of DRAM
bandwidth (and beats 3× cuBLAS by ~5% via single-launch structure).
Pre-concat-once-at-rollout is not slower per step (its cost is just the
single cuBLAS call; freshness is recoverable by re-concatenating at each
generate's re-capture), but its per-step gain is 0.8% on this op ≈ ~0.15%
end-to-end — inside measurement noise — while costing a ~375 MB weight
copy, re-capture refresh logic, and extra injection state.
First L2-resident run (52 MB working set) was discarded — it measured L2
bandwidth, not decode conditions.

## 2. temp > 0 (sampling) under CUDA graph — MEASURED (route a is ~free)

Today the graph path is greedy-only; sampling falls back to HF generate.

**Route a (sample outside the graph) cost, Qwen3.5-4B b=1, slope method:**

| Config | decode-only tok/s | delta |
|---|---|---|
| full-step greedy (argmax in graph) | 70.3 | — |
| graph + Python argmax | 70.2 | −0.1% |
| graph + Python sampling (softmax + multinomial) | 69.7 | **−0.9%** |

Sampling outside the graph costs ≈ 0.9% decode throughput (~0.1 ms/step) —
essentially free, vs falling back to HF generate (−60%). Route a is the
recommended design: graph captures forward only; per step, Python reads
static_logits, applies temperature/top-k/top-p, and calls multinomial.
(Measured with EOS masked to keep generation running; masked-softmax adds
negligible cost. Temporary patch reverted after measurement.)

Implementation landed in e56f41f94 (sample on the captured graph; standard
HF-equivalent temperature -> top-k -> top-p -> multinomial pipeline).

**Post-implementation measurement** (same session, slope method, b=1,
eos disabled for full-length runs, Qwen3.5-4B):

| Config | ours short | ours 2K | vLLM short | vLLM 2K | ours/vLLM |
|---|---|---|---|---|---|
| greedy | 70.5 | 67.1 | 76.5 | 74.5 | 90–92% |
| sample t=1.0 | 69.6 (−1.3%) | 66.4 (−1.0%) | 76.4 (−0.1%) | 74.6 (~0) | 89–91% |
| sample t=1.0 p=0.9 | 68.7 (−2.6%) | 65.6 (−2.2%) | 76.2 (−0.4%) | 74.2 (−0.4%) | 88–90% |

Notes: pure-temperature sampling costs us ~1% (matches the earlier
estimate); top-p adds another ~1.2% on our side (sort+cumsum+scatter over
the ~150k vocab per step, in Python) while vLLM's sampler absorbs it
(~0.1%). Relative position vs vLLM is stable at 88–92% across all modes.
Remaining: distribution-parity spot-check vs HF generate.

## 3. bs > 1 performance — FIRST MEASUREMENT AT b=8 (Qwen3.5-4B, greedy, uniform prompt, slope method)

| | ours graph-only | ours segKI | vLLM | segKI effect | ours/vLLM |
|---|---|---|---|---|---|
| b=8 short | 360.2 tok/s (22.2 ms/step) | 360.4 (22.2 ms) | 536.6 (15.0 ms) | **0%** | 67% |
| b=8 2K | 273.6 (29.2 ms/step) | 273.2 (29.2 ms) | 508.1 (15.7 ms) | **0%** | **54%** |
| (ref) b=1 short | 63.9 | 70.5 | 76.5 | +10% | 92% |
| (ref) b=1 2K | 59.3 | 67.1 | 74.5 | +13% | 90% |

Findings:
1. **segKI contributes nothing at b=8** — every injected kernel is b=1-only
   (dual_gemv/quad_gemv/triple_gemv/decode_attn/fused_add_norm guard on
   shape[0]==1); only gdn_gates survives, which is noise. Confirms the
   code-level prediction.
2. **The vLLM gap widens with batch**: 90–92% at b=1 → 67% short / **54% 2K**
   at b=8. Two compounding causes: (a) decode falls back to SDPA against the
   full-width static cache (the exact path decode_attn eliminates at b=1 —
   most visible at 2K), and (b) the per-step Python argmax/buffer loop.
3. Batch efficiency itself is fine on our path otherwise: 22.2 ms/step at
   b=8 vs 14.2 at b=1 (weights amortized across rows).
4. Priority order for the b>1 kernel work follows directly: batched decode
   attention > batched step kernel; GEMV batching is last (weights amortize
   anyway at b≥8, see item 1's concat note below).

**Follow-up (same investigation): batched decode_attn implemented and measured.**
The kernel was rewritten to batch (one block per (row, q-head), per-row or
shared write_pos; reference + tests updated: b=1 / b=4 shared / b=4 per-row;
qkv GEMV branch stays b=1-only, batched rows use the HF projections).
Correctness: b=8 greedy outputs token-identical to graph-only at both
contexts; full GPU suite 18 passed.

| b=8 decode-only | before (SDPA fallback) | after (batched kernel) | gain | vs vLLM |
|---|---|---|---|---|
| short | 360.4 tok/s | **445.6** | **+23.6%** | 67% → **83%** |
| 2K | 273.5 tok/s | **411.2** | **+50.3%** | 54% → **81%** |

The 2K jump confirms the SDPA-against-full-width-static-cache fallback was
the dominant b>1 bottleneck. Remaining b>1 gap (~17–19% to vLLM) is the
per-step Python argmax/buffer loop and GEMV batching.

**Follow-up 2: batched step kernel (greedy) implemented — ~0% effect.**
decode_step_graph now handles any batch (per-row argmax + per-row write_pos
+ row-strided mask/buffer; rollout uses per-row write_pos everywhere; the
per-row cache-update bounds check is skipped during capture). GPU suite 19
passed; b=8 outputs token-identical. Measured at b=8: short 443.0 vs 445.6
tok/s, 2K 411.0 vs 411.2 — **the Python per-step loop was never the
bottleneck** (~0.3% of a 17 ms step). The remaining ~17–19% gap to vLLM at
b=8 is GEMM/kernel efficiency, not stepping overhead. The kernel is still
the right structure going forward (single launch, sampling can reuse the
buffer/reveal half).

**Kernel × batch status (b=8):**

| Kernel | b=8 status | path taken at b=8 |
|---|---|---|
| decode_attn | ✅ batched (this investigation) | custom kernel |
| decode_step_graph (greedy) | ✅ batched (follow-up 2) | in-graph step |
| gdn_gates | ✅ batch-capable (elementwise) | custom kernel |
| fused_add_norm | ✅ batch-capable (row-parallel) but guard still requires b=1 | orig layer forward |
| dual_gemv (GLU) | ❌ single-vector kernel | composite GEMMs (cuBLAS) |
| quad_gemv (GDN in-proj) | ❌ single-vector kernel | composite GEMMs (cuBLAS) |
| triple_gemv (QKV) | ❌ single-vector kernel | 3× F.linear (cuBLAS) |
| sampling step | n/a — Python by design | _sample_from_logits |

**Follow-up 3: torch.profiler on the b=8 2K decode step** (128-token
generate, class-level aggregation; prefill/capture contamination removed by
cross-checking against the slope-measured 19.5 ms/step):

| class | ms/step (segki) | share | note |
|---|---|---|---|
| GEMM (cuBLAS, decode portion est.) | ~12.5 | ~64% | weights-bound, near floor |
| **elementwise (eager soup)** | **5.1** | **26%** | ~130 launches/step: eager RMSNorm ×72, rope, q/k norm, GDN composite casts |
| decode_attn (ours) | 1.6 | 8% | 8 full-attn layers (hybrid model) |
| other/reduce/memcpy | 3.0 | incl. contamination | |

graph-only comparison: SDPA path spends an extra **5.4 ms/step in one
elementwise kernel** (mask expansion over the full-width cache) — that is
what batched decode_attn removed (+50% at 2K).

**Conclusion: the remaining gap to vLLM (19.5 vs 15.7 ms/step) is the eager
elementwise soup, not GEMM and not stepping.** vLLM fuses rope/norm/etc.
into its kernels; our HF-structure path runs ~130 small elementwise launches
per step. Actionable, in order: (1) enable/batch fused_add_norm at b>1
(norms are a large slice of the soup and the kernel is already
row-parallel), (2) fuse rope into the attention kernel, (3) trim bf16 casts
in the GDN composite. This also revises the earlier estimate: the four GEMV
fusions remain low-value, but fused_add_norm-at-b>1 is worth more than the
b=1 ablation suggested.

**Follow-up 4: fused_add_norm at b>1 enabled — measured and verified.**
b=8 decode-only: short 445.6 → **454.4 (+2.0%)**, 2K 411.2 → **416.9 (+1.4%)**.
vs vLLM: 85% short / **82%** 2K. Token-equality vs graph-only no longer holds:
the norm kernel's rounding path differs from eager RMSNorm, so logits shift
slightly and the known "\\n\\n vs \\n" near-tie (Δlogit 0.125) flips — same
drift class as b=1 segki-vs-eager, both continuations coherent; per-kernel
tolerance tests remain the correctness contract. Cumulative b=8 progress
this investigation: 273.5 → 411.2 → 416.9 tok/s at 2K (54% → 82% of vLLM).

**Follow-up 5: post-norm-fix re-profile (b=8, 2K, segki):**
elementwise 5.110 → **4.756 ms/step**, reduce 0.426 → 0.337 (eager-RMSNorm
reduce kernels gone: 10611 → 6451 calls), decode_attn 1.637. Elementwise +
reduce = **5.09 ms/step ≈ 26% of the 19.2 ms step** — still the whole gap to
vLLM (15.7 ms). Largest single contributors: two elementwise kernel types at
1.20 + 0.75 ms/step (~230 launches/step combined); remaining candidates:
rope, q/k norms (28 GDN layers dominate by layer count), GDN composite casts.
Next actionable: attribute those two kernel types to ops (source-level
correlation), then fuse the top one.

**Follow-up 6: elementwise-soup fusion plan** (target: soup 5.1 → ~2 ms/step;
b=8 2K from 82% to ~95% of vLLM). Soup inventory per step (structural
attribution, to be confirmed by measurement — see step 0):

| op | source | launches/step |
|---|---|---|
| RoPE (gather, rotate_half neg+cat, mul-adds) | 8 full-attn layers | ~50–80 |
| q_norm/k_norm (pow/reduce/rsqrt/mul) | 8 full-attn layers | ~40 |
| output gate (sigmoid+mul) | 8 full-attn layers | 16 |
| GDN in-proj cat + .contiguous() repack | 28 GDN layers | ~60–90 (likely the largest item) |
| GDN z-gated output norm | 28 GDN layers | ~85 |
| MLP silu+mul (composite GLU activation) | all 36 layers | 72 |
| bf16↔fp32 casts, mask-to-padding, misc | various | ~50+ |

Fusion steps, in order:

0. **Source-level attribution first**: wrap the decode forward with
   `record_function` scopes (rope / gdn_proj / gdn_norm / mlp_act / gate) or
   profile `with_stack=True`, and map the two largest generic elementwise
   kernels (1.20 + 0.75 ms/step) to ops before writing any kernel.
1. **RoPE + q/k norm + output gate into `decode_attn`**: q is loaded into
   registers anyway — l2norm and cos/sin rotation in-register at load time;
   K rotated during row load (cos/sin indexed by position, graph-safe via
   write_pos); epilogue multiplies `out * sigmoid(gate)` from the 2x-wide
   q_proj output.
2. **Batched `quad_gemv` writing the concatenated layout directly** — kills
   the GDN cat + repack. (Revises the "GEMV batching is worthless" verdict:
   the GEMM half was worthless, the elementwise copies it removes are not.)
3. **Batched `dual_gemv_silu_mul`** — same pattern for the MLP activation
   (kills 72 launches/step).
4. **Gated-norm variant of `fused_add_norm`** for the GDN z-gated output
   norm (~85 launches/step).

**Follow-up 7: fusion plan executed (steps 0–4) — 2 kept, 2 reverted by measurement.**

| Step | Outcome |
|---|---|
| 0 attribution (eager, record_function) | ranking: MLP composite ~12ms/step eager, GDN in_proj ~6.8, qk_norm ~3.5, rope ~2.2 (launch-inflated; ordering valid) |
| 1 q-norm + partial-RoPE + gate into decode_attn | **KEPT** — correct (contract tests full+partial rope; partial-RoPE discovery: rope_dim=64 of head_dim 256; warp-divergent-shuffle hang found & fixed; pybind at::Tensor defaults fragile → dual overloads). b=8 perf neutral (within noise), b=1 held (68.5/65.4 tok/s) |
| 2 batched quad_gemv (concat layout) | **REVERTED** — b=8 e2e −10% vs baseline |
| 3 batched dual_gemv_silu_mul | **REVERTED** — b=8 e2e −60% (row-split re-streams MLP weights ×8; weight-stationary variant spills registers at in_features=2560) |
| 4 gated_norm kernel (GDN z-norm) | **KEPT** — correct, small win, folded into final numbers |

Final b=8 state (2925b7198): **444.8 tok/s short / 409.4 2K** (vs 441/411
pre-plan) — the plan's elementwise wins are inside noise at b=8; the kept
fusions are structural (fewer eager ops, b=1 path cleaner). Bisection
methodology note: the bench script's "graphonly" timings ran on the
segki-injected shared model (both rollouts mutate one model instance), which
is how the GEMV regressions surfaced in both rows; commit-level bisection
with an idle-GPU launcher isolated them. GPU suite 22 passed; e2e text
coherent (near-tie drift only, step-0 identical).

Still open under this item: pre-concat + single GEMM vs 3× GEMM at b ∈ {4, 16}
(the concat-amortization hypothesis from item 1).

Test-matrix addition (from the concat-amortization discussion): at b>1 the
pre-concat argument gains teeth — the tail-wave (SM wave-quantization)
effect does not vanish as the op grows, and freshness is recoverable by
re-concat at each generate's re-capture. (N-dim tile quantization is a
non-factor for our shapes: 8192/1024 are exact multiples of the 128-wide
tile.) Include "pre-concat weights + single GEMM
vs 3× GEMM" at b ∈ {1, 4, 16} in the micro-bench matrix; at b=1 it is
noise (measured: 0.8% op-level ≈ 0.15% e2e, both variants at 91–92% of
DRAM spec), but the batch scaling is unmeasured.


## 4. TP=2 performance — FIRST MEASUREMENT (Qwen3.5-4B, greedy, b=8, 2×RTX 4080 SUPER)

| | short ctx | 2K ctx | vs vLLM |
|---|---|---|---|
| DS (eager, no graph/segKI) | 94.1 tok/s | 92.6 tok/s | **11.6% / 12.0%** |
| vLLM 0.31 TP=2 | **810.4** | **772.9** | — |
| (ref) DS single-GPU segKI | 444.8 | 409.4 | 83%/81% of vLLM single |

Findings:
1. **Graph capture + TP=2 is BLOCKED**: `_create_static_cache` uses the full
   model config's head counts (e.g. num_kv_heads=4), but AutoTP-sharded model
   produces 2 heads per rank → tensor size mismatch at prefill. The rollout's
   cache, attention kernel, and step kernel are all designed for the unsharded
   model.
2. **Eager (HF generate) + TP=2 works correctly** — the model generates valid
   text; performance is the unaccelerated baseline.
3. **vLLM TP=2 is faster than its own single-GPU** (810 vs 537 tok/s short):
   splitting the model halves per-GPU weight traffic; the communication cost
   is dominated by the bandwidth win at this model size.
4. **The 8.6× gap** is almost entirely "no graph capture / no segKI at TP=2",
   not a kernel efficiency issue.

To unblock TP=2 graph capture:
- TP-aware static cache creation (head counts ÷ tp_size)
- TP-aware decode_attn kernel (per-rank head mapping, GQA ratio adjustment)
- All-reduce handling in graph (or skip graph and use eager + custom kernels)
- AutoTP config wrapping (module.config becomes a dict after AutoTP)


## 5. TP=2 AllGather explosion — Qwen3.5 GDN block's colwise_gather_output tp_plan (found via profile)

Profile (b=8, 2K, TP=2): NCCL comm = 60.7% of GPU time.
AllGather 127.6/step (30.6ms) + AllReduce 43.6/step (16.2ms).

### Root cause (corrected after code verification)

Qwen3.5's official HF tp_plan marks the ENTIRE GDN linear-attention block
`colwise_gather_output` (in_proj_qkv/z/b/a + out_proj = 5 gathers/layer),
because HF's GDN implementation (conv1d, chunk scan, gating) needs full-width
inputs. engine prefers HF tp_plan over the AutoTP heuristic; tp_plan_converter
maps colwise_gather_output → COLUMN + gather_output=True; AutoTP executes it.

Qwen3.5-4B = 32 layers (24 GDN + 8 full-attn; earlier notes saying 36=8+28
were wrong). Reconciliation:
- GDN: 24 layers × 5 colwise_gather_output projections = 120 AllGather/step
- o_proj (8) + down_proj (32) rowwise = 40 AllReduce; lm_head ≈ +3.6
- MLP gate/up are plain `colwise` — NO gather (an earlier version of this
  note and of issue #8877 wrongly attributed gathers to the MLP; corrected)

### Classification: deliberate HF plan semantics, not an AutoTP bug

Correctness is fine (gather then compute then reduce). The cost is specific
to hybrid models: pure full-attn tp_plans (Llama family) have zero gathers,
so this never surfaced before hybrid + TP decode profiling.

### Optimization directions

1. Near-term: concat the 5 GDN projection outputs locally, one gather per
   layer (120 → 24 calls)
2. Upstream: head-sharded GDN — the math is per-head independent (depthwise
   conv, per-head recurrent state, per-head norms); needs GDN/FLA kernels to
   accept sharded head layouts, then zero gathers

Issue: deepspeedai/DeepSpeed#8877 (edited with corrected attribution).


## 6. External bench feedback — single-GPU rollout (file: FEEDBACK_single_gpu.md)

Third-party bench on PR head 6647fb300 + their local fixes. Setup differs
from ours: different prompt per row (not shared prefix), temperature 1,
bs 1/4/32/128, all prompts exactly 256 tokens (no padding), 1x A100-SXM4-40GB,
torch 2.14.1+cu130, transformers 5.19.0. Reference: SGLang 0.5.23 (graphs on).

Results (decode step time, seg-KI vs SGLang): bs=1: 81%, bs=4: 76%,
bs=32: 66% — gap GROWS with batch size.

Kernel coverage at bs>1: decode_attn / fused_add_norm / gdn_gates /
gated_norm ON; dual/triple/quad GEMV and decode_step_graph OFF (separate
GEMMs). SGLang/vLLM use merged gate|up and qkv weights — suggestion:
merged-weight GEMM or small-batch GEMV for bs<=32.

### TWO REAL BUGS reported (both still present in deafce7ba):

BUG A — decode_attn strided query: Qwen3.5 q_proj emits [query|gate]
per head, so query_states.reshape(batch, heads, head_dim) is a strided
view; the kernel indexes it as packed → heads > 0 read WRONG values.
Greedy output diverges from HF within 1-8 tokens at every batch size.
Their fix: .contiguous() before the kernel + reject non-contiguous q in
the wrapper. NOTE: our own uniform-prompt benches showed "near-tie"
divergence at position ~2 — that may have been this bug, not numerics.
Our tests never caught it because kernel tests allocate contiguous q and
the HF-parity tests exercise the grad-enabled (composite) path, not
decode_attn.

BUG B — transformers 5.19 GDN scan binding: no longer binds
chunk_/recurrent_gated_delta_rule onto each layer → fused GDN forward
raises AttributeError. Their fix: fall back to module-level
torch_*_gated_delta_rule functions.

Both need fixes on our branch; GPU verification deferred (no free machine).


## 7. Feedback fixes verified + grouped GEMM results (RTX 4080 SUPER, different-prompt bench)

### Bug fixes verified on GPU
- Strided-q fix: segKI vs eager HF, different random prompts per row,
  greedy, 32 tokens: match rate 1.000 (run 1) — the feedback's
  systematic 1-8-token divergence is gone. A second run (grouped GEMM
  gated off at bs=8) showed 0.934 with first diff mid-sequence in a
  single row — near-tie noise on random-token prompts, not the bug
  signature.
- transformers 5.19 scan fallback: not exercised (remote has 5.15).

### Grouped GEMM (route 2, no-copy merged weights) results
Different prompts per row, 256-token prompts, greedy, slope method:

| bs | segKI | graphonly | delta |
|----|-------|-----------|-------|
| 8  | 451.2 | 350.9 | **+28.6%** (grouped OFF at bs<16, attention/norm kernels dominate) |
| 32 | 766.2 | 670.9 | **+14.2%** (grouped GEMM on) |

Measured at bs=8: grouped GEMM LOST to cuBLAS (-3.9%) — M=32 tile with
only 8 active rows wastes 3/4 of accumulators. Gated to 16<=bs<=32
(commit 1924a0dfd).

### bs=32 profile (after grouped GEMM)
| class | ms/step | % |
|---|---|---|
| GEMM (cuBLAS) | 21.9 | 42.3% |
| grouped GEMM (ours) | 14.6 | 28.1% |
| memcpy | 5.1 | 9.9% |
| GDN scan/conv | 4.4 | 8.5% |
| elementwise | 3.6 | 6.9% |
| decode_attn (ours) | 1.0 | 1.8% |

Known follow-ups (kernel headroom, for the kernel track):
1. grouped_linear_kernel is FMA (no tensor cores): ~1-2 TFLOP/s
   effective vs cuBLAS ~50 — MMA upgrade is the big win.
2. Only triple_gemm fires at decode: MLP (dual) and GDN (quad) forwards
   receive 3D [B,1,H] input and the dim()==2 dispatch skips them —
   flatten at the dispatch to extend coverage (measure first: the slow
   FMA kernel could regress those paths).
3. memcpy 9.9%: partly the new q .contiguous() — a strided-read kernel
   variant would remove it.


## 8. WMMA grouped GEMM + full dispatch + strided decode_attn (RTX 4090 D)

Four-step follow-up to the profile findings (commit bd1967fc9 + fixes):

### Changes
1. MLP/GDN 3D dispatch: [B,1,H] decode activations flattened at the
   dispatch so dual/quad_gemm fire at 16<=bs<=32 (was: only qkv triple).
2. Grouped GEMM rewritten with WMMA tensor cores (m16n16k16 bf16, fp32
   accum; FMA version achieved ~1-2 TFLOP/s vs cuBLAS ~50). WMMA smem
   ldm must be a multiple of 8 elements — the +1 padding caused UB.
3. decode_attn strided-read: q/gate read through a head-stride param;
   the chunked [q|gate] views pass with no .contiguous() copy.

### Results (4090 D, different prompts per row, 256-tok, greedy, slope)

| config | bs=8 | bs=32 |
|---|---|---|
| segKI | **561.3** | **780.4** |
| graphonly | 435.9 | 596.8 |
| vLLM 0.31 | 668.0 | 1127.2 |
| segKI vs graphonly | +28.7% | **+30.7%** (was +14.2% pre-WMMA) |
| segKI vs vLLM | **84.0%** | **69.2%** |

Token-match vs eager (different prompts, greedy): 1.000.

vLLM note: hybrid GDN state cache caps max_num_seqs (needed
max_num_seqs=32, max_model_len=3072 on 24GB).

### Remaining bs=32 gap to vLLM (30.8%)
Not yet profiled on 4090 D; candidates from the earlier 4080 profile:
remaining separate cuBLAS GEMMs (down_proj, o_proj, lm_head — not in
grouped kernels by design), GDN scan/conv (8.5%), memcpy (~10% pre-fix).


## 9. Single-GPU experiment sweep (RTX 4090 D, transformers 5.15 + 5.19)

### Bug found & fixed during the sweep
b=1 qkv dispatch: splitting the triple_gemv/triple_gemm branches orphaned
the chunk/k_norm/transpose lines into the grouped branch only —
query_states unbound at b=1. Caught by the bs=1 leg (prior benches only
ran bs>=4). Fixed, verified: bs=1 92.3 tok/s.

### Threshold experiment (WMMA grouped GEMM at 2..15)
| bs | gate 16 (cuBLAS) | gate 2 (grouped) | verdict |
|----|------------------|------------------|---------|
| 4  | 304.5            | 128.6 (-58%)     | cuBLAS  |
| 8  | 557.4            | 243.6 (-56%)     | cuBLAS  |
| 32 | 775.9            | 774.7 (=)        | grouped |
Even with WMMA tensor cores the M=32 tile wastes 3/4 of its rows at
bs<=8; the 16..32 gate is CORRECT with WMMA-era data. bs=1: 92.3/92.4.

### Correctness
- MATCH bs=8: 1.000; bs=4: 0.766 — seed-variation test (7/123/999)
  shows random single-row near-tie cascades (one seed perfect), not a
  systematic bug (which would be stable across seeds, all rows).
- Sampling b>1 (temp 1): works.
- Mixed-length left-padded batch (8 rows, 180..256 tok): works.
- transformers 5.19.0: generation + 183 tests pass (Bug B fallback
  verified on the version that exhibits the missing binding).

### bs=32 profile (current code; classify fix: glu_gemm was miscounted
as cuBLAS — it contains "gemm")
| class | ms/step | % |
|---|---|---|
| grouped GEMM (grouped_linear 10.3 + glu_gemm 17.9) | 28.2 | 57.9% |
| cuBLAS (down/o_proj/lm_head) | ~9.3 | 19.1% |
| memcpy | 3.5 | 7.2% |
| GDN scan/conv | 3.2 | 6.6% |
| elementwise | 2.8 | 5.8% |
| decode_attn | 0.7 | 1.4% |
Grouped kernels move ~500MB weights/step; at ~900GB/s that is ~0.6ms
theoretical — the WMMA tiles are ~45x off bandwidth-bound (small tiles,
no vectorized loads). Largest remaining headroom = kernel engineering.

### bs=64/128: OOM on 24GB (KV cache + dual caches) — not measured.


## 10. Vectorized staging (uint4) — bs=32 BEATS vLLM (RTX 4090 D)

Fix #1 of the pipeline list: smem staging loaded 2 bytes/thread-iteration;
now 8 bf16 per uint4 (16B transactions), with wrapper guards (K % 8 == 0,
16B-aligned pointers). kn and smem strides are multiples of 8 by
construction.

| bs | before | after | delta |
|----|--------|-------|-------|
| 1  | 92.3   | 91.9  | = (GEMV path) |
| 4  | 304.8  | 300.2 | = (cuBLAS path) |
| 8  | 557.6  | 557.5 | = (cuBLAS path) |
| 32 | 775.9  | **1233.1** | **+58.9%** |

**segKI bs=32 = 1233.1 vs vLLM 1127.2 = 109.4% — faster than vLLM.**

Correctness after the change: 183 tests passed; MATCH bs=8 = 1.000
(bit-identical to the pre-change run — the vectorization only changes
load width, not accumulation order); bs=4 = 0.766 (same seed-7 near-tie
value as baseline, deterministic); sampling + left-pad unchanged.

Transaction width was the dominant inefficiency: 2-byte loads wasted
7/8 of each 16-byte memory transaction. Remaining pipeline fixes
(cp.async double buffering, KT 32->128) target the latency exposure.


## 11. GEMV vectorization (b=1 path) — +1.2% at bs=1

Same uint4 treatment for the b=1 GEMV kernels (dual/triple/quad) via a
shared dot_partial_v8 helper. bs=1: 92.3 -> 93.4 tok/s (+1.2%); other
batch sizes unchanged (they do not run GEMV). Correctness: 183 tests,
MATCH bs=8 = 1.000, all e2e sections bit-identical.

Why the gain is small here (vs +59% for the grouped kernels): the GEMV
loop was already warp-coalesced (32 lanes reading 32 consecutive bf16 =
64B contiguous per warp instruction), so the memory transactions were
mostly utilized; vectorization only cuts instruction count, and at bs=1
there are enough warps in flight to hide the issue latency. The grouped
kernels' per-thread scattered 2-byte staging was the pathological case.


## 12. K-side fusion + deeper grouped tiles — final results (RTX 4090 D)

### Item 2: K norm + partial RoPE + cache write fused into decode_attn
Kernel normalizes/ropes this step's K in registers, writes K/V cache
slots itself (each lane writes exactly the elements it reads back at
idx==pos: same-thread visibility, no fences; sibling q-head blocks
write identical values). Skips k_norm, rope helper, cache-update copy.

### Item 1 (step A): grouped tiles KT 32->128, N 32->64 (2 wmma/warp)
4x fewer k-chunk barriers; partial tails zero-filled. Warp layout bug
(cols 48-63 uncovered) caught by the kernel test, fixed.

Step B (cp.async double buffering) deferred: KT=128 double buffers need
~96KB smem > the 48KB static limit — requires dynamic-smem refactor,
tracked for the kernel-engineering follow-up.

### Results (different prompts/row, greedy, slope)
| bs | session start | K-fusion | +tiles | total gain | vs vLLM |
|----|---------------|----------|--------|------------|---------|
| 1  | 92.3          | 95.4     | 95.8   | +3.8%      | -       |
| 4  | 304.8         | 309.6    | 310.8  | +2.0%      | -       |
| 8  | 557.6         | 567.4    | 566.7  | +1.6%      | 84.8% (668.0) |
| 32 | 775.9         | 1246.4*  | 1322.0 | +70.4%     | **117.3%** (1127.2) |
(*includes the uint4 vectorization step recorded separately)

Correctness after every item: 183 tests + MATCH 1.000 at bs=4/8
(K-fusion also resolved the seed-7 bs=4 near-tie).

### Final bs=32 profile (31.2ms GPU busy, corrected classes)
grouped (ours) 11.0ms 35% | cuBLAS (down/o/lm_head) 9.3ms 30% |
memcpy 3.5ms 11% | GDN scan 3.2ms 10% | elementwise 2.6ms 8%
(calls 746->580/st via K fusion) | decode_attn 0.7ms 2%
Grouped bandwidth now glu 429 / linear 226 GB/s (cuBLAS ref 840) —
remaining pipeline headroom documented for the kernel track.


## 13. cp.async double buffering — bs=32 = 136% of vLLM (RTX 4090 D)

Grouped GEMMs now stage through cp.async into double buffers on dynamic
shared memory (grouped: KT=128, 61KB; GLU: gate+up tiles staged together
at KT=64, 65KB; one-time cudaFuncSetAttribute opt-in past the 48KB
static limit). Chunk n+1 loads while chunk n computes.

Two bugs during bring-up, both caught by the kernel test:
1. GLU out_s offset used 2*NT ws rows but the layout needs 4*NT
   (2 slots x gate+up) — outputs ran past the smem allocation.
2. GLU stage lambda computed k0 = c * GRP_KT (128) but the chunk size
   is GRP_KT_GLU (64) — chunk 1 staged nothing, half the dot product
   silently missing (mixed right/wrong values).

### Results (different prompts/row, greedy, slope)
| bs | tiles | +cp.async | session total | vs vLLM |
|----|-------|-----------|---------------|---------|
| 1  | 95.8  | 95.8      | +3.8%         | -       |
| 4  | 310.8 | 310.7     | +2.0%         | -       |
| 8  | 566.7 | 568.5     | +2.0%         | 85.1%   |
| 32 | 1322.0| **1533.7**| **+97.7%**    | **136.1%** |

Correctness: 183 tests + MATCH 1.000 at bs=4/8 after the fix.

### Final bs=32 profile (27.8ms GPU busy; corrected classes)
| class | ms | % |
|---|---|---|
| grouped (ours): glu 4.9 + linear 2.6 | 7.6 | 27.2% |
| cuBLAS (down/o_proj/lm_head) | 9.3 | 33.5% |
| memcpy | 3.5 | 12.6% |
| GDN scan/conv | 3.1 | 11.3% |
| elementwise | 2.6 | 9.3% |
| decode_attn | 0.75 | 2.7% |

Grouped bandwidth now glu 608 / linear 345 GB/s (cuBLAS ref 840) —
within 1.4x of cuBLAS; remaining headroom = cuBLAS GEMMs, memcpy,
GDN scan.


## 14. memcpy investigation — GDN recurrent-state copy is by-design (HF)

The 3.5ms/st DtoD memcpy (29.4 calls, avg 0.12ms = ~60MB each) traced to:
1. **GDN recurrent-state copy per step** (~24 layers x 33MB): HF's
   cache_utils.update_recurrent_state does
   `self.recurrent_states[idx].copy_(recurrent_states)` after every scan,
   explicitly "to preserve the static address for cudagraphs" — the FLA
   scan returns a NEW tensor; the copy moves it back into the stable
   cache buffer the captured graph reads. Cannot be removed by rebinding
   (Python runs once at capture; replays re-read the same buffer).
   Fix = an in-place recurrent-update kernel (read state, apply 1-token
   delta rule, write state in place) — kernel-track item.
2. Prefill cache-management copies amortized into per-step accounting
   (per-row KV copies at lines ~717) — one-time per generate.
3. The decode loop's own glue is tiny (~7 small ops/step: token copy,
   fill, mask set, argmax, eos).

elementwise follow-up: the GDN q/k repeat_interleave (v_heads/k_heads=2)
at decode materializes duplicated heads — the scan kernel could take the
GQA ratio instead; kernel-track item.


## 15. Remaining kernel-track items — two done, one negative result

### lm_head on the grouped kernel (+0.3%, noise-level)
single_gemm entry + install_lm_head decode routing (16<=b<=32). Neutral
in practice; kept (principled routing, no risk). MATCH 1.000.

### In-place GDN recurrent step: honest negative result
gdn_recurrent_step mirrors FLA T=1 exactly (in-kernel l2norm eps 1e-6,
q scaled K^-0.5, precomputed beta/g), state in smem, updated in place —
half the state traffic on paper (4 passes -> 2). Measured:
- bs=32: NEUTRAL (1538.8 vs 1538.3) — the HF copy-back it removes was
  fully overlapped by cross-layer pipelining; GPU-busy time is not wall
  time at high occupancy.
- bs=1: -8% (B*H=32 blocks starve 128 SMs; FLA's tiling wins).
Gated to B>=16 (kept for large batches / bandwidth-starved GPUs where
the overlap argument weakens). Correctness: MATCH 1.000 everywhere.
Lesson recorded: profile shares flag candidates; overlap decides wins.

### Skipped: scan GQA-ratio fusion (in-graph launches already cheap);
grouped-linear NT tuning deferred (compile-time tile change, uncertain).


### vLLM small-batch baselines (same GPU, same prompts, added later)
| bs | segKI | vLLM | segKI % |
|---|---|---|---|
| 1 | 95.8 | 101.6 | 94.3% |
| 4 | 310.6 | 351.1 | 88.5% |


### bs=16 comparison (grouped-GEMM lower boundary, M=16 = half-full tile)
segKI 1041.2 vs vLLM 929.0 = **112.1%** — already faster than vLLM at the
gate's lower edge. Full curve: 1: 94.3% / 4: 88.5% / 8: 85.1% /
16: 112.1% / 32: 136%. The crossover sits exactly at the grouped-GEMM
threshold (16..32); below it segKI rides cuBLAS composite and vLLM's
fused kernels win.


## 16. One wmma tile per warp + gate lowered to 2..32 — sub-16 now WINS

8 warps (TPB 256), each exclusively owning one 16x16 output tile instead
of iterating two serially. At M<16 the old layout made half-idle warps
walk two tiles; the fix doubles in-block parallelism exactly where the
tile is underfilled.

| bs | cuBLAS composite (old gate) | 8-warp grouped (gate 2..32) | delta |
|----|------------------------------|------------------------------|-------|
| 4  | 310.6                        | 321.7                        | +3.6% |
| 8  | 568.1                        | 592.6                        | +4.3% |
| 16 | 1041.2                       | 1059.3                       | +1.7% |
| 32 | 1538                         | 1563.9                       | +1.7% |

vs vLLM full curve after: 1: 94.3% / 4: 91.6% / 8: 88.7% / 16: 114.1%
/ 32: 138.7%. Near-tie at bs=8 (0.934) verified benign across seeds
(7/999 perfect, 123 single-row mid-sequence flip); 183 tests pass.


### Full comparison table (fresh run, all cells same-session, RTX 4090 D)
| bs | HF eager | HF+graph | segKI+graph | vLLM 0.31 |
|----|---------|----------|-------------|-----------|
| 1  | 21.7    | 80.5     | 95.9        | 101.6     |
| 2  | 41.5    | 138.5    | 162.8       | 181.7     |
| 4  | 84.7    | 255.4    | 310.4       | 351.1     |
| 8  | 169.8   | 417.8    | 566.8       | 668.0     |
| 16 | 336.8   | 657.2    | 1055.0      | 929.0     |
| 32 | 661.9   | 876.0    | 1563.7      | 1127.2    |


### Sampling (temperature=1) — free for both engines, RL/OPD-ready
| bs | segKI greedy | segKI temp1 | vLLM greedy | vLLM temp1 |
|----|--------------|-------------|-------------|------------|
| 1  | 95.8         | 94.4        | 101.6       | 101.5      |
| 8  | 567.3        | 569.6       | 663.3       | 663.0      |
| 32 | 1564.8       | 1568.7      | 1127.5      | 1128.7     |
Sampling runs outside the graph (softmax+multinomial on [BS, 248K]
fp32 logits); cost is within noise at every batch size for both
engines (vLLM fuses it; ours is eager but amortized).

Sampler correctness: 2000 independent draws from a fixed logit row;
top-10 empirical frequencies vs softmax probabilities max |dev| = 1.9%
(~2 sigma at n=2000, binomial-consistent). Temperature/top-k/top-p
implemented in _sample_from_logits.


### Full sampling table (temperature=1.0, all engines, fresh cells)
| bs | HF eager | HF+graph | segKI+graph | vLLM 0.31 |
|----|---------|----------|-------------|-----------|
| 1  | 20.8 (20%)  | 79.4 (78%)  | 94.6 (93%)   | 101.5 |
| 2  | 40.7 (22%)  | 139.1 (77%) | 163.3 (90%)  | 181.5 |
| 4  | 80.5 (23%)  | 256.5 (73%) | 311.9 (89%)  | 351.4 |
| 8  | 163.5 (25%) | 417.5 (63%) | 569.6 (86%)  | 663.0 |
| 16 | 321.0 (35%) | 659.2 (71%) | 1064.2 (115%)| 926.8 |
| 32 | 628.2 (56%) | 876.8 (78%) | 1567.3 (139%)| 1128.7 |
Sampling costs <1.5% for every engine (within noise of greedy).


## 17. TP=2 with the grouped-GEMM generation (2x 4080 SUPER, auto-watcher run)

Watcher grabbed the dual-GPU machine at 01:02; automated pipeline ran
tests -> bench -> profile. All 184 tests pass under torchrun TP=2.

### Bench (256-token distinct prompts, greedy, slope)
| config | bs=8 | bs=32 |
|---|---|---|
| segKI + graph | 533.2 | **1120.5** (first TP=2 bs=32 measurement) |
| graphonly | 451.9 | 823.1 |
| segKI delta | +18.0% | **+36.1%** |

### Profile (rank0, b=8, both ranks profiled to keep NCCL in sync)
| class | ms/step | calls/step | % |
|---|---|---|---|
| NCCL communication | 27.43 | 166.1 | 55.3% |
| GEMM (cuBLAS) | 13.31 | 232.3 | 26.8% |
| elementwise | 4.93 | 1137.9 | 9.9% |
| GDN scan/conv | 1.29 | 49.7 | 2.6% |
| decode_attn (ours) | 0.81 | 8.2 | 1.6% |
| other/reduce/memcpy | 1.9 | — | 3.9% |
| TOTAL GPU busy | 49.6 | | |

### KEY FINDING: grouped GEMM (and fused norm/gates) did NOT engage at TP=2
The profile shows no grouped_linear/glu_gemm/fused norm kernels — only
decode_attn engaged. The segKI gain over graphonly (+18/+36%) is from
decode_attn + whatever partial installs applied. The AutoTP-wrapped
module tree apparently fails the MLP/GDN install checks (candidates:
projection wrapper attributes, grad-path branching, or the fused-norm
install's layer detection). This is optimization item #1 for TP.

### TP=2 optimization list (by expected impact)
1. Make grouped_linear engage under AutoTP wrappers (diagnose install
   checks; expected recovery = the same kernel gains as TP=1: GEMM is
   26.8% cuBLAS + elementwise 9.9% partially fusable)
2. NCCL 55.3% (27.4ms) — the #8877 AllGather pattern (Qwen3.5 tp_plan
   colwise_gather_output on the GDN block) + o/down_proj AllReduces;
   the custom-partition-config workaround (GDN replicate) remains
   untested
3. Graph-splitting / communication overlap (multi-graph architecture
   from the earlier design discussion)


## 18. TP=2 grouped-kernel engagement — diagnosis trail (open)

Goal: make MLP grouped kernels engage at TP=2 (profile showed none).

### Confirmed so far
1. Duck-typed _find_fusable_projections (LinearLayer wrappers pass).
2. GDN segment gated off at TP>1: with the finder open, the fused GDN
   forward engaged and crashed causal_conv1d — the conv weight is
   channel-sharded and the fused forward's width assumptions are not
   shard-aware. GDN stays native under TP (follow-up: shard-aware GDN).
3. find_glu_segments required gate/up/DOWN all comm-free; down is a
   LinearAllreduce under AutoTP -> 0 segments. Fixed: only gate/up are
   fused (down is delegated to its own forward, whose AllReduce must
   run). After the fix the report shows 32/32 segments installed and a
   direct mlp(x) call routes through the kernel (probe ok=True).
4. The full generate() path STILL shows no grouped kernels in the
   profile and bench-identical numbers (523-533/1114-1120), i.e. the
   rollout generate does not reach the installed MLP forward even
   though the install is live. A generate-path trace script died
   silently (no traceback, host VM at 79GB — suspected host OOM),
   leaving the routing gap unresolved.

### Where to look next (narrowed)
- rollout _generate_graph forward call chain under the DeepSpeed engine
  wrapper (engine.forward -> module forward -> decoder layer -> mlp):
  verify which object's forward actually runs at capture time
- whether the engine's forward re-dispatches through a compiled/
  scripted path that bypasses the patched module __call__
- rerun the trace with host memory headroom (the OOM suspicion)

TP=2 bench stands: segKI 523-533 / 1114-1120 (bs8/32), +18%/+36% over
graphonly, gains from decode_attn only.


### RESOLUTION: the "routing gap" was a classifier illusion
The TP=2 profile script's classify() jumped from NCCL straight to "gemm"
— glu_gemm_kernel contains "gemm" and was silently bucketed as cuBLAS.
Proof: after install the GEMM bucket's calls/step dropped 232.3 -> 199.6
(= one launch saved per MLP layer). With the classifier fixed:

| class | ms/step | calls/step | % |
|---|---|---|---|
| NCCL communication | 28.68 | 166.1 | 56.2% |
| GEMM (cuBLAS) | 10.52 | 166.8 | 20.6% |
| elementwise | 4.88 | 1072.4 | 9.6% |
| grouped GEMM (ours) | 3.44 | 40.9 | 6.7% |
| GDN scan/conv | 1.28 | 49.7 | 2.5% |
| decode_attn (ours) | 0.81 | 8.2 | 1.6% |
| TOTAL | 51.06 | | |

40.9 calls/step = 32 MLP + 8 qkv grouped — exactly as designed.

### Why the bench was flat: TP halves N, halving block count
Per-rank MLP N = 4608 (of 9216) -> 72 blocks on 128 SMs = underfilled;
cuBLAS is already strong at that shape. The TP=1 grouped advantage was
N-driven and does not transfer at halved N. Fix (kernel-track): a
smaller-N_TILE compile-time variant for sharded shapes.

### TP=2 priorities (final, evidence-based)
1. NCCL 56.2% (28.7ms) — #8877 AllGather pattern + o/down AllReduces;
   custom partition config (GDN replicate) untested; graph-splitting
   with comm overlap is the structural fix
2. elementwise 9.6% (1072 calls/step — GDN glue at TP since GDN segment
   is gated off; shard-aware GDN forward would restore its fusions)
3. grouped N_TILE variant for halved shapes (recover the wash)
Also: bench-side lesson — the earlier "silent death" was rank1 exiting
early under torchrun (not OOM); both ranks must always participate.


## 19. GDN column-column-row re-partition at TP=2 — SUCCESS (+48%)

The #8877 analysis led here: HF's tp_plan marks the whole GDN block
colwise_gather_output (6 collects/layer = 144/step). Because GDN math is
per-head independent, in_proj colwise (no gather) + out_proj rowwise makes
the block rank-local with ONE AllReduce per layer.

### Four correctness bugs found and fixed along the way
1. Shard-aware dims: derive split boundaries/head counts from per-rank
   weight shapes, not the global config.
2. in_proj_qkv is a FUSED [q|k|v] block weight — a contiguous column
   split lands mid-block (rank0 got [all-q, all-k]!). Fix: keep it
   replicated; each rank slices its head ranges per block as zero-copy
   views (z/b/a are single-block and shard head-aligned).
3. conv1d is not a Linear — AutoTP leaves it full-width. The conv weight
   channels follow the [q|k|v] block order, so a rank's local weight is
   the cat of its THREE block ranges (cached per module), not a prefix.
4. A_log/dt_bias slices were a prefix on every rank — rank-indexed head
   ranges required.

### Results (2x 4080 SUPER, bs=32, 256-token distinct prompts, greedy)
| config | tok/s | vs vLLM TP=2 |
|---|---|---|
| segKI + stock tp_plan | 1114-1120 | 64% |
| **segKI + C-C-R GDN partition** | **1649.4** | **95.5%** (vLLM 1727.2) |
| vLLM 0.31 TP=2 | 1727.2 | — |

Correctness vs full-width eager: 0.969 (6/8 rows perfect, 2 rows single
near-tie flips — TP-sharded numerics vs full-width, same class as bf16
tolerance; first 8 tokens identical).
184 tests still pass (TP=1 unchanged; the shard-aware forward is a
superset via tp_size=1 defaults).

Collectives/layer: 6 -> 1 (out_proj AR); the GDN fused forward re-engaged
under TP (gate lifted). opbuilder interface: unchanged throughout.


### TP=2 bs=64: segKI WINS at large batch (vLLM plateaus)
| bs | segKI C-C-R | vLLM TP=2 | segKI % |
|---|---|---|---|
| 32 | 1649.4 | 1727.2 | 95.5% |
| **64** | **2106.2** | 1729.6 | **121.7%** |
| 128 | OOM (state) | 1731.5 | — |

vLLM TP=2 saturates at ~1730 tok/s from bs=64 onward (its GDN mamba
cache blocks scaling — max_num_seqs); segKI keeps scaling to 2106 at
bs=64. segKI's bs=128 blocker is the fp32 GDN recurrent state (~4GB
over 2 cards at 128 rows); bf16 state or paging would unblock.


### TP=2 C-C-R correctness at bs=64 (vs full-width eager HF)
rate=0.959, full_rows=56/64, 8 rows diverge (first-diff positions
scattered 0-15, cascade after). Same near-tie class as bs=8 (0.969):
TP-sharded bf16 numerics vs full-width eager — expected divergence, not
a systematic bug (systematic errors would flip ALL rows at step 0-1 and
never match; 56/64 rows are token-identical for 32 steps).


### TP=2 sampling (temp=1.0) — free, RL/OPD-ready
| bs | greedy | temp=1.0 |
|---|---|---|
| 32 | 1660.1 | 1667.2 |
| 64 | 2106.2 | 2102.6 |
Sampling cost within noise at every batch size under TP=2 as well
(softmax+multinomial outside the graph, amortized).


### TP=2 profile AFTER C-C-R (bs=32, rank0) + hardware finding
| class | ms/step | calls/step | % |
|---|---|---|---|
| NCCL | 15.96 | **66.0** | 42.9% |
| GEMM (cuBLAS) | 12.25 | 216.3 | 33.0% |
| elementwise | 3.27 | 620.6 | 8.8% |
| grouped GEMM (ours) | 3.18 | 40.9 | 8.5% |
| decode_attn (ours) | 0.80 | 8.2 | 2.2% |
| GDN scan/conv | 0.70 | 49.7 | 1.9% |
| TOTAL | 37.17 | | |

Collectives 166 -> 66 calls (C-C-R confirmed). NCCL now 15.96ms
(242us/call average — pure per-call fixed latency on non-P2P PCIe).

**Hardware finding**: this box has NO P2P (can_device_access_peer False
both ways, topology cross-CPU). vLLM's custom one-shot AR is impossible
here too — vLLM also rides NCCL on this machine. Its remaining edge at
bs=32 is therefore NOT custom-AR; candidates are fused-projection
boundaries (fewer AR-adjacent sync points), better overlap, and its
GEMM library choices. The custom-AR plan is hardware-gated: worth
revisiting only on P2P-capable pairs (NVLink or same-socket PCIe with
P2P). On THIS box the honest lever list: GEMM 33% (down/o_proj cuBLAS
at halved N), elementwise 8.8%, and NCCL per-call latency (no software
fix without P2P).


### TP=2 bs=64 profile status (infra-blocked, honest)
Measured step times: segKI 30.4ms vs vLLM 36.9ms (+21.7%). segKI bs=32
composition scaled to 30-32ms at bs=64 — consistent with measurement,
no new bottleneck. Both detailed profiles blocked: segKI prof variant
OOMs on a 60.62GiB full-sequence lm_head alloc (modeling line 1725,
logits_to_keep default; trigger path unpinned — bench itself runs);
vLLM EngineCore is a separate process (needs tensorboard trace tools).
