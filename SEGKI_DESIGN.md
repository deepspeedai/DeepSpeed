# segKI Design Document

**Status: draft — to be submitted after PR #8602 merges.**
**Audience: AI coding agents (and humans who want the reasoning, not just the what).**

This document is the design guide for segKI. Agents and humans modifying
segKI are expected to read it before making changes, and the implementation
must stay in sync with it: any design-level change ships together with the
corresponding update to this document.

---

## 1. What segKI is

segKI injects hand-written, per-architecture-tuned CUDA kernels into
communication-free segments of hybrid models (e.g., Qwen3.5: full-attention +
gated-delta-net linear attention) for hybrid-engine rollout decode. It is
storage-agnostic: weights are read through `param.data` wherever it points
(including arena views bound by the resharding side, #9045), and the only
sync-time obligation is dropping derived-weight caches (`segki_refresh()`).

Layer map:

```
csrc/module_inject/fused_glu.cu      CUDA kernels (compute) + dispatcher
csrc/comm/custom_collective.cu       one-shot P2P all-reduce (PR #8997 owns it)
op_builder/fused_glu.py              build description (JIT/prebuilt)
op_builder/custom_collective.py      build description (#8997)
deepspeed/ops/module_inject/fused_glu.py   runtime lazy loader (process singleton)
deepspeed/module_inject/segment_ki.py      injection: installers, fused forwards,
                                           segki_refresh, TP guards
deepspeed/module_inject/kernel_reference.py  executable specification (7 fns)
deepspeed/utils/static_cache.py           graph-capture KV cache + GDN pass-through
deepspeed/runtime/rollout/hybrid_engine_rollout.py   engine: prefill/graph/continuous
```

## 2. The contracts (in order of authority)

### 2.1 The opbuilder interface — 10 exposed methods

The `.cu` file exposes exactly 10 `m.def` entries. Internal kernels (the gemv/gemm
variants the dispatcher routes to) are **not** exposed — routing is the
dispatcher's job and other backends must not see internal kernels leak through
the boundary.

| method | kind | notes |
|---|---|---|
| `grouped_linear` | compute | single entry for ALL GEMV/GEMM paths; flat / WMMA / composite routing inside |
| `decode_attn` | compute | single-query attention, GQA, fused partial-RoPE + q-norm + output gate |
| `gdn_gates` | compute | beta/g gating |
| `gated_norm` | compute | z-gated RMSNorm (multi-line lambda m.def — grep accordingly) |
| `fused_add_norm` | compute | residual add + RMSNorm |
| `gdn_recurrent_step` | compute | in-place recurrent update (TP=1 path; FLA wins at TP) |
| `decode_step_graph` | compute | single-token full step incl. argmax (b=1) |
| `repeat_qk`, `silu_mul` | glue | have `hasattr` eager fallbacks |
| `capabilities`, `clear_glu_cat_cache` | mgmt | no reference needed |

A backend porting segKI implements these methods — nothing else is the contract.

### 2.2 kernel_reference.py — the executable specification

Exactly 7 functions, one per compute interface method (gdn_recurrent_step's
oracle is FLA's public API; decode_step_graph is contract-tested). Three roles:
(1) specification for a porting agent, (2) runtime fallback when no kernel,
(3) test oracle. **Tests anchor on these, never on inline math** — when a
reference was added for `repeat_qk`, it caught a real stride bug (B*S stride
vs Hk stride, masked by B=1) that months of e2e parity had missed.

### 2.3 Weight shape is the interface

Kernels derive per-rank dimensions from `weight.shape` and read through
`data_ptr()`. The storage underneath (regular params, arena views, resharded
buffers) is invisible. This is what makes the arena integration zero-code on
our side: `param.data = arena_view` just works, and CUDA graphs stay valid
because in-place writes keep addresses.

### 2.4 The refresh handshake (PROVISIONAL — see #9109)

Derived caches (concat groups, conv slices, fp32 gate constants) are keyed by
weight identity/pointer. In-place syncs keep keys valid while values change.
`segki_refresh()` drops them. Known interim flaws, both tracked:

- centralized cache-name list vs distributed installers (the bug class #9109
  fixes with per-module registered callbacks — module-level registry, not
  model-level: registration co-located with injection, lifecycle automatic,
  traversal already exists);
- continuous-batching graphs bake derived-tensor addresses; the interim fix
  releases the graph on refresh (option (a)); option (b) — address-stable
  in-place refill — aligns better with the end state.

**End state (agreed with the resharding side):** derived tensors are affine
maps of the weights, so the sync writes them directly as targets
(#8575/#8706 fused target maps). Nothing derived → nothing cached → nothing
to refresh. The registered-callback framework composes: affine targets
register no-ops.

## 3. Design decisions and their reasoning

### 3.1 Hand-written kernels, no torch.compile in this PR

Compile was tried (route A: torch.library registration + `torch.compile` on the
GDN forward). Result: **-2.3%**, zero Triton kernels generated — the forward's
Python-object dependencies (`cache_params`) graph-break before any fusion.
Lesson recorded: compile's value depends on the Python structure being compiled,
not on compile itself (vLLM wins with it after rewriting model code to be
compile-friendly; DeepCompile uses the same custom-op pattern). If revived:
purify the forward to tensor-only args first; the registration layer pattern is
in git history (`segki_torchlib.py`, removed in 47d426d95).

### 3.2 Unfused arena layout (checkpoint lingua franca)

The arena speaks HF-checkpoint layout; concat derivation is segKI-internal
(e.g., GLU cat ~2GB/rank on 4B — measured, acceptable). Pre-fused layouts were
deferred: they turn storage into a three-party contract (trainer maps, reshard,
kernels) for a one-time ~ms concat. The end state (§2.4) supersedes this
cleanly via affine target maps — revisit only with measured memory pressure.

### 3.3 DSStaticGDNSlot: pass-through, not native GDN state management

HF's `LinearAttentionLayer` is already cudagraph-safe by construction. Our slot
binds it **by reference** (`__getattr__` forwarding) — zero state-management
code, single source of truth, graph addresses = the prefill buffers. HF-version
dependency surface = the same 6 protocol names the model code itself uses;
no wider.

### 3.4 Comm split: custom_collective lives in #8997

The one-shot P2P all-reduce was extracted to its own PR with a cleaner shape
(builder-based load behind the accelerator contract, lazy init inside the
`inference_all_reduce` hook, collective failure agreement, bounded spin
handshake — safe on fabrics where flag protocols fail). segKI only consumes the
hook. **Do not re-inline it.**

### 3.5 Arch-adaptive routing stays inside the dispatcher

`grouped_linear` routes by batch/weight-count/arch (flat GEMV at M=1; WMMA ≤
threshold; cuBLAS composite above; Ada keeps WMMA for GLU at M≤32, Ampere
prefers composite+concat-cache). Callers never route — every Python call site
goes through `grouped_linear` (the interface cleanup, 38ee22cd6). Known wart:
return format is inconsistent (n=4 flat concatenates into one tensor; other
paths return per-weight lists) — callers use the `_as_concat` normalization;
fixing the format is a small breaking change for a quiet day.

### 3.6 TP guards: in_proj replicated under TP>1

The injected GDN forward slices per-rank head ranges out of FULL in_proj
blocks — `in_proj_qkv/z/b/a` (and conv1d) must stay replicated. AutoTP specs
must mark them `partition_type: "skip"` (see the e2e script's TPSPECS). The
guard raises loudly with the fix pattern in the message; keep it loud.

### 3.7 Dual train/rollout without phase switches

Every native-kernel takeover is gated on `not torch.is_grad_enabled()`. No
inject/eject between training and rollout phases (RL contract test pins this).
The composite/fallback paths preserve autograd for training.

## 4. Agent extension guides

### Porting to a new accelerator (SYCL/XPU/...)
1. Read the 7 `kernel_reference.py` functions — that is the entire math contract.
2. Implement the 10 interface methods behind a builder
   (`get_accelerator().create_op_builder(...)` discovery).
3. `hasattr` discovery means partial implementations degrade to eager fallback
   — ship the high-value ops first (grouped_linear, decode_attn).
4. Validate against the reference tests (they are device-agnostic) and the
   greedy-parity harness (§6).

### Adding a kernel
1. Reference function in `kernel_reference.py` FIRST.
2. Direct reference-matched unit test NEXT (this ordering caught repeat_qk).
3. Then the CUDA kernel + dispatcher routing.
4. No new `m.def` unless it joins the portable contract.

### Adding an injection path
1. Installer sets `_ki_*` attributes on the target module.
2. Register the path's refresh callback per #9109 (module-level) — the
   centralized list is the bug class, do not extend it.
3. Wire the `hasattr` eager fallback (composite path must keep autograd).

## 5. Known pitfalls (each one bit us)

| pitfall | symptom | remedy |
|---|---|---|
| JIT extension cache staleness | old kernels after `.cu` edits; illegal memory access | `rm -rf ~/.cache/torch_extensions/*/fused_glu*` |
| Shallow git clones | "refusing to merge unrelated histories" on master merges | `git fetch --unshallow origin` |
| RunPod-class broken NCCL | hangs at transport setup; "Data inconsistency" misfires | infra problem, not code — do not weaken upstream checks (we reverted one such weakening) |
| transformers version drift | FLA per-layer binding (5.19→5.21), cache protocol (+1 semantics from #8852) | pin expectations to protocol names, verify parity on version bumps |
| `write_pos` off-by-one | mis-roped keys during decode | `get_seq_length()` counts the slot being written (wp+1); positions use `get_query_offset()` (wp) — and our decode paths pass `position_ids`/`cache_position` explicitly so neither feeds RoPE |
| B=1 masking in kernels | wrong rows for b>1 (stride bugs) | reference tests at multiple batch sizes, bitwise |

## 6. Verification pyramid

1. **Reference consistency** (unit, any device): kernel vs `kernel_reference`,
   multiple shapes/batches, bitwise where layout permits.
2. **Contract tests**: RL no-switch loop, graph-capture ordering, arena views.
3. **Greedy parity**: segKI+graph generate == HF `model.generate` greedy,
   b=1..8, per-row match rates (this is the harness that surfaced #9017 —
   graph-capture batch-size bug; keep it in CI-adjacent reach).
4. **ZeRO-3 e2e** (`DS_RUN_ZERO3_E2E=1`): train 2 steps → gather → in-place
   shard write → refresh → generate == fresh injection, bitwise both ranks.
   Note: needs a real optimizer in config (DummyOptim builds an offload
   parameter manager) and `engine.backward()` (not `loss.backward()`).
5. **Perf gates**: TP=2 bench per topology; vLLM same-box comparison for
   context (4080-class PCIe: segKI ≥ vLLM; A100 NVLink: 6-7 成 — the gap is
   attention MMA + glue, profiled but parked).

## 7. Open threads

- #9109 registered refresh callbacks (follow-up PR; this doc §2.4)
- Continuous-graph vs refresh semantics: interim (a) landed; (b) pending #9109
- Affine direct-write end state (with #9045; wiring lands on their side)
- grouped_linear return-format normalization (§3.5 wart)
- Attention MMA + glue remaining gap on A100 (parked, needs re-profile)
