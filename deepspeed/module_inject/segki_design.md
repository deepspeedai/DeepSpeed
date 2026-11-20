# segKI Design Document

**Audience: AI coding agents (and humans who want the reasoning, not just the what).**

This document is the design guide for segKI. Agents and humans modifying
segKI are expected to read it before making changes, and the implementation
must stay in sync with it: any design-level change ships together with the
corresponding update to this document.

---

## 1. What segKI is

segKI injects hand-written, per-architecture-tuned kernels into
communication-free segments of hybrid models (e.g., Qwen3.5: full-attention +
gated-delta-net linear attention) for hybrid-engine rollout decode. segKI is
**accelerator-agnostic by design**: the contract is the opbuilder interface
(§2.1) plus the executable reference (§2.2), and any accelerator can implement
it with its own opbuilder backend (SYCL/XPU/... — see the porting guide in
§5). It is
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
| `capabilities`, `refresh_weights` | mgmt | backend-owned derived-state invalidation (`clear_glu_cat_cache` is its alias; optional — a stateless backend omits it, §2.4) |

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

### 2.4 The refresh_weights mechanism (PROVISIONAL — see #9109)

Derived caches (concat groups, conv slices, fp32 gate constants) are keyed by
weight identity/pointer. In-place syncs keep keys valid while values change.
The mechanism to drop stale ones is single-trigger, two-tier by which side of
the opbuilder boundary the cache lives on:

- **Python-side caches** (`_ki_*` module attributes): each injector registers
  a refresh callback **on the module it injects into, at injection time** —
  registration co-located with cache creation, so a new injection path
  *structurally cannot* forget its refresh logic. Module-level (not
  model-level): lifecycle automatic (callbacks die with the module), zero
  added cost (`segki_refresh` already walks `model.modules()`).
- **Backend-side caches** (inside the compiled extension): invisible to the
  Python registry by design — reaching in would expose internal state. The
  backend owns them and implements a uniform `refresh_weights()` opbuilder
  method (`clear_glu_cat_cache` becomes its alias). The dispatcher calls it
  once; what it clears and how is private knowledge. A stateless backend
  simply does not implement it (hasattr discovery): *no derived state, no
  refresh obligation; own a cache, own its invalidation.*

`segki_refresh()` (engine entry: `refresh_weights`) is the pure dispatcher
over both tiers. Known interim flaws, both tracked:

- the current centralized cache-name list vs distributed installers (the bug
  class #9109 fixes with the registered callbacks above);
- continuous-batching graphs bake derived-tensor addresses; the interim fix
  releases the graph on refresh (option (a)); option (b) — address-stable
  in-place refill — aligns better with the end state.

**End state, two tiers:**

1. **Mandatory: the refresh framework as fallback.** Refresh survives,
   wherever the direct-write assumption breaks:
   - **non-affine derived tensors** — e.g. quantized weight caches:
     absmax/scale is not an affine map, cannot be a sync target; invalidate
     and recompute on weight change;
   - **weight writes that bypass the sync path** — manual loads, EMA
     overwrites, test code mutating `param.data` — do not know about targets,
     so values go stale unless a callback fires.
2. **Optional: affine direct-write via an `affine_targets()` opbuilder
   method.** A backend that keeps affine derived caches may publish them as
   opaque target descriptors — (source weights, affine map, destination) —
   and the sync executor then maintains them directly. The map crosses the
   boundary; the layout does not, so derived tensors may live in the arena
   *or* in backend-private memory — host choice is implementation freedom.
   Costs the backend two promises: destination address stability across
   syncs (no realloc/rehash) and single-writer discipline (sync writes, the
   kernel only reads). Backends that do not implement it keep the
   refresh-and-recompute path; implementing it removes the recompute.

With tier 2, "nothing derived → nothing cached → nothing to refresh" holds
for sync-driven updates: affine targets are written in place, and the
registered callbacks for them degrade to no-ops.

## 3. Design decisions and their reasoning

### 3.1 Hand-written kernels; implementation language is the accelerator owner's choice

The default opbuilder implementation uses native kernel language (CUDA) — this
is deliberate: it leaves the liberty to each accelerator owner to decide the
language to implement segKI kernels. An op can be implemented by any of the
following, all could be wrapped behind the opbuilder mechanism:

a. native kernel language (the default opbuilder implementation);
b. PyTorch ops, then optimized by a compiler;
c. an accelerator-specific library;
d. a portable kernel language such as Triton or TileLang.

Accelerator owners pick the most appropriate way per op — the contract (§2.1)
only fixes the 10 exposed methods, not how they are built.

External components (libraries, frameworks) are legitimate **only in "bring
your own pointers" mode**: they may lend math, never take ownership of state
or weights. State slots (KV/recurrent caches) belong to the upstream cache
(§3.3) — their addresses are what CUDA graphs bake (§6) and what the
native/fallback switch shares per call (§3.7); a library that insists on
managing its own KV or weight buffers would turn every architecture seam —
graph validity, per-call degradation, module lifecycle, state verifiability
— into a bidirectional sync problem. Math can be outsourced; the seams
cannot.

### 3.2 Unfused arena layout (checkpoint lingua franca)

The arena speaks HF-checkpoint layout; concat derivation is segKI-internal
(e.g., GLU cat ~2GB/rank on 4B — measured, acceptable). Pre-fused layouts were
deferred for two reasons. First, they turn storage into a three-party contract
(trainer maps, reshard, kernels) for a one-time ~ms concat. Second, and more
fundamental: pre-fusion would puncture the opbuilder boundary itself. Fusion
grouping is internal state today; with a pre-fused arena it leaks into the
storage format — writers must know it, `weight.shape` (§2.3) stops being a
self-describing interface, the eager fallback (HF modules expecting natural
shapes) needs a second layout, and whichever backend's fusion choices got
baked into storage would lock out other backends' implementation liberty
(§3.1). The end state (§2.4) supersedes this cleanly via affine target maps —
fusion stays declared by segKI as an affine target, so the layout knowledge
never leaves the kernel side — revisit only with measured memory pressure.

### 3.3 Stateful layer slots: pass-through binding, not native state management

segKI targets hybrid models generally; any stateful layer variant (GDN today;
Mamba/RWKV-class linears later) follows the same slot strategy. When the
upstream implementation is already graph-safe by construction, our slot binds
it **by reference** (`__getattr__` forwarding) — zero state-management code,
single source of truth, graph addresses = the prefill buffers. The upstream
dependency surface = the same few protocol names the model code itself uses;
no wider. DSStaticGDNSlot is the first instance of this pattern, not a
GDN-specific design.

### 3.4 Collective and compute: separate ops, separate component

Collectives and compute are separate ops by design, at two levels:

- **In the graph**: the collective is graph-capturable and is captured as its
  own op in the same stream as the compute kernels — the one-shot P2P
  all-reduce uses static buffers and flag-based signaling, so once its lazy
  init has run before capture, its addresses never change and the captured
  graph stays valid. It is never fused into `decode_step_graph`: the
  one-shot AR's essence is waiting on peers (spin); fusing it into a compute
  mega-kernel would pin SMs in a wait state and bake the blocking point in,
  and the fused op would no longer be implementable per-accelerator behind
  the §2.1 boundary.
- **At the interface**: segKI's only touchpoint is the
  `dist.inference_all_reduce` hook — a contract, not a dependency on any
  implementation. Backends fill the slot however they load (builder,
  accelerator library, plain NCCL); transport and topology evolve without
  touching segKI.

### 3.5 Arch-adaptive routing is transparent to the opbuilder interface

`grouped_linear` routes by batch/weight-count/arch (flat GEMV at M=1; WMMA ≤
threshold; cuBLAS composite above; Ada keeps WMMA for GLU at M≤32, Ampere
prefers composite+concat-cache) — all inside the method, behind the §2.1
boundary. The interface exposes a single `grouped_linear` entry regardless of
what routes underneath, so callers never route and a porting backend is free
to route differently for its own silicon. Every Python call site goes through
`grouped_linear` (the interface cleanup, 38ee22cd6). Known wart:
return format is inconsistent (n=4 flat concatenates into one tensor; other
paths return per-weight lists) — callers use the `_as_concat` normalization;
fixing the format is a small breaking change for a quiet day.

### 3.6 Layout assumptions of injected paths must be asserted loudly

An injected forward's weight-layout expectations are violated silently, not
crashingly — wrong partitioning yields wrong numbers, not an error. So every
layout expectation is asserted at config time, raising with the fix pattern
in the message. First (transient) instance: the injected GDN forward slices
per-rank head ranges out of FULL in_proj blocks, so `in_proj_qkv/z/b/a` (and
conv1d) must stay replicated — AutoTP specs mark them `partition_type:
"skip"`; a guard raises if they are sharded. Per-rank in_proj sharding (the
better TP efficiency end state) retires this guard.

### 3.7 Injection is model-agnostic: structural discovery + config cross-check

Injection never dispatches on model names. Installers walk `model.modules()`
and match **structural signatures** — attribute sets (`in_proj_qkv/z/b/a`
identifies a GDN segment), protocol shapes (`q_norm`/`k_norm` as modules,
norms with weight+eps), base-class isinstance for collective points — then
cross-check the config's own declaration (`config.layer_types[i]` must say
`full_attention` before an attention installer may take the module). The
replacement installed is per **segment type**, not per model: the fused
forward mirrors that segment family's native flow. A model is supported iff
every segment type it contains has an injection path — new models reusing
known segments are near-zero cost.

Discipline: a structural match is necessary, not sufficient. A future model
with the same attribute names but different gating semantics would match
structurally and be **silently mis-injected** (wrong numbers, no error — the
same failure class as §3.6). Hence: every installer includes the config-level
identity cross-check, and a new model family must pass greedy parity (§6)
before its structural hits are trusted — structure grants injection
eligibility, parity confirms semantic equivalence.

### 3.8 Dual train/rollout without phase switches

Every native-kernel takeover is gated on `not torch.is_grad_enabled()`. No
inject/eject between training and rollout phases (RL contract test pins this).
The composite/fallback paths preserve autograd for training.

## 4. Lineage: how segKI relates to inference V1/V2

DeepSpeed shipped two prior inference stacks; both had to answer the same
question — who holds the model knowledge that maps a model to kernels — and
answered it differently:

- **V1** (`module_inject`'s replace_module path, kernel inject): replace module
  classes in the HF model. Model knowledge lives in a user-supplied
  `injection_policy` dict or in hardcoded per-model replacement tables.
  Weights are reshaped into the fused layout the kernels want at injection
  time — the pre-fused storage §3.2 rejects, learned from this lineage.
- **V2** (`inference/v2`, ragged engine): no injection — per-model-family
  policies (`model_implementations/<family>/`) instantiate a DeepSpeed-native
  implementation and map checkpoint parameters into it. Model knowledge is
  explicit, one policy directory per family.
- **segKI**: no model list at all. Injection discovers segments by structural
  signature (attribute sets, cross-checked against `config.layer_types`) and
  replaces forwards in place; model knowledge is pushed down to per-segment
  protocol contracts.

| dimension | V1 | V2 | segKI |
|---|---|---|---|
| model→kernel mapping | user dict / hardcoded class tables | per-family policy directory (model rebuilt) | segment structural discovery (model-name-free) |
| intrusion | replaces module classes | no injection: native impl + param mapping | in-place forward swap, modules untouched |
| weight layout | reshaped to fused layout at inject | checkpoint engine maps into native impl | checkpoint layout preserved; derivation internal (§3.2) |
| relation to training | inference-only | inference-only | dual train/rollout, no phase switches (§3.8); storage-agnostic + refresh |
| portability | CUDA-centric builders | torch-op composites, fairly portable | opbuilder contract + executable reference + `hasattr` degradation |
| cost of a new model | one replacement table / user config | one policy directory | near-zero if segments reused; new segment shape = reference-first kernel (§5) |

The essential difference is not the injection mechanism but the problem
domain: V1/V2 are pure inference engines that load weights once — segKI lives
in the RL train/rollout alternation, which is why it carries §2.3
storage-agnosticity, §2.4 refresh, and the arena/reshard handshake at all.
Conversely, V2's ragged continuous-batch scheduling is what segKI currently
approximates with graph capture. segKI's structural discovery + `layer_types`
cross-check is effectively V1's "trust the user's injection_policy" replaced
by "trust the model's own config declaration".

### Possible evolution: declarative segment manifests (a possibility, not a plan)

Pure structural discovery has known pressure points that grow with the model
zoo: same-name/different-semantics false hits, heterogeneous per-family config
declarations (cross-checks accumulating per-family knowledge — the registry
creeping back in through the side door), and no authoritative "which families
are verified" list. One candidate answer — recorded here as a possibility to
be weighed when the pressure is real, not a commitment — is an optional
per-family **declarative manifest**: the family binds its blocks to segment
types from the closed contract vocabulary, with pinned protocol versions.

The key property would be a trust inversion: the declaration becomes the
primary judgement, and structural discovery is demoted to verifying that
declared segments look like what they claim (mismatch fails loudly). Models
without a manifest keep the structural-only path unchanged — the manifest is
a layer of determinism on top, not a new admission requirement.

In lineage terms this borrows V2's knowledge management (explicit, curated,
versioned, independently testable per family) on top of V1's injection
mechanism (in-place forward swap, model untouched), while staying distinct
from both: mappings target a closed vocabulary with mathematical definitions
(§2.1/§2.2) instead of arbitrary classes; the artifact is optional and
layered instead of a hard prerequisite; dual train/rollout and checkpoint
layout preservation are unaffected. Declared layout expectations (the §3.6
guard data, today scattered in AutoTP specs) have a natural home here too.

If pursued, the same caveats apply as to any registry: manifests rot
(upstream refactors outpace declarations — version pins must turn staleness
into loud errors), and parity (§6) remains the gatekeeper — a manifest grants
admission, never correctness.

## 5. Agent extension guides

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

### Adding a new model
1. Audit its segment composition (§3.7): every segment type needs an
   existing injection path. If all segments are known types, wiring is
   config-only — no code.
2. For each structural hit, verify the config identity check matches the
   model's `layer_types`/`block_type` declarations; extend the cross-check
   if the family declares itself differently.
3. If the model redefines a segment's math (new attention variant, new
   linear-attention recurrence), that is a new op family: follow "Adding a
   kernel" above — reference first, kernel second, new `m.def`/opbuilder type
   only if it joins the portable contract.
4. Gate support on greedy parity (§6) for the new family — structure granted
   injection; parity confirms semantics.

## 6. Verification pyramid

1. **Reference consistency** (unit, any device): kernel vs `kernel_reference`,
   multiple shapes/batches, bitwise where layout permits.
2. **Contract tests**: RL no-switch loop, graph-capture ordering, arena views.
3. **Greedy parity**: segKI+graph generate == HF `model.generate` greedy,
   b=1..8, per-row match rates (this is the harness that surfaced #9017 —
   graph-capture batch-size bug; keep it in CI-adjacent reach).
4. **End-to-end train/rollout loop**: train → weight sync (in-place write) →
   refresh → generate, verified against a fresh injection. This is the level
   that catches integration bugs no unit test can (e.g. stale derived caches,
   graph/address invalidation) — the storage-agnosticity and refresh
   contracts (§2.3, §2.4) are only proven here.
