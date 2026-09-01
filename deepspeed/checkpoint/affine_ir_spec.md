# Affine IR for Universal Checkpoint

Specification for the geometric shard description in `deepspeed/checkpoint/affine.py`.

Design discussion: [#8252](https://github.com/deepspeedai/DeepSpeed/issues/8252),
sub-issue of [#8230](https://github.com/deepspeedai/DeepSpeed/issues/8230). Measurements
here were taken against `master` at `92843ad70`.

Code is referred to by symbol rather than line number, so that this document does not go
stale every time a file above it shifts.

---

## 1. Scope and boundary

Universal Checkpoint stores every parameter as one full logical tensor `F`. Getting into
and out of that form currently requires knowing *what a parameter means* — is it a
vocabulary embedding, is it row-parallel, does it hold fused sub-parameters — and that
knowledge is encoded as regex lists over parameter names. This spec replaces the meaning
with geometry.

**The rule this document follows throughout:**

> The IR states what is true. The planner decides what to do.

Everything that is a fact about how bytes are laid out belongs in the IR. Everything that
is a choice — which replica to read from, what order to move things in, where to place a
tensor next — belongs to the caller. Section 4 applies this test to every key in the
current schema, and two of them (`output_shape`, `target_partition_shape`) fail it: they
describe a destination, not a fact about the source, and so they are not part of the IR.

**Non-goals.** This spec does not change how any layer partitions its weights at runtime.
It changes only how that partitioning is *described*. Every layout in tree today keeps the
exact bytes it has on every rank.

---

## 2. The representation

### 2.1 Piece

A **piece** is a block of elements, described by where it sits in the full tensor `F` and
where it sits in the shard:

```python
Piece = (shape, source_offset, source_strides, dest_offset, dest_strides, locations)
```

- `shape` — the extent of the block, **shared by both sides**: a piece holds the same
  elements in the same arrangement wherever it lives
- `source_offset`, `source_strides` — where the block sits in `F`
- `dest_offset`, `dest_strides` — where it sits in the rank's shard
- `locations` — the set of ranks holding this piece
- `scale` — the factor the shard holds the block by: `shard == full * scale`

`locations` is not the same thing as which rank's shard a piece belongs to. A map lists
pieces per rank because that is what reading or writing one shard needs; `locations` says
which *other* ranks hold identical data for that same block. For an ordinary sharded
piece the two coincide. For `bigcodetype`'s replicated kv block they do not: the block
appears in every rank's list, each entry naming all four ranks, and a converter reads it
once from whichever rank is cheapest.

Each side is exactly `torch.as_strided`'s argument list, so a piece is directly executable
with no interpretation step, in either direction.

`scale` exists because a row-parallel layer whose output is sum-all-reduced pre-divides its
replicated bias by the world size, so that summing the outputs adds the bias exactly once.
The divisor changes with the world size, so a checkpoint that does not record it cannot be
restored at a different TP degree without a rule naming which parameters are biases — which
is the kind of semantic category this IR exists to remove. §4.3 draws the line between this
and the average op.

**`scale` applies to the parameter, not to everything stored beside it.** A checkpoint also
carries optimizer state, and Adam's moments live in the parameter's *scaled* coordinate:
scaling a parameter by `s` scales its gradient by `1/s`, so the first moment carries `s⁻¹`
and the second `s⁻²` where the parameter carries `s`. A converter that applies the
parameter's own factor to all three corrupts the optimizer state and changes the trajectory
after a resume. The map records the geometry and the factor; the caller says which power of
it applies to the tensor being moved.

**Homogeneity.** A piece must cover elements that are all held by the same set of ranks and
all carry the same scale. This is what makes `locations` exact rather than advisory, and it
constrains merging: see §3 (P5) and §6.3.

The shard side has to be described explicitly. It is tempting to say a shard is simply its
pieces concatenated in order, but that is false for a **column** split, where the shard
interleaves its pieces row by row rather than appending them. An implementation built on
the concatenation assumption reproduces row-split layouts correctly and silently
transposes column-split ones.

### 2.2 Parameter map

A parameter's map pairs each rank's shard shape with the pieces that fill it:

```python
ParamMap = { logical_shape, { rank: (shard_shape, [Piece, ...]) } }
```

Because both ends of every piece are affine views, the two directions are the same loop
with the copy reversed — `extract` writes `source_view -> dest_view`, `rebuild` writes
`dest_view -> source_view`. Neither direction needs to know what the parameter *means*.

**Implementation note.** Piece offsets are storage offsets, because that is what
`as_strided` takes. A shard handed in by a caller is frequently a *view* into a larger
buffer — splitting a tensor produces exactly that — and its elements then begin partway
into that storage. Applying a piece to such a tensor reads from the wrong place, with no
error raised. An implementation must normalise a shard to a buffer starting at its first
element before applying pieces to it.

### 2.3 The two maps

- `M_s` — the **source** map: the layout the checkpoint was written from. Conversion to
  universal form is `M_s⁻¹`.
- `M_t` — the **target** map: the layout being restored into.

Restore is `M_t`. Phase 2 — moving weights between two live topologies without ever
materialising `F` — is `M_t ∘ M_s⁻¹`.

### 2.4 Why it is a relation, not a function

Under replication, one position in `F` lives on several ranks, so `M_s⁻¹` maps one element
to a *set* of locations. This is deliberate. A function would have to name a single owner,
and naming an owner is a decision — the cheapest source depends on topology, link
bandwidth, and what else is in flight, none of which the checkpoint format knows. The IR
records that all of these ranks hold the byte; the planner picks one.

This is the same argument that removes the average/combine op (§4.3): a reduce is not
invertible, so keeping it would make `M_s⁻¹` undefined for part of the language.

---

## 3. Properties

For a parameter map to be usable it must satisfy:

**P1 — Coverage.** The union of all pieces over all ranks covers every element of `F`
exactly once or more. If some element is covered by no piece, `F` cannot be rebuilt.

**P2 — Consistency under replication.** Where pieces from different ranks overlap, the
underlying data is identical. This is what makes picking any one of them safe, and it is
checkable — the current converter already asserts it for replicated parameters
(the replicated branch of `merge_tp_slices`).

**P3 — Invertibility.** A piece may carry an invertible elementwise map — a non-zero
`scale` — but never a reduction over several source elements. Given P1–P3, `M_s⁻¹` is total
and well defined as a relation.

**P4 — Composability.** `M_t ∘ M_s⁻¹` is again a set of affine pieces, so a transfer plan
can be computed without materialising `F`. This is not assumed: §8.2 measures it against
the real partition functions for every in-tree layout. It holds, but only for a composer
that folds pieces in N dimensions and groups them by source rank — §8.2 states both as
normative, because a composer missing either produces a correct map whose size grows with
model size.

**P5 — Homogeneity.** Every element a piece covers is held by the same set of ranks, and
that set is the piece's `locations`. This makes `locations` authoritative: if pieces `P`
(locations `L`) and `Q` (locations `M`) share an element `e`, then P5 gives
`holders(e) = L` and `holders(e) = M`, so `L = M`. **Two pieces with different location
sets cannot overlap.** A planner can therefore trust one rank's map about what is shared,
instead of scanning every other rank to discover replication it was not told about.

P5 does not forbid overlap. Two pieces with the *same* location set may overlap partially;
both already declare the same sharing, so nothing is hidden.

P3 is the property that forces the average op out of the language, and the property that
vocabulary truncation currently violates (§8).

---

## 4. Lowering: every current key

This is the compatibility contract. Every key that exists today must have a defined image
in the IR, or be explicitly declared out of scope.

### 4.1 Convert path — `merge_tp_slices` in `ds_to_universal.py`

The current implementation is a branch chain over regex-matched name categories. Each
branch lowers to a piece list:

| Branch (`ds_to_universal.py`) | Current behaviour | Lowers to |
|---|---|---|
| `TP_REPLICATED_PARAMETER_PATTERNS` | take `slices[0]`, assert all equal | **1 piece** covering all of `F`, `locations` = every rank |
| `PARAMETER_TO_AVERAGE_PATTERNS` | `sum(slices) / len(slices)` | **not a view** — see §4.3 |
| `PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0` | chunk each slice in 2, cat groupwise | **2 pieces/rank** |
| `PARAMETER_WITH_SUB_PARAMS` + `SUB_PARAM_SHAPE` | N sub-params, per-rank widths | **N pieces/rank** |
| default | `cat(slices, dim=cat_dim)` where `cat_dim = 1` if `PARAMETER_WITH_ROW_PARALLELISM_PATTERNS` else `0` | **1 piece/rank**, split along `cat_dim` |
| `VOCABULARY_PARAMETER_PATTERNS` | `param[:original_vocab_size, :]` | **not a view** — see §8 |

Two keys are written *into* the produced checkpoint rather than read from metadata, and
both become redundant:

| Key | Why it disappears |
|---|---|
| `CAT_DIM` | absorbed into `strides` — a row split and a column split differ only in stride |
| `PARAM_N_SUB_PARAMS` | absorbed into the length of the piece list |

`SUB_PARAM_SHARD_WIDTHS` (UCP 0.4, #8185) is the closest thing in tree to the IR already:
it records the physical extent each rank holds of each sub-parameter. It lowers directly
to per-piece `shape` entries, and it is the key that makes uneven splits expressible.

### 4.2 Restore path — `_resolve_autotp_partition` in `universal_checkpoint.py`

| Meta key | Current use | Lowers to |
|---|---|---|
| `replicated` | return all of `full_hp_param` | 1 piece, all of `F` |
| `partition_dim` | axis to narrow on | the axis whose piece offsets vary |
| `logical_shape` | `full_hp_param.view(...)` | `shape` of `F` |
| `sub_param_sizes` / `sub_param_shape` (:82, :81) | `_narrow_sub_params` | N pieces |
| `sub_param_shard_widths` | per-rank widths inside `_narrow_sub_params` | per-piece `shape` |
| `partition_sizes` | `narrow(dim, sum(sizes[:rank]), sizes[rank])` | 1 piece, offset = prefix sum |
| *fallback* (:141-148) | `full_view.chunk(world)[rank]`, asserts divisibility | 1 piece, even split — **the assert disappears**, because uneven extents are expressible |
| `unsupported_reason` (:91-93) | raises `RuntimeError` | **eliminated** — see §5 |

`output_shape` and `target_partition_shape` are carried in the restore meta
(by `_build_param_uc_restore_meta`) but describe the *destination*. By the boundary rule they are planner
inputs, not IR. They stay where they are; the IR does not absorb them.

### 4.3 Reductions, and why `scale` is not one

`PARAMETER_TO_AVERAGE_PATTERNS` is the one branch that cannot be expressed: an average is a
reduce over several source elements, and it is the one branch that would break P3 for the
whole language.

The distinction that matters is invertibility, not arithmetic. A reduce is `N -> 1` and
destroys information, so `M_s⁻¹` is undefined on it. A scale is `1 -> 1` and is a bijection
for any non-zero constant, so `M_s⁻¹` is the reciprocal. The IR therefore admits an
invertible elementwise map and refuses a reduction.

Dropping it is safe in tree. The constant has exactly three references repo-wide — its
declaration (in `constants.py`), its import, and its `.get` in the converter — and **no
producer anywhere in the repository** writes the key. Nothing in tree can reach that
branch. This matches delock's reading in #8230 that it papers over a historical rank-drift
issue rather than describing a real layout.

---

## 5. Coverage: the layouts #8185 marks unsupported

#8185 introduced `AUTOTP_UNSUPPORTED_PARAMETER_PATTERNS` — layouts whose partitioning the
current schema cannot describe, which conversion therefore refuses. If the IR is to
replace the semantic categories, it has to cover these. This was tested rather than
assumed.

**Method.** Per layout, per rank: run the real DeepSpeed partition function on a marker
tensor to recover which element of `F` each shard element came from; compress that index
map into maximal runs to obtain pieces; then re-run the same partition function on
**independent random data** and check that those pieces, materialised with
`torch.as_strided`, reproduce the shard bit-exactly. The last step is the actual test — if
pieces derived from markers reproduce a random tensor's shard, the mapping is a pure view
(P3), not something data-dependent.

**Result — all four are affine-expressible.**

| Layout | Reason string in `master` | `F` | tp | Pieces/rank | Covers `F` |
|---|---|---|---|---|---|
| `bigcodetype` (`fused_LinearLayer`, `fused_LinearLayer`) | "interleaves or replicates blocks … cannot be reassembled from the shards" | 48×8 | 4 | 2 | yes |
| `codegentype` (same class) | same | 24×8 | 2 | 11 | yes |
| Yuan value, dim 0 (`Yuan_LinearLayer`, `Yuan_LinearLayer`) | "selects noncontiguous head groups … cannot currently describe" | 32×8 | 2 | 2 | yes |
| Yuan o_proj, dim 1 (`Yuan_LinearAllreduce`, `Yuan_LinearAllreduce`) | same | 8×32 | 2 | 2, strides `(32, 1)` | yes |

Piece count is bounded by block structure, not model size — CodeGen holds at 11
pieces/rank while rows-per-rank grows 32×:

```
 hidden   F rows   tp  kv_heads  rows/rank  pieces/rank
      8       24    2         8         12           11
     16       48    2         8         24           11
     32       96    2         8         48           11
     64      192    2         8         96           11
    128      384    2         8        192           11
    256      768    2         8        384           11
```

Three consequences for this spec:

1. **No information is lost in any of these cases.** All four cover `F` exactly (P1), and
   Yuan's pieces are disjoint — it was always a clean partition. "Cannot be reassembled" is
   a limit of the *description format*, not of the data. That is what makes it fixable by
   an IR change alone.

2. **`bigcodetype` is the concrete case for the location set.** Its kv block is
   byte-identical on every rank — partial replication *within a single parameter*. A schema
   with one `partition_dim` per parameter cannot say "these rows sharded, those rows
   replicated"; a per-piece `locations` set says it with no extra mechanism. GPTBigCode /
   StarCoder is an in-tree model this unblocks.

3. **Strides are load-bearing.** Yuan's dim-1 case is `strides=(32, 1)` — a column
   selection, not a contiguous span. The representation cannot be narrowed to "a list of
   contiguous ranges."

4. **Merging must stop at a replication boundary.** On bigcode's last rank the q slice ends
   exactly where the replicated kv block begins, so merging by adjacency alone produces one
   piece that is rank-private in its first half and replicated in its second — and a single
   `locations` cannot describe it. Splitting there (P5) costs one extra piece across the
   whole parameter, 7 to 8 at TP=4; every other layout was already homogeneous.

The harness that produced this table is now a test —
`tests/unit/checkpoint/test_affine_shard_map.py`, 14 cases covering coverage, round-trip
against unseen data, and rebuild-inverts-extract for all four layouts. It needs no process
group or accelerator, because the partition functions take an explicit rank.

---

## 6. On-disk format

### 6.1 What is stored, and what is not

The file stores **`M_s` only** — how the job that wrote the checkpoint had the parameter
laid out. `M_t` is not stored and must not be: the job doing the restore derives its own
map from its own layers, and a target map baked into the file would be a map for somebody
else's topology.

Within `M_s`, §1's boundary rule splits the content in two:

| | stable across jobs? | stored | why |
|---|---|---|---|
| geometry (`shape`, offsets, strides) | yes — a property of the parameter | **yes** | this is what makes the checkpoint portable |
| `locations` (which ranks held a piece) | no — a property of one grid | **yes, as provenance** | the converter must know which shard file to read |

Storing locations is not a violation of the boundary. The file records *where the bytes
were*, which is a fact; it does not record where they should go next, which is a decision.
A reader that wants a different placement ignores the field entirely.

### 6.2 The document

Written into `UNIVERSAL_CHECKPOINT_INFO` under a new `affine_map` key, keyed by the same
exact-match patterns `collect_autotp_universal_checkpoint_info` already emits — one
`rf"^{re.escape(full_name)}$"` per parameter:

```python
{
  "affine_map": {
    "version": 1,
    "params": {
      "^transformer.h.0.attn.c_attn.weight$": {
        "logical_shape": [48, 8],
        "ranks": {
          0: {"shard_shape": [16, 8],
              "pieces": [
                {"shape": [8, 8],  "source": [0,   [8, 1]], "dest": [0,  [8, 1]], "locations": [0]},
                {"shape": [8, 8],  "source": [256, [8, 1]], "dest": [64, [8, 1]], "locations": [0,1,2,3]}
              ]},
          ...
        }
      }
    }
  }
}
```

Each piece is `shape` plus `[offset, strides]` on each side — the argument list of
`torch.as_strided` twice over, so a reader applies it with no interpretation step. `scale`
is omitted when it is 1, which is almost every piece.

Per-parameter entries live under `params` rather than beside `version`, so that a parameter
whose name matched a metadata key could never be confused for one.

### 6.3 Rules

- **Additive.** A checkpoint carrying `affine_map` also carries today's keys. Old readers
  ignore the new key and take the existing branch chain; new readers prefer `affine_map`
  when present and fall back otherwise. Nothing written today becomes unreadable, and
  nothing written by this scheme becomes unreadable by an old converter.
- **Version.** `UNIVERSAL_CHECKPOINT_VERSION_VALUE` goes 0.4 → 0.5. The inner
  `"version": 1` covers the map encoding itself, so the two can move independently.
- **Pieces are canonical.** A writer must emit maximally merged pieces, folded in N
  dimensions (§8.2), **subject to homogeneity** (P5): a merge may not cross a change in
  `locations` or in `scale`. Both halves are normative. Skipping the merge gives a map whose
  size grows with model size instead of block structure; skipping the homogeneity constraint
  gives a map whose `locations` is wrong, which is worse because it still round-trips
  correctly on a single topology and only misleads a phase-2 planner.
- **Plain scalars only.** No tensors, no pickled classes, no `SubparamShape` objects. The
  map should be readable without importing DeepSpeed, which matters for external tooling
  and for debugging a checkpoint that will not load.
- **`locations` is per piece, not per parameter.** This is the field that expresses
  `bigcodetype`, whose kv block is held identically by every rank (§5).

### 6.4 Size

Measured on a fused-QKV parameter at TP=8: 96 pieces, 6836 bytes of compact JSON —
about **71 bytes per piece**. Extrapolated by tensor count:

| model | tensors | `affine_map` |
|---|---|---|
| Llama-3-8B | ~291 | ~1.9 MB |
| Llama-3-70B | ~723 | ~4.7 MB |

Megabytes, not kilobytes, and it scales with **tensor count × TP degree** rather than with
parameter size — a 70B and a 7B with the same layer count cost the same. Against
checkpoints measured in hundreds of gigabytes this is negligible, but it is large enough
that the per-rank redundancy (most ranks differ only in offset) is worth revisiting if the
map is ever loaded somewhere latency-sensitive. That is an encoding question and is
deliberately outside the semantics.

---

## 7. Phase 1 — the converter

### 7.1 The algorithm

With the map present, `merge_tp_slices` loses its branch chain entirely:

```
for each parameter:
    F = empty(logical_shape)
    for rank, pieces in M_s:
        shard = flat_buffer(load_shard(rank))
        for piece in pieces:
            F[piece.source_view] = shard[piece.dest_view]
    assert M_s.uncovered_offsets() == []
    if the parameter is a padded vocabulary:
        F = F[:V_orig]                       # outside the IR, see §8.1
    write F
```

Restore is the same loop with the copy reversed, against the target's own `M_t`:

```
for each rank r of this job:
    shard = empty(M_t.shard_shape[r])
    for piece in M_t.pieces[r]:
        shard[piece.dest_view] = F[piece.source_view]
```

Both directions are driven by the same piece list. That is the structural reason the
current convert/restore asymmetry — the two builders `_build_param_uc_conversion_meta` and `_build_param_uc_restore_meta` emitting two
different key sets — stops being expressible.

### 7.2 Two things an implementation must get right

**Normalise the shard buffer.** Piece offsets are storage offsets, because that is what
`as_strided` takes. A loaded shard is often a *view* into a larger buffer whose elements
begin partway into that storage — splitting a tensor produces exactly that — and applying
a piece to it then reads from the wrong place, silently and with no error. `flat_buffer`
above is not decorative.

**Take any one holder of a replicated piece.** Where `locations` has more than one rank,
every listed rank holds identical data by construction (P2), so the converter reads
whichever is cheapest and skips the rest. Phase 1 is the degenerate case of the location
set; phase 2 is where the choice becomes interesting.

### 7.3 What changes

Removed: the regex categories, the `cat_dim` decision, the sub-parameter narrowing
arithmetic, the `chunk` fallback with its divisibility assert
(the `chunk` fallback in `_resolve_autotp_partition`), and the `unsupported_reason` refusal path
(its `unsupported_reason` guard) — the layouts it refuses are expressible (§5).

Added: a coverage assertion that is a real correctness property. Today's checks are shape
heuristics — they confirm the numbers line up, not that every element of the parameter has
a home. `uncovered_offsets()` answers the second question, and it is the check that fails
loudly if a future layout outgrows the language.

Unchanged: parallelism. Pieces write to disjoint regions of `F` (or identical values where
they overlap), and parameters remain independent, so the existing `ProcessPoolExecutor`
fan-out over `merge_tp_slices` carries over untouched.

---

## 8. Open cases

**8.1 Vocabulary truncation.** The vocabulary branch of `merge_tp_slices` does
`param = param[:original_vocab_size, :]`, dropping padding rows. This is not a view, so it
violates P3 and makes `M_s⁻¹` non-total over the shards. Two options:

- **(a)** `F` carries the padding; truncation becomes a post-step outside the IR. Every
  piece stays a view and `M_s⁻¹` stays well defined for the whole language. Cost: the
  padded size must be recorded.
- **(b)** The IR admits pieces that map to nothing. Cost: invertibility, i.e. P3, for the
  whole language rather than for one case.

**Resolved: (a)**, agreed with delock on #8252. One key is cheaper than one weakened
property, and the case is narrow — only legacy TP needs an even split and therefore
padding at all; AutoTP's uneven sharding does not pad `F`.

**(a) also survives phase 2**, which was the real risk. When a source and target topology
pad to different heights (TP=4 → 3 padding rows, TP=2 → 1), the truncate/pad step is the
identity restricted to `[0, V_orig)`. Restricting an affine map to a range leaves it
affine, so `M_t ∘ truncate ∘ M_s⁻¹` composes to plain overlap arithmetic and neither
`Fsrc` nor `Ftgt` is ever materialised. Verified over all 64 source/target pairs for
TP ∈ {1,2,4,8} and vocab ∈ {13, 32, 100, 50257}, in both the truncating and padding
directions, with source padding poisoned so that reading it would fail the check.

The composed map is **partial over the target** — target padding rows have no source.
That is much weaker than `M_s` being non-total: padding is not data, and the target has to
initialise it regardless. Piece count is bounded by the two TP degrees rather than by the vocabulary: across
vocab 13 → 262144 and TP up to 16 it never exceeds `src_tp + tgt_tp`, though the exact
count varies with how the two padded heights align.

Consequence: legacy TP does not need excluding from phase 2, and padding need not enter
the UC file even in legacy mode — the composition needs only `V_orig`.

Relevant: `PADDED_VOCAB_SIZE` is declared at `constants.py` and referenced **nowhere
else in the repository** (verified against `92843ad70`) — it looks like exactly this slot,
never wired up. Worth confirming before adding a new key.

**8.2 Do we ever need explicit indices? — RESOLVED: no.** Every layout in tree is
expressible with affine pieces at bounded count (§5), and composition `M_t ∘ M_s⁻¹` between
two TP degrees stays affine for all of them, verified against the real partition functions
with `F` never materialised. No escape hatch to explicit index lists is needed.

Two rules are **normative for any composer**, because getting either wrong yields a piece
count that grows with model size — which looks like evidence the language is insufficient:

- **Compress in N dimensions.** A column split emits one run per row with identical strides
  and constant start deltas; that is an outer dimension, and the stack is one piece. In 1-D
  only, Yuan o_proj needs one piece per row (32 → 1024 as rows go 4 → 128); in N-D it is 4.
- **Group runs by source rank before folding.** A target shard interleaves blocks from
  several source ranks, so runs that stack are usually not adjacent in shard order. For
  Yuan o_proj 4→2 they alternate rank 0/rank 1, and a neighbour-only merge never fires:
  256 pieces instead of 4.

Grouping is only legal because a piece carries its own destination offset (§2.1), so pieces
may be reordered freely.

Measured piece counts are bounded by topology and block structure, not model size: across
an 8× hidden sweep, codegen 4→8 holds at 40, bigcode 2→4 at 7, Yuan o_proj at 4.

**8.3 Scale and floating point.** `(x / N) * N` is exact for power-of-two `N` in fp32, bf16
and fp16 alike, because dividing by `2^k` only shifts the exponent — so a bias round-trips
bit-exactly at every TP degree in normal use. It is lossy for non-power-of-two `N` (3, 6,
12), where a converted-and-restored bias may differ in the last bits from the original.

**8.4 ZeRO and offload placement.** delock's extension in #8230 — "a subset of a parameter
combined with a list of ranks holding this subset" — is what §2.1's `locations` implements.
ZeRO-1/3 partitions and offload replicas should fall out as pieces whose `locations`
describe the DP group rather than the TP group, but this spec does not yet work through
AutoEP's expert placement, where locations are per-expert rather than per-parameter.

---

## 9. Staging

1. **Done.** IR data structure (`deepspeed/checkpoint/affine.py`: `AffinePiece`,
   `ParamAffineMap`), behind no flag — pure addition, nothing reads it yet.
2. **Done.** The §5 harness as a test, asserting all four layouts lower and round-trip.
3. Converter reads `affine_map` when present, existing chain otherwise. Parity test:
   both paths produce byte-identical `F` for every layout in tree.
4. Emit `affine_map` from `collect_autotp_universal_checkpoint_info`, and drop
   `unsupported_reason` for the four layouts §5 covers.
5. Phase 2 — `M_t ∘ M_s⁻¹` as a transfer plan — builds on this and is out of scope here.

Steps 1–3 change no behaviour: they add a representation and prove it agrees with the
current one. Only step 4 changes what a checkpoint contains, and only by adding a key.
