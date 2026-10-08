# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Batch-axis tiled rematerialization.

``TiledRematerialize`` runs a ``no_grad`` forward and recomputes one row
at a time in backward so rematerialize workspace stays 1x row. Default
forward is full-batch (same kernel shape as an uncheckpointed pass).
``tile_forward=True`` shards that first pass too (same split as remat).
``fwd_tile_size`` / ``fwd_shards`` pick a different first-forward tile
(rows per chunk, or chunk count) than remat. When native
``cpu_checkpointing`` is configured, saved full-batch inputs are offloaded
through ``_ActivationOffloadEngine`` (same path as DeepSpeed checkpoint)
so they do not need an outer checkpoint wrap. The Function-input
hidden is emptied after D2H and refilled on the way back;
``ctx`` holds only a token and a weak handle so residuals do not pin
through the ZeRO-3 epilogue. Step peak is
``max(fwd workspace, tiled bwd, keep-last saved input)``.

ZeRO-3 ``compute_params`` are gathered at forward and again at rematerialize
entry and held across the row loop. The hold itself works: measured allgather
counts do not grow with the row count. The rematerialize gather asks
``GatheredParameters`` for its coalesced path, since the default one launches a
collective per parameter and put tiling at ~2x plain checkpointing.
The reliable win versus GAS (``mbs=1``, ``gas=B``) is on the
gradient side: the row loop is a single ``engine.backward``, so
``ds_grad_is_ready`` flips True only on the last row and ZeRO-2/3 reduce
grads once instead of once per row.

Under ZeRO-3 the tiled module must be marked a leaf
(``deepspeed.utils.set_z3_leaf_modules``) *before* ``deepspeed.initialize``:
the marking is read while ZeRO-3 registers its hooks, and an unmarked module
re-runs its per-submodule hooks once per shard, registering a parameter for
reduction more than once. Whether that reaches the ``ds_id`` assert in
``stage3`` depends on bucketing, so entry points warn rather than refuse --
the assert names neither tiling nor the fix.

Use for batch-independent, sequence-mixing modules (attention, GDN). For
token-independent MLP compute use ``TiledMLP`` in
``deepspeed.runtime.sequence_parallel.ulysses_sp``.

``sequential_chain_rematerialize`` is the one-sequence case packed tiling
refuses to split: GDN tiles run in order, carrying the recurrent state.
``packed_seq_tiled_rematerialize_compute(chain_tile_fn=...)`` nests that
chain inside an oversized packed sample.
"""

import contextvars
import sys
import weakref
from contextlib import contextmanager

import torch
import torch.nn.functional as F

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partition_parameters import GatheredParameters, is_zero_param
from deepspeed.utils import logger, z3_leaf_parameter

# One-shot so the ZeRO-3 leaf warning does not repeat per layer per step.
_warned_not_z3_leaf = False

# Function.forward always runs with grad disabled; capture mode in the wrappers.
_tiled_fwd_grad_enabled = True

# Nested tilers (batch or packed remat around TiledMLP) share this. ZeRO-2
# reduces a parameter the first time a hook sees ds_grad_is_ready, then
# rejects every later grad on that parameter. An inner last shard is not
# ready while an outer tile is still outstanding.
_grad_reduce_allowed = contextvars.ContextVar("ds_tiled_grad_reduce_allowed", default=True)


@contextmanager
def limit_grad_reduce(is_last):
    """Allow a grad reduce only on this shard and every outer shard still open."""
    allowed = bool(is_last) and _grad_reduce_allowed.get()
    token = _grad_reduce_allowed.set(allowed)
    try:
        yield allowed
    finally:
        _grad_reduce_allowed.reset(token)


def grad_reduce_allowed():
    return _grad_reduce_allowed.get()


def _shards_for_tile_size(batch, tile_size):
    # Dual of shard count: tile_size is max rows per chunk.
    rows = max(1, min(int(tile_size), int(batch)))
    return (int(batch) + rows - 1) // rows


def _resolve_forward_shards(tile_forward, fwd_tile_size, fwd_shards, remat_shards, batch, device):
    # tiled_fwd may use a coarser split than remat (e.g. mbs=16 remat rows=2,
    # first-forward tiles of 8). Ranks agree on the fwd shard count.
    use_fwd = bool(tile_forward)
    requested = remat_shards
    if fwd_tile_size is not None and int(fwd_tile_size) > 0:
        use_fwd = True
        requested = _shards_for_tile_size(batch, fwd_tile_size)
    elif fwd_shards is not None and int(fwd_shards) > 0:
        use_fwd = True
        requested = int(fwd_shards)
    if not use_fwd:
        return False, remat_shards
    return True, _agree_rematerialize_shards(requested, batch, device)


def _usable_shard_count(shards, batch):
    # Shrink requested shards until every ceil-split slice is non-empty
    # (B=5, shards=4 would otherwise yield a trailing empty slice).
    resolved = max(1, min(int(shards), int(batch)))
    while resolved > 1:
        step = (batch + resolved - 1) // resolved
        if step * (resolved - 1) < batch:
            return resolved
        resolved -= 1
    return 1


# Agreement depends only on (shards, batch, world_size), but the call sites are
# per layer per microstep, so an uncached all_reduce costs L collectives and L
# device syncs on every step. Every rank reaches the same keys in the same order
# because ranks must share the microbatch size, so caching stays collective-safe.
_agreed_shard_counts = {}


def _agree_rematerialize_shards(shards, batch, device):
    resolved = _usable_shard_count(shards, batch)
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return resolved

    key = (int(shards), int(batch), dist.get_world_size())
    if key in _agreed_shard_counts:
        return _agreed_shard_counts[key]

    tensor = torch.tensor(resolved, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    resolved = int(tensor.item())
    if resolved > batch:
        raise RuntimeError(f"tiled rematerialize shard count {resolved} > local batch {batch}; "
                           "all ranks must use the same microbatch size")
    _agreed_shard_counts[key] = resolved
    return resolved


def _reject_rematerialize_kwarg(name, value):
    if value is None:
        return
    lname = name.lower()
    if lname in ("past_key_value", "past_key_values", "cache_params", "cache"):
        raise TypeError(f"tiled_rematerialize_compute rejects {name}: cache objects cannot be batch-chunked")
    if "cu_seqlens" in lname:
        raise TypeError(
            f"tiled_rematerialize_compute rejects {name}: chunking cumulative seqlens is not a batch split")
    type_name = type(value).__name__
    if value is not None and "Cache" in type_name:
        raise TypeError(f"tiled_rematerialize_compute rejects {name}: {type_name} cannot be batch-chunked")


def _classify_tensor_dim0(name, value, batch):
    if value.dim() == 0:
        return "pass"
    if value.shape[0] == batch:
        return "shard"
    if value.shape[0] == 1:
        return "pass"
    raise TypeError(f"tiled_rematerialize_compute: {name} dim0={value.shape[0]} is neither batch={batch} nor 1")


def _classify_tuple_kwarg(name, value, batch):
    # All dim0==B → shard (Qwen3.5 RoPE). All None / scalar / dim0==1 → pass.
    kinds = []
    for i, item in enumerate(value):
        item_name = f"{name}[{i}]"
        if item is None or isinstance(item, (int, float, bool, str)):
            kinds.append("pass")
            continue
        if not torch.is_tensor(item):
            raise TypeError(f"tiled_rematerialize_compute cannot handle {item_name} of type {type(item).__name__}")
        kinds.append(_classify_tensor_dim0(item_name, item, batch))
    if not kinds:
        return "pass"
    unique = set(kinds)
    if len(unique) > 1:
        raise TypeError(f"tiled_rematerialize_compute: {name} mixes shard and pass tensors")
    return kinds[0]


def classify_rematerialize_kwarg(name, value, batch):
    """Classify a kwarg as shard (dim0==B), pass, or raise."""
    _reject_rematerialize_kwarg(name, value)
    if value is None or isinstance(value, (int, float, bool, str)):
        return "pass"
    if isinstance(value, tuple):
        return _classify_tuple_kwarg(name, value, batch)
    if not torch.is_tensor(value):
        raise TypeError(f"tiled_rematerialize_compute cannot handle {name} of type {type(value).__name__}")
    return _classify_tensor_dim0(name, value, batch)


def _chunk_along_batch(value, num_shards):
    if torch.is_tensor(value):
        return list(torch.chunk(value, chunks=num_shards, dim=0))
    member_chunks = [list(torch.chunk(item, chunks=num_shards, dim=0)) for item in value]
    return [tuple(parts[i] for parts in member_chunks) for i in range(num_shards)]


@contextmanager
def _hold_gathered_params(params):
    """Keep ZeRO-3 weights gathered for every row of one layer.

    Wrapping a module ``forward`` already gets one gather from the ZeRO-3
    pre-hook; that module stays in ``ds_active_sub_modules`` until its
    post-hook, so child Linears cannot release between rows. Partitioning
    here would crash ``free_param``.

    Backward rematerialize runs outside the layer's own module hooks, so we
    allgather once and block per-row Linear releases with ``ds_persist`` /
    ``is_external_param`` (the first-step incomplete trace ignores persist).
    The original ``fn`` still runs child submodule forward/backward, whose
    ZeRO-3 hooks touch ``ds_active_sub_modules``; a hook that fires early or
    never fires (e.g. a submodule whose inputs do not require grad) can leave
    a stale active entry. ``GatheredParameters.__exit__`` then force-partitions
    at the layer boundary and ``free_param`` refuses to release a param still
    marked active, so clear that stale accounting first -- matching
    ``PartitionedParameterCoordinator.release_and_reset_all``.
    """
    zero_params = [p for p in (params or []) if is_zero_param(p)]
    if not zero_params:
        yield
        return
    saved_persist = [p.ds_persist for p in zero_params]
    saved_external = [p.is_external_param for p in zero_params]
    for p in zero_params:
        p.ds_persist = True
        p.is_external_param = True
    parent_held = any(p.ds_active_sub_modules for p in zero_params)
    try:
        if parent_held:
            yield
        else:
            with GatheredParameters(zero_params, modifier_rank=None, coalesced=True):
                try:
                    yield
                finally:
                    # Clear on the failure path too. Otherwise an exception from the body
                    # (an OOM, at the microbatch sizes where tiling is used) leaves the stale
                    # entry in place, ``__exit__`` force-partitions into ``free_param``, and
                    # its active-submodule guard raises a parameter-lifecycle error that
                    # replaces the OOM, sending users after a bug they do not have.
                    for p in zero_params:
                        p.ds_active_sub_modules.clear()
    finally:
        for p, was_persist, was_external in zip(zero_params, saved_persist, saved_external):
            p.ds_persist = was_persist
            p.is_external_param = was_external


def _mark_compute_params_ready(compute_params, is_last):
    ready = bool(is_last) and grad_reduce_allowed()
    for param in compute_params:
        param.ds_grad_is_ready = ready


def _require_grad_params(compute_params):
    if compute_params is None:
        return []
    return [p for p in compute_params if p.requires_grad]


def _warn_if_not_z3_leaf(compute_params):
    """Warn once when a tiled module was not registered as a ZeRO-3 leaf.

    Tiling calls the wrapped module once per shard, so without leafing, ZeRO-3
    runs its per-submodule hooks once per shard and can register the same
    parameter into a gradient bucket more than once, failing an opaque ``ds_id``
    uniqueness assert in ``stage3`` during gradient reduction.

    Warn rather than raise: whether the duplicate registration actually trips
    the assert depends on bucketing, so small unleafed models do run clean
    (``TestPackedSeqTiledRematerialize`` is one) and refusing them would break
    working code. The warning lands before the assert would, which is the point
    -- the assert names neither tiling nor the fix.

    The fix cannot be applied from here: ``_register_deepspeed_module`` stamps
    ``ds_z3_leaf_module`` while registering hooks and skips a leaf module's
    children entirely, so the marking has to exist before
    ``deepspeed.initialize`` and setting it later has no effect.
    """
    zero_params = [p for p in (compute_params or []) if is_zero_param(p)]
    if not zero_params or any(z3_leaf_parameter(p) for p in zero_params):
        return
    global _warned_not_z3_leaf
    if _warned_not_z3_leaf:
        return
    _warned_not_z3_leaf = True
    logger.warning("tiled rematerialization is running on a module whose ZeRO-3 parameters carry no leaf "
                   "marking. Call deepspeed.utils.set_z3_leaf_modules(model, [<tiled module class>]) before "
                   "deepspeed.initialize(); the marking is read while ZeRO-3 registers its hooks, so applying "
                   "it afterwards has no effect. Without it ZeRO-3 re-runs its per-submodule hooks once per "
                   "shard, which can register a parameter for reduction twice and fail a ds_id uniqueness "
                   "assert in stage3.")


def _activation_offload_engine(enabled):
    # Lazy import: checkpointing.py does not import this module.
    from deepspeed.runtime.activation_checkpointing.checkpointing import _get_cpu_offload_engine
    if not enabled:
        return None
    return _get_cpu_offload_engine()


def _checkpoint_recomputes_remat_input():
    """True inside ``non_reentrant_checkpoint``'s forward and its backward recompute.

    That region drops the mixer input and rebuilds it, so a direct offload would
    pin a second full hidden per layer and pin it again on recompute. Other
    saved-tensor hooks do not do that. The engine-wide offload hook returns an
    unmarked tensor unchanged, and tiling must still offload it.
    """
    from deepspeed.runtime.activation_checkpointing.checkpointing import _tiled_remat_offload_suspended
    return _tiled_remat_offload_suspended.get()


def _offload_remat_input(ctx, x):
    """Offload the Function-input hidden, or ``save_for_backward`` it.

    Offload empties the Function-input storage after D2H so forward peak is
    workspace + keep-last. The emptied object is the restore handle, but it is
    held *weakly*: under a ZeRO-3 leaf module PyTorch's full-backward hook
    hands ``forward`` a view whose ``_base`` is the caller's hidden, so a
    strong handle would pin L un-emptied bases for the whole step and reclaim
    nothing. Token lives on ``ctx``; the engine owns the buffer. Do not empty
    shared kwargs (RoPE / masks).

    Inside ``non_reentrant_checkpoint`` this tensor must go through
    ``save_for_backward`` so the checkpoint pack hook can drop it.
    """
    engine = _activation_offload_engine(getattr(ctx, 'ds_tiled_grad_enabled', False))
    if engine is not None and _checkpoint_recomputes_remat_input():
        engine = None
    ctx.ds_x_requires_grad = x.requires_grad
    ctx.ds_offload_engine = engine
    ctx.ds_offload_token = engine.offload_input(x.detach()) if engine is not None else None
    if ctx.ds_offload_token is None:
        ctx.save_for_backward(x)
        ctx.ds_x_handle = None
        return
    ctx.ds_x_handle = weakref.ref(x)
    x.data = torch.empty([], device=x.device).data


def _load_remat_input(ctx):
    """Restore the Function-input hidden as a local. Do not leave it on ``ctx``."""
    token = getattr(ctx, 'ds_offload_token', None)
    if token is None:
        saved = ctx.saved_tensors
        if not saved:
            raise RuntimeError("tiled rematerialize input was already consumed; retain_graph is not supported")
        return saved[0]
    handle_ref = ctx.ds_x_handle
    ctx.ds_offload_token = None
    ctx.ds_x_handle = None
    restored = ctx.ds_offload_engine.restore_input(token)
    # The handle is weak, so refill the caller's object when it is still live
    # and fall back to the engine's tensor when it is not. Backward only reads
    # this for recompute and to shape ``x_grad``, never returns it, so a fresh
    # object is equivalent.
    handle = handle_ref()
    if handle is None:
        restored.requires_grad_(ctx.ds_x_requires_grad)
        return restored
    handle.data = restored.data
    handle.requires_grad_(ctx.ds_x_requires_grad)
    return handle


def tiled_rematerialize(fn,
                        module,
                        x,
                        shards,
                        compute_params=None,
                        tile_forward=False,
                        fwd_tile_size=None,
                        fwd_shards=None):
    """Tile ``fn(module, x)`` along batch. No-op when B==1 or shards<=1.

    ``fwd_tile_size`` is rows per first-forward chunk; ``fwd_shards`` is the
    chunk count. Either implies tiled first forward. Default tiled_fwd uses
    the remat split.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    resolved = _agree_rematerialize_shards(shards, x.shape[0], x.device)
    if x.shape[0] <= 1 or resolved <= 1:
        return fn(module, x)
    _warn_if_not_z3_leaf(compute_params)
    use_fwd, fwd_resolved = _resolve_forward_shards(tile_forward, fwd_tile_size, fwd_shards, resolved, x.shape[0],
                                                    x.device)
    global _tiled_fwd_grad_enabled
    _tiled_fwd_grad_enabled = torch.is_grad_enabled()
    return TiledRematerialize.apply(fn, module, x, resolved, compute_params, bool(use_fwd), int(fwd_resolved))


def tiled_rematerialize_compute(fn,
                                hidden_states,
                                shards,
                                compute_params=None,
                                kwargs_to_shard=None,
                                kwargs_to_pass=None,
                                tile_forward=False,
                                fwd_tile_size=None,
                                fwd_shards=None):
    """Tile ``fn(hidden_states, **kwargs)`` along batch.

    ``kwargs_to_shard`` values must have dim0 == batch (tensors or tuples of
    tensors such as batch-major ``(cos, sin)``). ``kwargs_to_pass`` is forwarded
    on every shard (None, scalars, dim0==1 tensors, or dim0==1 tuples). Cache
    objects and ``cu_seqlens`` are rejected. Native cpu_checkpointing offloads
    only ``hidden_states``; kwargs stay resident because later layers reuse
    them. Kwarg tensors must not ``requires_grad``.
    """
    kwargs_to_shard = {} if kwargs_to_shard is None else dict(kwargs_to_shard)
    kwargs_to_pass = {} if kwargs_to_pass is None else dict(kwargs_to_pass)
    batch = hidden_states.shape[0]
    for name, value in kwargs_to_shard.items():
        kind = classify_rematerialize_kwarg(name, value, batch)
        if kind != "shard":
            raise TypeError(f"kwargs_to_shard[{name}] must be shardable, got {kind}")
    for name, value in kwargs_to_pass.items():
        classify_rematerialize_kwarg(name, value, batch)
    for name, value in list(kwargs_to_shard.items()) + list(kwargs_to_pass.items()):
        tensors = value if isinstance(value, tuple) else (value, )
        for item in tensors:
            if torch.is_tensor(item) and item.requires_grad:
                raise TypeError(f"tiled_rematerialize_compute rejects requires_grad on {name}")
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    resolved = _agree_rematerialize_shards(shards, batch, hidden_states.device)
    if batch <= 1 or resolved <= 1:
        return fn(hidden_states, **kwargs_to_shard, **kwargs_to_pass)
    _warn_if_not_z3_leaf(compute_params)
    use_fwd, fwd_resolved = _resolve_forward_shards(tile_forward, fwd_tile_size, fwd_shards, resolved, batch,
                                                    hidden_states.device)
    global _tiled_fwd_grad_enabled
    _tiled_fwd_grad_enabled = torch.is_grad_enabled()
    keys_shard = list(kwargs_to_shard.keys())
    keys_pass = list(kwargs_to_pass.keys())
    return TiledRematerializeCompute.apply(
        fn,
        hidden_states,
        resolved,
        _require_grad_params(compute_params),
        keys_shard,
        keys_pass,
        bool(use_fwd),
        int(fwd_resolved),
        *kwargs_to_shard.values(),
        *kwargs_to_pass.values(),
    )


class TiledRematerialize(torch.autograd.Function):
    """Tile ``fn(module, hidden_states)`` along the batch dimension.

    Use for batch-independent, sequence-mixing modules (attention, GDN). For
    token-independent MLP compute use ``TiledMLP``. Rematerialize workspace
    of ``fn`` shrinks to one row. The saved input stays full-batch.
    Default forward is full-batch; ``tile_forward`` shards it (optionally
    with a different ``fwd_tile_size`` / ``fwd_shards`` than remat).

    ``fn`` must be deterministic (no dropout, or dropout disabled). Under an
    outer activation checkpoint the first forward runs again before tiled
    rematerialize (three layer-body passes).

    ``compute_params`` lists the ZeRO-3 weights to gather and hold across the
    row loop. A Python loop of ``engine.backward`` per row is GAS and does not
    use this path.
    """

    @staticmethod
    def forward(
        ctx,
        fn,
        self,
        x,
        shards,
        compute_params,
        tile_forward,
        fwd_shards,
    ) -> torch.Tensor:
        ctx.ds_tiled_grad_enabled = _tiled_fwd_grad_enabled
        ctx.fn = fn
        ctx.self = self
        ctx.compute_params = _require_grad_params(compute_params)
        x_in = x
        if not x.is_contiguous():
            x = x.contiguous()

        ctx.num_shards = len(list(torch.chunk(x, chunks=max(1, min(int(shards), int(x_in.shape[0]))), dim=0)))
        ctx.fwd_num_shards = ctx.num_shards
        if tile_forward:
            ctx.fwd_num_shards = len(
                list(torch.chunk(x, chunks=max(1, min(int(fwd_shards), int(x_in.shape[0]))), dim=0)))

        with _hold_gathered_params(ctx.compute_params):
            with torch.no_grad():
                if tile_forward:
                    output_shards = []
                    for x_shard in torch.chunk(x, chunks=ctx.fwd_num_shards, dim=0):
                        shard_out = fn(self, x_shard)
                        if not torch.is_tensor(shard_out):
                            raise TypeError(
                                f"TiledRematerialize expects a single tensor from fn, got {type(shard_out)}")
                        output_shards.append(shard_out)
                    output = torch.cat(output_shards, dim=0)
                else:
                    output = fn(self, x)
        if not torch.is_tensor(output):
            raise TypeError(f"TiledRematerialize expects a single tensor from fn, got {type(output)}")
        # Empty the apply() input, not a contiguous working copy.
        _offload_remat_input(ctx, x_in)
        return output

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        fn = ctx.fn
        x = _load_remat_input(ctx)
        self = ctx.self
        compute_params = ctx.compute_params
        num_shards = ctx.num_shards

        x_requires_grad = x.requires_grad
        x = x.detach()
        x.requires_grad_(x_requires_grad)
        if not x.is_contiguous():
            x = x.contiguous()

        incoming_grad = grads[0]
        if not incoming_grad.is_contiguous():
            incoming_grad = incoming_grad.contiguous()

        x_shards = list(torch.chunk(x, chunks=num_shards, dim=0))
        grad_shards = list(torch.chunk(incoming_grad, chunks=num_shards, dim=0))
        x_grad = torch.zeros_like(x) if x_requires_grad else None

        with _hold_gathered_params(compute_params):
            offset = 0
            for i, x_shard in enumerate(x_shards):
                with limit_grad_reduce(i + 1 == len(x_shards)):
                    _mark_compute_params_ready(compute_params, i + 1 == len(x_shards))
                    x_shard.requires_grad_(x_requires_grad)
                    shard_batch = x_shard.shape[0]
                    if x_grad is not None:
                        x_shard.grad = x_grad.narrow(0, offset, shard_batch).view_as(x_shard)
                    with torch.enable_grad():
                        output = fn(self, x_shard)
                    torch.autograd.backward(output, grad_shards[i])
                    offset += shard_batch

        return (None, None, x_grad, None, None, None, None)


class TiledRematerializeCompute(torch.autograd.Function):
    """Generic batch-dim tiler used by ``tiled_rematerialize_compute``."""

    @staticmethod
    def forward(
        ctx,
        fn,
        x,
        shards,
        compute_params,
        keys_to_shard,
        keys_to_pass,
        tile_forward,
        fwd_shards,
        *args,
    ) -> torch.Tensor:
        ctx.ds_tiled_grad_enabled = _tiled_fwd_grad_enabled
        ctx.fn = fn
        ctx.compute_params = compute_params if compute_params is not None else []
        ctx.keys_to_shard = keys_to_shard
        ctx.keys_to_pass = keys_to_pass
        ctx.total_args = len(args)
        args = list(args)
        kwargs_to_shard = {k: args.pop(0) for k in keys_to_shard}
        kwargs_to_pass = {k: args.pop(0) for k in keys_to_pass}
        ctx.kwargs_to_pass = kwargs_to_pass
        # RoPE tuples are not tensors; keep them on ctx. Do not empty them:
        # the same cos/sin objects are reused by later layers.
        tensor_kwargs_to_shard = {}
        tuple_kwargs_to_shard = {}
        for key, value in kwargs_to_shard.items():
            if torch.is_tensor(value):
                tensor_kwargs_to_shard[key] = value
            else:
                tuple_kwargs_to_shard[key] = value
        ctx.tuple_kwargs_to_shard = tuple_kwargs_to_shard
        ctx.tensor_kwargs_to_shard = tensor_kwargs_to_shard
        x_in = x
        if not x.is_contiguous():
            x = x.contiguous()

        ctx.num_shards = len(list(torch.chunk(x, chunks=max(1, min(int(shards), int(x_in.shape[0]))), dim=0)))
        ctx.fwd_num_shards = ctx.num_shards
        if tile_forward:
            ctx.fwd_num_shards = len(
                list(torch.chunk(x, chunks=max(1, min(int(fwd_shards), int(x_in.shape[0]))), dim=0)))

        with _hold_gathered_params(ctx.compute_params):
            with torch.no_grad():
                if tile_forward:
                    x_shards = list(torch.chunk(x, chunks=ctx.fwd_num_shards, dim=0))
                    shard_lists = {k: _chunk_along_batch(v, ctx.fwd_num_shards) for k, v in kwargs_to_shard.items()}
                    output_shards = []
                    for i, x_shard in enumerate(x_shards):
                        kw = {k: shard_lists[k][i] for k in keys_to_shard}
                        kw.update(kwargs_to_pass)
                        shard_out = fn(x_shard, **kw)
                        if not torch.is_tensor(shard_out):
                            raise TypeError(
                                f"tiled_rematerialize_compute expects a single tensor from fn, got {type(shard_out)}")
                        output_shards.append(shard_out)
                    output = torch.cat(output_shards, dim=0)
                else:
                    output = fn(x, **kwargs_to_shard, **kwargs_to_pass)
        if not torch.is_tensor(output):
            raise TypeError(f"tiled_rematerialize_compute expects a single tensor from fn, got {type(output)}")
        # Empty only the apply() hidden. Kwargs (RoPE, masks) are shared
        # across layers; emptying them breaks the next layer's forward.
        _offload_remat_input(ctx, x_in)
        return output

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        fn = ctx.fn
        x = _load_remat_input(ctx)
        keys_to_shard = ctx.keys_to_shard
        kwargs_to_shard = dict(ctx.tensor_kwargs_to_shard)
        kwargs_to_shard.update(ctx.tuple_kwargs_to_shard)
        kwargs_to_pass = ctx.kwargs_to_pass
        compute_params = ctx.compute_params
        num_shards = ctx.num_shards

        x_requires_grad = x.requires_grad
        x = x.detach()
        x.requires_grad_(x_requires_grad)
        if not x.is_contiguous():
            x = x.contiguous()

        incoming_grad = grads[0]
        if not incoming_grad.is_contiguous():
            incoming_grad = incoming_grad.contiguous()

        x_shards = list(torch.chunk(x, chunks=num_shards, dim=0))
        grad_shards = list(torch.chunk(incoming_grad, chunks=num_shards, dim=0))
        shard_lists = {k: _chunk_along_batch(v, num_shards) for k, v in kwargs_to_shard.items()}
        x_grad = torch.zeros_like(x) if x_requires_grad else None

        with _hold_gathered_params(compute_params):
            offset = 0
            for i, x_shard in enumerate(x_shards):
                with limit_grad_reduce(i + 1 == len(x_shards)):
                    _mark_compute_params_ready(compute_params, i + 1 == len(x_shards))
                    x_shard.requires_grad_(x_requires_grad)
                    shard_batch = x_shard.shape[0]
                    if x_grad is not None:
                        x_shard.grad = x_grad.narrow(0, offset, shard_batch).view_as(x_shard)
                    kw = {k: shard_lists[k][i] for k in keys_to_shard}
                    kw.update(kwargs_to_pass)
                    with torch.enable_grad():
                        output = fn(x_shard, **kw)
                    torch.autograd.backward(output, grad_shards[i])
                    offset += shard_batch

        grad_outputs = [None, x_grad, None, None, None, None, None, None]
        arg_outputs = [None] * ctx.total_args
        return tuple(grad_outputs + arg_outputs)


# --- Packed-sequence tiling (mixer-only, token axis) ---
#
# A physical row can pack several independent, whole samples end to end, with
# boundaries given by ``cu_seqlens``. A tile here is a contiguous group of
# whole samples, not an equal ``torch.chunk`` slice, and ``cu_seqlens`` must be
# rebased per tile rather than sliced like a batch-major tensor. See
# tiled_remat/PLAN_packed_seq_tiled_remat.md.


def rebase_cu_seqlens(cu_seqlens, start_sample, end_sample):
    """Tile-local ``cu_seqlens`` for samples ``[start_sample, end_sample)``.

    Starts at 0 and ends at the tile's own token count -- the shape a varlen
    attention kernel expects for an independent call.
    """
    return cu_seqlens[start_sample:end_sample + 1] - cu_seqlens[start_sample]


def plan_packed_tiles(cu_seqlens, shards=None, token_budget=None):
    """Group whole samples of one packed row into contiguous tiles.

    Returns a list of ``(start_sample, end_sample, start_tok, end_tok)``
    tuples, end-exclusive on both axes, covering every sample exactly once in
    order. Pure tensor/Python logic: no device sync beyond reading
    ``cu_seqlens`` (already resident), no collectives.

    ``token_budget``, when given, is used directly as the max tokens per
    tile; ``shards`` is ignored for sizing in that case but still caps the
    tile count (see below). When only ``shards`` is given, an implied budget
    is derived as ``ceil(total_tokens / shards)`` first. A sample longer than
    the budget becomes its own oversized tile -- the budget is soft, never
    split a sample.

    ``shards`` upper-bounds the tile count as a stop condition: once closing
    another tile would reach the cap, the walk stops closing and the
    remainder accumulates in the final tile, even past budget.
    """
    num_samples = int(cu_seqlens.shape[0]) - 1
    if num_samples == 1:
        return [(0, 1, 0, int(cu_seqlens[1]))]
    if token_budget is None:
        if shards is None:
            raise ValueError("plan_packed_tiles requires shards or token_budget")
        total_tokens = int(cu_seqlens[-1])
        token_budget = (total_tokens + int(shards) - 1) // int(shards)
    token_budget = int(token_budget)
    max_tiles = int(shards) if shards is not None else num_samples

    tiles = []
    start_sample = 0
    start_tok = int(cu_seqlens[0])
    for sample in range(num_samples):
        tile_has_sample = sample > start_sample
        end_tok_with_sample = int(cu_seqlens[sample + 1])
        over_budget = tile_has_sample and (end_tok_with_sample - start_tok) > token_budget
        at_tile_cap = len(tiles) + 1 >= max_tiles
        if over_budget and not at_tile_cap:
            close_tok = int(cu_seqlens[sample])
            tiles.append((start_sample, sample, start_tok, close_tok))
            start_sample = sample
            start_tok = close_tok
    tiles.append((start_sample, num_samples, start_tok, int(cu_seqlens[num_samples])))
    return tiles


def _merge_smallest_adjacent_tile_pair(tiles):
    sizes = [end_tok - start_tok for _, _, start_tok, end_tok in tiles]
    pair_sizes = [sizes[i] + sizes[i + 1] for i in range(len(tiles) - 1)]
    merge_at = pair_sizes.index(min(pair_sizes))
    start_sample, _, start_tok, _ = tiles[merge_at]
    _, end_sample, _, end_tok = tiles[merge_at + 1]
    merged = (start_sample, end_sample, start_tok, end_tok)
    return tiles[:merge_at] + [merged] + tiles[merge_at + 2:]


def _agree_packed_tile_count(tiles, device):
    """Merge local ``tiles`` down to the tile count every rank can match.

    Uncached, unlike ``_agree_rematerialize_shards``: batch is always 1 on
    this axis (one physical row), so a cache key derived from (shards, batch,
    world_size) would collide across different packings on the same rank and
    return a stale count for the next call. ``agreed_count =
    min_i(local_count_i) <= N_local`` for every rank (a tile always needs at
    least one sample), so every rank can always merge down to
    ``agreed_count`` non-empty groups from its own samples -- this is always
    a shrink, never a split.
    """
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return tiles
    tensor = torch.tensor(len(tiles), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    agreed_count = int(tensor.item())
    while len(tiles) > agreed_count:
        tiles = _merge_smallest_adjacent_tile_pair(tiles)
    return tiles


_PACKED_CU_SEQLENS_NAMES = ("cu_seqlens", "cu_seq_lens_q", "cu_seq_lens_k")
_PACKED_META_KWARG_NAMES = _PACKED_CU_SEQLENS_NAMES + ("max_seqlen_q", "max_seqlen_k", "max_length_q", "max_length_k",
                                                       "seq_idx")


def _reject_packed_kwarg(name, value):
    if value is None:
        return
    lname = name.lower()
    if lname in ("past_key_value", "past_key_values", "cache_params", "cache"):
        raise TypeError(f"packed_seq_tiled_rematerialize_compute rejects {name}: cache objects cannot be tiled")
    type_name = type(value).__name__
    if "Cache" in type_name:
        raise TypeError(f"packed_seq_tiled_rematerialize_compute rejects {name}: {type_name} cannot be tiled")


def _classify_tensor_token(name, value):
    if value.dim() == 0 or value.shape[0] == 1:
        return "pass"
    return "token"


def _classify_tuple_token_kwarg(name, value):
    # Same all-or-nothing rule as the batch-axis tuple classifier: every item
    # must agree, so a tile split never has to split some items and not others.
    kinds = []
    for i, item in enumerate(value):
        item_name = f"{name}[{i}]"
        if item is None or isinstance(item, (int, float, bool, str)):
            kinds.append("pass")
            continue
        if not torch.is_tensor(item):
            raise TypeError(f"packed_seq_tiled_rematerialize_compute cannot handle "
                            f"{item_name} of type {type(item).__name__}")
        kinds.append(_classify_tensor_token(item_name, item))
    if not kinds:
        return "pass"
    unique = set(kinds)
    if len(unique) > 1:
        raise TypeError(f"packed_seq_tiled_rematerialize_compute: {name} mixes token and pass tensors")
    return kinds[0]


def classify_packed_kwarg(name, value, num_samples):
    """Classify a kwarg as pack_meta (cu_seqlens family), token (dim0==total_tokens), pass, or raise.

    Unlike the batch-axis classifier, an unrecognized tensor defaults to
    "token" rather than being rejected: a packed row's companion tensors
    (e.g. ``position_ids``) are conventionally token-major, so the caller
    -- which knows the tile's actual token count -- does the real shape
    validation when it slices.
    """
    _reject_packed_kwarg(name, value)
    lname = name.lower()
    if lname in _PACKED_META_KWARG_NAMES:
        if lname in _PACKED_CU_SEQLENS_NAMES and torch.is_tensor(value):
            expected_len = num_samples + 1
            if value.shape[0] != expected_len:
                raise TypeError(f"packed_seq_tiled_rematerialize_compute: {name} has length "
                                f"{value.shape[0]}, expected num_samples+1={expected_len}")
        return "pack_meta"
    if value is None or isinstance(value, (int, float, bool, str)):
        return "pass"
    if isinstance(value, tuple):
        return _classify_tuple_token_kwarg(name, value)
    if not torch.is_tensor(value):
        raise TypeError(f"packed_seq_tiled_rematerialize_compute cannot handle {name} of type "
                        f"{type(value).__name__}")
    return _classify_tensor_token(name, value)


def _resolve_forward_packed_tiles(tile_forward, fwd_tile_size, fwd_shards, cu_seqlens, tiles):
    # Mirrors _resolve_forward_shards: tiled_fwd may use a coarser split than
    # remat. Single-rank only for now -- cross-rank agreement is a separate
    # step layered on top of plan_packed_tiles, not done here.
    if fwd_tile_size is not None and int(fwd_tile_size) > 0:
        return plan_packed_tiles(cu_seqlens, token_budget=fwd_tile_size)
    if fwd_shards is not None and int(fwd_shards) > 0:
        return plan_packed_tiles(cu_seqlens, shards=fwd_shards)
    if tile_forward:
        return tiles
    num_samples = int(cu_seqlens.shape[0]) - 1
    return [(0, num_samples, 0, int(cu_seqlens[-1]))]


def _narrow_token_kwarg(value, start_tok, tile_len):
    if torch.is_tensor(value):
        return value.narrow(0, start_tok, tile_len)
    return tuple(item.narrow(0, start_tok, tile_len) for item in value)


def _packed_tile_kwargs(cu_seqlens, start_sample, end_sample, start_tok, end_tok, kwargs_to_token_shard,
                        kwargs_to_pass, max_seqlen_keys):
    local_cu_seqlens = rebase_cu_seqlens(cu_seqlens, start_sample, end_sample)
    kw = {"cu_seqlens": local_cu_seqlens}
    if max_seqlen_keys:
        # The row's global max can be larger than any one tile's; recompute
        # exactly instead of passing a stale, oversized hint through.
        local_max = int(local_cu_seqlens.diff().max())
        for key in max_seqlen_keys:
            kw[key] = local_max
    tile_len = end_tok - start_tok
    for key, value in kwargs_to_token_shard.items():
        kw[key] = _narrow_token_kwarg(value, start_tok, tile_len)
    for key, value in kwargs_to_pass.items():
        if key not in max_seqlen_keys:
            kw[key] = value
    return kw


def packed_seq_tiled_rematerialize_compute(fn,
                                           hidden_states,
                                           cu_seqlens,
                                           shards=None,
                                           token_budget=None,
                                           compute_params=None,
                                           kwargs_to_token_shard=None,
                                           kwargs_to_pass=None,
                                           tile_forward=False,
                                           fwd_tile_size=None,
                                           fwd_shards=None,
                                           chain_tile_fn=None,
                                           chain_token_budget=None,
                                           chain_align=64):
    """Tile ``fn(hidden_states, **kwargs)`` over contiguous whole-sample tiles of one packed row.

    ``hidden_states`` is ``[1, T, H]`` or ``[T, H]``; ``cu_seqlens`` locates
    the packed sample boundaries (length ``num_samples+1``). No-op when
    ``cu_seqlens`` describes a single sample and no chain applies.
    ``kwargs_to_token_shard`` values must be token-major (see
    ``classify_packed_kwarg``) and are narrowed per tile; ``kwargs_to_pass``
    is forwarded to every tile unchanged, except the ``max_seqlen_q/k`` /
    ``max_length_q/k`` family, which is recomputed per tile from the rebased
    ``cu_seqlens`` instead of passed through stale. Other pack_meta kwargs
    (e.g. ``seq_idx``) are not yet supported by this wrapper.

    ``chain_tile_fn`` (GDN mixers only, see ``sequential_chain_rematerialize``
    for its contract) nests a sequential chain inside any tile that holds a
    single sample longer than ``chain_token_budget``. The chain starts from
    zero states, as the packed forward does at every sample boundary, and
    never crosses one. It sees only the sample's ``[1, t, H]`` hidden states,
    so token-major kwargs are refused and pass kwargs are not forwarded to it.
    This call keeps the parameter hold and ``ds_grad_is_ready`` for itself;
    the chain neither gathers nor marks readiness on its own.
    """
    kwargs_to_token_shard = {} if kwargs_to_token_shard is None else dict(kwargs_to_token_shard)
    kwargs_to_pass = {} if kwargs_to_pass is None else dict(kwargs_to_pass)
    num_samples = int(cu_seqlens.shape[0]) - 1

    for name, value in kwargs_to_token_shard.items():
        kind = classify_packed_kwarg(name, value, num_samples)
        if kind != "token":
            raise TypeError(f"kwargs_to_token_shard[{name}] must be token-major, got {kind}")
    max_seqlen_keys = []
    for name, value in kwargs_to_pass.items():
        kind = classify_packed_kwarg(name, value, num_samples)
        if kind != "pack_meta":
            continue
        if name.lower() in ("max_seqlen_q", "max_seqlen_k", "max_length_q", "max_length_k"):
            max_seqlen_keys.append(name)
        else:
            raise TypeError(f"kwargs_to_pass[{name}] is pack_meta and not yet supported by "
                            "packed_seq_tiled_rematerialize_compute; only the max-seqlen/max-length "
                            "family is recomputed per tile")
    for name, value in list(kwargs_to_token_shard.items()) + list(kwargs_to_pass.items()):
        tensors = value if isinstance(value, tuple) else (value, )
        for item in tensors:
            if torch.is_tensor(item) and item.requires_grad:
                raise TypeError(f"packed_seq_tiled_rematerialize_compute rejects requires_grad on {name}")
    if chain_tile_fn is not None:
        if chain_token_budget is None:
            raise ValueError("packed_seq_tiled_rematerialize_compute: chain_tile_fn requires chain_token_budget")
        if kwargs_to_token_shard:
            raise TypeError("packed_seq_tiled_rematerialize_compute: chain_tile_fn cannot take token-major kwargs "
                            f"{sorted(kwargs_to_token_shard)}")
        if fwd_tile_size is not None or fwd_shards is not None:
            raise TypeError("packed_seq_tiled_rematerialize_compute: chain_tile_fn walks the remat tiles in "
                            "forward too; fwd_tile_size/fwd_shards cannot be combined with it")

    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()

    tiles = plan_packed_tiles(cu_seqlens, shards=shards, token_budget=token_budget)
    # Agree before the single-sample shortcut: a one-sample rank still has to
    # enter the MIN all_reduce its multi-sample peers are waiting in. Its
    # local count of 1 is also the right vote, since packed tiling cannot
    # split a sample, so every rank then merges down to one tile.
    tiles = _agree_packed_tile_count(tiles, hidden_states.device)
    chain_plans = {}
    if chain_tile_fn is not None:
        chain_plans = _plan_packed_chains(tiles, chain_token_budget, chain_align)
    if num_samples == 1 and not chain_plans:
        return fn(hidden_states, cu_seqlens=cu_seqlens, **kwargs_to_token_shard, **kwargs_to_pass)

    if len(tiles) > 1 or chain_plans:
        _warn_if_not_z3_leaf(compute_params)
    fwd_tiles = _resolve_forward_packed_tiles(tile_forward, fwd_tile_size, fwd_shards, cu_seqlens, tiles)
    if chain_plans:
        # A whole-row first forward would pay the full-sequence workspace the chain exists to drop.
        fwd_tiles = tiles

    global _tiled_fwd_grad_enabled
    _tiled_fwd_grad_enabled = torch.is_grad_enabled()
    keys_to_token_shard = list(kwargs_to_token_shard.keys())
    keys_to_pass = list(kwargs_to_pass.keys())
    return PackedSeqTiledRematerializeCompute.apply(
        fn,
        hidden_states,
        cu_seqlens,
        tiles,
        fwd_tiles,
        _require_grad_params(compute_params),
        keys_to_token_shard,
        keys_to_pass,
        tuple(max_seqlen_keys),
        chain_tile_fn,
        chain_plans,
        *kwargs_to_token_shard.values(),
        *kwargs_to_pass.values(),
    )


class PackedSeqTiledRematerializeCompute(torch.autograd.Function):
    """Token-axis analog of ``TiledRematerializeCompute``.

    Tiles are whole-sample groups from ``plan_packed_tiles``, not equal
    ``torch.chunk`` shards -- see ``packed_seq_tiled_rematerialize_compute``.
    """

    @staticmethod
    def forward(
        ctx,
        fn,
        hidden_states,
        cu_seqlens,
        tiles,
        fwd_tiles,
        compute_params,
        keys_to_token_shard,
        keys_to_pass,
        max_seqlen_keys,
        chain_tile_fn,
        chain_plans,
        *args,
    ) -> torch.Tensor:
        ctx.ds_tiled_grad_enabled = _tiled_fwd_grad_enabled
        ctx.fn = fn
        ctx.compute_params = compute_params if compute_params is not None else []
        ctx.keys_to_token_shard = keys_to_token_shard
        ctx.max_seqlen_keys = max_seqlen_keys
        ctx.tiles = tiles
        ctx.chain_tile_fn = chain_tile_fn
        ctx.chain_plans = chain_plans
        ctx.chain_states = {}
        ctx.total_args = len(args)
        args = list(args)
        kwargs_to_token_shard = {k: args.pop(0) for k in keys_to_token_shard}
        kwargs_to_pass = {k: args.pop(0) for k in keys_to_pass}
        ctx.kwargs_to_pass = kwargs_to_pass
        # Token-major companions (e.g. reset position_ids) are not tensors we
        # can safely empty like the Function-input hidden; keep them on ctx.
        tensor_kwargs_to_token_shard = {}
        tuple_kwargs_to_token_shard = {}
        for key, value in kwargs_to_token_shard.items():
            if torch.is_tensor(value):
                tensor_kwargs_to_token_shard[key] = value
            else:
                tuple_kwargs_to_token_shard[key] = value
        ctx.tuple_kwargs_to_token_shard = tuple_kwargs_to_token_shard
        ctx.tensor_kwargs_to_token_shard = tensor_kwargs_to_token_shard

        x_in = hidden_states
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        token_dim = 1 if hidden_states.dim() == 3 else 0
        ctx.token_dim = token_dim

        with _hold_gathered_params(ctx.compute_params):
            with torch.no_grad():
                output_shards = []
                for start_sample, end_sample, start_tok, end_tok in fwd_tiles:
                    kw = _packed_tile_kwargs(cu_seqlens, start_sample, end_sample, start_tok, end_tok,
                                             kwargs_to_token_shard, kwargs_to_pass, max_seqlen_keys)
                    x_tile = hidden_states.narrow(token_dim, start_tok, end_tok - start_tok)
                    if start_sample in chain_plans:
                        sample_x = x_tile.unsqueeze(0) if token_dim == 0 else x_tile
                        shard_out, saved_recurrent, saved_conv = _chain_forward(chain_tile_fn, sample_x,
                                                                                chain_plans[start_sample])
                        ctx.chain_states[start_sample] = (saved_recurrent, saved_conv)
                        output_shards.append(shard_out.squeeze(0) if token_dim == 0 else shard_out)
                        continue
                    shard_out = fn(x_tile, **kw)
                    if not torch.is_tensor(shard_out):
                        raise TypeError("packed_seq_tiled_rematerialize_compute expects a single tensor "
                                        f"from fn, got {type(shard_out)}")
                    output_shards.append(shard_out)
                output = torch.cat(output_shards, dim=token_dim)
        if not torch.is_tensor(output):
            raise TypeError(f"packed_seq_tiled_rematerialize_compute expects a single tensor from fn, "
                            f"got {type(output)}")
        ctx.ds_cu_seqlens = cu_seqlens
        # Empty only the apply() hidden. cu_seqlens/kwargs are shared across
        # layers; emptying them breaks the next layer's forward.
        _offload_remat_input(ctx, x_in)
        return output

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        fn = ctx.fn
        x = _load_remat_input(ctx)
        cu_seqlens = ctx.ds_cu_seqlens
        kwargs_to_token_shard = dict(ctx.tensor_kwargs_to_token_shard)
        kwargs_to_token_shard.update(ctx.tuple_kwargs_to_token_shard)
        kwargs_to_pass = ctx.kwargs_to_pass
        max_seqlen_keys = ctx.max_seqlen_keys
        compute_params = ctx.compute_params
        tiles = ctx.tiles
        token_dim = ctx.token_dim

        x_requires_grad = x.requires_grad
        x = x.detach()
        x.requires_grad_(x_requires_grad)
        if not x.is_contiguous():
            x = x.contiguous()

        incoming_grad = grads[0]
        if not incoming_grad.is_contiguous():
            incoming_grad = incoming_grad.contiguous()

        x_grad = torch.zeros_like(x) if x_requires_grad else None

        with _hold_gathered_params(compute_params):
            for i, (start_sample, end_sample, start_tok, end_tok) in enumerate(tiles):
                # TiledMLP inside this tile must not reduce until the last packed tile.
                with limit_grad_reduce(i + 1 == len(tiles)):
                    if start_sample in ctx.chain_plans:
                        _packed_chain_backward(ctx, x, x_grad, incoming_grad, start_sample, start_tok, end_tok,
                                               i + 1 == len(tiles))
                        continue
                    _mark_compute_params_ready(compute_params, i + 1 == len(tiles))
                    tile_len = end_tok - start_tok
                    x_tile = x.narrow(token_dim, start_tok, tile_len)
                    x_tile.requires_grad_(x_requires_grad)
                    if x_grad is not None:
                        x_tile.grad = x_grad.narrow(token_dim, start_tok, tile_len).view_as(x_tile)
                    kw = _packed_tile_kwargs(cu_seqlens, start_sample, end_sample, start_tok, end_tok,
                                             kwargs_to_token_shard, kwargs_to_pass, max_seqlen_keys)
                    grad_tile = incoming_grad.narrow(token_dim, start_tok, tile_len)
                    with torch.enable_grad():
                        output = fn(x_tile, **kw)
                    torch.autograd.backward(output, grad_tile)

        grad_outputs = [None, x_grad, None, None, None, None, None, None, None, None, None]
        arg_outputs = [None] * ctx.total_args
        return tuple(grad_outputs + arg_outputs)


def _plan_packed_chains(tiles, chain_token_budget, chain_align):
    """Chain plans keyed by sample index, for single-sample tiles over the chain budget.

    A tile holding several samples stays one call: a chain carries state
    across its cuts, and a packed boundary must restart from zero.
    """
    chain_plans = {}
    for start_sample, end_sample, start_tok, end_tok in tiles:
        if end_sample - start_sample != 1:
            continue
        chain_tiles = plan_chain_tiles(end_tok - start_tok, chain_token_budget, align=chain_align)
        if len(chain_tiles) > 1:
            chain_plans[start_sample] = chain_tiles
    return chain_plans


def _packed_chain_backward(ctx, x, x_grad, incoming_grad, sample, start_tok, end_tok, is_last_tile):
    # No per-sample chain agreement here: ranks hold different numbers of
    # oversized samples, so a collective per chain would hang, and the inner
    # loop issues none (the outer hold keeps weights gathered and the ready
    # flag flips once for the whole call).
    tile_len = end_tok - start_tok
    token_dim = ctx.token_dim
    sample_x = x.narrow(token_dim, start_tok, tile_len)
    sample_grad = incoming_grad.narrow(token_dim, start_tok, tile_len)
    sample_x_grad = None
    if x_grad is not None:
        sample_x_grad = x_grad.narrow(token_dim, start_tok, tile_len)
    if token_dim == 0:
        sample_x = sample_x.unsqueeze(0)
        sample_grad = sample_grad.unsqueeze(0)
        if sample_x_grad is not None:
            sample_x_grad = sample_x_grad.unsqueeze(0)
    saved_recurrent, saved_conv = ctx.chain_states.pop(sample)
    _chain_backward(ctx.chain_tile_fn,
                    sample_x,
                    sample_x_grad,
                    sample_grad,
                    ctx.chain_plans[sample],
                    saved_recurrent,
                    saved_conv,
                    ctx.compute_params,
                    ready_at_end=is_last_tile)


# Sequential chain for one sequence. Packed tiling never splits a sample, so a
# single long GDN sequence is otherwise one tile and pays the full-sequence
# scan workspace. The chain is exact only while each cut (except the tail)
# sits on a chunk boundary and the carried recurrent state stays fp32.


def plan_chain_tiles(seq_len, token_budget, align=64):
    """Split one sequence into contiguous ``(start, end)`` tiles.

    Every tile but the last has length a multiple of ``align`` (FLA's chunk
    size). The last tile holds the remainder. ``token_budget`` is rounded
    down to a multiple of ``align``, and a budget below one chunk is raised
    to ``align``. A sequence that already fits in the budget is one tile,
    even when its length is not a multiple of ``align``.
    """
    seq_len = int(seq_len)
    align = int(align)
    if seq_len < 1:
        raise ValueError(f"plan_chain_tiles seq_len must be positive, got {seq_len}")
    if align < 1:
        raise ValueError(f"plan_chain_tiles align must be positive, got {align}")
    if token_budget is None:
        raise ValueError("plan_chain_tiles requires token_budget")
    budget = int(token_budget)
    if budget < 1:
        raise ValueError(f"plan_chain_tiles token_budget must be positive, got {budget}")
    if seq_len <= budget:
        return [(0, seq_len)]
    if budget < align:
        budget = align
    else:
        budget = (budget // align) * align
    if seq_len <= budget:
        return [(0, seq_len)]
    tiles = []
    start = 0
    while start + budget < seq_len:
        tiles.append((start, start + budget))
        start += budget
    tiles.append((start, seq_len))
    return tiles


def _merge_smallest_chain_pair(tiles):
    sizes = [end - start for start, end in tiles]
    pair_sizes = [sizes[i] + sizes[i + 1] for i in range(len(tiles) - 1)]
    merge_at = pair_sizes.index(min(pair_sizes))
    merged = (tiles[merge_at][0], tiles[merge_at + 1][1])
    return tiles[:merge_at] + [merged] + tiles[merge_at + 2:]


def _agree_chain_tile_count(tiles, device):
    """Merge local chain tiles down to the count every rank can match.

    A longer tile is still a valid chain (the scan state is carried either
    way). Merging never moves a cut onto a token the shorter plan did not
    already use as a boundary or a tail.
    """
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return tiles
    tensor = torch.tensor(len(tiles), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    agreed_count = int(tensor.item())
    while len(tiles) > agreed_count:
        tiles = _merge_smallest_chain_pair(tiles)
    return tiles


def _torch_causal_conv_carry(x, weight, bias, activation, initial_states):
    """Depthwise causal conv that carries the last ``width - 1`` inputs.

    Matches ``causal_conv1d``'s reference for one sequence: a missing initial
    state is a zero pad, and the carried state is the pre-activation input,
    not the activated output. ``x`` is ``[B, D, T]``, ``weight`` is ``[D, W]``.
    """
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError(f"causal conv activation must be None, silu, or swish, got {activation}")
    width = weight.shape[1]
    seqlen = x.shape[-1]
    x_work = x.to(dtype=weight.dtype)
    if initial_states is None:
        out = F.conv1d(x_work, weight.unsqueeze(1), bias, padding=width - 1, groups=x_work.shape[1])
        tail_src = x_work
    else:
        extended = torch.cat([initial_states.to(dtype=weight.dtype), x_work], dim=-1)
        out = F.conv1d(extended, weight.unsqueeze(1), bias, padding=0, groups=x_work.shape[1])
        tail_src = extended
    out = out[..., :seqlen]
    tail_len = width - 1
    if tail_src.shape[-1] >= tail_len:
        final_states = tail_src[..., -tail_len:]
    else:
        final_states = F.pad(tail_src, (tail_len - tail_src.shape[-1], 0))
    if activation is not None:
        out = F.silu(out)
    return out.to(dtype=x.dtype), final_states.to(dtype=x.dtype)


def _as_channel_last_bdt(x):
    # causal_conv1d only returns final states when dim is the contiguous axis.
    if x.stride(1) == 1:
        return x
    return x.transpose(1, 2).contiguous().transpose(1, 2)


def _causal_conv_carry(x, weight, bias, activation, initial_states):
    """Conv carry. CUDA uses the same kernel Qwen's training forward uses.

    The torch fallback inside transformers drops ``initial_states`` and
    zero-pads, so a tile that called it would not continue the previous tile.
    """
    if not get_accelerator().on_accelerator(x):
        return _torch_causal_conv_carry(x, weight, bias, activation, initial_states)
    from causal_conv1d import causal_conv1d_fn
    x = _as_channel_last_bdt(x)
    if initial_states is not None:
        initial_states = _as_channel_last_bdt(initial_states)
    out, final_states = causal_conv1d_fn(x,
                                         weight,
                                         bias,
                                         initial_states=initial_states,
                                         return_final_states=True,
                                         activation=activation)
    return out, final_states


def _chunk_gated_delta_rule_for(module):
    # The kernel swap lives on the class's module (HF replaces this name),
    # not on the instance. Calling Qwen's forward cannot see the state.
    owner = sys.modules[type(module).__module__]
    scan = getattr(owner, "torch_chunk_gated_delta_rule", None)
    if scan is None:
        raise RuntimeError("qwen35_gdn_tile_forward needs torch_chunk_gated_delta_rule defined on "
                           f"{type(module).__module__}")
    return scan


def qwen35_gdn_tile_forward(module, hidden_states, recurrent_state=None, conv_state=None, attention_mask=None):
    """One tile of ``Qwen3_5GatedDeltaNet`` that returns the states training drops.

    Same padding mask, projections, conv, gated norm, and out-proj as the
    training forward (``cache_params is None``). The scan is called with
    ``output_final_state`` so the next tile can take this tile's fp32 state
    as ``initial_state``. ``conv_state`` is ``[B, conv_dim, kernel-1]``
    pre-activation inputs. No cache: that is outside this carry.

    ``attention_mask`` is the ``[B, T]`` padding mask for exactly these
    tokens. ``sequential_chain_rematerialize`` does not slice one per tile,
    so under a chain mask the full input once before chaining instead; the
    mask only scales the input, so that is the same product per token.
    """
    if attention_mask is not None:
        if tuple(attention_mask.shape) != tuple(hidden_states.shape[:2]):
            raise ValueError(f"qwen35_gdn_tile_forward: attention_mask shape {tuple(attention_mask.shape)} must be "
                             f"[B, T] = {tuple(hidden_states.shape[:2])} for this tile")
        # Qwen's apply_mask_to_padding_states: zero padded tokens before every projection.
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv = module.in_proj_qkv(hidden_states)
    if not mixed_qkv.is_contiguous():
        mixed_qkv = mixed_qkv.contiguous()
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = module.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, module.head_v_dim)
    b = module.in_proj_b(hidden_states)
    a = module.in_proj_a(hidden_states)

    mixed_qkv, conv_state = _causal_conv_carry(mixed_qkv, module.conv1d.weight.squeeze(1), module.conv1d.bias,
                                               module.activation, conv_state)

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, module.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, module.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, module.head_v_dim)

    beta = b.sigmoid()
    # fp32 matches Qwen: fp16 A_log.exp() underflows to zero.
    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
    repeat = module.num_v_heads // module.num_k_heads
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    core_attn_out, last_recurrent_state = _chunk_gated_delta_rule_for(module)(query,
                                                                              key,
                                                                              value,
                                                                              g=g,
                                                                              beta=beta,
                                                                              initial_state=recurrent_state,
                                                                              output_final_state=True,
                                                                              use_qk_l2norm_in_kernel=True)
    if last_recurrent_state is None:
        raise RuntimeError("GDN tile scan did not return a final state")

    core_attn_out = core_attn_out.reshape(-1, module.head_v_dim)
    z = z.reshape(-1, module.head_v_dim)
    core_attn_out = module.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return module.out_proj(core_attn_out), last_recurrent_state, conv_state


def sequential_chain_rematerialize(tile_fn, hidden_states, token_budget, compute_params=None, align=64):
    """Rematerialize one sequence as a chain of tiles.

    ``tile_fn(x_tile, recurrent_state, conv_state)`` returns
    ``(out, recurrent_state, conv_state)``. The first tile is called with
    both states ``None``. Forward runs under ``no_grad`` and keeps only the
    boundary states. Backward recomputes one tile and feeds that tile's
    incoming-state gradients to the previous tile as its outgoing-state
    cotangents. Keeping every tile in one autograd graph would retain the
    full-sequence scan activations this exists to drop.

    ``hidden_states`` is ``[B, T, H]`` or ``[T, H]``. Every batch row is one
    sequence of length ``T`` (this does not pack samples). ``align`` is the
    scan chunk size; GDN uses 64. For a packed row, pass ``tile_fn`` as
    ``chain_tile_fn`` to ``packed_seq_tiled_rematerialize_compute`` instead:
    calling this per sample would gather weights the packed hold already
    holds and flip ``ds_grad_is_ready`` once per sample.
    """
    squeezed = hidden_states.dim() == 2
    if squeezed:
        hidden_states = hidden_states.unsqueeze(0)
    if hidden_states.dim() != 3:
        raise ValueError("sequential_chain_rematerialize expects [B, T, H] or [T, H], "
                         f"got {tuple(hidden_states.shape)}")
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()

    tiles = plan_chain_tiles(hidden_states.shape[1], token_budget, align=align)
    tiles = _agree_chain_tile_count(tiles, hidden_states.device)
    if len(tiles) > 1:
        _warn_if_not_z3_leaf(compute_params)

    global _tiled_fwd_grad_enabled
    _tiled_fwd_grad_enabled = torch.is_grad_enabled()
    output = SequentialChainRematerialize.apply(tile_fn, hidden_states, tiles, _require_grad_params(compute_params))
    if squeezed:
        output = output.squeeze(0)
    return output


def _detach_state_grad(state):
    if state is None:
        return None
    if state.grad is None:
        raise RuntimeError("sequential chain tile did not produce a gradient for a carried state")
    return state.grad.detach()


def _chain_forward(tile_fn, hidden_states, tiles):
    """Run the chain once under the caller's ``no_grad`` and parameter hold.

    Returns the full-length output and the boundary states ``_chain_backward``
    restarts each tile from. The first tile starts from zero states.
    """
    recurrent = None
    conv_state = None
    saved_recurrent = []
    saved_conv = []
    output = None
    for start, end in tiles:
        tile_len = end - start
        out, recurrent, conv_state = tile_fn(hidden_states.narrow(1, start, tile_len), recurrent, conv_state)
        if not torch.is_tensor(out):
            raise TypeError(f"sequential chain tile_fn must return a tensor, got {type(out)}")
        if out.shape[1] != tile_len:
            raise RuntimeError(f"sequential chain tile output length {out.shape[1]} != {tile_len}")
        if recurrent is None or conv_state is None:
            raise RuntimeError("sequential chain tile_fn must return both carried states")
        # Keep a private copy. The next tile is handed the other one: some
        # kernels write into the initial-state buffer they are given.
        recurrent = recurrent.detach().clone()
        conv_state = conv_state.detach().clone()
        saved_recurrent.append(recurrent.clone())
        saved_conv.append(conv_state.clone())
        # Write into one full-length buffer so tile outputs do not stack up to 2x.
        if output is None:
            output = out.new_empty(out.shape[0], hidden_states.shape[1], out.shape[-1])
        output.narrow(1, start, tile_len).copy_(out)
        del out
    return output, saved_recurrent, saved_conv


def _chain_backward(tile_fn, x, x_grad, grad_output, tiles, saved_recurrent, saved_conv, compute_params, ready_at_end):
    """Reverse walk of one chain inside the caller's parameter hold.

    Does not gather: the caller owns that, because a second
    ``GatheredParameters`` on weights the caller already holds is what a
    nested chain must avoid. ``ready_at_end`` is the caller's verdict on
    whether this chain's last inner backward (tile 0) is the last backward
    of the whole wrapped call. ZeRO reads ``ds_grad_is_ready`` in a hook that
    fires on every inner ``torch.autograd.backward``, and reduces a parameter
    the first time it sees True, so True on any earlier step reduces a
    partial gradient and rejects the remaining ones as computed twice.
    """
    # Cotangents of the tile we just finished, applied to the previous tile's outputs.
    dht = None
    dconv = None
    for tile_i in range(len(tiles) - 1, -1, -1):
        # Ready only once every tile has contributed. Reverse order makes that tile 0.
        # An inner TiledMLP reads the same gate, so its last shard cannot reduce early.
        is_last = ready_at_end and tile_i == 0
        with limit_grad_reduce(is_last):
            _mark_compute_params_ready(compute_params, is_last)
            start, end = tiles[tile_i]
            tile_len = end - start
            x_tile = x.narrow(1, start, tile_len)
            x_tile.requires_grad_(x_grad is not None)
            if x_grad is not None:
                x_tile.grad = x_grad.narrow(1, start, tile_len).view_as(x_tile)
            if tile_i == 0:
                h0 = None
                c0 = None
            else:
                h0 = saved_recurrent[tile_i - 1].detach().clone().requires_grad_(True)
                c0 = saved_conv[tile_i - 1].detach().clone().requires_grad_(True)
            grad_tile = grad_output.narrow(1, start, tile_len)
            with torch.enable_grad():
                out, ht, ct = tile_fn(x_tile, h0, c0)
            outputs = [out]
            grads = [grad_tile]
            # The tail has no consumer of its final state. Earlier tiles do:
            # dL/dS_i includes dL/dS_{i+1} through the next tile's scan.
            if dht is not None:
                outputs.append(ht)
                grads.append(dht)
            if dconv is not None:
                outputs.append(ct)
                grads.append(dconv)
            torch.autograd.backward(outputs, grads)
            dht = _detach_state_grad(h0)
            dconv = _detach_state_grad(c0)
            del out, ht, ct, outputs, grads


class SequentialChainRematerialize(torch.autograd.Function):
    """no_grad chain forward, one-tile recompute backward. See ``sequential_chain_rematerialize``."""

    @staticmethod
    def forward(ctx, tile_fn, hidden_states, tiles, compute_params) -> torch.Tensor:
        ctx.ds_tiled_grad_enabled = _tiled_fwd_grad_enabled
        ctx.tile_fn = tile_fn
        ctx.tiles = tiles
        ctx.compute_params = compute_params if compute_params is not None else []

        with _hold_gathered_params(ctx.compute_params):
            with torch.no_grad():
                output, ctx.saved_recurrent, ctx.saved_conv = _chain_forward(tile_fn, hidden_states, tiles)
        _offload_remat_input(ctx, hidden_states)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x = _load_remat_input(ctx)
        x_requires_grad = x.requires_grad
        x = x.detach()
        x.requires_grad_(x_requires_grad)
        if not x.is_contiguous():
            x = x.contiguous()
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        x_grad = torch.zeros_like(x) if x_requires_grad else None
        with _hold_gathered_params(ctx.compute_params):
            _chain_backward(ctx.tile_fn,
                            x,
                            x_grad,
                            grad_output,
                            ctx.tiles,
                            ctx.saved_recurrent,
                            ctx.saved_conv,
                            ctx.compute_params,
                            ready_at_end=True)

        return None, x_grad, None, None
