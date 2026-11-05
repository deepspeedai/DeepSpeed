# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Hugging Face integration for packed-sequence tiled rematerialization and native CPU
activation offload.

``install_packed_seq_tiled_rematerialize(model, token_budget=None, shards=None)`` patches the
``self_attn`` / ``linear_attn`` mixer classes of a transformers causal LM so that, for a
packed row (batch 1, samples separated by ``position_ids`` resets), each mixer forward runs
through ``packed_seq_tiled_rematerialize_compute``: the row is split into tiles of whole
samples, at most ``token_budget`` tokens each, and each tile is recomputed in backward.

Packed callers often pass only ``position_ids``; transformers derives ``cu_seqlens`` from
them lazily inside ``_flash_attention_forward``, past the mixer boundary this patch
intercepts. So the tiled forward derives ``cu_seqlens`` (and ``seq_idx`` for linear-attention
GDN mixers) from ``position_ids`` itself, using the same helper transformers would call, and
captures ``position_ids`` at the top-level call for mixers that are not passed them
(Qwen3.5's decoder layer forwards them to ``self_attn`` only). Explicit
``cu_seq_lens_q`` / ``seq_idx`` kwargs are used when present.

GDN mixers are only patched when their kernels reset recurrent state at packed-sample
boundaries; otherwise installation raises, because tiling would silently change results.

``enable_native_activation_checkpoint_offload(model)`` switches the model's gradient
checkpointing to DeepSpeed's non-reentrant checkpoint with the CPU activation-offload
engine.

Requires transformers with ``modeling_flash_attention_utils.prepare_fa_kwargs_from_position_ids``
and flash attention. Use ZeRO stage 2: under ZeRO-3 the tiled mixers must be marked as leaf
modules before ``deepspeed.initialize``.
"""

import inspect
import sys

import torch

_logged_position_ids_fallback = False
_logged_captured_position_ids = False

# position_ids as seen at the top-level model(...) call. Qwen3.5's decoder layer forwards
# position_ids to self_attn but NOT to linear_attn (modeling_qwen3_5.py:844-859), so a GDN
# mixer cannot be given them from above; they have to be captured here instead. Not cleared
# after the forward: under the offload arms the checkpoint re-runs the layer forward during
# backward, and the value must still be present at recompute time.
_captured_position_ids = None

# (kernel function, kwarg that resets state at packed boundaries, state, package providing it)
# Ported from bench_qwen3_offload.py's module-level _GDN_BOUNDARY_RESET_KERNELS table.
_GDN_BOUNDARY_RESET_KERNELS = (
    ("torch_chunk_gated_delta_rule", "cu_seqlens", "recurrent state", "fla (flash-linear-attention)"),
    ("causal_conv1d_fn", "seq_idx", "short-conv state", "causal-conv1d"),
)


def _hf_kernel_applicable_params(fn):
    """Kwarg names transformers' use_kernel_func_from_hub_with_fallback lets through, or None if not that wrapper.

    That wrapper keeps only kwargs *named* in the resolved implementation's signature, so
    the torch fallback silently drops cu_seqlens / seq_idx (a ``**kwargs`` catch-all does
    not count - its name is just "kwargs"). Ported from bench_qwen3_offload.py as-is.
    """
    while callable(fn):
        try:
            nonlocals = inspect.getclosurevars(fn).nonlocals
        except TypeError:
            return None
        if "applicable_params" in nonlocals:
            return nonlocals["applicable_params"]
        fn = getattr(fn, "__wrapped__", None)
    return None


def _gdn_boundary_reset_gaps(mixer_cls):
    """Reasons ``mixer_cls`` would NOT reset GDN state at every packed boundary; empty if it does.

    Ported from bench_qwen3_offload.py as-is.
    """
    module = sys.modules.get(mixer_cls.__module__)
    gaps = []
    for func_name, reset_kwarg, state, package in _GDN_BOUNDARY_RESET_KERNELS:
        params = _hf_kernel_applicable_params(getattr(module, func_name, None))
        if params is None:
            gaps.append(f"{state}: no transformers kernel-fallback wrapper named {func_name} in "
                        f"{mixer_cls.__module__}, so boundary resets cannot be verified")
        elif reset_kwarg not in params:
            gaps.append(
                f"{state}: {func_name} resolved to an implementation that drops '{reset_kwarg}' (install {package})")
    return gaps


def _seq_idx_from_cu_seqlens(cu_seqlens, dtype):
    lengths = cu_seqlens.diff().long()
    sample_ids = torch.arange(lengths.numel(), device=cu_seqlens.device, dtype=dtype)
    return torch.repeat_interleave(sample_ids, lengths)


def _is_text_decoder_layer(mod):
    """Ported from bench_qwen3_offload.py's module-level _is_text_decoder_layer as-is."""
    cls_name = type(mod).__name__
    if "Vision" in cls_name:
        return False
    if "DecoderLayer" in cls_name:
        return True
    has_mixer = hasattr(mod, "self_attn") or hasattr(mod, "linear_attn")
    return has_mixer and hasattr(mod, "mlp")


def _install_top_level_position_ids_capture(model):
    """Record `position_ids` from the top-level model(...) call for mixers that never see it.

    Qwen3.5's decoder layer passes position_ids to `self_attn` only, so its GDN
    (`linear_attn`) mixers are called without them and cannot derive packed-sample
    boundaries. A pre-hook on the top-level module keeps the batch's position_ids where the
    tiled mixer forward can reach them. Registered once per model; harmless for
    full-attention models, which never consult it.
    """
    if getattr(model, "_at_position_ids_capture_installed", False):
        return

    def _capture(_module, _args, kwargs):
        global _captured_position_ids
        pids = kwargs.get("position_ids")
        if pids is not None:
            _captured_position_ids = pids
        return None

    model.register_forward_pre_hook(_capture, with_kwargs=True)
    model._at_position_ids_capture_installed = True


def install_packed_seq_tiled_rematerialize(model, token_budget=None, shards=None):
    """Install token-axis tiled rematerialization on every decoder layer's self_attn/
    linear_attn forward, for packed-sequence training.

    Important: like `enable_tiled_mlp_compute`, this monkey-patches the mixer *class*
    (shared across all instances of that class), so it must run before the model is
    instantiated via `self.config.model.factory(self)()`.
    """
    if token_budget is None and shards is None:
        raise ValueError("install_packed_seq_tiled_rematerialize needs token_budget or shards")

    global _logged_position_ids_fallback

    try:
        from deepspeed.runtime.activation_checkpointing.tiled_rematerialization import classify_packed_kwarg
        from deepspeed.runtime.activation_checkpointing.tiled_rematerialization import (
            packed_seq_tiled_rematerialize_compute, )
    except ImportError as e:
        raise ImportError("packed_seq_tiled_remat requires the sfc-gh-truwase/DeepSpeed fork "
                          "(docs/cpu-checkpointing-perf-note branch or newer) ahead of the installed "
                          "deepspeed on PYTHONPATH, e.g.:\n"
                          '  export PYTHONPATH="<fork checkout>:$PYTHONPATH"\n'
                          f"Original import error: {e}") from e

    from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids

    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        dropout = getattr(mod, "attention_dropout", None)
        if isinstance(dropout, float):
            mod.attention_dropout = 0.0

    _install_top_level_position_ids_capture(model)

    mixers = []
    linear_attn_classes = set()
    for layer in model.modules():
        if not _is_text_decoder_layer(layer):
            continue
        mixer = getattr(layer, "self_attn", None)
        if mixer is None:
            mixer = getattr(layer, "linear_attn", None)
            if mixer is not None:
                linear_attn_classes.add(type(mixer))
        if mixer is not None:
            mixers.append(mixer)
    for cls in linear_attn_classes:
        gaps = _gdn_boundary_reset_gaps(cls)
        if gaps:
            # Refuse rather than tile: the untiled packed baseline already leaks state
            # across samples, and tiling would reset it only at tile boundaries -
            # different numbers from both baseline and truth.
            raise RuntimeError(f"packed_seq_tiled_remat: {cls.__name__} does not reset at packed sample boundaries: " +
                               "; ".join(gaps))

    _HF_PACKED_MAX_LEN_NAMES = ("max_length_q", "max_length_k", "max_seqlen_q", "max_seqlen_k")

    def _squeeze_batch(value):
        if torch.is_tensor(value):
            if value.dim() >= 2 and value.shape[0] == 1:
                return value.squeeze(0), True
            return value, False
        if isinstance(value, tuple):
            items = [_squeeze_batch(v) for v in value]
            if items and all(sq for _, sq in items):
                return tuple(v for v, _ in items), True
            return value, False
        return value, False

    patched = []
    for mixer in mixers:
        cls = type(mixer)
        if cls in patched:
            continue
        orig_fwd = cls.forward
        sig = inspect.signature(orig_fwd)

        def _make(orig, fwd_sig, cls_budget, needs_seq_idx):

            def tiled_forward(self, hidden_states, *args, **kwargs):
                global _logged_position_ids_fallback, _logged_captured_position_ids

                bound = fwd_sig.bind_partial(self, hidden_states, *args, **kwargs)
                bound.apply_defaults()
                call_kw = {}
                for name, value in bound.arguments.items():
                    if name in ("self", "hidden_states"):
                        continue
                    param = fwd_sig.parameters.get(name)
                    if param is not None and param.kind == param.VAR_KEYWORD:
                        call_kw.update(value or {})
                    else:
                        call_kw[name] = value

                cu_seqlens = call_kw.pop("cu_seq_lens_q", None)
                if cu_seqlens is None:
                    cu_seqlens = call_kw.pop("cu_seq_lens_k", None)
                else:
                    call_kw.pop("cu_seq_lens_k", None)
                for meta_name in _HF_PACKED_MAX_LEN_NAMES:
                    call_kw.pop(meta_name, None)

                position_ids = call_kw.get("position_ids")
                if position_ids is None and _captured_position_ids is not None:
                    # GDN mixers are called without position_ids (the decoder layer gives
                    # them to self_attn only), so fall back to what the top-level model call
                    # was given. Validate the token count: a capture left over from a
                    # different microbatch would otherwise be used silently and produce
                    # wrong sample boundaries.
                    candidate = _captured_position_ids
                    n_tokens = hidden_states.shape[1] if hidden_states.dim() == 3 else hidden_states.shape[0]
                    if int(candidate.reshape(-1).shape[0]) != int(n_tokens):
                        raise RuntimeError("packed_seq_tiled_remat: captured position_ids describe "
                                           f"{int(candidate.reshape(-1).shape[0])} tokens but "
                                           f"{type(self).__name__}.forward was handed {int(n_tokens)}. "
                                           "The capture is from a different forward pass; this is only "
                                           "safe with one forward per backward "
                                           "(micro_batch_size=1, gradient_accumulation_steps=1).")
                    # Used locally to derive cu_seqlens/seq_idx only -- deliberately NOT
                    # added to call_kw, since a GDN mixer's forward does not accept
                    # position_ids and would raise TypeError on the unexpected kwarg.
                    position_ids = candidate
                    if not _logged_captured_position_ids:
                        print(
                            "[deepspeed] packed_seq_tiled_remat: using position_ids "
                            "captured at the top-level model(...) call for "
                            f"{type(self).__name__} (its decoder layer forwards them to "
                            "self_attn only)",
                            file=sys.stderr,
                        )
                        _logged_captured_position_ids = True
                if cu_seqlens is None and position_ids is not None:
                    (cu_seq_lens_q, _), _ = prepare_fa_kwargs_from_position_ids(position_ids)
                    cu_seqlens = cu_seq_lens_q
                    if not _logged_position_ids_fallback:
                        print(
                            "[deepspeed] packed_seq_tiled_remat: derived cu_seqlens from "
                            "position_ids (no explicit cu_seq_lens_q/cu_seq_lens_k at the "
                            "self_attn.forward boundary)",
                            file=sys.stderr,
                        )
                        _logged_position_ids_fallback = True

                if cu_seqlens is None:
                    raise RuntimeError(f"packed_seq_tiled_remat: {type(self).__name__}.forward has no "
                                       "cu_seq_lens_q/cu_seq_lens_k, no position_ids, and nothing was "
                                       "captured at the top-level model(...) call. Packed training needs "
                                       "position_ids in the batch (a packed SFT collator emits them "
                                       "when pack_samples is on).")
                num_samples = int(cu_seqlens.shape[0]) - 1
                seq_idx = call_kw.pop("seq_idx", None)
                if seq_idx is None and needs_seq_idx and position_ids is not None:
                    seq_idx = _seq_idx_from_cu_seqlens(cu_seqlens, torch.int32).reshape(position_ids.shape)
                if seq_idx is None and needs_seq_idx:
                    raise RuntimeError(f"packed_seq_tiled_remat: {type(self).__name__}.forward needs seq_idx, "
                                       "or position_ids in the batch so it can be derived, otherwise the "
                                       "untiled baseline's short conv slides across packed samples")
                if seq_idx is not None:
                    expected_seq_idx = _seq_idx_from_cu_seqlens(cu_seqlens, seq_idx.dtype)
                    if not torch.equal(seq_idx.reshape(-1), expected_seq_idx):
                        raise RuntimeError("packed_seq_tiled_remat: seq_idx disagrees with cu_seq_lens_q")

                had_batch_dim = hidden_states.dim() == 3
                if had_batch_dim:
                    if hidden_states.shape[0] != 1:
                        raise RuntimeError("packed_seq_tiled_remat requires batch=1, got "
                                           f"hidden_states.shape={tuple(hidden_states.shape)}")
                    hidden_states_2d = hidden_states.squeeze(0)
                else:
                    hidden_states_2d = hidden_states

                squeezed_names = set()
                to_shard = {}
                to_pass = {}
                for name, value in call_kw.items():
                    lname = name.lower()
                    if name != "cache_position" and (lname in ("past_key_value", "past_key_values", "cache_params",
                                                               "cache") or "cache" in lname):
                        if value is not None and not isinstance(value, bool):
                            raise TypeError(
                                f"packed_seq_tiled_remat: {name} must be None in training, got {type(value).__name__}")
                        to_pass[name] = None
                        continue
                    sq_value, was_sq = _squeeze_batch(value)
                    if was_sq:
                        squeezed_names.add(name)
                    try:
                        kind = classify_packed_kwarg(name, sq_value, num_samples)
                    except TypeError:
                        if value is None:
                            to_pass[name] = None
                            continue
                        raise
                    if kind == "token":
                        to_shard[name] = sq_value
                    else:
                        to_pass[name] = sq_value

                orig_tuple_len = []

                def _fn(h, **kw):
                    tile_cu_seqlens = kw.pop("cu_seqlens")
                    local_max = int(tile_cu_seqlens.diff().max())
                    for name in squeezed_names:
                        if name not in kw:
                            continue
                        v = kw[name]
                        kw[name] = v.unsqueeze(0) if torch.is_tensor(v) else tuple(x.unsqueeze(0) for x in v)
                    kw["cu_seq_lens_q"] = tile_cu_seqlens
                    kw["cu_seq_lens_k"] = tile_cu_seqlens
                    kw["max_length_q"] = local_max
                    kw["max_length_k"] = local_max
                    if seq_idx is not None:
                        tile_seq_idx = _seq_idx_from_cu_seqlens(tile_cu_seqlens, seq_idx.dtype)
                        kw["seq_idx"] = tile_seq_idx.reshape(*seq_idx.shape[:-1], -1)
                    h_in = h.unsqueeze(0) if had_batch_dim else h
                    out = orig(self, h_in, **kw)
                    if torch.is_tensor(out):
                        result = out
                    elif isinstance(out, tuple) and out and torch.is_tensor(out[0]):
                        extra = [x for x in out[1:] if x is not None]
                        if extra:
                            raise TypeError(f"packed_seq_tiled_remat: {type(self).__name__} returned extra outputs")
                        orig_tuple_len[:] = [len(out)]
                        result = out[0]
                    else:
                        raise TypeError(f"packed_seq_tiled_remat: {type(self).__name__} returned {type(out)}")
                    return result.squeeze(0) if had_batch_dim else result

                params = [p for p in self.parameters() if p.requires_grad]
                out2d = packed_seq_tiled_rematerialize_compute(
                    _fn,
                    hidden_states_2d,
                    cu_seqlens,
                    shards=shards,
                    token_budget=cls_budget,
                    compute_params=params,
                    kwargs_to_token_shard=to_shard,
                    kwargs_to_pass=to_pass,
                )
                out = out2d.unsqueeze(0) if had_batch_dim else out2d
                if not orig_tuple_len:
                    return out
                return (out, ) + (None, ) * (orig_tuple_len[0] - 1)

            return tiled_forward

        cls.forward = _make(orig_fwd, sig, token_budget, cls in linear_attn_classes)
        patched.append(cls)

    n = sum(1 for m in model.modules() if type(m) in patched)
    print(
        f"[deepspeed] packed_seq_tiled_remat token_budget={token_budget} shards={shards} "
        f"classes={[c.__name__ for c in patched]} modules={n}",
        file=sys.stderr,
    )
    if n == 0:
        raise RuntimeError("packed_seq_tiled_remat requested but no self_attn/linear_attn mixer modules found")
    return patched


def enable_native_activation_checkpoint_offload(model):
    """Swap this model's activation checkpointing over to the DeepSpeed fork's native,
    non-reentrant, CPU-offloading engine, in place of HF's default reentrant checkpoint
    function.

    Must run after the model is instantiated (and after `model.gradient_checkpointing_enable()`
    has already run inside `HFModelFactory.post_create_model_callback`, which sets each
    layer's default `_gradient_checkpointing_func`) - this only overrides that func.
    """
    try:
        import deepspeed.runtime.activation_checkpointing.checkpointing as ds_ckpt
        from deepspeed.runtime.activation_checkpointing.offload_activations import _ActivationOffloadEngine
    except ImportError as e:
        raise ImportError("native_activation_checkpoint_cpu_offload requires the sfc-gh-truwase/DeepSpeed "
                          "fork (docs/cpu-checkpointing-perf-note branch or newer) ahead of the installed "
                          "deepspeed on PYTHONPATH, e.g.:\n"
                          '  export PYTHONPATH="<fork checkout>:$PYTHONPATH"\n'
                          f"Original import error: {e}") from e

    import deepspeed
    from transformers.modeling_layers import GradientCheckpointingLayer

    # Bound the pinned-buffer pool. The engine's default cap is 4096 *buffers*, which is
    # harmless at short sequences (~256 MiB each) and fatal at long ones (~1.94 GiB each at
    # 253,952 tokens): host peaked at 1890/2000 GiB with only 79 GiB of GPU in use, so the
    # job was killed for host exhaustion while HBM sat idle. Only a few buffers are ever
    # in flight (keep_last_count=1 plus undrained D2H copies) and reuse is keyed by shape,
    # so one per layer is plenty -- this allows 2x that.
    # Pool size must be bounded by BYTES, not by layer count. Measured the hard way:
    # capping at 2*num_hidden_layers (72 for Qwen3-8B) changed nothing -- host stayed at
    # 1891 GiB and 294,912 still died -- because at 253,952 tokens each buffer is 1.94 GiB,
    # so 72 buffers still permits ~140 GiB/rank (~1.1 TB across 8 ranks), which is the
    # entire excess. That is the same mistake as the engine's own 4096 default: a count cap
    # means something completely different at 256 MiB/buffer than at 1.94 GiB/buffer.
    #
    # What actually bounds the requirement is concurrency, not depth. Buffers are keyed by
    # shape, and every decoder layer offloads the same shape, so one buffer can serve all
    # layers in turn as each D2H drains; only keep_last_count=1 plus a few undrained copies
    # are ever live at once. A small constant is therefore both sufficient and
    # length-independent. Kept at 8 rather than 2 so reuse still absorbs copies that have
    # not drained -- the pool exists to avoid cold cudaHostAlloc, whose stall is its own
    # pathology (see autorun/DIAGNOSTIC_packed_1m_allreduce_wedge.md).
    n_layers = getattr(getattr(model, "config", None), "num_hidden_layers", 0) or 32
    pool_buffers = 8
    ds_ckpt._cpu_offload_engine = _ActivationOffloadEngine(
        keep_last_count=1,
        min_offload_bytes=0,
        use_pin_memory=True,
        use_streams=True,
        prefetch=bool(getattr(ds_ckpt, "_cpu_offload_prefetch", False)),
        max_cpu_buffer_pool_count=pool_buffers,
    )
    print(
        f"[deepspeed] activation-offload pinned pool capped at {pool_buffers} buffers "
        f"(model has {n_layers} layers; engine default 4096 is unbounded in bytes)",
        file=sys.stderr,
    )
    deepspeed.checkpointing.configure(mpu_=None, checkpoint_in_cpu=True)

    n = 0
    for m in model.modules():
        if isinstance(m, GradientCheckpointingLayer):
            m.gradient_checkpointing = True
            m._gradient_checkpointing_func = ds_ckpt.non_reentrant_checkpoint
            n += 1
    print(f"[deepspeed] native_activation_checkpoint_cpu_offload enabled on {n} layers", file=sys.stderr)
    if n == 0:
        raise RuntimeError(
            "native_activation_checkpoint_cpu_offload requested but no GradientCheckpointingLayer modules found")


__all__ = ["install_packed_seq_tiled_rematerialize", "enable_native_activation_checkpoint_offload"]
