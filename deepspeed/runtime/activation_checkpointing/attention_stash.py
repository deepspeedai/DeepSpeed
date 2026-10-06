# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Keep attention outputs for the reentrant recompute of Hugging Face decoder layers (opt-in).

With Hugging Face per-layer reentrant gradient checkpointing, backward recomputes each decoder layer, including
its attention forward. For the last ``num_layers`` decoder layers given to ``install_attention_stash``, the
first forward instead calls the cuDNN SDPA op that ``F.scaled_dot_product_attention`` dispatches to, with the
log-sum-exp its backward needs, and keeps the output. The recompute still recomputes the projections, norms and
rotary embedding, then builds the attention node from the kept output; its backward is the call PyTorch's own
derivative makes. This trades one attention output plus its log-sum-exp per stashed layer, held from that layer's
forward to its recompute, for one attention forward per layer and micro-batch.

Each kept output is keyed by the decoder layer's input. The reentrant recompute runs on a detached alias of that
same tensor, so a recompute can only take the output its own forward produced; a key that does not match fails.
"""

from __future__ import annotations

import weakref
from functools import partial

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend
from torch.utils.checkpoint import checkpoint as torch_checkpoint

_WRAPPER_MARKER = "_deepspeed_attention_stash_wrapper"


def _cudnn_forward(query, key, value, scale):
    return torch.ops.aten._scaled_dot_product_cudnn_attention(query,
                                                              key,
                                                              value,
                                                              None,
                                                              True,
                                                              0.0,
                                                              True,
                                                              False,
                                                              scale=scale)


def _cudnn_backward(grad_output, query, key, value, output, logsumexp, philox_seed, philox_offset, cum_seq_q,
                    cum_seq_k, max_q, max_k, scale):
    # The call autograd makes for _scaled_dot_product_cudnn_attention without bias or dropout, causal.
    return torch.ops.aten._scaled_dot_product_cudnn_attention_backward(grad_output,
                                                                       query,
                                                                       key,
                                                                       value,
                                                                       output,
                                                                       logsumexp,
                                                                       philox_seed,
                                                                       philox_offset,
                                                                       None,
                                                                       cum_seq_q,
                                                                       cum_seq_k,
                                                                       max_q,
                                                                       max_k,
                                                                       0.0,
                                                                       True,
                                                                       scale=scale)


def _sdp_backend(query, key, value, scale) -> int:
    return torch._fused_sdp_choice(query, key, value, None, 0.0, True, scale=scale, enable_gqa=True)


class _StashedAttention(torch.autograd.Function):
    """The recompute's attention node: returns the kept output; backward is cuDNN SDPA's."""

    @staticmethod
    def forward(ctx, query, key, value, kept):
        output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, scale = kept
        ctx.save_for_backward(query, key, value, output, logsumexp, philox_seed, philox_offset)
        ctx.attention_args = (cum_seq_q, cum_seq_k, max_q, max_k, scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, output, logsumexp, philox_seed, philox_offset = ctx.saved_tensors
        cum_seq_q, cum_seq_k, max_q, max_k, scale = ctx.attention_args
        grad_query, grad_key, grad_value = _cudnn_backward(grad_output, query, key, value, output, logsumexp,
                                                           philox_seed, philox_offset, cum_seq_q, cum_seq_k, max_q,
                                                           max_k, scale)
        return grad_query, grad_key, grad_value, None


class AttentionStash:
    """The kept attention outputs of one decoder layer, keyed by that layer's input."""

    def __init__(self, decoder_layer: nn.Module, layer_index: int):
        # Held weakly: the decoder layer owns the attention module that holds this object.
        self._decoder_layer = weakref.ref(decoder_layer)
        self.layer_index = layer_index
        self.entries: dict[tuple, tuple] = {}
        # The decoder layer's input for the call in progress, when it requires grad: (key, tensor).
        self.pending: tuple | None = None

    def record_layer_input(self, module, args, kwargs) -> None:
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        self.pending = None
        if isinstance(hidden_states, torch.Tensor) and hidden_states.requires_grad:
            key = (hidden_states.data_ptr(), hidden_states._version, tuple(hidden_states.shape))
            self.pending = (key, hidden_states)

    def clear_layer_input(self, module, args, output) -> None:
        self.pending = None

    def raise_not_checkpointed(self) -> None:
        raise RuntimeError(
            f"attention stash: decoder layer {self.layer_index} ran a training forward outside a reentrant "
            "checkpoint. It requires Hugging Face per-layer gradient checkpointing with use_reentrant=True, e.g. "
            "model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True}). Do not "
            "install it for other checkpointing.")

    def assert_reentrant_checkpoint(self) -> None:
        decoder_layer = self._decoder_layer()
        checkpoint_func = getattr(decoder_layer, "_gradient_checkpointing_func", None)
        reentrant = (isinstance(checkpoint_func, partial) and checkpoint_func.func is torch_checkpoint
                     and checkpoint_func.keywords.get("use_reentrant") is True)
        if not getattr(decoder_layer, "gradient_checkpointing", False) or not reentrant:
            self.raise_not_checkpointed()

    def keep(self, kept: tuple) -> None:
        key, layer_input = self.pending
        if key in self.entries:
            raise RuntimeError(f"attention stash: decoder layer {self.layer_index} kept two attention "
                               "outputs for the same input.")
        self.entries[key] = kept
        # Dropped if the input's graph is freed without a backward through this layer.
        weakref.finalize(layer_input, self.entries.pop, key, None)

    def take(self) -> tuple:
        kept = None if self.pending is None else self.entries.pop(self.pending[0], None)
        if kept is None:
            raise RuntimeError(f"attention stash: the recompute of decoder layer {self.layer_index} found no "
                               "attention output kept for its input. The forward must run inside the same reentrant "
                               "checkpoint, with an input that requires grad, and not modify that input in place.")
        return kept


def _check_supported(stash: AttentionStash, module, query, key, value, attention_mask, dropout, scaling,
                     is_causal) -> None:
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    if attention_mask is not None or dropout != 0.0 or not causal or query.shape[2] <= 1:
        raise RuntimeError(f"attention stash: decoder layer {stash.layer_index} needs causal attention "
                           "without an attention mask (no padding) and without attention dropout.")
    backend = _sdp_backend(query, key, value, scaling)
    if backend != SDPBackend.CUDNN_ATTENTION.value:
        raise RuntimeError(f"The attention stash keeps cuDNN SDPA outputs, but PyTorch selects SDPA backend "
                           f"{backend} for decoder layer {stash.layer_index}.")


def _stashing_sdpa(original,
                   module,
                   query,
                   key,
                   value,
                   attention_mask,
                   dropout=0.0,
                   scaling=None,
                   is_causal=None,
                   **kwargs):
    stash = getattr(module, "_deepspeed_attention_stash", None)
    if stash is None:
        return original(module,
                        query,
                        key,
                        value,
                        attention_mask,
                        dropout=dropout,
                        scaling=scaling,
                        is_causal=is_causal,
                        **kwargs)
    in_backward = torch._C._current_graph_task_id() != -1
    recompute = torch.is_grad_enabled() and in_backward
    # Inside a reentrant checkpoint the first forward runs without grad; its input requiring grad means a
    # backward through this layer will recompute it.
    first_forward = not torch.is_grad_enabled() and not in_backward and stash.pending is not None
    if not recompute and not first_forward:
        # A reentrant checkpoint's forward runs without grad, so a training forward with grad is not checkpointed.
        if module.training and torch.is_grad_enabled():
            stash.raise_not_checkpointed()
        # Evaluation, and forwards whose input does not require grad, never recompute.
        return original(module,
                        query,
                        key,
                        value,
                        attention_mask,
                        dropout=dropout,
                        scaling=scaling,
                        is_causal=is_causal,
                        **kwargs)
    stash.assert_reentrant_checkpoint()
    _check_supported(stash, module, query, key, value, attention_mask, dropout, scaling, is_causal)
    if first_forward:
        output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, _ = _cudnn_forward(
            query, key, value, scaling)
        stash.keep((output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, scaling))
    else:
        kept = stash.take()
        if kept[0].shape != query.shape[:3] + value.shape[3:]:
            raise RuntimeError(f"attention stash: the output kept for decoder layer {stash.layer_index} does "
                               "not match its recompute.")
        output = _StashedAttention.apply(query, key, value, kept)
    # As in Hugging Face's sdpa_attention_forward.
    return output.transpose(1, 2).contiguous(), None


def _install_sdpa_wrapper() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    current = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(current, _WRAPPER_MARKER, False):
        return
    # Attention modules without a stash pass straight through to the original function.
    wrapper = partial(_stashing_sdpa, current)
    setattr(wrapper, _WRAPPER_MARKER, True)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapper


def install_attention_stash(model: nn.Module, num_layers: int) -> list[int]:
    """Keep the attention outputs of the last ``num_layers`` decoder layers for their reentrant recompute.

    Returns the indices, among the model's decoder layers, of the layers that keep them.
    """
    from transformers.modeling_layers import GradientCheckpointingLayer

    if isinstance(num_layers, bool) or not isinstance(num_layers, int):
        raise ValueError(f"install_attention_stash needs an integer num_layers, got {num_layers!r}")
    # The recompute is recognized as a grad-enabled forward inside a backward pass, and the kept output is replayed
    # through PyTorch's cuDNN SDPA backward.
    missing = [
        name for owner, name in ((torch._C, "_current_graph_task_id"), (torch, "_fused_sdp_choice"),
                                 (torch.ops.aten, "_scaled_dot_product_cudnn_attention_backward"))
        if not hasattr(owner, name)
    ]
    if missing:
        raise ValueError(f"The attention stash needs {missing}, which this PyTorch build lacks.")
    attn_implementation = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if attn_implementation != "sdpa":
        raise ValueError(f"The attention stash requires attn_implementation='sdpa', but the model uses "
                         f"{attn_implementation!r}.")
    decoder_layers = [
        module for module in model.modules()
        if isinstance(module, GradientCheckpointingLayer) and isinstance(getattr(module, "self_attn", None), nn.Module)
    ]
    if not 0 < num_layers <= len(decoder_layers):
        raise ValueError(f"install_attention_stash got num_layers={num_layers}, but the model has "
                         f"{len(decoder_layers)} decoder layers with a self_attn module.")
    _install_sdpa_wrapper()
    first = len(decoder_layers) - num_layers
    for index, decoder_layer in enumerate(decoder_layers[first:], start=first):
        if hasattr(decoder_layer.self_attn, "_deepspeed_attention_stash"):
            raise ValueError(f"The attention stash is already installed on decoder layer {index}.")
        stash = AttentionStash(decoder_layer, index)
        decoder_layer.self_attn._deepspeed_attention_stash = stash
        decoder_layer.register_forward_pre_hook(stash.record_layer_input, with_kwargs=True)
        decoder_layer.register_forward_hook(stash.clear_layer_input)
    return list(range(first, len(decoder_layers)))
