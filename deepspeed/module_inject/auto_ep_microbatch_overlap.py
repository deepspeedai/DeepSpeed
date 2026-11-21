# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Hide AutoEP's DeepEP transfers behind the computation of another micro-batch, as Megatron's combined 1F1B does.

Within one optimizer step, the backward of micro-batch i runs layer by layer together with the forward of micro-batch
i+1. At each layer step the forward of one decoder layer of micro-batch i+1 is paired with the backward of one decoder
layer of micro-batch i, and every transfer of one is queued beside a computation of the other::

    transfer on DeepEP's stream            computation beside it
    replayed dispatch (i), combine         attention + router (i+1)
      backward (i)
    dispatch (i+1)                         expert recompute and expert backward (i)
    dispatch backward (i)                  experts (i+1)
    combine (i+1)                          attention backward (i)

Both micro-batches keep their full size, so no matrix multiply shrinks. Every decoder layer is recomputed in its
backward, keeping only its input and its DeepEP handle, as with full activation checkpointing; the recompute replays
the forward's dispatch layout and does not repeat the forward's combine.

Each micro-batch's loss is the output of an autograd Function whose backward runs that micro-batch's backward
together with the next micro-batch's forward, so ``engine.backward(loss)`` keeps DeepSpeed's loss scaling, ZeRO
gradient hooks and accumulation-boundary bookkeeping. Each parameter is used by exactly one piece of each micro-batch,
so its gradient is accumulated once per micro-batch, as without the overlap.

Usage, replacing the forward/backward of one optimizer step's micro-batches::

    overlap = MicroBatchOverlap(engine)
    losses = overlap.forward_backward([{"input_ids": ids, "attention_mask": mask, "labels": ids}, ...])
    engine.step()

``forward_backward`` marks the last micro-batch as the gradient accumulation boundary and calls ``engine.backward``
for every micro-batch; the caller makes the one ``engine.step`` call, as with
``engine.set_gradient_accumulation_boundary`` in a plain loop (and as accelerate drives DeepSpeed).

Supported: Hugging Face causal LMs whose decoder layers' ``mlp`` are AutoEP layers on ``comm_backend="deepep"`` (DeepEP
v2) with ``autoep_size > 1``, ZeRO stages 0-2. Not supported, and rejected at construction: ZeRO-3, shared experts,
the router auxiliary loss, AutoEP+AutoTP folding, tied embeddings, attention dropout, and the layer options
``overlap_weight_grad`` and ``skip_recompute_combine`` (this schedule replaces both).
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.auto_ep_comm import DEEPEP_BACKEND, _conform_rows, sm_carveout
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer, apply_deepep_row_weights
from deepspeed.utils import log_dist


def _leaf(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """A graph input for one piece of the layer, so that piece can be differentiated on its own."""
    if tensor is None:
        return None
    return tensor.detach().requires_grad_(tensor.is_floating_point())


class _StopForward(Exception):
    """Raised by the layer-0 pre-hook once the model's preamble has produced what the layers need."""


@dataclass
class _Forward:
    """One decoder layer's forward for one micro-batch, between its pieces."""
    x: torch.Tensor = None
    residual: torch.Tensor = None
    tokens: torch.Tensor = None
    top_scores: torch.Tensor = None
    selected_experts: torch.Tensor = None
    counts: torch.Tensor = None
    ready: object = None
    received: torch.Tensor = None
    received_weights: torch.Tensor = None
    handle: object = None
    dispatched: object = None
    expert_output: torch.Tensor = None
    combined: torch.Tensor = None
    combined_event: object = None


@dataclass
class _Backward:
    """One decoder layer's backward (with its recompute) for one micro-batch, between its pieces."""
    x: torch.Tensor = None
    handle: object = None
    grad_out: torch.Tensor = None
    residual: torch.Tensor = None
    tokens: torch.Tensor = None
    top_scores: torch.Tensor = None
    selected_experts: torch.Tensor = None
    received_buffer_rows: int = 0
    weights_buffer_rows: int = 0
    received: torch.Tensor = None
    received_weights: torch.Tensor = None
    replayed: object = None
    expert_output: torch.Tensor = None
    grad_rows: torch.Tensor = None
    grad_rows_event: object = None
    sent: tuple = None


@dataclass
class _MicroBatch:
    batch: dict
    index: int = 0
    embeds: torch.Tensor = None
    layer_kwargs: dict = None
    hidden: torch.Tensor = None
    saved: list = field(default_factory=list)
    final_input: torch.Tensor = None
    loss: torch.Tensor = None
    loss_value: torch.Tensor = None
    grad: torch.Tensor = None
    next: "_MicroBatch" = None


class _OverlappedLoss(torch.autograd.Function):
    """A micro-batch's loss; its backward runs this micro-batch's backward together with the next one's forward."""

    @staticmethod
    def forward(ctx, runner, micro_batch, anchor):
        ctx.runner = runner
        ctx.micro_batch = micro_batch
        return micro_batch.loss_value.clone()

    @staticmethod
    def backward(ctx, grad):
        ctx.runner._backward_with_next_forward(ctx.micro_batch, grad)
        ctx.runner = ctx.micro_batch = None
        return None, None, None


class MicroBatchOverlap:
    """Runs one optimizer step's micro-batches with each forward beside the previous micro-batch's backward."""

    # Without a host wait of its own, a backward that is not paired with a forward lets the host queue layer after
    # layer while DeepEP's asynchronous calls keep each touched block unreusable until the GPU catches up. Waiting for
    # the layer two steps earlier bounds that to two layers.
    _BACKWARD_LOOKAHEAD = 2

    def __init__(self,
                 engine,
                 sm_carveout: bool = True,
                 dispatch_from_event: bool = True,
                 saved_input_buffer: bool = True):
        """
        Args:
            engine: the DeepSpeed engine wrapping the model.
            sm_carveout: while a DeepEP transfer is in flight, run the computation beside it on all SMs but the
                ``comm_num_sm`` the transfer uses, so the two do not wait for each other.
            dispatch_from_event: start each paired forward dispatch from an event captured after the forward
                attention, so the host's wait for the received row counts overlaps the backward's expert work.
            saved_input_buffer: keep the layer inputs saved for the backward in one preallocated buffer instead of
                allocating them one by one (see ``_saved_input_slot``).
        """
        self.engine = engine
        model = engine.module
        base = getattr(model, "model", None)
        missing = [name for name in ("embed_tokens", "layers", "norm") if not hasattr(base, name)]
        if base is None or missing or not hasattr(model, "lm_head") or not hasattr(model, "loss_function"):
            raise ValueError("MicroBatchOverlap runs a Hugging Face causal LM (model.model.embed_tokens/layers/norm, "
                             "model.lm_head, model.loss_function).")
        if engine.zero_optimization_stage() > 2:
            raise ValueError("MicroBatchOverlap supports ZeRO stages 0-2: it runs each decoder layer's pieces itself, "
                             "outside the module hooks through which ZeRO-3 gathers parameters.")
        config = model.config
        if getattr(config, "tie_word_embeddings", False):
            raise ValueError("MicroBatchOverlap does not support tied input and output embeddings: the shared "
                             "weight would receive two gradient accumulations per micro-batch.")
        if getattr(config, "output_router_logits", False):
            raise ValueError("MicroBatchOverlap does not compute the router auxiliary loss; set "
                             "output_router_logits=False.")
        if getattr(config, "attention_dropout", 0.0):
            raise ValueError("MicroBatchOverlap recomputes attention without restoring the random state, so it "
                             "needs attention_dropout=0.")
        for index, layer in enumerate(base.layers):
            moe = getattr(layer, "mlp", None)
            if not isinstance(moe, AutoEPMoELayer):
                raise ValueError(f"MicroBatchOverlap needs every decoder layer's mlp to be an AutoEP layer; layer "
                                 f"{index} has {type(moe).__name__}.")
            if moe.comm_backend != DEEPEP_BACKEND or moe.ep_size == 1:
                raise ValueError('MicroBatchOverlap overlaps DeepEP transfers; set comm_backend="deepep" with '
                                 "autoep_size > 1.")
            if moe.shared_experts is not None or moe.return_router_logits or moe.folding_group_handles is not None:
                raise ValueError("MicroBatchOverlap does not yet run shared experts, router logits or "
                                 "AutoEP+AutoTP folding.")
            if moe.overlap_weight_grad or moe.skip_recompute_combine:
                # Both act inside the layer's own forward, which this schedule does not call.
                raise ValueError("MicroBatchOverlap replaces overlap_weight_grad and skip_recompute_combine; leave "
                                 "both unset in the expert_parallel config.")
        if os.environ.get("EP_AVOID_RECORD_STREAM") != "1":
            # DeepEP reads this on every asynchronous call. Unset, it marks every tensor of an in-flight transfer as in
            # use by both streams, so the allocator cannot reuse those blocks until each stream's later work has run;
            # with two micro-batches in flight that fragments the cache. Set, DeepEP keeps the tensors alive in the
            # returned event instead, and this schedule waits on every event before it drops it.
            log_dist(
                "MicroBatchOverlap: set EP_AVOID_RECORD_STREAM=1 in the environment for DeepEP's asynchronous "
                "calls; without it, the allocator cache holds more memory and can flush in each step.",
                ranks=[0],
                level=logging.WARNING)
        self.model = model
        self.base = base
        self.layers = list(base.layers)
        self.sm_carveout = sm_carveout
        # The forward dispatch then allocates its outputs from DeepEP's stream's cache, so that it can start before the
        # compute stream reaches it.
        self.dispatch_from_event = dispatch_from_event
        # Every layer's input is kept until its backward. Allocated as they come, the inputs of the micro-batch going
        # forward land between the large recompute temporaries of the one going backward and stay there until their
        # own backward, which fragments the cache step after step. One allocation of num_layers + 1 slots holds them
        # instead: an even micro-batch keeps layer l's input in slot l, an odd one in slot num_layers - l, so each
        # forward layer writes into exactly the slot the backward layer paired with it has just finished with.
        self.saved_input_buffer = saved_input_buffer
        self._saved_input_slots = None
        self._anchor = None
        self._backward_done = deque()

    # ----------------------------------------------------------------------------------------------- public entry

    def forward_backward(self, micro_batches: list[dict]) -> list[torch.Tensor]:
        """Forward and backward of ``micro_batches``, in order, overlapped. Returns their detached losses."""
        if not micro_batches:
            return []
        states = [_MicroBatch(batch=dict(batch), index=index) for index, batch in enumerate(micro_batches)]
        for current, following in zip(states, states[1:]):
            current.next = following
        if self._anchor is None:
            self._anchor = torch.zeros((), device=get_accelerator().current_device_name(), requires_grad=True)

        self._forward_preamble(states[0])
        for index in range(len(self.layers)):
            self._forward_layer_alone(states[0], index)
        self._forward_loss(states[0])

        from deepspeed.runtime.utils import register_output_backward_hooks
        losses = []
        for position, state in enumerate(states):
            loss = _OverlappedLoss.apply(self, state, self._anchor)
            # engine.forward attaches these to the model output; the layers here run outside it. Without them the
            # engine's backward prologue (ZeRO's backward_prologue and enter_backward) would never run.
            register_output_backward_hooks(loss,
                                           preprocess_once_fn=self.engine._backward_prologue,
                                           preprocess_per_tensor_fn=self.engine._backward_prologue_per_tensor)
            self.engine.set_gradient_accumulation_boundary(position == len(states) - 1)
            self.engine.backward(loss)
            losses.append(state.loss_value)
        return losses

    # ---------------------------------------------------------------------------------------- whole-model pieces

    def _forward_preamble(self, micro_batch: _MicroBatch) -> None:
        """Embedding, mask and rotary tables, from the model's own forward, stopped before the first layer."""
        captured = {}

        def capture(_module, args, kwargs):
            captured["hidden"] = args[0] if args else kwargs["hidden_states"]
            captured["kwargs"] = {key: value for key, value in kwargs.items() if key != "hidden_states"}
            raise _StopForward()

        first_layer = self.layers[0]
        handle = first_layer.register_forward_pre_hook(capture, with_kwargs=True)
        # Off for this call, so that the hook stops the forward before Hugging Face's checkpoint wrapper starts; this
        # runs inside a backward pass when it prepares the next micro-batch.
        checkpointing = getattr(first_layer, "gradient_checkpointing", False)
        first_layer.gradient_checkpointing = False
        batch = micro_batch.batch
        try:
            with torch.enable_grad():
                self.base(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), use_cache=False)
        except _StopForward:
            pass
        finally:
            handle.remove()
            first_layer.gradient_checkpointing = checkpointing
        micro_batch.embeds = captured["hidden"]
        kwargs = captured["kwargs"]
        kwargs["use_cache"] = False
        kwargs["past_key_values"] = None
        micro_batch.layer_kwargs = kwargs
        if self.saved_input_buffer:
            slot = self._saved_input_slot(micro_batch, 0, micro_batch.embeds)
            slot.copy_(micro_batch.embeds.detach())
            micro_batch.hidden = slot
        else:
            micro_batch.hidden = micro_batch.embeds.detach()

    def _forward_loss(self, micro_batch: _MicroBatch) -> None:
        """Final norm, output layer and loss, keeping their graph for the backward."""
        micro_batch.final_input = _leaf(micro_batch.hidden)
        micro_batch.hidden = None
        batch = micro_batch.batch
        with torch.enable_grad():
            logits = self.model.lm_head(self.base.norm(micro_batch.final_input))
            micro_batch.loss = self.model.loss_function(logits, batch["labels"], self.model.config.vocab_size)
        micro_batch.loss_value = micro_batch.loss.detach()

    def _backward_loss(self, micro_batch: _MicroBatch, grad: torch.Tensor) -> None:
        torch.autograd.backward(micro_batch.loss, grad)
        micro_batch.grad = micro_batch.final_input.grad
        micro_batch.loss = micro_batch.final_input = None

    def _backward_preamble(self, micro_batch: _MicroBatch) -> None:
        torch.autograd.backward(micro_batch.embeds, micro_batch.grad)
        micro_batch.embeds = micro_batch.grad = None

    # ------------------------------------------------------------------------------------------- the schedule

    def _backward_with_next_forward(self, micro_batch: _MicroBatch, grad: torch.Tensor) -> None:
        following = micro_batch.next
        self._backward_loss(micro_batch, grad)
        if following is not None:
            self._forward_preamble(following)
        num_layers = len(self.layers)
        for step in range(num_layers):
            backward_index = num_layers - 1 - step
            if following is None:
                self._limit_backward_lookahead()
                self._backward_layer_alone(micro_batch, backward_index)
                self._record_backward_done()
            else:
                self._paired_layer_step(following, step, micro_batch, backward_index)
        self._backward_preamble(micro_batch)
        if following is not None:
            self._forward_loss(following)

    def _paired_layer_step(self, forward_mb: _MicroBatch, forward_index: int, backward_mb: _MicroBatch,
                           backward_index: int) -> None:
        """One forward layer of ``forward_mb`` and one backward layer of ``backward_mb``, interleaved."""
        f_layer, b_layer = self.layers[forward_index], self.layers[backward_index]
        f, b = _Forward(), self._start_backward(backward_mb, backward_index)
        exchange = b_layer.mlp._deepep_exchange
        carve = self._beside_transfer(exchange)

        self._recompute_attention(b_layer, backward_mb, b)  # nothing of this pair is in flight yet
        self._replay_dispatch(b_layer, b)
        self._send_combine_backward(b_layer, b)
        with carve():
            self._attention(f_layer, forward_mb, f)
        if self.dispatch_from_event:
            f.ready = f_layer.mlp._deepep_exchange.capture_event()
        else:
            self._dispatch(f_layer, f, previous_event=None)
        with carve():
            self._recompute_experts(b_layer, b)
            self._expert_backward(b)
        if self.dispatch_from_event:
            # The host waits here for the forward dispatch's row counts, while the GPU runs the expert work above.
            self._dispatch(f_layer, f, previous_event=f.ready)
        self._send_dispatch_backward(b_layer, b)
        f.dispatched.current_stream_wait()
        with carve():
            self._experts(f_layer, f)
        self._combine(f_layer, f)
        with carve():
            self._attention_backward(backward_mb, b)
        self._finish_forward(forward_mb, f)

    def _forward_layer_alone(self, micro_batch: _MicroBatch, index: int) -> None:
        layer = self.layers[index]
        f = _Forward()
        self._attention(layer, micro_batch, f)
        self._dispatch(layer, f, previous_event=None)
        f.dispatched.current_stream_wait()
        self._experts(layer, f)
        self._combine(layer, f)
        self._finish_forward(micro_batch, f)

    def _backward_layer_alone(self, micro_batch: _MicroBatch, index: int) -> None:
        layer = self.layers[index]
        b = self._start_backward(micro_batch, index)
        carve = self._beside_transfer(layer.mlp._deepep_exchange)
        self._recompute_attention(layer, micro_batch, b)
        self._replay_dispatch(layer, b)
        self._send_combine_backward(layer, b)
        with carve():
            self._recompute_experts(layer, b)
            self._expert_backward(b)
        self._send_dispatch_backward(layer, b)
        self._attention_backward(micro_batch, b)

    # ----------------------------------------------------------------------------------------- forward pieces

    def _attention(self, layer: nn.Module, micro_batch: _MicroBatch, f: _Forward) -> None:
        f.x = micro_batch.hidden
        with torch.no_grad():
            f.residual, f.tokens, f.top_scores, f.selected_experts, f.counts = _attention_and_router(
                layer, f.x, micro_batch.layer_kwargs)

    def _dispatch(self, layer: nn.Module, f: _Forward, previous_event) -> None:
        moe = layer.mlp
        moe._ensure_deepep_exchange(f.tokens)
        f.received, f.received_weights, f.handle, f.dispatched = moe._deepep_exchange.dispatch_async(
            f.tokens, f.selected_experts, f.top_scores, previous_event=previous_event)

    def _experts(self, layer: nn.Module, f: _Forward) -> None:
        arrived = f.handle.num_expanded_tokens
        with torch.no_grad():
            f.expert_output = _run_experts(layer.mlp, f.received[:arrived],
                                           None if f.received_weights is None else f.received_weights[:arrived],
                                           f.handle)
        f.received = f.received_weights = None

    def _combine(self, layer: nn.Module, f: _Forward) -> None:
        f.combined, f.combined_event = layer.mlp._deepep_exchange.combine_async(f.expert_output, f.handle)
        f.expert_output = None

    def _finish_forward(self, micro_batch: _MicroBatch, f: _Forward) -> None:
        layer_index = len(micro_batch.saved)
        next_index = layer_index + 1
        with torch.no_grad():
            f.combined_event.current_stream_wait()
            combined = _conform_rows(f.combined, f.tokens.shape).view_as(f.residual)
            if self.saved_input_buffer and next_index < len(self.layers):
                output = self._saved_input_slot(micro_batch, next_index, f.residual)
                torch.add(f.residual, combined, out=output)
            else:
                output = f.residual + combined
            self.layers[layer_index].mlp.tokens_per_expert.add_(f.counts)
        micro_batch.saved.append((f.x, f.handle))
        micro_batch.hidden = output

    # ---------------------------------------------------------------------------------------- backward pieces

    def _start_backward(self, micro_batch: _MicroBatch, index: int) -> _Backward:
        x, handle = micro_batch.saved[index]
        micro_batch.saved[index] = None
        b = _Backward(x=_leaf(x), handle=handle, grad_out=micro_batch.grad)
        micro_batch.grad = None
        return b

    def _recompute_attention(self, layer: nn.Module, micro_batch: _MicroBatch, b: _Backward) -> None:
        with torch.enable_grad():
            b.residual, b.tokens, b.top_scores, b.selected_experts, _ = _attention_and_router(
                layer, b.x, micro_batch.layer_kwargs)

    def _replay_dispatch(self, layer: nn.Module, b: _Backward) -> None:
        received, weights, b.replayed = layer.mlp._deepep_exchange.replay_dispatch_async(
            b.tokens, b.selected_experts, b.top_scores, b.handle)
        b.received_buffer_rows = received.shape[0]
        b.weights_buffer_rows = 0 if weights is None else weights.shape[0]
        arrived = b.handle.num_expanded_tokens
        b.received = _leaf(received[:arrived])
        b.received_weights = None if weights is None else _leaf(weights[:arrived])

    def _send_combine_backward(self, layer: nn.Module, b: _Backward) -> None:
        grad = b.grad_out.reshape(-1, b.grad_out.shape[-1]).contiguous()
        b.grad_rows, b.grad_rows_event = layer.mlp._deepep_exchange.dispatch_with_handle_async(grad, b.handle)

    def _recompute_experts(self, layer: nn.Module, b: _Backward) -> None:
        b.replayed.current_stream_wait()
        with torch.enable_grad():
            b.expert_output = _run_experts(layer.mlp, b.received, b.received_weights, b.handle)

    def _expert_backward(self, b: _Backward) -> None:
        b.grad_rows_event.current_stream_wait()
        grad_output = _conform_rows(b.grad_rows, b.expert_output.shape)
        b.grad_rows = None
        torch.autograd.backward(b.expert_output, grad_output)
        b.expert_output = None

    def _send_dispatch_backward(self, layer: nn.Module, b: _Backward) -> None:
        # A rank that received no rows still takes part in the transfer, with an empty gradient.
        grad_received = b.received.grad if b.received.grad is not None else torch.zeros_like(b.received)
        grad_received = _conform_rows(grad_received, (b.received_buffer_rows, )).contiguous()
        grad_weights = None
        if b.received_weights is not None and b.received_weights.grad is not None:
            grad_weights = _conform_rows(b.received_weights.grad, (b.weights_buffer_rows, )).contiguous()
        b.received = b.received_weights = None
        b.sent = layer.mlp._deepep_exchange.combine_with_weight_grad_async(grad_received, b.handle, grad_weights)

    def _attention_backward(self, micro_batch: _MicroBatch, b: _Backward) -> None:
        grad_tokens, grad_scores, event = b.sent
        b.sent = None
        event.current_stream_wait()
        grad_tokens = _conform_rows(grad_tokens, b.tokens.shape)
        outputs, grads = [b.residual, b.tokens], [b.grad_out, grad_tokens]
        if grad_scores is not None:
            outputs.append(b.top_scores)
            grads.append(_conform_rows(grad_scores, b.top_scores.shape).reshape(b.top_scores.shape))
        torch.autograd.backward(outputs, grads)
        micro_batch.grad = b.x.grad
        b.residual = b.tokens = b.top_scores = b.x = None

    # -------------------------------------------------------------------------------------------------- helpers

    def _saved_input_slot(self, micro_batch: _MicroBatch, layer_index: int, like: torch.Tensor) -> torch.Tensor:
        """The slot holding ``micro_batch``'s input to layer ``layer_index`` (see ``saved_input_buffer``)."""
        num_layers = len(self.layers)
        slots = self._saved_input_slots
        if slots is None or slots[0].shape != like.shape or slots[0].dtype != like.dtype:
            storage = torch.empty((num_layers + 1, ) + tuple(like.shape), dtype=like.dtype,
                                  device=like.device).untyped_storage()
            # Separate tensors over one allocation, not views: views share one version counter, so writing one
            # slot would make autograd reject another slot's saved input as modified in place.
            numel = like.numel()
            slots = [
                torch.empty(0, dtype=like.dtype, device=like.device).set_(storage, index * numel, tuple(like.shape))
                for index in range(num_layers + 1)
            ]
            self._saved_input_slots = slots
        if micro_batch.index % 2 == 0:
            return slots[layer_index]
        return slots[num_layers - layer_index]

    def _beside_transfer(self, exchange):
        if not self.sm_carveout:
            return contextlib.nullcontext
        return lambda: sm_carveout(exchange.num_sms)

    def _limit_backward_lookahead(self) -> None:
        while len(self._backward_done) >= self._BACKWARD_LOOKAHEAD:
            self._backward_done.popleft().synchronize()

    def _record_backward_done(self) -> None:
        event = get_accelerator().Event()
        event.record()
        self._backward_done.append(event)


def _attention_and_router(layer: nn.Module, x: torch.Tensor, kwargs: dict):
    """The decoder layer up to the router: as Hugging Face's pre-norm layer, then AutoEP's router."""
    hidden = layer.input_layernorm(x)
    hidden, _ = layer.self_attn(hidden_states=hidden, **kwargs)
    residual = x + hidden
    normed = layer.post_attention_layernorm(residual)
    tokens = normed.reshape(-1, normed.shape[-1])
    moe = layer.mlp
    top_scores, selected_experts, counts = moe.router(tokens, moe.expert_bias)
    return residual, tokens, top_scores, selected_experts, counts


def _expert_counts(handle) -> torch.Tensor:
    prefix = handle.psum_num_recv_tokens_per_expert
    return torch.diff(prefix, prepend=prefix.new_zeros(1)).to(torch.int32)


def _run_experts(moe, received: torch.Tensor, received_weights: torch.Tensor | None, handle) -> torch.Tensor:
    """The expert MLP on the received rows, with the routing weights applied as AutoEPMoELayer._deepep_route does."""
    weights = None if received_weights is None else received_weights.reshape(-1, 1)
    if weights is not None and moe.score_apply == "pre":
        received = apply_deepep_row_weights(received, weights, moe.row_weighting_impl)
        weights = None
    output = moe.experts(received, _expert_counts(handle))
    if weights is not None:
        output = apply_deepep_row_weights(output, weights, moe.row_weighting_impl)
    return output
