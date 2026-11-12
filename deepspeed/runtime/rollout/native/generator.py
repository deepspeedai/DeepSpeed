# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Batch generation with a :class:`TPDecoder`: an eager prefill, then one CUDA graph per decode step.

The decode graph is captured once per batch size and replayed for every step of every later generation. Its inputs
(the previous token, the cache slot, the rows' padding, the temperature) are tensors at fixed addresses that the
loop writes before each replay; the weights, the KV cache and the rotary tables it reads are the decoder's, which
are never reallocated. A weight update that writes the arena in place therefore needs no re-capture. Sampling
(temperature and Gumbel-max) runs inside the graph, so a step is one replay with no host synchronization.

Every tensor-parallel rank computes the same full logits, so ranks seeded alike sample the same tokens.

With ``compact=True`` a generation with EOS tokens drops finished rows as it goes: at each periodic EOS check, once at
most half of the batch is still running, the running rows' KV cache and padding move to the front and decoding
continues with the graph of the next power-of-two batch size (captured on first use), so a long tail of a few rows
no longer costs full-batch steps. Rows are sampled independently, so results differ from an uncompacted run only in
the random draws (and, under greedy decoding, in the GEMMs' rounding at another batch size). Ranks decide alike: the
decision depends only on the tokens.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.native.decoder import TPDecoder


def prefill_inputs(pad: torch.Tensor, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotary positions ``[B, T]`` and attention mask ``[B, 1, T, T]`` of left-padded prompts with ``pad`` padding
    tokens per row. A padding query attends to itself only, so its softmax stays finite."""
    B = pad.shape[0]
    i = torch.arange(T, device=pad.device)
    positions = (i.view(1, T) - pad.view(B, 1)).clamp(min=0)
    key, query = i.view(1, 1, T), i.view(1, T, 1)
    mask = ((key >= pad.view(B, 1, 1)) & (key <= query)) | (key == query)
    return positions, mask.view(B, 1, T, T)


@dataclass
class _DecodeStep:
    tokens: torch.Tensor  # [B, 1] previous token
    slot: torch.Tensor  # [1] cache slot it is written to
    pad: torch.Tensor  # [B] left padding of each row
    temperature: torch.Tensor  # [] fp32; <= 0 is greedy
    next_tokens: Optional[torch.Tensor] = None  # [B]
    next_logprobs: Optional[torch.Tensor] = None  # [B]
    graph: Optional[object] = None  # the accelerator's graph


class NativeGenerator:
    """Generates with a decoder; ``use_graph=False`` runs the same decode step eagerly (always so on CPU)."""

    def __init__(self, decoder: TPDecoder, use_graph: bool = True, compact: bool = False, min_batch: int = 8):
        self.decoder = decoder
        # A graph captures device work only: on CPU, capture and replay would do nothing.
        self.use_graph = use_graph and decoder.device.type != "cpu"
        self.compact = compact
        self.min_batch = min_batch
        self._steps: Dict[int, _DecodeStep] = {}
        self.captures = 0
        self.compactions = 0  # over this generator's lifetime

    def invalidate(self) -> None:
        """Drop the captured graphs (needed only when the decoder's tensors or its process group change)."""
        self._steps.clear()

    # ------------------------------------------------------------------ sampling

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        greedy = temperature <= 0
        scale = torch.where(greedy, torch.ones_like(temperature), temperature)
        logprobs = torch.log_softmax(logits / scale, dim=-1)
        uniform = torch.rand_like(logprobs).clamp_(min=1e-20)
        gumbel = -torch.log(-torch.log(uniform))
        noise = torch.where(greedy, torch.zeros_like(gumbel), gumbel)
        tokens = torch.argmax(logprobs + noise, dim=-1)
        return tokens, logprobs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    # ------------------------------------------------------------------ decode step

    def _decode_body(self, s: _DecodeStep) -> None:
        hidden = self.decoder.decode(s.tokens, s.slot, s.pad)
        s.next_tokens, s.next_logprobs = self._sample(self.decoder.logits(hidden[:, -1]), s.temperature)

    def _step_for(self, B: int) -> _DecodeStep:
        step = self._steps.get(B)
        if step is not None:
            return step
        dev = self.decoder.device
        step = _DecodeStep(tokens=torch.zeros(B, 1, dtype=torch.long, device=dev),
                           slot=torch.full((1, ), self.decoder.max_len - 1, dtype=torch.long, device=dev),
                           pad=torch.zeros(B, dtype=torch.long, device=dev),
                           temperature=torch.ones((), dtype=torch.float32, device=dev))
        if self.use_graph:
            # Warm up on a side stream (cuBLAS workspaces, NCCL communicators), then capture. The warm-up writes the
            # KV cache only at the last slot, which a step always overwrites before reading it.
            acc = get_accelerator()
            side = acc.Stream(device=dev)
            side.wait_stream(acc.current_stream(dev))
            with acc.stream(side):
                for _ in range(2):
                    self._decode_body(step)
            acc.current_stream(dev).wait_stream(side)
            step.graph = acc.create_graph()
            with acc.capture_to_graph(step.graph):
                self._decode_body(step)
            self.captures += 1
        self._steps[B] = step
        return step

    def prepare(self, batch_size: int) -> None:
        """Capture the decode graphs a generation of ``batch_size`` rows can use (with ``compact``, also the smaller
        power-of-two sizes it may continue at), so that no capture or compilation lands inside a later generation."""
        sizes = [batch_size]
        if self.compact:
            size = 1 << max(batch_size - 1, 0).bit_length()
            while size // 2 >= self.min_batch:
                size //= 2
                if size < batch_size:
                    sizes.append(size)
        for size in sizes:
            self._step_for(size)

    # ------------------------------------------------------------------ compaction

    def _compact(self, step: _DecodeStep, rows: Optional[torch.Tensor], done: torch.Tensor, B: int,
                 used: int) -> Tuple[_DecodeStep, Optional[torch.Tensor]]:
        """Move the running rows to the front of the KV cache and continue at a smaller batch size when at most half
        of the current batch still runs. ``done``: ``[B]`` per output row; ``used``: cache slots written so far."""
        cur = step.tokens.shape[0]
        positions = torch.arange(cur, device=done.device) if rows is None else rows
        # batch positions whose row still runs (padding positions map to the spare row B, never running)
        running = torch.cat([~done, done.new_zeros(1)])[positions].nonzero().flatten()
        k = int(running.numel())
        size = max(self.min_batch, 1 << max(k - 1, 0).bit_length())
        if k == 0 or size > cur // 2:
            return step, rows
        kv = self.decoder.kv_cache
        for layer in kv:  # one layer at a time: the gather's temporary is a layer's running rows, not the whole cache
            layer[:, :k, :used] = layer[:, running, :used]
        nxt = self._step_for(size)
        nxt.pad[:k] = step.pad[running]
        nxt.pad[k:] = step.pad[running[0]]  # any valid row's padding: the padding positions' outputs are dropped
        nxt.temperature.copy_(step.temperature)
        new_rows = torch.full((size, ), B, dtype=torch.long, device=done.device)
        new_rows[:k] = positions[running]
        self.compactions += 1
        return nxt, new_rows

    # ------------------------------------------------------------------ generation

    @torch.no_grad()
    def generate(self,
                 prompt_ids: torch.Tensor,
                 prompt_mask: torch.Tensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 eos_token_ids: Sequence[int] = (),
                 seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample up to ``max_new_tokens`` per left-padded prompt.

        Returns ``(tokens [B, N], logprobs [B, N], lengths [B])``: each row's tokens up to and including its first
        EOS (``lengths``), with whatever follows it unspecified.
        """
        dec = self.decoder
        B, T = prompt_ids.shape
        if B > dec.max_batch_size or T + max_new_tokens - 1 > dec.max_len:
            raise ValueError(f"batch {B} x {T}+{max_new_tokens} exceeds the cache ({dec.max_batch_size} x "
                             f"{dec.max_len})")
        step = self._step_for(B)
        if seed is not None:
            get_accelerator().manual_seed(seed)
        dev = dec.device
        prompt_ids, prompt_mask = prompt_ids.to(dev), prompt_mask.to(dev)
        pad = T - prompt_mask.long().sum(1)
        temp = torch.tensor(float(temperature), device=dev)

        positions, mask = prefill_inputs(pad, T)
        hidden = dec.prefill(prompt_ids, positions, mask)
        tokens, logprobs = self._sample(dec.logits(hidden[:, -1]), temp)

        # one spare row (index B) takes the outputs of the padding positions of a compacted batch
        out_tokens = torch.empty(B + 1, max_new_tokens, dtype=torch.long, device=dev)
        out_logprobs = torch.empty(B + 1, max_new_tokens, dtype=torch.float32, device=dev)
        out_tokens[:B, 0], out_logprobs[:B, 0] = tokens, logprobs
        out_tokens[B].zero_()  # the spare row feeds padding positions: keep its ids valid
        eos = torch.tensor(list(eos_token_ids), dtype=torch.long, device=dev)
        step.pad.copy_(pad)
        step.temperature.copy_(temp)
        rows = None  # batch position -> row of the outputs, once compacted (None: the identity)
        for n in range(1, max_new_tokens):
            if rows is None:
                step.tokens.copy_(out_tokens[:B, n - 1:n])
            else:
                step.tokens.copy_(out_tokens[rows, n - 1:n])
            step.slot.fill_(T + n - 1)
            if step.graph is not None:
                get_accelerator().replay_graph(step.graph)
            else:
                self._decode_body(step)
            if rows is None:
                out_tokens[:B, n], out_logprobs[:B, n] = step.next_tokens, step.next_logprobs
            else:
                out_tokens[rows, n], out_logprobs[rows, n] = step.next_tokens, step.next_logprobs
            # A host check now and then: every rank sees the same tokens, so all stop at the same step.
            if eos.numel() and n % 32 == 0:
                done = torch.isin(out_tokens[:B, :n + 1], eos).any(1)
                if bool(done.all()):
                    out_tokens, out_logprobs = out_tokens[:, :n + 1], out_logprobs[:, :n + 1]
                    break
                if self.compact:
                    step, rows = self._compact(step, rows, done, B, T + n)
        out_tokens, out_logprobs = out_tokens[:B], out_logprobs[:B]
        lengths = torch.full((B, ), out_tokens.shape[1], dtype=torch.long, device=dev)
        if eos.numel():
            hit = torch.isin(out_tokens, eos)
            first = torch.where(hit.any(1), hit.float().argmax(1) + 1, lengths)
            lengths = first
        return out_tokens, out_logprobs, lengths
