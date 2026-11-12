# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""A tensor-parallel decoder for rollout whose weights live in a :class:`WeightArena`.

Covers the Llama family as HuggingFace configures it, with Qwen3's per-head q/k RMSNorm: fused qkv and gate-up
projections, column- and row-parallel linears (one all-reduce after the attention output and one after the MLP),
and a vocabulary-parallel embedding and LM head.

Its loader (:meth:`TPDecoder.load_weights`) takes HuggingFace names and writes every slice with a plain ``copy_``
into the arena, so the layout is a set of pieces (:meth:`TPDecoder.layout_pieces`) and shard-to-shard plans can
write the arena directly. Nothing the forward reads is ever reallocated: the arena, the
static KV cache and the rotary tables keep their addresses for the engine's lifetime.

Prompts are left-padded and share cache slots: a batch's prompt fills slots ``0 .. T-1`` (:meth:`TPDecoder.prefill`)
and decode step ``n`` writes slot ``T + n - 1`` for every row (:meth:`TPDecoder.decode`), each row attending from its
first real token on.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine import AffinePiece
from deepspeed.runtime.rollout.native.arena import WeightArena

NamedTensors = Iterable[Tuple[str, torch.Tensor]]

try:  # decode attention over the KV cache; without it, a masked SDPA over the whole cache (slow, but exact)
    from flash_attn import flash_attn_with_kvcache as _flash_attn_with_kvcache
except ImportError:
    _flash_attn_with_kvcache = None


@dataclass(frozen=True)
class DecoderSpec:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    qk_norm: bool

    @classmethod
    def from_hf_config(cls, cfg) -> "DecoderSpec":
        if getattr(cfg, "attention_bias", False) or getattr(cfg, "mlp_bias", False):
            raise NotImplementedError("biased projections")
        # transformers 5 keeps the rotary settings in rope_parameters (rope_scaling aliases it); 4.x has rope_theta.
        rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise NotImplementedError(f"rope_type {rope.get('rope_type', rope.get('type'))!r}")
        rope_theta = rope.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))
        if getattr(cfg, "hidden_act", "silu") != "silu":
            raise NotImplementedError(f"hidden_act {cfg.hidden_act!r}")
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return cls(vocab_size=cfg.vocab_size,
                   hidden_size=cfg.hidden_size,
                   intermediate_size=cfg.intermediate_size,
                   num_layers=cfg.num_hidden_layers,
                   num_heads=cfg.num_attention_heads,
                   num_kv_heads=cfg.num_key_value_heads,
                   head_dim=head_dim,
                   rms_norm_eps=cfg.rms_norm_eps,
                   rope_theta=rope_theta,
                   tie_word_embeddings=getattr(cfg, "tie_word_embeddings", False),
                   qk_norm=cfg.model_type == "qwen3")


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # In fp32, as HuggingFace's RMSNorm does, then back to the activation dtype before the weight.
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


_COMPILED = {}


def _compiled(fn):
    # Once per process and on the plain function: compiling a decoder's bound methods would keep every decoder ever
    # built (and its arena) alive in the compile cache.
    if fn not in _COMPILED:
        _COMPILED[fn] = torch.compile(fn)
    return _COMPILED[fn]


class TPDecoder:
    """One tensor-parallel rank of the decoder.

    Args:
        spec: model shape.
        group: the tensor-parallel process group (``None``: a single rank, no communication).
        max_batch_size, max_len: shape of the static KV cache (prompt plus generated tokens).
        tp_rank, tp_size: this rank's position without a group, e.g. to plan another layout on ``device="meta"``.
        fuse: compile the elementwise chains (RMSNorm, rotary, SiLU-and-multiply) into one kernel each with
            ``torch.compile``.
    """

    def __init__(self,
                 spec: DecoderSpec,
                 group=None,
                 dtype: torch.dtype = torch.bfloat16,
                 device=None,
                 max_batch_size: int = 1,
                 max_len: int = 1024,
                 tp_rank: Optional[int] = None,
                 tp_size: Optional[int] = None,
                 fuse: bool = False):
        self.spec = spec
        self.group = group
        if tp_size is None:
            tp_size = dist.get_world_size(group) if group is not None else 1
            tp_rank = dist.get_rank(group) if group is not None else 0
        self.tp_size, self.tp_rank = tp_size, tp_rank
        tp = self.tp_size
        for what, n in (("num_heads", spec.num_heads), ("num_kv_heads", spec.num_kv_heads), ("intermediate_size",
                                                                                             spec.intermediate_size)):
            if n % tp:
                raise ValueError(f"{what}={n} is not divisible by the tensor-parallel size {tp}")
        self.heads = spec.num_heads // tp
        self.kv_heads = spec.num_kv_heads // tp
        self.inter = spec.intermediate_size // tp
        self.vocab_per_rank = -(-spec.vocab_size // tp)
        self.vocab_lo = self.tp_rank * self.vocab_per_rank
        self.vocab_hi = min(self.vocab_lo + self.vocab_per_rank, spec.vocab_size)
        self.dtype = dtype
        self.device = torch.device(device if device is not None else get_accelerator().current_device_name())
        self.arena = WeightArena(self._layout(), dtype, self.device)

        d = spec.head_dim
        self.max_batch_size, self.max_len = max_batch_size, max_len
        # [layer, k/v, batch, position, kv head, head dim]: the layout flash-attn's KV-cache kernel takes
        self.kv_cache = torch.zeros(spec.num_layers,
                                    2,
                                    max_batch_size,
                                    max_len,
                                    self.kv_heads,
                                    d,
                                    dtype=dtype,
                                    device=self.device)
        inv_freq = 1.0 / (spec.rope_theta**(torch.arange(0, d, 2, dtype=torch.float32, device=self.device) / d))
        freqs = torch.outer(torch.arange(max_len, dtype=torch.float32, device=self.device), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # In the activation dtype, as HuggingFace applies them.
        self.rope_cos, self.rope_sin = emb.cos().to(dtype), emb.sin().to(dtype)
        # Unfused, these chains launch a dozen small kernels per layer: about a third of a small-batch decode step.
        ops = (rms_norm, rotate, silu_mul)
        if fuse:
            ops = tuple(_compiled(op) for op in ops)
        self._rms_norm_op, self._rope, self._silu_mul = ops

    def _layout(self) -> Dict[str, Tuple[int, ...]]:
        s, d = self.spec, self.spec.head_dim
        layout = {"embed_tokens": (self.vocab_per_rank, s.hidden_size)}
        for i in range(s.num_layers):
            p = f"layers.{i}."
            layout[p + "input_layernorm"] = (s.hidden_size, )
            layout[p + "qkv_proj"] = ((self.heads + 2 * self.kv_heads) * d, s.hidden_size)
            if s.qk_norm:
                layout[p + "q_norm"] = (d, )
                layout[p + "k_norm"] = (d, )
            layout[p + "o_proj"] = (s.hidden_size, self.heads * d)
            layout[p + "post_attention_layernorm"] = (s.hidden_size, )
            layout[p + "gate_up_proj"] = (2 * self.inter, s.hidden_size)
            layout[p + "down_proj"] = (s.hidden_size, self.inter)
        layout["norm"] = (s.hidden_size, )
        if not s.tie_word_embeddings:
            layout["lm_head"] = (self.vocab_per_rank, s.hidden_size)
        return layout

    # ------------------------------------------------------------------ loading

    @torch.no_grad()
    def load_weights(self, named_tensors: NamedTensors) -> None:
        """Copy HuggingFace-named full weights into this rank's slices of the arena."""
        for name, tensor in named_tensors:
            dest, src = self._slice(name, tensor)
            if dest is not None:
                dest.copy_(src)

    def layout_pieces(self, shapes: Dict[str, Sequence[int]]) -> Dict[str, List[Tuple[str, AffinePiece]]]:
        """Where :meth:`load_weights` writes each weight, as pieces into ``"arena"``, without loading anything.

        Computed on meta tensors, so a layout can be planned before any rank holds the full weights.
        """
        out = {}
        for name, shape in shapes.items():
            dest, src = self._slice(name, torch.empty(shape, device="meta"))
            if dest is None:
                continue
            piece = AffinePiece(tuple(dest.shape),
                                src.storage_offset(),
                                src.stride(),
                                dest.storage_offset(),
                                dest.stride(),
                                locations=[0])
            out[name] = [("arena", piece)]
        return out

    def _slice(self, name: str, w: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """(arena view, view of ``w``) for one HuggingFace weight; ``(None, None)`` for weights the decoder ignores."""
        a, d, r = self.arena, self.spec.head_dim, self.tp_rank
        if name == "model.embed_tokens.weight":
            n = self.vocab_hi - self.vocab_lo
            return a["embed_tokens"][:n], w[self.vocab_lo:self.vocab_hi]
        if name == "lm_head.weight":
            if self.spec.tie_word_embeddings:
                return None, None
            n = self.vocab_hi - self.vocab_lo
            return a["lm_head"][:n], w[self.vocab_lo:self.vocab_hi]
        if name == "model.norm.weight":
            return a["norm"], w
        if not name.startswith("model.layers."):
            raise KeyError(f"unexpected weight {name!r}")
        i, rest = name[len("model.layers."):].split(".", 1)
        p = f"layers.{i}."
        q_rows, kv_rows = self.heads * d, self.kv_heads * d
        if rest == "self_attn.q_proj.weight":
            return a[p + "qkv_proj"][:q_rows], w[r * q_rows:(r + 1) * q_rows]
        if rest == "self_attn.k_proj.weight":
            return a[p + "qkv_proj"][q_rows:q_rows + kv_rows], w[r * kv_rows:(r + 1) * kv_rows]
        if rest == "self_attn.v_proj.weight":
            return a[p + "qkv_proj"][q_rows + kv_rows:], w[r * kv_rows:(r + 1) * kv_rows]
        if rest == "self_attn.o_proj.weight":
            return a[p + "o_proj"], w[:, r * q_rows:(r + 1) * q_rows]
        if rest == "mlp.gate_proj.weight":
            return a[p + "gate_up_proj"][:self.inter], w[r * self.inter:(r + 1) * self.inter]
        if rest == "mlp.up_proj.weight":
            return a[p + "gate_up_proj"][self.inter:], w[r * self.inter:(r + 1) * self.inter]
        if rest == "mlp.down_proj.weight":
            return a[p + "down_proj"], w[:, r * self.inter:(r + 1) * self.inter]
        replicated = {
            "input_layernorm.weight": "input_layernorm",
            "post_attention_layernorm.weight": "post_attention_layernorm",
            "self_attn.q_norm.weight": "q_norm",
            "self_attn.k_norm.weight": "k_norm",
        }
        if rest in replicated and p + replicated[rest] in a:
            return a[p + replicated[rest]], w
        raise KeyError(f"unexpected weight {name!r}")

    # ------------------------------------------------------------------ forward

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self._rms_norm_op(x, weight, self.spec.rms_norm_eps)

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            dist.all_reduce(x, group=self.group)
        return x

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.embedding(tokens, self.arena["embed_tokens"])
        local = tokens - self.vocab_lo
        outside = (local < 0) | (local >= self.vocab_per_rank)
        x = F.embedding(local.masked_fill(outside, 0), self.arena["embed_tokens"])
        x = x.masked_fill(outside.unsqueeze(-1), 0)
        return self._all_reduce(x)

    def prefill(self, tokens: torch.Tensor, positions: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Hidden states (after the final norm) of a batch of prompts, whose keys and values fill cache slots
        ``0 .. T-1``.

        Args:
            tokens: ``[B, T]`` token ids (``B`` up to ``max_batch_size``), left-padded.
            positions: ``[B, T]`` rotary positions.
            attn_mask: ``[B, 1, T, T]`` bool, True where a query may attend to a key.
        """
        B, T = tokens.shape

        def attend(i, q, k, v):
            self.kv_cache[i, 0, :B, :T].copy_(k)
            self.kv_cache[i, 1, :B, :T].copy_(v)
            out = F.scaled_dot_product_attention(q.transpose(1, 2),
                                                 k.transpose(1, 2),
                                                 v.transpose(1, 2),
                                                 attn_mask=attn_mask,
                                                 enable_gqa=True)
            return out.transpose(1, 2)

        return self._layers(tokens, positions, attend)

    def decode(self, tokens: torch.Tensor, slot: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """Hidden states of one new token per row, written to cache slot ``slot`` (the same for every row).

        Args:
            tokens: ``[B, 1]`` token ids.
            slot: ``[1]`` cache slot of the new tokens; row ``b`` attends to slots ``pad[b] .. slot``.
            pad: ``[B]`` left padding of each row.
        """
        B = tokens.shape[0]
        positions = (slot - pad).view(B, 1)
        if _flash_attn_with_kvcache is not None and self.device.type == "cuda" and self.dtype != torch.float32:
            # Appends the new keys and values in place and attends over each row's [pad, slot] without expanding
            # the KV heads or materializing a mask.
            seqlens = slot.to(torch.int32).expand(B).contiguous()
            leftpad = pad.to(torch.int32)

            def attend(i, q, k, v):
                return _flash_attn_with_kvcache(q,
                                                self.kv_cache[i, 0, :B],
                                                self.kv_cache[i, 1, :B],
                                                k=k,
                                                v=v,
                                                cache_seqlens=seqlens,
                                                cache_leftpad=leftpad,
                                                causal=True)
        else:
            cache_pos = torch.arange(self.max_len, device=tokens.device)
            mask = ((cache_pos >= pad.view(B, 1)) & (cache_pos <= slot)).view(B, 1, 1, -1)

            def attend(i, q, k, v):
                keys, values = self.kv_cache[i, 0, :B], self.kv_cache[i, 1, :B]
                keys.index_copy_(1, slot, k)
                values.index_copy_(1, slot, v)
                out = F.scaled_dot_product_attention(q.transpose(1, 2),
                                                     keys.transpose(1, 2),
                                                     values.transpose(1, 2),
                                                     attn_mask=mask,
                                                     enable_gqa=True)
                return out.transpose(1, 2)

        return self._layers(tokens, positions, attend)

    def _layers(self, tokens: torch.Tensor, positions: torch.Tensor, attend) -> torch.Tensor:
        """The decoder stack; ``attend(layer, q, k, v)`` (``[B, T, heads, d]``, rotary applied) does attention and
        the KV cache."""
        s, a = self.spec, self.arena
        B, T = tokens.shape
        d = s.head_dim
        cos = self.rope_cos[positions].unsqueeze(2)  # [B, T, 1, d]
        sin = self.rope_sin[positions].unsqueeze(2)
        x = self._embed(tokens)
        for i in range(s.num_layers):
            p = f"layers.{i}."
            h = self._rms_norm(x, a[p + "input_layernorm"])
            qkv = F.linear(h, a[p + "qkv_proj"])
            q, k, v = qkv.split([self.heads * d, self.kv_heads * d, self.kv_heads * d], dim=-1)
            q = q.view(B, T, self.heads, d)
            k = k.view(B, T, self.kv_heads, d)
            if s.qk_norm:
                q = self._rms_norm(q, a[p + "q_norm"])
                k = self._rms_norm(k, a[p + "k_norm"])
            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)
            v = v.view(B, T, self.kv_heads, d)
            attn = attend(i, q, k, v).reshape(B, T, self.heads * d)
            x = x + self._all_reduce(F.linear(attn, a[p + "o_proj"]))
            h = self._rms_norm(x, a[p + "post_attention_layernorm"])
            act = self._silu_mul(F.linear(h, a[p + "gate_up_proj"]))
            x = x + self._all_reduce(F.linear(act, a[p + "down_proj"]))
        return self._rms_norm(x, a["norm"])

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary fp32 logits of ``hidden`` (``[..., hidden_size]``), identical on every rank."""
        head = self.arena["embed_tokens"] if self.spec.tie_word_embeddings else self.arena["lm_head"]
        local = F.linear(hidden, head).float()
        if self.tp_size > 1:
            # Concatenated along dim 0 (the stacked [tp, ...] output is NCCL-only), then one slice per rank.
            local = local.contiguous()
            parts = torch.empty((self.tp_size * local.shape[0], ) + local.shape[1:],
                                dtype=local.dtype,
                                device=local.device)
            dist.all_gather_into_tensor(parts, local, group=self.group)
            local = torch.cat(list(parts.view((self.tp_size, ) + local.shape).unbind(0)), dim=-1)
        return local[..., :self.spec.vocab_size]
