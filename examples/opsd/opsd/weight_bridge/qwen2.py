# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Weight bridge for Qwen2 / Qwen2.5 dense models.

Naming follows the standard HF Qwen2 layout::

    model.embed_tokens.weight
    model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}
    model.layers.{i}.mlp.{gate,up,down}_proj.weight
    model.layers.{i}.{input,post_attention}_layernorm.weight
    model.norm.weight
    lm_head.weight     # may be tied to embed_tokens

Parallel kinds:
    * Q/K/V projections — column-parallel (split heads across ranks)
    * Attention output projection — row-parallel
    * MLP gate / up projections — column-parallel
    * MLP down projection — row-parallel
    * Layer norms / final norm — replicated
    * Token embedding & LM head — vocab-parallel (split vocab dim)
    * Bias on Q/K/V — column-parallel (1-D bias on a column-parallel linear)
    * Bias on o_proj / down_proj — replicated (row-parallel linears have a
      replicated bias under vLLM's convention; the partial sums are reduced
      before the bias add)
"""

import re

from opsd.weight_bridge.base import ParallelKind, WeightBridge

_LAYER_RE = re.compile(r"^model\.layers\.\d+\.(?P<rest>.+)$")


class Qwen2WeightBridge(WeightBridge):
    arch = "qwen2"

    # Suffix → parallel kind. Keyed by the part after "model.layers.{i}." for
    # transformer-block params, plus a few full names for embeddings / norms.
    _LAYER_RULES = {
        "self_attn.q_proj.weight": ParallelKind.COLUMN,
        "self_attn.k_proj.weight": ParallelKind.COLUMN,
        "self_attn.v_proj.weight": ParallelKind.COLUMN,
        "self_attn.q_proj.bias": ParallelKind.COLUMN,
        "self_attn.k_proj.bias": ParallelKind.COLUMN,
        "self_attn.v_proj.bias": ParallelKind.COLUMN,
        "self_attn.o_proj.weight": ParallelKind.ROW,
        "self_attn.o_proj.bias": ParallelKind.REPLICATED,
        "mlp.gate_proj.weight": ParallelKind.COLUMN,
        "mlp.up_proj.weight": ParallelKind.COLUMN,
        "mlp.down_proj.weight": ParallelKind.ROW,
        "mlp.down_proj.bias": ParallelKind.REPLICATED,
        "input_layernorm.weight": ParallelKind.REPLICATED,
        "post_attention_layernorm.weight": ParallelKind.REPLICATED,
    }

    _GLOBAL_RULES = {
        "model.embed_tokens.weight": ParallelKind.VOCAB,
        "model.norm.weight": ParallelKind.REPLICATED,
        "lm_head.weight": ParallelKind.VOCAB,
    }

    def parallel_kind(self, hf_name: str) -> ParallelKind:
        if hf_name in self._GLOBAL_RULES:
            return self._GLOBAL_RULES[hf_name]
        m = _LAYER_RE.match(hf_name)
        if m is not None:
            rest = m.group("rest")
            if rest in self._LAYER_RULES:
                return self._LAYER_RULES[rest]
            # Per-layer name not in our table — surface a clear error so the
            # weight sync isn't silently wrong for an unrecognised tensor.
            extra = self._extra_layer_kind(rest)
            if extra is not None:
                return extra
            raise KeyError(f"Unknown per-layer Qwen2 parameter suffix {rest!r}; add a rule "
                           f"in Qwen2WeightBridge._LAYER_RULES")
        raise KeyError(f"Unknown Qwen2 parameter name {hf_name!r}")

    def _extra_layer_kind(self, _suffix: str):  # noqa: D401, ARG002
        """Hook for subclasses (Qwen3) to add per-layer rules without
        duplicating the rest of the table."""
        return None
