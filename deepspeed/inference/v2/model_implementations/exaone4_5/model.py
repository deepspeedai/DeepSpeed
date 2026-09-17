# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
from typing import Any

import torch

from deepspeed.accelerator import get_accelerator

from ...config_v2 import RaggedInferenceEngineConfig
from ...inference_utils import DtypeEnum
from ...modules import heuristics
from ...modules.configs import DSSelfAttentionConfig, PositionalEmbeddingType, RotateHalfConfig
from ...ragged import DSSequenceDescriptor, RaggedBatchWrapper
from ..exaone4.model import Exaone4InferenceModel


def _get_rope_parameters(config: Any) -> dict:
    rope_parameters = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
    if not isinstance(rope_parameters, dict):
        raise ValueError("EXAONE 4.5 requires a RoPE parameter dictionary")

    # transformers v5 may normalize per-layer RoPE settings into nested dictionaries.
    sliding_parameters = rope_parameters.get("sliding_attention")
    if isinstance(sliding_parameters, dict):
        rope_parameters = sliding_parameters

    return rope_parameters


def _llama3_inverse_frequencies(config: Any, device: Any) -> torch.Tensor:
    rope_parameters = _get_rope_parameters(config)
    rope_type = rope_parameters.get("rope_type", rope_parameters.get("type"))
    if rope_type != "llama3":
        raise ValueError(f"EXAONE 4.5 requires llama3 RoPE scaling, got {rope_type!r}")

    theta = rope_parameters.get("rope_theta", getattr(config, "rope_theta", None))
    required_parameters = ("factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings")
    missing_parameters = [name for name in required_parameters if name not in rope_parameters]
    if theta is None or missing_parameters:
        missing_parameters += ["rope_theta"] if theta is None else []
        raise ValueError(f"EXAONE 4.5 is missing RoPE parameters: {', '.join(missing_parameters)}")

    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0))
    rotary_dim = int(head_dim * partial_rotary_factor)
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError(f"Rotary dimension must be a positive even number, got {rotary_dim}")

    freq_indices = torch.arange(0, rotary_dim, 2, dtype=torch.int64, device=device).float()
    inv_freq = 1.0 / (float(theta)**(freq_indices / rotary_dim))

    factor = float(rope_parameters["factor"])
    low_freq_factor = float(rope_parameters["low_freq_factor"])
    high_freq_factor = float(rope_parameters["high_freq_factor"])
    original_context = float(rope_parameters["original_max_position_embeddings"])

    low_freq_wavelen = original_context / low_freq_factor
    high_freq_wavelen = original_context / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    scaled_inv_freq = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (original_context / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * scaled_inv_freq / factor + smooth_factor * scaled_inv_freq
    is_medium_freq = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, scaled_inv_freq)


def _supported_sequence_length(config: Any) -> int:
    max_sequence_length = int(config.max_position_embeddings)
    if "sliding_attention" not in getattr(config, "layer_types", ()):
        return max_sequence_length

    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError("EXAONE 4.5 sliding attention requires sliding_window")

    # Dense blocked attention does not yet implement a local mask. At or below the
    # window size, its causal mask is equivalent to the checkpoint's local mask.
    return min(max_sequence_length, int(sliding_window))


class Exaone4_5InferenceModel(Exaone4InferenceModel):
    """
    Inference model for the language-model (text) portion of EXAONE 4.5.

    The post-norm, QK-Norm, and parameter layout are inherited from EXAONE 4.0.
    EXAONE 4.5 additionally uses hybrid attention: sliding layers apply scaled
    Llama 3 RoPE, while full-attention layers use NoPE. Separate attention
    modules preserve that per-layer behavior.

    DeepSpeed's dense blocked attention kernel does not currently implement a
    local attention mask, so sequence length is capped at ``sliding_window``.
    Within that bound, dense causal attention and sliding attention are
    equivalent. The vision tower and MTP head remain intentionally unloaded.
    """

    @property
    def max_sequence_length(self) -> int:
        return _supported_sequence_length(self._config)

    @property
    def positional_embedding_config(self) -> RotateHalfConfig:
        return RotateHalfConfig(use_trained_freqs=True)

    @property
    def activation_dtype(self) -> DtypeEnum:
        dtype = getattr(self._config, "torch_dtype", None)
        if dtype is None:
            dtype = getattr(self._config, "dtype", None)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace("torch.", ""), dtype)
        if dtype == torch.float16:
            return DtypeEnum.fp16
        elif dtype == torch.bfloat16:
            return DtypeEnum.bf16
        else:
            raise NotImplementedError("Only fp16 and bf16 are supported")

    def __init__(self, config: Any, engine_config: RaggedInferenceEngineConfig, base_mp_group: Any) -> None:
        layer_types = tuple(getattr(config, "layer_types", ()))
        if len(layer_types) != config.num_hidden_layers:
            raise ValueError(
                f"Expected one EXAONE 4.5 layer type per layer, got {len(layer_types)} for {config.num_hidden_layers}")
        unsupported_layer_types = set(layer_types) - {"sliding_attention", "full_attention"}
        if unsupported_layer_types:
            raise ValueError(f"Unsupported EXAONE 4.5 layer types: {sorted(unsupported_layer_types)}")

        super().__init__(config=config, engine_config=engine_config, base_mp_group=base_mp_group)

        self._layer_types = layer_types
        self._global_attn = self._build_global_attention()

        rope_inv_freqs = _llama3_inverse_frequencies(config, get_accelerator().current_device())
        self.register_buffer("_rope_inv_freqs", rope_inv_freqs.float(), persistent=False)

    def _build_global_attention(self):
        attn_config = DSSelfAttentionConfig(
            max_tokens=self._engine_config.state_manager.max_ragged_batch_size,
            n_heads_q=self.n_heads_q_local,
            n_heads_kv=self.n_heads_kv_local,
            head_size=self.head_size,
            max_sequences=self._engine_config.state_manager.max_ragged_sequence_count,
            scale_factor=1.0 / (self.head_size**0.5),
            input_dtype=self.activation_dtype,
            output_dtype=self.activation_dtype,
            positional_embedding_type=PositionalEmbeddingType.none,
        )
        return heuristics.instantiate_attention(attn_config, self._engine_config)

    def prepare_batch(self, wrapped_batch: RaggedBatchWrapper) -> None:
        super().prepare_batch(wrapped_batch)
        self._global_attn.build_atoms(wrapped_batch)

    def get_kv_requirements(self, sequence: DSSequenceDescriptor, max_new_tokens: int, max_new_blocks: int):
        remaining_tokens = max(self.max_sequence_length - sequence.seen_tokens, 0)
        return super().get_kv_requirements(sequence, min(max_new_tokens, remaining_tokens), max_new_blocks)

    def maybe_allocate_kv(self, sequence: DSSequenceDescriptor, n_new_tokens: int) -> None:
        requested_length = sequence.seen_tokens + n_new_tokens
        if requested_length > self.max_sequence_length:
            raise ValueError(
                f"EXAONE 4.5 Inference V2 supports at most {self.max_sequence_length} tokens, got {requested_length}")
        super().maybe_allocate_kv(sequence, n_new_tokens)

    def _forward_attention(self, layer_idx: int, qkv: torch.Tensor, kv_cache: torch.Tensor,
                           ragged_batch_info: RaggedBatchWrapper) -> torch.Tensor:
        if self._layer_types[layer_idx] == "sliding_attention":
            return self.attn(qkv, kv_cache, ragged_batch_info, self._rope_inv_freqs)
        return self._global_attn(qkv, kv_cache, ragged_batch_info)
