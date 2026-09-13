# Copyright (c) DeepSpeed Team
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""CUDA-graph-compatible static KV cache for hybrid engine rollout.

Derived from HuggingFace transformers ``StaticCache`` / ``StaticLayer``, but
with a critical difference: the write position is supplied externally via a
shared tensor instead of an internal ``cumulative_length`` counter.

Why this matters
----------------
Transformers' ``StaticLayer.update()`` maintains its own ``cumulative_length``
tensor that advances on every call.  During CUDA graph capture the captured
forward "freezes" this counter at whatever value it had at capture time.
On replay the counter does *not* advance, so subsequent KV writes go to the
wrong positions and the model silently produces incorrect logits.

Our ``DeepSpeedStaticCache`` instead reads the write position from a shared
tensor (``write_position``) that the caller updates in-place before each graph
replay.  Because ``write_position`` is a real tensor at a fixed address, CUDA
graph replays read the current value each time.

The caller (HybridEngineRollout) must call ``cache.set_write_position(pos)``
before each replay, where ``pos`` is a scalar ``torch.long`` tensor on the
correct device.
"""

import torch


class DeepSpeedStaticLayer:
    """A single layer's static KV cache whose write position is externally set.

    Parameters
    ----------
    max_cache_len : int
        Maximum number of tokens the cache can hold (last dim size).
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int):
        self.max_cache_len = max_cache_len
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.is_initialized = False
        self._write_position: torch.Tensor | None = None

    def set_write_position(self, pos: torch.Tensor):
        self._write_position = pos

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        max_batch_size, num_heads = key_states.shape[:2]
        self.max_batch_size = max_batch_size
        self.num_heads = num_heads
        self.k_head_dim = key_states.shape[-1]
        self.v_head_dim = value_states.shape[-1]

        self.keys = torch.zeros(
            (max_batch_size, num_heads, self.max_cache_len, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (max_batch_size, num_heads, self.max_cache_len, self.v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        torch._dynamo.mark_static_address(self.keys)
        torch._dynamo.mark_static_address(self.values)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        kv_length = key_states.shape[-2]

        if self._write_position is not None:
            cache_position = torch.arange(kv_length, device=self.device) + self._write_position
        else:
            cache_position = torch.arange(kv_length, device=self.device)

        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states

        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # Mirror HF StaticLayer exactly: static caches answer with the full
        # buffer width, which matches the full-width K/V that update()
        # returns (position-based sizes are the DynamicLayer formula).
        return self.max_cache_len, 0

    def get_seq_length(self):
        # The write position doubles as the cached-token count (HF
        # cumulative_length semantics): the next token is written AT this
        # index, so the count of already-cached tokens equals it. Returning
        # +1 here shifts every model-derived decode position by one and
        # mis-ropes the stored keys. Kept as a tensor expression: this is
        # read inside captured regions and must not sync.
        if not self.is_initialized:
            return 0
        if self._write_position is not None:
            return self._write_position
        return 0

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def reset(self) -> None:
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.is_initialized:
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.values.device))


class DSStaticGDNSlot:
    """Pass-through cache slot for hybrid models' linear-attention layers.

    HF's ``LinearAttentionLayer`` is already cudagraph-safe by construction:
    static shapes, ``mark_static_address`` buffers, and in-place ``copy_``
    state updates (its own comments say "to preserve the static address for
    cudagraphs"). Duplicating that state management here would only drift;
    the graph path therefore binds and reuses the prefill cache's slot
    object directly - zero copies, and decode-time updates land in the very
    buffers the captured graph reads."""

    is_compileable = True
    supports_early_init = False

    def __init__(self, max_cache_len: int):
        self._slot = None
        self.max_cache_len = max_cache_len
        self._write_position: torch.Tensor | None = None

    def bind(self, hf_slot) -> None:
        """Adopt a prefill cache's linear-attention slot (by reference)."""
        self._slot = hf_slot

    def __getattr__(self, name):
        # Forward the HF slot API (conv_states, recurrent_states,
        # has_previous_state, update_*, ...) so the model's GDN modules see
        # the exact object semantics they were written against.
        slot = object.__getattribute__(self, "_slot")
        if slot is None:
            raise AttributeError(f"DSStaticGDNSlot not bound: {name}")
        return getattr(slot, name)

    def set_write_position(self, pos: torch.Tensor):
        self._write_position = pos  # tracked for symmetry; GDN state is position-free

    def get_seq_length(self) -> int:
        # Linear-attention slots carry fixed-size state, not a sequence.
        return 0

    def get_max_cache_shape(self) -> int:
        return 0

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # Match the HF CacheLayer base formula for linear-attention slots:
        # they carry no sequence dimension, so kv_length is just the query
        # length (see transformers cache_utils.CacheLayer.get_mask_sizes).
        return query_length, 0

    def reset(self) -> None:
        if self._slot is not None:
            self._slot.reset()


class DeepSpeedStaticCache:
    # Mirrors HF StaticCache: static buffers answer mask builders with
    # full-width masks (is_compileable), which must match the full-width
    # K/V buffers update() returns; a position-truncated mask against a
    # full-width buffer crashes SDPA at decode.
    is_compileable = True
    """CUDA-graph-compatible static KV cache.

    Drop-in replacement for ``transformers.StaticCache`` in the graph-capture
    decode path of ``HybridEngineRollout``.  All layers share a single
    ``write_position`` tensor that the caller updates before each graph replay.

    Parameters
    ----------
    config : PreTrainedConfig
        HuggingFace model config (used to determine number of layers and head
        dimensions).
    batch_size : int
        Batch size for eager initialization.
    max_cache_len : int
        Maximum sequence length (prompt + generated tokens).
    device : torch.device | int | str | None
        Device for eager initialization.
    dtype : torch.dtype | None
        Dtype for eager initialization.
    """

    def __init__(
        self,
        config,
        batch_size: int = 1,
        max_cache_len: int = 4096,
        device=None,
        dtype=None,
    ):
        self.config = config
        text_config = getattr(config, "text_config", config)
        num_layers = getattr(text_config, "num_hidden_layers", 1)
        # Hybrid families (e.g. qwen3_5) mix full_attention and
        # linear_attention blocks behind one decoder-layer list; mirror the
        # split so KV slots and GDN pass-through slots line up by index.
        layer_types = getattr(text_config, "layer_types", None) or ["full_attention"] * num_layers
        self._layers = [
            DSStaticGDNSlot(max_cache_len) if t == "linear_attention" else DeepSpeedStaticLayer(max_cache_len)
            for t in layer_types
        ]
        self._max_cache_len = max_cache_len
        self._write_position: torch.Tensor | None = None

        if dtype is not None and device is not None and batch_size > 0:
            num_heads = getattr(text_config, "num_key_value_heads", getattr(text_config, "num_attention_heads", 1))
            # head_dim can be decoupled from hidden_size/heads (qwen3_5 family);
            # prefer the explicit config field when present.
            head_dim = getattr(text_config, "head_dim", 0) or (getattr(text_config, "hidden_size", 1) //
                                                               getattr(text_config, "num_attention_heads", 1))
            self.early_initialization(batch_size, num_heads, head_dim, dtype, device)

    @property
    def layers(self):
        return self._layers

    def set_write_position(self, pos: torch.Tensor):
        """Set the write position shared by all layers.

        Must be called before each graph replay with the decode step position
        as a scalar ``torch.long`` tensor on the correct device.  The tensor is
        stored by reference so subsequent in-place updates (e.g.
        ``pos.fill_(new_val)``) are immediately visible to all layers.
        """
        self._write_position = pos
        for layer in self._layers:
            layer.set_write_position(pos)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self._layers):
            raise IndexError(f"layer_idx {layer_idx} out of range (cache has {len(self._layers)} layers)")
        layer = self._layers[layer_idx]
        if isinstance(layer, DSStaticGDNSlot):
            raise TypeError("linear-attention slots manage conv/recurrent state, not key/value pairs")
        return layer.update(key_states, value_states, *args, **kwargs)

    def early_initialization(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device,
    ):
        for layer in self._layers:
            if isinstance(layer, DSStaticGDNSlot):
                continue  # bound to the prefill cache's HF slot later
            fake_k = torch.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype, device=device)
            fake_v = torch.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype, device=device)
            layer.lazy_initialization(fake_k, fake_v)

    def _first_attention_idx(self) -> int | None:
        # HF convention (cache_utils): alternating caches answer container
        # length queries with attention-slot semantics; a linear-attention
        # slot never tracks sequence length.
        for idx, layer in enumerate(self._layers):
            if not isinstance(layer, DSStaticGDNSlot):
                return idx
        return None

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._layers):
            return 0
        if isinstance(self._layers[layer_idx], DSStaticGDNSlot):
            first = self._first_attention_idx()
            if first is None:
                return 0
            return self._layers[first].get_seq_length()
        return self._layers[layer_idx].get_seq_length()

    def update_recurrent_state(self, recurrent_state, layer_idx: int = 0, state_idx: int = 0):
        return self._layers[layer_idx].update_recurrent_state(recurrent_state, state_idx=state_idx)

    def update_conv_state(self, conv_state, layer_idx: int = 0, **kwargs):
        return self._layers[layer_idx].update_conv_state(conv_state, state_idx=0, **kwargs)

    def has_previous_state(self, layer_idx: int = 0, state_idx: int = 0) -> bool:
        # transformers cache protocol: GDN blocks ask the container whether a
        # previous recurrent state exists before reading it.
        if layer_idx >= len(self._layers):
            return False
        layer = self._layers[layer_idx]
        if isinstance(layer, DSStaticGDNSlot):
            return layer.has_previous_state[state_idx]
        return self.get_seq_length(layer_idx) > 0

    def get_query_offset(self, layer_idx: int = 0) -> int:
        # transformers >= 5 cache protocol: the query offset equals the cached
        # sequence length for non-MTP layers (see HF cache_utils).
        return self.get_seq_length(layer_idx=layer_idx)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._layers):
            return self._max_cache_len
        return self._layers[layer_idx].get_max_cache_shape()

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        # HF convention: mask builders sample shared masks with the default
        # layer_idx, which on hybrid models may land on a linear-attention
        # slot; redirect such queries to the first attention slot so the
        # shared mask gets attention-sized spans.
        if layer_idx < len(self._layers) and isinstance(self._layers[layer_idx], DSStaticGDNSlot):
            first = self._first_attention_idx()
            if first is None:
                return query_length, 0
            return self._layers[first].get_mask_sizes(query_length)
        if layer_idx >= len(self._layers):
            return query_length, 0
        return self._layers[layer_idx].get_mask_sizes(query_length)

    def reset(self):
        for layer in self._layers:
            layer.reset()

    def __len__(self):
        return len(self._layers)
