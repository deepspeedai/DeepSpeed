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
        return self.max_cache_len, 0

    def get_seq_length(self) -> int:
        if not self.is_initialized:
            return 0
        if self._write_position is not None:
            return self._write_position + 1
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


class DeepSpeedStaticCache:
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
        self._layers = [DeepSpeedStaticLayer(max_cache_len) for _ in range(num_layers)]
        self._max_cache_len = max_cache_len
        self._write_position: torch.Tensor | None = None

        if dtype is not None and device is not None and batch_size > 0:
            num_heads = getattr(text_config, "num_key_value_heads", getattr(text_config, "num_attention_heads", 1))
            head_dim = getattr(text_config, "hidden_size", 1) // getattr(text_config, "num_attention_heads", 1)
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
        return self._layers[layer_idx].update(key_states, value_states, *args, **kwargs)

    def early_initialization(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device,
    ):
        for layer in self._layers:
            fake_k = torch.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype, device=device)
            fake_v = torch.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype, device=device)
            layer.lazy_initialization(fake_k, fake_v)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._layers):
            return 0
        return self._layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._layers):
            return self._max_cache_len
        return self._layers[layer_idx].get_max_cache_shape()

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        if layer_idx >= len(self._layers):
            return self._max_cache_len, 0
        return self._layers[layer_idx].get_mask_sizes(query_length)

    def reset(self):
        for layer in self._layers:
            layer.reset()

    def __len__(self):
        return len(self._layers)
