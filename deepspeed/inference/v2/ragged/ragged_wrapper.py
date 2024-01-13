# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import RaggedUtilsBuilder

from .sequence_descriptor import DSSequenceDescriptor
from .manager_configs import DSStateManagerConfig


def to_padded(original_size: int) -> int:
    """
    Pad to a backend friendly granularity.
    """

    def _pad_to_mul_of_pow2(val: int, pow_2_val: int) -> int:
        return val + (pow_2_val - 1) & ~(pow_2_val - 1)

    # TODO(cmikeh2): Tune this approach. This is mainly a placeholder right now.
    granularity = 64 if original_size <= 512 else 128

    return _pad_to_mul_of_pow2(original_size, granularity)


class RaggedBatchWrapper:
    """
    Container for all the auxiliary Tensors used in the management of a ragged batch.

    For each Tensor, we maintain a shadow Tensor on the host. This Tensor is what is
    directly populated when constructing the ragged batch. The shadow Tensors, when possible,
    should be allocated so as to support fast host-to-accelerator copies.
    """

    # Tensors to populate the ragged batch into.
    _input_ids_shadow: torch.Tensor
    _input_ids: torch.Tensor
    """
    Forward pass input buffer.
    """

    _batch_metadata_storage: torch.Tensor
    _batch_metadata_storage_shadow: torch.Tensor
    """
    Holds the number of inflight sequences and tokens for the ragged batch.
    """

    _token_to_seq_storage: torch.Tensor
    _token_to_seq_storage_shadow: torch.Tensor
    """
    Linear mapping for each of the tokens. Let's say we have 8 tokens in the batch,
    with the sequence breakdown being [4, 1, 3]. Then, the mapping would be:
    [0, 0, 0, 0, 1, 2, 2, 2]
    """

    _inflight_seq_descriptors: torch.Tensor
    _inflight_seq_descriptors_shadow: torch.Tensor
    """
    For each sequence in the batch, we store the start token in the batch, the number of tokens
    the number of tokens in the history of this sequence, and an unused 4th reserved for alignment.
    For the above example this would give:
    [[0, 4, H0, X], [4, 1, H1, X], [5, 3, H2, X]]
    """

    # Holds the block ids for each sequence in the ragged batch.
    _kv_ptrs: torch.Tensor
    _kv_ptrs_shadow: torch.Tensor
    """
    List of ptrs pointing to the GPU buffer that holds the KV-block ids for each sequence.
    If there are multiple allocation groups associated with each of the sequences, then
    then accessing the Nth cache will require accessing the Nth block id
    """

    def __init__(self, config: DSStateManagerConfig) -> None:
        """
        Convenience wrapper around the data structures used to represent a ragged
        batch for inference. Only a single `RaggedBatchWrapper` should be used per
        ragged inference engine.

        The underlying data structures are implemented in `ragged_batch_descriptor.h`.
        """
        self._config = config
        self._input_ids = torch.zeros((self._config.max_ragged_batch_size),
                                      dtype=torch.int64,
                                      device=get_accelerator().current_device())

        self._batch_metadata_storage = torch.zeros(2, dtype=torch.int32, device=get_accelerator().current_device())

        self._token_to_seq_storage = torch.zeros((self._config.max_ragged_batch_size),
                                                 dtype=torch.int32,
                                                 device=get_accelerator().current_device())
        self._inflight_seq_descriptors = torch.zeros((self._config.max_ragged_sequence_count, 4),
                                                     dtype=torch.int32,
                                                     device=get_accelerator().current_device())
        self._kv_ptrs = torch.zeros((self._config.max_ragged_sequence_count),
                                    dtype=torch.int64,
                                    device=get_accelerator().current_device())

        self._utils_module = RaggedUtilsBuilder().load()
        host_alloc = self._utils_module.allocate_fast_host_buffer

        self._input_ids_shadow = host_alloc(self._input_ids)
        self._batch_metadata_storage_shadow = host_alloc(self._batch_metadata_storage)
        self._token_to_seq_storage_shadow = host_alloc(self._token_to_seq_storage)
        self._inflight_seq_descriptors_shadow = host_alloc(self._inflight_seq_descriptors)
        self._kv_ptrs_shadow = host_alloc(self._kv_ptrs)

        # Default behavior should be no padding
        self._is_padded = False

        self._current_tokens = 0
        self._current_sequences = 0
        self._batch_tokens = []
        self._inflight_seq_descriptors_shadow_buf = []
        self._kv_blocks_ptr_buf = []
        self._token_to_seq_storage_shadow_buf = []

    def clear(self) -> None:
        """
        Clear the ragged batch. This will reset the number of tokens and sequences to 0.
        """
        self._current_tokens = 0
        self._current_sequences = 0
        self._batch_tokens = []
        self._inflight_seq_descriptors_shadow_buf = []
        self._kv_blocks_ptr_buf = []
        self._token_to_seq_storage_shadow_buf = []

    def insert_sequence(self, seq_descriptor: DSSequenceDescriptor, tokens: torch.Tensor, do_checks=True) -> None:
        """
        Incrementally insert a sequence into the ragged batch. This will update the
        metadata for the ragged batch and the sequence.

        Arguments:
            seq_descriptor ()
        """
        if tokens.device != torch.device("cpu"):
            # This doesn't really fall under schedulability, so we'll unconditionally check for it.
            raise RuntimeError(f"Expected tokens to be on host but found device '{tokens.device}'")

        if do_checks and self.current_sequences == self._config.max_ragged_sequence_count:
            raise RuntimeError(f"Ragged batch is full due to sequence limit: {self._config.max_ragged_sequence_count}")

        seq_tokens = tokens.numel()

        if do_checks and self.current_tokens + seq_tokens > self._config.max_ragged_batch_size:
            raise RuntimeError(f"Ragged batch is full due to capacity limit: {self._config.max_ragged_batch_size})")

        # The values in _inflight_seq_descriptors_shadow_buf, _token_to_seq_storage_shadow_buf, _kv_blocks_ptr_buf, etc.,
        # are ultimately stored in PyTorch tensors: _inflight_seq_descriptors_shadow, _token_to_seq_storage_shadow, _kv_ptrs_shadow, etc.
        # However, we found it inefficient to iterate over and substitute values into tensor slices or to use copy/fill calls for this purpose.
        # Therefore, we initially store the values in Python lists or primitive data types and then copy them collectively in the finalize() method,
        # instead of updating the tensors directly in each iteration.
        self._batch_tokens.append(tokens)
        self._inflight_seq_descriptors_shadow_buf.append(self.current_tokens)
        self._inflight_seq_descriptors_shadow_buf.append(seq_tokens)
        self._inflight_seq_descriptors_shadow_buf.append(seq_descriptor.seen_tokens)
        self._inflight_seq_descriptors_shadow_buf.append(0)  # alignment

        self._token_to_seq_storage_shadow_buf.extend([self.current_sequences] * seq_tokens)

        self._kv_blocks_ptr_buf.append(seq_descriptor.kv_blocks_ptr)

        self._current_tokens += seq_tokens
        self._current_sequences += 1

    @property
    def tensor_toks(self) -> torch.Tensor:
        """
        The number of tokens in the in-flight ragged batch. This will not trigger
        synchronization with the device.
        """
        cur_toks = self.current_tokens
        if self._is_padded:
            return to_padded(cur_toks)
        else:
            return cur_toks

    def finalize(self, padding: Optional[bool] = False) -> None:
        """
        Completes construction of the ragged batch by flushing the host buffers to the device.
        """
        cur_toks = self.current_tokens

        # Batch-copy the values recorded in insert_sequence() into PyTorch tensors to enhance efficiency.
        self._inflight_seq_descriptors_shadow.flatten()[:len(self._inflight_seq_descriptors_shadow_buf)].copy_(
            torch.tensor(self._inflight_seq_descriptors_shadow_buf))
        self._input_ids_shadow[:self.current_tokens].copy_(torch.cat(self._batch_tokens, dim=0))
        self._token_to_seq_storage_shadow[:len(self._token_to_seq_storage_shadow_buf)].copy_(
            torch.tensor(self._token_to_seq_storage_shadow_buf))
        self._kv_ptrs_shadow[:len(self._kv_blocks_ptr_buf)].copy_(torch.tensor(self._kv_blocks_ptr_buf))
        self._batch_metadata_storage_shadow.copy_(torch.tensor([cur_toks, self.current_sequences]))

        if padding:
            padded_toks = to_padded(cur_toks)
            self._input_ids_shadow[cur_toks:padded_toks].fill_(-1)
            self._token_to_seq_storage_shadow[cur_toks:padded_toks].fill_(-1)
            self._is_padded = True
        else:
            padded_toks = cur_toks
            self._is_padded = False

        current_sequences = self.current_sequences

        def _noblock_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
            dst.copy_(src, non_blocking=True)

        _noblock_copy(self._input_ids[:padded_toks], self._input_ids_shadow[:padded_toks])
        _noblock_copy(self._batch_metadata_storage, self._batch_metadata_storage_shadow)
        _noblock_copy(self._token_to_seq_storage[:padded_toks], self._token_to_seq_storage_shadow[:padded_toks])
        _noblock_copy(self._inflight_seq_descriptors[:current_sequences],
                      self._inflight_seq_descriptors_shadow[:current_sequences])
        _noblock_copy(self._kv_ptrs[:current_sequences], self._kv_ptrs_shadow[:current_sequences])

    def input_ids(self, on_device: bool = True) -> torch.Tensor:
        """
        The input ids tensor for the ragged batch. If the device Tensor is requested, the Tensor
        is truncated to the number of tokens in the batch.
        """
        if on_device:
            return self._input_ids[:self.tensor_toks]
        else:
            return self._input_ids_shadow

    def batch_metadata_buffer(self, on_device: bool = True) -> torch.Tensor:
        """
        Buffer associated with the batch metadata tensor that can
        be populated in preparation for passing a new input to the device.
        """
        if on_device:
            return self._batch_metadata_storage
        else:
            return self._batch_metadata_storage_shadow

    def tokens_to_seq(self, on_device: bool = True) -> torch.Tensor:
        """
        Mapping of token to which sequence it belongs to in the ragged batch. If the device Tensor
        is requested, the Tensor is truncated to the number of tokens in the batch.
        """
        if on_device:
            return self._token_to_seq_storage[:self.tensor_toks]
        else:
            return self._token_to_seq_storage_shadow

    def inflight_seq_descriptors(self, on_device: bool = True) -> torch.Tensor:
        """
        Buffer associated with the metadata of each sequence in the ragged batch. If the device Tensor
        is requested, the Tensor is truncated to the number of sequences in the batch.
        """
        if on_device:
            return self._inflight_seq_descriptors[:self.current_sequences]
        else:
            return self._inflight_seq_descriptors_shadow

    def kv_ptrs(self, on_device: bool = True) -> torch.Tensor:
        """
        Pointer to where the list of KV ids associated with a sequence are. If the device Tensor
        is requested, the Tensor is truncated to the number of sequences in the batch.
        """
        if on_device:
            return self._kv_ptrs[:self.current_sequences]
        else:
            return self._kv_ptrs_shadow

    def masks(self, on_device: bool = True) -> Optional[torch.Tensor]:
        """
        Placeholder for supporting complex masks. Currently not supported.

        Models that will need this will be BERT-like, not generative.
        """
        return None

    @property
    def current_tokens(self) -> int:
        """
        The number of tokens in the in-flight ragged batch. This will not trigger
        synchronization with the device.
        """
        return self._current_tokens

    @property
    def current_sequences(self) -> int:
        """
        The number of sequences in the in-flight ragged batch. This will not trigger
        synchronization with the device.
        """
        return self._current_sequences
