# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""WeightBridge ABC: per-tensor TP slicing for vLLM weight sync."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, Iterator, Tuple

import torch


class ParallelKind(str, Enum):
    """How a single parameter is split across vLLM TP ranks.

    Notation matches the standard Megatron-style decomposition:

      * ``COLUMN`` — output dim (dim 0) is split. Each rank owns
        ``out_features / tp`` rows. Used for attention Q/K/V and MLP
        gate/up.
      * ``ROW`` — input dim (dim 1) is split. Each rank owns
        ``in_features / tp`` columns. Used for attention output projection
        and MLP down projection.
      * ``VOCAB`` — like COLUMN but applied to the embedding / LM head where
        the partitioned dim is the vocab axis. Treated the same as COLUMN
        for slicing purposes; the kind is kept distinct to make divisibility
        diagnostics clearer at debug time.
      * ``REPLICATED`` — the same tensor lives on every rank
        (layer norms, RMSNorm scalars, per-head q_norm/k_norm in Qwen3).
    """

    COLUMN = "column"
    ROW = "row"
    VOCAB = "vocab"
    REPLICATED = "replicated"


def _even_slice(t: torch.Tensor, dim: int, rank: int, tp_size: int) -> torch.Tensor:
    """Return rank ``rank`` 's contiguous chunk of ``t`` along ``dim``.

    Refuses uneven divisions so that bugs surface here rather than as silent
    layout mismatches once weights are loaded into vLLM.
    """
    total = int(t.shape[dim])
    if total % tp_size != 0:
        raise ValueError(f"Shape {tuple(t.shape)} dim {dim} (={total}) not divisible by "
                         f"tp_size {tp_size}")
    per = total // tp_size
    return t.narrow(dim, rank * per, per).contiguous()


class WeightBridge(ABC):
    """Strategy object that maps HuggingFace param names to a parallel kind.

    Subclasses only need to implement :meth:`parallel_kind`; the slicing
    machinery is inherited.
    """

    # Subclasses set this to a human-readable tag, e.g. "qwen2".
    arch: str = "base"

    @abstractmethod
    def parallel_kind(self, hf_name: str) -> ParallelKind:
        """Return how parameter ``hf_name`` should be partitioned across TP."""

    def slice_for_rank(
        self,
        hf_name: str,
        tensor: torch.Tensor,
        tp_rank: int,
        tp_size: int,
    ) -> torch.Tensor:
        """Return the slice of ``tensor`` that belongs to rank ``tp_rank``."""
        if tp_size < 1 or not (0 <= tp_rank < tp_size):
            raise ValueError(f"invalid tp_rank={tp_rank} for tp_size={tp_size}")
        if tp_size == 1:
            return tensor
        kind = self.parallel_kind(hf_name)
        if kind is ParallelKind.REPLICATED:
            return tensor
        # COLUMN and VOCAB partition dim 0 (output / vocab). ROW partitions
        # dim 1 (input). Both kinds may apply to 1-D tensors (biases): for a
        # 1-D bias on a COLUMN-parallel linear, dim 0 IS the partitioned dim.
        if kind in (ParallelKind.COLUMN, ParallelKind.VOCAB):
            return _even_slice(tensor, dim=0, rank=tp_rank, tp_size=tp_size)
        if kind is ParallelKind.ROW:
            if tensor.dim() < 2:
                # Row-parallel linears have a replicated bias (vLLM convention),
                # so a 1-D tensor reaching this branch is a bug.
                raise ValueError(f"ROW parallel kind requires >=2-D tensor for {hf_name}; "
                                 f"got shape {tuple(tensor.shape)}")
            return _even_slice(tensor, dim=1, rank=tp_rank, tp_size=tp_size)
        raise ValueError(f"unhandled parallel kind {kind!r}")

    def map_state_dict(
        self,
        hf_named_tensors: Iterable[Tuple[str, torch.Tensor]],
        tp_rank: int,
        tp_size: int,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Yield ``(vllm_name, sliced_tensor)`` for every input pair.

        For Qwen-family models the vLLM parameter name is identical to the
        HF name (vLLM's loader handles QKV/gate-up fusion internally), so the
        emitted names are unchanged.
        """
        for name, tensor in hf_named_tensors:
            yield name, self.slice_for_rank(name, tensor, tp_rank, tp_size)
