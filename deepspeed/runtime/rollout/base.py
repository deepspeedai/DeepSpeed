# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engine interface.

The trainer talks to its rollout engine through three small dataclasses
(``RolloutRequest`` in / ``RolloutBatch`` out / ``SamplingConfig``) and one
ABC. This keeps engine-specific concerns out of the trainer loop.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class RolloutConfig:
    """Configuration for the rollout engine."""
    engine: str = "hybrid_engine"

    # Use CUDA graph capture for decode acceleration.
    use_graph_capture: bool = False


@dataclass
class SamplingConfig:
    """Sampling knobs that the trainer passes to ``generate`` each step."""

    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    n_samples_per_prompt: int = 1


@dataclass
class RolloutRequest:
    """Input to ``RolloutEngine.generate``.

    Prompts arrive *left-padded* (i.e. real tokens at the right edge) so that
    causal generation appends naturally after them.
    """

    prompt_ids: torch.Tensor  # [B, T_p] left-padded with pad_token_id
    prompt_attention_mask: torch.Tensor  # [B, T_p], 1 on real prompt tokens

    def __post_init__(self) -> None:
        if self.prompt_ids.dim() != 2:
            raise ValueError(f"prompt_ids must be 2-D [B, T_p]; got {tuple(self.prompt_ids.shape)}")
        if self.prompt_attention_mask.shape != self.prompt_ids.shape:
            raise ValueError(f"prompt_attention_mask shape {tuple(self.prompt_attention_mask.shape)} "
                             f"does not match prompt_ids {tuple(self.prompt_ids.shape)}")


@dataclass
class RolloutBatch:
    """Output of ``RolloutEngine.generate``.

    ``input_ids`` holds the *concatenation* of (left-padded) prompt and
    response, right-padded to the longest sequence in the batch.
    """

    input_ids: torch.Tensor  # [B', T_p + T_r]; B' = B * n_samples_per_prompt
    attention_mask: torch.Tensor  # [B', T_p + T_r]
    response_start_idx: torch.Tensor  # [B'] int

    def __post_init__(self) -> None:
        if self.input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2-D; got {tuple(self.input_ids.shape)}")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError(f"attention_mask shape {tuple(self.attention_mask.shape)} does not "
                             f"match input_ids {tuple(self.input_ids.shape)}")
        B = self.input_ids.shape[0]
        if self.response_start_idx.shape != (B, ):
            raise ValueError(f"response_start_idx must be 1-D of length {B}; got "
                             f"{tuple(self.response_start_idx.shape)}")

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])


class RolloutEngine(ABC):
    """Abstract base for rollout engines."""

    name: str = "base"

    @abstractmethod
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        """Run generation, return prompt+response in one tensor."""

    @abstractmethod
    def sync_weights(self, step: int) -> None:
        """Push updated weights into the rollout backend.

        No-op when the rollout engine is co-located with the training engine
        (e.g. hybrid engine shares weights directly).
        """

    def shutdown(self) -> None:
        """Release any backend resources. Default no-op."""
        return None
