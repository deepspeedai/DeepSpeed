# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engine interface.

The trainer talks to its rollout engine through three small dataclasses
(``RolloutRequest`` in / ``RolloutBatch`` out / ``SamplingConfig``) and one
ABC. This keeps engine-specific concerns out of the trainer loop.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class RolloutConfig:
    """Selects the rollout backend; each backend's config subclasses this with its own settings."""
    engine: str = "hybrid_engine"


@dataclass
class SamplingConfig:
    """Sampling knobs that the trainer passes to ``generate`` each step.

    ``top_k <= 0`` disables top-k filtering. ``repetition_penalty=None`` uses the
    value in the model's ``generation_config``; 1.0 disables the penalty.
    """

    max_new_tokens: int
    max_new_tokens_per_turn: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    n_samples_per_prompt: int = 1
    repetition_penalty: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.max_new_tokens_per_turn is not None and self.max_new_tokens_per_turn <= 0:
            raise ValueError("max_new_tokens_per_turn must be positive")
        if self.repetition_penalty is not None and self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")


@dataclass
class RolloutRecord:
    """One prompt and its rollout across all turns."""

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"

    group_index: int | None = None
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    attention_mask: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None
    status: Status = Status.PENDING
    # Turns the model has generated, each followed by the output of the tools it called, if any.
    num_turns: int = 0


@dataclass
class RolloutRequest:
    """Input to ``RolloutEngine.generate``."""

    samples: list[RolloutRecord]

    def __post_init__(self) -> None:
        for i, sample in enumerate(self.samples):
            if len(sample.attention_mask) != len(sample.tokens):
                raise ValueError(f"samples[{i}] attention_mask length {len(sample.attention_mask)} does not "
                                 f"match tokens length {len(sample.tokens)}")
            if sample.num_turns != 0 or sample.loss_mask is not None:
                raise ValueError(f"samples[{i}] already holds a rollout; pass the prompt only")


@dataclass
class RolloutBatch:
    """Output of ``RolloutEngine.generate``."""

    samples: list[RolloutRecord]

    def __post_init__(self) -> None:
        for i, sample in enumerate(self.samples):
            if len(sample.attention_mask) != len(sample.tokens):
                raise ValueError(f"samples[{i}] attention_mask length {len(sample.attention_mask)} does not "
                                 f"match tokens length {len(sample.tokens)}")

    @property
    def batch_size(self) -> int:
        return len(self.samples)

    @property
    def seq_len(self) -> int:
        return max(len(sample.tokens) for sample in self.samples)


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
