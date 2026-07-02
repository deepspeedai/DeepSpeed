# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Abstract base class for RLHF trainers.

Concrete implementations (e.g. :class:`~deepspeed.runtime.rlhf.trainer.opsd.OPSDTrainer`)
inherit from :class:`RLHFTrainer` and implement the abstract methods to define
their algorithm-specific training loop.
"""

from abc import ABC, abstractmethod
from typing import Any


class RLHFTrainer(ABC):
    """Base class for all RLHF training loops.

    Subclasses must implement :meth:`train` and :meth:`_train_step`.  The base
    class deliberately imposes no constraints on the constructor signature so
    each algorithm can accept whatever components it needs (rollout engine,
    reference model, reward model, etc.).
    """

    @abstractmethod
    def train(self) -> None:
        """Run the full training loop (all epochs / steps)."""
        ...

    @abstractmethod
    def _train_step(self, batch: Any) -> dict:
        """Execute one optimizer step and return a metrics dict.

        Args:
            batch: A single batch from the dataloader.  The expected structure
                is algorithm-specific.

        Returns:
            A ``dict`` of scalar metrics (``loss``, timing fields, token
            counts, …) suitable for logging.
        """
        ...
