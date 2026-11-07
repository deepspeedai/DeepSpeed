# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from pydantic import Field
from typing import Literal, Optional

from deepspeed.runtime.config_utils import DeepSpeedConfigModel


class ReflowConfig(DeepSpeedConfigModel):
    """Configuration options for the Reflow asynchronous CPU-offload optimizer (ZeRO stage 3)."""

    num_threads: Optional[int] = Field(None, ge=1)
    """Threads for the CPU optimizer update. ``None`` uses every core available to the rank."""

    enable_cpu_affinity: bool = False
    """Restrict the main process to its NUMA-local cores. Optimizer workers are always pinned."""

    main_thread_cores: int = Field(3, ge=1)
    """Cores reserved for the main thread; the rest go to the optimizer workers.
    Worker placement uses this even when ``enable_cpu_affinity`` is false."""

    main_thread_core_type: Literal["logical", "physical"] = "logical"
    """Count main-thread reservations as logical CPUs or physical cores including their SMT siblings."""

    pin_main_thread: bool = False
    """Pin the initializing thread to the reserved main CPUs and restore its affinity on destroy."""

    worker_core_type: Literal["logical", "physical"] = "logical"
    """Use all worker CPU IDs, or one logical CPU per worker physical core."""

    bucketwise_worker_affinity: Literal["task", "thread"] = "task"
    """Choose a CPU mask for each task, or keep each pool thread on its initial mask."""

    bucketwise_cores_per_worker: int = Field(8, ge=1)
    """Cores per bucketwise optimizer worker. The worker count follows from the cores available
    to the rank divided by this value."""

    state_update_cores: int = Field(2, ge=1)
    """Worker cores the background optimizer-state commit may use while it overlaps the next
    forward. Every busy core lowers the turbo frequency the CPU allows, which slows the
    launch-bound forward; once backward starts the commit uses every worker core."""

    state_update_backward_cores: Optional[int] = Field(None, ge=1)
    """Limit the state commit's backward CPU mask; ``None`` uses all worker CPUs."""
