# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from pydantic import Field
from typing import Optional

from deepspeed.runtime.config_utils import DeepSpeedConfigModel


class ReflowConfig(DeepSpeedConfigModel):
    """Configuration options for the Reflow asynchronous CPU-offload optimizer (ZeRO stage 3)."""

    num_threads: Optional[int] = Field(None, ge=1)
    """Threads for the CPU optimizer update. ``None`` uses every core available to the rank."""

    enable_cpu_affinity: bool = False
    """Pin the main (forward/backward) thread and the optimizer workers to separate CPU cores."""

    main_thread_cores: int = Field(3, ge=1)
    """Cores reserved for the main thread; the rest go to the optimizer workers.
    Only used with ``enable_cpu_affinity``."""

    bucketwise_cores_per_worker: int = Field(1, ge=1)
    """Cores per bucketwise optimizer worker. The worker count follows from the cores available
    to the rank divided by this value."""

    state_update_cores: int = Field(2, ge=1)
    """Worker cores the background optimizer-state commit may use while it overlaps the next
    forward. Every busy core lowers the turbo frequency the CPU allows, which slows the
    launch-bound forward; once backward starts the commit uses every worker core."""
