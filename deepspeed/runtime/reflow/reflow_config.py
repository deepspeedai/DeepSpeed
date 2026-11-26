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
    The reservation shrinks if needed to leave a worker core; a single available core is shared.
    Worker placement uses this even when ``enable_cpu_affinity`` is false."""

    main_thread_core_type: Literal["logical", "physical"] = "logical"
    """Count main-thread reservations as logical CPUs or physical cores including their SMT siblings."""

    pin_main_thread: bool = True
    """Pin the initializing thread to the reserved main CPUs and restore its affinity on destroy."""

    worker_core_type: Literal["logical", "physical"] = "logical"
    """Use all worker CPU IDs, or one logical CPU per worker physical core."""

    bucketwise_worker_affinity: Literal["task", "thread"] = "thread"
    """Choose a CPU mask per bucketwise task, or reuse a fixed bucketwise mask on each pool thread."""

    bucketwise_cores_per_worker: Optional[int] = Field(None, ge=1)
    """By default one optimizer worker uses the rank's worker CPUs. Set a limit to split
    each NUMA-local CPU set among multiple workers."""

    h2d_chunk_size_mib: int = Field(16, ge=1)
    """Maximum MiB per host-to-device transfer of Ready parameters."""

    state_update_cores: Optional[int] = Field(None, ge=1)
    """Limit worker CPUs used by the background state commit during forward; ``None`` uses all worker CPUs."""

    state_update_backward_cores: Optional[int] = Field(None, ge=1)
    """Limit the state commit's backward CPU mask; ``None`` uses all worker CPUs."""
