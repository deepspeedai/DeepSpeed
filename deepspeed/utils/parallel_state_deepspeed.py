# SPDX-License-Identifier: Apache-2.0
# Copyright (c) DeepSpeed Team

# DeepSpeed Team

# The file has been adapted from https://github.com/NVIDIA/Megatron-LM and retains the following license from the original file

# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DeepSpeed Compatibility Layer for parallel_state.

This module provides module-level functions compatible with DeepSpeed's
groups.py API, allowing code written for DeepSpeed to work with the
refactored parallel_state module.

Key Features:
- Supports multiple parallel state instances (for RL scenarios with different models)
- Backward compatible with single global instance
- Context manager for switching between different parallel configurations

Usage:
    # Basic usage (single global instance):
    from parallel_state_deepspeed import get_data_parallel_group
    dp_group = get_data_parallel_group()

    # Multi-instance usage (for RL scenarios):
    from parallel_state_deepspeed import (
        get_parallel_state_instance,
        set_current_parallel_state,
        get_data_parallel_group,
    )

    # Create different instances for different models
    actor_state = get_parallel_state_instance("actor")
    critic_state = get_parallel_state_instance("critic")

    # Initialize with different DP sizes
    actor_state.initialize_model_parallel(tensor_model_parallel_size=2, data_parallel_size=4)
    critic_state.initialize_model_parallel(tensor_model_parallel_size=1, data_parallel_size=8)

    # Use context manager to switch
    with set_current_parallel_state("actor"):
        actor_dp_group = get_data_parallel_group()  # Uses actor's DP group

    with set_current_parallel_state("critic"):
        critic_dp_group = get_data_parallel_group()  # Uses critic's DP group
"""

from contextlib import contextmanager
from typing import Optional
from parallel_state import ParallelState, get_parallel_state as _get_default_parallel_state

# Registry for multiple parallel state instances
_parallel_state_registry = {}
_default_instance_name = "__default__"

# Current active instance name (thread-local would be better, but using global for simplicity)
_current_instance_name = _default_instance_name


def get_parallel_state_instance(name: Optional[str] = None) -> ParallelState:
    """Get or create a named ParallelState instance.

    Args:
        name: Name of the instance. If None, returns the default global instance.
              Use different names for different models in RL scenarios.

    Returns:
        ParallelState instance

    Example:
        # For RL with actor and critic models
        actor_state = get_parallel_state_instance("actor")
        critic_state = get_parallel_state_instance("critic")
    """
    if name is None:
        return _get_default_parallel_state()

    if name not in _parallel_state_registry:
        _parallel_state_registry[name] = ParallelState()

    return _parallel_state_registry[name]


def set_current_parallel_state(name: Optional[str] = None):
    """Set the current active parallel state instance.

    Args:
        name: Name of the instance to activate. If None, uses the default instance.

    Returns:
        Context manager for temporarily switching the active instance

    Example:
        with set_current_parallel_state("actor"):
            dp_group = get_data_parallel_group()  # Uses actor's DP group
    """

    @contextmanager
    def _context():
        global _current_instance_name
        old_name = _current_instance_name
        _current_instance_name = name if name is not None else _default_instance_name
        try:
            yield
        finally:
            _current_instance_name = old_name

    return _context()


def get_current_parallel_state() -> ParallelState:
    """Get the currently active parallel state instance.

    Returns:
        The currently active ParallelState instance
    """
    return get_parallel_state_instance(_current_instance_name)


def get_parallel_state(name: Optional[str] = None) -> ParallelState:
    """Get parallel state instance (backward compatible).

    If name is provided, returns the named instance.
    Otherwise, returns the currently active instance.

    Args:
        name: Optional name of the instance. If None, returns current active instance.

    Returns:
        ParallelState instance
    """
    if name is not None:
        return get_parallel_state_instance(name)
    return get_current_parallel_state()


# ============================================================================
# Core Tensor/Model/Data Parallel Functions
# ============================================================================


def get_tensor_model_parallel_group(name: Optional[str] = None):
    """Get the tensor model parallel group the caller rank belongs to.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.
              Use this in RL scenarios to specify which model's parallel groups to use.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_tensor_model_parallel_group()


def get_model_parallel_group(name: Optional[str] = None):
    """Get the model parallel group the caller rank belongs to.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_model_parallel_group()


def get_data_parallel_group(name: Optional[str] = None,
                            with_context_parallel: bool = False,
                            partial_data_parallel: bool = False):
    """Get the data parallel group the caller rank belongs to.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.
              Use this in RL scenarios to specify which model's DP group to use.
              For example, "actor" vs "critic" may have different DP sizes.
        with_context_parallel: Whether to include context parallel in the group.
        partial_data_parallel: Whether to use partial data parallel group.

    DeepSpeed-compatible interface.

    Example:
        # In RL scenario with different DP sizes:
        actor_dp = get_data_parallel_group("actor")  # Actor's DP group
        critic_dp = get_data_parallel_group("critic")  # Critic's DP group

        # Or use context manager:
        with set_current_parallel_state("actor"):
            dp_group = get_data_parallel_group()  # Uses actor's DP group
    """
    return get_parallel_state(name).get_data_parallel_group(with_context_parallel=with_context_parallel,
                                                            partial_data_parallel=partial_data_parallel)


def get_tensor_model_parallel_world_size(name: Optional[str] = None):
    """Return world size for the tensor model parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_tensor_model_parallel_world_size()


def get_model_parallel_world_size(name: Optional[str] = None):
    """Return world size for the model parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_tensor_model_parallel_world_size()


def get_tensor_model_parallel_rank(name: Optional[str] = None):
    """Return caller's rank for the tensor-model-parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_tensor_model_parallel_rank()


def get_model_parallel_rank(name: Optional[str] = None):
    """Return caller's rank for the model parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_tensor_model_parallel_rank()


def get_data_parallel_world_size(name: Optional[str] = None,
                                 with_context_parallel: bool = False,
                                 partial_data_parallel: bool = False):
    """Return world size for the data parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.
        with_context_parallel: Whether to include context parallel.
        partial_data_parallel: Whether to use partial data parallel.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_data_parallel_world_size(with_context_parallel=with_context_parallel,
                                                                 partial_data_parallel=partial_data_parallel)


def get_data_parallel_rank(name: Optional[str] = None,
                           with_context_parallel: bool = False,
                           partial_data_parallel: bool = False):
    """Return caller's rank in the data-parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.
        with_context_parallel: Whether to include context parallel.
        partial_data_parallel: Whether to use partial data parallel.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_data_parallel_rank(with_context_parallel=with_context_parallel,
                                                           partial_data_parallel=partial_data_parallel)


def get_tensor_model_parallel_src_rank(name: Optional[str] = None):
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    import torch.distributed as dist
    global_rank = dist.get_rank()
    local_world_size = get_tensor_model_parallel_world_size(name)
    return (global_rank // local_world_size) * local_world_size


def set_tensor_model_parallel_world_size(world_size, name: Optional[str] = None):
    """Set the tensor model parallel size.

    Args:
        world_size: World size to set.
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    ps = get_parallel_state(name)
    ps.mpu_tensor_model_parallel_world_size = world_size


def set_tensor_model_parallel_rank(rank, name: Optional[str] = None):
    """Set tensor model parallel rank.

    Args:
        rank: Rank to set.
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    ps = get_parallel_state(name)
    ps.mpu_tensor_model_parallel_rank = rank


# ============================================================================
# Pipeline Parallel Functions
# ============================================================================


def get_pipeline_model_parallel_group(name: Optional[str] = None):
    """Get the pipeline-model-parallel group the caller rank belongs to.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_pipeline_model_parallel_group()


def get_pipeline_model_parallel_world_size(name: Optional[str] = None):
    """Return world size for the pipeline-model-parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_pipeline_model_parallel_world_size()


def get_pipeline_model_parallel_rank(name: Optional[str] = None):
    """Return caller's rank for the pipeline-model-parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_pipeline_model_parallel_rank()


# ============================================================================
# Context Parallel Functions
# ============================================================================


def get_context_parallel_group(name: Optional[str] = None):
    """Get the context-parallel group the caller rank belongs to.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_context_parallel_group()


def get_context_parallel_world_size(name: Optional[str] = None):
    """Return world size for the context parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_context_parallel_world_size()


def get_context_parallel_rank(name: Optional[str] = None):
    """Return caller's rank in the context-parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_context_parallel_rank()


# ============================================================================
# Expert Parallel Functions
# ============================================================================


def get_expert_model_parallel_group(name: Optional[str] = None):
    """Get the expert-model-parallel group the caller rank belongs to.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_expert_model_parallel_group()


def get_expert_model_parallel_world_size(name: Optional[str] = None):
    """Return world size for the expert-model-parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_expert_model_parallel_world_size()


def get_expert_model_parallel_rank(name: Optional[str] = None):
    """Return caller's rank in the expert-model-parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_expert_model_parallel_rank()


def get_expert_tensor_parallel_group(name: Optional[str] = None):
    """Get the expert-tensor-parallel group the caller rank belongs to.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_expert_tensor_parallel_group()


def get_expert_tensor_parallel_world_size(name: Optional[str] = None):
    """Return world size for the expert tensor parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_expert_tensor_parallel_world_size()


def get_expert_tensor_parallel_rank(name: Optional[str] = None):
    """Return my rank for the expert tensor parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_expert_tensor_parallel_rank()


def get_expert_data_parallel_group(name: Optional[str] = None, partial_expert_data_parallel: bool = False):
    """Get expert data parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.
        partial_expert_data_parallel: Whether to use partial expert data parallel.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_expert_data_parallel_group(
        partial_expert_data_parallel=partial_expert_data_parallel)


def get_expert_data_parallel_world_size(name: Optional[str] = None, partial_expert_data_parallel: bool = False):
    """Return world size for the expert data parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.
        partial_expert_data_parallel: Whether to use partial expert data parallel.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_expert_data_parallel_world_size(
        partial_expert_data_parallel=partial_expert_data_parallel)


def get_expert_data_parallel_rank(name: Optional[str] = None, partial_expert_data_parallel: bool = False):
    """Return caller's rank in the expert data parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.
        partial_expert_data_parallel: Whether to use partial expert data parallel.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_expert_data_parallel_rank(
        partial_expert_data_parallel=partial_expert_data_parallel)


# ============================================================================
# Additional Helper Functions
# ============================================================================


def get_embedding_group(name: Optional[str] = None):
    """Get the embedding group the caller rank belongs to.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_embedding_group()


def get_tensor_and_data_parallel_group(name: Optional[str] = None, with_context_parallel: bool = False):
    """Get the tensor- and data-parallel group the caller rank belongs to.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.
        with_context_parallel: Whether to include context parallel.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_tensor_and_data_parallel_group(with_context_parallel=with_context_parallel)


def get_tensor_and_context_parallel_group(name: Optional[str] = None):
    """Get the tensor- and context-parallel group the caller rank belongs to.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_tensor_and_context_parallel_group()


def is_initialized(name: Optional[str] = None):
    """Check if parallel state has been initialized.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).is_initialized()
