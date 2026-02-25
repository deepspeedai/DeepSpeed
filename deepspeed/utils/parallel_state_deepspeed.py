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
- Configuration-based initialization from config.json

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

    # Initialize from config.json:
    from deepspeed import DeepSpeedConfig
    ds_config = DeepSpeedConfig("config.json")
    initialize_parallel_state_from_config(ds_config)
"""

from contextlib import contextmanager
from typing import Optional, Union, Dict, Any, List
from .parallel_state import ParallelState, get_parallel_state as _get_default_parallel_state

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
    import deepspeed.comm as dist
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
# Sequence Parallel Functions
# ============================================================================


def get_sequence_parallel_group(name: Optional[str] = None):
    """Get the sequence-parallel group the caller rank belongs to.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_sequence_parallel_group()


def get_sequence_parallel_world_size(name: Optional[str] = None):
    """Return world size for the sequence parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_sequence_parallel_world_size()


def get_sequence_parallel_rank(name: Optional[str] = None):
    """Return caller's rank in the sequence-parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_sequence_parallel_rank()


def get_sequence_and_data_parallel_group(name: Optional[str] = None):
    """Get the sequence and data parallel group the caller rank belongs to.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_sequence_and_data_parallel_group()


def get_sequence_and_data_parallel_world_size(name: Optional[str] = None):
    """Return world size for the sequence and data parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_sequence_and_data_parallel_world_size()


def get_sequence_and_data_parallel_rank(name: Optional[str] = None):
    """Return caller's rank in the sequence and data parallel group.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_sequence_and_data_parallel_rank()


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


# ============================================================================
# All-to-All Groups for ZeRO++ Quantized Gradients
# ============================================================================


def initialize_all_to_all_groups(name: Optional[str] = None):
    """Initialize All-to-All groups for quantized gradient communication.

    Creates local and global All-to-All groups based on node topology.
    Used by ZeRO++ when zero_quantized_gradients is enabled.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    Returns:
        Dictionary of All-to-All groups

    Example:
        # Initialize for default instance
        all_to_all_groups = initialize_all_to_all_groups()

        # Initialize for named instance (RL scenario)
        actor_groups = initialize_all_to_all_groups("actor")
        critic_groups = initialize_all_to_all_groups("critic")

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).initialize_all_to_all_groups()


def get_all_to_all_groups(name: Optional[str] = None):
    """Get All-to-All groups dictionary.

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    Returns:
        Dictionary of All-to-All groups

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_all_to_all_groups()


def _get_local_all_to_all_group(name: Optional[str] = None):
    """Get All-to-All groups for current rank (backward compatible with groups.py).

    This function provides backward compatibility with the groups.py interface.
    It returns all All-to-All groups (both local and global).

    Args:
        name: Optional name of the parallel state instance. If None, uses current active instance.

    Returns:
        Dictionary of All-to-All groups

    Note:
        This is a compatibility wrapper. New code should use get_all_to_all_groups() instead.

    DeepSpeed-compatible interface.
    """
    return get_parallel_state(name).get_all_to_all_groups()


# ============================================================================
# Configuration-based Initialization
# ============================================================================


def initialize_parallel_state_from_config(
    config: Union[Dict[str, Any], Any],
    name: Optional[str] = None,
    # Optional parameters to override config values
    tensor_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_size: Optional[int] = None,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_comm_backend: Optional[str] = None,
    context_parallel_size: Optional[int] = None,
    hierarchical_context_parallel_sizes: Optional[List[int]] = None,
    expert_model_parallel_size: Optional[int] = None,
    num_distributed_optimizer_instances: Optional[int] = None,
    expert_tensor_parallel_size: Optional[int] = None,
    sequence_parallel_size: Optional[int] = None,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: Optional[int] = None,
    order: Optional[str] = None,
    create_gloo_process_groups: Optional[bool] = None,
    high_priority_stream_groups: Optional[List[str]] = None,
) -> None:
    """Initialize parallel state from DeepSpeed config.json with optional parameter overrides.

    This function reads parallelism configuration from the DeepSpeed config file
    (top-level fields) and automatically initializes the ParallelState instance.
    This allows code to work with both explicit initialization and config-based initialization.

    Configuration priority: function parameters > config file values > default values (1)

    Args:
        config: Either a DeepSpeedConfig object or a config dictionary.
                If DeepSpeedConfig, will access its _param_dict attribute.
                If dict, will use it directly.
        name: Optional name of the parallel state instance to initialize.
              If None, initializes the default global instance.

        # Parallelism dimension parameters (override config if provided):
        tensor_model_parallel_size: Size of tensor model parallel group. Default: 1
        pipeline_model_parallel_size: Size of pipeline model parallel group. Default: 1
        virtual_pipeline_model_parallel_size: Virtual pipeline model parallel size. Default: None
        pipeline_model_parallel_comm_backend: Communication backend for pipeline. Default: None
        context_parallel_size: Size of context parallel group. Default: 1 (MUST be 1, CP not supported)
        hierarchical_context_parallel_sizes: Hierarchical context parallel sizes. Default: None (NOT supported)
        expert_model_parallel_size: Size of expert model parallel group. Default: 1
        num_distributed_optimizer_instances: Number of distributed optimizer instances. Default: 1
        expert_tensor_parallel_size: Size of expert tensor parallel group. Default: None
        sequence_parallel_size: Size of sequence parallel group. Default: 1
        nccl_communicator_config_path: Path to NCCL communicator config. Default: None
        distributed_timeout_minutes: Timeout for distributed operations. Default: 30
        order: Order of parallelism dimensions. Default: "tp-ep-dp-pp"
        create_gloo_process_groups: Whether to create Gloo process groups. Default: False
        high_priority_stream_groups: High priority stream groups. Default: None

    Example config.json (using existing DeepSpeed config fields):
        {
          "train_batch_size": 8,
          "sequence_parallel_size": 1,
          "zero_optimization": {
            "stage": 1
          }
        }

    Note:
    - Currently only "sequence_parallel_size" can be read from config (existing field)
    - Other parallelism parameters must be passed via function parameters or use defaults
    - Context Parallel is NOT supported (cp must be 1)

    Example usage:
        # Basic usage from config file:
        from deepspeed import DeepSpeedConfig
        ds_config = DeepSpeedConfig("config.json")
        initialize_parallel_state_from_config(ds_config)

        # Override specific parameters:
        initialize_parallel_state_from_config(
            ds_config,
            tensor_model_parallel_size=4,  # Override config value
            expert_model_parallel_size=2
        )

        # From config dictionary:
        import json
        with open("config.json") as f:
            config_dict = json.load(f)
        initialize_parallel_state_from_config(config_dict)

        # For named instances (RL scenarios):
        initialize_parallel_state_from_config(ds_config, name="actor")
        initialize_parallel_state_from_config(
            ds_config,
            name="critic",
            tensor_model_parallel_size=2  # Override for critic
        )
    """
    # Extract config dictionary
    if hasattr(config, '_param_dict'):
        # DeepSpeedConfig object
        config_dict = config._param_dict
    elif isinstance(config, dict):
        # Already a dictionary
        config_dict = config
    else:
        raise ValueError(f"config must be a DeepSpeedConfig object or a dict, got {type(config)}")

    # Get the parallel state instance
    ps = get_parallel_state_instance(name)

    # Check if already initialized
    if ps.is_initialized():
        # Already initialized, skip
        return

    # Import logging
    import logging
    logger = logging.getLogger(__name__)

    # Helper function to get value with proper priority handling
    # Priority: function parameter > config file value > default value
    def get_value(param_name, param_value, config_key, default_value):
        """
        Get value with priority handling.

        Priority:
        1. If function parameter is provided -> use parameter value
        2. If config file has the value -> use config value
        3. Otherwise -> use default value
        """
        # Case 1: Function parameter provided
        if param_value is not None:
            return param_value

        # Case 2: Config file has the key
        if config_key in config_dict:
            config_value = config_dict[config_key]
            return config_value

        # Case 3: Use default
        return default_value

    # Extract parameters with proper priority: function param > config value > default
    init_kwargs = {
        "tensor_model_parallel_size":
        get_value("tensor_model_parallel_size", tensor_model_parallel_size, "tensor_model_parallel_size", 1),
        "pipeline_model_parallel_size":
        get_value("pipeline_model_parallel_size", pipeline_model_parallel_size, "pipeline_model_parallel_size", 1),
        "virtual_pipeline_model_parallel_size":
        get_value("virtual_pipeline_model_parallel_size", virtual_pipeline_model_parallel_size,
                  "virtual_pipeline_model_parallel_size", None),
        "pipeline_model_parallel_comm_backend":
        get_value("pipeline_model_parallel_comm_backend", pipeline_model_parallel_comm_backend,
                  "pipeline_model_parallel_comm_backend", None),
        "context_parallel_size":
        get_value("context_parallel_size", context_parallel_size, "context_parallel_size", 1),
        "sequence_parallel_size":
        get_value("sequence_parallel_size", sequence_parallel_size, "sequence_parallel_size", 1),
        "hierarchical_context_parallel_sizes":
        get_value("hierarchical_context_parallel_sizes", hierarchical_context_parallel_sizes,
                  "hierarchical_context_parallel_sizes", None),
        "expert_model_parallel_size":
        get_value("expert_model_parallel_size", expert_model_parallel_size, "expert_model_parallel_size", 1),
        "num_distributed_optimizer_instances":
        get_value("num_distributed_optimizer_instances", num_distributed_optimizer_instances,
                  "num_distributed_optimizer_instances", 1),
        "expert_tensor_parallel_size":
        get_value("expert_tensor_parallel_size", expert_tensor_parallel_size, "expert_tensor_parallel_size", None),
        "nccl_communicator_config_path":
        get_value("nccl_communicator_config_path", nccl_communicator_config_path, "nccl_communicator_config_path",
                  None),
        "distributed_timeout_minutes":
        get_value("distributed_timeout_minutes", distributed_timeout_minutes, "distributed_timeout_minutes", 30),
        "order":
        get_value("order", order, "order", "tp-ep-dp-pp"),
        "create_gloo_process_groups":
        get_value("create_gloo_process_groups", create_gloo_process_groups, "create_gloo_process_groups", False),
        "high_priority_stream_groups":
        get_value("high_priority_stream_groups", high_priority_stream_groups, "high_priority_stream_groups", None),
    }

    # Validate context_parallel_size
    cp_size = init_kwargs["context_parallel_size"]
    if cp_size != 1:
        raise NotImplementedError(
            f"DeepSpeed currently does not support context_parallel_size > 1. "
            f"Got context_parallel_size={cp_size}. Please set context_parallel_size=1 in your config.")

    # Validate hierarchical_context_parallel_sizes
    hcp_sizes = init_kwargs["hierarchical_context_parallel_sizes"]
    if hcp_sizes is not None:
        raise NotImplementedError(
            f"DeepSpeed currently does not support hierarchical_context_parallel_sizes. "
            f"Got hierarchical_context_parallel_sizes={hcp_sizes}. Please remove this configuration.")

    # Remove None values for optional parameters (except those that can be None)
    # Keep None for: virtual_pipeline_model_parallel_size, pipeline_model_parallel_comm_backend,
    # hierarchical_context_parallel_sizes, expert_tensor_parallel_size
    # Note: nccl_communicator_config_path and high_priority_stream_groups are not supported by initialize_model_parallel
    filtered_kwargs = {}
    supported_params = {
        "tensor_model_parallel_size", "pipeline_model_parallel_size", "virtual_pipeline_model_parallel_size",
        "pipeline_model_parallel_comm_backend", "context_parallel_size", "sequence_parallel_size",
        "hierarchical_context_parallel_sizes", "expert_model_parallel_size", "num_distributed_optimizer_instances",
        "expert_tensor_parallel_size", "distributed_timeout_minutes", "order", "create_gloo_process_groups"
    }
    
    for key, value in init_kwargs.items():
        # Skip unsupported parameters
        if key not in supported_params:
            continue
        # Keep None for parameters that can be None
        if value is not None or key in [
                "virtual_pipeline_model_parallel_size", "pipeline_model_parallel_comm_backend",
                "hierarchical_context_parallel_sizes", "expert_tensor_parallel_size"
        ]:
            filtered_kwargs[key] = value

    # Initialize parallel state
    ps.initialize_model_parallel(**filtered_kwargs)
