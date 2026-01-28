# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Configurable AutoTP API

This module provides a unified specification for tensor parallel layer partitioning.
The design is inspired by Universal Checkpointing's SubparamShape and provides
a single, well-defined format that users can easily understand, customize, and extend.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional
from enum import Enum
from deepspeed.utils.logging import warning_once


class PartitionType(Enum):
    """How the layer should be partitioned for tensor parallelism."""
    COLUMN = "column"  # Partition output dim, AllReduce in backward
    ROW = "row"  # Partition input dim, AllReduce in forward
    SKIP = "skip"  # Do not partition this layer


@dataclass
class TPLayerSpec:
    """
    Unified specification for tensor parallel layer partitioning.

    This is inspired by Universal Checkpointing's SubparamShape but extended
    for AutoTP's needs (forward/backward communication patterns).

    The `shape` parameter supports at most 1-level nesting at the partition dimension:
    - (3, -1)           -> 3 equal-size sub-params
    - ((q, k, v), -1)   -> 3 unequal-size sub-params (1-level nesting)

    Examples:
        # Simple row-parallel layer (e.g., o_proj, down_proj)
        TPLayerSpec(
            patterns=[".*\\.o_proj$", ".*\\.down_proj$"],
            partition_type=PartitionType.ROW,
        )

        # Simple column-parallel layer (e.g., q_proj, k_proj, v_proj)
        TPLayerSpec(
            patterns=[".*\\.[qkv]_proj$"],
            partition_type=PartitionType.COLUMN,
        )

        # Fused QKV - GLM style [Q, K, V] concatenated on dim 0
        TPLayerSpec(
            patterns=[".*\\.query_key_value\\.weight$"],
            partition_type=PartitionType.COLUMN,
            shape=(3, -1),  # 3 equal sub-params, -1 = infer
            partition_dim=0,
        )

        # Fused QKV - Bloom style [q1,k1,v1,q2,k2,v2,...]
        TPLayerSpec(
            patterns=[".*\\.query_key_value\\.weight$"],
            partition_type=PartitionType.COLUMN,
            # No reshape needed, just split along dim 0
        )

        # GQA with different Q/K/V sizes (1-level nesting)
        TPLayerSpec(
            patterns=[".*\\.qkv_proj\\.weight$"],
            partition_type=PartitionType.COLUMN,
            shape=((q_size, k_size, v_size), -1),  # Unequal sub-params
            partition_dim=0,
        )

        # Chunked MLP (gate_up_proj)
        TPLayerSpec(
            patterns=[".*\\.gate_up_proj\\.weight$"],
            partition_type=PartitionType.COLUMN,
            shape=(2, -1),  # [gate, up] packed
            partition_dim=0,
        )

        # MoE FFN with expert dimension
        TPLayerSpec(
            patterns=[".*\\.experts\\..*\\.w1\\.weight$"],
            partition_type=PartitionType.COLUMN,
            shape=(num_experts, -1, hidden_in),  # View as 3D
            partition_dim=1,  # Partition the hidden_out dimension
        )

        # Skip layer (e.g., MoE gate)
        TPLayerSpec(
            patterns=[".*\\.gate$", ".*\\.router$"],
            partition_type=PartitionType.SKIP,
        )
    """

    # Layer identification - regex patterns to match parameter names
    patterns: List[str]

    # Partition type determines communication pattern
    partition_type: PartitionType = PartitionType.COLUMN

    # Optional: logical shape for partitioning
    # - Use -1 for dimensions that should be inferred
    # - Use tuple of ints at partition_dim for unequal sub-params (1-level nesting only)
    # Examples:
    #   (3, -1)           -> 3 equal sub-params
    #   ((4096, 1024, 1024), -1) -> 3 unequal sub-params (GQA)
    #   (n_experts, -1, hidden) -> MoE reshape
    shape: Optional[Tuple[Union[int, Tuple[int, ...]], ...]] = None

    # Which dimension to partition (after optional reshape)
    # Default: 0 for COLUMN, 1 for ROW (standard 2D weight matrix)
    partition_dim: Optional[int] = None

    # Optional: model type constraint (only apply for specific models)
    model_types: Optional[List[str]] = None

    def __post_init__(self):
        if isinstance(self.partition_type, str):
            self.partition_type = PartitionType(self.partition_type.lower())
        if self.shape is not None:
            self.shape = self._normalize_shape(self.shape)
            self._validate_shape_format()

    @staticmethod
    def _normalize_shape(shape):
        if isinstance(shape, list):
            return tuple(TPLayerSpec._normalize_shape(item) for item in shape)
        if isinstance(shape, tuple):
            return tuple(TPLayerSpec._normalize_shape(item) if isinstance(item, list) else item for item in shape)
        return shape

    def _validate_shape_format(self):
        if not isinstance(self.shape, tuple):
            raise ValueError("AutoTP shape must be a tuple of ints or a tuple at partition_dim.")
        partition_dim = self.get_partition_dim()
        if partition_dim < 0 or partition_dim >= len(self.shape):
            raise ValueError(
                f"AutoTP partition_dim {partition_dim} is out of range for shape length {len(self.shape)}.")
        nested_tuple_seen = False
        for idx, dim in enumerate(self.shape):
            if isinstance(dim, tuple):
                if idx != partition_dim:
                    raise ValueError(
                        f"AutoTP shape nested tuple only allowed at partition_dim={partition_dim}, got at {idx}.")
                if nested_tuple_seen:
                    raise ValueError("AutoTP shape supports only 1-level nesting at partition_dim.")
                nested_tuple_seen = True
                if len(dim) == 0:
                    raise ValueError("AutoTP shape nested tuple cannot be empty.")
                for val in dim:
                    if isinstance(val, tuple):
                        raise ValueError("AutoTP shape supports only 1-level nesting at partition_dim.")
                    if not isinstance(val, int) or val <= 0:
                        raise ValueError("AutoTP nested sub-parameter sizes must be positive integers.")
            elif isinstance(dim, int):
                if dim == 0 or dim < -1:
                    raise ValueError("AutoTP shape dimensions must be positive integers or -1.")
            else:
                raise ValueError("AutoTP shape must contain only integers or a tuple at partition_dim.")

    def get_partition_dim(self) -> int:
        """Get effective partition dimension."""
        if self.partition_dim is not None:
            return self.partition_dim
        # Default based on partition type for 2D weight matrices
        return 0 if self.partition_type == PartitionType.COLUMN else 1

    def has_unequal_sub_params(self) -> bool:
        """Check if this spec has unequal sub-parameters (nested tuple at partition_dim)."""
        if self.shape is None:
            return False
        dim = self.get_partition_dim()
        if dim >= len(self.shape):
            return False
        return isinstance(self.shape[dim], tuple)

    def get_sub_param_sizes(self) -> Optional[Tuple[int, ...]]:
        """Get sub-parameter sizes if using unequal sub-params."""
        if not self.has_unequal_sub_params():
            return None
        return self.shape[self.get_partition_dim()]

    def get_num_sub_params(self) -> Optional[int]:
        """Get the number of sub-parameters."""
        if self.shape is None:
            return None
        dim = self.get_partition_dim()
        if dim >= len(self.shape):
            return None
        if isinstance(self.shape[dim], tuple):
            return len(self.shape[dim])
        elif isinstance(self.shape[dim], int) and self.shape[dim] > 0:
            return self.shape[dim]
        return None

    def matches(self, param_name: str, model_type: Optional[str] = None) -> bool:
        """Check if this spec matches the given parameter."""
        # Check model type constraint
        if self.model_types:
            if model_type is None:
                return False
            model_type_norm = str(model_type).lower()
            model_types_norm = [str(mt).lower() for mt in self.model_types]
            if model_type_norm not in model_types_norm:
                return False
        # Check pattern match
        return any(re.match(pattern, param_name) for pattern in self.patterns)


@dataclass
class AutoTPConfig:
    """
    Configuration for Automatic Tensor Parallelism.

    Example usage:
        config = AutoTPConfig(
            tp_size=4,
            layer_specs=[
                # Row-parallel layers (AllReduce after forward)
                TPLayerSpec(
                    patterns=[".*\\.o_proj", ".*\\.down_proj"],
                    partition_type=PartitionType.ROW,
                ),
                # Column-parallel layers
                TPLayerSpec(
                    patterns=[".*\\.[qkv]_proj", ".*\\.up_proj", ".*\\.gate_proj"],
                    partition_type=PartitionType.COLUMN,
                ),
                # Skip MoE gates
                TPLayerSpec(
                    patterns=[".*\\.gate$"],
                    partition_type=PartitionType.SKIP,
                ),
            ],
        )
    """

    tp_size: int = 1

    # Unified layer specifications
    layer_specs: List[TPLayerSpec] = field(default_factory=list)

    # Embedding configuration
    embedding_partition_dim: int = 1  # Usually partition vocab dim

    # LM head configuration
    lm_head_patterns: List[str] = field(default_factory=lambda: ["lm_head", "embed_out"])

    # Behavior flags
    use_default_specs: bool = True  # Merge with built-in specs
    strict_mode: bool = False  # Fail if unmatched Linear layers found

    def find_matching_spec(self, param_name: str, model_type: Optional[str] = None) -> Optional[TPLayerSpec]:
        """Find the first matching spec for a parameter."""
        matches = [spec for spec in self.layer_specs if spec.matches(param_name, model_type)]
        if not matches:
            return None
        if len(matches) > 1:
            warning_once(f"AutoTPConfig: parameter {param_name} matched multiple layer_specs; using the first match.")
        return matches[0]

    @classmethod
    def from_dict(cls, config_dict: dict) -> "AutoTPConfig":
        """Create config from dictionary (JSON config)."""
        layer_specs = []
        for spec_dict in config_dict.get("layer_specs", []):
            # Convert partition_type string to enum
            partition_type_str = spec_dict.get("partition_type", "column")
            if isinstance(partition_type_str, str):
                partition_type = PartitionType(partition_type_str.lower())
            else:
                partition_type = partition_type_str

            # Convert shape from list to tuple if necessary
            shape = spec_dict.get("shape")
            if shape is not None:
                shape = cls._convert_shape(shape)

            layer_specs.append(
                TPLayerSpec(
                    patterns=spec_dict.get("patterns", []),
                    partition_type=partition_type,
                    shape=shape,
                    partition_dim=spec_dict.get("partition_dim"),
                    model_types=spec_dict.get("model_types"),
                ))

        return cls(
            tp_size=config_dict.get("tp_size", 1),
            layer_specs=layer_specs,
            embedding_partition_dim=config_dict.get("embedding_partition_dim", 1),
            lm_head_patterns=config_dict.get("lm_head_patterns", ["lm_head", "embed_out"]),
            use_default_specs=config_dict.get("use_default_specs", True),
            strict_mode=config_dict.get("strict_mode", False),
        )

    @staticmethod
    def _convert_shape(shape):
        """Convert shape from list to tuple, handling nested structures."""
        if isinstance(shape, list):
            return tuple(AutoTPConfig._convert_shape(item) if isinstance(item, list) else item for item in shape)
        return shape


class AutoTPPresets:
    """Built-in presets for common model architectures."""

    @staticmethod
    def llama() -> AutoTPConfig:
        """LLaMA-style models (separate Q, K, V projections)."""
        return AutoTPConfig(layer_specs=[
            TPLayerSpec(
                patterns=[r".*\.self_attn\.o_proj\.weight$"],
                partition_type=PartitionType.ROW,
            ),
            TPLayerSpec(
                patterns=[r".*\.self_attn\.[qkv]_proj\.weight$"],
                partition_type=PartitionType.COLUMN,
            ),
            TPLayerSpec(
                patterns=[r".*\.mlp\.down_proj\.weight$"],
                partition_type=PartitionType.ROW,
            ),
            TPLayerSpec(
                patterns=[r".*\.mlp\.(up|gate)_proj\.weight$"],
                partition_type=PartitionType.COLUMN,
            ),
        ], )

    @staticmethod
    def llama_gqa(num_heads: int, num_kv_heads: int, head_dim: int) -> AutoTPConfig:
        """LLaMA with Grouped Query Attention (fused QKV variant)."""
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        return AutoTPConfig(
            layer_specs=[
                TPLayerSpec(
                    patterns=[r".*\.self_attn\.o_proj\.weight$"],
                    partition_type=PartitionType.ROW,
                ),
                # Fused QKV with unequal sizes (GQA)
                TPLayerSpec(
                    patterns=[r".*\.self_attn\.qkv_proj\.weight$"],
                    partition_type=PartitionType.COLUMN,
                    shape=((q_size, kv_size, kv_size), -1),  # 1-level nesting
                    partition_dim=0,
                ),
                TPLayerSpec(
                    patterns=[r".*\.mlp\.down_proj\.weight$"],
                    partition_type=PartitionType.ROW,
                ),
                TPLayerSpec(
                    patterns=[r".*\.mlp\.(up|gate)_proj\.weight$"],
                    partition_type=PartitionType.COLUMN,
                ),
            ], )

    @staticmethod
    def bloom() -> AutoTPConfig:
        """BLOOM-style models (fused QKV with interleaved heads)."""
        return AutoTPConfig(
            layer_specs=[
                TPLayerSpec(
                    patterns=[r".*\.self_attention\.dense\.weight$"],
                    partition_type=PartitionType.ROW,
                ),
                TPLayerSpec(
                    patterns=[r".*\.self_attention\.query_key_value\.weight$"],
                    partition_type=PartitionType.COLUMN,
                    # Bloom style: [q1,k1,v1,q2,k2,v2,...] - no reshape needed
                ),
                TPLayerSpec(
                    patterns=[r".*\.mlp\.dense_4h_to_h\.weight$"],
                    partition_type=PartitionType.ROW,
                ),
                TPLayerSpec(
                    patterns=[r".*\.mlp\.dense_h_to_4h\.weight$"],
                    partition_type=PartitionType.COLUMN,
                ),
            ], )

    @staticmethod
    def chatglm() -> AutoTPConfig:
        """ChatGLM-style models (GLM-style fused QKV)."""
        return AutoTPConfig(
            layer_specs=[
                TPLayerSpec(
                    patterns=[r".*\.self_attention\.dense\.weight$"],
                    partition_type=PartitionType.ROW,
                ),
                TPLayerSpec(
                    patterns=[r".*\.self_attention\.query_key_value\.weight$"],
                    partition_type=PartitionType.COLUMN,
                    shape=(3, -1),  # [Q, K, V] concatenated
                    partition_dim=0,
                ),
                TPLayerSpec(
                    patterns=[r".*\.mlp\.dense_4h_to_h\.weight$"],
                    partition_type=PartitionType.ROW,
                ),
                TPLayerSpec(
                    patterns=[r".*\.mlp\.dense_h_to_4h\.weight$"],
                    partition_type=PartitionType.COLUMN,
                    shape=(2, -1),  # [gate, up] packed
                    partition_dim=0,
                ),
            ], )

    @staticmethod
    def mixtral() -> AutoTPConfig:
        """Mixtral MoE model."""
        return AutoTPConfig(
            layer_specs=[
                TPLayerSpec(
                    patterns=[r".*\.self_attn\.o_proj\.weight$"],
                    partition_type=PartitionType.ROW,
                ),
                TPLayerSpec(
                    patterns=[r".*\.self_attn\.[qkv]_proj\.weight$"],
                    partition_type=PartitionType.COLUMN,
                ),
                # MoE experts
                TPLayerSpec(
                    patterns=[r".*\.block_sparse_moe\.experts\.\d+\.w2\.weight$"],
                    partition_type=PartitionType.ROW,
                ),
                TPLayerSpec(
                    patterns=[r".*\.block_sparse_moe\.experts\.\d+\.w[13]\.weight$"],
                    partition_type=PartitionType.COLUMN,
                ),
                # Skip MoE gate
                TPLayerSpec(
                    patterns=[r".*\.block_sparse_moe\.gate\.weight$"],
                    partition_type=PartitionType.SKIP,
                ),
            ], )

    @staticmethod
    def deepseek_v2() -> AutoTPConfig:
        """DeepSeek-V2 with MLA (Multi-head Latent Attention)."""
        return AutoTPConfig(
            layer_specs=[
                # Standard attention output
                TPLayerSpec(
                    patterns=[r".*\.self_attn\.o_proj\.weight$"],
                    partition_type=PartitionType.ROW,
                ),
                # MLA uses compressed KV, skip low-rank projections
                TPLayerSpec(
                    patterns=[r".*\.self_attn\.(q_a_proj|kv_a_proj_with_mqa)\.weight$"],
                    partition_type=PartitionType.SKIP,
                ),
                # Q/K/V projections from latent
                TPLayerSpec(
                    patterns=[r".*\.self_attn\.(q_b_proj|kv_b_proj)\.weight$"],
                    partition_type=PartitionType.COLUMN,
                ),
                # MoE experts
                TPLayerSpec(
                    patterns=[r".*\.mlp\.experts\.\d+\.down_proj\.weight$"],
                    partition_type=PartitionType.ROW,
                ),
                TPLayerSpec(
                    patterns=[r".*\.mlp\.experts\.\d+\.(up|gate)_proj\.weight$"],
                    partition_type=PartitionType.COLUMN,
                ),
                # Skip MoE gate
                TPLayerSpec(
                    patterns=[r".*\.mlp\.gate\.weight$"],
                    partition_type=PartitionType.SKIP,
                ),
                # Shared expert
                TPLayerSpec(
                    patterns=[r".*\.mlp\.shared_experts\.down_proj\.weight$"],
                    partition_type=PartitionType.ROW,
                ),
                TPLayerSpec(
                    patterns=[r".*\.mlp\.shared_experts\.(up|gate)_proj\.weight$"],
                    partition_type=PartitionType.COLUMN,
                ),
            ], )

    @staticmethod
    def qwen2() -> AutoTPConfig:
        """Qwen2 model."""
        return AutoTPConfig(layer_specs=[
            TPLayerSpec(
                patterns=[r".*\.self_attn\.o_proj\.weight$"],
                partition_type=PartitionType.ROW,
            ),
            TPLayerSpec(
                patterns=[r".*\.self_attn\.[qkv]_proj\.weight$"],
                partition_type=PartitionType.COLUMN,
            ),
            TPLayerSpec(
                patterns=[r".*\.mlp\.down_proj\.weight$"],
                partition_type=PartitionType.ROW,
            ),
            TPLayerSpec(
                patterns=[r".*\.mlp\.(up|gate)_proj\.weight$"],
                partition_type=PartitionType.COLUMN,
            ),
        ], )

    @staticmethod
    def phi3() -> AutoTPConfig:
        """Phi3 model with fused QKV and chunked MLP."""
        return AutoTPConfig(
            layer_specs=[
                TPLayerSpec(
                    patterns=[r".*\.self_attn\.o_proj\.weight$"],
                    partition_type=PartitionType.ROW,
                ),
                # Phi3 has fused qkv_proj
                TPLayerSpec(
                    patterns=[r".*\.self_attn\.qkv_proj\.weight$"],
                    partition_type=PartitionType.COLUMN,
                    shape=(3, -1),  # [Q, K, V] concatenated
                    partition_dim=0,
                ),
                TPLayerSpec(
                    patterns=[r".*\.mlp\.down_proj\.weight$"],
                    partition_type=PartitionType.ROW,
                ),
                # Phi3 has gate_up_proj fused
                TPLayerSpec(
                    patterns=[r".*\.mlp\.gate_up_proj\.weight$"],
                    partition_type=PartitionType.COLUMN,
                    shape=(2, -1),  # [gate, up] packed
                    partition_dim=0,
                ),
            ], )

    @staticmethod
    def get_preset(model_type: str) -> Optional[AutoTPConfig]:
        """Get a preset configuration by model type name."""
        presets = {
            "llama": AutoTPPresets.llama,
            "bloom": AutoTPPresets.bloom,
            "chatglm": AutoTPPresets.chatglm,
            "mixtral": AutoTPPresets.mixtral,
            "deepseek_v2": AutoTPPresets.deepseek_v2,
            "qwen2": AutoTPPresets.qwen2,
            "phi3": AutoTPPresets.phi3,
        }
        preset_fn = presets.get(model_type.lower())
        if preset_fn:
            return preset_fn()
        return None


def merge_autotp_configs(base: AutoTPConfig, override: AutoTPConfig) -> AutoTPConfig:
    """Merge two AutoTP configs, with override taking precedence."""
    # Combine layer specs - override specs come first (higher priority)
    merged_specs = list(override.layer_specs) + list(base.layer_specs)

    return AutoTPConfig(
        tp_size=override.tp_size if override.tp_size > 1 else base.tp_size,
        layer_specs=merged_specs,
        embedding_partition_dim=override.embedding_partition_dim,
        lm_head_patterns=override.lm_head_patterns or base.lm_head_patterns,
        use_default_specs=override.use_default_specs,
        strict_mode=override.strict_mode,
    )
