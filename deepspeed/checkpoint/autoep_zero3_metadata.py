# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Shared validation for AutoEP ZeRO-3 checkpoint metadata."""

from deepspeed.checkpoint.constants import (
    AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION,
    AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION_KEY,
    AUTOEP_ZERO3_EXPERT_STATE_FORMAT_KEY,
    AUTOEP_ZERO3_PARTITIONED_EXPERT_STATE_FORMAT,
)

AUTOEP_METADATA_REQUIRED_FIELDS = frozenset({
    'moe_layer_id',
    'module_path',
    'num_experts',
    'num_local_experts',
    'ep_size',
    'expert_key_prefix',
})

AUTOEP_ZERO3_PARTITIONED_METADATA_FIELDS = frozenset({
    AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION_KEY,
    'ep_group_name',
    'ep_rank',
    'expert_data_parallel_rank',
    'expert_data_parallel_world_size',
    'global_expert_start',
    'global_expert_end',
})


def is_autoep_zero3_partitioned_entry(entry):
    return (isinstance(entry, dict)
            and entry.get(AUTOEP_ZERO3_EXPERT_STATE_FORMAT_KEY) == AUTOEP_ZERO3_PARTITIONED_EXPERT_STATE_FORMAT)


def validate_autoep_zero3_partitioned_metadata(autoep_metadata,
                                               require_partitioned=True,
                                               expected_expert_prefixes=None,
                                               version_context="This DeepSpeed build"):
    if not isinstance(autoep_metadata, list):
        raise RuntimeError(f"ds_autoep_layers metadata is malformed: expected list, got "
                           f"{type(autoep_metadata).__name__}")

    seen_layer_ids = set()
    seen_prefixes = set()
    partitioned_count = 0

    for entry in autoep_metadata:
        if not isinstance(entry, dict):
            raise RuntimeError(f"ds_autoep_layers entry is malformed: expected dict, got "
                               f"{type(entry).__name__}")
        missing = AUTOEP_METADATA_REQUIRED_FIELDS - entry.keys()
        if missing:
            raise RuntimeError(f"ds_autoep_layers entry is invalid: missing fields {sorted(missing)}")

        layer_id = entry['moe_layer_id']
        if layer_id in seen_layer_ids:
            raise RuntimeError(f"ds_autoep_layers metadata has duplicate moe_layer_id: {layer_id}")
        seen_layer_ids.add(layer_id)

        prefix = entry['expert_key_prefix']
        if prefix in seen_prefixes:
            raise RuntimeError(f"ds_autoep_layers metadata has duplicate expert_key_prefix: {prefix}")
        seen_prefixes.add(prefix)

        if not is_autoep_zero3_partitioned_entry(entry):
            continue

        missing = AUTOEP_ZERO3_PARTITIONED_METADATA_FIELDS - entry.keys()
        if missing:
            raise RuntimeError(f"AutoEP ZeRO-3 checkpoint metadata is invalid: missing fields {sorted(missing)}")
        version = entry[AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION_KEY]
        if version != AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION:
            raise RuntimeError("Unsupported AutoEP ZeRO-3 checkpoint format version: "
                               f"{version}. {version_context} supports version "
                               f"{AUTOEP_ZERO3_EXPERT_STATE_FORMAT_VERSION}.")

        num_experts = entry['num_experts']
        num_local_experts = entry['num_local_experts']
        ep_size = entry['ep_size']
        if num_local_experts * ep_size != num_experts:
            raise RuntimeError("AutoEP ZeRO-3 checkpoint metadata is inconsistent: "
                               f"num_local_experts={num_local_experts}, ep_size={ep_size}, "
                               f"num_experts={num_experts}")

        expected_start = entry['ep_rank'] * num_local_experts
        expected_end = expected_start + num_local_experts
        if entry['global_expert_start'] != expected_start or entry['global_expert_end'] != expected_end:
            raise RuntimeError("AutoEP ZeRO-3 checkpoint metadata has inconsistent global expert range: "
                               f"got [{entry['global_expert_start']}, {entry['global_expert_end']}), "
                               f"expected [{expected_start}, {expected_end})")

        if expected_expert_prefixes is not None:
            module_path = entry['module_path']
            if module_path not in expected_expert_prefixes:
                raise RuntimeError(f"AutoEP ZeRO-3 checkpoint metadata references missing module: {module_path}")
            expected_prefix = expected_expert_prefixes[module_path]
            if prefix != expected_prefix:
                raise RuntimeError("AutoEP ZeRO-3 checkpoint metadata has unexpected expert key prefix: "
                                   f"got {prefix}, expected {expected_prefix}")

        partitioned_count += 1

    if require_partitioned and partitioned_count == 0:
        raise RuntimeError("AutoEP ZeRO-3 partition-native checkpoint metadata was expected but no "
                           "partitioned AutoEP layer entries were found")
