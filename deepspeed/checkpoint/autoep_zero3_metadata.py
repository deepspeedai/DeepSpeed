# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Shared validation for AutoEP ZeRO-3 checkpoint metadata."""

from deepspeed.checkpoint.autoep_affine import (legacy_uniform_autoep_placement_descriptor,
                                                validate_autoep_placement_descriptor)
from deepspeed.checkpoint.constants import (
    AUTOEP_EXPERT_PLACEMENT,
    AUTOEP_PLACEMENT_EP_SIZE,
    AUTOEP_PLACEMENT_EXPERTS,
    AUTOEP_PLACEMENT_NUM_EXPERTS,
    AUTOEP_PLACEMENT_RANK,
    AUTOEP_PLACEMENT_RANKS,
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
                                               expected_runtime_layers=None,
                                               version_context="This DeepSpeed build"):
    if not isinstance(autoep_metadata, list):
        raise RuntimeError(f"ds_autoep_layers metadata is malformed: expected list, got "
                           f"{type(autoep_metadata).__name__}")

    seen_layer_ids = set()
    seen_prefixes = set()
    partitioned_count = 0
    placements = []

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

        placement = _validated_placement(entry)
        placements.append(placement)
        _validate_runtime_layer(entry, placement, expected_runtime_layers)

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

        if AUTOEP_EXPERT_PLACEMENT not in entry:
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
    return placements


def _validated_placement(entry):
    try:
        if AUTOEP_EXPERT_PLACEMENT in entry:
            placement = entry[AUTOEP_EXPERT_PLACEMENT]
            validate_autoep_placement_descriptor(placement)
        else:
            placement = legacy_uniform_autoep_placement_descriptor(entry['num_experts'], entry['num_local_experts'],
                                                                   entry['ep_size'])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"AutoEP placement metadata is invalid: {exc}") from exc

    if placement[AUTOEP_PLACEMENT_NUM_EXPERTS] != entry['num_experts']:
        raise RuntimeError("AutoEP placement metadata num_experts does not match its enclosing layer entry: "
                           f"{placement[AUTOEP_PLACEMENT_NUM_EXPERTS]} != {entry['num_experts']}")
    if placement[AUTOEP_PLACEMENT_EP_SIZE] != entry['ep_size']:
        raise RuntimeError("AutoEP placement metadata ep_size does not match its enclosing layer entry: "
                           f"{placement[AUTOEP_PLACEMENT_EP_SIZE]} != {entry['ep_size']}")
    return placement


def _validate_runtime_layer(entry, placement, expected_runtime_layers):
    ep_rank = entry.get('ep_rank')
    if ep_rank is None:
        return
    ranks = {rank_entry[AUTOEP_PLACEMENT_RANK]: rank_entry for rank_entry in placement[AUTOEP_PLACEMENT_RANKS]}
    if ep_rank not in ranks:
        raise RuntimeError(f"AutoEP placement metadata does not contain current ep_rank {ep_rank}")
    listed_experts = ranks[ep_rank][AUTOEP_PLACEMENT_EXPERTS]
    if len(listed_experts) != entry['num_local_experts']:
        raise RuntimeError("AutoEP placement metadata current-rank expert count does not match num_local_experts: "
                           f"{len(listed_experts)} != {entry['num_local_experts']}")

    if expected_runtime_layers is None:
        return
    module_path = entry['module_path']
    if module_path not in expected_runtime_layers:
        raise RuntimeError(f"AutoEP checkpoint metadata references missing module: {module_path}")
    runtime = expected_runtime_layers[module_path]
    if ep_rank != runtime['ep_rank']:
        raise RuntimeError(f"AutoEP checkpoint metadata ep_rank {ep_rank} does not match runtime ep_rank "
                           f"{runtime['ep_rank']} for module {module_path}")
    if listed_experts != runtime['local_experts']:
        raise RuntimeError("AutoEP placement metadata current-rank expert order does not match runtime local packing: "
                           f"{listed_experts} != {runtime['local_experts']} for module {module_path}")
