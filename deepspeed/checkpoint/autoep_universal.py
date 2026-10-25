# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP universal checkpoint conversion utilities.

Consolidates per-expert checkpoint files (and their optimizer states) into
topology-agnostic universal format for EP resharding support.
"""

import os
import glob
import torch
from .affine import AFFINE_MAP_FORMAT_VERSION

from .constants import (
    AUTOEP_AFFINE_MAPS,
    AUTOEP_EP_SIZE,
    AUTOEP_EXPERT_PLACEMENT,
    AUTOEP_EXPERT_KEY_PREFIX,
    AUTOEP_NUM_EXPERTS,
    AUTOEP_NUM_LOCAL_EXPERTS,
    AUTOEP_PLACEMENT_EP_SIZE,
    AUTOEP_PLACEMENT_EXPERTS,
    AUTOEP_PLACEMENT_NUM_EXPERTS,
    AUTOEP_PLACEMENT_RANK,
    AUTOEP_PLACEMENT_RANKS,
    AUTOEP_ZERO12_REQUIRED_FIELDS,
    AFFINE_MAP_PARAMS,
    AFFINE_MAP_VERSION,
    PARAM,
    CAT_DIM,
    EP_IS_EXPERT_PARAM,
    EP_NUM_EXPERTS,
    FOLDING_METADATA_KEY,
    FOLDING_METADATA_VERSION,
    FOLDING_TP_SIZE,
    FOLDING_TP_RANK,
    FOLDING_EP_SIZE,
    FOLDING_EP_RANK,
    FOLDING_ETP_SIZE,
    FOLDING_ETP_RANK,
    FOLDING_ZERO_PARTITION_GROUP,
    FOLDING_ZERO_PARTITION_RANK,
    FOLDING_ZERO_PARTITION_COUNT,
    FOLDING_DISPATCH_STRATEGY,
    FOLDING_SHARED_EXPERT_PLACEMENT,
    FOLDING_FAMILY,
    FOLDING_PARAM_FAMILIES,
)
from .autoep_affine import autoep_metadata_to_affine_map, validate_autoep_placement_descriptor


def make_folding_metadata(*,
                          tp_size,
                          tp_rank,
                          ep_size,
                          ep_rank,
                          zero_partition_group,
                          zero_partition_rank,
                          zero_partition_count,
                          family,
                          param_families=None):
    metadata = {
        "version": FOLDING_METADATA_VERSION,
        FOLDING_TP_SIZE: tp_size,
        FOLDING_TP_RANK: tp_rank,
        FOLDING_EP_SIZE: ep_size,
        FOLDING_EP_RANK: ep_rank,
        FOLDING_ETP_SIZE: 1,
        FOLDING_ETP_RANK: 0,
        FOLDING_ZERO_PARTITION_GROUP: zero_partition_group,
        FOLDING_ZERO_PARTITION_RANK: zero_partition_rank,
        FOLDING_ZERO_PARTITION_COUNT: zero_partition_count,
        FOLDING_DISPATCH_STRATEGY: "route_full_partition_dispatch",
        FOLDING_SHARED_EXPERT_PLACEMENT: "tp_sharded",
        FOLDING_FAMILY: family,
    }
    if param_families is not None:
        metadata[FOLDING_PARAM_FAMILIES] = dict(param_families)
    return metadata


def validate_folding_metadata(metadata,
                              *,
                              tp_size,
                              ep_size,
                              etp_size=1,
                              tp_rank=None,
                              ep_rank=None,
                              etp_rank=None,
                              zero_partition_group=None,
                              zero_partition_rank=None,
                              zero_partition_count=None,
                              family=None,
                              param_families=None,
                              shared_expert_placement=None,
                              dispatch_strategy=None):
    if not isinstance(metadata, dict) or FOLDING_METADATA_KEY not in metadata:
        raise RuntimeError("Missing AutoEP+AutoTP folding metadata in folded checkpoint.")
    folding = metadata[FOLDING_METADATA_KEY]
    if folding.get("version") != FOLDING_METADATA_VERSION:
        raise RuntimeError(f"Unsupported folding metadata version: {folding.get('version')}")
    expected = {
        FOLDING_TP_SIZE: tp_size,
        FOLDING_EP_SIZE: ep_size,
        FOLDING_ETP_SIZE: etp_size,
    }
    optional_expected = {
        FOLDING_TP_RANK: tp_rank,
        FOLDING_EP_RANK: ep_rank,
        FOLDING_ETP_RANK: etp_rank,
        FOLDING_ZERO_PARTITION_GROUP: zero_partition_group,
        FOLDING_ZERO_PARTITION_RANK: zero_partition_rank,
        FOLDING_ZERO_PARTITION_COUNT: zero_partition_count,
        FOLDING_FAMILY: family,
        FOLDING_PARAM_FAMILIES: param_families,
        FOLDING_SHARED_EXPERT_PLACEMENT: shared_expert_placement,
        FOLDING_DISPATCH_STRATEGY: dispatch_strategy,
    }
    expected.update({key: value for key, value in optional_expected.items() if value is not None})
    for key, value in expected.items():
        if folding.get(key) != value:
            raise RuntimeError(f"Folding metadata mismatch for {key}: saved={folding.get(key)} runtime={value}")
    return folding


def resolve_expert_ckpt_path(checkpoint_dir, moe_layer_id, global_expert_id):
    """Find the expert checkpoint file for a given (layer, expert) pair.

    Resolves using glob pattern without assuming mp_rank=0.

    Returns:
        Path to the single matching expert checkpoint file.

    Raises:
        FileNotFoundError: No matching file found.
        NotImplementedError: Multiple matching files found (multi-mp_rank).
    """
    pattern = os.path.join(checkpoint_dir, f'layer_{moe_layer_id}_expert_{global_expert_id}_mp_rank_*_model_states.pt')
    matches = glob.glob(pattern)
    if len(matches) == 0:
        raise FileNotFoundError(f"Expert checkpoint file not found: layer_{moe_layer_id} "
                                f"expert_{global_expert_id} in {checkpoint_dir}")
    if len(matches) > 1:
        for match in matches:
            state = torch.load(match, map_location='cpu', weights_only=False)
            if FOLDING_METADATA_KEY in state:
                raise NotImplementedError("Universal checkpoint conversion for folded AutoEP+AutoTP expert shards "
                                          "is not supported yet. Load this checkpoint with a matching folded "
                                          "runtime, or consolidate the tensor-parallel expert shards before "
                                          "running ds_to_universal.")
        raise NotImplementedError(f"Multiple expert checkpoint files found for layer_{moe_layer_id} "
                                  f"expert_{global_expert_id}: {matches}. Multi-mp_rank expert files "
                                  f"are not yet supported.")
    return matches[0]


def get_autoep_zero12_expert_param_info(autoep_layers_metadata):
    """Validate AutoEP metadata and map fused expert parameter names to layer metadata."""
    if not isinstance(autoep_layers_metadata, list) or not autoep_layers_metadata:
        raise RuntimeError("AutoEP metadata must be a non-empty list for ZeRO-1/2 universal conversion.")

    param_info = {}
    for layer_info in autoep_layers_metadata:
        if not isinstance(layer_info, dict):
            raise RuntimeError("AutoEP layer metadata must contain dictionaries.")

        missing_fields = [field for field in AUTOEP_ZERO12_REQUIRED_FIELDS if field not in layer_info]
        if missing_fields:
            raise RuntimeError(f"AutoEP layer metadata is missing fields: {missing_fields}")

        prefix = layer_info[AUTOEP_EXPERT_KEY_PREFIX]
        if not isinstance(prefix, str) or not prefix:
            raise RuntimeError("AutoEP expert_key_prefix must be a non-empty string.")

        for field in (AUTOEP_NUM_EXPERTS, AUTOEP_EP_SIZE):
            value = layer_info[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RuntimeError(f"AutoEP {field} must be a positive integer, got {value!r}.")

        num_experts = layer_info[AUTOEP_NUM_EXPERTS]
        num_local_experts = layer_info[AUTOEP_NUM_LOCAL_EXPERTS]
        ep_size = layer_info[AUTOEP_EP_SIZE]
        placement = layer_info.get(AUTOEP_EXPERT_PLACEMENT)
        if isinstance(num_local_experts, bool) or not isinstance(num_local_experts, int):
            raise RuntimeError(f"AutoEP {AUTOEP_NUM_LOCAL_EXPERTS} must be an integer, got "
                               f"{num_local_experts!r}.")
        minimum_local_experts = 0 if placement is not None else 1
        if num_local_experts < minimum_local_experts:
            qualifier = "non-negative" if placement is not None else "positive"
            raise RuntimeError(f"AutoEP {AUTOEP_NUM_LOCAL_EXPERTS} must be a {qualifier} integer, got "
                               f"{num_local_experts!r}.")
        if placement is not None:
            try:
                validate_autoep_placement_descriptor(placement)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid AutoEP expert placement for {prefix}: {exc}") from exc
            if (placement[AUTOEP_PLACEMENT_NUM_EXPERTS] != num_experts
                    or placement[AUTOEP_PLACEMENT_EP_SIZE] != ep_size):
                raise RuntimeError(f"AutoEP expert placement disagrees with layer metadata for {prefix}.")
            ep_rank = layer_info.get('ep_rank')
            if ep_rank is not None:
                rank_entries = {entry[AUTOEP_PLACEMENT_RANK]: entry for entry in placement[AUTOEP_PLACEMENT_RANKS]}
                if ep_rank not in rank_entries:
                    raise RuntimeError(f"AutoEP expert placement does not contain metadata ep_rank {ep_rank}.")
                expected_local_experts = len(rank_entries[ep_rank][AUTOEP_PLACEMENT_EXPERTS])
                if num_local_experts != expected_local_experts:
                    raise RuntimeError("AutoEP num_local_experts disagrees with the placement entry for "
                                       f"EP rank {ep_rank}: {num_local_experts} != {expected_local_experts}.")
        elif num_experts != num_local_experts * ep_size:
            raise RuntimeError(f"AutoEP expert count mismatch for {prefix}: num_experts={num_experts}, "
                               f"num_local_experts={num_local_experts}, ep_size={ep_size}.")

        normalized = {
            'num_experts': num_experts,
            'num_local_experts': num_local_experts,
            'ep_size': ep_size,
            'expert_placement': placement,
        }
        for weight_name in ('w1', 'w2', 'w3'):
            param_name = f"{prefix}.{weight_name}"
            if param_name in param_info:
                raise RuntimeError(f"Duplicate AutoEP expert parameter metadata for {param_name}.")
            param_metadata = dict(normalized)
            affine_maps = layer_info.get(AUTOEP_AFFINE_MAPS, {})
            if affine_maps:
                try:
                    if affine_maps.get(AFFINE_MAP_VERSION) != AFFINE_MAP_FORMAT_VERSION:
                        raise RuntimeError(f"Unsupported AutoEP affine map format version "
                                           f"{affine_maps.get(AFFINE_MAP_VERSION)!r}.")
                    param_metadata['affine_map'] = affine_maps[AFFINE_MAP_PARAMS][param_name]
                except KeyError as exc:
                    raise RuntimeError(f"AutoEP layer metadata is missing affine map for {param_name}.") from exc
            param_info[param_name] = param_metadata

    return param_info


def _zero12_fragment_path(temp_dir, param_name, state_name, dp_rank):
    return os.path.join(temp_dir, param_name, "0", f"{state_name}.{dp_rank:0>2d}")


def _autoep_zero12_dp_ranks(ep_rank, dp_degree, ep_size, use_data_before_expert_parallel):
    """Return stage-local ZeRO DP ranks that own one EP rank's fragments."""
    if dp_degree % ep_size != 0:
        raise RuntimeError(f"ZeRO DP degree {dp_degree} is not divisible by AutoEP size {ep_size}.")
    if use_data_before_expert_parallel:
        edp_size = dp_degree // ep_size
        return range(ep_rank * edp_size, (ep_rank + 1) * edp_size)
    return range(ep_rank, dp_degree, ep_size)


def get_autoep_zero12_fp32_fallback_param_names(temp_dir, expert_param_info, slice_shapes, dp_degree,
                                                use_data_before_expert_parallel):
    """Find parameters that need per-expert weights as a fallback for missing master fragments."""
    fallback_param_names = set()
    for param_name, metadata in expert_param_info.items():
        if param_name not in slice_shapes:
            raise RuntimeError(f"AutoEP expert parameter {param_name} is missing from checkpoint parameter shapes.")

        ep_size = metadata['ep_size']
        local_shape = tuple(slice_shapes[param_name])
        affine_map = None
        if metadata.get('expert_placement') is not None:
            logical_shape = (metadata['num_experts'], ) + local_shape[1:]
            affine_map = autoep_metadata_to_affine_map(metadata, logical_shape)

        for ep_rank in range(ep_size):
            expected_shape = affine_map.shard_shapes[ep_rank] if affine_map is not None else local_shape
            if torch.Size(expected_shape).numel() == 0:
                continue
            dp_ranks = _autoep_zero12_dp_ranks(ep_rank, dp_degree, ep_size, use_data_before_expert_parallel)
            if any(not os.path.isfile(_zero12_fragment_path(temp_dir, param_name, "fp32", dp_rank))
                   for dp_rank in dp_ranks):
                fallback_param_names.add(param_name)
                break

    return fallback_param_names


def _zero12_values_equal(left, right):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    return left == right


def consolidate_autoep_zero12_expert_states(temp_dir,
                                            output_dir,
                                            expert_param_info,
                                            slice_shapes,
                                            dp_degree,
                                            tp_degree,
                                            use_data_before_expert_parallel,
                                            fp32_fallback_dir=None):
    """Consolidate AutoEP expert FP32 and Adam states from ZeRO-1/2 fragments."""
    if tp_degree != 1:
        raise NotImplementedError("ZeRO-1/2 Universal Checkpoint conversion for AutoEP with tensor parallelism "
                                  "is not supported.")

    for param_name, metadata in expert_param_info.items():
        if param_name not in slice_shapes:
            raise RuntimeError(f"AutoEP expert parameter {param_name} is missing from checkpoint parameter shapes.")

        ep_size = metadata['ep_size']
        num_experts = metadata['num_experts']
        placement = metadata.get('expert_placement')

        local_shape = tuple(slice_shapes[param_name])
        if not local_shape:
            raise RuntimeError(f"AutoEP local shape is empty for {param_name}.")

        affine_map = None
        if placement is not None:
            logical_shape = (num_experts, ) + local_shape[1:]
            affine_map = autoep_metadata_to_affine_map(metadata, logical_shape)
        elif local_shape[0] != metadata['num_local_experts']:
            raise RuntimeError(f"AutoEP local shape mismatch for {param_name}: shape={local_shape}, "
                               f"num_local_experts={metadata['num_local_experts']}.")

        param_dir = os.path.join(output_dir, "zero", param_name)
        os.makedirs(param_dir, exist_ok=True)

        for state_name in ('fp32', 'exp_avg', 'exp_avg_sq'):
            ep_tensors = {}
            fp32_fallback_tensor = None
            for ep_rank in range(ep_size):
                expected_shape = affine_map.shard_shapes[ep_rank] if affine_map is not None else local_shape
                fragments = []
                dp_ranks = _autoep_zero12_dp_ranks(ep_rank, dp_degree, ep_size, use_data_before_expert_parallel)
                for dp_rank in dp_ranks:
                    fragment_path = _zero12_fragment_path(temp_dir, param_name, state_name, dp_rank)
                    if not os.path.isfile(fragment_path):
                        continue
                    fragment = torch.load(fragment_path, map_location='cpu', weights_only=False)
                    if not torch.is_tensor(fragment):
                        raise RuntimeError(f"AutoEP {state_name} fragment is not a tensor: {fragment_path}")
                    if fragment.dtype != torch.float32:
                        raise RuntimeError(f"AutoEP {state_name} fragment must be FP32, got {fragment.dtype} "
                                           f"in {fragment_path}.")
                    fragments.append(fragment.flatten())

                expected_numel = torch.Size(expected_shape).numel()
                if not fragments and expected_numel == 0:
                    local_tensor = torch.empty(expected_shape, dtype=torch.float32)
                    ep_tensors[ep_rank] = local_tensor
                    continue
                if not fragments:
                    fallback_path = (os.path.join(fp32_fallback_dir, param_name, "fp32.pt")
                                     if fp32_fallback_dir is not None else os.path.join(param_dir, "fp32.pt"))
                    if state_name == "fp32" and os.path.isfile(fallback_path):
                        if fp32_fallback_tensor is None:
                            fp32_fallback_state = torch.load(fallback_path, map_location='cpu', weights_only=False)
                            fp32_fallback_tensor = fp32_fallback_state[PARAM]
                        if affine_map is not None:
                            local_tensor = affine_map.extract(fp32_fallback_tensor, ep_rank)
                        else:
                            start = ep_rank * metadata['num_local_experts']
                            end = start + metadata['num_local_experts']
                            local_tensor = fp32_fallback_tensor[start:end]
                        ep_tensors[ep_rank] = local_tensor.reshape(expected_shape)
                        continue
                    raise RuntimeError(f"Missing AutoEP {state_name} fragments for {param_name}, EP rank {ep_rank}.")

                local_tensor = torch.cat(fragments, dim=0)
                if local_tensor.numel() != expected_numel:
                    fallback_path = (os.path.join(fp32_fallback_dir, param_name, "fp32.pt")
                                     if fp32_fallback_dir is not None else os.path.join(param_dir, "fp32.pt"))
                    if (state_name == "fp32" and local_tensor.numel() < expected_numel
                            and os.path.isfile(fallback_path)):
                        if fp32_fallback_tensor is None:
                            fp32_fallback_state = torch.load(fallback_path, map_location='cpu', weights_only=False)
                            fp32_fallback_tensor = fp32_fallback_state[PARAM]
                        if affine_map is not None:
                            local_tensor = affine_map.extract(fp32_fallback_tensor, ep_rank)
                        else:
                            start = ep_rank * metadata['num_local_experts']
                            end = start + metadata['num_local_experts']
                            local_tensor = fp32_fallback_tensor[start:end]
                        ep_tensors[ep_rank] = local_tensor.reshape(expected_shape)
                        continue
                    raise RuntimeError(f"AutoEP {state_name} fragment size mismatch for {param_name}, "
                                       f"EP rank {ep_rank}: got {local_tensor.numel()}, expected {expected_numel}.")
                ep_tensors[ep_rank] = local_tensor.reshape(expected_shape)

            if affine_map is not None:
                full_tensor = affine_map.rebuild(ep_tensors)
            else:
                full_tensor = torch.cat([ep_tensors[rank] for rank in range(ep_size)], dim=0)
            if full_tensor.shape[0] != num_experts:
                raise RuntimeError(f"AutoEP consolidated expert count mismatch for {param_name}: "
                                   f"got {full_tensor.shape[0]}, expected {num_experts}.")

            torch.save({
                PARAM: full_tensor,
                CAT_DIM: 0,
                EP_IS_EXPERT_PARAM: True,
                EP_NUM_EXPERTS: num_experts,
            }, os.path.join(param_dir, f"{state_name}.pt"))

        step_values = []
        for dp_rank in range(dp_degree):
            step_path = _zero12_fragment_path(temp_dir, param_name, "step", dp_rank)
            if os.path.isfile(step_path):
                step_values.append(torch.load(step_path, map_location='cpu', weights_only=False))
        if not step_values:
            raise RuntimeError(f"Missing AutoEP optimizer step for {param_name}.")
        if not all(_zero12_values_equal(step_values[0], value) for value in step_values[1:]):
            raise RuntimeError(f"Inconsistent AutoEP optimizer steps for {param_name}.")
        torch.save(step_values[0], os.path.join(param_dir, "step.pt"))


def consolidate_autoep_expert_files(checkpoint_dir,
                                    output_dir,
                                    autoep_layers_metadata,
                                    fp32_fallback_param_names=None,
                                    fp32_fallback_dir=None):
    """Consolidate per-expert checkpoint files into full-expert universal format.

    For each AutoEP layer, loads all per-expert files, stacks into
    [E_total, H, D] tensors, and saves in universal checkpoint format.

    Args:
        checkpoint_dir: Path to DeepSpeed checkpoint directory.
        output_dir: Path to universal checkpoint output directory.
        autoep_layers_metadata: AutoEP metadata list from main checkpoint.
        fp32_fallback_param_names: Parameters whose ZeRO FP32 fragments are incomplete.
        fp32_fallback_dir: Temporary directory for fallback tensors consumed by ZeRO consolidation.

    Raises:
        FileNotFoundError: If expected expert files are missing.
        NotImplementedError: If multiple mp_rank files match one (layer, expert).
        RuntimeError: If metadata is missing or malformed.
    """
    if autoep_layers_metadata is None:
        raise RuntimeError("AutoEP metadata is missing from checkpoint. Cannot consolidate "
                           "expert files without ds_autoep_layers metadata.")
    if not isinstance(autoep_layers_metadata, list):
        raise RuntimeError(f"AutoEP metadata is malformed: expected list, got "
                           f"{type(autoep_layers_metadata).__name__}")

    for layer_info in autoep_layers_metadata:
        moe_layer_id = layer_info['moe_layer_id']
        num_experts = layer_info['num_experts']
        prefix = layer_info['expert_key_prefix']
        placement = layer_info.get(AUTOEP_EXPERT_PLACEMENT)
        if placement is not None:
            validate_autoep_placement_descriptor(placement)
            if placement[AUTOEP_NUM_EXPERTS] != num_experts:
                raise RuntimeError(f"AutoEP expert placement disagrees with num_experts for {prefix}.")

        for wname in ('w1', 'w2', 'w3'):
            param_name = f"{prefix}.{wname}"
            save_fp32 = fp32_fallback_param_names is None or param_name in fp32_fallback_param_names
            expert_tensors = [] if save_fp32 else None
            folding_metadata = None
            for global_eid in range(num_experts):
                ckpt_path = resolve_expert_ckpt_path(checkpoint_dir, moe_layer_id, global_eid)
                sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                if folding_metadata is None:
                    folding_metadata = sd.get(FOLDING_METADATA_KEY)
                key = f"{prefix}.{wname}.{global_eid}"
                if key not in sd:
                    raise RuntimeError(f"Expected key '{key}' not found in {ckpt_path}")
                if expert_tensors is not None:
                    expert_tensors.append(sd[key])

            # Stack to full fused tensor [E_total, H, D]
            if expert_tensors is None:
                continue
            full_tensor = torch.stack(expert_tensors, dim=0)

            # Store fallbacks separately so the ZeRO merger writes each final fp32.pt once.
            param_dir = (os.path.join(fp32_fallback_dir, param_name) if fp32_fallback_dir is not None
                         and fp32_fallback_param_names is not None else os.path.join(output_dir, "zero", param_name))
            os.makedirs(param_dir, exist_ok=True)
            universal_state = {
                PARAM: full_tensor,
                CAT_DIM: 0,
                EP_IS_EXPERT_PARAM: True,
                EP_NUM_EXPERTS: num_experts,
            }
            if folding_metadata is not None:
                universal_state[FOLDING_METADATA_KEY] = folding_metadata
            torch.save(universal_state, os.path.join(param_dir, "fp32.pt"))
