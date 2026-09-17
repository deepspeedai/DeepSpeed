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

from deepspeed.utils import logger

from .constants import (
    AUTOEP_EP_SIZE,
    AUTOEP_EXPERT_KEY_PREFIX,
    AUTOEP_NUM_EXPERTS,
    AUTOEP_NUM_LOCAL_EXPERTS,
    AUTOEP_ZERO12_REQUIRED_FIELDS,
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


def _state_entry(state, param_id):
    """Get optimizer state entry by param id, handling int/str key variants."""
    if param_id in state:
        return state[param_id]

    pid_str = str(param_id)
    if pid_str in state:
        return state[pid_str]

    if isinstance(param_id, str):
        try:
            pid_int = int(param_id)
        except ValueError:
            return None
        return state.get(pid_int)

    return None


def _ordered_param_ids(optim_sd):
    """Return optimizer param ids in param_groups order, deduplicated."""
    ordered = []
    seen = set()
    for group in optim_sd.get('param_groups', []):
        for param_id in group.get('params', []):
            key = str(param_id)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(param_id)

    if ordered:
        return ordered

    # Fallback for unexpected optimizer formats.
    state = optim_sd.get('state', {})
    return list(state.keys())


def _param_name_to_id(optim_sd):
    """Build optional mapping from parameter name to optimizer param id."""
    mapping = {}
    for group in optim_sd.get('param_groups', []):
        params = group.get('params', [])
        param_names = group.get('param_names', None)
        if not isinstance(param_names, list):
            continue
        if len(param_names) != len(params):
            continue
        for param_id, param_name in zip(params, param_names):
            mapping[param_name] = param_id
    return mapping


def _is_expert_optimizer_state(param_state, num_local):
    for state_key in ('exp_avg', 'exp_avg_sq'):
        tensor = param_state.get(state_key)
        if tensor is None:
            continue
        if tensor.dim() == 3 and tensor.shape[0] == num_local:
            return True
    return False


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

        for field in (AUTOEP_NUM_EXPERTS, AUTOEP_NUM_LOCAL_EXPERTS, AUTOEP_EP_SIZE):
            value = layer_info[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RuntimeError(f"AutoEP {field} must be a positive integer, got {value!r}.")

        num_experts = layer_info[AUTOEP_NUM_EXPERTS]
        num_local_experts = layer_info[AUTOEP_NUM_LOCAL_EXPERTS]
        ep_size = layer_info[AUTOEP_EP_SIZE]
        if num_experts != num_local_experts * ep_size:
            raise RuntimeError(f"AutoEP expert count mismatch for {prefix}: num_experts={num_experts}, "
                               f"num_local_experts={num_local_experts}, ep_size={ep_size}.")

        normalized = {
            'num_experts': num_experts,
            'num_local_experts': num_local_experts,
            'ep_size': ep_size,
        }
        for weight_name in ('w1', 'w2', 'w3'):
            param_name = f"{prefix}.{weight_name}"
            if param_name in param_info:
                raise RuntimeError(f"Duplicate AutoEP expert parameter metadata for {param_name}.")
            param_info[param_name] = normalized

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


def _zero12_values_equal(left, right):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    return left == right


def consolidate_autoep_zero12_expert_states(temp_dir, output_dir, expert_param_info, slice_shapes, dp_degree,
                                            tp_degree, use_data_before_expert_parallel):
    """Consolidate AutoEP expert FP32 and Adam states from ZeRO-1/2 fragments."""
    if tp_degree != 1:
        raise NotImplementedError("ZeRO-1/2 Universal Checkpoint conversion for AutoEP with tensor parallelism "
                                  "is not supported.")

    for param_name, metadata in expert_param_info.items():
        if param_name not in slice_shapes:
            raise RuntimeError(f"AutoEP expert parameter {param_name} is missing from checkpoint parameter shapes.")

        ep_size = metadata['ep_size']
        num_experts = metadata['num_experts']
        num_local_experts = metadata['num_local_experts']

        local_shape = tuple(slice_shapes[param_name])
        if not local_shape or local_shape[0] != num_local_experts:
            raise RuntimeError(f"AutoEP local shape mismatch for {param_name}: shape={local_shape}, "
                               f"num_local_experts={num_local_experts}.")

        param_dir = os.path.join(output_dir, "zero", param_name)
        os.makedirs(param_dir, exist_ok=True)

        for state_name in ('fp32', 'exp_avg', 'exp_avg_sq'):
            ep_tensors = []
            for ep_rank in range(ep_size):
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

                if not fragments:
                    raise RuntimeError(f"Missing AutoEP {state_name} fragments for {param_name}, EP rank {ep_rank}.")

                local_tensor = torch.cat(fragments, dim=0)
                expected_numel = torch.Size(local_shape).numel()
                if local_tensor.numel() != expected_numel:
                    raise RuntimeError(f"AutoEP {state_name} fragment size mismatch for {param_name}, "
                                       f"EP rank {ep_rank}: got {local_tensor.numel()}, expected {expected_numel}.")
                ep_tensors.append(local_tensor.reshape(local_shape))

            full_tensor = torch.cat(ep_tensors, dim=0)
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


def consolidate_autoep_expert_files(checkpoint_dir, output_dir, autoep_layers_metadata):
    """Consolidate per-expert checkpoint files into full-expert universal format.

    For each AutoEP layer, loads all per-expert files, stacks into
    [E_total, H, D] tensors, and saves in universal checkpoint format.

    Args:
        checkpoint_dir: Path to DeepSpeed checkpoint directory.
        output_dir: Path to universal checkpoint output directory.
        autoep_layers_metadata: AutoEP metadata list from main checkpoint.

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

        for wname in ('w1', 'w2', 'w3'):
            expert_tensors = []
            folding_metadata = None
            for global_eid in range(num_experts):
                ckpt_path = resolve_expert_ckpt_path(checkpoint_dir, moe_layer_id, global_eid)
                sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                if folding_metadata is None:
                    folding_metadata = sd.get(FOLDING_METADATA_KEY)
                key = f"{prefix}.{wname}.{global_eid}"
                if key not in sd:
                    raise RuntimeError(f"Expected key '{key}' not found in {ckpt_path}")
                expert_tensors.append(sd[key])

            # Stack to full fused tensor [E_total, H, D]
            full_tensor = torch.stack(expert_tensors, dim=0)

            # Save in universal format
            param_name = f"{prefix}.{wname}"
            param_dir = os.path.join(output_dir, "zero", param_name)
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


def consolidate_autoep_optimizer_states(checkpoint_dir, output_dir, autoep_layers_metadata, ep_size):
    """Consolidate expert optimizer states from expp_rank files into universal format.

    Loads optimizer states from all expp_rank_*_optim_states.pt files,
    extracts per-expert-parameter states (exp_avg, exp_avg_sq, etc.),
    concatenates along the expert dimension (dim 0) to form full
    [E_total, H, D] optimizer states, and saves alongside the model
    parameter in universal format.

    Args:
        checkpoint_dir: Path to DeepSpeed checkpoint directory.
        output_dir: Path to universal checkpoint output directory.
        autoep_layers_metadata: AutoEP metadata list from main checkpoint.
        ep_size: Expert parallel world size (number of expp_rank files to load).

    Raises:
        FileNotFoundError: If expected optimizer state files are missing.
        RuntimeError: If expert parameter states cannot be extracted.
    """
    if autoep_layers_metadata is None:
        raise RuntimeError("AutoEP metadata is missing. Cannot consolidate optimizer states.")

    # Load all expp_rank optimizer states
    optim_states = []
    for rank in range(ep_size):
        pattern = os.path.join(checkpoint_dir, f'expp_rank_{rank}_mp_rank_*_optim_states.pt')
        matches = glob.glob(pattern)
        if not matches:
            # No optimizer state files (e.g., ZeRO handles optimizer differently)
            return
        optim_path = matches[0]
        sd = torch.load(optim_path, map_location='cpu', weights_only=False)
        optim_states.append(sd)

    if not optim_states:
        return

    # Extract optimizer state dict
    optim_sd = optim_states[0].get('optimizer')
    if optim_sd is None:
        logger.warning("AutoEP per-expert optimizer checkpoint has no optimizer payload; ZeRO-1/2 checkpoints "
                       "must be consolidated from their ZeRO optimizer shards instead.")
        return

    state = optim_sd.get('state', {})

    if not state:
        return

    ordered_param_ids = _ordered_param_ids(optim_sd)
    name_to_param_id = _param_name_to_id(optim_sd)
    consumed_param_ids = set()

    # For each AutoEP layer, extract and consolidate optimizer states
    for layer_info in autoep_layers_metadata:
        prefix = layer_info['expert_key_prefix']
        num_experts = layer_info['num_experts']
        num_local = layer_info['num_local_experts']
        layer_param_ids = {}

        # If optimizer state carries param names, map weights by exact identity.
        for wname in ('w1', 'w2', 'w3'):
            param_name = f"{prefix}.{wname}"
            param_id = name_to_param_id.get(param_name)
            if param_id is None:
                continue
            layer_param_ids[wname] = param_id
            consumed_param_ids.add(str(param_id))

        # Fallback: consume expert-like params in optimizer param_groups order.
        missing_wnames = [w for w in ('w1', 'w2', 'w3') if w not in layer_param_ids]
        if missing_wnames:
            candidates = []
            for param_id in ordered_param_ids:
                if str(param_id) in consumed_param_ids:
                    continue
                param_state = _state_entry(state, param_id)
                if param_state is None:
                    continue
                if not _is_expert_optimizer_state(param_state, num_local):
                    continue
                candidates.append(param_id)

            for wname, param_id in zip(missing_wnames, candidates):
                layer_param_ids[wname] = param_id
                consumed_param_ids.add(str(param_id))

        for wname in ('w1', 'w2', 'w3'):
            param_name = f"{prefix}.{wname}"
            param_dir = os.path.join(output_dir, "zero", param_name)
            os.makedirs(param_dir, exist_ok=True)
            param_id = layer_param_ids.get(wname)
            if param_id is None:
                continue

            # Consolidate optimizer states for this specific expert parameter id.
            for state_key in ('exp_avg', 'exp_avg_sq'):
                rank_tensors = []

                for rank in range(ep_size):
                    rank_optim_sd = optim_states[rank].get('optimizer', {})
                    rank_state = rank_optim_sd.get('state', {})
                    param_state = _state_entry(rank_state, param_id)
                    if param_state is None:
                        rank_tensors = []
                        break
                    tensor = param_state.get(state_key)
                    if tensor is None:
                        rank_tensors = []
                        break
                    if tensor.dim() != 3 or tensor.shape[0] != num_local:
                        rank_tensors = []
                        break
                    rank_tensors.append(tensor)

                if len(rank_tensors) == ep_size:
                    full_tensor = torch.cat(rank_tensors, dim=0)
                    torch.save(
                        {
                            PARAM: full_tensor,
                            CAT_DIM: 0,
                            EP_IS_EXPERT_PARAM: True,
                            EP_NUM_EXPERTS: num_experts,
                        }, os.path.join(param_dir, f"{state_key}.pt"))
