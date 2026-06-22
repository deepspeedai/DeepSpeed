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

from .constants import (
    PARAM,
    CAT_DIM,
    EP_IS_EXPERT_PARAM,
    EP_NUM_EXPERTS,
)


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
        raise NotImplementedError(f"Multiple expert checkpoint files found for layer_{moe_layer_id} "
                                  f"expert_{global_expert_id}: {matches}. Multi-mp_rank expert files "
                                  f"are not yet supported.")
    return matches[0]


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
            for global_eid in range(num_experts):
                ckpt_path = resolve_expert_ckpt_path(checkpoint_dir, moe_layer_id, global_eid)
                sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
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
            torch.save({
                PARAM: full_tensor,
                CAT_DIM: 0,
                EP_IS_EXPERT_PARAM: True,
                EP_NUM_EXPERTS: num_experts,
            }, os.path.join(param_dir, "fp32.pt"))


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
