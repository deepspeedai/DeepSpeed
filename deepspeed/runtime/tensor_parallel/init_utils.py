#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import base64
import os
from typing import Optional, Union

import hjson
import torch

from deepspeed.runtime.config_utils import dict_raise_error_on_duplicate_keys

_TP_MODEL_INIT_ARGS = None


def load_ds_config(config: Union[str, dict]) -> dict:
    if isinstance(config, dict):
        return config
    if isinstance(config, str):
        if os.path.exists(config):
            return hjson.load(open(config, "r"), object_pairs_hook=dict_raise_error_on_duplicate_keys)
        try:
            config_decoded = base64.urlsafe_b64decode(config).decode('utf-8')
            return hjson.loads(config_decoded)
        except (UnicodeDecodeError, AttributeError, ValueError) as exc:
            raise ValueError(
                f"Expected a string path to an existing deepspeed config, or a dictionary or a valid base64. "
                f"Received: {config}") from exc
    raise ValueError(f"Expected a string path to an existing deepspeed config, or a dictionary or a valid base64. "
                     f"Received: {config}")


def _validate_same_tp_model_init_args(existing_args, tp_size, dtype, tp_group, dist_module):
    if existing_args["tp_size"] != tp_size or existing_args["dtype"] != dtype:
        raise ValueError("Conflicting tp_model_init arguments detected across multiple calls.")

    existing_group = existing_args.get("tp_group")
    if existing_group is None and tp_group is None:
        return
    if (existing_group is None) != (tp_group is None):
        raise ValueError("Conflicting tp_model_init arguments detected across multiple calls.")

    existing_group_size = tp_group_world_size(existing_group, dist_module)
    new_group_size = tp_group_world_size(tp_group, dist_module)
    if existing_group_size != new_group_size:
        raise ValueError("Conflicting tp_model_init arguments detected across multiple calls.")


def record_tp_model_init_args(tp_size, dtype, tp_group, dist_module, model=None):
    global _TP_MODEL_INIT_ARGS
    new_args = {
        "tp_size": tp_size,
        "dtype": dtype,
        "tp_group": tp_group,
    }

    if model is not None:
        existing_args = model.__dict__.get("_ds_tp_model_init_args")
        if existing_args is not None:
            _validate_same_tp_model_init_args(existing_args, tp_size, dtype, tp_group, dist_module)
            return
        setattr(model, "_ds_tp_model_init_args", new_args)
        return

    if _TP_MODEL_INIT_ARGS is None:
        _TP_MODEL_INIT_ARGS = new_args
        return

    _validate_same_tp_model_init_args(_TP_MODEL_INIT_ARGS, tp_size, dtype, tp_group, dist_module)


def tp_group_world_size(tp_group, dist_module):
    if tp_group is None or dist_module is None:
        return None
    return dist_module.get_world_size(group=tp_group)


def infer_config_dtype(config_dict: dict) -> Optional[torch.dtype]:
    bf16_config = config_dict.get("bf16", {})
    if isinstance(bf16_config, dict) and bf16_config.get("enabled", False):
        return torch.bfloat16
    fp16_config = config_dict.get("fp16", {})
    if isinstance(fp16_config, dict) and fp16_config.get("enabled", False):
        return torch.float16
    return None


def _find_tp_model_init_args(model):
    if isinstance(model, torch.nn.Module):
        for module in model.modules():
            args = module.__dict__.get("_ds_tp_model_init_args")
            if args is not None:
                return args
    return _TP_MODEL_INIT_ARGS


def merge_tp_model_init_into_config(config_dict: dict, mpu, mesh_param, dist_module, model=None):
    recorded_args = _find_tp_model_init_args(model)
    if recorded_args is None:
        return

    tp_size = recorded_args["tp_size"]
    dtype = recorded_args["dtype"]
    tp_group = recorded_args["tp_group"]

    if tp_group is not None and mpu is not None:
        raise ValueError("tp_model_init provided tp_group; deepspeed.initialize must not receive mpu.")
    from deepspeed.utils import groups
    if tp_group is None and mpu is None and mesh_param is None and tp_size <= 1 and groups.mpu is None \
            and groups.mesh_device is None and not groups._EXPERT_PARALLEL_GROUP:
        # Auto-create TP groups for compatibility with HF Trainer (mpu is not passed). With tp_size > 1, or after
        # an earlier engine, the engine creates its own groups, which may differ from an earlier engine's (#9081).
        groups._init_tp_mesh_device(tensor_model_parallel_size=tp_size)

    tp_section = config_dict.get("tensor_parallel")
    if tp_section is None:
        tp_section = {}
        config_dict["tensor_parallel"] = tp_section

    config_autotp_size = tp_section.get("autotp_size")
    if config_autotp_size is not None and config_autotp_size != tp_size:
        raise ValueError(
            f"Conflicting tensor_parallel.autotp_size in config ({config_autotp_size}) and tp_model_init ({tp_size}).")

    if config_autotp_size is None:
        tp_section["autotp_size"] = tp_size

    tp_config = tp_section.get("tp") or {}
    if not isinstance(tp_config, dict):
        raise ValueError("tensor_parallel.tp must be a dict when provided.")

    config_tp_size = tp_config.get("tp_size")
    if config_tp_size is not None and config_tp_size != tp_size:
        raise ValueError(
            f"Conflicting tensor_parallel.tp.tp_size in config ({config_tp_size}) and tp_model_init ({tp_size}).")
    if config_tp_size is None:
        tp_config["tp_size"] = tp_size

    tp_group_size = tp_group_world_size(tp_group, dist_module)
    if tp_group is not None:
        config_tp_group = tp_config.get("tp_group")
        if config_tp_group is not None and config_tp_group is not tp_group:
            raise ValueError("Conflicting tensor_parallel.tp.tp_group in config and tp_model_init.")
        tp_config["tp_group"] = tp_group

    size_mismatch = tp_group is not None and tp_group_size is not None and tp_group_size != tp_size
    if dist_module is not None and hasattr(dist_module, "is_initialized") and dist_module.is_initialized() \
            and hasattr(dist_module, "all_reduce"):
        from deepspeed.accelerator import get_accelerator
        flag = torch.tensor([int(size_mismatch)], device=get_accelerator().current_device_name())
        dist_module.all_reduce(flag, op=dist_module.ReduceOp.MAX)
        size_mismatch = bool(flag.item())
    if size_mismatch:
        raise ValueError(f"tp_model_init tp_size ({tp_size}) does not match tp_group size ({tp_group_size}).")

    tp_section["tp"] = tp_config

    config_dtype = infer_config_dtype(config_dict)
    if config_dtype is not None and config_dtype != dtype:
        raise ValueError(f"Conflicting dtype: config uses {config_dtype} but tp_model_init requested {dtype}.")

    tp_dtype = tp_section.get("dtype")
    if tp_dtype is not None:
        if isinstance(tp_dtype, str):
            tp_dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            tp_dtype_value = tp_dtype_map.get(tp_dtype.lower())
        else:
            tp_dtype_value = tp_dtype
        if tp_dtype_value is not None and tp_dtype_value != dtype:
            raise ValueError(f"Conflicting tensor_parallel.dtype in config ({tp_dtype}) and tp_model_init ({dtype}).")
