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


def record_tp_model_init_args(tp_size, dtype, tp_group, dist_module):
    global _TP_MODEL_INIT_ARGS
    new_args = {
        "tp_size": tp_size,
        "dtype": dtype,
        "tp_group": tp_group,
    }

    if _TP_MODEL_INIT_ARGS is None:
        _TP_MODEL_INIT_ARGS = new_args
        return

    if _TP_MODEL_INIT_ARGS["tp_size"] != tp_size or _TP_MODEL_INIT_ARGS["dtype"] != dtype:
        raise ValueError("Conflicting tp_model_init arguments detected across multiple calls.")

    existing_group = _TP_MODEL_INIT_ARGS.get("tp_group")
    if existing_group is None and tp_group is None:
        return
    if (existing_group is None) != (tp_group is None):
        raise ValueError("Conflicting tp_model_init arguments detected across multiple calls.")

    existing_group_size = tp_group_world_size(existing_group, dist_module)
    new_group_size = tp_group_world_size(tp_group, dist_module)
    if existing_group_size != new_group_size:
        raise ValueError("Conflicting tp_model_init arguments detected across multiple calls.")


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


def merge_tp_model_init_into_config(config_dict: dict, mpu, mesh_param, dist_module):
    if _TP_MODEL_INIT_ARGS is None:
        return

    tp_size = _TP_MODEL_INIT_ARGS["tp_size"]
    dtype = _TP_MODEL_INIT_ARGS["dtype"]
    tp_group = _TP_MODEL_INIT_ARGS["tp_group"]

    if tp_group is not None and mpu is not None:
        raise ValueError("tp_model_init provided tp_group; deepspeed.initialize must not receive mpu.")
    if tp_group is None and mpu is None and mesh_param is None:
        raise ValueError("tp_model_init did not provide tp_group; deepspeed.initialize requires mpu or mesh_param.")

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

    if tp_group is not None:
        config_tp_group = tp_config.get("tp_group")
        if config_tp_group is not None and config_tp_group is not tp_group:
            raise ValueError("Conflicting tensor_parallel.tp.tp_group in config and tp_model_init.")
        tp_config["tp_group"] = tp_group

        tp_group_size = tp_group_world_size(tp_group, dist_module)
        if tp_group_size is not None and tp_group_size != tp_size:
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
