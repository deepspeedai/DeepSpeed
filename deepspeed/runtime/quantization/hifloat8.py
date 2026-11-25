# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Resolve training module selections and pass their recipes to TorchAO."""

from fnmatch import fnmatchcase
import re

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.moe.ep_experts import GroupedExperts
from deepspeed.runtime.quantization.config import HIFLOAT8
from deepspeed.utils import logger


def _select_modules(model, config):
    patterns = config["module_name_patterns"]
    min_numel = config["min_numel"]
    candidates = {}
    modules = dict(model.named_modules())
    for name, module in modules.items():
        if patterns and not any(fnmatchcase(name, pattern) for pattern in patterns):
            continue
        if type(module) is torch.nn.Linear and module.weight.numel() >= min_numel:
            candidates[name] = module
        elif isinstance(module, GroupedExperts):
            if min(module.w1[0].numel(), module.w2[0].numel(), module.w3[0].numel()) >= min_numel:
                candidates[name] = module

    if not candidates:
        raise RuntimeError("hifloat8 module selection matched no nn.Linear or AutoEP GroupedExperts modules")
    unmatched = [pattern for pattern in patterns if not any(fnmatchcase(name, pattern) for name in candidates)]
    if unmatched:
        raise RuntimeError(f"hifloat8 patterns matched no eligible modules: {unmatched}")

    overrides = config.get("module_configs") or {}
    regexes = {key: re.compile(key[3:]) for key in overrides if key.startswith("re:")}
    for key in overrides:
        if key != "_default" and key not in candidates:
            if key not in regexes or not any(regexes[key].fullmatch(name) for name in candidates):
                raise RuntimeError(f"hifloat8 module_configs selector matched no eligible modules: {key!r}")

    groups = {}
    for name, module in candidates.items():
        key = _module_recipe_key(name, overrides, regexes)
        override = overrides.get(key, {} if patterns else None)
        if override is None:
            continue
        policy = dict(config.get("config") or {})
        policy.update(override)
        _, linears, experts = groups.setdefault(key, (policy, [], []))
        if type(module) is torch.nn.Linear:
            linears.append(name)
        else:
            if module.gate_up_impl == "fused":
                raise RuntimeError("hifloat8 experts require separate gate/up projections")
            parent = modules[name.rpartition(".")[0]]
            if getattr(parent, "experts", None) is module:
                if getattr(parent, "overlap_weight_grad", False) or getattr(parent, "dw_staging", None) is not None:
                    raise RuntimeError(f"HiFloat8 experts '{name}' do not support deferred or staged weight gradients")
            experts.append(name)

    count = sum(len(linears) + len(experts) for _, linears, experts in groups.values())
    if not count:
        raise RuntimeError("hifloat8 module selection matched no nn.Linear or AutoEP GroupedExperts modules")
    expected = config["expected_module_count"]
    if expected is not None and count != expected:
        raise RuntimeError(f"hifloat8 expected_module_count={expected}, selected {count} modules")
    return groups.values()


def _module_recipe_key(name, overrides, regexes):
    if name in overrides:
        return name
    for key, regex in regexes.items():
        if regex.fullmatch(name):
            return key
    return "_default"


def _hifloat8_recipe(config_class, options, linear, grouped):
    # The official AO config defaults to native ranges. Preserve the prior
    # DeepSpeed recipe for omitted fields, including in existing JSON files.
    policy = {"input_dst_type_max": 15, "weight_dst_type_max": 15, "grad_dst_type_max": 224}
    policy.update(options or {})
    return config_class.from_dict({"linear": policy if linear else None, "grouped": policy if grouped else None})


def configure_hifloat8(model, config, device):
    """Convert the selected modules in one pass after AutoEP and device placement."""
    accelerator = get_accelerator()
    module_configs = {}
    selected_linears, selected_experts = [], []
    for options, linears, experts in _select_modules(model, config):
        try:
            config_class = accelerator.get_training_quantization_config(HIFLOAT8)
            if config_class is None or torch.device(device).type != accelerator.device_name():
                raise RuntimeError(f"No hifloat8 training backend for {device}")
            recipe = _hifloat8_recipe(config_class, options, bool(linears), bool(experts))
        except (ImportError, RuntimeError, ValueError, TypeError) as error:
            raise RuntimeError(f"hifloat8 selected {len(linears)} Linear and {len(experts)} AutoEP modules, "
                               f"but its backend or recipe failed before conversion: {error}") from error

        leaves = [f"{name}.grouped_mm" if name else "grouped_mm" for name in experts]
        module_configs.update(dict.fromkeys((*linears, *leaves), recipe))
        selected_linears.extend(linears)
        selected_experts.extend(experts)

    from torchao.quantization import FqnToConfig, quantize_

    quantize_(model, FqnToConfig(module_configs), filter_fn=None)
    logger.info("Training quantization hifloat8 converted %d Linear and %d AutoEP modules", len(selected_linears),
                len(selected_experts))
    return tuple(selected_linears), tuple(selected_experts)
