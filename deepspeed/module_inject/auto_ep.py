# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP: Automatic Expert Parallelism for MoE models.

Phase 3: MoE layer detection and structural validation.
Phase 5: Layer replacement (replace_moe_layer filled in).
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import Literal

import torch
import torch.nn as nn

from deepspeed.module_inject.auto_ep_config import (
    fill_autoep_config_from_hf,
    AutoEPConfig,
    MoELayerSpec,
    MoEModelPreset,
)
from deepspeed.module_inject.auto_ep_presets.base import ForwardContract
from deepspeed.module_inject.auto_ep_presets.registry import (
    apply_config_overrides,
    get_preset_adapter,
    preset_name_for_hf_model_type,
    resolve_preset_candidates,
    unsupported_preset_for_hf_model_type,
)
from deepspeed.moe.fused_expert_layout import classify_fused_gate_up_layout
from deepspeed.utils import logger


def _remove_transformers_output_capture_hooks(model: nn.Module) -> int:
    """Remove HF output-capturing hooks so they can be reinstalled after AutoEP conversion."""
    removed = 0
    for module in model.modules():
        hooks = getattr(module, "_forward_hooks", None)
        if not hooks:
            continue

        for hook_id, hook in list(hooks.items()):
            if getattr(hook, "__name__", "") != "output_capturing_hook":
                continue
            del hooks[hook_id]
            removed += 1
            hooks_with_kwargs = getattr(module, "_forward_hooks_with_kwargs", None)
            if hooks_with_kwargs is not None:
                hooks_with_kwargs.pop(hook_id, None)
            hooks_always_called = getattr(module, "_forward_hooks_always_called", None)
            if hooks_always_called is not None:
                hooks_always_called.pop(hook_id, None)
    return removed


def _is_known_hf_model_type(model_type: str | None) -> bool:
    if model_type is None:
        return False
    return (preset_name_for_hf_model_type(model_type) is not None
            or unsupported_preset_for_hf_model_type(model_type) is not None)


def _raise_if_duplicate_moe_specs(specs: list[MoELayerSpec]) -> None:
    by_module: dict[str, list[MoELayerSpec]] = {}
    for spec in specs:
        by_module.setdefault(spec.moe_module_name, []).append(spec)

    duplicates = {name: matches for name, matches in by_module.items() if len(matches) > 1}
    if not duplicates:
        return

    details = "; ".join(f"{name}: {', '.join(spec.model_family for spec in matches)}"
                        for name, matches in sorted(duplicates.items()))
    raise ValueError("AutoEP detection is ambiguous and produced multiple replacement specs for the same "
                     f"MoE module(s): {details}. Set expert_parallel.preset_model or provide custom "
                     "AutoEP patterns so each MoE module matches exactly one preset.")


def _has_3d_expert_params(module: nn.Module, preset: MoEModelPreset) -> bool:
    """Check if module stores expert weights as 3D parameter tensors (transformers 5.0.0+).

    Returns True if the module has a parameter named preset.expert_w1 (e.g., "gate_up_proj")
    with 3 dimensions (num_experts, ..., ...).
    """
    w1_name = preset.expert_w1
    param = getattr(module, w1_name, None)
    if param is None:
        return False
    if isinstance(param, nn.Parameter) or isinstance(param, torch.Tensor):
        return param.ndim == 3
    return False


def _get_num_experts_from_config(model_config, preset: MoEModelPreset) -> int | None:
    """Extract num_experts from model.config using the preset's attribute name."""
    return getattr(model_config, preset.num_experts_attr, None)


def _get_top_k_from_config(model_config, preset: MoEModelPreset) -> int | None:
    """Extract top_k from model.config using the preset's attribute name."""
    return getattr(model_config, preset.top_k_attr, None)


def _as_finite_float(value, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")

    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")
    return value


def _resolve_route_scale(config: AutoEPConfig, model_config) -> float:
    """Resolve the single scale applied by TokenChoiceTopKRouter."""
    routed_scaling_factor = config.routed_scaling_factor

    if routed_scaling_factor != "auto":
        route_scale = _as_finite_float(routed_scaling_factor, "routed_scaling_factor")
        if config.route_scale != 1.0:
            logger.warning("AutoEP: routed_scaling_factor=%s overrides route_scale=%s.", routed_scaling_factor,
                           config.route_scale)
        return route_scale

    cfg_routed_scaling_factor = getattr(model_config, 'routed_scaling_factor', None)
    if cfg_routed_scaling_factor is not None:
        route_scale = _as_finite_float(cfg_routed_scaling_factor, "model.config.routed_scaling_factor")
        if config.route_scale != 1.0:
            logger.warning("AutoEP: model.config.routed_scaling_factor=%s overrides route_scale=%s.",
                           cfg_routed_scaling_factor, config.route_scale)
        return route_scale

    return _as_finite_float(config.route_scale, "route_scale")


def _detect_expert_storage(experts_module: nn.Module, preset: MoEModelPreset) -> Literal["fused_3d", "module_list"]:
    """Determine whether experts are stored as fused 3D tensors or nn.ModuleList."""
    if _has_3d_expert_params(experts_module, preset):
        return "fused_3d"
    if isinstance(experts_module, nn.ModuleList):
        return "module_list"
    # Check children for 3D params as fallback
    for name, param in experts_module.named_parameters(recurse=False):
        if param.ndim == 3:
            return "fused_3d"
    return "module_list"


def _infer_hidden_and_ffn_size(
    experts_module: nn.Module,
    preset: MoEModelPreset,
    storage: Literal["fused_3d", "module_list"],
    num_experts: int,
) -> tuple[int, int]:
    """Infer hidden_size and ffn_hidden_size from expert weight shapes."""
    if storage == "fused_3d":
        w1_param = getattr(experts_module, preset.expert_w1, None)
        w2_param = getattr(experts_module, preset.expert_w2, None)
        if w1_param is not None and w2_param is not None:
            if preset.expert_w3 is None:
                layout = classify_fused_gate_up_layout(tuple(w1_param.shape), tuple(w2_param.shape))
                if layout is None:
                    raise ValueError("expert_w3=None expects fused gate+up weights with either "
                                     f"[E, 2*ffn, hidden]/[E, hidden, ffn] or [E, hidden, 2*ffn]/[E, ffn, hidden], "
                                     f"but got {preset.expert_w1}={tuple(w1_param.shape)} and "
                                     f"{preset.expert_w2}={tuple(w2_param.shape)}.")
                hidden_size = layout.hidden_size
                ffn_hidden_size = layout.ffn_hidden_size
            else:
                # Separate gate and up: w1 shape is [E, ffn, hidden]
                w3_param = getattr(experts_module, preset.expert_w3, None)
                if w3_param is None:
                    raise ValueError(f"expert_w3='{preset.expert_w3}' is set but no such weight "
                                     f"exists on experts module.")
                hidden_size = w1_param.shape[2]
                ffn_hidden_size = w1_param.shape[1]
            return hidden_size, ffn_hidden_size
    elif storage == "module_list":
        # Legacy: individual expert modules
        if isinstance(experts_module, nn.ModuleList) and len(experts_module) > 0:
            expert0 = experts_module[0]
            w1 = getattr(expert0, preset.expert_w1, None)
            if w1 is None:
                # Try weight attribute for nn.Linear
                for name, child in expert0.named_children():
                    if preset.expert_w1 in name:
                        w1 = child.weight if hasattr(child, 'weight') else None
                        break
            if w1 is not None:
                if isinstance(w1, nn.Linear):
                    return w1.in_features, w1.out_features
                elif isinstance(w1, (nn.Parameter, torch.Tensor)):
                    if w1.ndim == 2:
                        return w1.shape[1], w1.shape[0]

    raise ValueError(f"Could not infer hidden_size/ffn_hidden_size from experts module "
                     f"with storage={storage}, preset.expert_w1={preset.expert_w1}")


def _detect_forward_contract(
    moe_module: nn.Module,
    router_module: nn.Module,
) -> ForwardContract:
    """Detect the forward contract for router logits capture.

    Returns:
        ForwardContract with router-logit return and capture metadata.
    """
    # Check for OutputRecorder on the model (transformers 5.0.0 pattern)
    # Look for _can_record_outputs attribute on parent modules
    capture_target: Literal["moe_block", "router", "none"] = "none"
    capture_index: int | None = None
    capture_layer_name: str | None = None
    return_router_logits = False

    # Check for OutputRecorder pattern on router class
    router_class = type(router_module)
    if hasattr(router_class, '_can_record_outputs'):
        capture_target = "router"
        record_config = router_class._can_record_outputs
        if isinstance(record_config, dict):
            for key, val in record_config.items():
                if isinstance(val, dict):
                    capture_index = val.get('index', 0)
                    capture_layer_name = val.get('layer_name', None)
                else:
                    capture_index = 0
        elif isinstance(record_config, (list, tuple)):
            capture_index = 0
        logger.debug(f"Detected OutputRecorder on router class {router_class.__name__}: "
                     f"index={capture_index}, layer_name={capture_layer_name}")

    # Check if MoE block has tuple return contract (legacy transformers)
    if hasattr(moe_module, '_can_record_outputs'):
        record_config = moe_module._can_record_outputs
        if record_config:
            capture_target = "moe_block"
            return_router_logits = True
            if isinstance(record_config, dict):
                for key, val in record_config.items():
                    if isinstance(val, dict):
                        capture_index = val.get('index', None)
                    elif isinstance(val, int):
                        capture_index = val

    return ForwardContract(
        return_router_logits=return_router_logits,
        capture_target=capture_target,
        capture_index=capture_index,
        capture_layer_name=capture_layer_name,
    )


class AutoEP:
    """Automatic Expert Parallelism: detect and replace MoE layers."""

    def __init__(self, model: nn.Module, config: AutoEPConfig) -> None:
        self.model = model
        self.config = config
        self.model_config = getattr(model, 'config', None)
        self._retargeted_transformers_output_recorders: set[str] = set()
        fill_autoep_config_from_hf(self.config, self.model_config)

    def ep_parser(self) -> list[MoELayerSpec]:
        """Traverse model and detect MoE layers. Returns list of MoELayerSpec."""
        specs = []

        # Determine which preset(s) to use
        presets_to_try = self._resolve_presets()

        for preset_name, preset in presets_to_try:
            adapter = get_preset_adapter(preset.preset_adapter)
            pattern = re.compile(preset.moe_layer_pattern)

            for module_name, module in self.model.named_modules():
                if not pattern.fullmatch(module_name):
                    continue

                # Structural validation: check for experts child
                experts_child = getattr(module, preset.experts_pattern, None)
                if experts_child is None:
                    logger.debug(
                        "Skipping %s: pattern matched but no '%s' child (likely dense FFN)",
                        module_name,
                        preset.experts_pattern,
                    )
                    continue

                expert_layout = adapter.resolve_expert_layout(experts_child, preset)

                # Accept both: nn.ModuleList (legacy) and Experts class (transformers 5.0.0+)
                has_expert_params = (isinstance(experts_child, nn.ModuleList)
                                     or _has_3d_expert_params(experts_child, expert_layout))
                if not has_expert_params:
                    logger.debug(
                        "Skipping %s: '%s' child exists but has no expert parameters",
                        module_name,
                        preset.experts_pattern,
                    )
                    continue

                # Check for router
                router_child = getattr(module, preset.router_pattern, None)
                if router_child is None:
                    logger.debug(
                        "Skipping %s: no router child '%s'",
                        module_name,
                        preset.router_pattern,
                    )
                    continue

                # Detect storage format
                storage = _detect_expert_storage(experts_child, expert_layout)

                # Get num_experts and top_k from config or weights
                num_experts = None
                top_k = None

                if self.model_config is not None:
                    num_experts = _get_num_experts_from_config(self.model_config, preset)
                    top_k = _get_top_k_from_config(self.model_config, preset)

                # Validate/derive from router weight shape
                router_weight = getattr(router_child, 'weight', None)
                if router_weight is not None and router_weight.ndim == 2:
                    num_experts_from_weight = router_weight.shape[0]
                    hidden_from_weight = router_weight.shape[1]
                    if num_experts is not None and num_experts != num_experts_from_weight:
                        raise ValueError(f"Config num_experts={num_experts} mismatches router weight "
                                         f"shape {router_weight.shape} (expected {num_experts_from_weight}) "
                                         f"in layer '{module_name}'")
                    num_experts = num_experts_from_weight

                if num_experts is None:
                    raise ValueError(f"Could not determine num_experts for layer '{module_name}'. "
                                     f"Set model.config.{preset.num_experts_attr} or use a preset.")

                # Override top_k from config if user specified
                if isinstance(self.config.top_k, int):
                    top_k = self.config.top_k
                elif top_k is None:
                    raise ValueError(f"Could not determine top_k for layer '{module_name}'. "
                                     f"Set model.config.{preset.top_k_attr} or config top_k.")

                # Infer hidden sizes
                try:
                    hidden_size, ffn_hidden_size = _infer_hidden_and_ffn_size(experts_child, expert_layout, storage,
                                                                              num_experts)
                except ValueError as e:
                    if self._requires_selected_preset_detection():
                        raise ValueError(f"AutoEP: preset '{preset_name}' matched layer '{module_name}' "
                                         f"with router and experts, but shape inference failed: {e}") from e
                    logger.warning(f"Skipping {module_name}: {e}")
                    continue

                # Cross-validate hidden_size with router
                if router_weight is not None and router_weight.ndim == 2:
                    if hidden_size != router_weight.shape[1]:
                        raise ValueError(f"hidden_size={hidden_size} from expert weights mismatches "
                                         f"router weight dim={router_weight.shape[1]} in '{module_name}'")

                # Validate top_k <= num_experts
                if top_k > num_experts:
                    raise ValueError(f"top_k={top_k} exceeds num_experts={num_experts} "
                                     f"in layer '{module_name}'")

                # Resolve score_func
                if self.config.score_func != "auto":
                    score_func = self.config.score_func
                else:
                    # Check model config for scoring_func attribute
                    cfg_score = getattr(self.model_config, 'scoring_func', None)
                    if cfg_score in ("softmax", "sigmoid"):
                        score_func = cfg_score
                    else:
                        score_func = preset.score_func

                # Resolve score_apply
                if self.config.score_apply != "auto":
                    score_apply = self.config.score_apply
                else:
                    score_apply = preset.score_apply

                route_norm = adapter.resolve_route_norm(self.config, preset, self.model_config)

                route_scale = _resolve_route_scale(self.config, self.model_config)

                group_routing = adapter.resolve_group_routing(self.config, self.model_config)

                # Check gate bias
                gate_bias = preset.gate_bias
                if router_weight is not None:
                    gate_bias = getattr(router_child, 'bias', None) is not None

                forward_contract = adapter.adjust_forward_contract(_detect_forward_contract(module, router_child))

                # Check shared experts
                has_shared = False
                shared_name = ""
                shared_gate_name = ""
                if preset.has_shared_experts and preset.shared_experts_pattern:
                    shared = getattr(module, preset.shared_experts_pattern, None)
                    if shared is not None:
                        has_shared = True
                        shared_name = preset.shared_experts_pattern
                        if preset.shared_experts_gate_pattern:
                            shared_gate = getattr(module, preset.shared_experts_gate_pattern, None)
                            if shared_gate is not None:
                                shared_gate_name = preset.shared_experts_gate_pattern

                # Warn about router stochasticity/precision settings
                if self.model_config is not None:
                    jitter = getattr(self.model_config, 'router_jitter_noise', 0.0)
                    if jitter and jitter > 0:
                        logger.warning(f"Layer {module_name}: model has router_jitter_noise={jitter}, "
                                       f"AutoEP router does not implement jitter.")
                    z_loss = getattr(self.model_config, 'router_z_loss_coef', 0.0)
                    if z_loss and z_loss > 0:
                        logger.warning(f"Layer {module_name}: model has router_z_loss_coef={z_loss}, "
                                       f"AutoEP router does not implement z-loss.")

                spec = MoELayerSpec(
                    moe_module_name=module_name,
                    model_family=preset_name,
                    router_name=preset.router_pattern,
                    experts_name=preset.experts_pattern,
                    expert_storage=storage,
                    expert_w1_name=expert_layout.expert_w1,
                    expert_w2_name=expert_layout.expert_w2,
                    expert_w3_name=expert_layout.expert_w3,
                    num_experts=num_experts,
                    top_k=top_k,
                    hidden_size=hidden_size,
                    ffn_hidden_size=ffn_hidden_size,
                    score_func=score_func,
                    score_apply=score_apply,
                    route_norm=route_norm,
                    gate_bias=gate_bias,
                    return_router_logits=forward_contract.return_router_logits,
                    router_logits_capture_target=forward_contract.capture_target,
                    router_logits_capture_index=forward_contract.capture_index,
                    router_logits_capture_layer_name=forward_contract.capture_layer_name,
                    has_shared_experts=has_shared,
                    shared_experts_name=shared_name,
                    shared_experts_gate_name=shared_gate_name,
                    route_scale=route_scale,
                    num_expert_groups=group_routing.num_expert_groups,
                    num_limited_groups=group_routing.num_limited_groups,
                    group_score_func=group_routing.group_score_func,
                    supports_expert_bias=preset.supports_expert_bias,
                    unsupported_router_bias_names=preset.unsupported_router_bias_names,
                    preset_adapter=preset.preset_adapter,
                    router_logits_capture_mode=forward_contract.router_logits_capture_mode,
                    moe_output_shape=forward_contract.moe_output_shape,
                )
                specs.append(spec)
                logger.debug(f"Detected MoE layer: {module_name} (family={preset_name}, "
                             f"experts={num_experts}, top_k={top_k}, storage={storage})")

        if not specs:
            if self._requires_selected_preset_detection():
                self._raise_no_moe_layers_detected(presets_to_try)
            logger.warning("AutoEP: no MoE layers detected in model.")
        else:
            _raise_if_duplicate_moe_specs(specs)

        return specs

    def _replace_moe_layer_without_retarget(
        self,
        spec: MoELayerSpec,
        ep_size: int,
        ep_rank: int,
    ) -> nn.Module:
        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

        # Navigate to the parent module and get the child name
        parts = spec.moe_module_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]
        source_module = getattr(parent, child_name)

        # Create replacement layer
        replacement = AutoEPMoELayer(
            spec=spec,
            source_module=source_module,
            ep_size=ep_size,
            ep_rank=ep_rank,
            config=self.config,
        )

        # Replace in-place on parent
        setattr(parent, child_name, replacement)
        return replacement

    def _retarget_transformers_output_recorders(self, spec: MoELayerSpec, replacement: nn.Module) -> None:
        adapter = get_preset_adapter(spec.preset_adapter)
        adapter.retarget_transformers_output_recorders(
            self.model,
            spec,
            replacement,
            self._retargeted_transformers_output_recorders,
            _remove_transformers_output_capture_hooks,
        )

    def replace_moe_layer(
        self,
        spec: MoELayerSpec,
        ep_size: int,
        ep_rank: int,
    ) -> None:
        """Replace a single MoE module with AutoEPMoELayer in-place on the model."""
        replacement = self._replace_moe_layer_without_retarget(spec, ep_size, ep_rank)
        self._retarget_transformers_output_recorders(spec, replacement)

        logger.info(f"AutoEP: replaced '{spec.moe_module_name}' with AutoEPMoELayer "
                    f"(ep_size={ep_size}, ep_rank={ep_rank}, "
                    f"local_experts={replacement.num_local_experts})")

    def replace_moe_layers(
        self,
        specs: list[MoELayerSpec],
        ep_size: int,
        ep_rank: int,
    ) -> None:
        """Replace multiple MoE modules and batch post-replacement recorder retargeting."""
        replacements: list[tuple[MoELayerSpec, nn.Module]] = []
        for spec in specs:
            replacement = self._replace_moe_layer_without_retarget(spec, ep_size, ep_rank)
            replacements.append((spec, replacement))
            logger.info(f"AutoEP: replaced '{spec.moe_module_name}' with AutoEPMoELayer "
                        f"(ep_size={ep_size}, ep_rank={ep_rank}, "
                        f"local_experts={replacement.num_local_experts})")

        retarget_groups: OrderedDict[tuple[str, str, type], tuple[MoELayerSpec, nn.Module]] = OrderedDict()
        for spec, replacement in replacements:
            retarget_key = (spec.preset_adapter, spec.model_family, replacement.__class__)
            retarget_groups.setdefault(retarget_key, (spec, replacement))

        for spec, replacement in retarget_groups.values():
            self._retarget_transformers_output_recorders(spec, replacement)

    def _apply_config_overrides(self, preset: MoEModelPreset) -> MoEModelPreset:
        return apply_config_overrides(self.config, preset)

    def _requires_selected_preset_detection(self) -> bool:
        """Return whether empty detection should fail for the selected preset."""
        if self.config.preset_model is not None:
            return True
        if self.config.moe_layer_pattern is not None:
            return True
        if self.model_config is None:
            return False
        model_type = getattr(self.model_config, 'model_type', None)
        return _is_known_hf_model_type(model_type)

    def _raise_no_moe_layers_detected(self, presets_to_try: list[tuple[str, MoEModelPreset]]) -> None:
        model_type = getattr(self.model_config, 'model_type', None)
        if self.config.preset_model is not None:
            source = f"preset_model='{self.config.preset_model}'"
        elif self.config.moe_layer_pattern is not None:
            source = f"moe_layer_pattern='{self.config.moe_layer_pattern}'"
        else:
            source = f"model_type='{model_type}'"

        expected = "; ".join(f"{preset_name}: moe_layer_pattern='{preset.moe_layer_pattern}', "
                             f"router='{preset.router_pattern}', experts='{preset.experts_pattern}'"
                             for preset_name, preset in presets_to_try)
        raise ValueError(f"AutoEP: no MoE layers detected for {source}. "
                         f"Expected MoE structure for selected preset(s): {expected}. "
                         "This usually means the selected preset does not match the model implementation, "
                         "or the installed Transformers version exposes a different structure. Choose a matching "
                         "preset, upgrade Transformers, or provide custom AutoEP patterns.")

    def _resolve_presets(self) -> list[tuple[str, MoEModelPreset]]:
        """Determine which preset(s) to use for detection."""
        return resolve_preset_candidates(self.config, self.model_config)
