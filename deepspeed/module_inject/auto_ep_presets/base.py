# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Shared AutoEP preset dataclasses and adapter interface."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, NoReturn

import torch.nn as nn
from packaging.version import InvalidVersion, Version

# Sentinel for "not specified in config, use preset default".
# Unlike None (which means "fused gate+up, no separate w3"), _UNSET means
# the user did not set the field at all. Compare with `is _UNSET`.
_UNSET = object()


def _raise_unsupported_load_balance_coeff(value: object) -> NoReturn:
    raise ValueError(f"load_balance_coeff={value!r} is not supported in this AutoEP build "
                     "(would register expert_bias and route through unsupported "
                     "auxiliary-loss-free load balancing). Set load_balance_coeff to null "
                     "or omit the key.")


@dataclass
class MoEModelPreset:
    """Preset configuration for a known MoE model family."""

    moe_layer_pattern: str
    router_pattern: str
    experts_pattern: str
    expert_storage: Literal["fused_3d", "module_list"]
    expert_w1: str
    expert_w2: str
    expert_w3: str | None
    num_experts_attr: str
    top_k_attr: str
    score_func: Literal["softmax", "sigmoid"]
    score_apply: Literal["pre", "post"]
    route_norm: bool
    gate_bias: bool
    has_shared_experts: bool = False
    shared_experts_pattern: str = ""
    shared_experts_gate_pattern: str = ""
    autoep_config_defaults: dict[str, Any] = field(default_factory=dict)
    supports_expert_bias: bool = True
    unsupported_router_bias_names: tuple[str, ...] = ()
    preset_adapter: str = "default"
    hf_model_types: tuple[str, ...] = ()
    unsupported_hf_model_type_notes: dict[str, str] = field(default_factory=dict)
    min_transformers_version: str | None = None
    validated_transformers_versions: str = ""
    docs_support_notes: str = ""


@dataclass
class MoELayerSpec:
    """Detected MoE layer specification for a single module in the model."""

    moe_module_name: str
    model_family: str
    router_name: str
    experts_name: str
    expert_storage: Literal["fused_3d", "module_list"]
    expert_w1_name: str
    expert_w2_name: str
    expert_w3_name: str | None
    num_experts: int
    top_k: int
    hidden_size: int
    ffn_hidden_size: int
    score_func: Literal["softmax", "sigmoid"]
    score_apply: Literal["pre", "post"]
    route_norm: bool
    gate_bias: bool
    return_router_logits: bool
    router_logits_capture_target: Literal["moe_block", "router", "none"]
    router_logits_capture_index: int | None
    router_logits_capture_layer_name: str | None
    has_shared_experts: bool
    shared_experts_name: str
    shared_experts_gate_name: str = ""
    route_scale: float = 1.0
    num_expert_groups: int | None = None
    num_limited_groups: int | None = None
    group_score_func: Literal["max", "top2_sum"] = "top2_sum"
    supports_expert_bias: bool = True
    unsupported_router_bias_names: tuple[str, ...] = ()
    preset_adapter: str = "default"
    router_logits_capture_mode: Literal["raw", "post_score"] = "post_score"
    moe_output_shape: Literal["batched", "flat"] = "batched"


@dataclass
class AutoEPConfig:
    """User-facing configuration parsed from DS config JSON."""

    enabled: bool = False
    autoep_size: int = 1
    expert_tensor_parallel_size: int = 1
    preset_model: str | None = None
    moe_layer_pattern: str | None = None
    expert_pattern: str | None = None
    router_pattern: str | None = None
    use_grouped_mm: bool = True
    route_norm: bool | None = None
    route_scale: float = 1.0
    score_apply: Literal["auto", "pre", "post"] = "auto"
    combine_impl: Literal["auto", "weighted_sum", "legacy_bmm"] = "auto"
    num_expert_groups: int | None = None
    num_limited_groups: int | None = None
    score_func: Literal["auto", "softmax", "sigmoid"] = "auto"
    top_k: int | str = "auto"
    load_balance_coeff: float | None | object = _UNSET
    routed_scaling_factor: float | str = "auto"
    expert_w1: str | None = None
    expert_w2: str | None = None
    expert_w3: object = _UNSET
    num_experts_attr: str | None = None
    top_k_attr: str | None = None
    has_shared_experts: bool | None = None
    shared_experts_pattern: str | None = None
    shared_experts_gate_pattern: str | None = None
    _load_balance_coeff_explicit: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.load_balance_coeff is _UNSET:
            self.load_balance_coeff = None
            self._load_balance_coeff_explicit = False
        else:
            self._load_balance_coeff_explicit = True


@dataclass(frozen=True)
class GroupRoutingConfig:
    num_expert_groups: int | None
    num_limited_groups: int | None
    group_score_func: Literal["max", "top2_sum"] = "top2_sum"


@dataclass(frozen=True)
class ForwardContract:
    return_router_logits: bool = False
    capture_target: Literal["moe_block", "router", "none"] = "none"
    capture_index: int | None = None
    capture_layer_name: str | None = None
    router_logits_capture_mode: Literal["raw", "post_score"] = "post_score"
    moe_output_shape: Literal["batched", "flat"] = "batched"


class AutoEPPresetAdapter:
    """Default behavior shared by presets without model-specific parser rules."""

    def validate_compatibility(
        self,
        preset_name: str,
        preset: MoEModelPreset,
        model_config,
    ) -> None:
        """Validate public HF compatibility metadata for a selected preset."""
        model_type = getattr(model_config, "model_type", None) if model_config is not None else None
        self._validate_hf_model_type(preset_name, preset, model_type)
        self._validate_transformers_version(preset_name, preset, model_type)

    def _validate_hf_model_type(
        self,
        preset_name: str,
        preset: MoEModelPreset,
        model_type: str | None,
    ) -> None:
        if model_type is None:
            return

        unsupported_note = preset.unsupported_hf_model_type_notes.get(model_type)
        if unsupported_note is None:
            return

        supported = ", ".join(repr(value) for value in preset.hf_model_types) or "none"
        raise ValueError(f"AutoEP preset '{preset_name}' does not support model_type='{model_type}'. "
                         f"{unsupported_note} Supported HF model_type value(s): {supported}.")

    def _validate_transformers_version(
        self,
        preset_name: str,
        preset: MoEModelPreset,
        model_type: str | None,
    ) -> None:
        min_version = preset.min_transformers_version
        if min_version is None or model_type is None:
            return
        if not self._requires_transformers_version_validation():
            return
        if model_type not in preset.hf_model_types and model_type not in preset.unsupported_hf_model_type_notes:
            return

        try:
            installed_version = self._installed_transformers_version()
        except Exception as exc:
            raise ValueError(f"AutoEP preset '{preset_name}' for model_type='{model_type}' requires "
                             f"Transformers >= {min_version}, but transformers could not be imported: {exc}.") from exc

        try:
            installed = Version(installed_version)
            minimum = Version(min_version)
        except InvalidVersion as exc:
            raise ValueError(f"AutoEP preset '{preset_name}' for model_type='{model_type}' requires "
                             f"Transformers >= {min_version}, but the installed Transformers version "
                             f"'{installed_version}' could not be parsed.") from exc

        if installed < minimum:
            raise ValueError(f"AutoEP preset '{preset_name}' for model_type='{model_type}' requires "
                             f"Transformers >= {min_version}, but installed transformers=={installed_version}. "
                             "Upgrade Transformers or choose a preset/model combination supported by the "
                             "installed Transformers version.")

    def _installed_transformers_version(self) -> str:
        import transformers
        return getattr(transformers, "__version__", "unknown")

    def _requires_transformers_version_validation(self) -> bool:
        # The default adapter also covers non-HF/mock/custom-compatible configs;
        # specialized HF-only adapters opt in to minimum Transformers checks.
        return False

    def resolve_route_norm(
        self,
        config: AutoEPConfig,
        preset: MoEModelPreset,
        model_config,
    ) -> bool:
        if config.route_norm is not None:
            return config.route_norm

        cfg_norm = getattr(model_config, 'norm_topk_prob', None)
        if cfg_norm is not None:
            return bool(cfg_norm)
        return preset.route_norm

    def resolve_group_routing(
        self,
        config: AutoEPConfig,
        model_config,
    ) -> GroupRoutingConfig:
        return GroupRoutingConfig(
            num_expert_groups=config.num_expert_groups,
            num_limited_groups=config.num_limited_groups,
        )

    def resolve_expert_layout(
        self,
        experts_module: nn.Module,
        preset: MoEModelPreset,
    ) -> MoEModelPreset:
        return preset

    def adjust_forward_contract(self, contract: ForwardContract) -> ForwardContract:
        return contract

    def retarget_transformers_output_recorders(
        self,
        model: nn.Module,
        spec: MoELayerSpec,
        replacement: nn.Module,
        retargeted_keys: set[str],
        remove_output_capture_hooks: Callable[[nn.Module], int],
    ) -> None:
        return


_MISSING_REGISTRY_ENTRY = object()


def _restore_transformers_output_capture_registry(
    registry: dict[str, Any],
    original_entries: dict[str, object],
) -> None:
    for registry_key, original_entry in original_entries.items():
        if original_entry is _MISSING_REGISTRY_ENTRY:
            registry.pop(registry_key, None)
        else:
            registry[registry_key] = original_entry


def _install_instance_transformers_output_recorders(
    model: nn.Module,
    registry_entries: dict[str, dict[str, Any]],
    output_capturing: Any,
    remove_output_capture_hooks: Callable[[nn.Module], int],
) -> bool:
    maybe_install_capturing_hooks = getattr(output_capturing, "maybe_install_capturing_hooks", None)
    registry = getattr(output_capturing, "_CAN_RECORD_REGISTRY", None)
    if not callable(maybe_install_capturing_hooks) or not isinstance(registry, dict):
        return False

    remove_output_capture_hooks(model)
    for module in model.modules():
        if hasattr(module, "_output_capturing_hooks_installed"):
            module._output_capturing_hooks_installed = False
    model._output_capturing_hooks_installed = False

    original_entries = {
        registry_key: registry.get(registry_key, _MISSING_REGISTRY_ENTRY)
        for registry_key in registry_entries
    }
    try:
        registry.update(registry_entries)
        maybe_install_capturing_hooks(model)
    finally:
        _restore_transformers_output_capture_registry(registry, original_entries)
    return True


def _retarget_transformers_output_recorders_for_modules(
    *,
    model: nn.Module,
    display_name: str,
    recorder_key: str,
    retargeted_keys: set[str],
    remove_output_capture_hooks: Callable[[nn.Module], int],
    module_matches: Callable[[nn.Module], bool],
    make_output_recorder: Callable[[Any], Any],
) -> int:
    try:
        from transformers.utils import output_capturing
    except Exception:
        return 0

    registry = getattr(output_capturing, "_CAN_RECORD_REGISTRY", None)
    if not isinstance(registry, dict):
        return 0

    registry_entries: dict[str, dict[str, Any]] = {}
    retargeted = 0
    for module in model.modules():
        if not module_matches(module):
            continue

        registry_key = str(module.__class__)
        record_outputs = getattr(module, "_can_record_outputs", None)
        registry_outputs = registry.get(registry_key)
        base_outputs = record_outputs if isinstance(record_outputs, dict) else registry_outputs
        if not isinstance(base_outputs, dict) or "router_logits" not in base_outputs:
            continue

        retargeted_outputs = dict(base_outputs)
        retargeted_outputs["router_logits"] = make_output_recorder(output_capturing.OutputRecorder)
        module._can_record_outputs = retargeted_outputs
        registry_entries[registry_key] = retargeted_outputs
        retargeted += 1

    if retargeted == 0:
        from deepspeed.utils import logger
        logger.warning(f"AutoEP: {display_name} conversion did not find a HF output-capture registry "
                       "entry for router_logits.")
        return 0

    if _install_instance_transformers_output_recorders(
            model,
            registry_entries,
            output_capturing,
            remove_output_capture_hooks,
    ):
        return retargeted

    if recorder_key in retargeted_keys:
        return retargeted
    retargeted_keys.add(recorder_key)
    registry.update(registry_entries)
    if getattr(model, "_output_capturing_hooks_installed", False):
        remove_output_capture_hooks(model)
    model._output_capturing_hooks_installed = False
    return retargeted


class TransformersTopLevelRouterLogitsAdapter(AutoEPPresetAdapter):
    """Retarget Transformers model-level router-logit recorders to AutoEP."""

    def __init__(
        self,
        *,
        display_name: str,
        hf_model_types: tuple[str, ...],
        class_name_fragments: tuple[str, ...],
    ) -> None:
        self.display_name = display_name
        self.hf_model_types = hf_model_types
        self.class_name_fragments = class_name_fragments

    def adjust_forward_contract(self, contract: ForwardContract) -> ForwardContract:
        # Mixtral/Qwen3/Qwen2 capture raw router logits through Transformers'
        # model-level OutputRecorder hooks. AutoEP keeps the MoE block tensor
        # return contract intact and retargets the recorder to router.gate.
        return replace(
            contract,
            return_router_logits=False,
            capture_target="router",
            capture_index=0,
            router_logits_capture_mode="raw",
        )

    def retarget_transformers_output_recorders(
        self,
        model: nn.Module,
        spec: MoELayerSpec,
        replacement: nn.Module,
        retargeted_keys: set[str],
        remove_output_capture_hooks: Callable[[nn.Module], int],
    ) -> None:
        recorder_key = f"{spec.model_family}:{replacement.__class__.__module__}.{replacement.__class__.__qualname__}"

        router_gate = getattr(getattr(replacement, "router", None), "gate", None)
        if router_gate is None:
            return

        def module_matches(module: nn.Module) -> bool:
            module_config = getattr(module, "config", None)
            model_type = getattr(module_config, "model_type", None)
            class_name = module.__class__.__name__
            return (model_type in self.hf_model_types
                    or any(fragment in class_name for fragment in self.class_name_fragments))

        _retarget_transformers_output_recorders_for_modules(
            model=model,
            display_name=self.display_name,
            recorder_key=recorder_key,
            retargeted_keys=retargeted_keys,
            remove_output_capture_hooks=remove_output_capture_hooks,
            module_matches=module_matches,
            make_output_recorder=lambda OutputRecorder: OutputRecorder(
                router_gate.__class__, index=0, layer_name="router.gate"),
        )
