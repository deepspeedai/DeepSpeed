# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Test that partition_config receives correct full hierarchical module paths.

The bug: AutoTP._replace_module built ``full_name`` from ``prev_name`` (the
immediate parent only) instead of ``class_name`` (the accumulated hierarchical
path).  Patterns like ``model.layers.0.self_attn.q_proj`` never matched
because the name was just ``0.self_attn.q_proj``.
"""

import pytest
import torch.nn as nn

from deepspeed.module_inject.auto_tp import AutoTP, AutoTPConfig, PartitionType, TPLayerSpec
from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer, LmHeadLinearAllreduce, set_autotp_mode
from deepspeed.module_inject.tp_plan_converter import TPPlanConverter


class SubAttn(nn.Module):

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 32, bias=False)
        self.k_proj = nn.Linear(32, 32, bias=False)
        self.v_proj = nn.Linear(32, 32, bias=False)
        self.o_proj = nn.Linear(32, 32, bias=False)


class DecoderLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.self_attn = SubAttn()
        self.mlp = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))


class DummyModel(nn.Module):

    def __init__(self, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(num_layers)])
        self.head = nn.Linear(32, 100, bias=False)


class OutputModel(nn.Module):

    def __init__(self, tied):
        super().__init__()
        self.config = type("Config", (), {"tie_word_embeddings": not tied})()
        self.embed_tokens = nn.Embedding(100, 32)
        self.lm_head = nn.Linear(32, 100, bias=False)
        if tied:
            self.lm_head.weight = self.embed_tokens.weight


def _build_config():
    """Partition config that matches q_proj and o_proj via regex."""
    return AutoTPConfig(layer_specs=[
        TPLayerSpec(patterns=[r".*\.self_attn\.q_proj"], partition_type=PartitionType.COLUMN),
        TPLayerSpec(patterns=[r".*\.self_attn\.o_proj"], partition_type=PartitionType.ROW),
    ])


def _capture_matched_names(model, config):
    """Run _replace_module and capture full_name values that match a spec."""
    matched_names = []
    original = AutoTP._replace_with_config

    def capture(self, child, full_name):
        # Only capture if a spec actually matches
        param_name = full_name + ".weight"
        model_type = self._get_model_type() if hasattr(self, '_get_model_type') else None
        spec = config.find_matching_spec(param_name, model_type)
        if spec is not None:
            matched_names.append(full_name)
        return None

    AutoTP._replace_with_config = capture
    try:
        autotp = AutoTP(
            module=model,
            all_reduce_linears=[],
            prefix="model",
            state_dict=None,
            linear_layer_setting=None,
            orig_layer_impl=None,
            partition_config=config,
            model_config=getattr(model, "config", None),
        )
        autotp._replace_module(model)
    finally:
        AutoTP._replace_with_config = original
    return matched_names


def test_partition_config_receives_full_path():
    """Verify that pattern matching sees the full hierarchical path."""
    model = DummyModel(num_layers=2)
    config = _build_config()
    matched_names = _capture_matched_names(model, config)

    for layer_idx in range(2):
        assert f"layers.{layer_idx}.self_attn.q_proj" in matched_names, \
            f"Expected 'layers.{layer_idx}.self_attn.q_proj', got: {matched_names}"
        assert f"layers.{layer_idx}.self_attn.o_proj" in matched_names, \
            f"Expected 'layers.{layer_idx}.self_attn.o_proj', got: {matched_names}"


def test_no_truncated_paths():
    """Ensure paths are never truncated to just the immediate parent prefix."""
    model = DummyModel(num_layers=3)
    config = _build_config()
    matched_names = _capture_matched_names(model, config)

    for name in matched_names:
        assert name.startswith("layers."), \
            f"Path should start with 'layers.', got: {name}"
        assert ".self_attn." in name, \
            f"Path should contain '.self_attn.', got: {name}"
        assert name.count(".") >= 3, \
            f"Path should have at least 3 dots (layers.N.self_attn.X_proj), got: {name}"


def test_nested_depth_correct():
    """Verify correct count and paths at 3 layers deep."""
    model = DummyModel(num_layers=3)
    config = _build_config()
    matched_names = _capture_matched_names(model, config)

    expected_count = 3 * 2  # 3 layers × (q_proj + o_proj)
    assert len(matched_names) == expected_count, \
        f"Expected {expected_count} matches, got {len(matched_names)}: {matched_names}"

    for layer_idx in range(3):
        assert f"layers.{layer_idx}.self_attn.q_proj" in matched_names
        assert f"layers.{layer_idx}.self_attn.o_proj" in matched_names


def _build_gathered_lm_head_autotp(model, mp_size=1):
    config = AutoTPConfig(layer_specs=[
        TPLayerSpec(
            patterns=[r".*lm_head\.weight$"],
            partition_type=PartitionType.COLUMN,
            gather_output=True,
        ),
    ])
    autotp = AutoTP(
        module=model,
        all_reduce_linears=[],
        prefix="",
        state_dict=None,
        linear_layer_setting=None,
        orig_layer_impl=None,
        partition_config=config,
        model_config=getattr(model, "config", None),
    )
    autotp.set_tensor_parallel_config(mp_size, None)
    autotp.update_linear_policies()
    return autotp


def _build_legacy_lm_head_autotp(model, training_mode=False):
    autotp = AutoTP(
        module=model,
        all_reduce_linears=("lm_head", "embed_out"),
        prefix="",
        state_dict=None,
        linear_layer_setting=(nn.Linear, nn.Embedding),
        orig_layer_impl=None,
        training_mode=training_mode,
    )
    autotp.mp_size = 1
    autotp.mp_group = None
    autotp.update_linear_policies()
    return autotp


def _build_row_output_head_autotp(model, head="lm_head", training_mode=False, mp_size=1):
    config = AutoTPConfig(layer_specs=[
        TPLayerSpec(patterns=[rf".*{head}\.weight$"], partition_type=PartitionType.ROW),
    ])
    autotp = AutoTP(
        module=model,
        all_reduce_linears=(),
        prefix="",
        state_dict=None,
        linear_layer_setting=(nn.Linear, nn.Embedding),
        orig_layer_impl=None,
        partition_config=config,
        training_mode=training_mode,
    )
    autotp.mp_size = mp_size
    autotp.mp_group = None
    autotp.update_linear_policies()
    return autotp


def test_gathered_lm_head_uses_column_parallel_layer_when_untied():
    model = OutputModel(tied=False)
    _build_gathered_lm_head_autotp(model)._replace_module(model)

    assert isinstance(model.lm_head, LinearLayer)
    assert model.lm_head.gather_output


def test_gathered_lm_head_falls_back_for_runtime_parameter_tie():
    model = OutputModel(tied=True)
    assert model.lm_head.weight is model.embed_tokens.weight

    _build_gathered_lm_head_autotp(model)._replace_module(model)

    assert isinstance(model.embed_tokens, nn.Embedding)
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.weight is model.embed_tokens.weight


def test_tied_embedding_plan_leaves_lm_head_and_embedding_replicated():
    model = OutputModel(tied=True)
    specs = TPPlanConverter.convert({"embed_tokens": "embedding_rowwise", "lm_head": "colwise_gather_output"})

    autotp = AutoTP(
        module=model,
        all_reduce_linears=[],
        prefix="",
        state_dict=None,
        linear_layer_setting=None,
        orig_layer_impl=None,
        partition_config=AutoTPConfig(layer_specs=specs),
    )
    autotp.set_tensor_parallel_config(1, None)
    autotp.update_linear_policies()
    autotp._replace_module(model)

    assert isinstance(model.embed_tokens, nn.Embedding)
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.weight is model.embed_tokens.weight


def test_gathered_lm_head_uses_column_parallel_layer_when_output_dim_is_uneven():
    model = OutputModel(tied=False)
    model.lm_head = nn.Linear(32, 101, bias=False)

    _build_gathered_lm_head_autotp(model, mp_size=2)._replace_module(model)

    assert isinstance(model.lm_head, LinearLayer)
    assert model.lm_head.gather_output


@pytest.mark.parametrize("head", ["lm_head", "embed_out"])
def test_legacy_output_head_defaults_to_column_parallel_during_training(head):
    model = OutputModel(tied=False)
    if head == "embed_out":
        model.embed_out = model.lm_head
        del model.lm_head

    _build_legacy_lm_head_autotp(model, training_mode=True)._replace_last_linear_module(model)

    output_head = getattr(model, head)
    assert isinstance(output_head, LinearLayer)
    assert output_head.gather_output


def test_legacy_lm_head_keeps_inference_allreduce_routing():
    model = OutputModel(tied=False)
    _build_legacy_lm_head_autotp(model)._replace_last_linear_module(model)

    assert isinstance(model.lm_head, LmHeadLinearAllreduce)


def test_global_training_mode_does_not_leak_into_inference_routing():
    model = OutputModel(tied=False)
    set_autotp_mode(training=True)
    try:
        _build_legacy_lm_head_autotp(model, training_mode=False)._replace_last_linear_module(model)
    finally:
        set_autotp_mode(training=False)

    assert isinstance(model.lm_head, LmHeadLinearAllreduce)


def test_legacy_tied_lm_head_stays_replicated_during_training():
    model = OutputModel(tied=True)
    tied_weight = model.embed_tokens.weight

    _build_legacy_lm_head_autotp(model, training_mode=True)._replace_last_linear_module(model)

    assert isinstance(model.embed_tokens, nn.Embedding)
    assert isinstance(model.lm_head, nn.Linear)
    assert model.embed_tokens.weight is tied_weight
    assert model.lm_head.weight is tied_weight


def test_explicit_row_parallel_lm_head_is_not_overridden_by_its_name():
    model = OutputModel(tied=False)
    _build_row_output_head_autotp(model, training_mode=True)._replace_module(model)

    assert isinstance(model.lm_head, LinearAllreduce)
    assert not isinstance(model.lm_head, LmHeadLinearAllreduce)


@pytest.mark.parametrize("tied", [False, True])
def test_explicit_row_parallel_lm_head_training_rejects_tp_greater_than_one(tied):
    model = OutputModel(tied=tied)

    with pytest.raises(NotImplementedError, match="row-parallel output heads"):
        _build_row_output_head_autotp(model, training_mode=True, mp_size=2)._replace_module(model)


def test_explicit_row_parallel_lm_head_keeps_inference_specialization():
    model = OutputModel(tied=False)
    _build_row_output_head_autotp(model)._replace_module(model)

    assert isinstance(model.lm_head, LmHeadLinearAllreduce)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
