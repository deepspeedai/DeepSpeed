# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Inference V2 must find rope_theta wherever the installed transformers keeps it.

transformers 5.0 folded the rotary settings into ``config.rope_parameters`` and dropped
the ``rope_theta`` attribute. Models that read the attribute directly raise against a
stock config on 5.x; ``exaone4`` read it through a ``getattr`` default and silently used
that default instead of the value the config actually carries.
"""

from types import SimpleNamespace

import pytest

from deepspeed.inference.v2.model_implementations.inference_transformer_base import DSTransformerModelBase


class _Model:
    """Minimal stand-in that borrows the property under test."""

    rope_theta = DSTransformerModelBase.rope_theta

    def __init__(self, config):
        self._config = config


def test_reads_the_legacy_attribute():
    assert _Model(SimpleNamespace(rope_theta=500000.0)).rope_theta == 500000.0


def test_reads_rope_parameters_when_the_attribute_is_gone():
    config = SimpleNamespace(rope_parameters={"rope_theta": 10000.0, "rope_type": "default"})

    assert _Model(config).rope_theta == 10000.0


def test_reads_rope_scaling_when_that_is_the_only_dict():
    config = SimpleNamespace(rope_scaling={"rope_theta": 1000000.0, "rope_type": "default"})

    assert _Model(config).rope_theta == 1000000.0


def test_prefers_the_attribute_when_both_are_present():
    config = SimpleNamespace(rope_theta=500000.0, rope_parameters={"rope_theta": 10000.0})

    assert _Model(config).rope_theta == 500000.0


def test_raises_instead_of_guessing_when_nothing_carries_it():
    # exaone4 used to fall back to 1e6 here, which is a silent 100x error against a
    # config whose real base is 1e4.
    with pytest.raises(ValueError, match="rope_theta"):
        _ = _Model(SimpleNamespace()).rope_theta


def test_unwraps_a_per_layer_type_dict():
    """A config that sets RoPE per layer type nests one level deeper.

    ``standardize_rope_params`` leaves the class default at the top level of the same
    dict, so reading the top level returns 10000.0 rather than the 1000000.0 the
    checkpoint asked for.
    """
    config = SimpleNamespace(
        rope_parameters={
            "sliding_attention": {
                "rope_theta": 1000000.0,
                "rope_type": "default"
            },
            "full_attention": {
                "rope_theta": 1000000.0,
                "rope_type": "default"
            },
            "rope_theta": 10000.0,
            "rope_type": "default",
        })

    assert _Model(config).rope_theta == 1000000.0


def test_refuses_a_config_whose_layer_types_disagree():
    """Callers feed one ``RotateHalfConfig.theta_base`` for the whole model.

    Picking either base would be wrong for the layers using the other one, so this is
    refused rather than resolved.
    """
    config = SimpleNamespace(
        rope_parameters={
            "sliding_attention": {
                "rope_theta": 1000000.0,
                "rope_type": "default"
            },
            "full_attention": {
                "rope_theta": 16000000.0,
                "rope_type": "default"
            },
            "rope_theta": 10000.0,
            "rope_type": "default",
        })

    with pytest.raises(ValueError, match="different rope_theta per layer type"):
        _ = _Model(config).rope_theta


@pytest.mark.parametrize(
    "module_name, config_name",
    [
        ("llama", "LlamaConfig"),
        ("mistral", "MistralConfig"),
        ("mixtral", "MixtralConfig"),
        ("phi", "PhiConfig"),
        ("phi3", "Phi3Config"),
        ("qwen2", "Qwen2Config"),
        ("qwen2_moe", "Qwen2MoeConfig"),
        ("exaone4", "Exaone4Config"),
    ],
)
def test_resolves_against_the_installed_transformers_configs(module_name, config_name):
    """Every config backing a V2 model must yield a base through one spelling or the other."""
    importlib = pytest.importorskip("importlib")
    try:
        module = importlib.import_module(f"transformers.models.{module_name}.configuration_{module_name}")
        config = getattr(module, config_name)()
    except (ImportError, AttributeError):
        pytest.skip(f"{config_name} is not available in the installed transformers")

    assert _Model(config).rope_theta > 0
