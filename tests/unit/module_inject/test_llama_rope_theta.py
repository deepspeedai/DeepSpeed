# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Llama kernel injection must find rope_theta wherever the installed transformers keeps it.

The value has moved twice. Older releases exposed ``config.rope_theta``; transformers 5.0
folded the rotary settings into ``config.rope_parameters`` and dropped the attribute, so an
injection policy that only knows the old spelling raises AttributeError against a stock
LlamaConfig. Older still, it lived on the attention module.
"""

from types import SimpleNamespace

import pytest

from deepspeed.module_inject.containers.llama import _get_rope_theta


def test_reads_the_legacy_config_attribute():
    self_attn = SimpleNamespace(config=SimpleNamespace(rope_theta=500000.0))

    assert _get_rope_theta(self_attn) == 500000.0


def test_reads_rope_parameters_when_the_attribute_is_gone():
    # transformers >= 5.0: the attribute is absent and the value sits in the dict.
    config = SimpleNamespace(rope_parameters={"rope_theta": 10000.0, "rope_type": "default"})
    self_attn = SimpleNamespace(config=config)

    assert _get_rope_theta(self_attn) == 10000.0


def test_prefers_the_attribute_when_both_are_present():
    config = SimpleNamespace(rope_theta=500000.0, rope_parameters={"rope_theta": 10000.0})
    self_attn = SimpleNamespace(config=config)

    assert _get_rope_theta(self_attn) == 500000.0


def test_falls_back_to_the_module_attribute():
    # No config at all, the layout the policy handled before configs were attached.
    self_attn = SimpleNamespace(rope_theta=1000000.0)

    assert _get_rope_theta(self_attn) == 1000000.0


def test_falls_back_when_rope_parameters_carries_no_theta():
    config = SimpleNamespace(rope_parameters={"rope_type": "default"})
    self_attn = SimpleNamespace(config=config, rope_theta=250000.0)

    assert _get_rope_theta(self_attn) == 250000.0


def test_resolves_against_a_real_llama_config():
    """The six cases above build the config by hand, so none of them pins the claim this
    change rests on: which spelling a stock `LlamaConfig` actually carries.

    This one stays meaningful on either side of the 5.0 boundary — it takes the legacy
    attribute on 4.x and `rope_parameters` on 5.x — and it uses a non-default theta, so a
    helper that returned the class default would fail it.
    """
    LlamaConfig = pytest.importorskip("transformers.models.llama.configuration_llama").LlamaConfig

    self_attn = SimpleNamespace(config=LlamaConfig(rope_theta=500000.0))

    assert _get_rope_theta(self_attn) == 500000.0


def test_the_two_spellings_do_not_disagree_on_a_real_config():
    """Guards the branch rather than the value.

    If transformers reinstates `rope_theta` as a deprecated property, the test above still
    passes while the injection path silently changes which branch it takes. That is only a
    problem if the two spellings can disagree, so this asserts they cannot.
    """
    LlamaConfig = pytest.importorskip("transformers.models.llama.configuration_llama").LlamaConfig
    config = LlamaConfig(rope_theta=500000.0)

    parameters = getattr(config, "rope_parameters", None) or {}
    carried = [
        value for value in (getattr(config, "rope_theta", None), parameters.get("rope_theta")) if value is not None
    ]

    assert carried, "neither spelling carries rope_theta on the installed transformers"
    assert all(value == 500000.0 for value in carried), f"the spellings disagree: {carried}"


def test_raises_when_nothing_carries_it():
    with pytest.raises(AttributeError):
        _get_rope_theta(SimpleNamespace(config=SimpleNamespace()))


# --- scaled rotary variants ----------------------------------------------------
#
# The injected kernel builds its rotary embedding from a scalar base
# (`InferenceContext.get_rotary(rotary_dim, rope_theta)`) and carries no scaling
# parameters at all, so a config asking for one cannot be served here.


@pytest.mark.parametrize("rope_type", ["llama3", "linear", "dynamic", "yarn", "longrope"])
def test_a_scaled_rope_variant_is_refused(rope_type):
    """Reading only rope_theta out of a scaled config is silently wrong.

    DeepSeek-R1-Distill-Llama-8B (#8340) is the live case: `rope_type="llama3"` with
    `factor`, `low_freq_factor`, `high_freq_factor` and `original_max_position_embeddings`.
    Dropping those and keeping the base runs the model with unscaled positions and no error,
    which is worse than the AttributeError this helper exists to remove.
    """
    config = SimpleNamespace(
        rope_parameters={
            "rope_type": rope_type,
            "rope_theta": 500000.0,
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        })

    with pytest.raises(ValueError, match="cannot serve rope_type"):
        _get_rope_theta(SimpleNamespace(config=config))


def test_a_scaled_variant_in_the_legacy_rope_scaling_spelling_is_refused():
    """transformers < 5.0 carries the same request under `rope_scaling`."""
    config = SimpleNamespace(rope_theta=500000.0, rope_scaling={"rope_type": "llama3", "factor": 8.0})

    with pytest.raises(ValueError, match="cannot serve rope_type"):
        _get_rope_theta(SimpleNamespace(config=config))


def test_the_default_rope_type_is_not_refused():
    """`rope_type: "default"` is what standardize_rope_params writes for plain RoPE."""
    config = SimpleNamespace(rope_parameters={"rope_theta": 500000.0, "rope_type": "default"})

    assert _get_rope_theta(SimpleNamespace(config=config)) == 500000.0


def test_a_real_llama_config_is_not_refused():
    """The stock config the crash fix targets carries no scaling and must still resolve."""
    LlamaConfig = pytest.importorskip("transformers.models.llama.configuration_llama").LlamaConfig

    assert _get_rope_theta(SimpleNamespace(config=LlamaConfig(rope_theta=500000.0))) == 500000.0
