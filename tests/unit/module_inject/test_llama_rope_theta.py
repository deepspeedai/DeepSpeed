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
