# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import pytest

transformers = pytest.importorskip("transformers")
Gemma4Config = getattr(transformers, "Gemma4Config", None)
pytestmark = pytest.mark.skipif(Gemma4Config is None, reason="Gemma4Config not available in this transformers version")


def test_gemma4_text_config_fallback():
    config = Gemma4Config()
    assert not hasattr(config, 'num_attention_heads'), \
        "Gemma4Config top-level should not have num_attention_heads"
    arch_cfg = config.get_text_config()
    assert hasattr(arch_cfg, 'num_attention_heads')
    assert arch_cfg.num_attention_heads > 0
    assert hasattr(arch_cfg, 'num_key_value_heads')
    assert arch_cfg.num_key_value_heads > 0
    assert hasattr(arch_cfg, 'num_hidden_layers')
    assert arch_cfg.num_hidden_layers > 0
    assert hasattr(arch_cfg, 'hidden_size')
    assert arch_cfg.hidden_size > 0


def test_gemma4_text_config_matches_text_config():
    config = Gemma4Config()
    arch_cfg = config.get_text_config()
    assert arch_cfg is config.text_config
    assert arch_cfg.num_attention_heads == config.text_config.num_attention_heads
    assert arch_cfg.num_key_value_heads == config.text_config.num_key_value_heads
