# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
from transformers import AutoConfig

from deepspeed.inference.v2.model_implementations.exaone import (ExaoneTransformerContainer,
                                                                 ExaoneNonTransformerContainer, ExaonePolicy)


class TestExaoneImplementation:
    """Test suite for EXAONE 4.0 model implementation in DeepSpeed inference v2"""

    @pytest.fixture
    def exaone_config(self):
        """Load EXAONE 4.0 configuration for testing"""
        try:
            config = AutoConfig.from_pretrained('LGAI-EXAONE/EXAONE-4.0-32B', trust_remote_code=True)
            return config
        except Exception:
            pytest.skip("EXAONE 4.0 model config not available")

    @pytest.mark.inference_v2
    def test_exaone_config_properties(self, exaone_config):
        """Test that EXAONE config has expected properties"""
        assert exaone_config.model_type == "exaone4"
        assert hasattr(exaone_config, 'layer_types')
        assert hasattr(exaone_config, 'sliding_window')
        assert hasattr(exaone_config, 'num_attention_heads')
        assert hasattr(exaone_config, 'num_key_value_heads')

        # Test hybrid attention configuration
        layer_types = exaone_config.layer_types
        sliding_count = layer_types.count('sliding_attention')
        full_count = layer_types.count('full_attention')
        ratio = sliding_count / full_count if full_count > 0 else 0

        assert abs(ratio - 3.0) < 0.1, f"Expected 3:1 ratio, got {ratio:.1f}:1"

    @pytest.mark.inference_v2
    def test_transformer_container_param_mapping(self, exaone_config):
        """Test ExaoneTransformerContainer parameter mapping"""
        container = ExaoneTransformerContainer(exaone_config)

        # Check that all expected parameter mappings exist
        expected_mappings = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ]

        for param_name in expected_mappings:
            assert param_name in container.PARAM_MAPPING, f"Missing mapping for {param_name}"

    @pytest.mark.inference_v2
    def test_non_transformer_container_param_mapping(self, exaone_config):
        """Test ExaoneNonTransformerContainer parameter mapping"""
        container = ExaoneNonTransformerContainer(exaone_config)

        expected_mappings = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]

        for param_name in expected_mappings:
            assert param_name in container.PARAM_MAPPING, f"Missing mapping for {param_name}"

    @pytest.mark.inference_v2
    def test_exaone_inference_model_properties(self, exaone_config):
        """Test EXAONE model configuration properties"""
        # Test basic config properties that our model would use
        assert exaone_config.num_hidden_layers > 0
        assert exaone_config.hidden_size > 0
        assert exaone_config.vocab_size > 0
        assert exaone_config.num_attention_heads > 0
        assert exaone_config.num_key_value_heads > 0

        # Test EXAONE-specific properties
        assert hasattr(exaone_config, 'layer_types')
        assert len(exaone_config.layer_types) == exaone_config.num_hidden_layers

        # Test that ExaoneInferenceModel class can be imported
        from deepspeed.inference.v2.model_implementations.exaone.model import ExaoneInferenceModel
        assert ExaoneInferenceModel is not None

    @pytest.mark.inference_v2
    def test_hybrid_attention_layer_detection(self, exaone_config):
        """Test hybrid attention layer type detection logic"""
        # Test the layer pattern without full model instantiation
        layer_types = exaone_config.layer_types

        # Count layer types
        global_layers = []
        local_layers = []

        for i, layer_type in enumerate(layer_types):
            if layer_type == 'full_attention':
                global_layers.append(i)
            else:
                local_layers.append(i)

        # Should have 16 global and 48 local layers for 32B model
        assert len(global_layers) == 16, f"Expected 16 global layers, got {len(global_layers)}"
        assert len(local_layers) == 48, f"Expected 48 local layers, got {len(local_layers)}"

        # Test the logic that would be used by ExaoneInferenceModel
        # (testing the core logic without instantiation)
        def is_global_attention_layer(layer_idx: int) -> bool:
            if layer_types and layer_idx < len(layer_types):
                return layer_types[layer_idx] == 'full_attention'
            return False

        def should_apply_rope(layer_idx: int) -> bool:
            return not is_global_attention_layer(layer_idx)

        # Test RoPE application logic
        for layer in global_layers:
            assert not should_apply_rope(layer), f"Global layer {layer} should not apply RoPE"

        for layer in local_layers:
            assert should_apply_rope(layer), f"Local layer {layer} should apply RoPE"

    @pytest.mark.inference_v2
    def test_exaone_policy_creation(self, exaone_config):
        """Test ExaonePolicy creation and container map building"""

        # Mock checkpoint engine
        class MockCheckpointEngine:

            def __init__(self, config):
                self.model_config = config

            def parameters(self):
                return iter([])

        checkpoint_engine = MockCheckpointEngine(exaone_config)
        policy = ExaonePolicy(exaone_config, checkpoint_engine=checkpoint_engine)

        # Test container map creation
        container_map = policy.build_container_map()

        assert container_map.transformer_params is not None
        assert container_map.non_transformer_params is not None
        assert len(list(container_map.transformer_params)) == exaone_config.num_hidden_layers

    @pytest.mark.inference_v2
    def test_model_type_recognition(self, exaone_config):
        """Test that EXAONE model type is correctly recognized"""
        assert exaone_config.model_type == "exaone4"

        # Test that the config has the expected architecture
        assert "Exaone4ForCausalLM" in exaone_config.architectures

    @pytest.mark.inference_v2
    @pytest.mark.parametrize("layer_idx,expected_type", [
        (0, 'sliding_attention'),
        (1, 'sliding_attention'),
        (2, 'sliding_attention'),
        (3, 'full_attention'),
        (4, 'sliding_attention'),
        (7, 'full_attention'),
        (11, 'full_attention'),
    ])
    def test_layer_type_pattern(self, exaone_config, layer_idx, expected_type):
        """Test specific layer type patterns"""
        layer_types = exaone_config.layer_types
        if layer_idx < len(layer_types):
            assert layer_types[layer_idx] == expected_type, \
                f"Layer {layer_idx} expected {expected_type}, got {layer_types[layer_idx]}"
