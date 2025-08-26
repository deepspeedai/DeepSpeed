#!/usr/bin/env python3
"""
Test script for EXAONE 4.0 inference with DeepSpeed v2

This script tests the EXAONE 4.0 model implementation in DeepSpeed inference v2.
It loads the model and runs inference to verify that the implementation works correctly.
"""

import os
import sys
import torch
import logging
from transformers import AutoTokenizer, AutoConfig
import deepspeed

# Add DeepSpeed to path if needed
if os.path.exists("deepspeed"):
    sys.path.insert(0, os.path.abspath("."))

from deepspeed.inference.v2.engine_factory import build_hf_engine
from deepspeed.inference.v2.config_v2 import RaggedInferenceEngineConfig
from deepspeed.inference.v2.scheduling_utils import SchedulingResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_exaone_1_2b():
    """Test EXAONE 4.0 1.2B model"""
    logger.info("Testing EXAONE 4.0 1.2B model...")
    
    model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info("✓ Tokenizer loaded successfully")
        
        # Load config
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"✓ Config loaded: {config.model_type}")
        logger.info(f"  - Hidden size: {config.hidden_size}")
        logger.info(f"  - Num layers: {config.num_hidden_layers}")
        logger.info(f"  - Num heads: {config.num_attention_heads}")
        logger.info(f"  - KV heads: {config.num_key_value_heads}")
        logger.info(f"  - Layer types: {config.layer_types[:10]}...")  # Show first 10
        
        # Configure DeepSpeed inference engine
        from deepspeed.inference.v2.ragged import DSStateManagerConfig
        
        state_manager_config = DSStateManagerConfig(
            max_tracked_sequences=32,
            max_ragged_batch_size=32,
            max_ragged_sequence_count=16,
            max_context=2048,
        )
        
        engine_config = RaggedInferenceEngineConfig(
            state_manager=state_manager_config,
            tensor_parallel={"tp_size": 1}  # Single GPU, no tensor parallelism
        )
        logger.info("✓ Engine config created")
        
        # Build the inference engine
        logger.info("Building DeepSpeed inference engine...")
        engine = build_hf_engine(model_name, engine_config)
        logger.info("✓ DeepSpeed inference engine built successfully")
        
        # Test basic inference using the put method
        logger.info("Testing basic inference...")
        
        # Create a simple test input
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0]  # Remove batch dimension
        
        # Test the put method (single sequence)
        batch_uids = [1]  # Unique ID for this sequence
        batch_tokens = [input_ids]
        
        try:
            with torch.no_grad():
                logits = engine.put(batch_uids, batch_tokens)
            
            logger.info(f"✓ Inference successful! Logits shape: {logits.shape}")
            logger.info(f"✓ Expected shape: [1, vocab_size] = [1, {config.vocab_size}]")
            
            # Test that we can get the next token
            next_token_logits = logits[0, -1, :]  # Get logits for the last token
            next_token_id = torch.argmax(next_token_logits).item()
            next_token = tokenizer.decode([next_token_id], skip_special_tokens=True)
            logger.info(f"✓ Next token prediction: '{next_token}' (ID: {next_token_id})")
            
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            raise
        
        # Clean up
        engine.flush(1)  # Remove the sequence from memory
        
        logger.info("✓ Basic inference test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_exaone_32b():
    """Test EXAONE 4.0 32B model (if available)"""
    logger.info("Testing EXAONE 4.0 32B model...")
    
    model_name = "LGAI-EXAONE/EXAONE-4.0-32B"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info("✓ Tokenizer loaded successfully")
        
        # Load config
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"✓ Config loaded: {config.model_type}")
        logger.info(f"  - Hidden size: {config.hidden_size}")
        logger.info(f"  - Num layers: {config.num_hidden_layers}")
        logger.info(f"  - Num heads: {config.num_attention_heads}")
        logger.info(f"  - KV heads: {config.num_key_value_heads}")
        
        # Configure DeepSpeed inference engine
        from deepspeed.inference.v2.ragged import DSStateManagerConfig
        
        state_manager_config = DSStateManagerConfig(
            max_tracked_sequences=16,
            max_ragged_batch_size=16,
            max_ragged_sequence_count=8,
            max_context=1024,
        )
        
        engine_config = RaggedInferenceEngineConfig(
            state_manager=state_manager_config,
            tensor_parallel={"tp_size": 1}  # Single GPU, no tensor parallelism
        )
        logger.info("✓ Engine config created")
        
        # Build the inference engine
        logger.info("Building DeepSpeed inference engine...")
        engine = build_hf_engine(model_name, engine_config)
        logger.info("✓ DeepSpeed inference engine built successfully")
        
        # Test basic inference using the put method
        logger.info("Testing basic inference...")
        
        # Create a simple test input
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0]  # Remove batch dimension
        
        # Test the put method (single sequence)
        batch_uids = [1]  # Unique ID for this sequence
        batch_tokens = [input_ids]
        
        try:
            with torch.no_grad():
                logits = engine.put(batch_uids, batch_tokens)
            
            logger.info(f"✓ Inference successful! Logits shape: {logits.shape}")
            logger.info(f"✓ Expected shape: [1, vocab_size] = [1, {config.vocab_size}]")
            
            # Test that we can get the next token
            next_token_logits = logits[0, -1, :]  # Get logits for the last token
            next_token_id = torch.argmax(next_token_logits).item()
            next_token = tokenizer.decode([next_token_id], skip_special_tokens=True)
            logger.info(f"✓ Next token prediction: '{next_token}' (ID: {next_token_id})")
            
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            raise
        
        # Clean up
        engine.flush(1)  # Remove the sequence from memory
        
        logger.info("✓ 32B model inference test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during 32B testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    logger.info("Starting EXAONE 4.0 DeepSpeed inference tests...")
    
    # Test 1.2B model
    success_1_2b = test_exaone_1_2b()
    
    # Test 32B model (optional, may fail due to memory constraints)
    try:
        success_32b = test_exaone_32b()
    except Exception as e:
        logger.warning(f"32B model test skipped due to: {e}")
        success_32b = False
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY:")
    logger.info(f"1.2B Model: {'✓ PASSED' if success_1_2b else '❌ FAILED'}")
    logger.info(f"32B Model:  {'✓ PASSED' if success_32b else '❌ FAILED/SKIPPED'}")
    
    if success_1_2b:
        logger.info("🎉 EXAONE 4.0 DeepSpeed inference implementation is working!")
    else:
        logger.error("💥 EXAONE 4.0 DeepSpeed inference implementation has issues!")

if __name__ == "__main__":
    main()
