# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import os
import shutil
import logging
from concurrent.futures import ProcessPoolExecutor
from deepspeed.accelerator import get_accelerator
from tqdm import tqdm
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hard-coded constants for parameter patterns
VOCAB_PARAMETER_PATTERNS = [
    'word_embeddings',
    'embed_tokens',
    'embedding',
    'wte',  # GPT style embeddings
    'lm_head'  # Language model head, often tied with embeddings
]


def get_parameter_type(name: str) -> dict:
    """Determine parameter type and required fields based on name."""
    param_info = {
        'cat_dim': 0  # Default concatenation dimension
    }

    # Check for vocabulary tensors (embeddings, etc.)
    if any(pattern in name.lower() for pattern in VOCAB_PARAMETER_PATTERNS):
        param_info['vocab_tensor'] = True

    # TODO: figure out if we need to check for row-parallel parameters
    return param_info


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint to Universal Checkpoint format')
    parser.add_argument('--hf_checkpoint_dir',
                        type=str,
                        required=True,
                        help='Path to the HuggingFace checkpoint directory')
    parser.add_argument('--safe_serialization',
                        action='store_true',
                        default=False,
                        help='Use safetensors for serialization')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use for saving checkpoints')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save checkpoints')
    args = parser.parse_args()

    # Create a temporary directory for atomic operations
    temp_save_dir = args.save_dir + '.tmp'

    def save_parameter(name: str, param: torch.Tensor, save_dir: str):
        """Save a parameter and its optimizer states in universal format."""
        # Create parameter directory under zero/
        param_dir = os.path.join(save_dir, name)
        os.makedirs(param_dir, exist_ok=True)

        # Get parameter type and required fields
        param_info = get_parameter_type(name)

        # Save parameter in fp32 with proper dictionary structure
        param_path = os.path.join(param_dir, "fp32.pt")
        param_dict = {
            'param': param.to(torch.float32),  # Main tensor goes in 'param' field
            **param_info  # Include all determined parameter info
        }
        torch.save(param_dict, param_path)

        # Since HuggingFace checkpoints do not have optimizer states,
        # we initialize them with zeros
        for state in ("exp_avg", "exp_avg_sq"):
            state_path = os.path.join(param_dir, f"{state}.pt")
            state_dict = {
                'param': torch.zeros_like(param, dtype=torch.float32),
                **param_info  # Include same parameter info in optimizer states
            }
            torch.save(state_dict, state_path)

    def process_shard(shard_file, checkpoint_dir, save_dir, safe_serialization):
        """Process a single shard file."""
        try:
            shard_path = os.path.join(checkpoint_dir, shard_file)
            logger.info(f"Loading shard from: {shard_path}")

            if safe_serialization:
                from safetensors.torch import load_file
                shard_dict = load_file(shard_path)
            else:
                shard_dict = torch.load(shard_path, map_location='cpu')

            # Create progress bar for parameters within this shard
            pbar = tqdm(total=len(shard_dict),
                        desc=f"Processing {os.path.basename(shard_file)}",
                        position=1,
                        leave=False)

            for key, param in shard_dict.items():
                save_parameter(key, param, save_dir)
                del param
                pbar.update(1)
                pbar.set_postfix({'key': key[:20] + '...' if len(key) > 20 else key})

            pbar.close()
            del shard_dict
            get_accelerator().empty_cache()
            logger.info(f"Completed processing shard: {shard_file}")

        except Exception as e:
            logger.error(f"Error processing shard {shard_file}: {str(e)}")
            raise

    def get_shard_list(checkpoint_dir):
        """Get list of shards from index file."""
        if args.safe_serialization:
            index_file = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        else:
            index_file = os.path.join(checkpoint_dir, "pytorch_model.bin.index.json")

        if os.path.exists(index_file):
            import json
            with open(index_file, 'r') as f:
                index = json.load(f)
            return list(set(index['weight_map'].values()))
        else:
            # Handle single file case
            if args.safe_serialization and os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
                return ["model.safetensors"]
            elif os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")):
                return ["pytorch_model.bin"]
            else:
                raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    def process_shard_batch(shard_files: List[str], checkpoint_dir: str, save_dir: str, safe_serialization: bool):
        """Process a batch of shards in parallel."""
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for shard_file in shard_files:
                future = executor.submit(process_shard, shard_file, checkpoint_dir, save_dir, safe_serialization)
                futures.append((shard_file, future))

            # Create progress bar for this batch
            batch_pbar = tqdm(total=len(futures), desc=f"Processing shard batch", position=0, leave=True)

            # Wait for all futures to complete
            for shard_file, future in futures:
                try:
                    future.result()  # This will raise any exceptions that occurred
                    batch_pbar.update(1)
                    batch_pbar.set_postfix({'last_completed': os.path.basename(shard_file)})
                except Exception as e:
                    logger.error(f"Failed processing shard {shard_file}: {str(e)}")
                    raise

            batch_pbar.close()

    try:
        # Create zero subdirectory in temp directory
        temp_zero_dir = os.path.join(temp_save_dir, 'zero')
        if os.path.exists(temp_zero_dir):
            logger.info(f"Removing existing temp directory: {temp_zero_dir}")
            shutil.rmtree(temp_zero_dir)

        shard_files = get_shard_list(args.hf_checkpoint_dir)
        total_shards = len(shard_files)
        logger.info(f"Found {total_shards} shards to process")
        # Process shards in batches equal to the number of workers
        batch_size = args.num_workers
        for i in range(0, total_shards, batch_size):
            batch_shards = shard_files[i:i + batch_size]
            logger.info(
                f"Processing batch of {len(batch_shards)} shards ({i+1}-{min(i+batch_size, total_shards)} of {total_shards})"
            )
            process_shard_batch(
                batch_shards,
                args.hf_checkpoint_dir,
                temp_zero_dir,  # Changed from temp_save_dir to temp_zero_dir
                args.safe_serialization)

            # Clear CUDA cache after each batch to free up memory
            get_accelerator().empty_cache()

        logger.info("All shard batches processed successfully")

        final_save_dir = os.path.join(args.save_dir, 'zero')
        if os.path.exists(final_save_dir):
            shutil.rmtree(final_save_dir)

        # Create the parent directory if it doesn't exist
        os.makedirs(os.path.dirname(final_save_dir), exist_ok=True)
        # Move the zero directory to its final location
        os.rename(temp_zero_dir, final_save_dir)

        # Clean up the temporary directory
        if os.path.exists(temp_save_dir):
            shutil.rmtree(temp_save_dir)

        # Write identifier file
        with open(os.path.join(args.save_dir, 'source.txt'), 'w') as f:
            f.write("Huggingface checkpoint")

        logger.info(f"Successfully saved checkpoint to {final_save_dir}")

        # Update latest file
        checkpoint_root_folder = os.path.dirname(args.save_dir)
        step_folder = os.path.basename(args.save_dir)
        latest_file = os.path.join(checkpoint_root_folder, 'latest_universal')
        with open(latest_file, 'w') as f:
            f.write(step_folder)

        logger.info(f"Checkpoint conversion completed successfully. Latest file updated at {latest_file}")

    except Exception as e:
        logger.error(f"Failed to process checkpoint: {str(e)}")
        if os.path.exists(temp_save_dir):
            shutil.rmtree(temp_save_dir)
        raise
