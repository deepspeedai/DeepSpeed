---
title: "Converting a Hugging Face checkpoint to Universal Checkpointing format"
tags: checkpointing, training, deepspeed, huggingface
---

## Introduction to Universal Checkpointing

Universal Checkpointing in DeepSpeed abstracts away the complexities of saving and loading model states, optimizer states, and training scheduler states. This feature is designed to work out of the box with minimal configuration, supporting a wide range of model sizes and types, from small-scale models to large, distributed models with different parallelism topologies trained across multiple GPUs and other accelerators.

See more: https://www.deepspeed.ai/tutorials/universal-checkpointing/

## Converting a pretrained Hugging Face checkpoint to Universal Checkpointing format

### Step 1: Download a pretrained Hugging Face checkpoint
Download a pretrained Hugging Face checkpoint from the Hugging Face Hub using [snapshot_download](https://huggingface.co/docs/huggingface_hub/en/guides/download)

Hugging Face checkpoints are one or many files in the `pytorch_model.bin` or `safetensors format`.

### Step 2: Convert Hugging Face checkpoint to Universal Checkpointing format

To convert a Hugging Face checkpoint to Universal Checkpointing format, you can use the `hf_to_universal.py` script provided in the DeepSpeed repository. This script will take a Hugging Face checkpoint directory and convert it to a Universal Checkpointing format.

```bash
python deepspeed/checkpoint/hf_to_universal.py --hf_checkpoint_dir /path/to/huggingface/checkpoint --save_dir /path/to/universal/checkpoint
```

This script will process the Hugging Face checkpoint and generate a new checkpoint in the Universal Checkpointing format. Note that `hf_to_universal.py` script supports both `.safetensors` and `pytorch.bin` checkpoint format. Use `--safe_serialization` flag to convert from `.safetensors` format.

### Step 3: Resume Training with Universal Checkpoint
With the Universal checkpoint ready, you can now resume training on potentially with different parallelism topologies or training configurations. To do this add `--universal-checkpoint` to your DeepSpeed config (json) file


## Conclusion
DeepSpeed Universal Checkpointing simplifies the management of model states, making it easier to save, load, and transfer model states across different training sessions and parallelism techniques. By converting a Hugging Face checkpoint to Universal Checkpointing format, you can load pretrained weights of any model in the Hugging Face Hub and resume training with DeepSpeed under any parallelism topologies.

For more detailed examples and advanced configurations, please refer to the [Megatron-DeepSpeed examples](https://github.com/deepspeedai/Megatron-DeepSpeed/tree/main/examples_deepspeed/universal_checkpointing).
