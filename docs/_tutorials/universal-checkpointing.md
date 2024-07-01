---
title: "Universal Checkpointing with DeepSpeed: A Practical Guide"
tags: checkpointing, training, deepspeed
---

DeepSpeed Universal Checkpointing feature is a powerful tool for saving and loading model checkpoints in a way that is both efficient and flexible, enabling seamless model training continuation and finetuning across different model architectures, different parallelism techniques and training configurations. This tutorial, tailored for both begininers and experienced users, provides a step-by-step guide on how to leverage Universal Checkpointing in your DeepSpeed-powered applications. This tutorial will guide you through the process of creating ZeRO checkpoints, converting them into a Universal format, and resuming training with these universal checkpoints. This approach is crucial for leveraging pre-trained models and facilitating seamless model training across different setups.


## Introduction to Universal Checkpointing

Universal Checkpointing in DeepSpeed abstracts away the complexities of saving and loading model states, optimizer states, and training scheduler states. This feature is designed to work out of the box with minimal configuration, supporting a wide range of model sizes and types, from small-scale models to large, distributed models with different parallelism topologies trained across multiple GPUs and other accelerators.

## Prerequisites

Before you begin, ensure you have the following:
- DeepSpeed installed, installation can be done via `pip install deepspeed`.
- A model training script that utilizes DeepSpeed for distributed training.

## How to use DeepSpeed Universal Checkpointing

Follow the three simple steps below:

### Step 1: Create ZeRO Checkpoint

The first step in leveraging DeepSpeed Universal Checkpointing is to create a ZeRO checkpoint. [ZeRO](/tutorials/zero/) (Zero Redundancy Optimizer) is a memory optimization technology in DeepSpeed that allows for efficient training of large models. To create a ZeRO checkpoint, you'll need to:

 - Initialize your model with DeepSpeed using the ZeRO optimizer.
 - Train your model to the desired state (iterations).
 - Save a checkpoint using DeepSpeed's checkpointing feature.


### Step 2: Convert ZeRO Checkpoint to Universal Format

Once you have a ZeRO checkpoint, the next step is to convert it into the Universal format. This format is designed to be flexible and compatible across different model architectures and DeepSpeed configurations. To convert a checkpoint:

 - Use the [ds_to_universal.py](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/checkpoint/ds_to_universal.py) script provided by DeepSpeed.
 - Specify the path to your ZeRO checkpoint and the desired output path for the Universal checkpoint.

```bash
python ds_to_universal.py --input_folder /path/to/zero/checkpoint --output_folder /path/to/universal/checkpoint
```

This script will process the ZeRO checkpoint and generate a new checkpoint in the Universal format. Pass `--help` flag to see other options.

### Step 3: Resume Training with Universal Checkpoint
With the Universal checkpoint ready, you can now resume training on potentially with different parallelism topologies or training configurations. To do this add `--universal-checkpoint` to your DeepSpeed config (json) file


## Conclusion
DeepSpeed Universal Checkpointing simplifies the management of model states, making it easier to save, load, and transfer model states across different training sessions and parallelism techniques. By following the steps outlined in this tutorial, you can integrate Universal Checkpointing into your DeepSpeed applications, enhancing your model training and development workflow.

For more detailed examples and advanced configurations, please refer to the [Megatron-DeepSpeed examples](https://github.com/microsoft/Megatron-DeepSpeed/tree/main/examples_deepspeed/universal_checkpointing).

For technical in-depth of DeepSpeed Universal Checkpointing, please see [arxiv manuscript](https://arxiv.org/abs/2406.18820) and [blog](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ucp/).

Happy training!
