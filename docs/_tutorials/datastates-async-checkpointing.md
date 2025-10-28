---
title: "DataStates-LLM Checkpointing Engine"
tags: asynchronous checkpointing for minimizing I/O overheads.
---
This tutorial will show how to use [DataStates-LLM](https://github.com/DataStates/datastates-llm) for asynchronous checkpointing. DataStates-LLM introduces a lazy asynchronous checkpointing mechanism tailored for LLMs, aiming to minimize I/O overhead and enhance training efficiency. This tutorial provides a guide on integrating DataStates-LLM with the DeepSpeed framework.

## Overview of DataStates-LLM

DataStates-LLM is designed to address the challenges of frequent checkpointing in LLM training by introducing a lazy asynchronous multi-level approach. It leverages the immutability of model parameters and optimizer states during forward and backward passes to perform non-blocking data transfers, thereby reducing interference with the training process. This method has demonstrated up to 48x faster checkpointing and 2.2x faster end-to-end training times compared to traditional approaches as outlined in [DataStates-LLM: Lazy Asynchronous Checkpointing for Large Language Models](https://arxiv.org/abs/2406.10707).

## Prerequisites

Before integrating DataStates-LLM with DeepSpeed, ensure the following:

- **DeepSpeed Installation**: DeepSpeed should be installed in your environment. If not, refer to the [DeepSpeed Getting Started Guide](https://github.com/microsoft/DeepSpeed/blob/master/docs/_tutorials/getting-started.md) for installation instructions.

- **DataStates-LLM Repository**: Access the DataStates-LLM source code from its [GitHub repository](https://github.com/DataStates/datastates-llm) and follow the installation instructions provided therein.

## Configuring DeepSpeed for DataStates-LLM

To enable DataStates-LLM's asynchronous checkpointing within DeepSpeed, please modify the `deepspeed_config.json` file to include specific configurations under the `datastates_ckpt` section. Below is an example configuration:

```json
{
    // ... other DeepSpeed configuration options
    "datastates_ckpt": {
        "host_cache_size": 16
    }
}
```

### Configuration Parameters

- **`host_cache_size`**: Specifies the amount of pinned host memory (in gigabytes) reserved for asynchronous checkpoint flushing. Adjust this value based on your system's memory capacity and the size of your model checkpoints.

## Implementing DataStates-LLM in Your Training Script

After enabling datastates checkpointing the `deepspeed_config.json`, the frequency of checkpointing can be configured by specifying the number of iterations after which the checkpoints should be captured using command-line parameter ` --save-interval`.

## Limitations and Ongoing Work

1. DataStates-LLM currently only supports the CUDA runtime on Nvidia-based GPUs.


2. DataStates-LLM has only been tested with ZeRO stage-1 without offloading to any other tiers.


3. While the checkpoint layout of datastates matches Huggingface's [safetensor](https://huggingface.co/docs/safetensors/) format, due to pickled objects required by DeepSpeed during restart, it is not fully compatible with safetensor library yet.

4. DataStates-LLM does not yet support universal or elastic checkpointing.


## Questions and Support

Please use the [DataStates-LLM Github repository](https://github.com/DataStates/datastates-llm) for any questions, issues, or feature requests.
