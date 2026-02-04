---
title: "AutoTP Training API"
tags: training tensor-parallelism
---

# AutoTP Training API

This tutorial covers the **AutoTP Training API** for combining tensor parallelism with ZeRO optimization during training. For inference-only tensor parallelism, see [Automatic Tensor Parallelism for HuggingFace Models](automatic-tensor-parallelism).

## Contents
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Custom Layer Specifications](#custom-layer-specifications)
- [Limitations](#limitations)

## Introduction

The AutoTP Training API enables hybrid parallelism by combining:
- **Tensor Parallelism (TP)**: Split model weights across GPUs within a node
- **Data Parallelism (DP)**: Replicate model across GPU groups
- **ZeRO Optimization**: Memory-efficient optimizer states (Stage 0, 1, or 2)

Tensor parallelism (TP) splits the computations and parameters of large layers
across multiple GPUs so each rank holds only a shard of the weight matrix. This
is an efficient way to train large-scale transformer models by reducing per-GPU
memory pressure while keeping the layer math distributed across the TP group.


## Quick Start

### Basic Usage

AutoTP training can be enabled entirely through the DeepSpeed config. When
`tensor_parallel` is set in the config, `deepspeed.initialize(...)` applies
AutoTP sharding during engine initialization, so the training loop itself does
not change.

```python
import torch
import deepspeed

# 1. Create your model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# 2. Define the DeepSpeed config with tensor_parallel settings
ds_config = {
    "train_micro_batch_size_per_gpu": 1,
    "zero_optimization": {"stage": 2},
    "bf16": {"enabled": True},
    "tensor_parallel": {"autotp_size": 4},
}

# 3. Initialize DeepSpeed with AutoTP + ZeRO
engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config,
    mpu=mpu  # Model parallel unit (optional if you provide tp_group elsewhere)
)

# 4. Train as usual
for batch in dataloader:
    outputs = engine(input_ids=batch["input_ids"], labels=batch["labels"])
    engine.backward(outputs.loss)
    engine.step()
```

Compatibility note: For backward compatibility, you can still call
`set_autotp_mode(training=True)` and `deepspeed.tp_model_init(...)`, but they
are not required when the DeepSpeed config provides the necessary
`tensor_parallel` settings.

### Preset-based Sharding

If your model matches a built-in preset, set `tensor_parallel.preset_model` in the DeepSpeed config:

```json
{
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 1,
    "bf16": { "enabled": true },
    "zero_optimization": { "stage": 2 },
    "tensor_parallel": {
        "autotp_size": 4,
        "preset_model": "llama"
    }
}
```

For the list of available presets, see [supported models](../code-docs/training#autotp-supported-models).



## Custom Patterns

If you are training a custom model, define regex-based patterns and partition rules in `tensor_parallel.partition_config`:

```json
{
    "tensor_parallel": {
        "autotp_size": 4,
        "partition_config": {
            "use_default_specs": false,
            "layer_specs": [
                {
                    "patterns": [".*\\.o_proj\\.weight$", ".*\\.down_proj\\.weight$"],
                    "partition_type": "row"
                },
                {
                    "patterns": [".*\\.[qkv]_proj\\.weight$"],
                    "partition_type": "column"
                },
                {
                    "patterns": [".*\\.gate_up_proj\\.weight$"],
                    "partition_type": "column",
                    "shape": [2, -1],
                    "partition_dim": 0
                }
            ]
        }
    }
}
```

## Custom Layer Specifications

For models not covered by presets, define custom layer specs:

```json
{
    "tensor_parallel": {
        "autotp_size": 4,
        "partition_config": {
            "use_default_specs": false,
            "layer_specs": [
                {
                    "patterns": [".*\\.o_proj\\.weight$", ".*\\.down_proj\\.weight$"],
                    "partition_type": "row"
                },
                {
                    "patterns": [".*\\.[qkv]_proj\\.weight$"],
                    "partition_type": "column"
                },
                {
                    "patterns": [".*\\.gate_up_proj\\.weight$"],
                    "partition_type": "column",
                    "shape": [2, -1],
                    "partition_dim": 0
                }
            ]
        }
    }
}
```

### Fused Layers with Unequal Sub-parameters (GQA)

For Grouped Query Attention with different Q/K/V sizes:

```json
{
    "tensor_parallel": {
        "partition_config": {
            "layer_specs": [
                {
                    "patterns": [".*\\.qkv_proj\\.weight$"],
                    "partition_type": "column",
                    "shape": [[q_size, kv_size, kv_size], -1],
                    "partition_dim": 0
                }
            ]
        }
    }
}
```

## Limitations

1. **ZeRO Stage 3 not supported**: AutoTP currently only works with ZeRO stages 0, 1, and 2.

2. **TP size must divide model dimensions**: The tensor parallel size must evenly divide the attention head count and hidden dimensions.


## See Also

- [Automatic Tensor Parallelism for Inference](automatic-tensor-parallelism)
- [ZeRO Optimization](zero)
- [DeepSpeed Configuration](config-json)
