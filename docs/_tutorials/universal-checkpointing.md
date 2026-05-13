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

 - Use the [ds_to_universal.py](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/checkpoint/ds_to_universal.py) script provided by DeepSpeed.
 - Specify the path to your ZeRO checkpoint and the desired output path for the Universal checkpoint.

```bash
python ds_to_universal.py --input_folder /path/to/zero/checkpoint --output_folder /path/to/universal/checkpoint
```

This script will process the ZeRO checkpoint and generate a new checkpoint in the Universal format. Pass `--help` flag to see other options.

### Step 3: Resume Training with Universal Checkpoint
With the Universal checkpoint ready, you can now resume training on potentially with different parallelism topologies or training configurations. To do this add `--universal-checkpoint` to your DeepSpeed config (json) file


## Universal Checkpointing with AutoTP (Automatic Tensor Parallelism)

DeepSpeed AutoTP (Automatic Tensor Parallelism) can produce checkpoints that are compatible with Universal
Checkpoint conversion and restore.

### What gets saved

When AutoTP is enabled, DeepSpeed will attach Universal Checkpoint metadata (`UNIVERSAL_CHECKPOINT_INFO`)
to the saved training checkpoint. This metadata describes how tensor-parallel parameters were partitioned
(e.g. row-parallel vs column-parallel, replicated parameters, and fused/sub-parameter layouts).

This enables:
- converting a TP-sharded training checkpoint into a Universal checkpoint via `ds_to_universal.py`
- restoring the checkpoint correctly even when TP partitioning uses fused weights (e.g. QKV)

### Enablement

AutoTP is enabled by setting `tensor_parallel` in your DeepSpeed config:

```json
{
  "zero_optimization": { "stage": 2 },
  "bf16": { "enabled": true },
  "tensor_parallel": { "autotp_size": 4 }
}
```

Save a regular DeepSpeed checkpoint during training:

```
engine.save_checkpoint(save_dir, tag=tag)
```

### Conversion

Convert the saved DeepSpeed checkpoint to the universal format:

```
python deepspeed/checkpoint/ds_to_universal.py \
  --input_folder /path/to/ds_checkpoint \
  --output_folder /path/to/universal_checkpoint
```


## Universal Checkpointing with AutoEP (Automatic Expert Parallelism)

AutoEP checkpoints are saved as regular DeepSpeed checkpoints. With AutoEP enabled, DeepSpeed writes the routed expert weights (`w1`, `w2`, and `w3`) into per-expert files named like `layer_<moe_layer_id>_expert_<global_expert_id>_mp_rank_<NN>_model_states.pt`. The regular model checkpoint records AutoEP metadata in `ds_autoep_layers`; older checkpoints may use the legacy `autoep_layers` key. Router, gate, shared-expert, and other non-routed-expert parameters stay in the regular `mp_rank_*_model_states.pt` files and use the standard Universal Checkpointing path.

Use ZeRO Stage 1 or ZeRO Stage 2 for the current AutoEP Universal Checkpoint conversion path:

```json
{
  "zero_optimization": { "stage": 2 },
  "expert_parallel": {
    "enabled": true,
    "autoep_size": 4,
    "preset_model": "mixtral"
  }
}
```

Save the checkpoint through the normal DeepSpeed API:

```python
engine.save_checkpoint(save_dir, tag=tag)
```

Regular AutoEP checkpoint load requires the target run to use the same `autoep_size` as the save run. To change `autoep_size` for the same AutoEP-detected model topology, convert the saved checkpoint to Universal Checkpoint format:

```bash
python deepspeed/checkpoint/ds_to_universal.py \
  --input_folder /path/to/ds_checkpoint \
  --output_folder /path/to/universal_checkpoint
```

During conversion, `ds_to_universal.py` reads `ds_autoep_layers` (or the legacy `autoep_layers` key), consolidates each AutoEP layer's routed expert files, and writes the full expert tensors to paths such as `zero/<expert_key_prefix>.w1/fp32.pt`. These files are tagged with `is_expert_param` and `ep_num_experts`, which are the load-time signals used for AutoEP expert resharding. When matching expert optimizer shards are available, the converter also writes optimizer state files such as `exp_avg.pt` and `exp_avg_sq.pt` next to the converted parameter.

Load the converted checkpoint through the Universal Checkpoint path:

```json
{
  "checkpoint": {
    "load_universal": true
  },
  "expert_parallel": {
    "enabled": true,
    "autoep_size": 2,
    "preset_model": "mixtral"
  }
}
```

```python
engine.load_checkpoint("/path/to/universal_checkpoint", tag=tag)
```

In the Universal Checkpoint load path, AutoEP routed experts are restored from the `zero/` parameter layout rather than from the regular `layer_*_expert_*_model_states.pt` files. The target run's AutoEP process group supplies the load-side expert-parallel rank and size. For each tagged expert tensor, the loader slices the saved expert dimension by `ep_rank` and `ep_size` when `ep_size > 1`.

The target model still needs to expose matching AutoEP parameter names and compatible shapes, for example `<module_path>.experts.w1`, `<module_path>.experts.w2`, and `<module_path>.experts.w3`. Universal Checkpointing changes the expert-parallel sharding for matching tensors; it does not translate between different model families, different module paths, or arbitrary expert parameter names. The target AutoEP configuration must also be valid before checkpoint loading, including `autoep_size` divisibility by the target stage size and by every detected target layer's expert count.

Topology changes are limited to `autoep_size` resharding for matching AutoEP-managed expert parameters. For every AutoEP layer in the checkpoint, the saved `ep_num_experts` must be divisible by the target `autoep_size` when the target `ep_size > 1`. For example, an 8-expert checkpoint can load with target `autoep_size` values of 1, 2, 4, or 8, but not 3. With `autoep_size=1`, the expert tensor is not sliced, but the target parameter must still have the compatible full expert shape.

Current limits and failure cases:

- ZeRO Stage 3 AutoEP Universal Checkpoint conversion is not supported. When AutoEP metadata is present, the converter raises `NotImplementedError` with the message that "AutoEP currently requires ZeRO Stage 1 or 2."
- For ZeRO Stage 1 and ZeRO Stage 2 conversion, expert checkpoint files without `ds_autoep_layers` / `autoep_layers` metadata raise a `RuntimeError`.
- Existing DeepSpeed MoE or Megatron-DeepSpeed expert checkpoint files may share the `layer_<moe_layer_id>_expert_<global_expert_id>_mp_rank_<NN>_model_states.pt` naming convention, but they use native `deepspeed_moe` expert parameter names and do not carry AutoEP metadata. Loading or converting those checkpoints into AutoEP requires a separate model-specific migration step.
- If AutoEP metadata is present but an expected per-expert model file is missing, conversion raises `FileNotFoundError`.
- More than one `mp_rank_*` expert file for the same `(layer, expert)` pair raises `NotImplementedError`; combined AutoEP + AutoTP topology changes are not documented by this path.
- AutoEP optimizer-state consolidation is best effort. It succeeds for the usual ZeRO Stage 1 / 2 AutoEP training checkpoints that include matching expert optimizer shards. If `expp_rank_*_mp_rank_*_optim_states.pt` files or matching state entries are absent, the converter still writes the model parameter `fp32.pt` files and skips unavailable optimizer state files.


## Conclusion
DeepSpeed Universal Checkpointing simplifies the management of model states, making it easier to save, load, and transfer model states across different training sessions and parallelism techniques. By following the steps outlined in this tutorial, you can integrate Universal Checkpointing into your DeepSpeed applications, enhancing your model training and development workflow.

For more detailed examples and advanced configurations, please refer to the [Megatron-DeepSpeed examples](https://github.com/deepspeedai/Megatron-DeepSpeed/tree/main/examples_deepspeed/universal_checkpointing).

For technical in-depth of DeepSpeed Universal Checkpointing, please see [arxiv manuscript](https://arxiv.org/abs/2406.18820) and [blog](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ucp/).

Happy training!
