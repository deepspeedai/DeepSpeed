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

Universal Checkpointing uses the same high-level flow for dense models, AutoTP
(Automatic Tensor Parallelism), and AutoEP (Automatic Expert Parallelism): save a
regular DeepSpeed ZeRO checkpoint, convert that checkpoint to Universal format,
then load it with `checkpoint.load_universal` enabled.

### Step 1: Create ZeRO Checkpoint

Start by creating a regular DeepSpeed checkpoint from a run that uses
[ZeRO](/tutorials/zero/) (Zero Redundancy Optimizer). Use the normal DeepSpeed
checkpoint API from your training script:

```python
engine.save_checkpoint(save_dir, tag=tag)
```

This is the same save call used for AutoTP and AutoEP training runs. AutoTP
checkpoints include Universal Checkpoint metadata that describes tensor-parallel
parameter layouts. AutoEP checkpoints also use the normal save API; AutoEP's
expert-specific layout is described in the AutoEP requirements section below.

### Step 2: Convert ZeRO Checkpoint to Universal Format

Once you have a ZeRO checkpoint, convert it to Universal format with the
`ds_to_universal.py` script provided by DeepSpeed:

```bash
python deepspeed/checkpoint/ds_to_universal.py \
  --input_folder /path/to/ds_checkpoint \
  --output_folder /path/to/universal_checkpoint
```

This script processes the saved ZeRO checkpoint and writes a Universal
checkpoint to the output folder. Pass the `--help` flag to see additional
options.

For AutoTP checkpoints, the converter uses the saved Universal Checkpoint
metadata (`UNIVERSAL_CHECKPOINT_INFO`) to reconstruct tensor-parallel parameters
correctly, including row-parallel, column-parallel, replicated, fused, and
sub-parameter layouts.

### Step 3: Resume Training with Universal Checkpoint

With the Universal checkpoint ready, resume training by enabling Universal
Checkpoint loading in your DeepSpeed config:

```json
{
  "checkpoint": {
    "load_universal": true
  }
}
```

Then load the converted checkpoint through the normal DeepSpeed checkpoint API:

```python
engine.load_checkpoint("/path/to/universal_checkpoint", tag=tag)
```

The target run still needs the DeepSpeed parallelism configuration that matches
the model and topology you want to use for resumed training.

### AutoEP Requirements and Limitations

AutoEP checkpoints are saved as regular DeepSpeed checkpoints, but routed expert
weights have an additional layout that depends on the ZeRO stage. With ZeRO
Stage 1 or ZeRO Stage 2, DeepSpeed writes the routed expert weights (`w1`,
`w2`, and `w3`) into per-expert files named like
`layer_<moe_layer_id>_expert_<global_expert_id>_mp_rank_<NN>_model_states.pt`,
while router, gate, shared-expert, and other non-routed-expert parameters stay
in the regular `mp_rank_*_model_states.pt` files and use the standard
Universal Checkpointing path. With ZeRO Stage 3, AutoEP checkpoints are
partition-native: no per-expert files are produced, and expert weights are
stored as ZeRO partition shards in the `zero_pp_rank_*_model_states.pt` and
optimizer shard files together with the recorded partition-group metadata. In
both layouts the regular model checkpoint records AutoEP metadata in
`ds_autoep_layers`; older checkpoints may use the legacy `autoep_layers`
key.

Both ZeRO Stage 1/2 and ZeRO Stage 3 AutoEP checkpoints can be converted to
Universal Checkpoint format. For ZeRO Stage 3, `ds_to_universal.py` detects the
partition-native AutoEP metadata in `zero_pp_rank_*_model_states.pt`,
consolidates each expert parameter from its partition shards across the expert
replica group, and writes the same `zero/` parameter layout as the other
stages. ZeRO Stage 3 AutoEP also supports module-only loads
(`load_module_only=True`) and optimizer-state-free loads
(`load_optimizer_states=False`) from both partition shards and Universal
Checkpoint format. After conversion to Universal Checkpoint format, ZeRO Stage 3
AutoEP can load optimizer-including or weights-only/module-only checkpoints at a
different data-parallel world size, a different `autoep_size`, or both, as long
as the target AutoEP topology is valid for the same model parameter names and
expert count. `zero_to_fp32.py` consolidation is not supported for
partition-native AutoEP checkpoints (the script raises `NotImplementedError`;
use `ds_to_universal.py` instead).

During conversion, `ds_to_universal.py` reads `ds_autoep_layers` or the legacy
`autoep_layers` key, consolidates each AutoEP layer's routed expert state (the
per-expert files for ZeRO Stage 1/2, the partition shards for ZeRO Stage 3), and
writes full expert tensors to paths such as `zero/<expert_key_prefix>.w1/fp32.pt`.
These files are tagged with `is_expert_param` and `ep_num_experts`, which are the
load-time signals used for AutoEP expert resharding. When matching expert
optimizer shards are available, the converter also writes optimizer state files
such as `exp_avg.pt` and `exp_avg_sq.pt` next to the converted parameter.

Regular AutoEP checkpoint load requires the target run to use the same
`autoep_size` as the save run. To change `autoep_size` or data-parallel world
size for the same AutoEP-detected model topology, convert the checkpoint to
Universal format and load the Universal checkpoint. For ZeRO Stage 3 AutoEP,
optimizer-including loads reslice routed expert parameters and their Adam
`fp32`, `exp_avg`, and `exp_avg_sq` states using the target runtime topology;
weights-only/module-only loads reslice routed expert parameters and standard
parameters from the universal `fp32.pt` files without requiring optimizer state.

In the Universal Checkpoint load path, AutoEP routed experts are restored from
the `zero/` parameter layout rather than from the regular
`layer_*_expert_*_model_states.pt` files. The target run's AutoEP process group
supplies the load-side expert-parallel rank and size. For each tagged expert
tensor, the loader slices the saved expert dimension by `ep_rank` and `ep_size`
and then applies the target ZeRO partitioning group and padding.

The target model still needs to expose matching AutoEP parameter names and
compatible shapes, for example `<module_path>.experts.w1`,
`<module_path>.experts.w2`, and `<module_path>.experts.w3`. Universal
Checkpointing changes the expert-parallel sharding for matching tensors; it does
not translate between different model families, different module paths, or
arbitrary expert parameter names. The target AutoEP configuration must also be
valid before checkpoint loading: `autoep_size` must divide the target pipeline
stage size (`world_size / pp_size`) and every detected target layer's expert
count.

Topology changes are limited to data-parallel world-size changes and
`autoep_size` resharding for matching AutoEP-managed expert parameters. For
every AutoEP layer in the checkpoint, the saved `ep_num_experts` must be
divisible by the target `autoep_size`. For example, an 8-expert checkpoint can
load with target
`autoep_size` values of 1, 2, 4, or 8, but not 3. With `autoep_size=1`, the expert
tensor is not sliced, but the target parameter must still have the compatible
full expert shape.

Additional AutoEP failure cases:

- For ZeRO Stage 1 and ZeRO Stage 2 conversion, expert checkpoint files without
  `ds_autoep_layers` or `autoep_layers` metadata raise a `RuntimeError`.
- Existing DeepSpeed MoE or Megatron-DeepSpeed expert checkpoint files may share
  the `layer_<moe_layer_id>_expert_<global_expert_id>_mp_rank_<NN>_model_states.pt`
  naming convention, but they use native `deepspeed_moe` expert parameter names
  and do not carry AutoEP metadata. Loading or converting those checkpoints into
  AutoEP requires a separate model-specific migration step.
- If AutoEP metadata is present but an expected per-expert model file is missing,
  conversion raises `FileNotFoundError`.
- More than one `mp_rank_*` expert file for the same `(layer, expert)` pair
  raises `NotImplementedError`; combined AutoEP + AutoTP topology changes are
  not documented by this path.
- AutoEP optimizer-state consolidation is best effort. It succeeds for the usual
  ZeRO Stage 1 or ZeRO Stage 2 AutoEP training checkpoints that include matching
  expert optimizer shards. If `expp_rank_*_mp_rank_*_optim_states.pt` files or
  matching state entries are absent, the converter still writes the model
  parameter `fp32.pt` files and skips unavailable optimizer state files.


## Conclusion
DeepSpeed Universal Checkpointing simplifies the management of model states, making it easier to save, load, and transfer model states across different training sessions and parallelism techniques. By following the steps outlined in this tutorial, you can integrate Universal Checkpointing into your DeepSpeed applications, enhancing your model training and development workflow.

For more detailed examples and advanced configurations, please refer to the [Megatron-DeepSpeed examples](https://github.com/deepspeedai/Megatron-DeepSpeed/tree/main/examples_deepspeed/universal_checkpointing).

For technical in-depth of DeepSpeed Universal Checkpointing, please see [arxiv manuscript](https://arxiv.org/abs/2406.18820) and [blog](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ucp/).

Happy training!
