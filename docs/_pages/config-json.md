---
title: "DeepSpeed Configuration JSON"
---

### Batch Size Related Parameters

**Note:** configuring ***train\_batch\_size*** is required.
{: .notice--warning}

***train\_batch\_size***: [integer]

| Value                                                                                                                                                                                                                                                                                                                                                                             | Example |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| The effective training batch size. This is the amount of data samples that leads to one step of model update. ***train\_batch\_size*** is aggregated by the batch size that a single GPU processes in one forward/backward pass (a.k.a., ***train\_step\_batch\_size***),  the gradient accumulation steps (a.k.a., ***gradient\_accumulation\_steps***), and the number of GPUs. | `32`    |


***train\_micro\_batch\_size\_per\_gpu***: [integer]

| Description                                                                                                                                                                                                                                                                                                                    | Default                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------ |
| Batch size to be processed by one GPU in one step (without gradient accumulation). When specified, ***gradient\_accumulation\_steps*** is automatically calculated using ***train\_batch\_size*** and number of GPUs. Should not be concurrently specified with ***gradient\_accumulation\_steps*** in the configuration JSON. | ***train\_batch\_size*** value |

***gradient\_accumulation\_steps***: [integer]

| Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Default |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Number of training steps to accumulate gradients before averaging and applying them. This feature is sometimes useful to improve scalability since it results in less frequent communication of gradients between steps. Another impact of this feature is the ability to train with larger batch sizes per GPU. When specified, ***train\_step\_batch\_size*** is automatically calculated using ***train\_batch\_size*** and number of GPUs. Should not be concurrently specified with ***train\_step\_batch\_size*** in the configuration JSON. | `1`     |



### Optimizer Parameters

***optimizer***: [dictionary]

| Fields | Value                                                                                                                                                                                                                                                                                        | Example                      |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| type   | The optimizer name. DeepSpeed natively supports **Adam**, **AdamW**, **OneBitAdam**, and **Lamb** optimizers (See [here](https://deepspeed.readthedocs.io/en/latest/optimizers.html) for details) and will import other optimizers from [torch](https://pytorch.org/docs/stable/optim.html). | `"Adam"`                     |
| params | Dictionary of parameters to instantiate optimizer. The parameter names must match the optimizer constructor signature (e.g., for [Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam)).                                                                                       | `{"lr": 0.001, "eps": 1e-8}` |

  Example of ***optimizer*** with Adam

```json
"optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [
        0.8,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  }
```
The Adam optimizer also supports the following two params keys/values in addition to the standard parameters from [torch.optim.Adam](https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam):

| "params" key  | Description                                                                 | Default |
| ------------- | --------------------------------------------------------------------------- | ------- |
| torch\_adam   | Use torch's implementation of adam instead of our fused adam implementation | false   |
| adam\_w\_mode | Apply L2 regularization (also known as AdamW)                               | true    |

  Another example of ***optimizer*** with 1-bit Adam

```json
"optimizer": {
    "type": "OneBitAdam",
    "params": {
      "lr": 0.001,
      "betas": [
        0.8,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 3e-7,
      "freeze_step": 400,
      "cuda_aware": false,
      "comm_backend_name": "nccl"
    }
  }
```

The 1-bit Adam optimizer supports the following three params keys/values in addition to the standard Adam (learn more in our [tutorial](/tutorials/onebit-adam/)):

| "params" key  | Description                                                                 | Default |
| ------------- | --------------------------------------------------------------------------- | ------- |
| freeze\_step   | Number of warm up steps before 1-bit compression gets applied to the communication | 100000   |
| cuda\_aware | To indicate that the underlying MPI library supports CUDA-Aware communication         | false    |
| comm\_backend\_name | To indicate which backend implementation to use                               | "nccl"   |

### Scheduler Parameters

***scheduler***: [dictionary]

| Fields | Value                                                                                                                      | Example                                        |
| ------ | -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| type   | The scheduler name. See [here](https://deepspeed.readthedocs.io/en/latest/schedulers.html) for list of support schedulers. | `"WarmupLR"`                                   |
| params | Dictionary of parameters to instantiate scheduler. The parameter names should match scheduler constructor signature.       | `{"warmup_min_lr": 0, "warmup_max_lr": 0.001}` |

Example of ***scheduler***

```json
 "scheduler": {
      "type": "WarmupLR",
      "params": {
          "warmup_min_lr": 0,
          "warmup_max_lr": 0.001,
          "warmup_num_steps": 1000
      }
  }
```

### Communication options

***fp32\_allreduce***: [boolean]

| Description                                                    | Default |
| -------------------------------------------------------------- | ------- |
| During gradient averaging perform allreduce with 32 bit values | `false` |

***prescale\_gradients***: [boolean]

| Description                            | Default |
| -------------------------------------- | ------- |
| Scale gradients before doing allreduce | `false` |

***gradient_predivide_factor***: [float]

| Description                                                                                                                                       | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Before gradient averaging predivide gradients by a specified factor, can sometimes help with fp16 stability when scaling to large numbers of GPUs | `1.0`   |

***sparse\_gradients***: [boolean]

| Description                                                                                                              | Default |
| ------------------------------------------------------------------------------------------------------------------------ | ------- |
| Enable sparse compression of [torch.nn.Embedding](https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding) gradients. | `false` |

### FP16 training options

**Note:** this mode cannot be combined with the `amp` mode described below.
{: .notice--warning}

***fp16***: [dictionary]

| Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Default |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Configuration for using mixed precision/FP16 training that leverages [NVIDIA's Apex package](https://nvidia.github.io/apex/). An example, including the available dictionary keys is illustrated below. NOTE: this does not use Apex's AMP mode that allows for more flexibility in mixed precision training modes, this mode is similar to AMP's O2 mode. Please see AMP support below if you want to use more complex mixed precision modes. If you want to use ZeRO (currently) you must use this mode. | None    |

```json
"fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 32,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
}
```

***fp16:enabled***: [boolean]

| Description                                                                            | Default |
| -------------------------------------------------------------------------------------- | ------- |
| ***enabled*** is a **fp16** parameter indicating whether or not FP16 training enabled. | `false` |

***fp16:loss\_scale***: [float]

| Description                                                                                                                                                                                                                  | Default |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| ***loss\_scale*** is a ***fp16*** parameter representing the loss scaling value for FP16 training. The default value of 0.0 results in dynamic loss scaling, otherwise the value will be used for static fixed loss scaling. | `0.0`   |

***fp16:initial\_scale\_power***: [integer]

| Description                                                                                                                                                                                       | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| ***initial\_scale\_power*** is a **fp16** parameter representing the power of the initial dynamic loss scale value. The actual loss scale is computed as 2<sup>***initial\_scale\_power***</sup>. | `32`    |

***fp16:loss\_scale\_window***: [integer]

| Description                                                                                                                       | Default |
| --------------------------------------------------------------------------------------------------------------------------------- | ------- |
| ***loss\_scale\_window*** is a **fp16** parameter representing the window over which to raise/lower the dynamic loss scale value. | `1000`  |

***fp16:hysteresis***: [integer]

| Description                                                                                    | Default |
| ---------------------------------------------------------------------------------------------- | ------- |
| ***hysteresis*** is a **fp16** parameter representing the delay shift in dynamic loss scaling. | `2`     |

***fp16:min\_loss\_scale***: [integer]

| Description                                                                                        | Default |
| -------------------------------------------------------------------------------------------------- | ------- |
| ***min\_loss\_scale*** is  a **fp16** parameter representing the minimum dynamic loss scale value. | `1000`  |

### Automatic mixed precision (AMP) training options

**Note:** this mode cannot be combined with the `fp16` mode described above. In addition this mode is not currently compatible with ZeRO.
{: .notice--warning}

***amp***: [dictionary]

| Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Default |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Configuration for using automatic mixed precision (AMP) training that leverages [NVIDIA's Apex AMP package](https://nvidia.github.io/apex/). An example, including the available dictionary keys is illustrated below. Is not compatible with `fp16` mode above or ZeRO. Any parameters outside of "enabled" will be passed to AMP's initialize call, see the API and descriptions here at the [apex.amp.initialize documentation](https://nvidia.github.io/apex/amp.html#apex.amp.initialize). | None    |

```json
"amp": {
    "enabled": true,
    ...
    "opt_level": "O1",
    ...
}
```

***amp:enabled***: [boolean]

| Description                                                                              | Default |
| ---------------------------------------------------------------------------------------- | ------- |
| ***enabled*** is an **amp** parameter indicating whether or not AMP training is enabled. | `false` |

***amp params***: [various]

| Description                                                                                                                                                                                                            | Default |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Any parameters outside of "enabled" will be passed to AMP's initialize call, see the API and descriptions here at the [apex.amp.initialize documentation](https://nvidia.github.io/apex/amp.html#apex.amp.initialize). | None    |

### Gradient Clipping

***gradient\_clipping***: [float]

| Description                         | Default |
| ----------------------------------- | ------- |
| Enable gradient clipping with value | `0`     |



### ZeRO Optimizations for FP16 Training

Enabling and configuring ZeRO memory optimizations
```json
  "zero_optimization": {
    "stage": [0|1|2|3],
    "allgather_partitions": [true|false],
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": [true|false],
    "reduce_bucket_size": 5e8,
    "contiguous_gradients" : [true|false],
    "cpu_offload": [true|false],
    "cpu_offload_params" : [true|false],
    "cpu_offload_use_pin_memory" : [true|false],
    "stage3_max_live_parameters" : 1e9,
    "stage3_max_reuse_distance" : 1e9,
    "stage3_prefetch_bucket_size" : 5e8,
    "stage3_param_persistence_threshold" : 1e6,
    "sub_group_size" : 1e12,
    "elastic_checkpoint" : [true|false]
    }
```

***zero\_optimization***: [dictionary]

| Description                                                                                               | Default |
| --------------------------------------------------------------------------------------------------------- | ------- |
| Enable ZeRO memory optimization wrapper for FP16 Training. Currently compatible only with Adam optimizer. | `false` |

***stage***: [integer]

| Description                                                                                                                                                           | Default |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Chooses different stages of ZeRO Optimizer. Stage 0, 1, 2, and 3 refer to disabled, optimizer state partitioning, and optimizer+gradient state partitioning, and optimizer+gradient+parameter partitioning, respectively. | `0`     |

***allgather_partitions***: [boolean]

| Description                                                                                                                                      | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| Chooses between allgather collective or a series of broadcast collectives to gather updated parameters from all the GPUs at the end of each step | `true`  |

***allgather_bucket_size***: [boolean]

| Description                                                                                                  | Default |
| ------------------------------------------------------------------------------------------------------------ | ------- |
| Number of elements allgathered at a time. Limits the memory required for the allgather for large model sizes | `5e8`   |

***overlap_comm***: [boolean]

| Description                                                                  | Default |
| ---------------------------------------------------------------------------- | ------- |
| Attempts to overlap the reduction of the gradients with backward computation | `false` |

***reduce_scatter***: [boolean]

| Description                                                             | Default |
| ----------------------------------------------------------------------- | ------- |
| Uses reduce or reduce scatter instead of allreduce to average gradients | `true`  |

***reduce_bucket_size***: [boolean]

| Description                                                                                                         | Default |
| ------------------------------------------------------------------------------------------------------------------- | ------- |
| Number of elements reduced/allreduced at a time. Limits the memory required for the allgather for large model sizes | `5e8`   |

***contiguous_gradients***: [boolean]

| Description                                                                                                                                                     | Default |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Copies the gradients to a contiguous buffer as they are produced. Avoids memory fragmentation during backward pass. Only useful when running very large models. | `False` |

***cpu_offload***: [boolean]

| Description                                                                                                              | Default |
| ------------------------------------------------------------------------------------------------------------------------ | ------- |
| Enable offloading of optimizer memory and computation to CPU. This frees up GPU memory for larger models or batch sizes. | `False` |

***cpu_offload_params***: [boolean]

| Description                                                                                                                       | Default |
| --------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Enable offloading of model parameters to CPU. This frees up GPU memory for larger models or batch sizes. Valid only with stage 3. | `False` |

***cpu_offload_use_pin_memory***: [boolean]

| Description                                                                               | Default |
| ----------------------------------------------------------------------------------------- | ------- |
| Use pinned CPU memory when offloading. Can improve performance. Valid only with stage 3.  | `False` |

***stage3_max_live_parameters***: [integer]

| Description                                                                                                                           | Default |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| The maximum number of parameters resident per GPU before releasing. Smaller values use less memory, but perform more communication. | `1e9`   |

***stage3_max_reuse_distance***: [integer]

| Description                                                                                                      | Default |
| ---------------------------------------------------------------------------------------------------------------- | ------- |
| Do not release a parameter if it will be reused within this threshold of parameters. Smaller values use less memory, but perform more communication. | `1e9`   |

***stage3_prefetch_bucket_size***: [integer]

| Description                                                                                                                     | Default |
| ------------------------------------------------------------------------------------------------------------------------------- | ------- |
| The size of the fixed buffer for prefetching parameters. Smaller values use less memory, but can increase stalls due to communication. | `5e8`   |


***stage3_param_persistence_threshold***: [integer]
| Description                                                                                                                                                          | Default |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Do not partition parameters smaller than this threshold. Smaller values use less memory, but can greatly increase communication (especially latency-bound messages). | `1e6`   |


### Logging

***steps\_per\_print***: [integer]

| Description                    | Default |
| ------------------------------ | ------- |
| Print train loss every N steps | `10`    |

***wall\_clock\_breakdown***: [boolean]

| Description                                                             | Default |
| ----------------------------------------------------------------------- | ------- |
| Enable timing of the latency of forward/backward/update training phases | `false` |

***dump_state***: [boolean]

| Description                                                          | Default |
| -------------------------------------------------------------------- | ------- |
| Print out state information of DeepSpeed object after initialization | `false` |

### Flops Profiler
```json
{
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": true,
    }
}
```
***enabled***: [boolean]

| Description                 | Default |
| --------------------------- | ------- |
| Enables the flops profiler. | `false` |

***profile\_step***: [integer]

| Description                                                                                                     | Default |
| --------------------------------------------------------------------------------------------------------------- | ------- |
| The global training step at which to profile. Note that warm up steps are needed for accurate time measurement. | `1`     |

***module\_depth***: [integer]

| Description                                                                                                                                                            | Default |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| The depth of the model at which to print the aggregated module information. When set to `-1`, it prints information on the innermost modules (with the maximum depth). | `-1`    |

***top\_modules***: [integer]

| Description                                                                  | Default |
| ---------------------------------------------------------------------------- | ------- |
| Limits the aggregated profile output to the number of top modules specified. | `3`     |

***detailed***: [boolean]

| Description                                  | Default |
| -------------------------------------------- | ------- |
| Whether to print the detailed model profile. | `true`  |

### Activation Checkpointing
```json
  "activation_checkpointing": {
    "partition_activations": false,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
    }
```
***partition\_activations***: [boolean]

| Description                                                   | Default |
| ------------------------------------------------------------- | ------- |
| Enables partition activation when used with model parallelism | `false` |

***cpu\_checkpointing***: [boolean]

| Description                                                                 | Default |
| --------------------------------------------------------------------------- | ------- |
| Offloads partitioned activations to CPU if partition_activations is enabled | `false` |


***contiguous\_memory\_optimization***: [boolean]

| Description                                                          | Default |
| -------------------------------------------------------------------- | ------- |
| Copies partitioned activations so that they are contiguous in memory | `false` |

***number_checkpoints***: [integer]

| Description                                                                                              | Default |
| -------------------------------------------------------------------------------------------------------- | ------- |
| Total number of activation checkpoints used to allocate memory buffer for contiguous_memoty_optimization | `None`  |

***synchronize\_checkpoint\_boundary***: [boolean]

| Description                                                   | Default |
| ------------------------------------------------------------- | ------- |
| Inserts torch.cuda.synchronize() at each checkpoint boundary. | `false` |


***profile***: [boolean]

| Description                                                     | Default |
| --------------------------------------------------------------- | ------- |
| Logs the forward and backward time for each checkpoint function | `false` |

### Sparse Attention

***sparse\_attention***: [dictionary]

| Fields                           | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Example           |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| mode                             | A string determining sparsity structure type. Deepspeed currently supports `"dense"`, `"fixed"`, `"bigbird"`, `"bslongformer"`, and `"variable"`.                                                                                                                                                                                                                                                                                                                                                              | `"fixed"`         |
| block                            | An integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.                                                                                                                                                                                                                                                                                                              | 16                |
| different\_layout\_per\_head     | A boolean determining if each head should be assigned a different sparsity layout; this will be satisfied based on availability.                                                                                                                                                                                                                                                                                                                                                                               | false             |
| num\_local\_blocks               | An integer determining the number of random blocks in each block row; only used in `"fixed"` mode.                                                                                                                                                                                                                                                                                                                                                                                                             | 4                 |
| num\_global\_blocks              | An integer determining how many consecutive blocks in a local window is used as the representative of the window for global attention; used in `"fixed"` and `"bigbird"` modes.                                                                                                                                                                                                                                                                                                                                | 1                 |
| attention                        | A string determining attention type. Attention can be `"unidirectional"`, such as autoregressive models, in which tokens attend only to tokens appear before them in the context. Considering that, the upper triangular of attention matrix is empty. Or it can be `"bidirectional"`, such as BERT, in which tokens can attend to any other tokens before or after them. Then, the upper triangular part of the attention matrix is mirror of the lower triangular; used in `"fixed"` and `"variable"` modes. | `"bidirectional"` |
| horizontal\_global\_attention    | A boolean determining if blocks that are global representative of a local window, also attend to all other blocks. This is valid only if attention type is `"bidirectional"`. Looking at the attention matrix, that means global attention not only includes the vertical blocks, but also horizontal blocks; used in `"fixed"` and `"variable"` modes.                                                                                                                                                        | false             |
| num\_different\_global\_patterns | An integer determining number of different global attentions layouts. While global attention can be fixed by which block/s are representative of any local window, since there are multi-heads, each head can use a different global representative; used only in `"fixed"` mode.                                                                                                                                                                                                                              | 4                 |
| num\_random\_blocks              | An integer determining the number of random blocks in each block row; used in `"variable"` and `"bigbird"` modes.                                                                                                                                                                                                                                                                                                                                                                                              | 0                 |
| local\_window\_blocks            | A list of integers determining the number of blocks in each local attention window. It assumes first number determines # of blocks in the first local window, second the second window, ..., and the last number determines the number of blocks in the remaining local windows; only used in `"variable"` mode.                                                                                                                                                                                               | [4]               |
| global\_block\_indices           | A list of integers determining which blocks are considered as global attention. Given indices, determine the blocks that all other token blocks attend to and they attend to all other token blocks. Notice that if global\_block\_end\_indices parameter is set, this parameter is used as starting index of each global window; used in `"variable"` and `"bslongformer"` modes.                                                                                                                             | [0]               |
| global\_block\_end\_indices      | A list of integers determining end indices of global window blocks. By default this is not used. But if it is set, it must have the same size of global\_block\_indices parameter, and combining this two parameters, for each index i, blocks from global\_block\_indices[i] to global\_block\_end\_indices[i], exclusive, are considered as global attention; used in `"variable"` and `"bslongformer"` modes.                                                                                               | None              |
| num\_sliding\_window\_blocks     | An integer determining the number of blocks in sliding local attention window; used in `"bigbird"` and `"bslongformer"` modes.                                                                                                                                                                                                                                                                                                                                                                                 | 3                 |

  Example of ***sparse\_attention***

```json
  "sparse_attention": {
    "mode": "fixed",
    "block": 16,
    "different_layout_per_head": true,
    "num_local_blocks": 4,
    "num_global_blocks": 1,
    "attention": "bidirectional",
    "horizontal_global_attention": false,
    "num_different_global_patterns": 4,
    "num_random_blocks": 0,
    "local_window_blocks": [4],
    "global_block_indices": [0],
    "global_block_end_indices": None,
    "num_sliding_window_blocks": 3
  }
```
