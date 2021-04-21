---
title: "DeepSpeed Configuration JSON"
---

### Batch Size Related Parameters

**Note:** configuring <i>**train_batch_size**</i> is required.
{: .notice--warning}

<i>**train_batch_size**</i>: [integer]

| Value                                                                                                                                                                                                                                                                                                                                                                             | Example |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| The effective training batch size. This is the amount of data samples that leads to one step of model update. <i>**train_batch_size**</i> is aggregated by the batch size that a single GPU processes in one forward/backward pass (a.k.a., <i>**train_step_batch_size**</i>),  the gradient accumulation steps (a.k.a., <i>**gradient_accumulation_steps**</i>), and the number of GPUs. | `32`    |


<i>**train_micro_batch_size_per_gpu**</i>: [integer]

| Description                                                                                                                                                                                                                                                                                                                    | Default                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------ |
| Batch size to be processed by one GPU in one step (without gradient accumulation). When specified, <i>**gradient_accumulation_steps**</i> is automatically calculated using <i>**train_batch_size**</i> and number of GPUs. Should not be concurrently specified with <i>**gradient_accumulation_steps**</i> in the configuration JSON. | <i>**train_batch_size**</i> value |

<i>**gradient_accumulation_steps**</i>: [integer]

| Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Default |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Number of training steps to accumulate gradients before averaging and applying them. This feature is sometimes useful to improve scalability since it results in less frequent communication of gradients between steps. Another impact of this feature is the ability to train with larger batch sizes per GPU. When specified, <i>**train_step_batch_size**</i> is automatically calculated using <i>**train_batch_size**</i> and number of GPUs. Should not be concurrently specified with <i>**train_step_batch_size**</i> in the configuration JSON. | `1`     |



### Optimizer Parameters

<i>**optimizer**</i>: [dictionary]

| Fields | Value                                                                                                                                                                                                                                                                                        | Example                      |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| type   | The optimizer name. DeepSpeed natively supports **Adam**, **AdamW**, **OneBitAdam**, **Lamb**, and **OneBitLamb** optimizers (See [here](https://deepspeed.readthedocs.io/en/latest/optimizers.html) for details) and will import other optimizers from [torch](https://pytorch.org/docs/stable/optim.html). | `"Adam"`                     |
| params | Dictionary of parameters to instantiate optimizer. The parameter names must match the optimizer constructor signature (e.g., for [Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam)).                                                                                       | `{"lr": 0.001, "eps": 1e-8}` |

  Example of <i>**optimizer**</i> with Adam

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

Another example of <i>**optimizer**</i> with 1-bit Adam specific parameters is as follows.

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

| "params" key        | Description                                                                        | Default |
| ------------------- | ---------------------------------------------------------------------------------- | ------- |
| freeze\_step        | Number of warm up steps before 1-bit compression gets applied to the communication | 100000  |
| cuda\_aware         | To indicate that the underlying MPI library supports CUDA-Aware communication      | false   |
| comm\_backend\_name | To indicate which backend implementation to use                                    | "nccl"  |

Another example of ***optimizer*** with 1-bit LAMB

```json
"optimizer": {
    "type": "OneBitLamb",
    "params": {
      "lr": 11e-3,
      "weight_decay": 0.01,
      "bias_correction": false,
      "max_coeff": 0.3,
      "min_coeff": 0.01,
      "freeze_step": 1000,
      "cuda_aware": false,
      "comm_backend_name": "nccl",
      "coeff_beta": 0.9,
      "factor_max": 4.0,
      "factor_min": 0.5,
      "factor_threshold": 0.1
    }
  }
```

The 1-bit LAMB optimizer supports the following params keys/values in addition to the standard LAMB (learn more in our [tutorial](/tutorials/onebit-lamb/)):

| "params" key  | Description                                                                 | Default |
| ------------- | --------------------------------------------------------------------------- | ------- |
| max\_coeff   | Scaling coefficient upper bound for original LAMB algorithm and 1-bit LAMB's warmup stage   | 10.0   |
| min\_coeff   | Scaling coefficient lower bound for original LAMB algorithm and 1-bit LAMB's warmup stage   | 0.01   |
| freeze\_step   | Number of warm up steps before 1-bit compression gets applied to the communication   | 100000   |
| cuda\_aware | To indicate that the underlying MPI library supports CUDA-Aware communication           | false    |
| comm\_backend\_name | To indicate which backend implementation to use                                 | "nccl"   |
| coeff\_beta | Coefficient used for computing running averages of lamb coefficient                     | 0.9      |
| factor\_max | Maximum value of scaling factor to the frozen lamb coefficient during compression stage | 4.0      |
| factor\_min | Minimum value of scaling factor to the frozen lamb coefficient during compression stage | 0.5      |
| factor\_threshold | Threshold of how much the scaling factor can fluctuate between steps              | 0.1      |

### Scheduler Parameters


DeepSpeed calls the `step()` method of the scheduler at every training step when `model_engine.step()` is executed.

***scheduler***: [dictionary]

| Fields | Value                                                                                                                      | Example                                        |
| ------ | -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| type   | The scheduler name. See [here](https://deepspeed.readthedocs.io/en/latest/schedulers.html) for list of support schedulers. | `"WarmupLR"`                                   |
| params | Dictionary of parameters to instantiate scheduler. The parameter names should match scheduler constructor signature.       | `{"warmup_min_lr": 0, "warmup_max_lr": 0.001}` |

Example of <i>**scheduler**</i>

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

<i>**fp32_allreduce**</i>: [boolean]

| Description                                                    | Default |
| -------------------------------------------------------------- | ------- |
| During gradient averaging perform allreduce with 32 bit values | `false` |

<i>**prescale_gradients**</i>: [boolean]

| Description                            | Default |
| -------------------------------------- | ------- |
| Scale gradients before doing allreduce | `false` |

<i>**gradient_predivide_factor**</i>: [float]

| Description                                                                                                                                       | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Before gradient averaging predivide gradients by a specified factor, can sometimes help with fp16 stability when scaling to large numbers of GPUs | `1.0`   |

<i>**sparse_gradients**</i>: [boolean]

| Description                                                                                                              | Default |
| ------------------------------------------------------------------------------------------------------------------------ | ------- |
| Enable sparse compression of [torch.nn.Embedding](https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding) gradients. | `false` |

### FP16 training options

**Note:** this mode cannot be combined with the `amp` mode described below.
{: .notice--warning}

<i>**fp16**</i>: [dictionary]

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

<i>**fp16:enabled**</i>: [boolean]

| Description                                                                            | Default |
| -------------------------------------------------------------------------------------- | ------- |
| <i>**enabled**</i> is a **fp16** parameter indicating whether or not FP16 training enabled. | `false` |

<i>**fp16:loss_scale**</i>: [float]

| Description                                                                                                                                                                                                                  | Default |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| <i>**loss_scale**</i> is a <i>**fp16**</i> parameter representing the loss scaling value for FP16 training. The default value of 0.0 results in dynamic loss scaling, otherwise the value will be used for static fixed loss scaling. | `0.0`   |

<i>**fp16:initial_scale_power**</i>: [integer]

| Description                                                                                                                                                                                                   | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| <i>**initial_scale_power**</i> is a **fp16** parameter representing the power of the initial dynamic loss scale value. The actual loss scale is computed as 2<sup><i>**initial_scale_power**</i></sup>. | `32`    |

<i>**fp16:loss_scale_window**</i>: [integer]

| Description                                                                                                                       | Default |
| --------------------------------------------------------------------------------------------------------------------------------- | ------- |
| <i>**loss_scale_window**</i> is a **fp16** parameter representing the window over which to raise/lower the dynamic loss scale value. | `1000`  |

<i>**fp16:hysteresis**</i>: [integer]

| Description                                                                                    | Default |
| ---------------------------------------------------------------------------------------------- | ------- |
| <i>**hysteresis**</i> is a **fp16** parameter representing the delay shift in dynamic loss scaling. | `2`     |

<i>**fp16:min_loss_scale**</i>: [integer]

| Description                                                                                        | Default |
| -------------------------------------------------------------------------------------------------- | ------- |
| <i>**min_loss_scale**</i> is  a **fp16** parameter representing the minimum dynamic loss scale value. | `1000`  |

### Automatic mixed precision (AMP) training options

**Note:** this mode cannot be combined with the `fp16` mode described above. In addition this mode is not currently compatible with ZeRO.
{: .notice--warning}

<i>**amp**</i>: [dictionary]

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

<i>**amp:enabled**</i>: [boolean]

| Description                                                                              | Default |
| ---------------------------------------------------------------------------------------- | ------- |
| <i>**enabled**</i> is an **amp** parameter indicating whether or not AMP training is enabled. | `false` |

***amp params***: [various]

| Description                                                                                                                                                                                                            | Default |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Any parameters outside of "enabled" will be passed to AMP's initialize call, see the API and descriptions here at the [apex.amp.initialize documentation](https://nvidia.github.io/apex/amp.html#apex.amp.initialize). | None    |

### Gradient Clipping

<i>**gradient_clipping**</i>: [float]

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
    "offload_param": {
      ...
    },
    "offload_optimizer": {
      ...
    },
    "stage3_max_live_parameters" : 1e9,
    "stage3_max_reuse_distance" : 1e9,
    "stage3_prefetch_bucket_size" : 5e8,
    "stage3_param_persistence_threshold" : 1e6,
    "sub_group_size" : 1e12,
    "elastic_checkpoint" : [true|false],
    "stage3_gather_fp16_weights_on_model_save": [true|false]
    }
```

<i>**zero_optimization**</i>: [dictionary]

| Description                                                                                               | Default |
| --------------------------------------------------------------------------------------------------------- | ------- |
| Enable ZeRO memory optimization wrapper for FP16 Training. Currently compatible only with Adam optimizer. | `false` |

<i>**stage**</i>: [integer]

| Description                                                                                                                                                                                                               | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Chooses different stages of ZeRO Optimizer. Stage 0, 1, 2, and 3 refer to disabled, optimizer state partitioning, and optimizer+gradient state partitioning, and optimizer+gradient+parameter partitioning, respectively. | `0`     |

<i>**allgather_partitions**</i>: [boolean]

| Description                                                                                                                                      | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| Chooses between allgather collective or a series of broadcast collectives to gather updated parameters from all the GPUs at the end of each step | `true`  |

***allgather_bucket_size***: [integer]

| Description                                                                                                  | Default |
| ------------------------------------------------------------------------------------------------------------ | ------- |
| Number of elements allgathered at a time. Limits the memory required for the allgather for large model sizes | `5e8`   |

<i>**overlap_comm**</i>: [boolean]

| Description                                                                  | Default |
| ---------------------------------------------------------------------------- | ------- |
| Attempts to overlap the reduction of the gradients with backward computation | `false` |

<i>**reduce_scatter**</i>: [boolean]

| Description                                                             | Default |
| ----------------------------------------------------------------------- | ------- |
| Uses reduce or reduce scatter instead of allreduce to average gradients | `true`  |

***reduce_bucket_size***: [integer]

| Description                                                                                                         | Default |
| ------------------------------------------------------------------------------------------------------------------- | ------- |
| Number of elements reduced/allreduced at a time. Limits the memory required for the allgather for large model sizes | `5e8`   |

<i>**contiguous_gradients**</i>: [boolean]

| Description                                                                                                                                                     | Default |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Copies the gradients to a contiguous buffer as they are produced. Avoids memory fragmentation during backward pass. Only useful when running very large models. | `False` |


***offload_param***: [dictionary]

| Description                                                                                                                       | Default |
| --------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Enable offloading of model parameters to CPU or NVMe. This frees up GPU memory for larger models or batch sizes. Valid only with stage 3. See [here](#parameter-offloading) for more details. | `False` |

***offload_optimizer***: [dictionary]

| Description                                                                               | Default |
| ----------------------------------------------------------------------------------------- | ------- |
| Enable offloading of optimizer state to CPU or NVMe, and optimizer computation to CPU. This frees up GPU memory for larger models or batch sizes. Valid only with stage 3. See [here](#optimizer-offloading) for more details. | `False` |

***stage3_max_live_parameters***: [integer]

| Description                                                                                                                         | Default |
| ----------------------------------------------------------------------------------------------------------------------------------- | ------- |
| The maximum number of parameters resident per GPU before releasing. Smaller values use less memory, but perform more communication. | `1e9`   |

***stage3_max_reuse_distance***: [integer]

| Description                                                                                                                                          | Default |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Do not release a parameter if it will be reused within this threshold of parameters. Smaller values use less memory, but perform more communication. | `1e9`   |

***stage3_prefetch_bucket_size***: [integer]

| Description                                                                                                                            | Default |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| The size of the fixed buffer for prefetching parameters. Smaller values use less memory, but can increase stalls due to communication. | `5e8`   |


***stage3_param_persistence_threshold***: [integer]

| Description                                                                                                                                                          | Default |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Do not partition parameters smaller than this threshold. Smaller values use less memory, but can greatly increase communication (especially latency-bound messages). | `1e6`   |


***stage3_gather_fp16_weights_on_model_save***: [boolean]

| Description                                                                                                                                                          | Default |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Consolidate the weights before saving the model by `save_fp16_model()`. Since the weights are partitioned across GPUs, they aren't part of `state_dict`, so this function automatically gather the weights when this option is enabled and then saves the fp16 model weights. | `False` |

***cpu_offload***: [boolean]

**Deprecated:** **cpu_offload** is disabled and will be removed in future, please use `offload_optimizer` instead.
{: .notice--warning}

| Description                                                                                                              | Default |
| ------------------------------------------------------------------------------------------------------------------------ | ------- |
| Enable offloading of optimizer memory and computation to CPU. This frees up GPU memory for larger models or batch sizes. Valid only with stage 2.| `False` |


### Parameter offloading
Enabling and configuring ZeRO optimization of parameter offloading to CPU/NVMe. Available only with ZeRO stage 3.
```json
  "offload_param": {
    "device": "[none|cpu|nvme]",
    "nvme_path": "/local_nvme",
    "buffer_count": 5,
    "buffer_size": 1e8,
    "max_in_cpu": 1e9
  }
```
***device***: [string]

| Description                                                                                                                           | Default |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Device memory to offload model parameters. Supported options are `cpu` and `nvme`. | `cpu`   |

***nvme_path***: [string]

| Description                                                                                                                           | Default |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Filesystem path for NVMe device for parameter offloading. | `/local_nvme`   |

***buffer_count***: [integer]

| Description                                                                                                                           | Default |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Number of buffers in buffer pool for parameter offloading to NVMe. | 5  |


***buffer_size***: [integer]

| Description                                                                                                                           | Default |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Size of buffers in buffer pool for parameter offloading to NVMe. | 1e8  |

***max_in_cpu***: [integer]

| Description                                                                                                                           | Default |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Number of parameter elements to maintain in CPU memory when offloading to NVMe is enabled. | 1e9  |

### Optimizer offloading
Enabling and configuring ZeRO optimization of offloading optimizer computation to CPU and state to CPU/NVMe. CPU offloading is available with ZeRO stage 2 or 3. NVMe offloading is available only with ZeRO stage 3.
```json
  "offload_optimizer": {
    "device": "[none|cpu|nvme]",
    "nvme_path": "/local_nvme",
    "buffer_count": 4,
    "pin_memory": [true|false],
    "fast_init": false
  }
```
***device***: [string]

| Description                                                                                                                           | Default |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Device memory to offload optimizer state. Supported options are `cpu` and `nvme`. Optimizer computation is offload to CPU regardless of device option. | `cpu`   |

***nvme_path***: [string]

| Description                                                                                                                           | Default |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Filesystem path for NVMe device for optimizer state offloading. | `/local_nvme`   |

***buffer_count***: [integer]

| Description                                                                                                                           | Default |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Number of buffers in buffer pool for optimizer state offloading to NVMe. This should be at least the number of states maintained per parameter by the optimizer. For example, Adam optimizer has 4 states (parameter, gradient, momentum, and variance). | 4  |


***pin_memory***: [boolean]

| Description                                                                                                                           | Default |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Offload to page-locked CPU memory. This could boost throughput at the cost of extra memory overhead. | `false`  |

***fast_init***: [boolean]

| Description                                                                                                                           | Default |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Enable fast optimizer initialization when offloading to NVMe. | `false`  |

### Logging

<i>**steps_per_print**</i>: [integer]

| Description                    | Default |
| ------------------------------ | ------- |
| Print train loss every N steps | `10`    |

<i>**wall_clock_breakdown**</i>: [boolean]

| Description                                                             | Default |
| ----------------------------------------------------------------------- | ------- |
| Enable timing of the latency of forward/backward/update training phases | `false` |

<i>**dump_state**</i>: [boolean]

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
<i>**enabled**</i>: [boolean]

| Description                 | Default |
| --------------------------- | ------- |
| Enables the flops profiler. | `false` |

<i>**profile_step**</i>: [integer]

| Description                                                                                                     | Default |
| --------------------------------------------------------------------------------------------------------------- | ------- |
| The global training step at which to profile. Note that warm up steps are needed for accurate time measurement. | `1`     |

<i>**module_depth**</i>: [integer]

| Description                                                                                                                                                            | Default |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| The depth of the model at which to print the aggregated module information. When set to `-1`, it prints information on the innermost modules (with the maximum depth). | `-1`    |

<i>**top_modules**</i>: [integer]

| Description                                                                  | Default |
| ---------------------------------------------------------------------------- | ------- |
| Limits the aggregated profile output to the number of top modules specified. | `3`     |

<i>**detailed**</i>: [boolean]

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
<i>**partition_activations**</i>: [boolean]

| Description                                                   | Default |
| ------------------------------------------------------------- | ------- |
| Enables partition activation when used with model parallelism | `false` |

<i>**cpu_checkpointing**</i>: [boolean]

| Description                                                                 | Default |
| --------------------------------------------------------------------------- | ------- |
| Offloads partitioned activations to CPU if partition_activations is enabled | `false` |


<i>**contiguous_memory_optimization**</i>: [boolean]

| Description                                                          | Default |
| -------------------------------------------------------------------- | ------- |
| Copies partitioned activations so that they are contiguous in memory | `false` |

<i>**number_checkpoints**</i>: [integer]

| Description                                                                                              | Default |
| -------------------------------------------------------------------------------------------------------- | ------- |
| Total number of activation checkpoints used to allocate memory buffer for contiguous_memoty_optimization | `None`  |

<i>**synchronize_checkpoint_boundary**</i>: [boolean]

| Description                                                   | Default |
| ------------------------------------------------------------- | ------- |
| Inserts torch.cuda.synchronize() at each checkpoint boundary. | `false` |


<i>**profile**</i>: [boolean]

| Description                                                     | Default |
| --------------------------------------------------------------- | ------- |
| Logs the forward and backward time for each checkpoint function | `false` |

### Sparse Attention

<i>**sparse_attention**</i>: [dictionary]

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

  Example of <i>**sparse_attention**</i>

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
