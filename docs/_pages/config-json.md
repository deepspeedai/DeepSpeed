---
title: "DeepSpeed Configuration JSON"
---

### Batch Size Related Parameters

**Note:** <i>**train_batch_size**</i> must be equal to  <i>**train_micro_batch_size_per_gpu**</i> * <i>**gradient_accumulation**</i> * number of GPUs. For simplicity, you can choose to only specify two of the three parameters, the last one will be inferred automatically by DeepSpeed.
{: .notice--warning}

<i>**train_batch_size**</i>: [integer]

| Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Example |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| The effective training batch size. This is the amount of data samples that leads to one step of model update. <i>**train_batch_size**</i> is aggregated by the batch size that a single GPU processes in one forward/backward pass (a.k.a., <i>**train_micro_batch_size_per_gpu**</i>),  the gradient accumulation steps (a.k.a., <i>**gradient_accumulation_steps**</i>), and the number of GPUs. Can be omitted if both <i>**train_micro_batch_size_per_gpu**</i> and <i>**gradient_accumulation_steps**</i> are provided. | `32`    |


<i>**train_micro_batch_size_per_gpu**</i>: [integer]

| Description                                                                                                                                                                                    | Default                           |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| Batch size to be processed by one GPU in one step (without gradient accumulation). Can be omitted if both <i>**train_batch_size**</i> and <i>**gradient_accumulation_steps**</i> are provided. | <i>**train_batch_size**</i> value |

<i>**gradient_accumulation_steps**</i>: [integer]

| Description                                                                                                                                                                                                                                                                                                                                                                                                                     | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Number of training steps to accumulate gradients before averaging and applying them. This feature is sometimes useful to improve scalability since it results in less frequent communication of gradients between steps. Another impact of this feature is the ability to train with larger batch sizes per GPU. Can be omitted if both <i>**train_batch_size**</i> and <i>**train_micro_batch_size_per_gpu**</i> are provided. | `1`     |



### Optimizer Parameters

<i>**optimizer**</i>: [dictionary]

| Fields | Value                                                                                                                                                                                                                                                                                                        | Example                      |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- |
| type   | The optimizer name. DeepSpeed natively supports **Adam**, **AdamW**, **OneBitAdam**, **Lamb**, and **OneBitLamb** optimizers (See [here](https://deepspeed.readthedocs.io/en/latest/optimizers.html) for details) and will import other optimizers from [torch](https://pytorch.org/docs/stable/optim.html). | `"Adam"`                     |
| params | Dictionary of parameters to instantiate optimizer. The parameter names must match the optimizer constructor signature (e.g., for [Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam)).                                                                                                       | `{"lr": 0.001, "eps": 1e-8}` |

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

A variant ***optimizer*** for 1-bit Adam is 0/1 Adam, which further optimizes 1-bit Adam via adaptive variance freezing and 1-bit synchronization over optimizer states.
```json
"optimizer": {
    "type": "ZeroOneAdam",
    "params": {
      "lr": 1e-3,
      "weight_decay": 0.01,
      "bias_correction": false,
      "var_freeze_step": 1000,
      "var_update_scaler": 16,
      "local_step_scaler": 1000,
      "local_step_clipper": 16,
      "cuda_aware": false,
      "comm_backend_name": "nccl"
    }
  }
```
0/1 Adam supports  the following params key/values in addition to standard Adam (learn more in our [tutorial](/tutorial/zero-one-adam/).)
| "params" key        | Description                                                                        | Default |
| ------------------- | ---------------------------------------------------------------------------------- | ------- |
| var\_freeze\_step   | The latest step to update the variance                                             | 100000  |
| var\_update\_scaler | The interval to update the variance                                                | 16  |
| local\_step\_scaler | The interval to scale the local steps interval according to the learning rate policy   | 32678  |
| local\_step\_clipper | The largest interval for local steps with learning rate policy                     | 16  |
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

| "params" key        | Description                                                                               | Default |
| ------------------- | ----------------------------------------------------------------------------------------- | ------- |
| max\_coeff          | Scaling coefficient upper bound for original LAMB algorithm and 1-bit LAMB's warmup stage | 10.0    |
| min\_coeff          | Scaling coefficient lower bound for original LAMB algorithm and 1-bit LAMB's warmup stage | 0.01    |
| freeze\_step        | Number of warm up steps before 1-bit compression gets applied to the communication        | 100000  |
| cuda\_aware         | To indicate that the underlying MPI library supports CUDA-Aware communication             | false   |
| comm\_backend\_name | To indicate which backend implementation to use                                           | "nccl"  |
| coeff\_beta         | Coefficient used for computing running averages of lamb coefficient                       | 0.9     |
| factor\_max         | Maximum value of scaling factor to the frozen lamb coefficient during compression stage   | 4.0     |
| factor\_min         | Minimum value of scaling factor to the frozen lamb coefficient during compression stage   | 0.5     |
| factor\_threshold   | Threshold of how much the scaling factor can fluctuate between steps                      | 0.1     |

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

<i>**communication_data_type**</i>: [boolean]

| Description                                                                                                                   | Default |
| ----------------------------------------------------------------------------------------------------------------------------- | ------- |
| During gradient averaging perform communication with selected data type. Buy default it will be determined by selected regime |  None   |

<i>**prescale_gradients**</i>: [boolean]

| Description                            | Default |
| -------------------------------------- | ------- |
| Scale gradients before doing allreduce | `false` |

<i>**gradient_predivide_factor**</i>: [float]

| Description                                                                                                                                       | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Before gradient averaging predivide gradients by a specified factor, can sometimes help with fp16 stability when scaling to large numbers of GPUs | `1.0`   |

<i>**sparse_gradients**</i>: [boolean]

| Description                                                                                                                                                                                                                                                                                                                                                 | Default |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Enable sparse compression of [torch.nn.Embedding](https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding) gradients. This feature is essentially deprecated as we don't see use cases for it as much anymore. It should be noted that this feature is not compatible with [torch.sparse](https://pytorch.org/docs/stable/sparse.html) related features. | `false` |

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

| Description                                                                                 | Default |
| ------------------------------------------------------------------------------------------- | ------- |
| <i>**enabled**</i> is a **fp16** parameter indicating whether or not FP16 training enabled. | `false` |

<i>**fp16:loss_scale**</i>: [float]

| Description                                                                                                                                                                                                                           | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| <i>**loss_scale**</i> is a <i>**fp16**</i> parameter representing the loss scaling value for FP16 training. The default value of 0.0 results in dynamic loss scaling, otherwise the value will be used for static fixed loss scaling. | `0.0`   |

<i>**fp16:initial_scale_power**</i>: [integer]

| Description                                                                                                                                                                                             | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| <i>**initial_scale_power**</i> is a **fp16** parameter representing the power of the initial dynamic loss scale value. The actual loss scale is computed as 2<sup><i>**initial_scale_power**</i></sup>. | `32`    |

<i>**fp16:loss_scale_window**</i>: [integer]

| Description                                                                                                                          | Default |
| ------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| <i>**loss_scale_window**</i> is a **fp16** parameter representing the window over which to raise/lower the dynamic loss scale value. | `1000`  |

<i>**fp16:hysteresis**</i>: [integer]

| Description                                                                                         | Default |
| --------------------------------------------------------------------------------------------------- | ------- |
| <i>**hysteresis**</i> is a **fp16** parameter representing the delay shift in dynamic loss scaling. | `2`     |

<i>**fp16:min_loss_scale**</i>: [integer]

| Description                                                                                           | Default |
| ----------------------------------------------------------------------------------------------------- | ------- |
| <i>**min_loss_scale**</i> is  a **fp16** parameter representing the minimum dynamic loss scale value. | `1000`  |

### BFLOAT16 training options

**Note:** this mode cannot be combined with the `amp` mode described below.
{: .notice--warning}

**Note:** this mode cannot be combined with the `fp16` mode described above.
{: .notice--warning}

<i>**bf16**</i>: [dictionary]

| Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Default |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Configuration for using [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) floating-point format as an alternative to FP16. BFLOAT16 requires hardware support (e.g., NVIDIA A100). An example, including the available dictionary keys is illustrated below. Training with bfloat16 does not require loss scaling. | None    |

```json
"bf16": {
   "enabled": true
 }
```

<i>**bf16:enabled**</i>: [boolean]

| Description                                                        | Default |
|--------------------------------------------------------------------| ------- |
| <i>**enabled**</i> indicates whether BFLOAT16 training is enabled. | `false` |


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

| Description                                                                                   | Default |
| --------------------------------------------------------------------------------------------- | ------- |
| <i>**enabled**</i> is an **amp** parameter indicating whether or not AMP training is enabled. | `false` |

***amp params***: [various]

| Description                                                                                                                                                                                                            | Default |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Any parameters outside of "enabled" will be passed to AMP's initialize call, see the API and descriptions here at the [apex.amp.initialize documentation](https://nvidia.github.io/apex/amp.html#apex.amp.initialize). | None    |

### Gradient Clipping

<i>**gradient_clipping**</i>: [float]

| Description                         | Default |
| ----------------------------------- | ------- |
| Enable gradient clipping with value | `1.0`   |



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
    "stage3_gather_16bit_weights_on_model_save": [true|false],
    "ignore_unused_parameters": [true|false]
    "round_robin_gradients": [true|false]
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

| Description                                                                                                         | Default |
| ------------------------------------------------------------------------------------------------------------------- | ------- |
| Copies the gradients to a contiguous buffer as they are produced. Avoids memory fragmentation during backward pass. | `True`  |

<i>**grad_hooks**</i>: [boolean]

| Description                                                                                                                               | Default |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| For use with ZeRO stage 1, enable backward hooks to reduce gradients during the backward pass or wait until the end of the backward pass. | `True`  |

***round_robin_gradients***: [boolean]

| Description                                                                                                                                                                                                                                                                         | Default |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Stage 2 optimization for CPU offloading that parallelizes gradient copying to CPU memory among ranks by fine-grained gradient partitioning. Performance benefit grows with gradient accumulation steps (more copying between optimizer steps) or GPU count (increased parallelism). | `False` |

***offload_param***: [dictionary]

| Description                                                                                                                                                                                   | Default |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Enable offloading of model parameters to CPU or NVMe. This frees up GPU memory for larger models or batch sizes. Valid only with stage 3. See [here](#parameter-offloading) for more details. | `False` |

***offload_optimizer***: [dictionary]

| Description                                                                                                                                                                                                                          | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| Enable offloading of optimizer state to CPU or NVMe, and optimizer computation to CPU. This frees up GPU memory for larger models or batch sizes. Valid only with stage 2 and 3. See [here](#optimizer-offloading) for more details. | `False` |

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


***stage3_gather_16bit_weights_on_model_save***: [boolean]

| Description                                                                                                                                                                                                                                                                    | Default |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ------- |
| Consolidate the weights before saving the model by `save_16bit_model()`. Since the weights are partitioned across GPUs, they aren't part of `state_dict`, so this function automatically gathers the weights when this option is enabled and then saves the fp16 model weights. | `False` |


***cpu_offload***: [boolean]

**Deprecated:** **cpu_offload** is deprecated and will be removed in future, please use `offload_optimizer` instead.
{: .notice--warning}

| Description                                                                                                                                       | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Enable offloading of optimizer memory and computation to CPU. This frees up GPU memory for larger models or batch sizes. Valid only with stage 2. | `False` |


### Parameter offloading
Enabling and configuring ZeRO optimization of parameter offloading to CPU/NVMe. Available only with ZeRO stage 3.
Note that if the value of "device" is not specified or not supported, an assertion will be triggered.

```json
  "offload_param": {
    "device": "[cpu|nvme]",
    "nvme_path": "/local_nvme",
    "pin_memory": [true|false],
    "buffer_count": 5,
    "buffer_size": 1e8,
    "max_in_cpu": 1e9
  }
```
***device***: [string]

| Description                                                                        | Default |
| ---------------------------------------------------------------------------------- | ------- |
| Device memory to offload model parameters. Supported options are `cpu` and `nvme`. | `cpu`   |

***nvme_path***: [string]

| Description                                               | Default       |
| --------------------------------------------------------- | ------------- |
| Filesystem path for NVMe device for parameter offloading. | `/local_nvme` |

***pin_memory***: [boolean]

| Description                                                                                          | Default |
| ---------------------------------------------------------------------------------------------------- | ------- |
| Offload to page-locked CPU memory. This could boost throughput at the cost of extra memory overhead. | `false` |

***buffer_count***: [integer]

| Description                                                        | Default |
| ------------------------------------------------------------------ | ------- |
| Number of buffers in buffer pool for parameter offloading to NVMe. | 5       |


***buffer_size***: [integer]

| Description                                                      | Default |
| ---------------------------------------------------------------- | ------- |
| Size of buffers in buffer pool for parameter offloading to NVMe. | 1e8     |

***max_in_cpu***: [integer]

| Description                                                                                | Default |
| ------------------------------------------------------------------------------------------ | ------- |
| Number of parameter elements to maintain in CPU memory when offloading to NVMe is enabled. | 1e9     |

### Optimizer offloading
Enabling and configuring ZeRO optimization of offloading optimizer computation to CPU and state to CPU/NVMe. CPU offloading is available with ZeRO stage 2 or 3. NVMe offloading is available only with ZeRO stage 3.
Note that if the value of "device" is not specified or not supported, an assertion will be triggered.
```json
  "offload_optimizer": {
    "device": "[cpu|nvme]",
    "nvme_path": "/local_nvme",
    "pin_memory": [true|false],
    "buffer_count": 4,
    "fast_init": false
  }
```
***device***: [string]

| Description                                                                                                                                            | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| Device memory to offload optimizer state. Supported options are `cpu` and `nvme`. Optimizer computation is offload to CPU regardless of device option. | `cpu`   |

***nvme_path***: [string]

| Description                                                     | Default       |
| --------------------------------------------------------------- | ------------- |
| Filesystem path for NVMe device for optimizer state offloading. | `/local_nvme` |

***pin_memory***: [boolean]

| Description                                                                                          | Default |
| ---------------------------------------------------------------------------------------------------- | ------- |
| Offload to page-locked CPU memory. This could boost throughput at the cost of extra memory overhead. | `false` |

***buffer_count***: [integer]

| Description                                                                                                                                                                                                                                              | Default |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Number of buffers in buffer pool for optimizer state offloading to NVMe. This should be at least the number of states maintained per parameter by the optimizer. For example, Adam optimizer has 4 states (parameter, gradient, momentum, and variance). | 4       |

***fast_init***: [boolean]

| Description                                                   | Default |
| ------------------------------------------------------------- | ------- |
| Enable fast optimizer initialization when offloading to NVMe. | `false` |


### Asynchronous I/O
Configuring the asynchronous I/O module for offloading parameter and optimizer states to persistent (NVMe) storage. This module uses Linux native asynchronous I/O (libaio).
```json
  "aio": {
    "block_size": 1048576,
    "queue_depth": 8,
    "thread_count": 1,
    "single_submit": false,
    "overlap_events": true
  }
```
***block_size***: [integer]

| Description              | Default |
| ------------------------ | ------- |
| I/O block size in bytes. | 1048576 |

***queue_depth***: [integer]

| Description      | Default |
| ---------------- | ------- |
| I/O queue depth. | 8       |

***thread_count***: [integer]

| Description                                                               | Default |
| ------------------------------------------------------------------------- | ------- |
| Intra-request parallelism for each read/write submitted by a user thread. | 1       |

***single_submit***: [boolean]

| Description                                                                                            | Default |
| ------------------------------------------------------------------------------------------------------ | ------- |
| Submit requests to storage device as multiple individual requests as opposed to one block of requests. | `false` |

***overlap_events***: [boolean]

| Description                                                                                                    | Default |
| -------------------------------------------------------------------------------------------------------------- | ------- |
| Submit requests to storage device in an overlapped fashion without waiting for completion of earlier requests. | `true`  |

***ignore_unused_parameters***: [boolean]

| Description                                                                                                                                                                                                                                                                                                                                                     | Default |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Unused parameters in modules may be unexpected in static networks, but could be normal in dynamic networks. This controls whether or not training should terminate with an error message when unused parameters are detected. This is set to `False` by default, which means unused parameters are ignored and training continues. Now is just used in stage 2. | `True`  |

### Logging

<i>**steps_per_print**</i>: [integer]

| Description                                                                                                                                                                                                                             | Default |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Print progress report every N training steps. The report includes the number of training steps, number of skipped optimizer updates (likely due to overflows in mixed-precision training), current learning rate, and current momentum. | `10`    |

<i>**wall_clock_breakdown**</i>: [boolean]

| Description                                                             | Default |
| ----------------------------------------------------------------------- | ------- |
| Enable timing of the latency of forward/backward/update training phases | `false` |

<i>**dump_state**</i>: [boolean]

| Description                                                          | Default |
| -------------------------------------------------------------------- | ------- |
| Print out state information of DeepSpeed object after initialization | `false` |


### Autotuning

```json
{
  "autotuning": {
    "enabled": false,
    "results_dir": null,
    "exps_dir": null,
    "overwrite": false,
    "metric": "throughput",
    "start_profile_step": 3,
    "end_profile_step": 5,
    "fast": true,
    "max_train_batch_size": null,
    "mp_size": 1,
    "num_tuning_micro_batch_sizes": 3,
    "tuner_type": "model_based",
    "tuner_early_stopping": 5,
    "tuner_num_trials": 50,
    "arg_mappings": null
  }
}
```
<i>**enabled**</i>: [boolean]

| Description            | Default |
| ---------------------- | ------- |
| Enables the autotuner. | `false` |


<i>**results_dir**</i>: [string]

| Description                                                                                                                      | Default |
| -------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Path to the autotuning experiment results directory. If None, "autotuning_results" under the training script launching path is used. | `null`  |

<i>**exps_dir**</i>: [string]

| Description                                                                                                                        | Default |
| ---------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Path to the auotuning experiment descriptions directory. If None, "autotuning_exps" under the train script launching path is used. | `null`  |

<i>**overwrite**</i>: [boolean]

| Description                                                                                                               | Default |
|---------------------------------------------------------------------------------------------------------------------------| ------- |
| Whether to run autotuing experiments whose results already exist. Setting it to true would overwrite the existing result. | `false` |


<i>**metric**</i>: [string]

| Description                                                                                                                                                                                                                                                            | Default      |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| The performance metric to use for ranking autotuning experiments. `latency`, `throughput`, and `FLOPS` are currently supported, referring to training step latency, training samples per second, and floating-point operations per second achieved per GPU respectively. | `throughput` |

<i>**start_profile_step**</i>: [integer]

| Description                                                                                                                                         | Default |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| The global training step at which to start profiling in an autotuning experiment. Note that warm-up is needed for accurate performance measurement. | `3`     |

<i>**end_profile_step**</i>: [integer]

| Description                                                                                                               | Default |
| ------------------------------------------------------------------------------------------------------------------------- | ------- |
| The global training step at which to end profiling in an autotuning experiment. Must not be less than start_profile_step. | `5`     |


<i>**fast**</i>: [boolean]

| Description                                                                                  | Default |
| -------------------------------------------------------------------------------------------- | ------- |
| Enables fast-model autotuning where only Zero stages and micro-batch sizes per GPU are tuned. | `true` |

<i>**max_train_batch_size**</i>: [int]

| Description                                                                       | Default |
| --------------------------------------------------------------------------------- | ------- |
| The maximum train batch size (global effective batch size) for the model training. | `null`  |

<i>**mp_size**</i>: [int]

| Description              | Default |
| ------------------------ | ------- |
| Model parallelism degree. | `1`     |


<i>**num_tuning_micro_batch_sizes**</i>: [integer]

| Description                                     | Default |
| ----------------------------------------------- | ------- |
| The number of micro-batch sizes to explore. | `3`     |

<i>**tuner_type**</i>: [string]

| Description                                                                              | Default       |
| ---------------------------------------------------------------------------------------- | ------------- |
| The algorithm defines the order of autotuning space exploration within a ZeRO stage. | `model_based` |


<i>**tuner_early_stopping**</i>: [integer]

| Description                                                                                                                                                | Default |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| The number of experiments to run beyond the current best experiment. If no better experiment is found within that number, the Autotuner stops the exploration. | `5`     |

<i>**tuner_num_trials**</i>: [integer]

| Description                                                                           | Default |
| ------------------------------------------------------------------------------------- | ------- |
| The maximum number of experiments to explore in the tuning space within a ZeRO stage. | `50`    |


### Flops Profiler
```json
{
  "flops_profiler": {
    "enabled": false,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null,
    }
}
```
<i>**enabled**</i>: [boolean]

| Description                                                              | Default |
| ------------------------------------------------------------------------ | ------- |
| Enables the flops profiler. This would also enables wall_clock_breakdown | `false` |

<i>**profile_step**</i>: [integer]

| Description                                                                                                     | Default |
| --------------------------------------------------------------------------------------------------------------- | ------- |
| The global training step at which to profile. Note that warm up steps are needed for accurate time measurement. | `1`     |

<i>**module_depth**</i>: [integer]

| Description                                                                                                                                                                           | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| The depth of the model at which to print the aggregated module information. When set to `-1`, it prints information from the top module to the innermost modules (the maximum depth). | `-1`    |

<i>**top_modules**</i>: [integer]

| Description                                                                  | Default |
| ---------------------------------------------------------------------------- | ------- |
| Limits the aggregated profile output to the number of top modules specified. | `1`     |

<i>**detailed**</i>: [boolean]

| Description                                  | Default |
| -------------------------------------------- | ------- |
| Whether to print the detailed model profile. | `true`  |

<i>**output_file**</i>: [string]

| Description                                                       | Default |
| ----------------------------------------------------------------- | ------- |
| Path to the output file. If None, the profiler prints to stdout.. | `null`  |


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
| Total number of activation checkpoints used to allocate memory buffer for contiguous_memory_optimization | `None`  |

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

### Curriculum Learning
```json
  "curriculum_learning": {
    "enabled": true,
    "curriculum_type": "seqlen",
    "min_difficulty": 8,
    "max_difficulty": 1024,
    "schedule_type": "fixed_linear",
    "schedule_config": {
      "total_curriculum_step": 40000,
      "difficulty_step": 8
    }
  }
```
<i>**enabled**</i>: [boolean]

| Description                               | Default |
| ----------------------------------------- | ------- |
| Set to true to enable curriculum learning | `false` |

<i>**curriculum_type**</i>: [string]

| Description                                                       | Default |
| ----------------------------------------------------------------- | ------- |
| Type of curriculum difficulty metric. Currently support `seqlen`. | N/A     |


<i>**min_difficulty**</i>: [integer]

| Description                   | Default |
| ----------------------------- | ------- |
| The starting difficulty level | N/A     |

<i>**max_difficulty**</i>: [integer]

| Description                 | Default |
| --------------------------- | ------- |
| The ending difficulty level | N/A     |

<i>**schedule_type**</i>: [string]

| Description                                                                                        | Default |
| -------------------------------------------------------------------------------------------------- | ------- |
| Type of curriculum schedule. Currently support `fixed_linear`, `fixed_root`, and `fixed_discrete`. | N/A     |


<i>**total_curriculum_step**</i>: [integer]

| Description                                                                                                                                      | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| Total number of steps for the curriculum learning. One of the `schedule_config` when the `fixed_linear` and `fixed_root` schedule_type are used. | N/A     |

<i>**difficulty_step**</i>: [integer]

| Description                                                                                                                                                                                                                                                                                          | Default |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| At any time, the curriculum learning difficulty must be multiple of this `difficulty_step`. Set this to multiple of 8 (for FP16 data) or 16 (for INT8 data) to enable NVIDIA Tensor Core acceleration. One of the `schedule_config` when the `fixed_linear` and `fixed_root` schedule_type are used. | N/A     |

<i>**root_degree**</i>: [integer]

| Description                                                                                                                | Default |
| -------------------------------------------------------------------------------------------------------------------------- | ------- |
| Root degree of the curriculum schedule function. One of the `schedule_config` when the `fixed_root` schedule_type is used. | N/A     |

<i>**difficulty**</i>: [list of integer]

| Description                                                                                                                         | Default |
| ----------------------------------------------------------------------------------------------------------------------------------- | ------- |
| List of difficulty levels to be used during schedule. One of the `schedule_config` when the `fixed_discrete` schedule_type is used. | N/A     |

<i>**max_step**</i>: [list of integer]

| Description                                                                                                                  | Default |
| ---------------------------------------------------------------------------------------------------------------------------- | ------- |
| List of which step to change difficulty level. One of the `schedule_config` when the `fixed_discrete` schedule_type is used. | N/A     |

### Logging to Tensorboard

**Note:** Deepspeed logs to TensorBoard through PyTorch. Logging to TensorBoard requires that the `tensorboard` package is installed (read more in the [PyTorch documentation](https://pytorch.org/docs/1.8.0/tensorboard.html)).
{: .notice--warning}


Deepspeed can log training details into a [Tensorboard](https://www.tensorflow.org/tensorboard)-compatible file. Below is an overview of what deepspeed will log.

| Field | Description                                                                                                                                                                                                                                                                                               |Conditions |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| `Train/Samples/train_loss`   | The training loss. | None |
| `Train/Samples/lr`           | The learning rate during training. | None |
| `Train/Samples/loss_scale`   | The loss scale when training using `fp16`. | `fp16` must be enabled. |
| `Train/Eigenvalues/ModelBlockParam_{i}`   | Eigen values per param block. | `eigenvalue` must be enabled. |
| `Train/Samples/elapsed_time_ms_forward`   | The global duration of the forward pass. | `flops_profiler.enabled` or `wall_clock_breakdown`. |
| `Train/Samples/elapsed_time_ms_backward`   | The global duration of the forward pass. | `flops_profiler.enabled` or `wall_clock_breakdown`.  |
| `Train/Samples/elapsed_time_ms_backward_inner`   | The backward time that does not include the the gradient reduction time. Only in cases where the gradient reduction is not overlapped, if it is overlapped then the inner time should be about the same as the entire backward time. | `flops_profiler.enabled` or `wall_clock_breakdown`.  |
| `Train/Samples/elapsed_time_ms_backward_allreduce`   | The global duration of the allreduce operation. | `flops_profiler.enabled` or `wall_clock_breakdown`.  |
| `Train/Samples/elapsed_time_ms_step`   | The optimizer step time | `flops_profiler.enabled` or `wall_clock_breakdown`.  |

<i>**tensorboard**</i>: [dictionary]

| Fields | Value                                                                                                                                                                                                                                                                                                        |Default |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| enabled   | Whether logging to [Tensorboard](https://www.tensorflow.org/tensorboard) is enabled. | `false` |
| job_name  | Name for the current job. This will become a new directory inside `output_path` | `"DeepSpeedJobName"` |
| output_path | Path to where the Tensorboard logs will be written.                           | `~/tensorboard/` |


Example of <i>** tensorboard**</i> configuration:

```json
"tensorboard": {
    "enabled": true,
    "output_path": "output/ds_logs/",
    "job_name": "train_bert"
}
```
