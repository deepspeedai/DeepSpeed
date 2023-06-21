---
title: "DeepSpeed Configuration JSON"
toc: true
toc_label: "Contents"
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

<i>**communication_data_type**</i>: [string]

| Description                                                                                                                   | Default |
| ----------------------------------------------------------------------------------------------------------------------------- | ------- |
| During gradient averaging perform communication with selected data type. By default it will be determined by selected regime  |  None   |

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
    "auto_cast": false,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "consecutive_hysteresis": false,
    "min_loss_scale": 1
}
```

<i>**fp16:enabled**</i>: [boolean]

| Description                                                                                 | Default |
| ------------------------------------------------------------------------------------------- | ------- |
| <i>**enabled**</i> is a **fp16** parameter indicating whether or not FP16 training enabled. | `false` |

<i>**fp16:auto_cast**</i>: [boolean]

| Description                                                  | Default |
| -------------------------------------------------------------| ------- |
| <i>**auto_cast**</i> automatically casts inputs to **fp16**  | `false` |

<i>**fp16:loss_scale**</i>: [float]

| Description                                                                                                                                                                                                                           | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| <i>**loss_scale**</i> is a <i>**fp16**</i> parameter representing the loss scaling value for FP16 training. The default value of 0.0 results in dynamic loss scaling, otherwise the value will be used for static fixed loss scaling. | `0.0`   |

<i>**fp16:initial_scale_power**</i>: [integer]

| Description                                                                                                                                                                                             | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| <i>**initial_scale_power**</i> is a **fp16** parameter representing the power of the initial dynamic loss scale value. The actual loss scale is computed as 2<sup><i>**initial_scale_power**</i></sup>. | `16`    |

<i>**fp16:loss_scale_window**</i>: [integer]

| Description                                                                                                                          | Default |
| ------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| <i>**loss_scale_window**</i> is a **fp16** parameter representing the window over which to raise/lower the dynamic loss scale value. | `1000`  |

<i>**fp16:hysteresis**</i>: [integer]

| Description                                                                                         | Default |
| --------------------------------------------------------------------------------------------------- | ------- |
| <i>**hysteresis**</i> is a **fp16** parameter representing the delay shift in dynamic loss scaling. | `2`     |

<i>**fp16:consecutive_hysteresis**</i>: [boolean]

| Description                                                                                         | Default |
| --------------------------------------------------------------------------------------------------- | ------- |
| <i>**consecutive_hysteresis**</i> is a **fp16** parameter representing whether to refill the hysteresis if we reach an iteration that doesn't overflow | `false`     |

<i>**fp16:min_loss_scale**</i>: [integer]

| Description                                                                                           | Default |
| ----------------------------------------------------------------------------------------------------- | ------- |
| <i>**min_loss_scale**</i> is  a **fp16** parameter representing the minimum dynamic loss scale value. | `1`     |

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
    "zero_hpz_partition_size": 1
    "zero_quantized_weights": [true|false]
    "zero_quantized_gradients": [true|false]
    }
```

<i>**zero_optimization**</i>: [dictionary]

| Description                                                                                               | Default |
| --------------------------------------------------------------------------------------------------------- | ------- |
| Enable ZeRO memory optimizations, compatible with FP16/BF16/FP32 and the Adam optimizer. | `false` |

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
| Stage 1 and 2 optimization for CPU offloading that parallelizes gradient copying to CPU memory among ranks by fine-grained gradient partitioning. Performance benefit grows with gradient accumulation steps (more copying between optimizer steps) or GPU count (increased parallelism). | `False` |

***offload_param***: [dictionary]

| Description                                                                                                                                                                                   | Default |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Enable offloading of model parameters to CPU or NVMe. This frees up GPU memory for larger models or batch sizes. Valid only with stage 3. See [here](#parameter-offloading) for more details. | `False` |

***offload_optimizer***: [dictionary]

| Description                                                                                                                                                                                                                          | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| Enable offloading of optimizer state to CPU or NVMe, and optimizer computation to CPU. This frees up GPU memory for larger models or batch sizes. Valid for ZeRO stage 1, 2, 3. See [here](#optimizer-offloading) for more details. | `False` |

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

***zero_hpz_partition_size***: [integer]

| Description                                                                                                                         | Default |
| ----------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Number of ranks in hiearchical partitioning ZeRO (hpZ) secondary tensor group of ZeRO++, default is 1 meaning no hpZ, ideal is number of ranks (gpus) per node. | `1`   |

***zero_quantized_weights***: [boolean]

| Description                                                                                                                         | Default |
| ----------------------------------------------------------------------------------------------------------------------------------- | ------- |
|Boolean indicating whether to enable communication efficient quantized weights of ZeRO++. | `False`   |

***zero_quantized_gradients***: [boolean]

| Description                                                                                                                         | Default |
| ----------------------------------------------------------------------------------------------------------------------------------- | ------- |
|Boolean indicating whether to enable communication efficient quantized gradients of ZeRO++. | `False`   |

***cpu_offload***: [boolean]

**Deprecated:** **cpu_offload** is deprecated and will be removed in future, please use `offload_optimizer` instead.
{: .notice--warning}

| Description                                                                                                                                       | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Enable offloading of optimizer memory and computation to CPU. This frees up GPU memory for larger models or batch sizes. Valid with stage 1 and 2. | `False` |


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
Enabling and configuring ZeRO optimization of offloading optimizer computation to CPU and state to CPU/NVMe. CPU offloading is available with ZeRO stage 1, 2, 3. NVMe offloading is available only with ZeRO stage 3.
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
    "results_dir": "autotuning_results",
    "exps_dir": "autotuning_exps",
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

| Description                                                                                                                           | Default |
| ------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| Path to the autotuning experiment results directory.  The default appears in the working directory from which Deepspeed was launched. | "autotuning_results"  |

<i>**exps_dir**</i>: [string]

| Description                                                                                                                              | Default |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| Path to the auotuning experiment descriptions directory. The default appears in the working directory from which Deepspeed was launched. | "autotuning_exps"  |

<i>**overwrite**</i>: [boolean]

| Description                                                                                                               | Default |
|---------------------------------------------------------------------------------------------------------------------------| ------- |
| Whether to run autotuning experiments whose results already exist. Setting it to true would overwrite the existing result. | `false` |


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
| Inserts get_accelerator().synchronize() at each checkpoint boundary. | `false` |


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

### Data Efficiency
DeepSpeed Data Efficiency Library includes two techniques: curriculum learning and random layerwise token dropping (random-LTD). Read more about how to use the DeepSpeed Data Efficiency Library in our [tutorial](/tutorials/data-efficiency/).

```json
"data_efficiency": {
  "enabled": true,
  "seed": 1234,
  "data_routing": {
    "enabled": true,
    "random_ltd":{
      "enabled": true,
      "total_layer_num": 24,
      "random_ltd_layer_num": 22,
      "random_ltd_layer_id": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
      "model_mask_name": "attention_mask",
      "model_type": "decoder",
      "hidden_state_order": "seq_batch_dim",
      "random_ltd_schedule": {
        "min_value": 128,
        "max_value": 2048,
        "schedule_type":"fixed_linear",
        "schedule_config": {
          "require_steps": 200000,
          "seq_per_step": 16
        }
      }
    }
  },
  "data_sampling": {
    "enabled": true,
    "num_epochs": 1,
    "num_workers": 0,
    "curriculum_learning": {
      "enabled": true,
      "data_cluster_path": "/path/to/data_clusters",
      "curriculum_metrics": {
        "vocabularyrarity": {
          "index_to_sample_path": "/path/to/index_to_sample",
          "index_to_metric_path": "/path/to/index_to_metric",
          "difficulty_type": "percentile",
          "clustering_type": "schedule_based",
          "min_difficulty": 1,
          "max_difficulty": 100,
          "schedule_type": "fixed_root",
          "schedule_config": {
            "total_curriculum_step": 110000,
            "difficulty_step": 1,
            "root_degree": 2
          }
        }
      }
    }
  }
}
```

<i>**data_efficiency**</i>: [dictionary]

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**enabled**</i>: [boolean] | Enable data efficiency or not. | `false` |
| <i>**seed**</i>: [integer] | Random seed for data sampling. | 1234 |
| <i>**data_routing**</i>: [dictionary] | Configs for data routing techniques. | N/A |
| <i>**data_sampling**</i>: [dictionary] | Configs for data sampling techniques. | N/A |

<i>**data_routing**</i>: [dictionary]

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**enabled**</i>: [boolean] | Enable data routing techniques or not. | `false` |
| <i>**random_ltd**</i>: [dictionary] | Configs for random-LTD technique. | N/A |

<i>**data_sampling**</i>: [dictionary]

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**enabled**</i>: [boolean] | Enable data sampling techniques or not. | `false` |
| <i>**num_epochs**</i>: [integer] | At most how many epoches of the original dataset will be iterated. | 1000 |
| <i>**num_workers**</i>: [integer] | Data loader number of workers. | 0 |
| <i>**curriculum_learning**</i>: [dictionary] | Configs for curriculum learing technique. | N/A |

<i>**random_ltd**</i>: [dictionary]

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**enabled**</i>: [boolean] | Enable random-LTD technique or not. | `false` |
| <i>**total_layer_num**</i>: [integer] | The number of layer (or the depth) for the pretraining/fine-tuning model. | N/A |
| <i>**random_ltd_layer_num**</i>: [integer] | The number of layers that will be applied with random-LTD. | N/A |
| <i>**random_ltd_layer_id**</i>: [list] | The exact layer_id that will be applied with random-LTD. The length of this list must be the same as `random_ltd_layer_num`. | N/A |
| <i>**model_mask_name**</i>: [str] | The variable name of the attention_mask. Different libraries have different names, such as att_mask. For huggingface model, it’s named “attention_mask”. Users need to check the forward function in the original model files. If the attention mask input in the original model's forward function is not a keyword/named argument (e.g., attention_mask=None), user would need to change it to a keyword/named argument and provide that keyword as `model_mask_name`. | N/A |
| <i>**model_type**</i>: [str] | Users need to identify whether the model is `decoder` or `encoder`. Currently we only support these two. | N/A |
| <i>**hidden_state_order**</i>: [str] | Users need to know the input order of the hidden state tensor. Normally, it’s batch, sequence and then the hidden dimension, which is `batch_seq_dim`. Somethings, the order between batch and sequence will be switch like `seq_batch_dim`. Currently, we support these two.  | N/A |
| <i>**random_ltd_schedule**</i>: [dictionary] | The schedule of the effective sequence length after token dropping. It's a linear function where random-LTD gradually drops less tokens and increases effective sequence length. | N/A |
| <i>&emsp;&emsp;**min_value**</i>: [integer] | The initial effective sequence length (after token dropping) at step/iteration 0. | N/A |
| <i>&emsp;&emsp;**max_value**</i>: [integer] | The max effective sequence length (usually the case without any token dropping). Usually this is set as baseline's seqlen. | N/A |
| <i>&emsp;&emsp;**schedule_type**</i>: [str] | The sequence length follows a linear increasing function starting from `min_value` and reaching `max_value`. We currently only support this type. | N/A |
| <i>&emsp;&emsp;**schedule_config**</i>: [dictionary] | Configs for the linear increasing function. | N/A |
| <i>&emsp;&emsp;&emsp;&emsp;**require_steps**</i>: [integer] | How many iterations will be needed to reach max_value from min_value. | N/A |
| <i>&emsp;&emsp;&emsp;&emsp;**seq_per_step**</i>: [integer] | At any time, the effective sequence length be multiple of this `seq_per_step`. Set this to multiple of 8 (for FP16 data) or 16 (for INT8 data) to enable NVIDIA Tensor Core acceleration. | N/A |

<i>**curriculum_learning**</i>: [dictionary]

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**enabled**</i>: [boolean] | Enable curriculum learing technique or not. | `false` |
| <i>**data_cluster_path**</i>: [str] | Path to directory where curriculum learning will store the indexes of data samples within the same difficulty ranges. | N/A |
| <i>**curriculum_metrics**</i>: [dictionary] | This dictionary includes all desired curriculum metrics and their configs. Each metric will be a separate sub-dictionary, where the key is the metric name and the values are configs below. | N/A |
| <i>&emsp;&emsp;**index_to_sample_path**</i>: [str] | Path to the index_to_sample file generated during offline data analysis. Note that data analysis will generate two kinds of index_to_sample files: The metric_name_index_to_sample_percentile_merged file is a concatenated index for perf improvement, but it only works when you set difficulty_type=`percentile`. If you use difficulty_type=`value`, you need to change this to use the metric_name_index_to_sample file. | N/A |
| <i>&emsp;&emsp;**index_to_metric_path**</i>: [str] | Path to the index_to_metric_path file generated during offline data analysis. | N/A |
| <i>&emsp;&emsp;**difficulty_type**</i>: [str] | During training, how to increase the max accepted difficulty. Currently support `value` (increase by absolute value) and `percentile` (increase by difficulty percentile). | N/A |
| <i>&emsp;&emsp;**clustering_type**</i>: [str] | Currently support `schedule_based` (cluster data based on the difficulty schedule (pacing function) below) and `single_cluster` (no clustering required and probably CL is achieved by data postprocessing, such as sequence length truncation). | N/A |
| <i>&emsp;&emsp;**min_difficulty**</i>: [integer] | Starting difficulty at first step. When difficulty_type=`value` the `min_difficulty` is an absolute difficulty value. When difficulty_type=`percentile` the `min_difficulty` is a difficulty percentile value. | N/A |
| <i>&emsp;&emsp;**max_difficulty**</i>: [integer] | Final max difficulty. When difficulty_type=`value` the `max_difficulty` is an absolute difficulty value. When difficulty_type=`percentile` the `max_difficulty` is a difficulty percentile value. | N/A |
| <i>&emsp;&emsp;**schedule_type**</i>: [str] | The difficulty schedule (pacing function) that defines how the max accepted difficulty increases from `min_difficulty` to `max_difficulty` during training. Currently support `fixed_linear`, `fixed_root`, `fixed_discrete`, and `custom`. | N/A |
| <i>&emsp;&emsp;**schedule_config**</i>: [dictionary] | Configs for the pacing function. When schedule_type=`custom` this dictionary is not necessary. Instead user needs to provide a callback function (via the `set_custom_curriculum_learning_schedule` API in deepspeed/runtime/engine.py) which will update the max accepted difficulty during training. Configs below are all belongs to `schedule_config`. | N/A |
| <i>&emsp;&emsp;&emsp;&emsp;**total_curriculum_step**</i>: [integer] | How many steps the curriculum learning takes to go from min difficulty to max difficulty. Used by `fixed_linear` and `fixed_root` schedule. | N/A |
| <i>&emsp;&emsp;&emsp;&emsp;**difficulty_step**</i>: [integer] | The max accepted difficulty level determined every step must be a multiple of this `difficulty_step`. This is used to ensure the use of NVIDIA Tensor Core acceleration (requires multiple of 8 (FP16) or 16 (INT8)). Used by `fixed_linear` and `fixed_root` schedule. | N/A |
| <i>&emsp;&emsp;&emsp;&emsp;**root_degree**</i>: [integer] | The degree of the root function. Degree of 2 means square root and degree of 3 means cube root. Degree of 1 is equivalent to linear. Used by `fixed_root` schedule. | N/A |
| <i>&emsp;&emsp;&emsp;&emsp;**difficulty**</i>: [list] | List of max accepted difficulty levels to be used during schedule. Used by `fixed_discrete` schedule. | N/A |
| <i>&emsp;&emsp;&emsp;&emsp;**max_step**</i>: [list] | List of which step to change max accepted difficulty level. Used by `fixed_discrete` schedule. | N/A |


### Curriculum Learning

**Note:** On 12/12/2022, we released [DeepSpeed Data Efficiency Library](/tutorials/data-efficiency/) which provides a more general curriculum learning support. This legacy curriculum learning feature below is still supported but we recommend to use the Data Efficiency Library.

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

### Monitoring Module (TensorBoard, WandB, CSV)

**Note:** Deepspeed logs to TensorBoard through PyTorch. Logging to TensorBoard requires that the `tensorboard` package is installed (read more in the [PyTorch documentation](https://pytorch.org/docs/1.8.0/tensorboard.html)).
{: .notice--warning}
**Note:** Logging to WandB requires that the `wandb` package is installed (read more in the [WandB documentation](https://docs.wandb.ai/quickstart)).
{: .notice--warning}


Deepspeed's Monitor module can log training details into a [Tensorboard](https://www.tensorflow.org/tensorboard)-compatible file, to [WandB](https://wandb.ai/site), or to simple CSV files. Below is an overview of what DeepSpeed will log automatically.

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
| output_path | Path to where the Tensorboard logs will be written. If None, the output path is set under the training script's launching path.     | `null` |
| job_name  | Name for the current job. This will become a new directory inside `output_path`. | `"DeepSpeedJobName"` |


Example of <i>**tensorboard**</i> configuration:

```json
"tensorboard": {
    "enabled": true,
    "output_path": "output/ds_logs/",
    "job_name": "train_bert"
}
```

<i>**wandb**</i>: [dictionary]

| Fields | Value                                                                                                                                                                                                                                                                                                        |Default |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| enabled   | Whether logging to [WandB](https://wandb.ai/site) is enabled. | `false` |
| group  | Name for the WandB group. This can be used to group together runs. | `None` |
| team | Name for the WandB team.       | `None` |
| project | Name for the WandB project.       | `deepspeed` |


Example of <i>**wandb**</i> configuration:

```json
"wandb": {
    "enabled": true,
    "group": "my_group",
    "team": "my_team",
    "project": "my_project"
}
```

<i>**csv_monitor**</i>: [dictionary]

| Fields | Value                                                                                                                                                                                                                                                                                                        |Default |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| enabled   | Whether logging to local CSV files is enabled. | `false` |
| output_path | Path to where the csv files will be written. If None, the output path is set under the training script's launching path.      | `null` |
| job_name  | Name for the current job. This will become a new directory inside `output_path` | `"DeepSpeedJobName"` |


Example of <i>**csv_monitor**</i> configuration:

```json
"csv_monitor": {
    "enabled": true,
    "output_path": "output/ds_logs/",
    "job_name": "train_bert"
}
```

### Elastic Training Config (V0.1 and V0.2)

```json
  "elasticity": {
    "enabled": true,
    "max_train_batch_size": "seqlen",
    "micro_batch_sizes": 8,
    "min_gpus": 1024,
    "max_gpus": "fixed_linear",
    "min_time": "seqlen",
    "version": 8,
    "ignore_non_elastic_batch_info": 1024,
    "num_gpus_per_node": "fixed_linear",
    "model_parallel_size": MODEL_PARALLEL_SIZE
  }
```

| Field | Description                                                                                                                                                                                                                                                                                                   |Default|
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| `enabled`   | Enables computation of global batch size in elastic training. | false |
| `max_train_batch_size` | Max acceptable batch size can be used in training. | 2000 |
| `micro_batch_sizes` | Acceptable micro batch sizes, same as train_micro_batch_size_per_gpu | [2,4,6] |
| `min_gpus` | Min number of GPUs to search over when computing highly composite batch size in v0.1 and v0.2. | 1 |
| `max_gpus` | Max number of GPUs to search over when computing highly composite batch size in v0.1 and v0.2. | 10000 |
| `min_time` |Minimum running time (minutes) before the scheduler will scale again (only used in v0.1). 0 implies it's unknown | 0 |
| `prefer_large_batch` | When finding a suitable batch size, attempt to find one that is closest to the max train batch size given. | true |
| `version` | Version of elastic logic to use. | 0.2 |
| `ignore_non_elastic_batch_info` | Ignore all batch info provided outside the elastic config. To reduce confusion, we require all batch related info to be given in elastic config only. | false |
| `num_gpus_per_node` | Number of GPUs per node. This information is used by v0.2 to support model-parallel training (only used by v0.2) | 1 |
| `model_parallel_size` | Tensor or model parallel size (only used by v0.2) | 1 |


### Communication Logging


DeepSpeed provides a flexible communication logging tool which can automatically detect and record communication operations launched via `deepspeed.comm`. NOTE: All logging communication calls are synchronized in order to provide accurate timing information. This may hamper performance if your model heavily uses asynchronous communication operations.

Once the logs are populated, they can be summarized with `deepspeed.comm.log_summary()`. For more detail and example usage, see the [tutorial](/tutorials/comms-logging/)




<i>**comms_logger**</i>: [dictionary]

| Fields | Value                                                                                                                                                                                                                                                                                                        |Default |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| enabled   | Whether communication logging is enabled. | `false` |
| verbose | Whether to immediately print every communication operation  | `false` |
| prof_all  | Whether to profile all operations. | `true` |
| debug  | Appends the caller function to each communication operation's `log_name`. | `false` |
| prof_ops  | A list of communication operations to log (only the specified ops will be profiled). | `[]` |


Example of recommended <i>**comms_logger**</i> configuration:

```json
"comms_logger": {
  "enabled": true,
  "verbose": false,
  "prof_all": true,
  "debug": false
}
```

Example of <i>**comms_logger**</i> configuration for logging specific operations only:

```json
"comms_logger": {
  "enabled": true,
  "verbose": false,
  "prof_all": false,
  "debug": false,
  "prof_ops": ["all_reduce", "all_gather"]
}
```
### Compression
**Note:** <i>**Compression**</i> has seven different components, including layer reduction, weight quantization, activation quantization, sparse pruning, row pruning, head pruning, and channel pruning. We explain them one by one with simple json examples. Read more about how to use the DeepSpeed Compression library in our [tutorial](/tutorials/model-compression/).

#### Layer Reduction
**Note:** Layer reduction works much better when using knowledage distillation (learn more in our [tutorial](/tutorials/model-compression/)):

```json
"compression_training": {
    "layer_reduction": {
      "enabled": true,
      "keep_number_layer": 5,
      "module_name_prefix": "bert.encoder.layer",
      "teacher_layer": [
        2,
        4,
        6,
        8,
        10
      ],
      "other_module_name": [
        "bert.pooler",
        "bert.embeddings",
        "classifier"
      ]
    }
  }
```

<i>**layer_reduction**</i>: [dictionary]

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**enabled**</i>: [boolean] | Enable layer reduction or not. | `false` |
| <i>**keep_number_layer**</i>: [list] | The number of layer in the model to be kept. | N/A |
| <i>**module_name_prefix**</i>: [str] | The (uniform) name prefix of the model's modules of which the associated weight parameters are to be reinitialized. | N/A |
| <i>**teacher_layer**</i>: [list] | The layer of the weight parameters are to be reinitialized. The length of the list equals to 'keep_number_layer'. | N/A |
| <i>**other_module_name**</i>: [list] | The name of modules of which the associated weight parameters are to be reinitialized. It is an complemenatory or alternative of module_name_prefix. For instance,  "other_module_name": ["bert.encoder.layer.2","bert.encoder.layer.4"] equals to "module_name_prefix":"bert.encoder.layer" and  "teacher_layer": [2,4]. | N/A |

#### Weight Quantization
```json
  "compression_training": {
  "weight_quantization": {
    "shared_parameters":{
      "enabled": true,
      "quantizer_kernel": false,
      "schedule_offset": 0,
      "quantize_groups": 1,
      "quantize_verbose": false,
      "quantization_type": "symmetric",
      "rounding": "nearest",
      "quantize_weight_in_forward": false,
      "fp16_mixed_quantize":{
        "enabled": false,
        "quantize_change_ratio": 0.001
      }
    },
    "different_groups":{
      "wq1": {
        "params": {
            "start_bits": 8,
            "target_bits": 8,
            "quantization_period": 50
        },
        "modules": [
          "attention.self",
          "intermediate"
        ]
      },
      "wq2": {
        "params": {
            "start_bits": 4,
            "target_bits": 4,
            "quantization_period": 50
        },
        "modules": [
          "attention.output"
        ]
      }
    }
  }
  }
```

<i>**shared_parameters**</i>: [dictionary]

Shared parameters for all weight quantization groups.

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**enabled**</i>: [boolean] | Enable weight quantization or not. | `false` |
| <i>**quantizer_kernel**</i>: [boolean] | Use DeepSpeed quantization kernel for >=4 bit quantization. This can only be enabled when using DeepSpeed FP16 optimizer. | `false` |
| <i>**schedule_offset**</i>: [integer] | Enable weight quantization after scheduled steps (can be treated as warmup steps). | `0` |
| <i>**quantize_groups**</i>: [integer] | Split the weight matrix into different number of groups, and each of them has its own scaling factor. | `1` |
| <i>**quantize_verbose**</i>: [boolean] | Print the quantization related logs. | `false` |
| <i>**quantization_type**</i>: [string] | Choose the quantization algorithm, symmetric or asymmetric. | `"symmetric"` |
| <i>**rounding**</i>: [string] | Rounding algorithm associated with quantization, nearest or stochastic. | `"nearest"` |
| <i>**quantize_weight_in_forward**</i>: [boolean] | Quantize weight in optimizer or forward step, must set to be true for FP32 optimizer training. | `false` |
| <i>**fp16_mixed_quantize**</i>: [dictionary] | Using the value mixed by FP16 value and the quantized value. | N/A |
| <i>&emsp;&emsp;**enabled**</i>: [boolean] | Whether fp16 mixed quantization is enabled. | `false` |
| <i>&emsp;&emsp;**quantize_change_ratio**</i>: [float] | Initial quantize value ratio, will gradually increase to 1. | `0.001` |

<i>**different_groups**</i>: [dictionary]

Different quantization sets, this is used for different quantization parameters. In this example, we give two different sets. In practice, you can choose the number of sets based on your requirements.

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**params**</i>: [dictionary] | | |
| <i>&emsp;&emsp;**start_bits**</i>: [integer] | Quantization starting bits, will gradaully reduce to target bits. | `8` |
| <i>&emsp;&emsp;**target_bits**</i>: [integer] | Quantization target bits, need to be <= start_bits. | `8` |
| <i>&emsp;&emsp;**quantization_period**</i>: [integer] | For every n steps, the quantization bits will be reduce by 1. | `1` |
| <i>**modules**</i>: [list] | Scope of weight parameters associated to the params setting. | `"All Linear and CONV2D layers"` |

#### Activation Quantization
```json
"compression_training": {
  "activation_quantization": {
    "shared_parameters":{
      "enabled": true,
      "quantization_type": "asymmetric",
      "range_calibration": "dynamic",
      "schedule_offset": 50
    },
    "different_groups":{
      "aq1": {
        "params": {
            "bits": 8
        },
        "modules": [
          "attention.output"
        ]
      }
    }
  }
```

<i>**shared_parameters**</i>: [dictionary]

Shared parameters for all activation quantization groups.

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**enabled**</i>: [boolean] | Enable activation quantization or not. | `false` |
| <i>**quantization_type**</i>: [string] | Choose the quantization algorithm, symmetric or asymmetric. | `"symmetric"` |
| <i>**range_calibration**</i>: [string] | Using dynamic (per token or per image) or static (fixed min/max using momentum) for inference. | `"static"` |
| <i>**schedule_offset**</i>: [integer] | Enable activation quantization after scheduled steps (can be treated as warmup steps). | `0` |

<i>**different_groups**</i>: [dictionary]

Different quantization sets, this is used for different quantization parameters. In this example, we give one set. In practice, you can choose the number of sets based on your requirements.

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**params**</i>: [dictionary] | | |
| <i>&emsp;&emsp;**bits**</i>: [integer] | Number of bits used for activation target bits, need to be >= 4. | `8` |
| <i>**modules**</i>: [list] | Scope of weight parameters associated to the params setting. | `"All Linear and CONV2D layers"` |

#### Sparse Pruning
```json
"compression_training": {
  "sparse_pruning":{
    "shared_parameters":{
      "enabled": true,
      "schedule_offset": 30,
      "method": "l1"
    },
    "different_groups":{
      "sp1": {
        "params": {
            "dense_ratio": 0.5
        },
        "modules": [
          "attention.self"
        ]
      }
    }
  }
}
```

```json
"compression_training": {
  "sparse_pruning":{
    "shared_parameters":{
      "enabled": true,
      "schedule_offset": 30,
      "schedule_offset_end": 90,
      "schedule_offset_stride": 15,
      "method": "snip_momentum",
      "block_pattern": "4x1",
      "dense_ratio": 0.4,
      "excluded_modules": ['classifier', 'pooler']
    },
    "different_groups":{
    }
  }
}
```

<i>**shared_parameters**</i>: [dictionary]

Shared parameters for all sparse pruning groups.

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**enabled**</i>: [boolean] | Enable sparse pruning or not. | `false` |
| <i>**schedule_offset**</i>: [integer] | Enable sparse pruning after scheduled steps (can be treated as warmup steps). | `0` |
| <i>**schedule_offset_end**</i>: [integer] | Disable sparse pruning after scheduled steps, mandotory for `snip_momentum`. | `0` |
| <i>**schedule_offset_stride**</i>: [integer] | The stride of pruning on training steps, mandotory for `snip_momentum`. | `"1"` |
| <i>**method**</i>: [string] | Choose different pruning methods, l1 (static, magnitude based), topk (dynamic, learnable) or snip_momentum (structured pruning). | `"l1"` |
| <i>**block_pattern**</i>: [string] | Choose different structured pruning block patterns, NxM or N:M (N and M are integers). For instance, "4x1" or "2:4" are common block patterns, mandotory for `snip_momentum`. | `"4x1"` |
| <i>**dense_ratio**</i>: [float] | Used to get the targeted global sparsity ratio, mandotory for `snip_momentum`. | `"0.1"` |
| <i>**excluded_modules**</i>: [list] | Excluded pruning scope on some special modules like output layer. | `[]` |

<i>**different_groups**</i>: [dictionary]

Different pruning sets, this is used for different pruning parameters. In this example, we give one set. In practice, you can choose the number of sets based on your requirements.
Note for `snip_momentum` method, you can leave it as empty.

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**params**</i>: [dictionary] | | |
| <i>&emsp;&emsp;**dense_ratio**</i>: [float] | The percentage of weights to keep after pruning. | `0.5` |
| <i>**modules**</i>: [list] | Scope of weight parameters associated to the params setting. | `"All Linear and CONV2D layers"` |

#### Row Pruning
**Note:** <i>**Row Pruning**</i> is a feature designed for two back-to-back linear layers (e.g., Feed Forward Network in Transformers). As such, we suggested use row pruning for the first linear layer (i.e., the `intermediate.dense` layer for BERT). Reducing the row dimension of this matrix can help reducing the column of the follow-up matrix (i.e., `layer.\\w+.output.dense` layer for BERT). It should also work for other linear layers as well.
```json
"compression_training": {
  "row_pruning":{
    "shared_parameters":{
      "enabled": true,
      "schedule_offset": 20,
      "method": "topk"
    },
    "different_groups":{
      "rp1": {
        "params": {
            "dense_ratio": 0.5
        },
        "modules": [
          "intermediate.dense"
        ],
        "related_modules":[
          ["layer.\\w+.output.dense"]
        ]
      }
    }
  }
}
```

<i>**shared_parameters**</i>: [dictionary]

Shared parameters for all row pruning groups.

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**enabled**</i>: [boolean] | Enable row pruning or not. | `false` |
| <i>**schedule_offset**</i>: [integer] | Enable row pruning after scheduled steps (can be treated as warmup steps). | `0` |
| <i>**method**</i>: [string] | Choose different pruning methods, l1 (static, magnitude based) or topk (dynamic, learnable). | `"l1"` |

<i>**different_groups**</i>: [dictionary]

Different pruning sets, this is used for different pruning parameters. In this example, we give one set. In practice, you can choose the number of sets based on your requirements.

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**params**</i>: [dictionary] | | |
| <i>&emsp;&emsp;**dense_ratio**</i>: [float] | The percentage of weights to keep after pruning. | `0.5` |
| <i>**modules**</i>: [list] | Scope of weight parameters associated to the params setting. | `"All Linear and CONV2D layers"` |
| <i>**related_modules**</i>: [list[list]] | Related module to the row pruned module, which can be performed column pruning. | `None` |

#### Head Pruning
**Note:** <i>**Head Pruning**</i> is a feature designed for two attention layers (e.g., Multi Head Attention in Transformers). For now, it can only be applied to output matrix of the Transformer (i.e., `attention.output.dense` in BERT). Pruning the output matrix can lead to the pruning of Query/Key/Value matrix as well.
```json
"compression_training": {
  "head_pruning":{
    "shared_parameters":{
      "enabled": true,
      "schedule_offset": 10,
      "method": "topk",
      "num_heads": 12
    },
    "different_groups":{
      "rp1": {
        "params": {
            "dense_ratio": 0.5
        },
        "modules": [
          "attention.output.dense"
        ],
        "related_modules":[
          ["self.query", "self.key", "self.value"]
        ]
      }
    }
  }
}

```

<i>**shared_parameters**</i>: [dictionary]

Shared parameters for all head pruning groups.

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**enabled**</i>: [boolean] | Enable head pruning or not. | `false` |
| <i>**schedule_offset**</i>: [integer] | Enable head pruning after scheduled steps (can be treated as warmup steps). | `0` |
| <i>**method**</i>: [string] | Choose different pruning methods. For now, we only support topk (dynamic, learnable). | `"topk"` |
| <i>**num_heads**</i>: [int] | Number of heads (must be provided by user). | N/A |

<i>**different_groups**</i>: [dictionary]

Different pruning sets, this is used for different pruning parameters. In this example, we give one set. In practice, you can choose the number of sets based on your requirements.

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**params**</i>: [dictionary] | | |
| <i>&emsp;&emsp;**dense_ratio**</i>: [float] | The percentage of weights to keep after pruning. | `0.5` |
| <i>**modules**</i>: [list] | Scope of weight parameters associated to the params setting. | `"All Linear and CONV2D layers"` |
| <i>**related_modules**</i>: [list[list]] | Related module (Usually Q/K/V) to the head pruned module (i.e., the output matrix). For now, this feature only works for BERT. | `None` |

#### Channel Pruning
**Note:** <i>**Channel Pruning**</i> is a feature designed for two back-to-back CONV2d layers (e.g., residual connection in ResNet). As such, we suggested use channel pruning for the first CONV2d layer. Reducing the number of output channels of this layer can help reducing the number of input channels the follow-up layer. It should also work for other CONV2d layers as well.
```json
"compression_training": {
"channel_pruning":{
      "shared_parameters":{
        "enabled": true,
        "schedule_offset": 0,
        "method": "topk"
      },
      "different_groups":{
        "cp1": {
          "params": {
              "dense_ratio": 0.5
          },
          "modules": [
            "layer....conv1"
          ],
          "related_modules": [
            ["layer....conv2", "layer....bn1"]
          ]
        }
      }
    }
}
```

<i>**shared_parameters**</i>: [dictionary]

Shared parameters for all channel pruning groups.

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**enabled**</i>: [boolean] | Enable channel pruning or not. | `false` |
| <i>**schedule_offset**</i>: [integer] | Enable channel pruning after scheduled steps (can be treated as warmup steps). | `0` |
| <i>**method**</i>: [string] | Choose different pruning methods, l1 (static, magnitude based) or topk (dynamic, learnable). | `"l1"` |

<i>**different_groups**</i>: [dictionary]

Different pruning sets, this is used for different pruning parameters. In this example, we give one set. In practice, you can choose the number of sets based on your requirements.

| Fields | Value | Default |
| ----- | ----- | ----- |
| <i>**params**</i>: [dictionary] | | |
| <i>&emsp;&emsp;**dense_ratio**</i>: [float] | The percentage of weights to keep after pruning. | `0.5` |
| <i>**modules**</i>: [list] | Scope of weight parameters associated to the params setting. | `"All CONV2D layers"` |
| <i>**related_modules**</i>: [list[list]] | Related module to the channel pruned module. | `None` |

### Checkpoint options

```json
"checkpoint": {
    "tag_validation"="Warn",
    "load_universal"=false,
    "use_node_local_storage"=false,
    "parallel_write":{
        "pipeline_stage": false
    }
}
```

<i>**tag_validation**</i>: ["Ignore"|"Warn"|"Fail"]

| Description                                                                                                                            | Default |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| Enables level of checking to ensure checkpoint tags are consistent across all ranks. Useful when restoring with different world sizes. |  "Warn" |

<i>**load_universal**</i>: [boolean]

| Description                            | Default |
| -------------------------------------- | ------- |
| Load the latest checkpoint for all.    | `false` |

<i>**use_node_local_storage**</i>: [boolean]

| Description                                                                                                                                                               | Default |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| If `true` DeepSpeed will store model parameter states and checkpoint states based on local rank allowing checkpoints to be loaded without access to a shared filesystem.  | `false` |

<i>**pipeline_stage**</i>: [boolean]

| Description                                                   | Default |
| ------------------------------------------------------------- | ------- |
| Use pipeline stages to parallelize the writing of checkpoints.| `false` |

### Data Type options

```json
"data_types": {
    "grad_accum_dtype"=["fp32"|"fp16"|"bf16"]
    }
}
```

<i>**grad_accum_dtype**</i>: ["fp32"|"fp16"|"bf16"]

| Description                                                                                                   | Default |
| --------------------------------------------------------------------------------------------------------------| ------- |
| Specifies the data type in which to do gradient accumulation. If None the default is to match the model type. |  None   |
