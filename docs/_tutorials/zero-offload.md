---
title: "ZeRO-Offload"
tags: training IO large-model
---
ZeRO-3 Offload consists of a subset of features in our newly released ZeRO-Infinity. Read our [ZeRO-Infinity blog](https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/) to learn more!

We recommend that you read the tutorials on [Getting Started](/getting-started/)  and [ZeRO](/tutorials/zero/) before stepping through this tutorial.

ZeRO-Offload is a ZeRO optimization that offloads the optimizer memory and computation from the GPU to the host CPU. ZeRO-Offload enables large models with up to 13 billion parameters to be efficiently trained on a single GPU. In this tutorial we will use ZeRO-Offload to train a 10-billion parameter GPT-2 model in DeepSpeed. Furthermore, *using ZeRO-Offload in a DeepSpeed model is quick and easy because all you need is to change a few configurations in the DeepSpeed configuration json*. No code changes are needed.

## ZeRO-Offload Overview
For large model training, optimizers such as [Adam](https://arxiv.org/abs/1412.6980), can consume a significant amount of GPU compute and memory. ZeRO-Offload reduces the GPU compute and memory requirements of such models by leveraging compute and memory resources on the host CPU  to execute the optimizer. Furthermore, to prevent the optimizer from becoming a bottleneck, ZeRO-Offload uses DeepSpeed's highly optimized CPU implementation of Adam called [DeepSpeedCPUAdam](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/ops/adam). DeepSpeedCPUAdam is 5X--7X faster than the standard PyTorch implementation. To deep dive into the design and performance of ZeRO-Offload, please see our [blog post](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/#toc-heading-3).

## Training Environment
For this tutorial, we will configure a 10 billion parameter GPT-2 model using the DeepSpeed [Megatron-LM](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM-v1.1.5-ZeRO3) GPT-2 code. We advise stepping through the Megatron-LM [tutorial](/tutorials/megatron/) if you have not previously done so. We will use a single [NVIDIA Tesla V100-SXM3 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/v100/) with 32GB RAM for this exercise.

## Training a 10B parameter GPT-2 on a single V100 GPU
We need to make changes to the Megatron-LM launch script and to the DeepSpeed configuration json.

### Megatron-LM GPT-2 launch script changes
We need to apply two changes to the launch script for the DeepSpeed Megatron-LM GPT-2 model. The first change is to configure a 10B parameter GPT-2 model with activation checkpointing enabled, which can be achieved by the following set of changes:

```bash
       --model-parallel-size 1 \
       --num-layers 50 \
       --hidden-size 4096 \
       --num-attention-heads 32 \
       --batch-size 10 \
       --deepspeed_config ds_zero_offload.config \
       --checkpoint-activations
```

Most of the flags in the changes above should be familiar if you have stepped through the Megatron-LM [tutorial](/tutorials/megatron/).

Second, we need to apply the following changes to ensure that only one GPU is used for training.
```bash
   deepspeed --num_nodes 1 --num_gpus 1 ...
```

### DeepSpeed Configuration Changes
ZeRO-Offload leverages many ZeRO stage 1 and 2 mechanisms, and so the configuration changes to enable ZeRO-Offload are an extension of those required to enable ZeRO stage 1 or 2. The `zero_optimization` configuration to enable ZeRO-Offload is shown below:

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
        }
        "contiguous_gradients": true,
        "overlap_comm": true
    }
}
```

As seen above, in addition to setting the _stage_ field to **2** (to enable ZeRO stage 2, but stage 1 also works), we also need to set the _offload\_optimizer_ device to **cpu** to enable ZeRO-Offload optimizations. In addition, we can set other ZeRO stage 2 optimization flags, such as _overlap\_comm_ to tune ZeRO-Offload performance.  With these changes we can now run the model. We share some screenshots of the training below.

Here is a screenshot of the training log:

<a href="/assets/images/zero_offload_dp1_10B_log.png">
<img src="/assets/images/zero_offload_dp1_10B_log.png">
</a>


Here is a screenshot of `nvidia-smi` showing that only GPU 0 is active during training:

<a href="/assets/images/zero_offload_dp1_10B_smi.png">
<img src="/assets/images/zero_offload_dp1_10B_smi.png">
</a>

Finally, here is a screenshot of `htop` showing host CPU and memory activity during optimizer computation:

<a href="/assets/images/zero_offload_dp1_10B_cpu.png">
<img src="/assets/images/zero_offload_dp1_10B_cpu.png">
</a>

Congratulations! You have completed the ZeRO-Offload tutorial.
