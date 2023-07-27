---
title: "1-bit LAMB: Communication Efficient Large-Scale Large-Batch Training with LAMB's Convergence Speed"
tags: training IO
---

**Watch out!**
1) The NCCL-based implementation requires PyTorch >= 1.8 (and NCCL >= 2.8.3 when you have 64 or more GPUs). See details below. 2) Although 1-bit LAMB is compatible with both FP16 and FP32, currently we only verified the convergence under mixed precision/FP16 training. 3) Currently the MPI-based implementation is not compatible with pipeline parallelism. 4) Frequent checkpoint loading could hurt 1-bit LAMB's convergence. See details below.
{: .notice--warning}

In this tutorial, we introduce DeepSpeed's 1-bit LAMB optimizer which enables communication-efficient large-scale large-batch training with LAMB's convergence speed. 1-bit LAMB can improve model training speed on communication-constrained clusters, especially for communication-intensive large models by reducing the overall communication volume by up to 4.6x. We also have a [paper](https://arxiv.org/abs/2104.06069) which provides the technical details including algorithm, system implementation, and evaluations.

To illustrate the benefits and usage of 1-bit LAMB optimizer, we use the BERT Pre-training task as example. For more details on this task, please refer to the [tutorial](/tutorials/bert-pretraining/).

## 1. Overview

### 1.1 Pre-requisites for installing DeepSpeed

If you don't already have a copy of the DeepSpeed repository, please clone it
now and checkout the DeepSpeedExamples submodule that contains the BERT Pre-training example.

```shell
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
git submodule update --init --recursive
cd DeepSpeedExamples/
```

### 1.2 Pre-requisites for 1-bit LAMB

#### 1.2.1 NCCL-based implementation

In DeepSpeed, we introduce a system implementation for compressed communication using the NCCL backend of PyTorch distributed. This implementation provides better performance and usability than the MPI-based implementation below. Thus we highly recommend users to choose this implementation.

**Watch out!**
This NCCL-based implementation requires PyTorch >= 1.8. It also requires NCCL >= 2.8.3 when you have 64 or more GPUs to avoid certain NCCL runtime bugs. Currently (2021/03/16) NCCL 2.8.3 is not officially supported by PyTorch. The solution we used is by hacking in NCCL 2.8.3 via `LD_PRELOAD`: 1) Install NCCL 2.8.3. This works for us on a CUDA 11 system: `apt-get install -y libnccl2=2.8.3-1+cuda11.0 libnccl-dev=2.8.3-1+cuda11.0`. 2) Set `LD_PRELOAD` to the library path. This works for us: `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.8.3`. To confirm `LD_PRELOAD` is working you can see the version it uses in the NCCL logs if you have `NCCL_DEBUG=INFO`, it should say: NCCL version 2.8.3+cuda11.0.
{: .notice--warning}

#### 1.2.2 MPI-based implementation

For this implementation, we rely on Message Passing Interface (MPI) for advanced communication primitives.

We package the necessary dependencies in the DeepSpeed docker images. However, if you are using a different build system, please install MPI and mpi4py on your system. To install the prerequisites run:

```shell
pip install deepspeed[1bit_adam]
```

We have tested CUDA-Aware MPI communication using the [MVAPICH2-GDR](http://mvapich.cse.ohio-state.edu/userguide/gdr/) library. However, any CUDA-Aware communication library including [OpenMPI](https://www.open-mpi.org/) should work fine with these examples.

An example launch command for 1-bit LAMB using the `deepspeed` launcher is as follows:

```shell
deepspeed --launcher=[mvapich|openmpi] script.py
```

Please note that for MPI-based implementation of 1-bit LAMB, the `--launcher=[mvapich|openmpi]` flag is required when using the `deepspeed` launcher.

Alternatively, the standard mpirun launcher can also be used as follows:

```shell
mpirun -np [num processes] -ppn [num GPUs on each node] -hostfile [hostfile] [MPI flags] python [training_script.py]
```

### 1.3 1-bit LAMB Algorithm

The detailed description of the 1-bit LAMB algorithm can be seen from our [paper](https://arxiv.org/abs/2104.06069).

### 1.4 Configuration of 1-bit LAMB
The 1-bit LAMB feature can be used by setting the optimizer configuration options as follows. An example json config file is shown below.

```json
{
  "train_batch_size": 65536,
  "train_micro_batch_size_per_gpu": 64,
  "optimizer": {
    "type": "OneBitLamb",
    "params": {
      "lr": 11e-3,
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
  },
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  }
}
```
Please note the new parameters `freeze_step`, `cuda_aware`, `comm_backend_name`, `coeff_beta`, `factor_max`, `factor_min`, and `factor_threshold` that have been added to support the 1-bit LAMB feature:

`freeze_step` is the number of warm up steps before 1-bit compression gets applied to the communication. In order to determine the number of warm up steps, one strategy is to set 15-25% of the total training steps for a given model (This is related to LAMB's variance/second moment term and scaling coefficient. See detailed analysis in our [paper](https://arxiv.org/abs/2104.06069)). If it provides the desired outcome, one can try to extract more performance by reducing the steps systematically. In future, we plan to introduce a threshold that can automatically search and decide for the number of warm up steps for different models. The examples below have been tuned for the number of warm up steps. The `freeze_step` parameter has already been set to the best number we found in the corresponding run scripts.

`cuda_aware` is used for MPI-based implementation to indicate that the underlying MPI library supports CUDA-Aware communication. This feature is only supported on systems with InfiniBand interconnect and a CUDA-Aware MPI library like [MVAPICH2-GDR](http://mvapich.cse.ohio-state.edu/userguide/gdr/) or OpenMPI built with CUDA-Aware support. Setting `cuda_aware` to False will allow training on Ethernet based systems. However, the communication will happen using sender as well as receiver side memory copies between CPU and GPU buffers before and after communication.

`comm_backend_name` is used to indicate which backend implementation to use. You can choose between NCCL and MPI-based implementations by setting `comm_backend_name` to "nccl" or "mpi". When using NCCL-based implementation, there is no need to set `cuda_aware`.

`coeff_beta` is used when calculating a moving average of the LAMB scaling coefficient during the warmup stage. This moving average is then used as the frozen base scaling coefficient during the compression stage.

`factor_max`, `factor_min`, and `factor_threshold` are used to regularize the adaptive scaling of the frozen base scaling coefficient during the compression stage. `factor_max` and `factor_min` are the scaling factor upper/lower bound. `factor_threshold` defines the threshold of how much the scaling factor can fluctuate between steps.

#### 1.4.1 Momentum masks for parameters with constant zero gradients
Because 1-bit compression cannot represent exact zero, the compression error would keep accumulating in the momentum if a parameter have constant zero gradients during training. For example, for BERT pre-training seq length 128, `bert.embeddings.position_embeddings.weight` has constant zeros in its gradient and momentum for row 129 to 512, because it only learns up to seq length 128 while the model supports up to seq length 512. Thus in 1-bit LAMB we added support of a momentum mask for users to specify those params that have constant exact zeros in their gradients. See [example script](https://github.com/microsoft/DeepSpeedExamples/blob/master/bing_bert/deepspeed_train.py) for how to configure this momentum mask. One thing to note is that we don't use momentum mask saved in checkpoints since this mask could change during training (e.g., BERT seqlen 128 and 512 require different masks). So you have to provide this mask every time in your training script.

**Watch out!**
1-bit LAMB relies on an compression error compensation mechanism to maintain the convergence speed at compression stage. When loading checkpoints, we actually reset the compression errors for 3 reasons: 1) The worker and server error at each GPU are distinct, so in current implementation only rank 0's errors are saved in the checkpoint. Thus we have to reset the errors. If we want to save them correctly we need O(num_gpu*model_size) memory in order to gather all the error, which is a very large memory requirement. It's possible to save them in a distributed way, but it will make the checkpoint saving/loading much more complicated. 2) Even if we are able to save the compression errors correctly, you need to have the exact same number of GPUs in order to load them correctly. 3) We verified on BERT pre-training that occasionally resetting the compression error at checkpoint loading does not affect the convergence. However, please avoid frequent checkpoint loading which could break the error compensation mechanism thus affect the convergence.
{: .notice--warning}

## 2. BERT Pre-training with 1-bit LAMB
For data downloading and pre-processing, please refer to the [BERT Pre-training tutorial](/tutorials/bert-pretraining/).

### 2.1 Running Pre-training with DeepSpeed and 1-bit LAMB

We provide example scripts under [DeepSpeedExamples/bing_bert/1-bit_lamb/](https://github.com/microsoft/DeepSpeedExamples/tree/master/bing_bert/1-bit_lamb). There are 3 sets of scripts corresponding to NCCL-based implementation, MPI-based implementation on Ethernet systems, and MPI-based implementation on InfiniBand systems. For MPI-based implementation, we provide both example scripts when launching with deepspeed or mpirun.

### 2.2 Configuration for BERT Pre-training with DeepSpeed and 1-bit LAMB enabled

The `deepspeed_bsz64k_onebitlamb_config_seq128_*.json` and `deepspeed_bsz32k_onebitlamb_config_seq512_*.json` files give the user the ability to specify DeepSpeed
options in terms of batch size, micro batch size, optimizer, learning rate, and other parameters. In these files we include the tuned hyperparameters to reproduce experiments in our [paper](https://arxiv.org/abs/2104.06069).

### 2.3 Performance Results for BERT Pre-training

Performance results can be seen in our [paper](https://arxiv.org/abs/2104.06069).
