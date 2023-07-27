---
title: "Maximizing Communication Efficiency for Large-scale Training via 0/1 Adam"
---

**Watch out!**
1) The NCCL-based implementation requires PyTorch >= 1.8 (and NCCL >= 2.8.3 when you have 64 or more GPUs). See details below. 2) Although 0/1 Adam is compatible with both FP16 and FP32, currently we only verified the convergence under mixed precision/FP16 training. 3) Currently the MPI-based implementation is not compatible with pipeline parallelism. 4) Frequent checkpoint loading could hurt 0/1 Adam's convergence. See details below.
{: .notice--warning}

In this tutorial, we introduce DeepSpeed's 0/1 Adam optimizer, which can improve model training speed on communication-constrained clusters, especially for communication-intensive large models. For instance, it is able to reduce the overall communication volume on BERT-large pre-training by up to 26x without affecting the end-to-end model accuracy.
Compared to the 1-bit Adam optimizer, 0/1 Adam provides a more flexible way of using compressed communication via adaptive variance state freezing. Additionally, it allows the computing nodes to skip communication rounds during training using a technique called 1-bit sync, without compromising the convergence speed.
We have a [paper](https://arxiv.org/abs/2202.06009) which provides the technical details including algorithm, system implementation, and evaluations.

To illustrate the benefits and usage of 0/1 Adam optimizer, we use the BERT Pre-training task as example. For more details on this task, please refer to the [tutorial](/tutorials/bert-pretraining/).

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

### 1.2 Pre-requisites for 0/1 Adam

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

An example launch command for 0/1 Adam using the `deepspeed` launcher is as follows:

```shell
deepspeed --launcher=[mvapich|openmpi] script.py
```

Please note that for MPI-based implementation of 0/1 Adam, the `--launcher=[mvapich|openmpi]` flag is required when using the `deepspeed` launcher.

Alternatively, the standard mpirun launcher can also be used as follows:

```shell
mpirun -np [num processes] -ppn [num GPUs on each node] -hostfile [hostfile] [MPI flags] python [training_script.py]
```

### 1.3 0/1 Adam Algorithm

The detailed description of the 0/1 Adam algorithm can be seen from our [paper](https://arxiv.org/abs/2202.06009).

### 1.4 Configuration of 0/1 Adam
The 0/1 Adam feature can be used by setting the optimizer configuration options as follows. An example json config file is shown below.

```json
{
  "train_batch_size": 4096,
  "train_micro_batch_size_per_gpu": 16,
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
  },
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  }
}
```
Please note the new parameters `var_freeze_step`, `var_update_scaler`, `local_step_scaler`, `local_step_clipper`, `cuda_aware` and `comm_backend_name` that have been added to support the 0/1 Adam feature:

`var_freeze_step` is the latest step to update the variance. Using the notation from [0/1 Adam paper](https://arxiv.org/abs/2202.06009), it denotes the $\max\{i|i \in \mathcal{T}_v\}$. Note that this is different from the `freeze_step` in 1-bit Adam. The `var_freeze_step` is usually the last step of the learning rate warmup and thus does not require tuning. Note that this hyperparameter is optional. In practice, we can avoid tuning this parameter by setting it to a sufficiently large number (larger than the total number of steps). Following this, 0/1 Adam still enjoys the non-trivial communication reduction without affecting the convergence speed.

`var_update_scaler` is the interval to update the variance. Note that the update policy for variance follows an exponential rule. Formally, if we denote $k_j$ as the step where $j$-th variance update takes place, then it follows that $k_{j+1} - k_j = 2\cdot\exp\{\lfloor j/\kappa\rfloor\}$ (please refer to the [0/1 Adam paper](https://arxiv.org/abs/2202.06009) for detailed explanation), and the `var_update_scaler` denotes the $\kappa$ factor in such expression.
In practice, we found its default value (16) is able to work well on most of the tasks, including BERT-Base/Large pretraining, GPT pretraining, and ImageNet training.

`local_step_scaler` and `local_step_clipper` are two hyperparameters for learning rate based local step policy in 0/1 Adam. Formally, if we denote $k_j$ as the step where $j$-th synchronization takes place among all the workers, then it follows that $k_{j+1} - k_j = 2\cdot\exp\{\min(\lfloor j/\alpha\rfloor, \beta )\}$ (please refer to the [0/1 Adam paper](https://arxiv.org/abs/2202.06009) for detailed explanation). Following such notations, `local_step_scaler` and `local_step_clipper` denote the $\alpha$ and $\beta$, respectively. Informally, `local_step_scaler` decides the frequency of synchronization while `local_step_clipper` denotes the maximal local step interval 0/1 Adam can use.
The learning rate policy is the default policy used in 0/1 Adam, and the value of `local_step_scaler` can be pre-calculated (see [0/1 Adam paper](https://arxiv.org/abs/2202.06009) Section 6). We can also trivially construct other policies by setting these two hyperparameters such as constant local step interval policy by setting `local_step_scaler=1` and `local_step_clipper=constant`.

`cuda_aware` is used for MPI-based implementation to indicate that the underlying MPI library supports CUDA-Aware communication. This feature is only supported on systems with InfiniBand interconnect and a CUDA-Aware MPI library like [MVAPICH2-GDR](http://mvapich.cse.ohio-state.edu/userguide/gdr/) or OpenMPI built with CUDA-Aware support. Setting `cuda_aware` to False will allow training on Ethernet based systems. However, the communication will happen using sender as well as receiver side memory copies between CPU and GPU buffers before and after communication.

`comm_backend_name` is used to indicate which backend implementation to use. You can choose between NCCL and MPI-based implementations by setting `comm_backend_name` to "nccl" or "mpi". When using NCCL-based implementation, there is no need to set `cuda_aware`.

#### 1.4.1 Momentum masks for parameters with constant zero gradients
Because 1-bit compression cannot represent exact zero, the compression error would keep accumulating in the momentum if a parameter have constant zero gradients during training. For example, for BERT pre-training seq length 128, `bert.embeddings.position_embeddings.weight` has constant zeros in its gradient and momentum for row 129 to 512, because it only learns up to seq length 128 while the model supports up to seq length 512. Thus in 0/1 Adam we added support of a momentum mask for users to specify those params that have constant exact zeros in their gradients. See [example script](https://github.com/microsoft/DeepSpeedExamples/blob/master/bing_bert/deepspeed_train.py) for how to configure this momentum mask. One thing to note is that we don't use momentum mask saved in checkpoints since this mask could change during training (e.g., BERT seqlen 128 and 512 require different masks). So you have to provide this mask every time in your training script.

**Watch out!**
0/1 Adam relies on an compression error compensation mechanism to maintain the convergence speed at compression stage. When loading checkpoints, aside from resetting the compression errors as 1-bit Adam, we additionally need to reset the local step buffer. Since the local step buffer can potentially fail to capture the training dynamics if the checkpoints are loaded by different number of nodes (GPUs).
{: .notice--warning}

## 2. BERT Pre-training with 0/1 Adam
For data downloading and pre-processing, please refer to the [BERT Pre-training tutorial](/tutorials/bert-pretraining/).

### 2.1 Running Pre-training with DeepSpeed and 0/1 Adam

We provide example scripts under [DeepSpeedExamples/bing_bert/01_adam/](https://github.com/microsoft/DeepSpeedExamples/tree/master/bing_bert/01_adam). There are 3 sets of scripts corresponding to NCCL-based implementation, MPI-based implementation on Ethernet systems, and MPI-based implementation on InfiniBand systems. For MPI-based implementation, we provide both example scripts when launching with deepspeed or mpirun.

### 2.2 Configuration for BERT Pre-training with DeepSpeed and 0/1 Adam enabled

The `deepspeed_bsz4k_01adam_config_seq128_*.json` and `deepspeed_bsz4k_01adam_config_seq512_*.json` files give the user the ability to specify DeepSpeed
options in terms of batch size, micro batch size, optimizer, learning rate, and other parameters. In these files we include the tuned hyperparameters to reproduce experiments in our [paper](https://arxiv.org/abs/2202.06009).

### 2.3 Performance Results for BERT Pre-training

Performance results can be seen in our [paper](https://arxiv.org/abs/2202.06009).

### 2.4 GLUE Fine-tuning
We additionally provide the fine-tuning scripts for BERT pre-training checkpoints over [GLUE tasks](https://gluebenchmark.com/). The scripts are available at [DeepSpeedExamples/BingBertGlue](https://github.com/microsoft/DeepSpeedExamples/tree/master/BingBertGlue). The `glue_bert_base.json` and `glue_bert_large.json` files give the user the ability to specify DeepSpeed
options/parameters like micro batch size over BERT-base and BERT-large checkpoints, respectively. Currently we use Adam as the default optimizer for GLUE fine-tuning since the fine-tuning tasks usually use small batch size (~32) and do not require large-scale systems. `run_glue_bert_base_finetune.sh` and `run_glue_bert_large_finetune.sh` give the scripts for launching fine-tuning tasks, where we can modify variables like task name, number of epochs, model, etc. Note that to launch the fine-tuning, we must specify the path for checkpoint, for instance,
```
bash run_glue_bert_base_finetune.sh <path to checkpoint>
```
Specific GLUE scores and hyperparameters for 0/1 Adam are included in our [paper](https://arxiv.org/abs/2202.06009) Table 1.
