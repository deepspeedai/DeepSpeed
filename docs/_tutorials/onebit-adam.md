---
title: "1-bit Adam: Up to 5x less communication volume and up to 3.4x faster training"
tags: training IO
toc: false
---

**Note:**
On 03/07/2022 we released 0/1 Adam, which is a new communication-efficient Adam optimizer partially following the 1-bit Adam's design. Compared to the 1-bit Adam described below, 0/1 Adam provides better communication efficiency and the same final model quality on different tasks including BERT, GPT-2, and ImageNet. Thus we would recommend to first try 0/1 Adam ([tutorial](/tutorials/zero-one-adam/)), and then try 1-bit Adam if 0/1 Adam couldn't provide baseline Adam's convergence in your task.
{: .notice--info}

**Note:**
This tutorial is updated on 03/04/2021 to reflect the 1-bit Adam v2. Changes include: 1) NCCL-based implementation which provides better performance and usability compared to the MPI-based implementation. 2) Add support to momentum masks for those parameters with constant zero gradients during training. 3) Bug fixes. See details below.
{: .notice--info}

**Watch out!**
1) The NCCL-based implementation requires PyTorch >= 1.8 (and NCCL >= 2.8.3 when you have 64 or more GPUs). See details below. 2) Although 1-bit Adam is compatible with both FP16 and FP32, currently we only verified the convergence under mixed precision/FP16 training. 3) Currently the MPI-based implementation is not compatible with pipeline parallelism. 4) Frequent checkpoint loading could hurt 1-bit Adam's convergence. See details below.
{: .notice--warning}

In this tutorial, we are going to introduce the 1-bit Adam optimizer in DeepSpeed. 1-bit Adam can improve model training speed on communication-constrained clusters, especially for communication-intensive large models by reducing the overall communication volume by up to 5x. Detailed description of the 1-bit Adam algorithm, its implementation in DeepSpeed, and performance evaluation is available from our [blog post](https://www.deepspeed.ai/2020/09/08/onebit-adam-blog-post.html). We also have a [paper](https://arxiv.org/abs/2102.02888) which provides the most complete details including algorithm, system implementation, theoretical analysis, and more evaluations.

To illustrate the benefits and usage of 1-bit Adam optimizer in DeepSpeed, we use the following two training tasks as examples:

1. BingBertSQuAD Fine-tuning
2. BERT Pre-training

For more details on these tasks, please refer to the tutorial posts on [BingBertSQuAD Fine-tuning](/tutorials/bert-finetuning/) and [BERT Pre-training](/tutorials/bert-pretraining/).

## 1. Overview

### 1.1 Pre-requisites for installing DeepSpeed

If you don't already have a copy of the DeepSpeed repository, please clone it
now and checkout the DeepSpeedExamples submodule that contains the BingBertSQuAD and BERT Pre-training examples.

```shell
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
git submodule update --init --recursive
cd DeepSpeedExamples/
```

### 1.2 Pre-requisites for 1-bit Adam

#### 1.2.1 (New in v2) NCCL-based implementation

In 1-bit Adam v2, we introduce a new system implementation for compressed communication using the NCCL backend of PyTorch distributed. This significantly improves the usability due to NCCL’s integration with PyTorch distributed. The performance of our new NCCL-based implementation is also better than our earlier MPI-based implementation for Ethernet-based systems and on-par for InfiniBand-based systems. Thus we highly recommend users to choose this implementation.

**Watch out!**
This NCCL-based implementation requires PyTorch >= 1.8. It also requires NCCL >= 2.8.3 when you have 64 or more GPUs to avoid certain NCCL runtime bugs. Currently (2021/03/16) NCCL 2.8.3 is not officially supported by PyTorch. The solution we used is by hacking in NCCL 2.8.3 via `LD_PRELOAD`: 1) Install NCCL 2.8.3. This works for us on a CUDA 11 system: `apt-get install -y libnccl2=2.8.3-1+cuda11.0 libnccl-dev=2.8.3-1+cuda11.0`. 2) Set `LD_PRELOAD` to the the library path. This works for us: `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.8.3`. To confirm `LD_PRELOAD` is working you can see the version it uses in the NCCL logs if you have `NCCL_DEBUG=INFO`, it should say: NCCL version 2.8.3+cuda11.0.
{: .notice--warning}

#### 1.2.2 MPI-based implementation

For this implementation, we rely on Message Passing Interface (MPI) for advanced communication primitives.

We package the necessary dependencies in the DeepSpeed docker images. However, if you are using a different build system, please install MPI and mpi4py on your system. To install the prerequisites run:

```shell
pip install deepspeed[1bit_adam]
```

We have tested CUDA-Aware MPI communication using the [MVAPICH2-GDR](http://mvapich.cse.ohio-state.edu/userguide/gdr/) library. However, any CUDA-Aware communication library including [OpenMPI](https://www.open-mpi.org/) should work fine with these examples.

An example launch command for 1-bit Adam using the `deepspeed` launcher is as follows:

```shell
deepspeed --launcher=[mvapich|openmpi] script.py
```

Please note that for MPI-based implementation of 1-bit Adam, the `--launcher=[mvapich|openmpi]` flag is required when using the `deepspeed` launcher.

Alternatively, the standard mpirun launcher can also be used as follows:

```shell
mpirun -np [#processes] -ppn [#GPUs on each node] -hostfile [hostfile] [MPI flags] python [training_script.py]
```

### 1.3 1-bit Algorithm

The detailed description of the 1-bit Algorithm can be seen from our [blog post](https://www.deepspeed.ai/2020/09/08/onebit-adam-blog-post.html) and our [paper](https://arxiv.org/abs/2102.02888).

### 1.4 Configuration of 1-bit Adam
The 1-bit Adam feature can be used by setting the optimizer configuration options as follows. An example json config file is shown below.

```json
{
  "train_batch_size": 4096,
  "train_micro_batch_size_per_gpu": 16,
  "optimizer": {
    "type": "OneBitAdam",
    "params": {
      "lr": 4e-4,
      "freeze_step": 23000,
      "cuda_aware": false,
      "comm_backend_name": "nccl"
    }
  },
  "fp16": {
    "enabled": true,
  }
}
```
Please note three new parameters `freeze_step`, `cuda_aware`, and `comm_backend_name` that have been added to support the 1-bit Adam feature.

`freeze_step` is the number of warm up steps before 1-bit compression gets applied to the communication. In order to determine the number of warm up steps, one strategy is to set 15-25% of the total training steps for a given model (This is related to Adam's variance/second moment term. See detailed analysis in our [paper](https://arxiv.org/abs/2102.02888)). If it provides the desired outcome, one can try to extract more performance by reducing the steps systematically. In future, we plan to introduce a threshold that can automatically search and decide for the number of warm up steps for different models. The examples below have been tuned for the number of warm up steps. The `freeze_step` parameter has already been set to the best number we found in the corresponding run scripts.

`cuda_aware` is used for MPI-based implementation to indicate that the underlying MPI library supports CUDA-Aware communication. This feature is only supported on systems with InfiniBand interconnect and a CUDA-Aware MPI library like [MVAPICH2-GDR](http://mvapich.cse.ohio-state.edu/userguide/gdr/) or OpenMPI built with CUDA-Aware support. Setting `cuda_aware` to False will allow training on Ethernet based systems. However, the communication will happen using sender as well as receiver side memory copies between CPU and GPU buffers before and after communication.

(New in v2) `comm_backend_name` is used to indicate which backend implementation to use. You can choose between NCCL and MPI-based implementations by setting `comm_backend_name` to "nccl" and "mpi". When using NCCL-based implementation, there is no need to set `cuda_aware`.

#### 1.4.1 (New in v2) Momentum masks for parameters with constant zero gradients
Because 1-bit compression cannot represent exact zero, the compression error would keep accumulating in the momentum if a parameter have constant zero gradients during training. For example, for BERT pre-training seq length 128, `bert.embeddings.position_embeddings.weight` has constant zeros in its gradient and momentum for row 129 to 512, because it only learns up to seq length 128 while the model supports up to seq length 512. Thus in 1-bit Adam v2 we added support of a momentum mask for users to specify those params that have constant exact zeros in their gradients. See [example script](https://github.com/microsoft/DeepSpeedExamples/blob/master/bing_bert/deepspeed_train.py) for how to configure this momentum mask. One thing to note is that we don't use momentum mask saved in checkpoints since this mask could change during training (e.g., BERT seqlen 128 and 512 require different masks). So you have to provide this mask every time in your training script.

**Watch out!**
1-bit Adam relies on an compression error compensation mechanism to maintain the convergence speed at compression stage. When loading checkpoints, we actually reset the compression errors for 3 reasons: 1) The worker and server error at each GPU are distinct, so in current implementation only rank 0's errors are saved in the checkpoint. Thus we have to reset the errors. If we want to save them correctly we need O(num_gpu*model_size) memory in order to gather all the error, which is a very large memory requirement. It's possible to save them in a distributed way, but it will make the checkpoint saving/loading much more complicated. 2) Even if we are able to save the compression errors correctly, you need to have the exact same number of GPUs in order to load them correctly. 3) We verified on BERT pre-training that occasionally resetting the compression error at checkpoint loading does not affect the convergence. However, please avoid frequent checkpoint loading which could break the error compensation mechanism thus affect the convergence.
{: .notice--warning}

## 2. BingBertSQuAD Fine-tuning with 1-bit Adam

* Download the SQuAD dataset:
  * Training set: [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
  * Validation set: [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
* Download the HuggingFace checkpoint and config files:
  * [bert-large-uncased-whole-word-masking](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin)
  * [bert json config](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json)

You can also use a pre-trained BERT model checkpoint from either DeepSpeed, [HuggingFace](https://github.com/huggingface/transformers), or [TensorFlow](https://github.com/google-research/bert#pre-trained-models) to run the fine-tuning.

**Note:** For details about loading checkpoint, argument parsing, initialization, forward pass, backward pass, weight update and evaluation, please refer to the [BingBertSQuAD Fine-tuning](/tutorials/bert-finetuning/) tutorial.

### 2.1 Running BingBertSQuAD with DeepSpeed and 1-bit Adam

We provide example scripts under [DeepSpeedExamples/BingBertSquad/1-bit_adam/](https://github.com/microsoft/DeepSpeedExamples/tree/master/BingBertSquad/1-bit_adam). There are 3 sets of scripts corresponding to NCCL-based implementation, MPI-based implementation on Ethernet systems, and MPI-based implementation on InfiniBand systems. For MPI-based implementation, we provide both example scripts when launching with deepspeed or mpirun.

<!-- The main part of training is done in `nvidia_run_squad_deepspeed.py`, which has
already been modified to use DeepSpeed. The `run_squad_deepspeed.sh` script
helps to invoke training and setup several different hyperparameters relevant
to the training process.

- **DeepSpeed-enabled:** Start training with DeepSpeed by providing the following 4 arguments to this script:

```shell
bash run_squad_deepspeed.sh <NUM_GPUS> <PATH_TO_CHECKPOINT> <PATH_TO_DATA_DIR> <PATH_TO_OUTPUT_DIR>`
```

The first argument is the number of GPUs to train with, second argument is the path to the pre-training checkpoint, third is the path to training and validation sets (e.g., train-v1.1.json), and fourth is path to an output folder where the results will be saved. This script will invoke `nvidia_run_squad_deepspeed.py`.

- **DeepSpeed with 1-bit Adam enabled:** In order to run with 1-bit Adam feature enabled, the same script (`nvidia_run_squad_deepspeed.py`) can be used but there are two options for launching this properly: 1) Launch using deepspeed launcher and 2) Launch with mpirun.

To enable the 1-bit compressed training, 1-bit Adam uses an MPI library (E.g. MVAPICH2-GDR, OpenMPI, etc.) as the communication backend, which means that we can use `mpirun` to launch the training job. However, our user-friendly launcher called `deepspeed` has been enhanced to launch MPI jobs as well.

### Launch with deepspeed

The following helper script in the DeepSpeedExamples/BingBertSQuAD will launch the training without the need for setting any `mpirun` parameters. The number of nodes and GPUs will be automatically detected and the job will be launched on all the available resources.

```shell
bash run_squad_deepspeed_onebitadam.sh <PATH_TO_OUTPUT_DIR>
```

### Launch with mpirun

Alternatively, we show how the standard `mpirun` launcher can be used for launching the fine-tuning job.

```shell
mpirun -np [#processes] -ppn [#GPUs on each node] -hostfile [hostfile] [MPI flags] bash run_squad_mpi_onebitadam.sh
```

For example, in order to use 32 GPUs (4GPUs/node, 8 nodes in total), with the support of InfiniBand, you can use the `mpirun` launcher packaged with the MVAPICH2 library. Please run the following command:

```shell
mpirun -np 32 -ppn 4 -hostfile hosts -env MV2_USE_CUDA=1 -env MV2_SUPPORT_DL=1 -env MV2_ENABLE_AFFINITY=0 -env MV2_SMP_USE_CMA=0 bash run_squad_mpi_onebitadam.sh
``` -->

### 2.2 Configuration for BingBertSQuAD with DeepSpeed and 1-bit Adam enabled

The `deepspeed_onebitadam_bsz96_config.json` file gives the user the ability to specify DeepSpeed
options in terms of batch size, micro batch size, optimizer, learning rate, and other parameters.
When running the `nvidia_run_squad_deepspeed.py`, in addition to the
`--deepspeed` flag to enable DeepSpeed, the appropriate DeepSpeed configuration
file must be specified using `--deepspeed_config deepspeed_onebitadam_bsz96_config.json`.

Table 1 shows the fine-tuning configuration we used in our experiments.

| Parameters                     | Value 		|
| ------------------------------ | ---------------------|
| Total batch size               | 96    		|
| Train micro batch size per GPU | 3     		|
| Optimizer                      | **"OnebitAdam"**  	|
| Learning rate                  | 3e-5  		|
| Sequence-length                | 384   		|
| Weight-decay                   | 0.0   		|
| Epoch count                    | 2     		|
| **freeze_step**                | 400     	   	|
| **comm_backend_name**          | "nccl"     		|

Table 1. Fine-tuning configuration

### 2.3 Performance Results for BingBertSQuAD Fine-tuning

<i>**Accuracy:**</i>
The results are summarized in the table below. The total batch size is set to 96 and training is conducted
on 32 GPUs for 2 epochs. A set of parameters (seeds and learning rates) were tried and the best ones were selected.
We fixed the learning rate to 3e-5. The table below shows the F1 and the EM scores we achieved that are on-par or better than the [HuggingFace results](https://github.com/huggingface/transformers/tree/master/examples/question-answering).

| Case        | Model                                 | Precision | EM    | F1    |
| ----------- | ------------------------------------- | --------- | ----- | ----- |
| HuggingFace | [Bert-large-uncased-whole-word-masking](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin) | FP16      | 87.26 | 93.32 |


***Training Speed and Scalability:***

<!-- 1-bit Adam enables up to 2.7x overall speedup in training speed for SQuAD fine-tuning. This is made possible by up to 6.2x faster throughput during the compressed stage of the algorithm as shown in Figure 1.

![SQuAD Finetuning](/assets/images/squad-scaling.png){: .align-center}

Figure 1: Scalability of 1-bit Adam for SQuAD Finetuning on V100 GPUs with batch size of 3/GPU. -->

Performance results of SQuAD Fine-tuning can be seen from our [blog post](https://www.deepspeed.ai/2020/09/08/onebit-adam-blog-post.html) and our [paper](https://arxiv.org/abs/2102.02888).



## 3. BERT Pre-training with 1-bit Adam
For data downloading and pre-processing, please refer to the [BERT Pre-training](/tutorials/bert-pretraining/) tutorial.

### 3.1 Running Pre-training with DeepSpeed and 1-bit Adam

We provide example scripts under [DeepSpeedExamples/bing_bert/1-bit_adam/](https://github.com/microsoft/DeepSpeedExamples/tree/master/bing_bert/1-bit_adam). There are 3 sets of scripts corresponding to NCCL-based implementation, MPI-based implementation on Ethernet systems, and MPI-based implementation on InfiniBand systems. For MPI-based implementation, we provide both example scripts when launching with deepspeed or mpirun.

<!-- The main part of training is done in `deepspeed_train.py`, which has
already been modified to use DeepSpeed. The `ds_train_bert_onebit_bsz4k_seq128.sh` and `ds_train_bert_bsz64k_seq128.sh`
are the shell scripts that help to invoke training and setup several different hyperparameters relevant
to the training process.

- **DeepSpeed-enabled:** Start training with DeepSpeed by running the command below:

```shell
bash ds_train_bert_bsz64k_seq128.sh
```

- **DeepSpeed with 1-bit Adam enabled:** In order to run with 1-bit Adam feature enabled, the same script (`deepspeed_train.py`) can be used but there are two options for launching this properly:

### Launch with deepspeed

As discussed for BingBertSQuAD fine-tuning, we can simply use the `deepspeed` launcher to launch our BERT pre-training jobs as follows.

```shell
bash ds_train_bert_onebit_bsz4k_seq128.sh
```

### Launch with mpirun

Alternatively, use the following command to launch using `mpirun`.

```shell
mpirun -np [#processes] -ppn [#GPUs on each node] -hostfile [hostfile] [MPI flags] bash mpi_train_bert_onebit_bsz4k_seq128.sh
```

For example, in order to use 32 GPUs (4GPUs/node, 8 nodes in total), with the support of InfiniBand, you can use MVAPICH2 as the launcher and run the following command:
```shell
mpirun -np 32 -ppn 4 -hostfile hosts -env MV2_USE_CUDA=1 -env MV2_SUPPORT_DL=1 -env MV2_ENABLE_AFFINITY=0 -env MV2_SMP_USE_CMA=0 bash ds_train_bert_onebit_bsz4k_seq128.sh
``` -->

### 3.2 Configuration for BERT Pre-training with DeepSpeed and 1-bit Adam enabled

The `deepspeed_bsz4k_onebit_config_seq128_*.json` file gives the user the ability to specify DeepSpeed
options in terms of batch size, micro batch size, optimizer, learning rate, and other parameters.

Below is the DeepSpeed configuration file for running BERT-large pre-training with sequence length of 128 using the 1-bit Adam optimizer.

```json
{
  "train_batch_size": 4096,
  "train_micro_batch_size_per_gpu": 16,
  "steps_per_print": 100,
  "prescale_gradients": false,
  "optimizer": {
    "type": "OneBitAdam",
    "params": {
      "lr": 4e-4,
      "weight_decay": 0.01,
      "bias_correction": false,
      "freeze_step": 23000,
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
The above file is for BERT-large. For BERT-base training (sequence length 128), the suggested `freeze_step` is 16000. For sequence 512 pre-training, we suggest to use a `freeze_step` of 1500 for both BERT-base and BERT-large. And make sure to set the `comm_backend_name` and `cuda_aware` correctly as described above.

### 3.3 Performance Results for BERT Pre-training

Performance results of BERT Pre-training can be seen from our [blog post](https://www.deepspeed.ai/2020/09/08/onebit-adam-blog-post.html) and our [paper](https://arxiv.org/abs/2102.02888).
