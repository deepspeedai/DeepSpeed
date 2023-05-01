---
title: "CIFAR-10 Tutorial"
excerpt: "Train your first model with DeepSpeed!"
tags: getting-started
---

If you haven't already, we advise you to first read through the
[Getting Started](/getting-started/) guide before stepping through this
tutorial.

In this tutorial we will be adding DeepSpeed to the CIFAR-10 model, which is a small image classification model.

First we will go over how to run the original CIFAR-10 model. Then we will proceed step-by-step in enabling this model to run with DeepSpeed.



## Running Original CIFAR-10

Original model code from the [CIFAR-10 Tutorial](https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/cifar10_tutorial.py), We've copied this repo under [DeepSpeedExamples/training/cifar/](https://github.com/microsoft/DeepSpeedExamples/tree/master/training/cifar) and made it available as a submodule. To download, execute:

```bash
git submodule update --init --recursive
```

To install the requirements for the CIFAR-10 model:
```bash
cd DeepSpeedExamples/cifar
pip install -r requirements.txt
```

Run `python cifar10_tutorial.py`, it downloads the training data set at first run.
```
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
170500096it [00:02, 61124868.24it/s]
Extracting ./data/cifar-10-python.tar.gz to ./data
Files already downloaded and verified
  cat  frog  frog  frog
[1,  2000] loss: 2.170
[1,  4000] loss: 1.879
[1,  6000] loss: 1.690
[1,  8000] loss: 1.591
[1, 10000] loss: 1.545
[1, 12000] loss: 1.467
[2,  2000] loss: 1.377
[2,  4000] loss: 1.374
[2,  6000] loss: 1.363
[2,  8000] loss: 1.322
[2, 10000] loss: 1.295
[2, 12000] loss: 1.287
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:    cat  ship plane plane
Accuracy of the network on the 10000 test images: 53 %
Accuracy of plane : 69 %
Accuracy of   car : 59 %
Accuracy of  bird : 56 %
Accuracy of   cat : 36 %
Accuracy of  deer : 37 %
Accuracy of   dog : 26 %
Accuracy of  frog : 70 %
Accuracy of horse : 61 %
Accuracy of  ship : 51 %
Accuracy of truck : 63 %
cuda:0
```




## Enabling DeepSpeed


### Argument Parsing

The first step to apply DeepSpeed is adding DeepSpeed arguments to CIFAR-10 model, using `deepspeed.add_config_arguments()` function as below.

```python
 import argparse
 import deepspeed

 def add_argument():

     parser=argparse.ArgumentParser(description='CIFAR')

     # Data.
     # Cuda.
     parser.add_argument('--with_cuda', default=False, action='store_true',
                         help='use CPU in case there\'s no GPU support')
     parser.add_argument('--use_ema', default=False, action='store_true',
                         help='whether use exponential moving average')

     # Train.
     parser.add_argument('-b', '--batch_size', default=32, type=int,
                         help='mini-batch size (default: 32)')
     parser.add_argument('-e', '--epochs', default=30, type=int,
                         help='number of total epochs (default: 30)')
     parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

     # Include DeepSpeed configuration arguments.
     parser = deepspeed.add_config_arguments(parser)

     args=parser.parse_args()

     return args
```



### Initialization

We create `model_engine`, `optimizer` and `trainloader` with the help of `deepspeed.initialize`, which is defined as following:

```python
def initialize(args,
               model,
               optimizer=None,
               model_params=None,
               training_data=None,
               lr_scheduler=None,
               mpu=None,
               dist_init_required=True,
               collate_fn=None):
```

Here we initialize DeepSpeed with the CIFAR-10 model (`net`), `args`, `parameters` and `trainset`:

```python
 parameters = filter(lambda p: p.requires_grad, net.parameters())
 args=add_argument()

 # Initialize DeepSpeed to use the following features
 # 1) Distributed model.
 # 2) Distributed data loader.
 # 3) DeepSpeed optimizer.
 model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=args, model=net, model_parameters=parameters, training_data=trainset)

```

After initializing DeepSpeed, the original `device` and `optimizer` are removed:

```python
 #from deepspeed.accelerator import get_accelerator
 #device = torch.device(get_accelerator().device_name(0) if get_accelerator().is_available() else "cpu")
 #net.to(device)

 #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```



### Training API

The `model` returned by `deepspeed.initialize` is the _DeepSpeed Model Engine_ that we will use to train the model using the forward, backward and step API.

```python
     for i, data in enumerate(trainloader):
         # Get the inputs; data is a list of [inputs, labels].
         inputs = data[0].to(model_engine.device)
         labels = data[1].to(model_engine.device)

         outputs = model_engine(inputs)
         loss = criterion(outputs, labels)

         model_engine.backward(loss)
         model_engine.step()
```

Zeroing the gradients is handled automatically by DeepSpeed after the weights have been updated using a mini-batch.



### Configuration

The next step to use DeepSpeed is to create a configuration JSON file (ds_config.json). This file provides DeepSpeed specific parameters defined by the user, e.g., batch size, optimizer, scheduler and other parameters.

```json
 {
   "train_batch_size": 4,
   "steps_per_print": 2000,
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
   },
   "scheduler": {
     "type": "WarmupLR",
     "params": {
       "warmup_min_lr": 0,
       "warmup_max_lr": 0.001,
       "warmup_num_steps": 1000
     }
   },
   "wall_clock_breakdown": false
 }
```



### Run CIFAR-10 Model with DeepSpeed Enabled

To start training the CIFAR-10 model with DeepSpeed applied, execute the following command, it will use all detected GPUs by default.

```bash
deepspeed cifar10_deepspeed.py --deepspeed_config ds_config.json
```

DeepSpeed usually prints more training details for the user to monitor, including training settings, performance statistics and loss trends.
```
deepspeed.pt cifar10_deepspeed.py --deepspeed_config ds_config.json
Warning: Permanently added '[192.168.0.22]:42227' (ECDSA) to the list of known hosts.
cmd=['pdsh', '-w', 'worker-0', 'export NCCL_VERSION=2.4.2; ', 'cd /data/users/deepscale/test/ds_v2/examples/cifar;', '/usr/bin/python', '-u', '-m', 'deepspeed.pt.deepspeed_launch', '--world_info=eyJ3b3JrZXItMCI6IFswXX0=', '--node_rank=%n', '--master_addr=192.168.0.22', '--master_port=29500', 'cifar10_deepspeed.py', '--deepspeed', '--deepspeed_config', 'ds_config.json']
worker-0: Warning: Permanently added '[192.168.0.22]:42227' (ECDSA) to the list of known hosts.
worker-0: 0 NCCL_VERSION 2.4.2
worker-0: WORLD INFO DICT: {'worker-0': [0]}
worker-0: nnodes=1, num_local_procs=1, node_rank=0
worker-0: global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0]})
worker-0: dist_world_size=1
worker-0: Setting CUDA_VISIBLE_DEVICES=0
worker-0: Files already downloaded and verified
worker-0: Files already downloaded and verified
worker-0:  bird   car horse  ship
worker-0: DeepSpeed info: version=2.1, git-hash=fa937e7, git-branch=master
worker-0: [INFO 2020-02-06 19:53:49] Set device to local rank 0 within node.
worker-0: 1 1
worker-0: [INFO 2020-02-06 19:53:56] Using DeepSpeed Optimizer param name adam as basic optimizer
worker-0: [INFO 2020-02-06 19:53:56] DeepSpeed Basic Optimizer = FusedAdam (
worker-0: Parameter Group 0
worker-0:     betas: [0.8, 0.999]
worker-0:     bias_correction: True
worker-0:     eps: 1e-08
worker-0:     lr: 0.001
worker-0:     max_grad_norm: 0.0
worker-0:     weight_decay: 3e-07
worker-0: )
worker-0: [INFO 2020-02-06 19:53:56] DeepSpeed using configured LR scheduler = WarmupLR
worker-0: [INFO 2020-02-06 19:53:56] DeepSpeed LR Scheduler = <deepspeed.pt.deepspeed_lr_schedules.WarmupLR object at 0x7f64c4c09c18>
worker-0: [INFO 2020-02-06 19:53:56] rank:0 step=0, skipped=0, lr=[0.001], mom=[[0.8, 0.999]]
worker-0: DeepSpeedLight configuration:
worker-0:   allgather_size ............... 500000000
worker-0:   allreduce_always_fp32 ........ False
worker-0:   disable_allgather ............ False
worker-0:   dump_state ................... False
worker-0:   dynamic_loss_scale_args ...... None
worker-0:   fp16_enabled ................. False
worker-0:   global_rank .................. 0
worker-0:   gradient_accumulation_steps .. 1
worker-0:   gradient_clipping ............ 0.0
worker-0:   initial_dynamic_scale ........ 4294967296
worker-0:   loss_scale ................... 0
worker-0:   optimizer_name ............... adam
worker-0:   optimizer_params ............. {'lr': 0.001, 'betas': [0.8, 0.999], 'eps': 1e-08, 'weight_decay': 3e-07}
worker-0:   prescale_gradients ........... False
worker-0:   scheduler_name ............... WarmupLR
worker-0:   scheduler_params ............. {'warmup_min_lr': 0, 'warmup_max_lr': 0.001, 'warmup_num_steps': 1000}
worker-0:   sparse_gradients_enabled ..... False
worker-0:   steps_per_print .............. 2000
worker-0:   tensorboard_enabled .......... False
worker-0:   tensorboard_job_name ......... DeepSpeedJobName
worker-0:   tensorboard_output_path ......
worker-0:   train_batch_size ............. 4
worker-0:   train_micro_batch_size_per_gpu  4
worker-0:   wall_clock_breakdown ......... False
worker-0:   world_size ................... 1
worker-0:   zero_enabled ................. False
worker-0:   json = {
worker-0:     "optimizer":{
worker-0:         "params":{
worker-0:             "betas":[
worker-0:                 0.8,
worker-0:                 0.999
worker-0:             ],
worker-0:             "eps":1e-08,
worker-0:             "lr":0.001,
worker-0:             "weight_decay":3e-07
worker-0:         },
worker-0:         "type":"Adam"
worker-0:     },
worker-0:     "scheduler":{
worker-0:         "params":{
worker-0:             "warmup_max_lr":0.001,
worker-0:             "warmup_min_lr":0,
worker-0:             "warmup_num_steps":1000
worker-0:         },
worker-0:         "type":"WarmupLR"
worker-0:     },
worker-0:     "steps_per_print":2000,
worker-0:     "train_batch_size":4,
worker-0:     "wall_clock_breakdown":false
worker-0: }
worker-0: [INFO 2020-02-06 19:53:56] 0/50, SamplesPerSec=1292.6411179579866
worker-0: [INFO 2020-02-06 19:53:56] 0/100, SamplesPerSec=1303.6726433398537
worker-0: [INFO 2020-02-06 19:53:56] 0/150, SamplesPerSec=1304.4251022567403

......

worker-0: [2, 12000] loss: 1.247
worker-0: [INFO 2020-02-06 20:35:23] 0/24550, SamplesPerSec=1284.4954513975558
worker-0: [INFO 2020-02-06 20:35:23] 0/24600, SamplesPerSec=1284.384033658866
worker-0: [INFO 2020-02-06 20:35:23] 0/24650, SamplesPerSec=1284.4433482972925
worker-0: [INFO 2020-02-06 20:35:23] 0/24700, SamplesPerSec=1284.4664449792422
worker-0: [INFO 2020-02-06 20:35:23] 0/24750, SamplesPerSec=1284.4950124403447
worker-0: [INFO 2020-02-06 20:35:23] 0/24800, SamplesPerSec=1284.4756105952233
worker-0: [INFO 2020-02-06 20:35:24] 0/24850, SamplesPerSec=1284.5251526215386
worker-0: [INFO 2020-02-06 20:35:24] 0/24900, SamplesPerSec=1284.531217073863
worker-0: [INFO 2020-02-06 20:35:24] 0/24950, SamplesPerSec=1284.5125323220368
worker-0: [INFO 2020-02-06 20:35:24] 0/25000, SamplesPerSec=1284.5698818883018
worker-0: Finished Training
worker-0: GroundTruth:    cat  ship  ship plane
worker-0: Predicted:    cat   car   car plane
worker-0: Accuracy of the network on the 10000 test images: 57 %
worker-0: Accuracy of plane : 61 %
worker-0: Accuracy of   car : 74 %
worker-0: Accuracy of  bird : 49 %
worker-0: Accuracy of   cat : 36 %
worker-0: Accuracy of  deer : 44 %
worker-0: Accuracy of   dog : 52 %
worker-0: Accuracy of  frog : 67 %
worker-0: Accuracy of horse : 58 %
worker-0: Accuracy of  ship : 70 %
worker-0: Accuracy of truck : 59 %
```
