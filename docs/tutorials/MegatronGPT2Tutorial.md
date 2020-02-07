# Tutorial: Megatron-LM GPT2 with DeepSpeed
**TODO: these two links are broken (not yet implemented).**

We advise you to first read through the guides for [Setup and
Onboarding](../../Onboard/onboard/onboard.md) and [Model
Training](../../Onboard/model_training/deepspeed_model_training.md).

In this tutorial we will be adding DeepSpeed to Megatron-LM GPT2 model, which
is a large, powerful transformer. Megatron-LM supports model-parallel and multi-node
training. Please see the corresponding paper for more details: [Megatron-LM:
Training Multi-Billion Parameter Language Models Using Model
Parallelism](https://arxiv.org/abs/1909.08053).

First, we discuss data and environment setup and how to train the GPT-2 model with the
original Megatron-LM. Next, we proceed step-by-step in enabling this model to run with
DeepSpeed. Finally, we demonstrate the **_performance gains_**, and **_memory footprint
reduction_** from using DeepSpeed.

## 1 Training GPT-2 with the Original Megatron-LM

The original model code from
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM).  We've copied this repo
under
[DeepSpeedExamples/Megatron-LM/](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM)
and made it available as a submodule. To download, execute:
```bash
git submodule update --init --recursive
```

### 1.1 Training Data Setup
* Follow Megatron's [instructions](https://github.com/NVIDIA/Megatron-LM#collecting-gpt2-webtext-data)
  to download the webtext data and place a symbolic link under `DeepSpeedExamples/Megatron-LM/data`:
  * (*Microsoft*:) Raw and pre-processed data has already been downloaded on
     all DLTS clusters: `/data/Megatron-LM/data/`. You can simply execute
      `ln -s /data/Megatron-LM/data DeepSpeedExamples/Megatron-LM/`.

### 1.2 Running Unmodified Megatron-LM GPT2 model

* For a single GPU run:
    - change `scripts/pretrain_gpt2.sh`, set its `--train-data` argument as `"webtext"`.
    - run `bash scripts/pretrain_gpt2.sh`

* For multiple GPUs and/or nodes run:
    - change `scripts/pretrain_gpt2_model_parallel.sh`
        - set its `--train-data` argument as `"webtext"`
        - `GPUS_PER_NODE` indicates how many GPUs per node involved in the testing
        - `NNODES` indicates how many nodes involved in the testing

    - run `bash scripts/pretrain_gpt2_model_parallel.sh`


## 2 Enabling DeepSpeed

To use DeepSpeed we will modify three files :

* `arguments.py` : Arguments configurations
* `pretrain_gpt2.py` : Main entry point for training
* `utils.py` : Checkpoints saving and loading utilities


### 2.1 Argument Parsing
The first step is to apply DeepSpeed is adding DeepSpeed arguments to
Megatron-LM GPT2 model, using `deepspeed.add_config_arguments()` in
`arguments.py`.

```python
def get_args():
    """Parse all the args."""

    parser = argparse.ArgumentParser(description='PyTorch BERT Model')
    parser = add_model_config_args(parser)
    parser = add_fp16_config_args(parser)
    parser = add_training_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_text_generate_args(parser)
    parser = add_data_args(parser)

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
```



### 2.2 Initialization and Training
We modify `pretrain.py` to enable training with DeepSpeed.

#### 2.2.1 Initialization
We use `deepspeed.initialize` to create `model_engine`, `optimizer` and LR
`scheduler`. Below is its definition:
```python
def initialize(args,
               model,
               optimizer=None,
               model_parameters=None,
               training_data=None,
               lr_scheduler=None,
               mpu=None,
               dist_init_required=True,
               collate_fn=None):
```

For the Megatron-LM GPT2 model, we initialize DeepSpeed in its
`setup_model_and_optimizer()` function as below, to pass the raw `model`,
`optimizer`, `args`, `lr_scheduler` and `mpu`.
```python
def setup_model_and_optimizer(args):
    """Setup model and optimizer."""

    model = get_model(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.deepspeed:
        import deepspeed

        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=False
       )
```


Note that when FP16 is enabled, Megatron-LM GPT2 adds a wrapper to the `Adam`
optimizer. DeepSpeed has its own FP16 Optimizer, so we need to pass the `Adam`
optimizer to DeepSpeed directly without any wrapper. We return the unwrapped
Adam optimizer from `get_optimizer()` when DeepSpeed is enabled.
```python
def get_optimizer(model, args):
    """Setup the optimizer."""

    ......

    # Use Adam.
    optimizer = Adam(param_groups,
                     lr=args.lr, weight_decay=args.weight_decay)

    if args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer
```

#### 2.2.2 Using the Training API
The `model` returned by `deepspeed.initialize` is the _DeepSpeed Model Engine_
that we will use to train the model using the forward, backward and step API.


##### Forward Propagation
The forward propagation API is compatible to PyTorch and no change is required.


##### Backward Propagation
Backward propagation is done by calling `backward(loss)` directly on the model engine.

```python
    def backward_step(optimizer, model, lm_loss, args, timers):
        """Backward step."""

        # Total loss.
        loss = lm_loss

        # Backward pass.
        if args.deepspeed:
            model.backward(loss)
        else:
            optimizer.zero_grad()
            if args.fp16:
                optimizer.backward(loss, update_master_grads=False)
            else:
                loss.backward()
```

Zeroing the gradients is handled automatically by DeepSpeed after the weights
have been updated using a mini-batch.

Furthermore, DeepSpeed addresses distributed data parallel and FP16 under the
hood, simplifying code in multiple places.

(A) DeepSpeed also performs gradient averaging automatically at the gradient
accumulation boundaries. So we skip the allreduce communication.

   ```python
        if args.deepspeed:
            # DeepSpeed backward propagation already addressed all reduce communication.
            # Reset the timer to avoid breaking timer logs below.
            timers('allreduce').reset()
        else:
            torch.distributed.all_reduce(reduced_losses.data)
            reduced_losses.data = reduced_losses.data / args.world_size
            if not USE_TORCH_DDP:
                timers('allreduce').start()
                model.allreduce_params(reduce_after=False,
                                       fp32_allreduce=args.fp32_allreduce)
                timers('allreduce').stop()

   ```

(B) We also skip updating master gradients, since DeepSpeed addresses it internally.

   ```python
        # Update master gradients.
        if not args.deepspeed:
            if args.fp16:
                optimizer.update_master_grads()

            # Clipping gradients helps prevent the exploding gradient.
            if args.clip_grad > 0:
                if not args.fp16:
                    mpu.clip_grad_norm(model.parameters(), args.clip_grad)
                else:
                    optimizer.clip_master_grads(args.clip_grad)

        return lm_loss_reduced

   ```

##### Updating the Model Parameters
The `step()` function in DeepSpeed engine updates the model parameters as well
as the learning rate.

```python
     if args.deepspeed:
         model.step()
     else:
         optimizer.step()

         # Update learning rate.
         if not (args.fp16 and optimizer.overflow):
             lr_scheduler.step()
         else:
             skipped_iter = 1

```



##### Loss Scaling
The GPT2 training script logs the loss scaling value during training. Inside,
the DeepSpeed optimizer, this value is stored as `cur_scale` instead of
`loss_scale` in Megatron's optimizer. Therefore, we appropriately replace it in
the logging string.

```python
             if args.fp16:
                 log_string += ' loss scale {:.1f} |'.format(
                     optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)

```


### 2.3 Checkpoints Saving & Loading

DeepSpeed engine has flexible APIs for checkpoint saving and loading, to handle
the states from both the client model and its own internal.

```python
def save_checkpoint(self, save_dir, tag, client_state={})
def load_checkpoint(self, load_dir, tag)
```

Applying DeepSpeed needs to update utils.py in which Megatron-LM GPT2 saves and
loads its checkpoints.

A new function `save_ds_checkpoint()` is created as below for DeepSpeed, it
collects the client model states and passes to DeepSpeed engine by calling
`save_checkpoint()` of DeepSpeed.

```python
 def save_ds_checkpoint(iteration, model, args):
     """Save a model checkpoint."""

     sd = {}
     sd['iteration'] = iteration
     # rng states.
     if not args.no_save_rng:
         sd['random_rng_state'] = random.getstate()
         sd['np_rng_state'] = np.random.get_state()
         sd['torch_rng_state'] = torch.get_rng_state()
         sd['cuda_rng_state'] = torch.cuda.get_rng_state()
         sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()

     model.save_checkpoint(args.save, iteration, client_state = sd)

```

In Megatron-LM GPT2 `save_checkpoint()` function, adds following lines to
invoke the above function for DeepSpeed.

```python
 def save_checkpoint(iteration, model, optimizer,
                     lr_scheduler, args):
     """Save a model checkpoint."""
     if args.deepspeed:
         save_ds_checkpoint(iteration, model, args)
     else:
		......

```

In `load_checkpoint()` function, use DeepSpeed loading checkpoint API as below,
and return the states for the client model.

```python
 def load_checkpoint(model, optimizer, lr_scheduler, args):
     """Load a model checkpoint."""

     iteration, release = get_checkpoint_iteration(args)

     if args.deepspeed:
         checkpoint_name, sd = model.load_checkpoint(args.load, iteration)

         if checkpoint_name is None:
             if mpu.get_data_parallel_rank() == 0:
                 print("Unable to load checkpoint.")
             return iteration
     else:
         ......

```

### 2.4 Train  scripts
Assume webtext data was prepared in previous step, to start training
Megatron-LM GPT2 model with DeepSpeed applied, execute the following command to
start training.

- Single GPU run
  - run `bash scripts/ds_pretrain_gpt2.sh`
- Multiple GPUs/Nodes run
  - run `bash scripts/ds_pretrain_gpt2_model_parallel.sh`



## 3 DeepSpeed Improvements over Megatron
DeepSpeed enables training very large models effectively via the advanced [ZeRO
optimizer](https://arxiv.org/abs/1910.02054v2). ZeRO significantly reduces the memory
footprint for training large models which means large models can be trained with i) less
model parallelism and ii) larger batch sizes. A lower model parallelism degree improves
training efficiency by increasing the granularity of the computation such as the matrix
multiplication where performance is directly related to the size of the matrices.
Furthermore, less model parallelism also results in less communication between model
parallel GPUs, which further boosts performance.  Larger batch size has a similar effect
of increasing the computational granularity as well as reducing communication, also
resulting in better performance. Therefore, using DeepSpeed with Megatron can be
significantly faster than using Megatron without DeepSpeed.

The observed performance improvements depend on several factors such as the memory per
GPU, the local GPU interconnect (i.e., PCI-E vs NVLINK vs NVSwitch), the model size,
inter node network interconnect, etc. Below, we show some of the performance improvements
from using DeepSpeed over Megatron on a 16 GPU Low Bandwidth (40 Gbps) cluster and a 400 GPU DGX-2 High Bandwidth (800 Gbps) cluster.
For details please see the [ZeRO Paper](https://arxiv.org/abs/1910.02054v2). We also
present performance improvement on a 64 GPU cluster along with detailed configuration
analysis to show where the improvements come from.

![DeepSpeed-vs-Megatron](../figures/DeepSpeed-vs-Megatron.png)

### 3.1 On Low Bandwidth GPU Cluster
The figure above shows that training 1.5B parameter model with DeepSpeed is
nearly 4x faster than without DeepSpeed on a cluster with 4 nodes, 4 GPU per
node, and 16 GPUs total. These GPUs have 16GB of memory each, and PCI-E
interconnects GPUs within a node, and 40 Gbps infiniband across nodes.

The performance improvement comes from lower model parallelism degree and
larger batch size as discussed earlier. Training 1.5B parameter model with
Megatron alone requires 4-way model parallelism, and can only fit an effective
batch size of 32 using all 16 GPUs. On the other hand, DeepSpeed does not
require any model-parallelism to train this model, and can support and
effective batch size of 128 without running out of memory, resulting in
significantly higher performance.


### 3.2 On High bandwidth DGX-2 GPU Cluster
Each GPU on the DGX-2 cluster has 32 GB of memory, and GPUs inside a box is connected via
the high-bandwidth NVSwitch. DGX-2 nodes are connected to each other via 800 Gbps (8 x 100Gbps) infiniband interconnect. As such, running a 1.5B model on DGX-2 requires less model
parallelism, and the performance improvement from DeepSpeed for this model size is not
significant. However, at larger model sizes, Megatron still requires significantly larger
model parallelism degree, and can only run much smaller batch sizes than DeepSpeed.
Therefore, as the model sizes get larger, DeepSpeed starts to significantly outperform
Megatron.


### 3.3 Performance Improvements with Configuration Details
The figure below compares DeepSpeed with Megatron on a 64 GPU cluster with 4
DGX-2 nodes. To give the readers a clear idea of source of the performance
improvements, we also present the configuration table for both Megatron and
DeepSpeed. It shows the smallest model parallelism degree and the largest batch
size that can be used to train these models without running out of memory. As
discussed above, the tables demonstrate that DeepSpeed can run with smaller
achieve better performance than Megatron.

![DeepSpeed Performance SpeedUp](../figures/megatron-gpt2-perf-test.png)

**a ) Megatron-LM GPT2 Baseline**

|      | Model Parallelism | Data Parallelism | #gpus | batch size | layers | hidden size | attention heads | samples / sec |
| ---- | ----------------: | ---------------: | ----: | ---------: | -----: | -----------:| --------------: | ------------: |
| 1.5B | 2                 | 32               | 64    | 512        | 48     | 1600        | 16              | 128.56        |
| 4B   | 4                 | 16               | 64    | 128        | 64     | 2304        | 16              | 49.36         |
| 8B   | 4                 | 16               | 64    | 128        | 72     | 3072        | 24              | 24.57         |
| 20B  | 16                | 4                | 64    | 16         | 111    | 3808        | 32              | 3.42          |



**b ) Megatron-LM GPT2 with DeepSpeed**

|      | Model Parallelism | Data Parallelism | #gpus | batch size | layers | hidden size | attention heads | samples / sec |
| ---- | ----------------: | ---------------: | ----: | ---------: | -----: | -----------:| --------------: | ------------: |
| 1.5B | 1                 | 64               | 64    | 2048       | 48     | 1600        | 16              | 151.35        |
| 4B   | 1                 | 64               | 64    | 512        | 64     | 2304        | 16              | 75.13         |
| 8B   | 2                 | 32               | 64    | 512        | 72     | 3072        | 24              | 43.52         |
| 20B  | 4                 | 16               | 64    | 128        | 111    | 3808        | 32              | 12.65         |
