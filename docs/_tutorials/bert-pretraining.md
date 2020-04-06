---
title: "Pre-training Bing BERT"
excerpt: ""
---

In this tutorial we will apply DeepSpeed to pre-train the BERT
(**B**idirectional **E**ncoder **R**epresentations from **T**ransformers),
which is widely used for many Natural Language Processing (NLP) tasks. The
details of BERT can be found here: [BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).


We will go through how to setup the data pipeline and how to run the original
BERT model. Then we will show step-by-step how to modify the model to
leverage DeepSpeed. Finally, we demonstrate the performance evaluation and
memory usage reduction from using DeepSpeed.

## Pre-training Bing BERT without DeepSpeed

We work from adaptations of
[huggingface/transformers](https://github.com/huggingface/transformers) and
[NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples).
We have forked this repo under
[DeepSpeedExamples/bing_bert](https://github.com/microsoft/DeepSpeedExamples/tree/master/bing_bert)
and made several modifications in their script:
  * We adopted the modeling code from NVIDIA's BERT under `bing_bert/nvidia/`.
  * We extended the data pipeline from [Project Turing](https://msturing.org/)
    under `bing_bert/turing/`.


### Training Data Setup

**Note:** *Downloading and pre-processing instructions are coming soon.*

Download the Wikipedia and BookCorpus datasets and specify their paths in the
model config file `DeepSpeedExamples/bing_bert/bert_large_adam_seq128.json`:

```json
{
  ...
  "datasets": {
      "wiki_pretrain_dataset": "/data/bert/bnorick_format/128/wiki_pretrain",
      "bc_pretrain_dataset": "/data/bert/bnorick_format/128/bookcorpus_pretrain"
  },
  ...
}
```


### Running the Bing BERT model

From `DeepSpeedExamples/bing_bert`, run:

```bash
python train.py  \
    --cf bert_large_adam_seq128.json \
    --train_batch_size 64 \
    --max_seq_length 128 \
    --gradient_accumulation_steps 1  \
    --max_grad_norm 1.0 \
    --fp16 \
    --loss_scale 0 \
    --delay_allreduce \
    --max_steps 10 \
    --output_dir <path-to-model-output>
```


## Enabling DeepSpeed

To use DeepSpeed we need to edit two files :

* `train.py`: Main entry point for training
* `utils.py`: Training parameters and checkpoints saving/loading utilities


### Argument Parsing

We first need to add DeepSpeed's argument parsing to `train.py`
using `deepspeed.add_config_arguments()`. This step allows the application to
recognize DeepSpeed specific configurations.

```python
def get_arguments():
    parser = get_argument_parser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args
```


### Initialization and Training

We modify the `train.py` to enable training with DeepSpeed.

#### Initialization

We use `deepspeed.initialize()` to create the model, optimizer, and learning
rate scheduler. For the Bing BERT model, we initialize DeepSpeed in its
`prepare_model_optimizer()` function as below, to pass the raw model and
optimizer (specified from the command option).
```python
def prepare_model_optimizer(args):
    # Loading Model
    model = BertMultiTask(args)

    # Optimizer parameters
    optimizer_parameters = prepare_optimizer_parameters(args, model)
    model.network, optimizer, _, _ = deepspeed.initialize(args=args,
                                         model=model.network,                                                                      model_parameters=optimizer_parameters,
                                         dist_init_required=False)
    return model, optimizer
```
Note that for Bing BERT, the raw model is kept in `model.network`, so we pass
`model.network` as a parameter instead of just model.

#### Training

The `model` returned by `deepspeed.initialize` is the DeepSpeed _model
engine_ that we will use to train the model using the forward, backward and
step API. Since the model engine exposes the same forward pass API as
`nn.Module` objects, there is no change in the forward pass.
Thus, we only modify the the backward pass and optimizer/scheduler steps.

Backward propagation is performed by calling `backward(loss)` directly with
the model engine.
```python
# Compute loss
if args.deepspeed:
    model.network.backward(loss)
else:
    if args.fp16:
        optimizer.backward(loss)
    else:
        loss.backward()
```

The `step()` function in DeepSpeed engine updates the model parameters as
well as the learning rate. Zeroing the gradients is handled automatically by
DeepSpeed after the weights have been updated after each step.
```python
if args.deepspeed:
    model.network.step()
else:
    optimizer.step()
    optimizer.zero_grad()
```

### Checkpoints Saving & Loading
DeepSpeed's model engine has flexible APIs for checkpoint saving and loading
in order to handle the both the client model state and its own internal
state.

```python
def save_checkpoint(self, save_dir, tag, client_state={})
def load_checkpoint(self, load_dir, tag)
```

In `train.py`, we use DeepSpeed's checkpointing API in the
`checkpoint_model()` function as below, where we collect the client model
states and pass them to the model engine by calling `save_checkpoint()`:
```python
def checkpoint_model(PATH, ckpt_id, model, epoch, last_global_step, last_global_data_samples, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {'epoch': epoch,
                             'last_global_step': last_global_step,
                             'last_global_data_samples': last_global_data_samples}
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.network.save_checkpoint(PATH, ckpt_id, checkpoint_state_dict)

    return
```

In the `load_training_checkpoint()` function, we use DeepSpeed's loading
checkpoint API and return the states for the client model:
```python
def load_training_checkpoint(args, model, PATH, ckpt_id):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """

    _, checkpoint_state_dict = model.network.load_checkpoint(PATH, ckpt_id)

    epoch = checkpoint_state_dict['epoch']
    last_global_step = checkpoint_state_dict['last_global_step']
    last_global_data_samples = checkpoint_state_dict['last_global_data_samples']
    del checkpoint_state_dict
    return (epoch, last_global_step, last_global_data_samples)
```



### DeepSpeed JSON Config File

The last step to use DeepSpeed is to create a configuration JSON file (e.g.,
`deepspeed_bsz4096_adam_config.json`). This file provides DeepSpeed specific
parameters defined by the user, e.g., batch size per GPU, optimizer and its
parameters, and whether enabling training with FP16.

```json
{
  "train_batch_size": 4096,
  "train_micro_batch_size_per_gpu": 64,
  "steps_per_print": 1000,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 2e-4,
      "max_grad_norm": 1.0,
      "weight_decay": 0.01,
      "bias_correction": false
    }
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  }
}
```

In particular, this sample json is specifying the following configuration parameters to DeepSpeed:

1. `train_batch_size`: use effective batch size of 4096
2. `train_micro_batch_size_per_gpu`: each GPU has enough memory to fit batch size of 64 instantaneously
3. `optimizer`: use Adam training optimizer
4. `fp16`: enable FP16 mixed precision training with an initial loss scale factor 2^16.

That's it! That's all you need do in order to use DeepSpeed in terms of
modifications. We have included a modified `train.py` file called
`DeepSpeedExamples/bing_bert/deepspeed_train.py` with all of the changes
applied.


### Start Training
An example of launching `deepspeed_train.py` on four nodes with four GPUs each would be:
```bash
deepspeed --num_nodes 4  \
    deepspeed_train.py \
    --deepspeed \
    --deepspeed_config  deepspeed_bsz4096_adam_config.json
    --cf /path-to-deepspeed/examples/tests/bing_bert/bert_large_adam_seq128.json \
    --train_batch_size 4096  \
    --max_seq_length 128 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 1.0 \
    --fp16 \
    --loss_scale 0 \
    --delay_allreduce \
    --max_steps 32 \
    --print_steps 1 \
    --output_dir <output_directory>
```
See the [Getting Started](/getting-started/) guide for more information on
launching DeepSpeed.

------

## Reproducing BERT Training Results with DeepSpeed

Our BERT training result is competitive across the industry in terms of
achieving F1 score of 90.5 or better on the SQUAD 1.1 dev set:

- Comparing with the original BERT training time from Google, it took them
about 96 hours to reach parity on 64 TPU2 chips, while it took us 14 hours on
4 DGX-2 nodes of 64 V100 GPUs.
- On 256 GPUs, it took us 3.7 hours, faster than state-of-art result (3.9
hours) from Nvidia using their superpod on the same number of GPUs
([link](https://devblogs.nvidia.com/training-bert-with-gpus/)).

![BERT Training Time](/assets/images/bert-large-training-time.png){: .align-center}

Our configuration for the BERT training result above can be reproduced with
the scripts/json configs in our DeepSpeed repo. Below is a table containing a
summary of the configurations. Specifically see the
`ds_train_bert_bsz16k_seq128.sh` and `ds_train_bert_bsz16k_seq512.sh` scripts
for more details in
[DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/tree/master/bing_bert).


| Parameters               | 128 Sequence              | 512 Sequence              |
| ------------------------ | ------------------------- | ------------------------- |
| Total batch size         | 16K                       | 16K                       |
| Train micro batch size per gpu | 64                  | 8                         |
| Optimizer                | Lamb                      | Lamb                      |
| Learning rate            | 4e-3                      | 1e-3                      |
| Min Lamb coefficient     | 0.08                      | 0.08                      |
| Max Lamb coefficient     | 0.5                       | 0.5                       |
| Learning rate scheduler  | `warmup_linear_decay_exp` | `warmup_linear_decay_exp` |
| Warmup proportion        | 0.02                      | 0.01                      |
| Decay rate               | 0.90                      | 0.70                      |
| Decay step               | 1000                      | 1000                      |
| Max Training steps       | 187000                    | 18700                     |
| Rewarm LR                | N/A                       | True                      |
| Output checkpoint number | 150                       | 162                       |
| Sample count             | 402679081                 | 34464170                  |
| Iteration count          | 24430                     | 2089                      |


## DeepSpeed Throughput Results

We have measured the throughput results of DeepSpeed using both the Adam
optimizer and LAMB optimizer. We measure the throughput by measuring the wall
clock time to process one mini-batch and dividing the mini-batch size with
the elapsed wall clock time. The table below shows that for sequence length 128,
DeepSpeed achieves 200 samples/sec throughput on a single V100 GPU, and it
obtains 53X and 57.4X speedups over the single GPU run for Adam and LAMB
respectively:

![](/assets/images/deepspeed-throughput-seq128.png){: .align-center}

![](/assets/images/deepspeed-throughput-seq512.png){: .align-center}
