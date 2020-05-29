---
title: "BingBertSQuAD Fine-tuning"
excerpt: ""
---

In this tutorial we will be adding DeepSpeed to the BingBert model for the SQuAD fine-tuning task, called "BingBertSquad" henceforth. We will also demonstrate performance gains.

## Overview

Please clone the repository and go to the `examples/BingBertSquad` folder to follow along.

### Pre-requisites

* Download SQuAD data:
  * Training set: [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
  * Validation set: [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

You also need a pre-trained BERT model checkpoint from DeepSpeed. We will use checkpoint 162 from the
BERT pre-training [tutorial](bert-pretraining).

  * Pre-training checkpoint: `training_state_checkpoint_162.tar`

Note that the BERT model in the file `train-v1.1.json_bert-large-uncased_384_128_64` is not strictly required as it will be downloaded automatically if it is not present locally on the cluster.

### Running BingBertSquad

- **Unmodified (BaseLine):** If you would like to run unmodified BingBertSquad with the pre-processed data, there is a helper script which you can invoke via: `bash run_squad_baseline.sh 8 <PATH_TO_CHECKPOINT>/training_state_checkpoint_162.tar` where the first argument `8` is the number of GPUs and the second argument is the path to the pre-training checkpoint. This bash script sets the parameters and invokes `nvidia_run_squad_baseline.py`.
- **Modified (DeepSpeed):** This is similar to baseline;  just substitute `run_squad_baseline.sh` with `run_squad_deepspeed.sh` which invokes `nvidia_run_squad_deepspeed.py`.

## DeepSpeed Integration

The main DeepSpeed modified script is `nvidia_run_squad_deepspeed.py`; the line `import deepspeed` enables you to use DeepSpeed.

Make sure that the number of GPUs specified in the job are available (else, this will yield an out of memory error). The wrapper script `run_BingBertSquad.sh` and the test script `run_tests.sh` essentially serve to automate training - they may also be used a guidelines to set parameters and launch the fine-tuning task.

### Configuration

The `deepspeed_bsz24_config.json` file gives the user to specify DeepSpeed options in terms of batch size, learning rate, precision and other parameters. When running the `nvidia_run_squad_deepspeed.py`, in addition to the `-d` flag to enable DeepSpeed, the appropriate DeepSpeed configuration file must be specified using `--deepspeed_config <deepspeed_bsz24_config.json>`.

### Argument Parsing

The first step to apply DeepSpeed is adding arguments to BingBertSquad, using `deepspeed.add_config_arguments()` in the beginning of the main entry point as in the `main()` function in `nvidia_run_squad_deepspeed.py`. The argument passed to `add_config_arguments()` is obtained from the `get_argument_parser()` function in utils.py.

```python
parser = get_argument_parser()
# Include DeepSpeed configuration arguments
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
```

Similar to this, all the options with their corresponding description are available in `utils.py`.


### Training

#### Initialization

DeepSpeed has an initialization function to create model, optimizer and LR scheduler. For BingBertSquad, we simply augment the Baseline script with the initialize function as follows.

```python
model, optimizer, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=optimizer_grouped_parameters,
                                              dist_init_required=False)

```

Another feature of DeepSpeed is its convenient `step()` function which can be called directly as `model.step()` which hides the `fp16_optimizer` away from the user as opposed to `optimizer.step()` in the baseline code (similar to other models in this tutorial) which needs explicit handling of the case of FP16 computation.

#### Forward pass

This is identical in both Baseline and DeepSpeed, and is performed by `loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)`.

#### Backward pass

In the Baseline script you need to handle the all-reduce operation at the gradient accumulation boundary explicitly by using `enable_need_reduction()` followed by `optimizer.backward(loss)` in FP16 and `loss.backward()` in FP32. In DeepSpeed, you may simply do `model.backward(loss)`.

#### Weight updates

In the Baseline Script, you are required to explicitly specify the optimizer as `FusedAdam` (along with the handling of dynamic loss scaling) in FP16 and `BertAdam` in FP32, followed by the call `optimizer.step()` and `optimizer.zero_grad()`. DeepSpeed handles this internally (by setting the optimizer using the JSON config) when `initialize()` is called and thus you don't need to explicitly write code but just do `model.step()`.

Congratulations! Porting into DeepSpeed is complete.

### Evaluation

Once training is complete, the EM and F1 scores may be obtained from the following command:

`python evaluate-v1.1.py <PATH_TO_DEVSET>/dev-v1.1.json <PATH_TO_PREDICTIONS>/predictions.json`

### DeepSpeed Improvements

The table summarizing the results are given below. In all cases, the batch size is set to 24 and the training is conducted on 8 GPUs for 2 epochs on the DLTS RR1 DGX-2 hypercluster. A set of parameters (seeds and learning rates) were tried and the best ones were selected. All learning rates was 3e-5; Baseline seeds were 42 and DeepSpeed seeds were 10.

| Case      | Precision | EM    | F1    | Throughput |
| --------- | --------- | ----- | ----- | ---------- |
| DeepSpeed | FP16      | 84.38 | 91.11 | 9.6        |
| Baseline  | FP16      | 84.39 | 91.29 | 8.4        |
| DeepSpeed | FP32      | 84.20 | 91.06 | 3.7        |
| Baseline  | FP32      | 84.20 | 90.91 | 2.7        |

In terms of throughput (expressed in iterations processed per second), we note that DeepSpeed outperforms the baseline for the desired accuracy (in terms of EM, F1 scores).

## Fine-tuning the model pre-trained with DeepSpeed Transformer Kernels
For pre-training your model, please see [BERT Pre-Training](\bert-pretraining\) tutorial for the detailed instrucions.
If you already obtained the checkpoint of your model, use the following configuration to finetune your pretrained checkpoint.

| Parameters               | Value             |
| ------------------------ | ------------------------- |
| Total batch size         | 24                       |
| Train micro batch size per gpu | 3                  |
| Optimizer                | Adam                      |
| Learning rate            | 4e-5                      |
| Sequence-length          | 384                      |
| Weight-decay             | 0.0                      |
| Epoch count               | 2                      |

### Enabling DeepSpeed's Transformer Kernel

DeepSpeed's optimized transformer kernel must be enabled during fine-tuning
if and only if it was used also during pre-training, because the transformer
kernel has its own parameters saved in checkpoint files.

To enable the transformer kernel for higher performance, first add an argument
`--deepspeed_transformer_kernel` in `utils.py`, we can set it as `False` by
default, for easily turning on/off.

```python
 parser.add_argument('--deepspeed_transformer_kernel',
                     default=False,
                     action='store_true',
                     help='Use DeepSpeed transformer kernel to accelerate.')
```

Then in the `BertEncoder` class of the modeling source file, instantiate
transformer layers using the DeepSpeed transformer kernel as below.

```python
         if args.deepspeed_transformer_kernel:
             from deepspeed import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig, DeepSpeedConfig

             if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
                 ds_config = DeepSpeedConfig(args.deepspeed_config)
             else:
                 raise RuntimeError('deepspeed_config is not found in args.')

             cuda_config = DeepSpeedTransformerConfig(
                 batch_size = ds_config.train_micro_batch_size_per_gpu,
                 max_seq_length = args.max_seq_length,
                 hidden_size = config.hidden_size,
                 heads = config.num_attention_heads,
                 attn_dropout_ratio = config.attention_probs_dropout_prob,
                 hidden_dropout_ratio = config.hidden_dropout_prob,
                 num_hidden_layers = config.num_hidden_layers,
                 initializer_range = config.initializer_range,
                 seed = args.seed,
                 fp16 = ds_config.fp16_enabled,
                 pre_layer_norm=True)

             self.layer = nn.ModuleList([copy.deepcopy(DeepSpeedTransformerLayer(i, cuda_config)) for i in range(config.num_hidden_layers)])
         else:
             layer = BertLayer(config)
             self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

```

All configuration settings come from the DeepSpeed configuration file and
command arguments and thus we must pass the `args` variable to here in this model.

Note:

1. `batch_size` is the maximum bath size of input data, all fine-tuning training data or prediction data shouldn't exceed this threshold, otherwise it will throw an exception. In the DeepSpeed configuration file micro batch size is defined as `train_micro_batch_size_per_gpu`, e.g. if it is set as 8 and prediction uses batch size of 12, we can use 12 as transformer kernel batch size, or using "--predict_batch_size" argument to set prediction batch size to 8 or a smaller number.
2. `local_rank` in DeepSpeedTransformerConfig is used to assign the transformer kernel to the correct device. Since the model already runs set_device() before here, so does not need to be set here.

For more details about the transformer kernel, please see [DeepSpeed Transformer Kernel](/transformer_kernel/) and [DeepSpeed Fast-Bert Training](/fast_bert/).

### Dropout Setting
For the fine-tuning, we only use the deterministic transformer to have reproducible the fine-tuning results. But, we choose different values for dropout based on whether pre-training was done using deterministic or stochastic transformer (Please see [Transformer tutorial](/transformer_kernel/) for more detail of selecting these two modes).

For model pre-trained with deterministic transformer, we use the same dropout ration used in pretraining (0.1). However, we slightly increase the dropout ratio when fine-tuning the model pre-trained using the stochastic transformer to compensate for the lack of stochastic noise during fune-tuning.


| Pretraining mode               | Dropout ratio             |
| ------------------------ | ------------------------- |
| Determinstic | 0.1                      |
| Stochastic | 0.12 - 0.14                |

### Results

Fine-tuning the model pre-trained usng DeepSpeed Transformer and the recepie in [DeepSpeed Fast-Bert Training](/fast_bert/) should yield F1 score of 90.5 and is expected to increase if you let the pre-training longer than suggested in the tutorial.
