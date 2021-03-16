---
title: "Flops Profiler"
excerpt: "Measure the parameters, latency, and floating point operations of your model"
---

In this tutorial, we introduce the DeepSpeed flops profiler and provide examples of its usage.

  - [Overview](#overview)
  - [Supported Models](#supported-models)
  - [Multi-GPU, Multi-node Runs](#multi-gpu-multi-node-runs)
  - [Usage](#usage)

## Overview

The DeepSpeed flops profiler profiles the forward pass of a PyTorch model and prints the model graph with the measured profile attached to each module.
It shows the parameters, latency, and number of floating point operations of the modules within the model to identify potential bottlenecks.
It also outputs the names of the top `k` modules in terms of aggregated time, flops, and number of parameters at depth `l` with `k` and `l` specified by the user.
The DeepSpeed flops profiler can be used with the DeepSpeed runtime or as a standalone package.

The output profile is computed for each batch of input and printed to the `stdout`. For each module, the measured profile is annotated after the name and is listed in the order of `number of parameters, percentage of total parameters, number of multiply-accumulate operations (MACs), percentage of total MACs, latency of the module, percentage of the total latency, floating point operations per second (FLOPS)`. Note that the number of floating point operations is estimated as `2 * MACs` in the profiler (each MAC operation is counted as 2 floating point operations).

Below is an example output for LeNet5 with batch size 1024:

```shell
-------------------------- DeepSpeed Flops Profiler --------------------------
Summary of forward pass:
Profile step:                   1
Number of parameters:           61.71 k
Number of multiply-accumulate operations (MACs):   439.56 M
Number of floating point operations ( = 2 * MACs):   879.12 M
Latency:                        25.7 ms
Floating point operations per second(FLOPS):   34.2 GFLOPS

----------------------------- Aggregated Profile -----------------------------
Top 3 modules in MACs at depth 2 are {'Conv2d': '421.91 MMACs', 'Linear': '11.18 MMACs', 'AvgPool2d': '6.46 MMACs'}
Top 3 modules in params at depth 2 are {'Conv2d': '50.69 k', 'Linear': '11.01 k', 'Tanh': '0'}
Top 3 modules in latency at depth 2 are {'Conv2d': '11.37 ms', 'Linear': '5.27 ms', 'AvgPool2d': '5.02 ms'}

------------------------------ Detailed Profile ------------------------------
Each module profile is listed after its name in the following order:
number of parameters, percentage of total parameters, number of multiply-accumulate operations (MACs), percentage of total MACs, latency, percentage of total latency, number of floating point operations per second (FLOPS, computed as 2 * MACs / latency).
Note:
1. A module can have torch.nn.functional (e.g. to compute logits) along with submodules, thus making the difference between the parent's MACs(or latency) and the sum of its submodules'.
2. Number of floating point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.

LeNet5(
  61.71 k, 100.00% Params, 439.56 MMACs, 100.00% MACs, 25.7 ms, 100.00% latency, 34.2 GFLOPS,
  (feature_extractor): Sequential(
    50.69 k, 82.15% Params, 428.37 MMACs, 97.45% MACs, 20.12 ms, 78.27% latency, 42.59 GFLOPS,
    (0): Conv2d(156, 0.25% Params, 125.24 MMACs, 28.49% MACs, 9.8 ms, 38.12% latency, 25.56 GFLOPS, 1, 6, kernel_size=(5, 5), stride=(1, 1))
    (1): Tanh(0, 0.00% Params, 0 MACs, 0.00% MACs, 2.85 ms, 11.08% latency, 0.0 FLOPS, )
    (2): AvgPool2d(0, 0.00% Params, 4.82 MMACs, 1.10% MACs, 4.01 ms, 15.59% latency, 2.4 GFLOPS, kernel_size=2, stride=2, padding=0)
    (3): Conv2d(2.42 k, 3.92% Params, 247.4 MMACs, 56.28% MACs, 924.83 us, 3.60% latency, 535.02 GFLOPS, 6, 16, kernel_size=(5, 5), stride=(1, 1))
    (4): Tanh(0, 0.00% Params, 0 MACs, 0.00% MACs, 672.1 us, 2.62% latency, 0.0 FLOPS, )
    (5): AvgPool2d(0, 0.00% Params, 1.64 MMACs, 0.37% MACs, 1.01 ms, 3.95% latency, 3.23 GFLOPS, kernel_size=2, stride=2, padding=0)
    (6): Conv2d(48.12 k, 77.98% Params, 49.27 MMACs, 11.21% MACs, 647.31 us, 2.52% latency, 152.25 GFLOPS, 16, 120, kernel_size=(5, 5), stride=(1, 1))
    (7): Tanh(0, 0.00% Params, 0 MACs, 0.00% MACs, 82.02 us, 0.32% latency, 0.0 FLOPS, )
  )
  (classifier): Sequential(
    11.01 k, 17.85% Params, 11.18 MMACs, 2.54% MACs, 5.41 ms, 21.06% latency, 4.13 GFLOPS,
    (0): Linear(10.16 k, 16.47% Params, 10.32 MMACs, 2.35% MACs, 2.47 ms, 9.60% latency, 8.37 GFLOPS, in_features=120, out_features=84, bias=True)
    (1): Tanh(0, 0.00% Params, 0 MACs, 0.00% MACs, 90.12 us, 0.35% latency, 0.0 FLOPS, )
    (2): Linear(850, 1.38% Params, 860.16 KMACs, 0.20% MACs, 2.8 ms, 10.91% latency, 613.62 MFLOPS, in_features=84, out_features=10, bias=True)
  )
)
------------------------------------------------------------------------------
```

## Supported Models

The flops estimation is partly inspired by [ptflops](https://github.com/sovrasov/flops-counter.pytorch) with the major difference being that the DeepSpeed flops profiler captures ```torch.nn.functional``` invoked in a module to estimate the flops. Thus the DeepSpeed flops profiler allows for customized modules in the model, e.g., ```ParallelTransformerLayerworks, ParallelSelfAttention, RowParallelLinear, etc.``` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). This is in contrast to tools that profile at ```torch.nn.module``` level, such as ptflops, which require users to write customized flops calculation functions for each customized module. Finally, the DeepSpeed flops profiler also supports flops computation at module level (for RNNs).

## Multi-GPU, Multi-node Runs

For models running on multi-GPU or multi-node, only the model parallelism (e.g. ```--model-parallel-size``` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)) affects the number of flops and parameters profiled, i.e.,
`model_parallel_size * flops = total_flops` and `model_parallel_size * parameters = total_parameters`. The number of GPUs or nodes does not affect the output profile.


## Usage

The DeepSpeed flops profiler can be used with the DeepSpeed runtime or as a standalone package. When using DeepSpeed for model training, the flops profiler can be configured in the deepspeed_config file without user code changes. To use the flops profiler outside of the DeepSpeed runtime, one can simply install DeepSpeed and import the flops_profiler package to use the APIs directly. Examples of each usage are given below.

  - [Usage With the DeepSpeed Runtime](#usage-with-the-deepspeed-runtime)
    - [Example: Megatron-LM](#example-megatron-lm)
  - [Usage Outside the DeepSpeed Runtime](#usage-outside-the-deepspeed-runtime)
    - [In Model Inference](#in-model-inference)
      - [Example: AlexNet](#example-alexnet)
      - [Example: Bert](#example-bert)
    - [In Model Training Workflow](#in-model-training-workflow)
      - [Example Training Workflow](#example-training-workflow)


### Usage With the DeepSpeed Runtime

When using DeepSpeed for model training, the flops profiler can be configured in the `deepspeed_config` file. No explicit API calls are needed to use the profiler. Refer to [flops profiler](https://www.deepspeed.ai/docs/config-json/#flops-profiler) for details.


#### Example: Megatron-LM

For information on running Megatron-LM with DeepSpeed, please refer to our tutorial [Megatron-LM](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM)

The flops profiler can be enabled by adding the following field to the `deepspeed_config` file.

```json
{
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": true,
    }
}
```

An example output of 4-layer Megatron-LM model (`hidden_size = 512, num_attention_heads = 16, batch_size = 8, seq_length = 1024`) is shown below.

```shell
-------------------------- DeepSpeed Flops Profiler --------------------------
Summary of forward pass:
Profile step:                   1
Number of parameters:           38.89 M
Number of multiply-accumulate operations (MACs):   314.61 G
Number of floating point operations ( = 2 * MACs):   629.21 G
Latency:                        33.81 ms
Floating point operations per second(FLOPS):   18.61 TFLOPS

----------------------------- Aggregated Profile -----------------------------
Top 3 modules in MACs at depth 8 are {'ColumnParallelLinear': '60.13 GMACs', 'RowParallelLinear': '42.95 GMACs', 'FusedScaleMaskSoftmax': '536.87 MMACs'}
Top 3 modules in params at depth 8 are {'ColumnParallelLinear': '7.35 M', 'RowParallelLinear': '5.25 M', 'FusedScaleMaskSoftmax': '0'}
Top 3 modules in latency at depth 8 are {'ColumnParallelLinear': '659.23 us', 'RowParallelLinear': '587.94 us', 'FusedScaleMaskSoftmax': '370.98 us'}

------------------------------ Detailed Profile ------------------------------
Each module profile is listed after its name in the following order:
number of parameters, percentage of total parameters, number of multiply-accumulate operations (MACs), percentage of total MACs, latency, percentage of total latency, number of floating point operations per second (FLOPS, computed as 2 * MACs / latency).
Note:
1. A module can have torch.nn.functional (e.g. to compute logits) along with submodules, thus making the difference between the parent's MACs(or latency) and the sum of its submodules'.
2. Number of floating point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.

DistributedDataParallel(
  38.89 M, 100.00% Params, 314.61 GMACs, 100.00% MACs, 33.81 ms, 100.00% latency, 18.61 TFLOPS,
  (module): FP16_Module(
    38.89 M, 100.00% Params, 314.61 GMACs, 100.00% MACs, 33.77 ms, 99.89% latency, 18.63 TFLOPS,
    (module): GPT2Model(
      38.89 M, 100.00% Params, 314.61 GMACs, 100.00% MACs, 33.69 ms, 99.66% latency, 18.67 TFLOPS,
      (language_model): TransformerLanguageModel(
        38.89 M, 100.00% Params, 103.62 GMACs, 32.94% MACs, 5.58 ms, 16.51% latency, 37.13 TFLOPS,
        (embedding): Embedding(
          26.28 M, 67.57% Params, 0 MACs, 0.00% MACs, 545.98 us, 1.61% latency, 0.0 FLOPS,
          (word_embeddings): VocabParallelEmbedding(25.76 M, 66.23% Params, 0 MACs, 0.00% MACs, 223.88 us, 0.66% latency, 0.0 FLOPS, )
          (position_embeddings): Embedding(524.29 k, 1.35% Params, 0 MACs, 0.00% MACs, 147.1 us, 0.44% latency, 0.0 FLOPS, 1024, 512)
          (embedding_dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 79.39 us, 0.23% latency, 0.0 FLOPS, p=0.1, inplace=False)
        )
        (transformer): ParallelTransformer(
          12.61 M, 32.43% Params, 103.62 GMACs, 32.94% MACs, 5.0 ms, 14.78% latency, 41.49 TFLOPS,
          (layers): ModuleList(
            12.61 M, 32.42% Params, 103.62 GMACs, 32.94% MACs, 4.4 ms, 13.01% latency, 47.13 TFLOPS,
            (0): ParallelTransformerLayer(
              3.15 M, 8.11% Params, 25.9 GMACs, 8.23% MACs, 1.36 ms, 4.02% latency, 38.09 TFLOPS,
              (input_layernorm): FusedLayerNorm(1.02 k, 0.00% Params, 0 MACs, 0.00% MACs, 92.51 us, 0.27% latency, 0.0 FLOPS, torch.Size([512]), eps=1e-05, elementwise_affine=True)
              (attention): ParallelSelfAttention(
                1.05 M, 2.70% Params, 8.72 GMACs, 2.77% MACs, 754.59 us, 2.23% latency, 23.12 TFLOPS,
                (query_key_value): ColumnParallelLinear(787.97 k, 2.03% Params, 6.44 GMACs, 2.05% MACs, 182.87 us, 0.54% latency, 70.46 TFLOPS, )
                (scale_mask_softmax): FusedScaleMaskSoftmax(0, 0.00% Params, 134.22 MMACs, 0.04% MACs, 120.4 us, 0.36% latency, 2.23 TFLOPS, )
                (attention_dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 47.45 us, 0.14% latency, 0.0 FLOPS, p=0.1, inplace=False)
                (dense): RowParallelLinear(262.66 k, 0.68% Params, 2.15 GMACs, 0.68% MACs, 81.78 us, 0.24% latency, 52.52 TFLOPS, )
              )
              (post_attention_layernorm): FusedLayerNorm(1.02 k, 0.00% Params, 0 MACs, 0.00% MACs, 57.22 us, 0.17% latency, 0.0 FLOPS, torch.Size([512]), eps=1e-05, elementwise_affine=True)
              (mlp): ParallelMLP(
                2.1 M, 5.40% Params, 17.18 GMACs, 5.46% MACs, 224.83 us, 0.67% latency, 152.83 TFLOPS,
                (dense_h_to_4h): ColumnParallelLinear(1.05 M, 2.70% Params, 8.59 GMACs, 2.73% MACs, 64.13 us, 0.19% latency, 267.87 TFLOPS, )
                (dense_4h_to_h): RowParallelLinear(1.05 M, 2.70% Params, 8.59 GMACs, 2.73% MACs, 90.36 us, 0.27% latency, 190.13 TFLOPS, )
              )
            )
            ...
            (3): ParallelTransformerLayer(...)
          (final_layernorm): FusedLayerNorm(1.02 k, 0.00% Params, 0 MACs, 0.00% MACs, 52.69 us, 0.16% latency, 0.0 TFLOPS, torch.Size([512]), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
)
```

###  Usage Outside the DeepSpeed Runtime

The flops profiler can be used as a standalone package outside of the DeepSpeed runtime.
One can simply install DeepSpeed and import the `flops_profiler` package to use the APIs directly.
Refer to [installation of DeepSpeed](https://www.deepspeed.ai/getting-started/#installation) for installing DeepSpeed.

#### In Model Inference

To profile a trained model in inference, use the `get_model_profile` function.
Examples are given below.

##### Example: AlexNet

The following example shows how to profile AlexNet using the DeepSpeed flops profiler.

```python
import torchvision.models as models
import torch
from deepspeed.profiling.flops_profiler import get_model_profile

with torch.cuda.device(0):
    model = models.alexnet()
    batch_size = 256
    macs, params = get_model_profile(model=model, # model
                                     input_res=(batch_size, 3, 224, 224), # input shape or input to the input_constructor
                                     input_constructor=None, # if specified, a constructor taking input_res is used as input to the model
                                     print_profile=True, # prints the model graph with the measured profile attached to each module
                                     detailed=True, # print the detailed profile
                                     module_depth=-1, # depth into the nested modules with -1 being the inner most modules
                                     top_modules=3, # the number of top modules to print aggregated profile
                                     warm_up=10, # the number of warm-ups before measuring the time of each module
                                     as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                     ignore_modules=None) # the list of modules to ignore in the profiling
```

An example output:

```shell
-------------------------- DeepSpeed Flops Profiler --------------------------
Summary of forward pass:
Profile step:                   10
Number of parameters:           61.1 M
Number of multiply-accumulate operations (MACs):   183.18 G
Number of floating point operations ( = 2 * MACs):   366.36 G
Latency:                        22.13 ms
Floating point operations per second(FLOPS):   16.56 TFLOPS

----------------------------- Aggregated Profile -----------------------------
Top 3 modules in MACs at depth 2 are {'Conv2d': '167.95 GMACs', 'Linear': '15.01 GMACs', 'ReLU': '126.26 MMACs'}
Top 3 modules in params at depth 2 are {'Linear': '58.63 M', 'Conv2d': '2.47 M', 'ReLU': '0'}
Top 3 modules in latency at depth 2 are {'Conv2d': '13.96 ms', 'Linear': '6.23 ms', 'ReLU': '730.75 us'}

------------------------------ Detailed Profile ------------------------------
Each module profile is listed after its name in the following order:
number of parameters, percentage of total parameters, number of multiply-accumulate operations (MACs), percentage of total MACs, latency, percentage of total latency, number of floating point operations per second (FLOPS, computed as 2 * MACs / latency).
Note:
1. A module can have torch.nn.functional (e.g. to compute logits) along with submodules, thus making the difference between the parent's MACs(or latency) and the sum of its submodules'.
2. Number of floating point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.

AlexNet(
  61.1 M, 100.00% Params, 183.18 GMACs, 100.00% MACs, 22.13 ms, 100.00% latency, 16.56 TFLOPS,
  (features): Sequential(
    2.47 M, 4.04% Params, 168.17 GMACs, 91.81% MACs, 15.17 ms, 68.57% latency, 22.17 TFLOPS,
    (0): Conv2d(23.3 k, 0.04% Params, 18.04 GMACs, 9.85% MACs, 633.0 us, 2.86% latency, 57.0 TFLOPS, 3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(0, 0.00% Params, 49.56 MMACs, 0.03% MACs, 163.79 us, 0.74% latency, 605.17 GFLOPS, inplace=True)
    (2): MaxPool2d(0, 0.00% Params, 49.56 MMACs, 0.03% MACs, 159.26 us, 0.72% latency, 622.38 GFLOPS, kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(307.39 k, 0.50% Params, 57.37 GMACs, 31.32% MACs, 6.15 ms, 27.81% latency, 18.64 TFLOPS, 64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(0, 0.00% Params, 35.83 MMACs, 0.02% MACs, 185.01 us, 0.84% latency, 387.34 GFLOPS, inplace=True)
    (5): MaxPool2d(0, 0.00% Params, 35.83 MMACs, 0.02% MACs, 134.23 us, 0.61% latency, 533.89 GFLOPS, kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(663.94 k, 1.09% Params, 28.72 GMACs, 15.68% MACs, 389.58 us, 1.76% latency, 147.47 TFLOPS, 192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(0, 0.00% Params, 16.61 MMACs, 0.01% MACs, 76.53 us, 0.35% latency, 434.15 GFLOPS, inplace=True)
    (8): Conv2d(884.99 k, 1.45% Params, 38.29 GMACs, 20.90% MACs, 6.38 ms, 28.82% latency, 12.01 TFLOPS, 384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(0, 0.00% Params, 11.08 MMACs, 0.01% MACs, 104.43 us, 0.47% latency, 212.12 GFLOPS, inplace=True)
    (10): Conv2d(590.08 k, 0.97% Params, 25.53 GMACs, 13.94% MACs, 405.79 us, 1.83% latency, 125.83 TFLOPS, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(0, 0.00% Params, 11.08 MMACs, 0.01% MACs, 65.57 us, 0.30% latency, 337.85 GFLOPS, inplace=True)
    (12): MaxPool2d(0, 0.00% Params, 11.08 MMACs, 0.01% MACs, 122.07 us, 0.55% latency, 181.46 GFLOPS, kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(0, 0.00% Params, 2.36 MMACs, 0.00% MACs, 259.4 us, 1.17% latency, 18.19 GFLOPS, output_size=(6, 6))
  (classifier): Sequential(
    58.63 M, 95.96% Params, 15.01 GMACs, 8.19% MACs, 6.54 ms, 29.54% latency, 4.59 TFLOPS,
    (0): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 42.68 us, 0.19% latency, 0.0 FLOPS, p=0.5, inplace=False)
    (1): Linear(37.75 M, 61.79% Params, 9.66 GMACs, 5.28% MACs, 301.36 us, 1.36% latency, 64.13 TFLOPS, in_features=9216, out_features=4096, bias=True)
    (2): ReLU(0, 0.00% Params, 1.05 MMACs, 0.00% MACs, 79.39 us, 0.36% latency, 26.41 GFLOPS, inplace=True)
    (3): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 39.58 us, 0.18% latency, 0.0 FLOPS, p=0.5, inplace=False)
    (4): Linear(16.78 M, 27.46% Params, 4.29 GMACs, 2.34% MACs, 234.37 us, 1.06% latency, 36.65 TFLOPS, in_features=4096, out_features=4096, bias=True)
    (5): ReLU(0, 0.00% Params, 1.05 MMACs, 0.00% MACs, 56.03 us, 0.25% latency, 37.43 GFLOPS, inplace=True)
    (6): Linear(4.1 M, 6.71% Params, 1.05 GMACs, 0.57% MACs, 5.69 ms, 25.72% latency, 368.42 GFLOPS, in_features=4096, out_features=1000, bias=True)
  )
)
------------------------------------------------------------------------------
```

##### Example: Bert

```python
from functools import partial
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from deepspeed.profiling.flops_profiler import get_model_profile


def bert_input_constructor(input_shape, tokenizer):
    fake_seq = ""
    for _ in range(input_shape[1] - 2):  # ignore the two special tokens [CLS] and [SEP]
      fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * input_shape[0],
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * input_shape[0])
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs


with torch.cuda.device(0):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    batch_size = 4
    seq_len = 128
    enable_profile = True
    if enable_profile:
      macs, params = get_model_profile(
          model,
          (batch_size, seq_len),
          input_constructor=partial(bert_input_constructor,
                                    tokenizer=tokenizer),
          print_profile=True,
          detailed=True,
      )
    else:
      inputs = bert_input_constructor((batch_size, seq_len), tokenizer)
      outputs = model(inputs)
```

An example output:

```
-------------------------- DeepSpeed Flops Profiler --------------------------
Summary of forward pass:
Profile step:                   1
Number of parameters:           109.48 M
Number of multiply-accumulate operations (MACs):   43.5 G
Number of floating point operations ( = 2 * MACs):   87.0 G
Latency:                        393.7 ms
Floating point operations per second(FLOPS):   220.97 GFLOPS

----------------------------- Aggregated Profile -----------------------------
Top 3 modules in MACs at depth 7 are {'Linear': '14.5 GMACs', 'Dropout': '0 MACs', 'LayerNorm': '0 MACs'}
Top 3 modules in params at depth 7 are {'Linear': '28.35 M', 'LayerNorm': '18.43 k', 'Dropout': '0'}
Top 3 modules in latency at depth 7 are {'Linear': '153.7 ms', 'LayerNorm': '4.74 ms', 'Dropout': '597.95 us'}

------------------------------ Detailed Profile ------------------------------
Each module profile is listed after its name in the following order:
number of parameters, percentage of total parameters, number of multiply-accumulate operations (MACs), percentage of total MACs, latency, percentage of total latency, number of floating point operations per second (FLOPS, computed as 2 * MACs / latency).
Note:
1. A module can have torch.nn.functional (e.g. to compute logits) along with submodules, thus making the difference between the parent's MACs(or latency) and the sum of its submodules'.
2. Number of floating point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.

BertForSequenceClassification(
  109.48 M, 100.00% Params, 43.5 GMACs, 100.00% MACs, 393.7 ms, 100.00% latency, 220.97 GFLOPS,
  (bert): BertModel(
    109.48 M, 100.00% Params, 43.5 GMACs, 100.00% MACs, 393.38 ms, 99.92% latency, 221.15 GFLOPS,
    (embeddings): BertEmbeddings(
      23.84 M, 21.77% Params, 0 MACs, 0.00% MACs, 1.79 ms, 0.45% latency, 0.0 FLOPS,
      (word_embeddings): Embedding(23.44 M, 21.41% Params, 0 MACs, 0.00% MACs, 485.18 us, 0.12% latency, 0.0 FLOPS, 30522, 768, padding_idx=0)
      (position_embeddings): Embedding(393.22 k, 0.36% Params, 0 MACs, 0.00% MACs, 111.1 us, 0.03% latency, 0.0 FLOPS, 512, 768)
      (token_type_embeddings): Embedding(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 215.53 us, 0.05% latency, 0.0 FLOPS, 2, 768)
      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 386.95 us, 0.10% latency, 0.0 FLOPS, (768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 20.27 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      85.05 M, 77.69% Params, 43.5 GMACs, 99.99% MACs, 391.03 ms, 99.32% latency, 222.47 GFLOPS,
      (layer): ModuleList(
        85.05 M, 77.69% Params, 43.5 GMACs, 99.99% MACs, 390.82 ms, 99.27% latency, 222.59 GFLOPS,
        (0): BertLayer(
          7.09 M, 6.47% Params, 3.62 GMACs, 8.33% MACs, 31.91 ms, 8.10% latency, 227.21 GFLOPS,
          (attention): BertAttention(
            2.36 M, 2.16% Params, 1.21 GMACs, 2.78% MACs, 16.39 ms, 4.16% latency, 147.47 GFLOPS,
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 906.76 MMACs, 2.08% MACs, 15.07 ms, 3.83% latency, 120.36 GFLOPS,
              (query): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.69% MACs, 3.66 ms, 0.93% latency, 164.91 GFLOPS, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.69% MACs, 3.72 ms, 0.94% latency, 162.36 GFLOPS, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.69% MACs, 4.52 ms, 1.15% latency, 133.65 GFLOPS, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 24.08 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 301.99 MMACs, 0.69% MACs, 1.29 ms, 0.33% latency, 469.21 GFLOPS,
              (dense): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.69% MACs, 504.26 us, 0.13% latency, 1.2 TFLOPS, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 437.97 us, 0.11% latency, 0.0 FLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 21.93 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 1.21 GMACs, 2.78% MACs, 9.57 ms, 2.43% latency, 252.35 GFLOPS,
            (dense): Linear(2.36 M, 2.16% Params, 1.21 GMACs, 2.78% MACs, 8.75 ms, 2.22% latency, 276.11 GFLOPS, in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 1.21 GMACs, 2.78% MACs, 5.77 ms, 1.47% latency, 418.39 GFLOPS,
            (dense): Linear(2.36 M, 2.16% Params, 1.21 GMACs, 2.78% MACs, 5.13 ms, 1.30% latency, 471.15 GFLOPS, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 310.9 us, 0.08% latency, 0.0 FLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 29.8 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
        )
        ...
        (11): BertLayer(...)
      )
    )
    (pooler): BertPooler(
      590.59 k, 0.54% Params, 2.36 MMACs, 0.01% MACs, 337.12 us, 0.09% latency, 14.0 GFLOPS,
      (dense): Linear(590.59 k, 0.54% Params, 2.36 MMACs, 0.01% MACs, 173.57 us, 0.04% latency, 27.19 GFLOPS, in_features=768, out_features=768, bias=True)
      (activation): Tanh(0, 0.00% Params, 0 MACs, 0.00% MACs, 46.01 us, 0.01% latency, 0.0 FLOPS, )
    )
  )
  (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 19.55 us, 0.00% latency, 0.0 FLOPS, p=0.1, inplace=False)
  (classifier): Linear(1.54 k, 0.00% Params, 6.14 KMACs, 0.00% MACs, 56.51 us, 0.01% latency, 217.47 MFLOPS, in_features=768, out_features=2, bias=True)
)
------------------------------------------------------------------------------
```

#### In Model Training Workflow

To profile model forward in a training workflow, use the `FlopsProfiler`class.
The `FlopsProfiler`class provides the follwing methods:
  * `start_profile()` - starts profiling
  * `get_total_flops(as_string=False)` - returns the total number of MACs in the model
  * `get_total_params(as_string=False)` - returns the total number of parameters in the model
  * `print_model_profile(profile_step=1, module_depth=-1, top_modules=3, detailed=True)` - prints the model profile
  * `end_profile()` - ends profiling and cleans up. This should be invoked at the end of the profiling and AFTER `get_total_flops`, `get_total_params` or `print_model_profile`.

##### Example Training Workflow

Below is an example of this usage in a typical training workflow. Note that the flops profiler only captures the forward pass in a training step. The flops of a backward pass can be roughly estimated from that of the forward pass (~2x).

```python
from deepspeed.profiling.flops_profiler import FlopsProfiler

model = Model()
prof = FlopsProfiler(model)

profile_step = 5
print_profile= True

for step, batch in enumerate(data_loader):
  # start profiling at training step "profile_step"
  if step == profile_step:
    prof.start_profile()

  # forward() method
  loss = model(batch)

  # end profiling and print output
  if step == profile_step: # if using multi nodes, check global_rank == 0 as well
    flops = prof.get_total_flops(as_string=True)
    params = prof.get_total_params(as_string=True)
    if print_profile:
        prof.print_model_profile(profile_step=profile_step)
    prof.end_profile()

  # runs backpropagation
  loss.backward()

  # weight update
  optimizer.step()
```
