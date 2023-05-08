---
title: "DeepSpeed Mixture-of-Quantization (MoQ)"
tags: training quantization
---

DeepSpeed introduces new support for model compression using quantization, called Mixture-of-Quantization (MoQ).  MoQ is designed on top of QAT (Quantization-Aware Training), with the difference that it schedules various data precisions across the training process. It starts with quantizing the model with a high precision, such as FP16 or 16-bit quantization, and reduce the precision through a pre-defined schedule until reaching the target quantization bits (like 8-bit). Moreover, we use second-order information of the model parameters to dynamically adjust the quantization schedule for each layer of the network separately. We have seen that by adding such schedule and using various data precision in the training process, we can quantize the model with better quality and preserve accuracy. For a better understanding of MoQ methodology, please refer to MoQ deep-dive, [here](https://www.deepspeed.ai/2021/05/04/MoQ.html).

Below, we use fine-tune for the GLUE tasks as an illustration of how to use MoQ.

## Prerequisites

To use MoQ for model quantization training, you should satisfy these two requirements:

1. Integrate DeepSpeed into your training script using the [Getting Started](https://www.deepspeed.ai/getting-started/) guide.
2. Add the parameters to configure your model, we will define MoQ parameters below.

MoQ quantization schedule is defined by a number of parameters which allow users to explore different configurations.

### MoQ Parameters

`enabled`: Whether to enable quantization training, default is False.

`quantize_verbose`: Whether to display verbose details, default is False.

`quantizer_kernel`: Whether to enable quantization kernel, default is False.

`quantize_type`: Quantization type, "symmetric" or "asymmetric", default is "symmetric".

`quantize_groups`: Quantization groups, which shows the number of scales used to quantize a model, default is 1.

`quantize_bits`, The number of bits to control the data-precision transition from a start-bit to the final target-bits (e.g. starting from 16-bit down to 8-bit).

    `start_bits`: The start bits in quantization training. Default is set to 16.
    `target_bits`: The target bits in quantization training. Default is set to 16.

`quantize_schedule`, This determines how to schedule the training steps at each precision level.

    `quantize_period`: indicates the period by which we reduce down the precision (number of bits) for quantization. By default, we use a period of 100 training steps, that will be doubled every time the precision reduces by 1 bit.
    `schedule_offset`: indicates when the quantization starts to happen (before this offset, we just use the normal training precision which can be either FP32/FP16). Default is set to 100 steps.

`quantize_algo`, The algorithm used to quantize the model.

    `q_type`: we currently support symmetric and asymmetric quantization that result in signed and unsigned integer values, respectively. Default is set to symmetric
    `rounding`: for the rounding of the quantized values, we can either round to the nearest value or use stochastic rounding. Default is set to nearest.

### Eigenvalue Parameters

`enabled`: Whether to enable quantization training with eigenvalue schedule, default value is set to False.

`verbose`: Whether to display verbose details of eigenvalue computation, default value is set to False.

`max_iter`: Max iteration in computing eigenvalue, default value is set to 100.

`tol`: The tolerance error in computing eigenvalue, default value is set to 1e-2.

`stability`: Variance stabilization factor, default value is set to 1e-6.

`gas_boundary_resolution`: Indicates eigenvalue computation by every N gas boundary, default value is set to 1.

`layer_name`: The model scope name pointing to all layers for eigenvalue computation, default value is set to "bert.encoder.layer".

`layer_num`: How many layers to compute eigenvalue.


## How to Use MoQ for GLUE Training Tasks

Before fine-tuning the GLUE tasks using DeepSpeed MoQ, you need:

1. Install DeepSpeed.
2. Checkout Huggingface transformers branch, install it with all required packages.

### DeepSpeed Configuration File

Prepare a config file `test.json` as below, please note the following important parameters for quantization training:

```
{
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": 2e-5,
        "weight_decay": 0.0,
        "bias_correction": true
      }
    },
    "gradient_clipping": 1.0,
    "fp16": {
      "initial_scale_power": 16,
      "enabled": true
    },
    "quantize_training": {
      "enabled": true,
      "quantize_verbose": true,
      "quantizer_kernel": true,
      "quantize-algo": {
        "q_type": "symmetric"
      },
      "quantize_bits": {
        "start_bits": 16,
        "target_bits": 8
      },
      "quantize_schedule": {
        "quantize_period": 400,
        "schedule_offset": 0
      },
      "quantize_groups": 8,
    }
}

```

### Test Script

Create a script file under `huggingface/examples` folder as below, enabling DeepSpeed using the json file prepared above.

Here we use `MRPC` task as an example.

```
TSK=mrpc
TEST_JSON=test.json

python text-classification/run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TSK \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TSK/ \
  --fp16 \
  --warmup_steps 2 \
  --deepspeed test.json
```

Running this script will get `MRPC` accuracy and F1 metric results with MoQ quantization.


### Quantization with dynamic schedule using second-order information (Eigenvalue)

Eigenvalues can be used as a proxy for layer sensitivity during training, and can be used to create a layer-wise quantization schedule. When eigenvalue calculation is enabled, DeepSpeed will compute the eigenvalues for each specified layer at the `gas_boundary_resolution` and use it to increase the `quantize_period` by up to 5x based on layer sensitivity to allow the layer enough iterations to adapt before the next precision reduction phase. The factor of 5x was chosen based on heuristics.

Please note:

1. Enabling eigenvalue will make the training much slower, it needs longer time to compute eigenvalue for each layer.
2. During fp16 training, some eigenvalues of some layers can become NaN/Inf due to limited range. For those layers, we return the max of all the non-NaN/Inf eigenvalues across all the layers. If all the eigenvalues are NaN, we return 1.0 for each of them.
3. Eigenvalues can increase the `quantize_period` by up to 5x (chosen based on heuristics). When combined with doubling of the `quantize_period` during each 1-bit precision reduction phase, this can result in very large `quantize_period` specially if the initial `quantize_period` was large to begin with. Therefore, it is important to start with a relatively small `quantize_period` when using eigenvalues to allow training to go through all the precision transition phases before the training ends.
4. Enabling eigenvalue doesn't guarantee better accuracy result, usually it needs tuning with other settings, such as `start_bits`, `quantize_period` and `quantize_groups`.

```
{
	......

    "quantize_training": {
      "enabled": true,
      "quantize_verbose": true,
      "quantizer_kernel": true,
      "quantize_type": "symmetric",
      "quantize_bits": {
        "start_bits": 12,
        "target_bits": 8
      },
      "quantize_schedule": {
        "quantize_period": 10,
        "schedule_offset": 0
      },
      "quantize_groups": 8,
      "fp16_mixed_quantize": {
        "enabled": false,
        "quantize_change_ratio": 0.001
      },
      "eigenvalue": {
        "enabled": true,
        "verbose": true,
        "max_iter": 50,
        "tol": 1e-2,
        "stability": 0,
        "gas_boundary_resolution": 1,
        "layer_name": "bert.encoder.layer",
        "layer_num": 12
      }
    }
}

```

### Finetuning Results

Here, we show the results for the GLUE tasks fine-tuning with quantization. The below table illustrates the scheduling parameters we used for each task to reach the reported accuracy. For all these experiments, we use symmetric grouped quantization with 8 groups.


|Task         |STSB	|MRPC |COLA |WNLI |SST2 |RTE  |QNLI |QQP  |MNLI |
|-------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|start-bits   |12	  |12   |12   |12   |12   |12   |12   |12   |14   |
|period       |10	  |10   |8    |8    |400  |8    |64   |18   |12   |
|Enable Eigenvalue       |False	  |True   |True    |True    |False  |True    |False   |True   |True   |

As we see in the following table, MoQ consistently preserve accuracy across different down-stream tasks.


|Task         |STSB	|MRPC |COLA |WNLI |SST2 |RTE  |QNLI |QQP  |MNLI |SQuAD|ACC+ |
|-------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|w/o QAT(FP16)|88.71|88.12|56.78|56.34|91.74|65.3 |90.96|90.67|84.04|90.56|0    |
|Basic QAT    |88.9 |88.35|52.78|55.3 |91.5 |64.2 |90.92|90.59|84.01|90.39|-0.87|
|MoQ          |88.93|89|59.33|56.34|92.09 |67.15 |90.63|90.94|84.55|90.71|0.75 |

### Tips

When using the MoQ, one needs to consider the number of samples and training iterations before setting the correct quantization period or offset to make sure that the quantization reaches the desired level of precision before training finishes.

Enabling eigenvalues for quantization dynamically adjust the quantization period on the different parts of the network. This has two positive impact: 1) the quantized network can potentially produce higher accuracy than quantizing each layer with same `quantize_period` ; 2) it automatically identifies a good quantization schedule for each layer based on its sensitivity.
