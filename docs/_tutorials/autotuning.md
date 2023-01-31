---
title: "Autotuning"
excerpt: "Automatically discover the optimal DeepSpeed configuration that delivers good training speed"
tags: training performance-tuning
---

Make sure you've read the DeepSpeed tutorials on [Getting Started](https://www.deepspeed.ai/getting-started/) and [Zero Redundancy Optimizer](https://www.deepspeed.ai/tutorials/zero/) before stepping through this tutorial.

One pain point in model training is to figure out good performance-relevant configurations such as micro-batch size to fully utilize the hardware and achieve a high throughput number. This configuration exploring process is commonly done manually but is important since model training is repeated many times and benefits from using a good configuration. Not only is the hand-tuning process time-consuming, but the outcome is hardware-dependent. This means that a good configuration on one hardware might not be the best on another different hardware. The user thus has to hand tune the configuration again. With DeepSpeed, there are more configuration parameters that could potentially affect the training speed, thus making it more tedious to manually tune the configuration.

The DeepSpeed Autotuner mitigates this pain point and automatically discovers the optimal DeepSpeed configuration that delivers good training speed. It not only reduces the time and resources users spend on tuning, but also can discover configurations better than hand-tuned methods. In this tutorial, we showcase the usage and benefits of the autotuning feature in DeepSpeed. For more details, please see the [README.md](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning).

## Tuning scope and strategy

The DeepSpeed Autotuner uses model information, system information, and heuristics to efficiently tune system knobs that affect compute and memory efficiencies, such as ZeRO optimization stages, micro-batch sizes, and many other ZeRO optimization configurations.
Currently, the DeepSpeed Autotuner tunes ZeRO stages, micro-batch size per GPU, and ZeRO configurations (offloading is not yet supported) on top of other configurations such as optimizer, scheduler, fp16 defined by the user in the DeepSpeed configuration file.
Note that ZeRO stages, micro-batch sizes, and other ZeRO configurations to tune are also configurable and can be overwritten by the user through the DeepSpeed configuration file. See [Configuring Tuning Scope](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning#configuring-tuning-scope) for details.


## Ease of use

DeepSpeed Autotuning is easy to use, requiring no code change from DeepSpeed users.
Compared to the original training script (`deepspeed your_program.py <normal cl args> --deepspeed ds_config.json`), invoking the autotuning feature in DeepSpeed only requires setting an `autotuning` flag after the DeepSpeed launcher (see [Usage](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning#usage) for details), and adding `" autotuning": {"enabled": true}` to the DeepSpeed configuration file. Users can further tailor the autotuning process by changing the autotuning configuration in the DeepSpeed configuration JSON file (See [Autotuning Configuration](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning#autotuning-configuration) for details).

## Example

We demonstrate the usage and benefit of autotuning using the training of a 0.77 billion parameter [GPT2-large model](https://huggingface.co/gpt2-large) from Hugging Face on 16 Nvidia V100 GPUs. For more examples, refer to [autotuning](https://github.com/microsoft/DeepSpeedExamples/tree/master/autotuning) in the DeepSpeedExamples repo. Note that autotuning works with any DeepSpeed-accelerated model training, not limited to Hugging Face models.

The model has:

- 36-layer
- 1280 hidden dimension
- 20 attention heads
- 774M parameters.

### Environment

The training use fp16 and runs on 1 node with 16 Nvidia V100 GPUs. The autotuning uses the same hardware resource as the training. `max_train_batch_size` is not defined. The HF packages below are used.

HF examples require installing the `transformers` package from source:
```bash
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install .
```
The `datasets` package can be installed by `pip install datasets`

Below are the versions used in this test.

- transformers (4.12.0.dev0)
- datasets (1.11.0)

### Enabling Autotuning

To enable the autotuning, add `--autotuning run` is added to the training script and add `"autotuning": {"enabled": true}` to the DeepSpeed configuration file. If the user training script uses DeepSpeed configuration parameters as training script arguments, the name mappings between the parameters in DeepSpeed configuration and the training script arguments must be provided in the `arg_mappings` dictionary in the `autotuning` section of the DeepSpeed configuration file.

Train script:
```bash
    deepspeed --autotuning run --num_nodes=$NNODES --num_gpus=$NGPUS $HF_PATH/transformers/examples/pytorch/language-modeling/run_clm.py --deepspeed $DS_CONFIG\
    --model_name_or_path $MODEL_NAME \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --fp16 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate 2e-5 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir
```

DeepSpeed configuration file:
```json
{
  "train_micro_batch_size_per_gpu": "auto",
  "fp16": {
    "enabled": true
  },
  "autotuning": {
    "enabled": true,
    "arg_mappings": {
      "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
      "gradient_accumulation_steps ": "--gradient_accumulation_steps"
    }
  }
}
```

### Throughput Comparison

The table below shows the throughput (samples per second) comparison. The corresponding micro-batch size per GPU (mbs or tmbspg) and ZeRO stage used to achieve the throughput value is also shown in the parentheses. Assume the strategy users would use in the hand-tuning process is to start from `mbs = 1` and increase mbs by 2 each time until running out of GPU memory.
 - `baseline` is the vanilla Hugging Face (HF) without DeepSpeed (DS) and mbs is hand-tuned.
 - `HF + DS hand-tuned` is HF with DS, and mbs is hand-tuned while other DS configuration uses default values.
 - `HF + DS autotuning` is HF with DS, and the DS configuration selected from autotuning.

Notation: Hugging Face (HF), DeepSpeed (DS), ZeRO stage (z), gradient accumulation steps (gas), micro-batch size per GPU (mbs or tmbspg).

| Model name | baseline (vanilla HF) | HF + DS hand-tuned       | HF + DS autotuning (fast-mode) |
| ---------- | -------------------- | ------------------------ | ------------------------------ |
| GPT2-large | 27.874 (mbs = 1)     | 56.797 (z = 1, mbs = 2), | 69.061 (z = 1, mbs = 3)        |

The detailed `HF + DS autotuning` result summary is shown below.

Note that the performance metric used in autotuning is calculated using the timings captured within DeepSpeed forward, backward and step functions. The sum of these timings is less than the actual training step latency, thus the throughput metric values used by autotuning would be higher than the end-to-end throughput in training.

- Fast-mode Autotuning time: 27 mins
- Number of experiments: 13
- Throughput Improvement over baseline: 2.48x

| tuning_space | num_experiments | best_metric_val | best_exp_name   |
| :----------- | --------------: | --------------: | :-------------- |
| z0           |               4 |         59.0229 | z0_gas1_tmbspg2 |
| z1           |               5 |         87.3017 | z1_gas1_tmbspg3 |
| z2           |               3 |         77.8338 | z2_gas1_tmbspg3 |
| z3           |               1 |               0 | z3_gas1_tmbspg3 |
| global       |              13 |         87.3017 | z1_gas1_tmbspg3 |

Tuning completed in 0:27:33.988447. Total number of experiments: 13.

As we can see the DeepSpeed Autotuner can select a better than hand-tuned configuration with a reasonable number of experiments. Examples in [Autotuning Hugging Face Examples](https://github.com/microsoft/DeepSpeedExamples/tree/master/autotuning/hf#autotuning-hugging-face-examples) would demonstrate the effectiveness of autotuning across different models.

### DeepSpeed Autotuning with AzureML

To try DeepSpeed autotuning with AzureML, please see the example [here](https://github.com/Azure/azureml-examples/tree/main/cli/jobs/deepspeed/deepspeed-autotuning).
