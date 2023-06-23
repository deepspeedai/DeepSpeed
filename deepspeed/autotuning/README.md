# DeepSpeed Autotuning
## Overview

One pain point in model training is to figure out good performance-relevant configurations such as micro-batch size to fully utilize the hardware and achieve a high throughput number. This configuration exploring process is commonly done manually but is important since model training is repeated many times and benefits from using a good configuration. Not only is the hand-tuning process time-consuming, but the outcome is hardware-dependent. This means that a good configuration on one hardware might not be the best on another different hardware. The user thus has to hand tune the configuration again. With DeepSpeed, there are more configuration parameters that could potentially affect the training speed, thus making it more tedious to manually tune the configuration.

The DeepSpeed Autotuner mitigates this pain point and automatically discovers the optimal DeepSpeed configuration that delivers good training speed.
The Autotuner uses model information, system information, and heuristics to efficiently tune system knobs that affect compute and memory efficiencies, such as ZeRO optimization stages, micro-batch sizes, and many other ZeRO optimization configurations.
It not only reduces the time and resources users spend on tuning, but also can discover configurations better than hand-tuned methods.

DeepSpeed Autotuning is easy to use, requiring no code change from DeepSpeed users.
Compared to the original training script (`deepspeed your_program.py <normal cl args> --deepspeed ds_config.json`), invoking the autotuning feature in DeepSpeed only requires setting an `autotuning` flag after the DeepSpeed launcher (see [Usage](#usage) for details), and adding `"autotuning": {"enabled": true}` to the DeepSpeed configuration file. Users can further tailor the autotuning process by changing the autotuning configuration in the DeepSpeed configuration JSON file (See [Autotuning Configuration](#autotuning-configuration) for details).

## Usage

To use DeepSpeed Autotuner, you need to do two things:

1. Add `"autotuning": {"enabled": true}` to the DeepSpeed configuration file. If the user training script uses DeepSpeed configuration parameters as command-line arguments, the name mappings between the parameters in DeepSpeed configuration and the training script arguments must be provided in the `arg_mappings` dictionary in the `autotuning` section of the DeepSpeed configuration file.
Common train scripts have micro-batch size per GPU as an argument, this mapping between the flag name and `train_micro_batch_size_per_gpu` must be provided. Below is the an example where the training script takes `--per_device_train_batch_size` as micro-batch size. Note that `--` is needed.

```json
{
    "autotuning": {
        "enabled": true,
        "arg_mappings": {
            "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
        }
    }
```

2. Specifying `--autotuning=[run|tune]` in the command line, shown as below.
```bash
deepspeed --autotuning=[run|tune] <user script> --deepspeed ds_config.json <other user args>
```

`--autotuning=run` finds the optimal DeepSpeed configuration and then launches the training with that configuration. If you want to just find the optimal configuration without running the training script, then set `--autotuning` to `tune`.


If users specify the [resource configuration](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) using the flags `--num_gpus` and `--num_nodes`, then the command becomes:

```bash
deepspeed --autotuning=[run|tune] --num_gpus=$NUM_GPUS --num_nodes=$NUM_NODES <user script> --deepspeed ds_config.json <other user args>
```

 Below shows an example where `train_micro_batch_size_per_gpu` and `gradient_accumulation_steps` are mapped to `--per_device_train_batch_size` and `--gradient_accumulation_steps` as training arguments.

Example script (some details omitted):

```bash
  deepspeed --autotuning run --num_nodes=1 --num_gpus=8 $HF_PATH/transformers/examples/pytorch/language-modeling/run_clm.py \
  --deepspeed $DS_CONFIG_PATH \
  --model_name_or_path gpt2 \
  --do_train \
  --do_eval \
  --fp16 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  ...
```

DeepSpeed configuration file:

```json
{
    "autotuning": {
        "enabled": true,
        "arg_mappings": {
            "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
            "gradient_accumulation_steps ": "--gradient_accumulation_steps"
        }
    }
```


By default, the Autotuner would only tune ZeRO optimization stages and micro-batch sizes per GPU (`fast` mode). This saves the autotuning time for a close-to-optimal tuning result. If you would like to tune other ZeRO optimization configurations,set `"fast"` to `false` in the [autotuning configuration](#autotuning-configuration).


## Autotuning Workflow and Scope

Currently, the DeepSpeed Autotuner tunes ZeRO stages, micro-batch size per GPU, and ZeRO configurations (offloading is not yet supported) on top of other configurations such as optimizer, scheduler, fp16 defined by the user in the DeepSpeed configuration file. A high-level workflow is described below:
  1. At the beginning of the autotuning process, the Autotuner launches a model information profiling experiment to get the number of model parameters and amount of activation memory.
  2. Then the Autotuner explores ZeRO stages in the order of `[0, 1, 2, 3]`. For each ZeRO stage, the Autotuner estimates the minimal memory required per GPU to train the model, and compares it with the available GPU memory. A less-than indicates that the model might be runnable with the given ZeRO stage, and the Autotuner then tunes the micro-batch size per GPU and other ZeRO configurations for that ZeRO stage.
     1. The Autotuner first tunes the micro-batch size per GPU along with gradient accumulation steps (users can specify the maximum global train batch size for the model), and selects a list of micro-batch sizes to explore next.
     2. Each ZeRO stage has a carefully-chosen default tuning space to explore for the other [ZeRO configurations](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training). Users can overwrite it through the DeepSpeed configuration file.
     3. Combinations of different micro-batch sizes and ZeRO configurations are then explored as experiments by the Autotuner using a supported algorithm (e.g., xgboost model-based algorithm). Early termination in this exploration is set by heuristics and is configurable by the user.
     4. An optimal configuration based on a metric (throughput, latency, or FLOPS) is returned for that ZeRO stage.
  3. The exploration of different ZeRO stages would stop if the optimal setup for the current ZeRO stage is no better than that of the previous ZeRO stage tuned (other heuristics are used as well for determining the termination).
  4. In the end, the global optimal setup is returned to the user. If the value of the `--autotuning` flag is set to `run`, the Autotuner launches the training with the found optimal setup.

Note that ZeRO stages, micro-batch sizes, and other ZeRO configurations to tune are also configurable and can be overwritten by the user through the DeepSpeed configuration file. See [Configuring Tuning Scope](#configuring-tuning-scope) for details.


## Configuring Tuning Scope

The DeepSpeed Autotuner tunes ZeRO stages, micro-batch size per GPU, and ZeRO configurations. Other DeepSpeed configurations are used as defined by the user in the DeepSpeed configuration file. Users can overwrite any of the tuning parameters.
### Configuring ZeRO Stage

By default, the DeepSpeed Autotuner does not tune ZeRO stages. If `"zero_optimization"` is not defined, DeepSpeed ZeRO is disabled. If `"zero_optimization"` is set to `"all"`, the Autotuner explores ZeRO stages in the order of `[0, 1, 2, 3]`. Users can overwrite this behavior if they already know what ZeRO stage(s) to use. For example, the below section in the DeepSpeed configuration file limits the Autotuner to only exploring ZeRO stage 2 and 3.

```json
{
  "zero_optimization": {
    "stage": [2, 3]
  }
}
```

### Configuring Train Micro-Batch Size

The DeepSpeed Autotuner tunes the micro-batch size per GPU (`train_micro_batch_size_per_gpu` in DeepSpeed configuration) along with gradient accumulation steps (`gradient_accumulation_steps` in DeepSpeed configuration). The `train_micro_batch_size_per_gpu` value specified by the user in the DeepSpeed configuration file is used as the minimal micro-batch size per GPU to tune if it's runnable.

When using Hugging Face and `train_micro_batch_size_per_gpu` is set to ["auto"](#using-autotuning-with-hugging-face), if `train_micro_batch_size_per_gpu` has a corresponding training script mapping provided in `args_mapping`, the command-line value is used as the minimal micro-batch size per GPU to tune; else, `1` would be used as the minimal micro-batch size per GPU in tuning.

`train_batch_size` in DeepSpeed configuration must be equal to `train_micro_batch_size_per_gpu * gradient_accumulation_steps * total_num_gpus // model_parallelism_size `. Currently, the DeepSpeed Autotuner ignores the `train_batch_size` parameter specified in the DeepSpeed configuration file, please use `train_micro_batch_size_per_gpu` and `gradient_accumulation_steps` in autotuning.

The configuration below asks the Autotuner to use `4` as the minimal micro-batch size per GPU in tuning. Note that the value passed to the training script through `--per_device_train_batch_size` is ignored (which is supposed to be equal to the `train_micro_batch_size_per_gpu` value set in the DeepSpeed configuration).

```json
{
    "train_micro_batch_size_per_gpu": 4,
    "autotuning": {
        "enabled": true,
    },
    "arg_mappings": {
        "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
    }
}
```

The configuration below asks the Autotuner to use the value of `"--per_device_train_batch_size"` in the training script as the minimal micro-batch size per GPU in tuning. Also, the training script takes `gradient_accumulation_steps` as an argument in training code.
```json
{
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "autotuning": {
        "enabled": true,
        "arg_mappings": {
            "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
            "gradient_accumulation_steps ": "--gradient_accumulation_steps"
        }
    }
}
```

Users can set the maximum train batch size (global effective batch size) for the autotuning process by specifying `max_train_batch_size` in the autotuning configuration section of the DeepSpeed configuration file, as shown below. If `max_train_batch_size` is not defined, the Autotuner would use `maximum_train_micro_batch_size_per_gpu_runnable * gradient_accumulation_steps * total_num_gpus // model_parallelism_size` as `max_train_batch_size` (`gradient_accumulation_steps` defined in the DeepSpeed configuration file or training script or `1` is used here).

```json
{
    "autotuning": {
        "enabled": true,
        "max_train_batch_size": 1024,
        "arg_mappings": {
          "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
        }
    }
}
```

By default, the DeepSpeed Autotuning would select at maximum `num_tuning_micro_batch_sizes` (micro-batch size per GPU, gradient accumulation steps) pairs to tune ZeRO configurations. `num_tuning_micro_batch_sizes` defaults to `3` and can be set in the [autotuning configuration](#autotuning-configuration).

Users can specify the list of micro-batch sizes to tune in the DeepSpeed configuration file.
For example, the following section in the DeepSpeed configuration file limits the autotuning to explore `train_micro_batch_size_per_gpu` in `[1, 4, 16]`, and `gradient_accumulation_steps = 2` is used. Combinations of the two parameters are considered in the tuning (constrained by `max_train_batch_size` if defined). Note that specifying a list of `gradient_accumulation_steps` to tune is not supported.

```json
{
  "train_micro_batch_size_per_gpu": [1, 4, 16],
  "gradient_accumulation_steps": 2
}
```

The entry below asks the Autotuner to use `4` as the micro-batch size per GPU in tuning (micro-batch size per GPU is fixed as 4). Note that it's different from using ` "train_micro_batch_size_per_gpu": [4]` which asks the Autotuner to tune micro-batch size per GPU starting from `4`.
```json
{
    "train_micro_batch_size_per_gpu": [4],
}
```

#### Learning rate scaling when the effective batch size changes

Given that DS Autotuner provides users the flexibility to explore the best performance configuration under a range of different batch sizes (e.g., by changing the `train_micro_batch_size_per_gpu`), it is possible that the total effective batch size `B'` per training iteration that maximizes the compute efficiency is different from the one `B` the user originally uses for training. If the user decides to choose the best-performing batch size `B'` identified by DS autotuner for training to achieve faster training speed, we suggest the user to scale the learning rate by `sqrt(B'/B)` while keeping the other hyperparameters unchanged. The rationale behind this scaling is that one should scale the learning rate such that the variance in the gradient expectation remains constant when the batch size changes. In the case of stochastic gradient descent, we recommend the user to scale the learning rate linearly by `B'/B` while keeping the other hyperparameters (momentum, weight decay, etc.) the same, which we empirically find to work better for SGD and momentum-based optimizer.


### Configuring ZeRO configurations

The DeepSpeed Autotuner explores a set of carefully-chosen default values for ZeRO configuration parameters, defined in [`DEFAULT_TUNING_SPACE_ZERO_0,1,2,3`](constants.py). Users can overwrite any of the parameters (using a value or a list of values) in the DeepSpeed configuration file.

For example, the default tuning space for ZeRO stage 1 is
```python
DEFAULT_TUNING_SPACE_ZERO_1 = {
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": [5e7,
                               5e8,
                               1e9],
        "allgather_bucket_size": [5e7,
                                  5e8,
                                  1e9],
    }
}
```
, where `3*3 = 9` combinations of different `reduce_bucket_size` and `allgather_bucket_size` values are explored in the tuning. Users can overwrite it in the DeepSpeed configuration file
by
```json
{
  "zero_optimization": {
    "stage": 1,
    "reduce_bucket_size": [5e7, 5e8],
    "allgather_bucket_size": 5e8,
  }
}
```
, where only `2*1` cases `{"reduce_bucket_size": 5e7, "allgather_bucket_size": 5e8}` and `{"reduce_bucket_size": 5e8, "allgather_bucket_size": 5e8}` would be explored in the tuning.
If `"stage"` is not defined or set as `"all"`, then the overwriting applies to all ZeRO stages.
#### Offloading and NVME

Currently, the DeepSpeed Autotuner does not tune offloading behaviors but instead uses the values defined in the offload section of the DeepSpeed configuration file. See [Parameter offloading](https://www.deepspeed.ai/docs/config-json/#parameter-offloading) and [Optimizer offloading](https://www.deepspeed.ai/docs/config-json/#optimizer-offloading) for details.

If using NVME for offloading, users can run a benchmark offline to select the optimal `aio` setup in DeepSpeed. Refer to [profiling NVMe and configuring aio param section](https://github.com/microsoft/DeepSpeed/issues/998).

## Autotuning Output

By default, the DeepSpeed Autotuner generates a folder named `"autotuning_exps"` to store the descriptions of the autotuning experiments, and a folder named `"autotuning_results"` to store the results of the autotuning experiments under the training script launching path. Users can specify other path to use by setting `"results_dir"` or `"exps_dir"` in the autotuning configuration ([Results and Experiments Path](#results-and-experiments-path)).

Each autotuning experiment has a unique name based on the tuning parameters and. For example, z1_tmbspg3_gas1 means the experiment uses ZeRO stage 1, train micro-batch size per GPU (tmbspg) of 3, and gradient accumulation steps (gas) of 1. Then the experiment description is store as file z1_tmbspg3_gas1.json in the `"exps_dir"` folder, and the experiment result is stored in a folder named z1_tmbspg3_gas1 under the `"results_dir"`.

Each experiment result folder could contain the following files:

```bash
z1_tmbspg3_gas1/ # z1_tmbspg4_gas1 experiment result folder
|-- cmd.txt # command used to launch the experiment
|-- ds_config.json # DeepSpeed configuration used in the experiment
|-- exp.json # experiment description, used by the Autotuner for experiment management
|-- metrics.json # performance metrics recorded for the experiment
|-- stderr.log # stderr of running the experiment
`-- stdout.log # stdout of running the experiment
```

After the autotuning is done, a table of tuning experiments summary and autotuning duration would be printed to the terminal, for example:

```
| tuning_space | num_exps | best_metric_val | best_exp_name   |
| :----------- | -------: | --------------: | :-------------- |
| z0           |        2 |         90.1269 | z0_tmbspg2_gas1 |
| z1           |        2 |         187.298 | z1_tmbspg3_gas1 |
| z2           |        2 |         148.154 | z2_tmbspg3_gas1 |
| global       |        6 |         187.298 | z1_tmbspg3_gas1 |

Tuning completed in 0:00:03.602291
```

A file named `summary.txt` with the same content is saved under the `"results_dir"` for reference as well.
Other than the tuning summary, `ds_config_optimal.json`, the optimal DeepSpeed configuration found by autotuning,  and the corresponding command to launch the experiment `cmd_optimal.txt` are also saved under the `"results_dir"` after autotuning finishes.

## Autotuning Configuration

While `"autotuning": {"enabled": true}` is the minimal requirement to enable autotuning, there are other parameters users can define to configure the autotuning process. Below shows major parameters and their default values. These parameters can be set in the "autotuning" section in DeepSpeed configuration file.
```json
{
  "autotuning": {
    "enabled": false,
    "results_dir": null,
    "exps_dir": null,
    "overwrite": false,
    "metric": "throughput",
    "start_profile_step": 3,
    "end_profile_step": 5,
    "fast": true,
    "max_train_batch_size": null,
    "mp_size": 1,
    "num_tuning_micro_batch_sizes": 3,
    "tuner_type": "model_based",
    "tuner_early_stopping": 5,
    "tuner_num_trials": 50,
    "arg_mappings": null
  }
}
```

### Results and Experiments Path

`"results_dir"` points to a folder where the results of all the autotuning experiments are stored. `"exps_dir"` points to a folder where the descriptions of the autotuning experiments are stored.
By default, `"exps_dir"` is set to a folder named `"autotuning_exps"` and `"results_dir"` is set to a folder named `"autotuning_results"` under the training script launching path. Users can specify other paths to use by setting these two parameters in the autotuning configuration.

By default, the Autotuner does not run experiments whose results already exist. To change this behavior and rerun experiments all the time, set `"overwrite"` to true.

### Autotuning Metric

The Autotuner ranks tuning experiments by a metric. Currently, three metric types are supported, `"latency"`, `"throughput"`, and `"FLOPS"`:
* "throughput": training samples per second (calculated as  `train_batch_size * 1000 / "latency"`)
* "latency": training step latency in ms (`training iteration latency * gradient accumulation steps`)
* "FLOPS": floating-point operations per second achieved per GPU (calculated as `the number of flops / training iteration latency`). Refer to [DeepSpeed flops profiler](https://www.deepspeed.ai/tutorials/flops-profiler/) for details on how the number of flops is measured.

By default, `"throughput"` is used for ranking. Users can select other metrics, e.g., setting `{"metric": "latency"}` would use latency as the ranking metric.

Note that the performance metric used in autotuning is calculated using the timings captured within DeepSpeed forward, backward and step functions. The sum of these timings is less than the actual training step latency, thus the throughput metric values used by autotuning would be higher than the end-to-end throughput in training.

### Autotuning Resources

The DeepSpeed Autotuner uses all the hardware resources in the environment to run the tuning experiments. Experiments can be scheduled and run in parallel if resources are available and parallelization applies to the tuning logic (some steps in the tuning workflow is sequential).
For example, in an environment with 2 nodes and 16 GPUs per node, if the user specifies `--num_gpus=16` and `--num_nodes=1` in the training script, then at most two autotuning experiments can be run in parallel at a time.

### Profile Steps

In each of the tuning experiments, profiling is performed over a continuous portion of training steps to collect performance metrics, which are then used to rank the tuning experiments. Users can specify when to start and end the profiling.
* start_step (int, defaults to 3): the training step to start recording performance metrics
* end_step (int, >= start_step, defaults to 5): the training step to end recording performance metrics

Note that setting `start_step` to large values could result in a noticeable longer run time for each tuning experiment.

### Fast Mode

Besides ZeRO stages and micro-batch sizes per GPU (`fast` mode), users can tune other ZeRO optimization configurations by setting `"fast"` to `false`. The autotuning time would increase as the tuning space gets larger and more tuning experiments are performed. The fast mode is by default enabled.

### Max Train Batch Size

Users can set the maximum train batch size (global effective batch size) for the autotuning process by specifying `max_train_batch_size` in the autotuning configuration section of the DeepSpeed configuration file. If `max_train_batch_size` is not defined, the Autotuner would use `maximum_train_micro_batch_size_per_gpu_runnable * gradient_accumulation_steps * total_num_gpus // model_parallelism_size` as `max_train_batch_size` (`gradient_accumulation_steps` defined in the DeepSpeed configuration file or training script or `1` is used here). See [Configuring Train Micro-Batch Size](#configuring-train-micro-batch-size) for its usage with micro-batch size and gradient accumulation steps.

# Model Parallelism Size

If model parallelism is used, set the `mp_size` in the autotuning configuration to be the model parallelism degree. `mp_size` defaults to 1 which means no model parallelism is used.
### Tuning algorithms

Within a ZeRO stage, combinations of micro-batch sizes and other ZeRO configurations form a tuning space if experiments where the DeepSpeed Autotuner explores in an order (tuner algorithm).

Currently, three types of tuner algorithms are supported:
* `"random"`: randomly select the next set of configurations to experiment with.
* `" gridsearch" `: sequentially select the next set of configurations to experiment with.
* `"model_based"`: xgboost cost model is used to select the next set of configurations to experiment with given the results of the finished experiments.

By default, `"model_based"` algorithm is used.

The Autotuner stops exploring the space when any of the following conditions meet:

* When there is no more promising configurations are likely to be found. `"tuner_early_stopping"` defines the number of experiments to explore beyond the current best experiment. If no better experiment is found within that number, the Autotuner stops the exploration. `"tuner_early_stopping"` defaults to `5`.
* When the total number of experiments explored exceeds the `"tuner_num_trials"`, which defaults to `50`.
* When all the experiments in the tuning space are explored.

## Using Autotuning with Hugging Face

Hugging Face users can set some configurations values to ["auto"](https://huggingface.co/transformers/main_classes/deepspeed.html?highlight=gradient_accumulation_steps#shared-configuration).
`"auto"` means the value will be set to the default in Hugging Face or be overwritten using the supplied values from the command line arguments.
In DeepSpeed Autotuning, if the user-provided DeepSpeed configuration file has "auto" keywords, they are treated as the value "auto".

##  GPT2-large Example

This section shows an example of using DeepSpeed autotuning. For more examples, refer to [autotuning](https://github.com/microsoft/DeepSpeedExamples/tree/master/autotuning) in the DeepSpeedExamples repo.

Example training script:

```bash
MODEL_NAME=gpt2-large
PER_DEVICE_TRAIN_BATCH_SIZE=1
HF_PATH=~/projects # REPLACE WITH YOUR HUGGING FACE PATH
DS_CONFIG_PATH=ds_config.json # REPLACE WITH YOUR DEEPSPEED CONFIGURATION FILE PATH

NEPOCHS=1
NGPUS=16
NNODES=1
OUTPUT_DIR=./output_b${PER_DEVICE_TRAIN_BATCH_SIZE}_g${NGPUS}

deepspeed --autotuning run --num_nodes=$NNODES --num_gpus=$NGPUS $HF_PATH/transformers/examples/pytorch/language-modeling/run_clm.py --deepspeed $DS_CONFIG_PATH \
--model_name_or_path $MODEL_NAME \
--dataset_name wikitext \
--dataset_config_name wikitext-2-raw-v1 \
--do_train \
--do_eval \
--fp16 \
--per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
--learning_rate 2e-5 \
--num_train_epochs $NEPOCHS \
--output_dir ${OUTPUT_DIR} \
--overwrite_output_dir
```

Example DeepSpeed configuration file:

```json
{
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "autotuning": {
    "enabled": true,
    "arg_mappings": {
      "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
      "gradient_accumulation_steps": "--gradient_accumulation_steps"
    },
  }
}
```

Example output (in `summary.txt`):

```
| tuning_space | num_experiments | best_metric_val | best_exp_name   |
| :----------- | --------------: | --------------: | :-------------- |
| z0           |               4 |         59.0229 | z0_gas1_tmbspg2 |
| z1           |               5 |         87.3017 | z1_gas1_tmbspg3 |
| z2           |               3 |         77.8338 | z2_gas1_tmbspg3 |
| z3           |               1 |               0 | z3_gas1_tmbspg3 |
| global       |              13 |         87.3017 | z1_gas1_tmbspg3 |

Tuning completed in 0:27:33.988447. Total number of experiments: 13.
```

The table below shows the throughput (samples per second) comparison. The corresponding train micro-batch size per GPU (mbs or tmbspg) and ZeRO stage used to achieve the throughput value is also shown in the parentheses. Assume the strategy users would use in the hand-tuning process is to start from `mbs = 1` and increase mbs by 2 each time until running out of GPU memory.
 - `baseline` is the vanilla Hugging Face (HF) without DeepSpeed (DS) and mbs is hand-tuned.
 - `HF + DS hand-tuned` is HF with DS, and mbs is hand-tuned while other DS configuration uses default values.
 - `HF + DS autotuning` is HF with DS, and the DS configuration selected from autotuning.

Notation: Hugging Face (HF), DeepSpeed (DS), ZeRO stage (z), gradient accumulation steps (gas), train micro-batch size per GPU (mbs or tmbspg).

| Model name | baseline (vanilla HF) | HF + DS hand-tuned       | HF + DS autotuning (fast-mode) |
| ---------- | -------------------- | ------------------------ | ------------------------------ |
| GPT2-large | 27.874 (mbs = 1)     | 56.797 (z = 1, mbs = 2), | 69.061 (z = 1, mbs = 3)        |

As we can see the DeepSpeed Autotuner can select a better than hand-tuned configuration with a reasonable number of experiments. Examples in [Autotuning Hugging Face Examples](https://github.com/microsoft/DeepSpeedExamples/tree/master/autotuning/hf#autotuning-hugging-face-examples) would demonstrate the effectiveness of autotuning across different models.
