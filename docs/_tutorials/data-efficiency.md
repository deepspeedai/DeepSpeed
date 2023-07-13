---
title: "DeepSpeed Data Efficiency: A composable library that makes better use of data, increases training efficiency, and improves model quality"
tags: training pre-training
---

**What is DeepSpeed Data Efficiency:** DeepSpeed Data Efficiency is a library purposely built to make better use of data, increases training efficiency, and improves model quality.

**Why use DeepSpeed Data Efficiency:** DeepSpeed Data Efficiency offers novel data efficiency techniques to achieve better training efficiency and/or better model quality. DeepSpeed Data Efficiency takes extensibility, flexibility, and composability into consideration, which makes it easier to customize the techniques, apply the techniques to various training tasks, and compose multiple techniques together. We highly recommend you also to read [our blog](https://www.deepspeed.ai/2022/12/11/data-efficiency.html) to learn more about (at a high level) why we build DeepSpeed Data Efficiency and what benefits it provides to users. Additional technical details can be found in our papers, “[Random-LTD: Random and Layerwise Token Dropping Brings Efficient Training for Large-scale Transformers](https://arxiv.org/abs/2211.11586)” which describes the random-LTD technique, and “[DeepSpeed Data Efficiency: Improving Deep Learning Model Quality and Training Efficiency via Efficient Data Sampling and Routing](https://arxiv.org/abs/2212.03597)” which describes the curriculum learning technique and overall DeepSpeed Data Efficiency framework.

**How to use DeepSpeed Data Efficiency:** In the following tutorial, the first two sections will describe the data efficiency techniques supported by the library. The third section will describe how to compose the two techniques to achieve even better training efficiency/model quality.

## 1. Curriculum Learning

### 1.1 What is Curriculum Learning
Curriculum learning (proposed by [Yoshua Bengio et al.](https://dl.acm.org/doi/abs/10.1145/1553374.1553380)) aims to improve training convergence speed by presenting relatively easier or simpler examples earlier during training. Building a curriculum learning solution usually requires two components: the difficulty metric (i.e., how to quantify the difficulty of each data sample) and the pacing function (i.e., how to decide the curriculum difficulty range when sampling next training data batch).

### 1.2 When to use Curriculum Learning
Curriculum learning has been successfully applied to various training tasks (see details in for example [this survey paper](https://arxiv.org/abs/2010.13166)), and last year we also released a specific curriculum learning technique (sequence length warmup) for GPT-style model pretraining (see technical details in our paper “[The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models](https://openreview.net/forum?id=JpZ5du_Kdh)” published in NeurIPS 2022 and the [tutorial for this legacy curriculum learning feature](/tutorials/curriculum-learning/)). This new general curriculum learning library inside DeepSpeed Data Efficiency enables users to employ curriculum learning to their models at **maximum extensibility**: users can easily analyze, index, and sample their training data based on various customizable strategies. Using this library, we were able to explore different CL strategies for GPT-3 and BERT pretraining and identify the best solution that provides up to **1.5x data saving** while still maintaining similar model quality.

### 1.3 How to use Curriculum Learning

#### 1.3.1 GPT-3 and BERT pretraining
The `examples_deepspeed/data_efficiency` directory in our [Megatron-DeepSpeed repo](https://github.com/microsoft/Megatron-DeepSpeed) includes our examples of how to apply curriculum learning to GPT-3 and BERT pretraining. There are 3 steps: data analysis, pretraining, and eval/finetuning.

**Data analysis:** Curriculum learning requires a data analysis before pretraining that calculate the difficulty of each data sample (based on the metric provided by user), and build an index that map difficulty value to corresponding data samples. (There are exceptions: for example the truncation-based sequence length metric can be achieved by data postprocessing without data analysis.) We provide a data analyzer to perform the offline CPU-only data analysis.

`examples_deepspeed/data_efficiency/gpt/ds_analyze_*.sh` and `examples_deepspeed/data_efficiency/bert/ds_analyze_*.sh` are example scripts for GPT-3 and BERT's data analysis. Our data analyzer employs a simple Map-Reduce scheme. First, at the Map stage the `ds_analyze_*_data_map.sh` is used to split the dataset and compute the difficulty value for each data sample. User would need to provide a function to compute the metric (we implement ours in `examples_deepspeed/data_efficiency/analyze_data.py`), the raw training dataset, and other configurations such as number of CPU nodes and number of threads per node. Then the data analyzer will automatically splits the dataset based on number of workers, compute the difficulty values in a batched fashion, and write the results to two indexes: one index maps each data sample to its difficulty value, and another index maps each distinct difficulty value to the corresponding samples. Second, at the Reduce stage the `ds_analyze_*_data_reduce.sh` is used to merge the index files produced by all workers. One thing to note is that in order to enable speedup by distribution yet still being able to merge all the output, the Map stage will potentially generate a lot of output files, which is proportional to number of CPU nodes, number of threads per node, and number of possible metric values. Thus to avoid generating too much output files, we recommend to start with a smaller number of nodes/threads (in the output log we provide an estimate required time for users to judge if they want to increase number of workers), and we recommend to limit number of possible difficulty values when designing your difficulty metric (our experience shows that a few thousands of distinct values is already sufficient to enjoy the benefit of curriculum learning).

**Pretraining** `examples_deepspeed/data_efficiency/gpt/pretrain` and `examples_deepspeed/data_efficiency/bert/pretrain` include the example pretraining scripts with curriculum learning feature. Several changes are needed to enable curriculum learning during pretraining: (1) User need to provide a DeepSpeed json config file which includes configurations for curriculum learning (see [list of configuration](/docs/config-json/#data-efficiency) for details). We provide tested example configurations in `examples_deepspeed/data_efficiency/gpt/pretrain/ds_pretrain_gpt_1.3B_dense_run.sh` and `examples_deepspeed/data_efficiency/bert/pretrain/ds_pretrain_bert_336M_run.sh`. (2) When initializing the DeepSpeed engine via `deepspeed.initialize`, user needs to provide the train dataset and use the dataloader returned by the initialization (this dataloader includes the curriculum learning capability). We provide an example implementation of this change in `megatron/training.py` function `setup_model_and_optimizer`. (3) If the curriculum learning metric requires data postprocessing (such as truncation-based sequence length), user needs to use the DeepSpeed engine's `set_data_post_process_func` API to provide the postprocessing function. We provide an example implementation of this change in `megatron/training.py`, `pretrain_bert.py`, and `pretrain_gpt.py`. (4) If the curriculum learning metric requires a custom scheduling strategy (the pacing function), user needs to use the DeepSpeed engine's `set_custom_curriculum_learning_schedule` API to provide the function to update the max accepted difficulty during training. DeepSpeed engine will provide a global train step input to this callback function.

**Eval/finetuning** `examples_deepspeed/data_efficiency/gpt/eval/` and `examples_deepspeed/data_efficiency/bert/finetune` include the example scripts for GPT-3 model's zero-/few-shot evaluation and BERT model's finetuning. Our [paper](https://arxiv.org/abs/2212.03597) includes the reference eval/finetune results if you follow our example scripts to perform the pretraining/eval/finetuning.

#### 1.3.2 GPT-2 finetuning
The `data_efficiency/gpt_finetuning` directory in our [DeepSpeedExamples repo](https://github.com/microsoft/DeepSpeedExamples) includes our examples of how to apply curriculum learning to GPT-2 finetuning. `data_efficiency/gpt_finetuning/finetune/ds_finetune_gpt2_run.sh` is the example finetuning script. For CL metrics that require data analysis (e.g., the vocabulary rarity metric), you need to first use ```data_efficiency/gpt_finetuning/finetune/ds_analyze_gpt_data_*``` to analyze and index the dataset, similar to the GPT-3 pre-training case described above in 1.3.1.

## 2. Random layerwise token dropping (random-LTD)

### 2.1 What is random-LTD
Random-LTD is an efficient token drop method applied to each layer with random assignment. Precisely, for each layer, as compared to the baseline, random-LTD randomly selects a subset of the tokens and feeds them into the transformer layer. Afterward, we combine the output of transformer layer with the dropped tokens to recover the full sequence length. Thus, the next layer still receives the full sequence and can repeat this process. For more technical details please read [our random-LTD paper](https://arxiv.org/abs/2211.11586).

### 2.2 When to use random-LTD
When you want to pretrain/fine-tune a transformer-based model, it is always a good idea to try random-LTD, as it can achieve a better performance than the standard baseline training given the same amount of computational cost. If you have limited resources, random-LTD achieves similar accuracy as the original baseline method with up to 33.3% theoretical cost saving and up to 25.6% wall-clock time saving. Particularly, if you need to train a much larger model with >=24 layers and with >=2048 sequence length, our method will be much more efficient than baseline.

### 2.3 How to use random-LTD

#### 2.3.1 GPT-3 and BERT pretraining
The `examples_deepspeed/data_efficiency` directory in our [Megatron-DeepSpeed repo](https://github.com/microsoft/Megatron-DeepSpeed) includes our examples of how to apply random-LTD to GPT-3 and BERT pretraining.

`examples_deepspeed/data_efficiency/gpt/pretrain` and `examples_deepspeed/data_efficiency/bert/pretrain` include the example pretraining scripts with random-LTD feature. Several changes are needed to enable random-LTD during pretraining: (1) User need to provide a DeepSpeed json config file which includes configurations for random-LTD (see [list of configuration](/docs/config-json/#data-efficiency) for details). We provide tested example configurations in `examples_deepspeed/data_efficiency/gpt/pretrain/ds_pretrain_gpt_1.3B_dense_run.sh` and `examples_deepspeed/data_efficiency/bert/pretrain/ds_pretrain_bert_336M_run.sh`. (2) After initializing the DeepSpeed engine via `deepspeed.initialize`, user needs to use the `convert_to_random_ltd` API to convert and wrap the model layers in order to enable the random-LTD feature. We provide an example implementation of this change in `megatron/training.py` function `setup_model_and_optimizer`. (3) In order for random-LTD to understand the input argument mapping of the forward function, user need to change all the input arguments (except the hidden_states input) into keyword/named argument. For example, in `megatron/model/transformer.py` we changed the forward function from `def forward(self, hidden_states, attention_mask, encoder_output=None, enc_dec_attn_mask=None, layer_past=None, get_key_value=False):` to `def forward(self, hidden_states, attention_mask=None, encoder_output=None, enc_dec_attn_mask=None, layer_past=None, get_key_value=False):`. (4) When saving model checkpoints, (especially if the state dictionary has non-traditional structure) user needs to use the `remove_random_ltd_state_dict` API to convert the random-LTD-wrapped layers back to original model layers. We provide an example implementation of this change in `megatron/model/language_model.py`.

For eval/finetuning of the pretrained model, see [previous section](#131-gpt-3-and-bert-pretraining) about how to use our example scripts.

#### 2.3.2 GPT-2 and ViT finetuning
The `data_efficiency` directory in our [DeepSpeedExamples repo](https://github.com/microsoft/DeepSpeedExamples) includes our examples of how to apply random-LTD to GPT-2 and ViT finetuning.

Just like pretraining case, similar changes are required to enable random-LTD for finetuning: (1) DeepSpeed json config file. (2) Use the `convert_to_random_ltd` API to convert and wrap the model layers. (3) When saving model checkpoints, use the `remove_random_ltd_state_dict` API to convert the random-LTD-wrapped layers back to original model layers.

One can run our GPT finetuning example by:

```shell
DeepSpeedExamples/data_efficiency/gpt_finetuning$ pip install -r requirement.txt
DeepSpeedExamples/data_efficiency/gpt_finetuning$ bash ./bash_script/run_base_random_ltd.sh
DeepSpeedExamples/data_efficiency/gpt_finetuning$ bash ./bash_script/run_medium_random_ltd.sh
```

And the reference final result is:

```shell
For run_base_random_ltd.sh:
End of training epoch 3 step 1344 consumed_token 2148032 best perplexity 22.552324221233757 time 0.17486039188173083 hr

For run_medium_random_ltd.sh:
End of training epoch 3 step 1373 consumed_token 2147024 best perplexity 17.332243199130996 time 0.4661190489927928 hr
```

One can run our ViT finetuning example by:

```shell
DeepSpeedExamples/data_efficiency/vit_finetuning$ pip install -r requirement.txt
DeepSpeedExamples/data_efficiency/vit_finetuning$ bash ./bash_script/run_cifar.sh
DeepSpeedExamples/data_efficiency/vit_finetuning$ bash ./bash_script/run_imagenet.sh
```

And the reference final result is:

```shell
For run_cifar.sh:
13 epoch at time 480.6546013355255s | reserved_length 197
iter 5474 | LR [0.0001]| val_acc 97.97000122070312 | layer_token 305784192
```

## 3. Composing curriculum learning and random-LTD to achieve more

### 3.1 GPT-3 and BERT pretraining
The `examples_deepspeed/data_efficiency` directory in our [Megatron-DeepSpeed repo](https://github.com/microsoft/Megatron-DeepSpeed) includes our examples of how to compose curriculum learning random-LTD, and apply both of them to GPT-3 and BERT pretraining.

The changes needed are the same as described in previous two sections, since DeepSpeed Data Efficiency already handles the complexity when composing the two techniques. However, one thing to note is that since both random-LTD and some of the curriculum learning metrics will change the sequence length, it could require some extra code to calculate the effective sequence length at each step. We provide an example implementation of this change in `megatron/training.py` function `train` where we calculate the `actual_seq_length`.

#### 3.2 GPT-2 finetuning
The `data_efficiency/gpt_finetuning` directory in our [DeepSpeedExamples repo](https://github.com/microsoft/DeepSpeedExamples) includes our examples of how to compose curriculum learning random-LTD for GPT-2 finetuning. `data_efficiency/gpt_finetuning/finetune/ds_finetune_gpt2_run.sh` is the example finetuning script.
