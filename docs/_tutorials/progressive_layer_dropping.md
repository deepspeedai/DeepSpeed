---
title: "Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping"
tags: training
---

In this tutorial, we are going to introduce the progressive layer dropping (PLD) in DeepSpeed and provide examples on how to use PLD. PLD allows to train Transformer networks such as BERT 24% faster under the same number of samples and 2.5 times faster to get similar accuracy on downstream tasks. Detailed description of PLD and the experimental results are available in our [technical report](https://arxiv.org/pdf/2010.13369.pdf).

To illustrate how to use PLD in DeepSpeed, we show how to enable PLD to pre-train a BERT model and fine-tune the pre-trained model on the GLUE datasets.

## Running Pre-training with DeepSpeed and PLD

To perform pre-training, one needs to first prepare the datasets. For this part, please refer our [BERT Pre-training](/tutorials/bert-pretraining/) post, which contains detailed information on how to do data downloading and pre-processing. For the below experiment, we use Wikipedia text and Bookcorpus, similar as [Devlin et. al.](https://arxiv.org/abs/1810.04805).

The main part of pre-training is done in `deepspeed_train.py`, which has
already been modified to use DeepSpeed. The  `ds_train_bert_progressive_layer_drop_bsz4k_seq128.sh` is the shell script that launches the pre-training with DeepSpeed and PLD.

```shell
bash ds_train_bert_progressive_layer_drop_bsz4k_seq128.sh
```

Most of the flags in the above script should be familiar if you have stepped through the BERT pre-training [tutorial](/tutorials/bert-pretraining/). To enable training with PLD, one needs to enable PLD in both the client script and in the DeepSpeed engine. To enable PLD in the client script, one needs to add the following command line flag to enable progressive layer dropping on Transformer blocks.

```bash
--progressive_layer_drop
```

To enable PLD in DeepSpeed, one needs to update the json configuration file with an appropriate PLD configuration dictionary like below:

```json
{
  ...
  "progressive_layer_drop": {
    "enabled": true,
    "theta": 0.5,
    "gamma": 0.001
  }
}
```

we recommend a PLD theta value of 0.5 and gamma of 0.001 because these have worked well in our experiments.

With these configuration changes, the DeepSpeed engine should print a runtime message as below:

    [INFO] [logging.py:60:log_dist] [Rank 0] Enabled progressive layer dropping (theta = 0.5)

The `deepspeed_bsz4k_progressive_layer_drop_config_seq128.json` file allows users to specify DeepSpeed options in terms of batch size, micro batch size, optimizer, learning rate, sequence length, and other parameters. Below is the DeepSpeed configuration file we use for running BERT and PLD.

```json
{
  "train_batch_size": 4096,
  "train_micro_batch_size_per_gpu": 16,
  "steps_per_print": 1000,
  "prescale_gradients": true,
  "gradient_predivide_factor": 8,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-3,
      "weight_decay": 0.01,
      "bias_correction": false
    }
  },
  "gradient_clipping": 1.0,

  "wall_clock_breakdown": false,

  "fp16": {
    "enabled": true,
    "loss_scale": 0
  },

  "progressive_layer_drop": {
    "enabled": true,
    "theta": 0.5,
    "gamma": 0.001
  }
}
```

Note that the above configuration assumes training on 64 X 32GB V100 GPUs. Each GPU uses a micro batch size of 16 and accumulates gradients until the effective batch size reaches 4096. If you have GPUs with less memory, you may need to reduce "train_micro_batch_size_per_gpu". Alternatively, if you have more GPUs, you can increase the "train_batch_size" to increase training speed. We use the following hyperparameters for pre-training BERT with PLD enabled.

| Parameters                     | Value                   |
| ------------------------------ | ----------------------- |
| Effective batch size           | 4K                      |
| Train micro batch size per GPU | 16                      |
| Optimizer                      | Adam                    |
| Peak learning rate             | 1e-3                    |
| Sequence-length                | 128                     |
| Learning rate scheduler        | Warmup linear decay exp |
| Warmup ratio                   | 0.02                    |
| Decay rate                     | 0.99                    |
| Decay step                     | 1000                    |
| Weight decay                   | 0.01                    |
| Gradient clipping              | 1.0                     |

Table 1. Pre-training hyperparameters

**Note:** DeepSpeed now supports PreLayerNorm as the default way for training BERT, because of its ability to avoid vanishing gradient, stabilize optimization, and performance gains, as described in our fastest BERT training [blog post](https://www.deepspeed.ai/2020/05/27/fastest-bert-training.html). We therefore support the switchable Transformer block directly on the BERT with PreLayerNorm. The implementation can be found at "example\bing_bert\nvidia\modelingpreln_layerdrop.py".

## Fine-tuning with DeepSpeed on GLUE Tasks

We use GLUE for fine-tuning tasks. GLUE (General Language Understanding Evaluation benchmark) (https://gluebenchmark.com/) is a  collection of sentence or sentence-pair natural language understanding tasks including question answering, sentiment analysis, and textual entailment.  It is designed to favor sample-efficient learning and knowledge-transfer across a range of different linguistic tasks in different domains.

One can download all GLUE data using the provided helper [script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e). Once the data has been downloaded, one can set up the data and move the data to "/data/GlueData", which is the default location for hosting GLUE data. We then can use the PLD pre-trained BERT model checkpoint to run the fine-tuning.

The main part of fine-tuning is done in `run_glue_classifier_bert_base.py`, which has
already been modified to use DeepSpeed. Before the fine-tuning, one needs to specify the BERT model configuration through the following config in `run_glue_classifier_bert_base.py`. In this case, it has already been modified to be the same as the configuration of the pre-trained model.

```json
    bert_model_config = {
        "vocab_size_or_config_json_file": 119547,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02
    }
```

Next, one can load a DeepSpeed style checkpoint with the following command, which has also already been added in the script.

```shell
model.load_state_dict(checkpoint_state_dict['module'], strict=False)
```

Finally, the `run_glue_classifier_bert_base.sh` script invokes pre-training and setups several hyperparameters relevant to fine-tuning.

```shell
bash run_glue_bert_base_finetune.sh [task] [batch size] [learning rate] [number of epochs] [job name] [checkpoint path]
```

An example would be:

```shell
bash run_glue_bert_base_finetune.sh MNLI 32 3e-5 5 "fine_tune_MNLI" deepspeed_checkpoint.pt
```



### Expected Results

The fine-tuning results can be found under the "logs" directory, and below are expected results for PLD on GLUE tasks. The "Lr" row indicates the learning rate we use for getting the corresponding accuracy result for each task.

|                        | RTE  | MRPC      | STS-B     | CoLA | SST-2 | QNLI | QQP       | MNLI-m/mm | GLUE |
| ---------------------- | :--: | --------- | --------- | ---- | ----- | ---- | --------- | --------- | ---- |
| Metrics                | Acc. | F1/Acc.   | PCC/SCC   | Acc. | Acc.  | Acc. | F1/Acc.   | Acc.      |      |
| Bert_{base} (original) | 66.4 | 88.9/84.8 | 87.1/89.2 | 52.1 | 93.5  | 90.5 | 71.2/89.2 | 84.6/83.4 | 80.7 |
| Bert_{base} (Our impl) | 67.8 | 88.0/86.0 | 89.5/89.2 | 52.5 | 91.2  | 87.1 | 89.0/90.6 | 82.5/83.4 | 82.1 |
| PLD                    | 69.3 | 86.6/84.3 | 90.0/89.6 | 55.8 | 91.6  | 90.7 | 89.6/91.2 | 84.1/83.8 | 82.9 |
| Lr                     | 7e-5 | 9e-5      | 7e-5      | 5e-5 | 7e-5  | 9e-5 | 2e-4      | 3e-5      |      |
