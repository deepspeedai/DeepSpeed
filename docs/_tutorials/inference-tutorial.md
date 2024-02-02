---
title: "Getting Started with DeepSpeed for Inferencing Transformer based Models"
tags: inference
---

>**DeepSpeed-Inference v2 is here and it's called DeepSpeed-FastGen! For the best performance, latest features, and newest model support please see our [DeepSpeed-FastGen release blog](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)!**

DeepSpeed-Inference introduces several features to efficiently serve transformer-based PyTorch models. It supports model parallelism (MP) to fit large models that would otherwise not fit in GPU memory. Even for smaller models, MP can be used to reduce latency for inference. To further reduce latency and cost, we introduce inference-customized kernels. Finally, we propose a novel approach to quantize models, called MoQ, to both shrink the model and reduce the inference cost at production. For more details on the inference related optimizations in DeepSpeed, please refer to our [blog post](https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/).

DeepSpeed provides a seamless inference mode for compatible transformer based models trained using DeepSpeed, Megatron, and HuggingFace, meaning that we don’t require any change on the modeling side such as exporting the model or creating a different checkpoint from your trained checkpoints. To run inference on multi-GPU for compatible models, provide the model parallelism degree and the checkpoint information or the model which is already loaded from a checkpoint, and DeepSpeed will do the rest. It will automatically partition the model as necessary, inject compatible high performance kernels into your model and manage the inter-gpu communication. For list of compatible models please see [here](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py).

## Initializing for Inference

For inference with DeepSpeed, use `init_inference` API to load the model for inference. Here, you can specify the MP degree, and if the model has not been loaded with the appropriate checkpoint, you can also provide the checkpoint description using a `json` file or the checkpoint path.

To inject the high-performance kernels, you need to set the `replace_with_kernel_inject` to True for the compatible models. For models not supported by DeepSpeed, the users can submit a PR that defines a new policy in [replace_policy class](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py) that specifies the different parameters of a Transformer layer, such as attention and feed-forward parts. The policy classes in DeepSpeed create a mapping between the parameters of the original user-supplied layer implementation with DeepSpeed's inference-optimized Transformer layer.

```python
# create the model
if args.pre_load_checkpoint:
    model = model_class.from_pretrained(args.model_name_or_path)
else:
    model = model_class()
...

import deepspeed

# Initialize the DeepSpeed-Inference engine
ds_engine = deepspeed.init_inference(model,
                                 tensor_parallel={"tp_size": 2},
                                 dtype=torch.half,
                                 checkpoint=None if args.pre_load_checkpoint else args.checkpoint_json,
                                 replace_with_kernel_inject=True)
model = ds_engine.module
output = model('Input String')
```

To run inference with only model-parallelism for the models that we don't support kernels, you can pass an injection policy that shows the two specific linear layers on a Transformer Encoder/Decoder layer: 1) the attention output GeMM and 2) layer output GeMM. We need these part of the layer to add the required all-reduce communication between GPUs to merge the partial results across model-parallel ranks. Below, we bring an example that shows how you can use deepspeed-inference with a T5 model:


```python
# create the model
import transformers
from transformers.models.t5.modeling_t5 import T5Block

import deepspeed

pipe = pipeline("text2text-generation", model="google/t5-v1_1-small", device=local_rank)
# Initialize the DeepSpeed-Inference engine
pipe.model = deepspeed.init_inference(
    pipe.model,
    tensor_parallel={"tp_size": world_size},
    dtype=torch.float,
    injection_policy={T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')}
)
output = pipe('Input String')
```

## Loading Checkpoints

For the models trained using HuggingFace, the model checkpoint can be pre-loaded using the `from_pretrained` API as shown above. For Megatron-LM models trained with model parallelism, we require a list of all the model parallel checkpoints passed in JSON config. Below we show how to load a Megatron-LM checkpoint trained using MP=2.

```json
"checkpoint.json":
{
    "type": "Megatron",
    "version": 0.0,
    "checkpoints": [
        "mp_rank_00/model_optim_rng.pt",
        "mp_rank_01/model_optim_rng.pt",
    ],
}
```
For models that are trained with DeepSpeed, the checkpoint `json` file only requires storing the path to the model checkpoints.
```json
"checkpoint.json":
{
    "type": "ds_model",
    "version": 0.0,
    "checkpoints": "path_to_checkpoints",
}
```

> DeepSpeed supports running different MP degree for inference than from training. For example, a model trained without any MP can be run with MP=2, or a model trained with MP=4 can be inferenced without any MP. DeepSpeed automatically merges or splits checkpoints during initialization as necessary.

## Launching

Use the DeepSpeed launcher `deepspeed` to launch inference on multiple GPUs:

```bash
deepspeed --num_gpus 2 inference.py
```

## End-to-End GPT NEO 2.7B Inference

DeepSpeed inference can be used in conjunction with HuggingFace `pipeline`. Below is the end-to-end client code combining DeepSpeed inference with HuggingFace `pipeline` for generating text using the GPT-NEO-2.7B model.

```python
# Filename: gpt-neo-2.7b-generation.py
import os
import deepspeed
import torch
from transformers import pipeline

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B',
                     device=local_rank)



generator.model = deepspeed.init_inference(generator.model,
                                           tensor_parallel={"tp_size": world_size},
                                           dtype=torch.float,
                                           replace_with_kernel_inject=True)

string = generator("DeepSpeed is", do_sample=True, min_length=50)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(string)

```
The above script modifies the model in HuggingFace text-generation pipeline to use DeepSpeed inference. Note that here we can run the inference on multiple GPUs using the model-parallel tensor-slicing across GPUs even though the original model was trained without any model parallelism and the checkpoint is also a single GPU checkpoint. To run the client simply run:

```bash
deepspeed --num_gpus 2 gpt-neo-2.7b-generation.py
```
Below is an output of the generated text.  You can try other prompt and see how this model generates text.

```log
[{
    'generated_text': 'DeepSpeed is a blog about the future. We will consider the future of work, the future of living, and the future of society. We will focus in particular on the evolution of living conditions for humans and animals in the Anthropocene and its repercussions'
}]
```

## Datatypes and Quantized Models

DeepSpeed inference supports fp32, fp16 and int8 parameters. The appropriate datatype can be set using dtype in `init_inference`, and DeepSpeed will choose the kernels optimized for that datatype. For quantized int8 models, if the model was quantized using DeepSpeed's quantization approach ([MoQ](https://www.deepspeed.ai/2021/05/04/MoQ.html)), the setting by which the quantization is applied needs to be passed to `init_inference`. This setting includes the number of groups used for quantization and whether the MLP part of transformer is quantized with extra grouping. For more information on these parameters, please visit our [quantization tutorial](https://www.deepspeed.ai/tutorials/MoQ-tutorial/).

```python
import deepspeed
model = deepspeed.init_inference(model,
                                 checkpoint='./checkpoint.json',
                                 dtype=torch.int8,
                                 quantization_setting=(quantize_groups,
                                                       mlp_extra_grouping)
                                )
```

Congratulations! You have completed DeepSpeed inference Tutorial.
