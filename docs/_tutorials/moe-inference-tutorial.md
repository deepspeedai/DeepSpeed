---
title: "Getting Started with DeepSpeed-MoE for Inferencing Large-Scale MoE Models"
---

DeepSpeed-MoE Inference introduces several important features on top of the inference optimization for dense models ([DeepSpeed-Inference blog post](https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/)). It embraces several different types of parallelism, i.e. data-parallelism and tensor-slicing for the non-expert parameters and expert-parallelism and expert-slicing for the expert parameters. To maximize the aggregate memory-bandwidth, we provide the communication scheduling with parallelism coordination to effectively group and route tokens with the same critical-data-path. Moreover, we propose new modeling optimizations, PR-MoE and MoS, to reduce MoE model size while maintaining accuracy. For more information on the DeepSpeed MoE inference optimization, please refer to our [blog post]({{ site.press_release_v6 }}).

DeepSpeed provides a seamless inference mode for the variant of MoE models that are trained via the DeepSpeed-MoE library ([MoE tutorial](https://www.deepspeed.ai/tutorials/mixture-of-experts-nlg/)). To do so, one needs to simply use the deepspeed-inference engine to initialize the model to run the model in the eval mode.
In the next part, we elaborate on how to run MoE models at inference-mode through DeepSpeed and describing the different features for adding different parallelism support to serve the model at unprecedented scale and speed.


## Initializing for Inference

First step to use DeepSpeed-MoE inferenece is to initialize the expert-parallel groups. To do so, one can use the group utility from DeepSpeed to initialize the group (`deepspeed.utils.groups.initialize`). This function creates the groups based on minimum of the world\_size (total number of GPUs) and expert size. By using this group, we can partition the experts among the expert-parallel GPUs. If number of experts is lower than total number of GPUs, DeepSpeed-MoE leverages expert-slicing for partitioning the expert parameters between the expert-parallel GPUs.

For inference with DeepSpeed-MoE, use `init_inference` API to load the MoE model for inference. Here, you can specify the Model-parallelism/tensor-slicing (MP) degree, number of experts, and if the model has not been loaded with the appropriate checkpoint, you can also provide the checkpoint description using a `json` file or simply pass the `'checkpoint'` path to load the moddel. To inject the high-performance inference kernels, you can pass int the `replace_method` as `'auto'` and set the `replace_with_kernel_inject` to True.

```python

import deepspeed
import torch.distributed as dist

# Set expert-parallel size
world_size = dist.get_world_size()
expert_parallel_size = min(world_size, args.num_experts)

# Initialize the expert- parallel group
deepspeed.utils.groups.initialize(expert_parallel_size)

# create the model
model = get_model(model_provider)
...

# Initialize the DeepSpeed-Inference engine
ds_engine = deepspeed.init_inference(model,
                                     mp_size=tensor_slicing_size,
                                     dtype=torch.half,
                                     moe_experts=args.num_experts,
                                     checkpoint==getattr(args, 'checkpoint_path')
                                     replace_method='auto',
                                     replace_with_kernel_inject=True,)
model = ds_engine.module
output = model('Input String')
```



## End-to-End MoE Inference Example

Here, we show a text-generation example using an MoE model for which we can specify the model-parallel size and number of experts.
DeepSpeed inference-engine takes care of creating the different parallelism groups using the tensor-slicing degree, number of experts, and the total number of GPUs used for running the MoE model. Regarding the expert parameters, we first use the expert-parallelism to assign each group of experts to one GPU. If number of GPUs is higher than experts, we use expert-slicing to partition each expert vertically/horizontally across the GPUs.

Let's take a look at some of the parameters passed to run our example. Please refer to [DeepSpeed-Example](https://github.com/microsoft/Megatron-DeepSpeed/blob/moe/examples/generate_text.sh) for a complete generate-text inference example.


```bash
generate_samples_gpt.py \
       --tensor-model-parallel-size 1 \
       --num-experts ${experts} \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 32 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --load $checpoint_path \
       --fp16 \
       --ds-inference \
```

In order to show the performance scaling of DeepSpeed-MoE inference with increasing number of GPUs, we consider a 52B model architecture with 128 experts and 1.3B dense model using the parameters shown in the script above. In this example, we set tensor-slicing degree to one since the non-expert part of the model is relatively small (805M parameters). We use the last flag, `ds-inference`, to switch between DeepSpeed-MoE and PyTorch implementations.

Figure 1 shows the inference performance of three different configuration, PyTorch, DeepSpeed-MoE (Generic), and DeepSpeed-MoE (Specialized), running on 8, 16, and 32 GPUs. Compared to PyTorch, DeepSpeed-MoE obtains significantly higher performance benefit as we increased the number of GPUs. By using the generic DeepSpeed-MoE inference, we can get between 24% to 60% performance improvement over PyTorch. Additionally, by enabling the full features of DeepSpeed-MoE inference, such as communication optimization and MoE customized kernels, the performance speedup gets boosted (2x – 3.2x).

![52b-MoE-128](/assets/images/1.3B-MoE-128.png)

## PR-MoE for Faster Performance and Lower Inference Cost

To select between different MoE structures, we add a new parameter in our inference example, called `mlp-type`, to select between the `'standard'` MoE structure and the `'residual'` one to enable the modeling optimizations offered by PR-MoE. In addition to changing the `mlp-type`, we need to pass the number of experts differently when using PR-MoE. In contrast to standard MoE which uses the same number of experts for each MoE layer, PR-MoE uses different expert-count for the initial layers than the deeper layers of the network. Below is an example of PR-MoE using a mixture of 64 and 128 experts for every other layers:

```bash
experts="64 64 64 64 64 64 64 64 64 64 128 128"
generate_samples_gpt.py \
       --tensor-model-parallel-size 1 \
       --num-experts ${experts} \
       --mlp_type 'residual' \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --load $checpoint_path \
       --fp16 \
       --ds-inference \
```

To evaluate the performance of PR-MoE, we use the two model structures, `'standard'` and `'residual'` and the configuration parameters as shown in the table below. Since we cannot fit the non-expert part of the 24B+MoE-128 on a single GPU, we use a model-parallel size larger than one. We choose the tensor-slicing degree in order to get the best performance benefit.

|Model          |Size (billions) |#Layers |Hidden size |MP degree |EP degree |
|-------------  |-----           |-----   |-----       |-----     |-----     |
|2.4B+MoE-128   |107.7           |16      |3584        |1         |64 - 128  |
|24B+MoE-128    |1046.9          |30      |8192        |8         |64 - 128  |

We use 1 node (8 A100 GPUs) to run inference on the 2.4B+MoE-128 and 8 nodes (64 A100 GPUs) for the 24B+MoE-128. Figure 2 shows the performance of three different configuration: MoE-Standard (PyTorch), MoE-Standard (DeepSpeed-Generic), PR-MoE (DeepSpeed-Generic). By using the standard-MoE DeepSpeed improves inference performance by 1.4x and 1.65x compared to PyTorch for the two models, respectively. Furthermore, by using the PR-MoE, we can improve the performance speedups to 1.81x and 1.87x, while keeping the model quality maintained.


![52b-MoE-128](/assets/images/prmoe.png)

Congratulations! You have completed DeepSpeed-MoE inference Tutorial.
