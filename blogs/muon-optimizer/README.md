# Using Muon Optimizer with DeepSpeed
## TL;DR
Muon optimizer has gain momentum with more and more use from community and also from Large Foundation Model like Kimi-K2-Thinking.  Now DeepSpeed supports Muon optimizer

## What is Muon optimizer?
Muon is an optimizer designed for hidden 2D weights of a neural network.  It takes gradient of the weight, computes its momentum, and applies NewtonSchulz iterations to orthogonalize the momentum matrix, then use this orthogonalized matrix to update weight[1](https://kellerjordan.github.io/posts/muon/).  With Muon optimizer, optimization are less likely to overfit, converge faster, and saves more memory than Adam optimizer.   It is used by Keller Jordan’s mod of NanoGPT[2](https://github.com/KellerJordan/modded-nanogpt), Andrej Karpathy’s nanochat[3](https://github.com/karpathy/nanochat), and a variant of Muon (MuonClip) is also used by product level LLM model Kimi-K2 from MoonShot[4](https://arxiv.org/pdf/2507.20534).

## Muon Optimizer support in DeepSpeed
One of the challenges of applying Muon optimizer to DeepSpeed is that previous optimizers (SGD, Adam) look at gradients as flattened buffers.   Thus it is hard to swap in Muon optimizer in the same place because the gradient buffers are already flattened.   We move the Muon update to the get_flat_partition function of stage 1 and 2 DeepSpeedZeroOptimizer in which per parameter gradients are still in unflattened stages, thus we can easily apply the Muon updates.

Muon optimizer works for hidden 2D gradients.   We apply a parse in model engine initializer to tag the model parameter with ‘use_muon’, if and only if the model parameter is 2D and is hidden.   When Muon optimizer is used, any gradient with parameter match ‘use_muon’ will use Muon optimizer to update weight.

## Running DeepSpeed finetune with Muon optimizer
Deepspeed finetune demo[5](https://github.com/delock/deepspeed_finetune_demo) is a demo to use different DeepSpeed training features and compare their performance in a single place.  You can use it to test finetune LLM models with Muon optimizer:
```
git clone https://github.com/delock/deepspeed_finetune_demo
cd deepspeed_finetune_demo
./finetune.sh <NUM_GPUS> <MODEL_NAME> z2_muon.json
```

## Muon Optimizer convergence experiment result
[TBD]

## Muon Optimizer memory overhead
Muon optimizer has significantly smaller memory requirements than Adam optimizer, making it particularly valuable for large-scale model training.

### Memory Usage Comparison
In theory, Muon optimizer has one momentum buffer while Adam has two, this makes Muon optimizer use 50% less memory for optimizer states compared to Adam

| Optimizer | Momentum Buffers | Memory per Parameter | Example: 3B Model |
|-----------|------------------|---------------------|-------------------|
| Adam      | 2 (m, v)         | 8 bytes             | ~24 GB            |
| Muon      | 1 (momentum)     | 4 bytes             | ~12 GB            |

With Muon optimizer, we can potentially use larger batch sizes and avoid CPU offloading.

### Real-world Example: 3B Model - No Offloading Required
We tested finetune Qwen2.5-3B model with tatsu-lab/aplaca dataset on 2xA100 (40GB GPU memory each) using batch size=8 and input length=512:

**Training Configuration:**
- Model: Qwen2.5-3B
- Dataset: tatsu-lab/alpaca (standard instruction-tuning dataset)
- Batch size: 8
- Sequence length: 512 tokens (standard for instruction-tuning)
- GPU memory: 80 GB total (2×A100 40GB)
- ZeRO Stage: Stage 2 (distributed optimizer states and gradients)

**Performance Test Results:**

BS=8, sequence length=512

| Optimizer | Offloading | Iteration Time |
|-----------|------------|----------------|
| Muon      | No         | 0.9s           |
| Adam      | No         | OOM (Crash)    |
| Adam      | Yes        | 4.5s           |

**Key Performance Insights:**

From this result, we can see in certain situation, Muon optimizer can use less memory and does not need CPU offloading, while Adam optimizer cannot fit GPU memory and requires CPU offloading.  This collaterally brings performance benefit even when Muon optimizer needs more computation, because no offloading needed.

## Future plan
Muon optimizer is getting more and more attention, and is verified by product level open LLM model such as Kimi-K2 which has 1T weights.  This makes Muon a second choice even potential replacement of Adam optimizer.   To make Muon optimizer more accessible in production environment, the following feature is needed:

- [ ] Muon optimizer with ZeRO stage 3
- [ ] CPU Offloading support
- [ ] MuonClip support
- [ ] Performance optimization to make Muon optimizer more efficient

If you have thoughts, feedback and contribution on Muon optimizer, welcome to start an issue for discussion, or submit a PR to DeepSpeed.  Let's make Muon optimizer rock solid and lightning fast in DeepSpeed!

## Contributors
This work is contributed from Wang, Zhipeng (@PKUWZP); Chi McIsaac(@qimcis) and Ma, Guokai (@delock)
