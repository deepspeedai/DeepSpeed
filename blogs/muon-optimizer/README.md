# Using Muon Optimizer with DeepSpeed
## TL;DR
Muon optimizer has gained momentum with more and more use from community and also from Large Foundation Model like Kimi-K2-Thinking.  Now DeepSpeed supports Muon optimizer.

## What is Muon optimizer?
Muon is an optimizer designed for hidden 2D weights of a neural network.  It takes gradient of the weight, computes its momentum, and applies Newton-Schulz iterations to orthogonalize the momentum matrix, then uses this orthogonalized matrix to update the weight[1](https://kellerjordan.github.io/posts/muon/).  Because Muon only maintains one momentum buffer (versus Adam’s two), it uses less memory for optimizer states.  It is used by Keller Jordan’s mod of NanoGPT[2](https://github.com/KellerJordan/modded-nanogpt), Andrej Karpathy’s nanochat[3](https://github.com/karpathy/nanochat), and a variant of Muon (MuonClip) is also used by the production-level LLM Kimi-K2 from MoonShot[4](https://arxiv.org/pdf/2507.20534).

## Muon Optimizer support in DeepSpeed
One of the challenges of applying Muon optimizer to DeepSpeed is that previous optimizers (SGD, Adam) look at gradients as flattened buffers.   Thus it is hard to swap in Muon optimizer in the same place because the gradient buffers are already flattened.   We move the Muon update to the get_flat_partition function of stage 1 and 2 DeepSpeedZeroOptimizer in which per parameter gradients are still in unflattened stages, thus we can easily apply the Muon updates.

Muon optimizer works for hidden 2D gradients.   We apply a parse in model engine initializer to tag the model parameter with 'use_muon', if and only if the model parameter is 2D and is hidden.   When Muon optimizer is used, any gradient with parameter match 'use_muon' will use Muon optimizer to update weight.

Note that Muon is a hybrid optimizer: it uses Muon updates only for 2D hidden weights and falls back to Adam for all other parameters (embeddings, layer norms, biases, lm_head).  The DeepSpeed config supports separate learning rates via `muon_lr` (for Muon parameters) and `adam_lr` (for Adam parameters).

## Running DeepSpeed finetune with Muon optimizer
Deepspeed finetune demo[5](https://github.com/delock/deepspeed_finetune_demo) is a demo to use different DeepSpeed training features and compare their performance in a single place.  You can use it to test finetune LLM models with Muon optimizer:
```
git clone https://github.com/delock/deepspeed_finetune_demo
cd deepspeed_finetune_demo
./finetune.sh <NUM_GPUS> <MODEL_NAME> z2_muon.json
```

## Muon Optimizer Convergence Experiment Result

We compared Muon optimizer with AdamW optimizer by finetuning a Qwen2.5-3B model on the tatsu-lab/alpaca dataset.  To ensure a fair comparison, we performed learning rate sweeps for both optimizers independently and report results at each optimizer’s best configuration.

**Training Configuration:**
- Model: Qwen2.5-3B
- Dataset: tatsu-lab/alpaca
- ZeRO Stage 2, bf16
- Batch size: 32 (4 per GPU), 8 GPUs (A100 40GB)
- 1 epoch (~1460 steps), eval every 100 steps
- LR schedule: constant (no warmup, no decay)
- Gradient clipping: 1.0

**AdamW Optimizer Hyperparameters:**
- betas: (0.9, 0.999)
- eps: 1e-8
- weight_decay: 0.01

**Muon Optimizer Hyperparameters:**
- momentum: 0.95 (Muon parameters)
- betas: (0.9, 0.999) (Adam parameters)
- eps: 1e-8
- weight_decay: 0.01

**Learning Rate Sweep Results:**

For AdamW, we swept lr across {1e-6, 2e-6, 5e-6, 1e-5}. For Muon, we first swept muon_lr across {1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2} with adam_lr=2e-6, then swept adam_lr across {2e-6, 5e-6, 1e-5} with muon_lr=5e-3.

| Optimizer | Learning Rate | Final Eval Loss |
|-----------|---------------|-----------------|
| AdamW     | lr=1e-5       | 1.2404          |
| AdamW     | lr=5e-6       | 1.2001          |
| **AdamW** | **lr=2e-6**   | **1.1842**      |
| AdamW     | lr=1e-6       | 1.1883          |
| Muon      | muon_lr=5e-3, adam_lr=2e-6 | 1.1996 |
| **Muon**  | **muon_lr=5e-3, adam_lr=5e-6** | **1.1966** |
| Muon      | muon_lr=5e-3, adam_lr=1e-5 | 1.1970 |

**Convergence Trajectory (Best Configuration per Optimizer):**

| Step | AdamW (lr=2e-6) | Muon (muon_lr=5e-3, adam_lr=5e-6) |
|------|-----------------|-----------------------------------|
| 0    | 1.3278          | 1.3300                            |
| 100  | 1.2205          | 1.2814                            |
| 200  | 1.2101          | 1.2300                            |
| 500  | 1.1969          | 1.2107                            |
| 1000 | 1.1894          | 1.2009                            |
| 1400 | **1.1842**      | **1.1966**                        |

In this finetuning experiment, AdamW achieves a slightly lower final eval loss (1.1842) compared to Muon (1.1966).  AdamW also converges faster in early training steps.  This result is consistent with the observation that Muon’s strength has been demonstrated primarily in pretraining settings, while finetuning a pretrained model on a small dataset may not fully benefit from Muon’s orthogonalization approach.

## Muon Optimizer Memory Savings
Muon optimizer uses less memory for optimizer states than Adam, because it maintains one momentum buffer per parameter instead of two (first and second moment).

### Memory Usage Comparison
Note that Muon is a hybrid optimizer: 2D hidden weights use Muon (1 buffer), while remaining parameters (embeddings, layer norms, lm_head) still use Adam (2 buffers).  The actual memory savings depend on the fraction of parameters that are 2D hidden weights.  For typical transformer models, 70-80% of parameters are 2D hidden weights, so optimizer state memory is reduced by roughly 35-40%.  However, because total GPU memory also includes model weights, gradients, and activations, the end-to-end memory reduction is smaller (see measured results below).

| Optimizer | State Buffers per Param | Memory per Parameter |
|-----------|------------------------|---------------------|
| Adam      | 2 (m, v)               | 8 bytes             |
| Muon      | 1 (momentum)           | 4 bytes             |

### Measured GPU Memory: Qwen2.5-3B Finetuning
We measured peak GPU memory during finetuning Qwen2.5-3B on tatsu-lab/alpaca using the same 8xA100 (40GB) configuration described above (batch size 32, ZeRO Stage 2, bf16).

| Optimizer | Peak Memory per GPU | Savings vs AdamW |
|-----------|---------------------|------------------|
| AdamW     | 34.5 GiB            | —                |
| Muon      | 31.4 GiB            | 9%               |

Muon reduces per-GPU memory by approximately 3 GiB (9%) compared to AdamW.  The savings come entirely from optimizer states: Muon parameters store one momentum buffer (4 bytes) instead of Adam's two (8 bytes).  However, because optimizer states are only one component of total GPU memory (alongside model weights, gradients, and activations), the end-to-end reduction is modest.  For larger models or tighter memory budgets, this 9% savings could make the difference between fitting a workload on-device versus requiring CPU offloading.

## Future plan
Muon optimizer is getting more and more attention, and is verified by production-level open LLM model such as Kimi-K2 which has 1T weights.  This makes Muon a strong second choice and a potential replacement of Adam optimizer.   To make Muon optimizer more accessible in production environment, the following features are needed:

- [ ] Muon optimizer with ZeRO stage 3
- [ ] CPU Offloading support
- [ ] MuonClip support
- [ ] Performance optimization to make Muon optimizer more efficient

If you have thoughts, feedback and contribution on Muon optimizer, welcome to start an issue for discussion, or submit a PR to DeepSpeed.  Let’s make Muon optimizer rock solid and lightning fast in DeepSpeed!

## Contributors
This work is contributed from Wang, Zhipeng (@PKUWZP); Chi McIsaac(@qimcis) and Ma, Guokai (@delock)
