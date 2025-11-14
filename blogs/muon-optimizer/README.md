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
Muon optimizer has smaller memory requirements than Adam optimizer.  Adam optimizer has to hold two momentum buffer as training state, Muon only has one momentum buffer.  Thus Muon need less memory of optimizer state.   For setup that exceeds GPU memory capacity and needs CPU offloading with Adam optimizer, Muon optimizer will need less GPU memory and may still be able to run without CPU offloading, avoid the overhead of CPU offloading.

## Future plan
Muon optimizer is getting more and more attention, and is verified by product level open LLM model such as Kimi-K2 which has 1T weights.  This makes Muon a second choice even potential replacement of Adam optimizer.   To make Muon optimizer more accessible in production environment, the following feature is needed:

- [ ] Muon optimizer with ZeRO stage 3
- [ ] CPU Offloading support
- [ ] MuonClip support
- [ ] Performance optimization to make Muon optimizer more efficient

If you have thoughts, feedback and contribution on Muon optimizer, welcome to start an issue for discussion, or submit a PR to DeepSpeed.  Let's make Muon optimizer rock solid and lightning fast in DeepSpeed!

## Contributors
This work is contributed from Wang, Zhipeng (@PKUWZP); Chi McIsaac(@qimcis) and Ma, Guokai (@delock)
