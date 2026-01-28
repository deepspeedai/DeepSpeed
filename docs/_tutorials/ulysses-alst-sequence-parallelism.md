---
title: Arctic Long Sequence Training (ALST) for HF Transformers integration
tags: training, finetuning, sequence-parallelism, long-sequence
---

1. Ulysses Sequence Parallelism for Hugging Face (HF) Transformers implements an efficient way of training on long sequences by employing sequence parallelism and attention head parallelism.
2. Arctic Long Sequence Training (ALST) enables even longer sequence lengths using a bag of tricks:
- Activation checkpoint offload to CPU
- Tiled MLP compute
- Liger-kernel
- PYTORCH_CUDA_ALLOC_CONF

It enables on LLama-8B training on 500K tokens on a single H100 GPU, 3.7M on a single node, and 15M on Llama-8B using just four nodes.

To learn about this technology please read this paper: [Arctic Long Sequence Training: Scalable And Efficient Training For Multi-Million Token Sequences](https://arxiv.org/abs/2506.13996).

It's already fully integrated into Arctic Training, see [this guide](https://github.com/snowflakedb/ArcticTraining/blob/main/projects/sequence-parallelism/).

The rest of the document explains how to integrate it into other frameworks or your own training loop.

There is another older version of UlyssesSP which only works with Megatron-Deepspeed and can be found [here](https://www.deepspeed.ai/tutorials/ds-sequence/).

## Part 1: Ulysses Sequence Parallelism for HF Transformers

If you want to integrate Ulysses Sequence Parallelism for HF Transformers into your framework, it's easy to do. Here is a full training loop with a hardcoded dataset:

```python
# train.py
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF, UlyssesSPDataLoaderAdapter
from deepspeed.runtime.utils import move_to_device
from deepspeed.utils import groups
from torch import tensor
from transformers import AutoModelForCausalLM
import deepspeed
import deepspeed.comm as dist
import torch

model_name_or_path = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
seq_length = 64
sequence_parallel_size = 2
micro_batch_size = 1

config_dict = {
    "train_micro_batch_size_per_gpu": 1,
    "zero_optimization": {
        "stage": 3,
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-3
        }
    },
    "sequence_parallel_size": sequence_parallel_size,
}

dtype = torch.bfloat16

# a simple Dataset
# replace with a real dataset but make sure `position_ids` are returned
input_ids = tensor([[1, 10, 10, 10, 2, 2], [1, 20, 20, 20, 2, 2]], )
position_ids = tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
ds = torch.utils.data.TensorDataset(input_ids, position_ids)
def collate_fn(batch):
    input_ids, position_ids = batch[0]
    return dict(input_ids=input_ids.unsqueeze(0),
                position_ids=position_ids.unsqueeze(0),
                labels=input_ids.unsqueeze(0))

dist.init_distributed(dist_backend='nccl', dist_init_required=True)

# Ulysses injection into HF Transformers
mpu = UlyssesSPAttentionHF.register_with_transformers(
    model_name_or_path=model_name_or_path,
    core_attn_implementation="sdpa",
    sequence_parallel_size=sequence_parallel_size,
    micro_batch_size=micro_batch_size,
    seq_length=seq_length,
    seq_length_is_variable=True,
)

# Deepspeed setup
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model, _, _, _ = deepspeed.initialize(config=config_dict,
                                        model=model,
                                        model_parameters=model.parameters(),
                                        mpu=mpu)

# UlyssesSPDataLoaderAdapter injection
sp_group = groups._get_sequence_parallel_group()
sp_world_size = groups._get_sequence_parallel_world_size()
sp_rank = groups._get_sequence_parallel_rank()
dl = torch.utils.data.DataLoader(ds, batch_size=micro_batch_size, collate_fn=collate_fn)
dl = UlyssesSPDataLoaderAdapter(
    dl,
    sp_rank=sp_rank,
    sp_group=sp_group,
    sp_world_size=sp_world_size,
    device=model.device,
)

# Normal training loop
for iter, batch in enumerate(dl):
    batch = move_to_device(batch, model.device)

    outputs = model(**batch)
    # as of this writing HF doesn't calculate loss with shift_labels yet and requires us to do it manually (liger does that automatically)
    shift_labels = batch["shift_labels"]
    loss = model.module.loss_function(
        logits=outputs.logits,
        labels=None,
        shift_labels=shift_labels,
        vocab_size=model.module.config.vocab_size,
    )

    # differentiable weighted per-shard-loss aggregation across ranks
    losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=sp_group)
    # special dealing with SFT that has prompt tokens that aren't used in loss computation
    good_tokens = (shift_labels != -100).view(-1).sum()
    good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)
    total_loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(sp_world_size))
    total_good_tokens = sum(good_tokens_per_rank)
    loss = total_loss / max(total_good_tokens, 1)

    if dist.get_rank() == 0:
        print(f"{iter}: {loss=}")

    model.backward(loss)
```

Now to train:

```bash
$ deepspeed --num_gpus 2 train.py
0: loss=tensor(10.4248, device='cuda:0', grad_fn=<DivBackward0>)
1: loss=tensor(10.4248, device='cuda:0', grad_fn=<DivBackward0>)
2: loss=tensor(10.3818, device='cuda:0', grad_fn=<DivBackward0>)
3: loss=tensor(10.3818, device='cuda:0', grad_fn=<DivBackward0>)
```

This example has been derived from the [UlyssesSP unit test](https://github.com/deepspeedai/DeepSpeed/blob/master/tests/unit/ulysses_alst/test_ulysses_sp_hf.py).

Let's study the parts not normally present in the vanilla training loop:

### UlyssesSPAttentionHF.register_with_transformers

`UlyssesSPAttentionHF.register_with_transformers` injects Ulysses Attention adapter into HF Transformers.

```python
mpu = UlyssesSPAttentionHF.register_with_transformers(
    model_name_or_path=model_name_or_path,
    core_attn_implementation="sdpa",
    sequence_parallel_size=sequence_parallel_size,
    micro_batch_size=micro_batch_size,
    seq_length=seq_length,
    seq_length_is_variable=True,
)
```

It also creates nccl process groups encapsulated by the `mpu` object it returns.

For the `model_name_or_path` argument you can also pass the already existing HF Transformers `model` object.

`UlyssesSPAttentionHF.register_with_transformers` has to be called before `from_pretrained` is called.

If `seq_length_is_variable` is `True` (which is also the default value), `UlyssesSPAttentionHF` will recalculate the shapes on each `forward` based on the incoming batch's shapes - in which case you don't need to set `seq_length` - you can just skip it like so:
```
mpu = UlyssesSPAttentionHF.register_with_transformers(
    model_name_or_path=model_name_or_path,
    core_attn_implementation="sdpa",
    sequence_parallel_size=sequence_parallel_size,
    micro_batch_size=micro_batch_size,
    seq_length_is_variable=True,
)
```

If, however, all your batches have an identical sequence length, then you'd save a few microseconds per run with using the `seq_length_is_variable=False` code path, which will pre-measure all shapes once and re-use them in all runs:

```
mpu = UlyssesSPAttentionHF.register_with_transformers(
    [...]
    seq_length=seq_length,
    seq_length_is_variable=False,
)
```

If you pass `seq_length`, remember that it has to be divisible by `sequence_parallel_size`. And of course, this also applies to all batches, even if you use `seq_length_is_variable=True`.


### UlyssesSPDataLoaderAdapter

```python
dl = UlyssesSPDataLoaderAdapter(
    dl,
    sp_rank=sp_rank,
    sp_group=sp_group,
    sp_world_size=sp_world_size,
    device=model.device,
)
```

This takes an existing DataLoader object and returns a new one that will shard the batches on the sequence dimension and synchronize all GPUs of the replica to return to each rank only its corresponding sequence shard.

It also takes care of replacing `labels` with `shift_labels` in the batch, by pre-shifting labels, which is crucial for the correct loss calculation when using Ulysses sequence parallelism.

### Loss averaging

Since each rank processes a segment we need to average loss. To get the gradients right we need to use a differentiable `all_gather`

```python
    # differentiable weighted per-shard-loss aggregation across ranks
    losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=sp_group)
    # special dealing with SFT that has prompt tokens that aren't used in loss computation
    good_tokens = (shift_labels != -100).view(-1).sum()
    good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)
    total_loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(sp_world_size))
    total_good_tokens = sum(good_tokens_per_rank)
    loss = total_loss / max(total_good_tokens, 1)
```

In theory you could just average `losses_per_rank`, but the system supports variable sequence length so the last rank is likely to have a shorter sequence length and also use cases like SFT may have a variable number of tokens that contribute to the loss calculation, so it's best to compute a weighted loss.

## Nuances

### Note on PyTorch Versions < 2.3

If you are using Sequence Parallelism with **PyTorch version < 2.3**, you may encounter an `IndexError: tuple index out of range` during the backward pass when `sequence_parallel_size < world_size`. This is due to a known issue in the `torch.distributed.all_gather` backward implementation in older versions.

**Workaround:** We recommend using a **weighted `all_reduce` pattern** instead of `all_gather` for loss averaging. You can refer to our [regression test case](https://github.com/deepspeedai/DeepSpeed/blob/master/tests/unit/sequence_parallelism/test_ulysses.py) for a code example of this workaround.

### Why do labels need to be pre-shifted?

When using batch sharding one can't let the upstream `loss` function do the labels shifting. Here is why:

When calculating loss in an unsharded batch we end up with (shift left):

```
input_ids: [1 2 3 4 5 6 7    8   ]
labels   : [1 2 3 4 5 6 7    8   ]
shiftedl : [2 3 4 5 6 7 8 -100]
```

When sharded we lose label 5 once shifted:

```
input_ids: [1 2 3    4] [5 6 7    8]
labels   : [1 2 3    4] [5 6 7    8]
shiftedl : [2 3 4 -100] [6 7 8 -100]
```

So a new API was added in HF transformers to support pre-shifted labels, and then we end up with the correct labels passed to the loss function for each shard:

```
input_ids: [1 2 3 4]  [5 6 7 8]
labels   : [1 2 3 4]  [5 6 7 8]
shiftedl : [2 3 4 5]  [6 7 8 -100]
```

## Part 2. Arctic Long Sequence Training (ALST) enables even longer sequence lengths using a bag of tricks

### Tiled loss computation

If you use [Liger-kernel](https://github.com/linkedin/Liger-Kernel) it'll automatically do the very memory efficient loss computation without manifesting intermediate full logits tensor, which consume a huge among of GPU memory when long sequence lengths are used.

If your model isn't supported by Liger-kernel you can use our implementation, which uses about the same amount of memory, but which is slightly slower since it's written in plain PyTorch. Here is a simplified version of it:

```python
    def loss(self, batch):
        num_shards = 4
        outputs = model(**batch, use_cache=False)
        hidden_states = outputs.last_hidden_state

        kwargs_to_shard = dict(
            hidden_states=hidden_states,
            shift_labels=batch["shift_labels"],
        )
        kwargs_to_pass = dict(model=model, vocab_size=model.config.vocab_size)
        grad_requiring_tensor_key = "hidden_states"
        compute_params = [model.lm_head.weight]
        seqlen = shift_labels.shape[1]

        total_loss_sum = sequence_tiled_compute(
            loss_fn,
            seqlen,
            num_shards,
            kwargs_to_shard,
            kwargs_to_pass,
            grad_requiring_tensor_key,
            compute_params,
            output_unshard_dimension=0,  # loss is a scalar
            output_reduction="sum",
        )
        total_good_items = (shift_labels != -100).squeeze().sum()
        loss = total_loss_sum / max(total_good_items, 1)

        # differentiable weighted per-shard-loss aggregation across ranks
        losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=self.sp_group)
        good_tokens = (shift_labels != -100).view(-1).sum()
        good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=self.sp_group)
        total_loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(self.sp_world_size))
        total_good_tokens = sum(good_tokens_per_rank)
        loss = total_loss / max(total_good_tokens, 1)

        return loss
```

You can see the full version [here](https://github.com/snowflakedb/ArcticTraining/blob/main/arctic_training/trainer/sft_trainer.py#L45).

### Tiled MLP computation

If you want to use Tiled MLP computation you'd need to monkey patch the model you work with, for a full example see this [unit test](https://github.com/deepspeedai/DeepSpeed/blob/master/tests/unit/ulysses_alst/test_tiled_compute.py).

```python
from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP
import transformers

def tiled_mlp_forward_common(self, x):
    """a monkey patch to replace modeling_llama.LlamaMLP.forward and other identical MLP implementations to perform a tiled compute of the same"""

    # figure out the number of shards
    bs, seqlen, hidden = x.shape
    num_shards = math.ceil(seqlen / hidden)
    # it's crucial that all ranks run the same number of shards, otherwise if one of the ranks
    # runs fewer shards than the rest, there will be a deadlock as that rank will stop running
    # sooner than others and will not supply its ZeRO-3 weights shard to other ranks. So we
    # will use the max value across all ranks.
    tensor = torch.tensor(num_shards, device=x.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    num_shards = tensor.item()
    # print(f"derived {num_shards} for {seqlen=} and {hidden=} max'ed across ranks")

    # only needed for deepspeed
    compute_params = [self.down_proj.weight, self.gate_proj.weight, self.up_proj.weight]

    def mlp_forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    return TiledMLP.apply(
        mlp_forward,
        self,
        x,
        num_shards,
        compute_params,
    )


from transformers.models.llama import modeling_llama
modeling_llama.LlamaMLP.forward = tiled_mlp_forward_common
```

You can of course come up with a different way of computing the number of shards to be used.

### Activation checkpoint offload to CPU

You will find a prototype implementation version [here](https://github.com/snowflakedb/ArcticTraining/blob/75758c863beff1c8a5c4e4987ba013ecaf377fc3/arctic_training/monkey_patches.py#L37)

```python
from arctic_training.monkey_patches import monkey_patch_checkpoint_function_with_cpu_offload
monkey_patch_checkpoint_function_with_cpu_offload()
```

We hope PyTorch core will provide an internal support for offloading. If not we will need to come up with some better solution - perhaps using a context manager.

This currently implementation isn't yet efficient (blocking), but it barely makes any difference for very long sequence lengths where `matmuls` dominate the compute.

### PYTORCH_CUDA_ALLOC_CONF

Before launching your script add:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This will help with minimizing memory fragmentation and will allow a longer sequence length.
