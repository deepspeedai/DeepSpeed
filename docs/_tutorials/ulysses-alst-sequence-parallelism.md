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

**Workaround:** We recommend using a **weighted `all_reduce` pattern** instead of `all_gather` for loss averaging. You can refer to our [regression test case](https://github.com/deepspeedai/DeepSpeed/blob/master/tests/unit/v1/sequence_parallelism/test_ulysses.py) for a code example of this workaround.

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

Do not use Tiled MLP on attention: tokens mix along `S`. For a side-by-side with batch-axis tiled rematerialization, see [Tiled rematerialization vs Tiled MLP](#tiled-rematerialization-vs-tiled-mlp).

### Tiled rematerialization

`TiledRematerialize` shards the **batch** axis (one row at a time) for modules that mix sequence but are independent across batch — attention and Gated DeltaNet, not token-independent MLPs (`TiledMLP` already covers those).

A packed `mbs=1` row — several independent samples end to end in one row, located by `cu_seqlens` — has physical batch=1, so this mechanism is a no-op on it; a separate entry point tiles *within* that row instead, see [Packed-sequence tiled rematerialization](#packed-sequence-tiled-rematerialization) below.

Only the inner workspace of the wrapped `forward` shrinks. The concatenated output and the saved input stay full-batch. Default first forward is full-batch; pass `tile_forward=True` to shard it too. Forward runs twice per step (three times under an outer activation checkpoint). The wrapped `forward` must be deterministic (disable dropout).

All ranks must use the same microbatch size so the shard loop count matches (required for ZeRO-3). Do **not** write a Python loop of `engine.backward` per row; that is gradient accumulation (GAS): ZeRO-3 allgathers the layer on every microbatch and reduce-scatters grads every backward (and trips ZeRO-2 `params_already_reduced`). `TiledRematerialize` gathers `compute_params` and holds them across the row loop, accumulates `dW` inside one `engine.backward`, and sets `ds_grad_is_ready` only on the last row, so grads reduce **once per step** instead of once per row.

```python
from deepspeed.runtime.activation_checkpointing.tiled_rematerialization import tiled_rematerialize
import deepspeed.comm as dist
import torch

def tiled_attn_forward(self, hidden_states, **kwargs):
    bs = hidden_states.shape[0]
    num_shards = bs
    tensor = torch.tensor(num_shards, device=hidden_states.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    num_shards = tensor.item()
    compute_params = [p for p in self.parameters() if p.requires_grad]
    return tiled_rematerialize(orig_forward, self, hidden_states, num_shards, compute_params)
```

`tiled_rematerialize_compute` shards extra batch-major tensors (`attention_mask`) and batch-major `(cos, sin)` position embeddings. dim0==1 tuples still pass through. It rejects `past_key_value` / `Cache` and `cu_seqlens`.

#### Tiled rematerialization vs Tiled MLP

These are not two APIs for the same knob. They tile **different axes** on **different modules**, and you normally want both.

| | Tiled MLP / `SequenceTiledCompute` | Tiled rematerialization |
|---|---|---|
| Axis | Sequence / tokens (dim −2 on `[B, S, H]` or `[S, H]`) | Batch (dim 0) |
| Legal on | Token-independent compute: SwiGLU MLP, tiled logits loss | Sequence-mixing, batch-independent: attention, Gated DeltaNet |
| Illegal on | Attention / GDN (would split the mixing axis) | MLP (use Tiled MLP); KV cache; `cu_seqlens` |
| What shrinks | MLP (or loss) GEMM workspace ~ `B · (S/shards) · intermediate` | Remat workspace of the wrap ~ `(B/shards) · S · …` |
| First forward | Always tiled `no_grad`, then concatenate | Default **full batch** (same kernel shapes as an uncheckpointed pass). `tile_forward=True` shards that pass too |
| `mbs=1` | Still helps **long S** (example: `shards = ceil(S / hidden)`) | **No-op** (`shards` collapses to 1) |
| `mbs>1`, moderate S | Often `shards=1` if `S` is not ≫ hidden | Cuts attention/GDN remat that scaled with `B` |
| Packed `mbs=1` row (`cu_seqlens`) | N/A (already token-major) | **Not** a no-op — separate entry point, see below |
| Ulysses | Complements SP: local `S` can still be huge; tile remaining tokens in the FFN | Orthogonal: SP already cut `S`; remat tiles `B` |
| ZeRO | `ds_grad_is_ready` only on the last **sequence** shard | Same last-shard reduce, but across **batch** rows, in one `engine.backward` (not GAS) |

Attention remat live set scales like `B · S`. Tiled remat cuts `B`; it does not cut MLP intermediates. MLP intermediates scale like `B · S · intermediate`. Tiled MLP cuts `S`; it does not cut attention/FA workspace.

The `mbs=1` row above is about `tiled_rematerialize`/`tiled_rematerialize_compute`, the batch-axis entry point: a physical `mbs=1` row has nothing for it to shard, so it collapses to a no-op. A packed `mbs=1` row — several independent samples in that one row — is a different case with its own entry point, `packed_seq_tiled_rematerialize_compute`, covered next; the two are separate functions on the same module, not two modes of one function.

Compose them on a decoder layer: wrap attention / GDN with tiled rematerialization, replace MLP (and optionally the fused logits loss) with Tiled MLP / `sequence_tiled_compute`. Tiled MLP is the ALST lever for million-token **`mbs=1`** runs. Tiled rematerialization is the lever when **`mbs>1`** made attention/GDN remat the bound — it does not replace Tiled MLP for long-`S` at `mbs=1`.

Qwen3.5 ZeRO-3 `0 vs intermediate_size` during backward is an emptied SwiGLU Linear weight under checkpoint rematerialize, not GDN `conv_dim`.

#### Packed-sequence tiled rematerialization

`packed_seq_tiled_rematerialize_compute` is a separate entry point for the case the table above marks a no-op: one physical row that **packs several independent samples end to end**, with `cu_seqlens` (or HF's `cu_seq_lens_q/k`) locating every sample boundary. It tiles the **token** axis of that one row over contiguous groups of *whole* samples, never splitting a sample — splitting would reset GDN recurrent/short-conv state and break causal attention across a document boundary. Like the batch-axis entry point, it wraps a sequence-mixer submodule only (attention, GDN), never a whole decoder layer or an MLP.

```python
from deepspeed.runtime.activation_checkpointing.tiled_rematerialization import packed_seq_tiled_rematerialize_compute

def tiled_attn_forward(self, hidden_states, cu_seqlens, position_ids=None, **kwargs):
    compute_params = [p for p in self.parameters() if p.requires_grad]
    return packed_seq_tiled_rematerialize_compute(
        orig_forward, hidden_states, cu_seqlens,
        token_budget=2048, compute_params=compute_params,
        kwargs_to_token_shard=dict(position_ids=position_ids),
    )
```

Give either `token_budget` (max tokens per tile) or `shards` (target tile count, converted to an implied budget). The budget is soft: a single sample longer than the budget still gets its own oversized tile rather than being split. A single-sample `cu_seqlens` is a bitwise no-op — the tiling machinery never runs, matching the batch axis's own `mbs<=1` no-op. On more than one rank it still joins the tile-count agreement below before taking the direct call, voting 1, so a rank holding one sample cannot leave its multi-sample peers waiting in the collective.

`cu_seqlens` is **rebased** per tile (`cu_seqlens[start:end+1] - cu_seqlens[start]`), not sliced like a batch-major tensor — passing `cu_seqlens` to `tiled_rematerialize_compute` instead still raises `TypeError`, since that entry point hard-rejects every `cu_seqlens`-named kwarg as a safety rail against exactly this mistake. `max_seqlen_q/k` / `max_length_q/k` are recomputed from each tile's own rebased `cu_seqlens`, never passed through stale from the full row. `position_ids` and other token-major companions go in `kwargs_to_token_shard` and are narrowed alongside `hidden_states`; an unrecognized tensor defaults to token-major rather than being rejected, since a packed row's companions are conventionally token-major.

Same cross-rank symmetry rule as the batch axis, but on tile **count** rather than shard count: two ranks packing different sample-length histograms into the same token budget can land on different local tile counts, and ZeRO-3 needs every rank to run the same number of loop iterations. Each rank computes its own greedy tile plan, then all ranks `all_reduce(MIN)` the tile count and every rank merges its own smallest-adjacent tile pair down to that minimum, repeatedly. This is always a shrink, never a split — a rank's tile count never exceeds its own sample count, and the cross-rank minimum never exceeds any rank's count — and it is deliberately **not cached**: unlike the batch axis's shard-count agreement, batch is always 1 on this axis (one physical row), so a cache key derived from `(shards, batch, world_size)` would collide across genuinely different packings within the same microstep and return a stale count.

#### One long GDN sequence

Packed tiling does not split a sample, so one 256k sequence is a single tile and pays the full GDN scan workspace. `sequential_chain_rematerialize` is the entry point for that case. It cuts the sequence into tiles whose lengths are a multiple of 64 (FLA's chunk), except the tail, and runs the tiles in order.

The link between tiles is the fp32 recurrent state and the width-4 conv's last 3 pre-activation inputs. Forward is a `no_grad` chain that stores those boundaries. Backward recomputes one tile and passes the next tile's state gradient back as `dht` / `dfinal_states`. One autograd graph for every tile would keep full-sequence `q`, `k`, and `v`, which is the workspace the `no_grad` forward drops.

Qwen's training forward never returns either state (`cache_params is None`), so the tile body is `qwen35_gdn_tile_forward`: the same projections, conv, scan, gated norm, and out-proj, with `output_final_state=True`. On CUDA the conv state goes through `causal_conv1d`'s `initial_states` / `return_final_states`. The torch fallback conv zero-pads and ignores that state, so it cannot continue a tile.

```python
from deepspeed.runtime.activation_checkpointing.tiled_rematerialization import (
    qwen35_gdn_tile_forward,
    sequential_chain_rematerialize,
)

def tiled_gdn(module, hidden_states):
    params = [p for p in module.parameters() if p.requires_grad]
    return sequential_chain_rematerialize(
        lambda x, recurrent, conv: qwen35_gdn_tile_forward(module, x, recurrent, conv),
        hidden_states,
        token_budget=32768,
        compute_params=params,
    )
```

A packed row whose samples already fit the budget does not need this. The chain matches that packed floor for a single long sequence; it does not go below it. Under ZeRO-3 the GDN module still has to be a leaf before `deepspeed.initialize`, and `ds_grad_is_ready` flips true on the last backward tile, which is the first tile of the sequence.

`qwen35_gdn_tile_forward` takes an optional `attention_mask` and zeroes padded tokens before the projections, as Qwen's `apply_mask_to_padding_states` does. The mask must cover exactly the tile's tokens; a mismatched shape raises. The chain hands `tile_fn` no mask, so for a padded batch multiply the full input by the mask once before chaining.

**Inside a packed row.** When a packed row mixes short samples with one longer than the chain budget, pass the tile body to the packed entry point instead of calling the chain per sample:

```python
def tiled_gdn_packed(module, hidden_states, cu_seqlens):
    params = [p for p in module.parameters() if p.requires_grad]
    return packed_seq_tiled_rematerialize_compute(
        orig_forward, hidden_states, cu_seqlens,
        token_budget=32768, compute_params=params,
        chain_tile_fn=lambda x, recurrent, conv: qwen35_gdn_tile_forward(module, x, recurrent, conv),
        chain_token_budget=32768,
    )
```

Packed tiling stays the outer loop. A packed tile that holds exactly one sample longer than `chain_token_budget` runs as a chain, from zero recurrent and conv states, since that is where the packed forward restarts every sample. The chain never crosses a sample boundary, and a packed tile that holds several short samples stays a single call. With a chain configured, the first forward walks the packed tiles too, rather than one whole-row call, so the long sample never pays its full scan workspace. The chain is GDN-specific (scan plus short conv); full-attention mixers stay on the plain packed path. `chain_tile_fn` sees only the sample's hidden states, so `kwargs_to_token_shard` is refused when it is set.

Two ownership rules keep the nesting safe under ZeRO:

1. **The outer loop owns the parameter gather.** The packed call gathers the ZeRO-3 weights once and holds them across every chain step. The nested chain does not open its own `GatheredParameters`, and does not touch `ds_persist` / `is_external_param`, because a second gather on weights the outer hold already owns is what breaks the lifecycle.
2. **The outer loop owns `ds_grad_is_ready`, and it is true for exactly one inner backward.** ZeRO reads the flag inside each parameter's gradient hook, and that hook fires on every inner `torch.autograd.backward`, not once per packed tile. While the flag is false, gradients accumulate in `param.grad`; the first backward that sees it true reduces the parameter, and any later gradient for it fails with "has already been reduced / Gradient computed twice". So if the last packed tile is chained into N steps, the flag stays false for the first N-1 reverse steps and turns true only on that chain's tile 0, the last inner backward of the whole mixer call. Marking the whole last packed tile ready up front would reduce after its first chain step and reject the rest.

The per-sample chain plan is not agreed across ranks, unlike the packed tile count. Ranks hold different numbers of oversized samples, so a collective per chain would hang. The inner loop issues no collectives, because the outer hold keeps the weights gathered and the ready flag flips once per call.

#### ZeRO-3: coalesced parameter gathers

A tiled layer used to issue roughly **2x** the allgathers of plain checkpointing (282 vs 129 on a 4-layer
Qwen3.5-4B across 8 ranks). The excess was per rematerialize pass, not per row: the count is identical for
2 and 4 shards, so the hold across rows was already working. The cost came from `GatheredParameters`, which
falls through to `_allgather_params_coalesced` -- a function that, despite its name, launches one
`all_gather_into_tensor` **per parameter** rather than flattening the layer into a single collective.

The rematerialize hold now opts into the coordinator's flattening path, which takes the tiled layer to
**126** allgathers, just under the **129** of plain checkpointing:

```python
# deepspeed/runtime/activation_checkpointing/tiled_rematerialization.py
with GatheredParameters(zero_params, modifier_rank=None, coalesced=True):
```

`coalesced=True` is opt-in, so every other `GatheredParameters` caller keeps the per-parameter path. The
flattening path is stricter -- it rejects a parameter that is not `NOT_AVAILABLE`, and it reads the
secondary-tensor and quantization choice off the first parameter of the list -- so the gather buckets by
process group, hpZ secondary tensor, and quantization before submitting.

#### ZeRO-3 leaf modules: slower, but sometimes required

`set_z3_leaf_modules` was the previous way to avoid the per-parameter wave, reaching 114 allgathers. Now
that coalescing gets within 10% of that (126) without changing gradient-reduction timing, the leaf costs
more than it returns. Measured with `cpu_checkpointing`, 3 interleaved repeats per arm:

| config | leaf | no leaf | verdict |
|---|---|---|---|
| Qwen3.5-4B @ 32k | 808.8 tok/s, 30.39 GB | 810.4 tok/s, 28.69 GB | -0.2%, ranges overlap |
| Qwen3.5-27B @ 8k | 579.9 tok/s, 66.56 GB | 616.9 tok/s, 64.61 GB | **-6.0%, ranges separated** |

Coalescing is also what moved the no-leaf numbers: at 27B it added **+14.3 tok/s** to the no-leaf arm while
the leaf arm gained only +3.8, which is the expected shape since the leaf already skipped the wave that
coalescing fixes. The leaf is slower and uses more memory at both sizes, so prefer tiled rematerialization
without it wherever the no-leaf path runs at all -- see the correctness caveat below. An earlier **+1.0%**
leaf result on 4B at 32k is superseded: it measured the leaf against a no-leaf path that was still paying
the per-parameter wave.

Correctness caveat: the table compares two arms that **both ran**, which is not the same as the no-leaf arm
always running. On the batch axis at `shards > 1`, the tiled forward calls the mixer once per shard, so an
unleafed module re-runs its ZeRO-3 per-submodule hooks per shard and can register the same parameter into a
gradient bucket twice, tripping
`assert len(set(p.ds_id for p in params_in_bucket)) == len(params_in_bucket)` in `stage3.py` during
reduction. Whether it trips depends on bucketing, so this is a scale effect rather than a flag you can set
once and forget: small models tile clean unleafed (the unit tests do, and the measurements above did), while
a full 36-layer Qwen3-4B at `mbs=2, shards=2` on 8 ranks asserts. The entry points log a warning naming the
fix whenever they tile a ZeRO-3 parameter that carries no leaf marking. On hitting either the warning or the
assert, mark the tiled module a leaf **before** `deepspeed.initialize`:

```python
from deepspeed.utils import set_z3_leaf_modules

set_z3_leaf_modules(model, [Qwen3DecoderLayer])  # before deepspeed.initialize(...)
```

The ordering is not stylistic. ZeRO-3 reads the marking while registering its hooks and does not recurse
into a leaf module's children, so marking a module after `deepspeed.initialize` has no effect at all. The
same applies to the packed-sequence entry point once it plans more than one tile.

Marking a module a ZeRO-3 leaf also makes ZeRO-3 attach a `register_full_backward_hook`, which is worth
knowing if you use one for other reasons alongside activation offload. That hook hands `forward` a *view*
whose `_base` is the caller's hidden. Rematerialization parks that view on its autograd context to restore
the input later, so the base storage stayed pinned for the whole step: offload copied every tensor to host
as usual, but **0%** of the bytes came back and ~10 GiB of saved layer inputs stayed resident. Holding the
restore handle weakly lets the view die at the end of forward, which releases the base; measured reclaim
went from 0% to 91%, matching the no-leaf path.

### Activation checkpoint offload to CPU

The one-line way to enable this for any DeepSpeedEngine training loop (HuggingFace
gradient checkpointing or native `deepspeed.checkpointing.checkpoint`) is:

```json
{
  "activation_checkpointing": {
    "cpu_checkpointing": true
  }
}
```

That flag offloads checkpointed hidden states to pinned CPU on a side stream, which
is usually enough to fit a several-times-longer sequence in the same GPU memory.
Keep `engine.backward(loss)` (or `loss.backward()` through the engine) so the offload
context stays open until unpack. See [Activation Checkpointing](/docs/config-json/#activation-checkpointing).

If you use the reentrant `torch.utils.checkpoint` API you can use the prototype monkeypatch from ArcticTraining [here](https://github.com/snowflakedb/ArcticTraining/blob/75758c863beff1c8a5c4e4987ba013ecaf377fc3/arctic_training/monkey_patches.py#L37):

```python
from arctic_training.monkey_patches import monkey_patch_checkpoint_function_with_cpu_offload
monkey_patch_checkpoint_function_with_cpu_offload()
```

For non-reentrant checkpointing (`use_reentrant=False`, the HF Transformers default)
outside `DeepSpeedEngine.forward`/`backward`, DeepSpeed also provides a context manager
that offloads only the checkpointed hidden-state inputs to pinned CPU memory on a side stream:

```python
from deepspeed.runtime.activation_checkpointing.offload_activations import (
    get_checkpoint_hidden_states_offloading_ctx_manager,
)

ctx = get_checkpoint_hidden_states_offloading_ctx_manager()
with ctx:
    loss = model(**batch).loss
    loss.backward()
```

Re-use one manager and wrap each training step (forward+backward) in it; `backward()` must run inside the same context as forward. Requires `transformers`: the marker identifying checkpoint inputs is installed on `GradientCheckpointingLayer` only while a manager is active (so the same model can be reused for HybridEngine rollout), and all other saved tensors pass through untouched. Tune with `use_pin_memory` (pin on/off for async D2H/H2D; default true), `use_streams`, `min_offload_bytes` (threshold), `max_fwd_stash_count` (in-flight GPU copies), `max_cpu_buffer_pool_count` (pooled CPU buffers), and `keep_last_count` (default 1: leave the last checkpoint input on GPU so the first backward is not stalled by a D2H of an activation that is needed immediately). Peak extra GPU activations retained is `max_fwd_stash_count + keep_last_count`. Host buffers are pinned with `get_accelerator().pin_memory()`, so pinning honors `DS_PIN_MEMORY_BACKEND` and is reflected in DeepSpeed's pinned-memory accounting. Keep the default `torch` backend here: `native` is `mlock` without `cudaHostRegister`, so its buffers are not registered for accelerator DMA and side-stream D2H would stall. Same pin-on/off story as DeepCompile `compile.offload_activation_pin_memory`. Do not nest this manager around `engine.forward()` when `cpu_checkpointing` is already enabled in ds_config.

### PYTORCH_CUDA_ALLOC_CONF

Before launching your script add:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This will help with minimizing memory fragmentation and will allow a longer sequence length.
