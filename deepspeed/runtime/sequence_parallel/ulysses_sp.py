# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
*** Arctic Long Sequence Training (ALST) components ***

1. Ulysses Sequence Parallelism for HF Transformers implements an efficient way of training on long sequences by employing sequence parallelism and attention head parallelism.
2. ALST enables even longer sequence lengths using a bag of tricks:
- Activation checkpoint offload to CPU
- Tiled MLP compute
- Liger-kernel
- PYTORCH_CUDA_ALLOC_CONF

ALST features found in this module:

- `UlyssesSPAttentionHF` - port of UlyssesAttention from Megatron-Deepspeed plus modern MHA-variations
- `UlyssesSPDataLoaderAdapter` - DL adapter to shard the normal DL batches to be used by `UlyssesSPAttentionHF`
- `SequenceTiledCompute` - generic autograd function to perform compute after tiling on the sequence dimension
- `TiledMLP` - a specific autograd function to perform tiled MLP (it's much easier to understand before trying to grok `SequenceTiledCompute`)
- `TiledFusedLogitsLoss` - a specific autograd function to perform loss computation without manifesting the full logits tensor and instead computing loss on shards of logits.

This module implements Arctic Long Sequence Training: Scalable And Efficient Training For Multi-Million Token Sequences: https://arxiv.org/abs/2506.13996

For integration docs see: https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/

The other ALST features live inside
https://github.com/snowflakedb/ArcticTraining/blob/main/projects/sequence-parallelism/

"""

from collections import defaultdict
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.sequence.layer import _DimZeroAllToAll
from deepspeed.utils.logging import logger
from einops import rearrange
from packaging import version
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Any
from typing import Tuple
import deepspeed.comm as dist
import importlib.metadata
import math
import torch
import torch.distributed.nn


class UlyssesSPAttentionHF(torch.nn.Module):
    """Re-Implementation of deepspeed.sequence.layer.DistributedAttention. This implementation enforces the input shape
    to be standard [sl, bs, hc, hs] form. Any deviation from this shape will raise an error.

    The primary reason for the re-implementation is to make this less error prone, and remove what seemed like bugs in scenarios where batch size > 1 and when using different versions of
    flash attention each of which takes different input shape. Those should be handled by
    the actual attn implementation, and not by this module.

    This class then has been further adapted to work with HF Transformers' supported attention mechanism.

    Dimension annotation:
        bs   = bs
        hc   = head count
        hc_l = head count local
        hs   = head_size
        sl   = seqlen
        sl_l = seqlen local
        ws   = world_size
        em    = embedding (hidden size)
        em_l  = embedding (hidden size) local

    Arguments:
        attn: normal attention implementation from transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS
        seq_length_is_variable (bool): whether global seqlen may change between batches
        local_seq_length (int): local sequence length per GPU or None if seq_length_is_variable is True
        global_seq_length (int): actual sequence length or None if seq_length_is_variable is True
        batch_size (int): batch size
        attn_head_size (int): size of each attention head
        attn_head_count (int): total number of attention heads
        kv_head_count (int): total number of kv heads
        num_hidden_layers (int): total number of layers
        process_group (dist.ProcessGroup): Ulysses process group


    Extras:
        - set self.skip_all_but_last_attention_debug_mode to True to enable fast debug which will skip calling all core attn layers but the last one, it will produce garbage of course quality-wise.
    """

    def __init__(
        self,
        attn,
        batch_size: int,
        attn_head_count: int,
        attn_head_size: int,
        kv_head_count: int,
        num_hidden_layers: int,
        process_group: dist.ProcessGroup,
        seq_length_is_variable: bool = False,
        local_seq_length: int = None,
        global_seq_length: int = None,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.sp_rank = dist.get_rank(process_group)

        self.batch_size = batch_size
        self.seq_length_is_variable = seq_length_is_variable
        self.local_seq_length = local_seq_length
        self.global_seq_length = global_seq_length

        self.attn_head_size = attn_head_size
        self.attn_head_count = attn_head_count
        self.global_kv_head_count = kv_head_count

        self.num_hidden_layers = num_hidden_layers
        self.skip_all_but_last_attention_debug_mode = False
        self.rotating_layer_counter = 0  # used for dev work

        self.local_q_head_count = attn_head_count // self.world_size

        # if we have 4 kv heads and sp 8, we need to replicate kv heads 2x
        self.kv_replication_factor = self.world_size // kv_head_count
        if self.kv_replication_factor > 1:
            self.local_kv_head_count = 1
        else:
            self.local_kv_head_count = kv_head_count // self.world_size

        transformers_version_min = "4.51.3"
        transformers_version_have = importlib.metadata.version("transformers")
        if version.parse(transformers_version_have) < version.parse(transformers_version_min):
            raise ValueError(
                f"transformers>={transformers_version_min} is required, but you have transformers=={transformers_version_have}"
            )

        if self.attn_head_count % self.world_size != 0:
            raise ValueError(f"Attention head count {attn_head_count} is not divisible by SP size {self.world_size}")
        if not (self.global_kv_head_count % self.world_size == 0 or self.world_size % self.global_kv_head_count == 0):
            raise ValueError(
                f"KV attention head count {self.global_kv_head_count} is not divisible by SP size {self.world_size} or"
                " vice versa")

        if self.seq_length_is_variable:
            # the self.required_*_shape depending on the following will get updated in `forward`
            # use 1 as a placeholder for dim=0 to keep torch.Size happy
            local_seq_length = 1
            global_seq_length = 1

        # [sl_l bs hc hs]
        self.required_query_shape = torch.Size([local_seq_length, batch_size, attn_head_count, attn_head_size])
        self.required_key_value_shape = torch.Size([local_seq_length, batch_size, kv_head_count, attn_head_size])

        # [sl bs em_l]
        self.required_context_shape = torch.Size(
            [global_seq_length, batch_size, attn_head_size * attn_head_count // self.world_size])

    def _combine_local_sequences(self, query, key, value) -> Tuple[Tensor, Tensor, Tensor]:

        def combine_sequence(input, head_type):
            """
            expects inputs in shape: [sl_l bs hc hs]
            returns output in shape: [sl bs hc_l hs]

            local_head_count could be different for k,v vs q if it's not an MHA situation
            """
            if head_type == "q":
                local_head_count = self.local_q_head_count
            else:  # kv
                local_head_count = self.local_kv_head_count

                # MQA and some GQA cases:
                if self.kv_replication_factor > 1:
                    # local_head_count *= self.kv_replication_factor
                    # replicate heads to the kv_replication_factor on hc dimension [sl_l bs hc hs] - so dim=2
                    input = input.repeat_interleave(self.kv_replication_factor, dim=2)

            # [sl_l bs hc hs] -> [sl_l bs ws hc_l hs]
            input = input.reshape(
                [self.local_seq_length, self.batch_size, self.world_size, local_head_count, self.attn_head_size])

            input = rearrange(input, "sl_l bs ws hc_l hs -> ws sl_l bs hc_l hs").contiguous()

            output = _DimZeroAllToAll.apply(self.process_group, input)

            # [ws sl_l bs hc_l hs] -> [sl bs hc_l hs]
            output = output.reshape([self.global_seq_length, *output.shape[2:]]).contiguous()

            # [sl bs hc_l hs]
            return output

        return (
            combine_sequence(query, head_type="q"),
            combine_sequence(key, head_type="kv"),
            combine_sequence(value, head_type="kv"),
        )

    def _partition_global_sequence(self, input) -> Tensor:
        """
        expects input in shape:  [sl bs em_l]
        returns output in shape: [sl_l bs em]
        """

        # [sl bs em_l] -> [ws sl_l bs em_l]
        input = input.reshape([
            self.world_size,
            self.local_seq_length,
            self.batch_size,
            self.attn_head_size * self.attn_head_count // self.world_size,
        ]).contiguous()

        output = _DimZeroAllToAll.apply(self.process_group, input)
        output = rearrange(output, "ws sl_l bs em_l -> sl_l bs ws em_l")

        # [sl_l bs ws em_l] -> [sl_l bs em]
        output = output.reshape([*output.shape[:2], -1]).contiguous()

        # [sl_l bs em]
        return output

    def forward(
        self,
        module: torch.nn.Module,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            attention_mask (Tensor): Attention mask
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # HF incoming shapes are:
        # [batch_size, num_heads, seqlen, head_size]
        # UlyssesSPAttentionHF expects:
        # [seqlen, batch_size, num_heads, head_size]
        # print_rank0(f"{query.shape=}")
        # print_rank0(f"{key.shape=}")
        # print_rank0(f"{value.shape=}")
        # print_rank0(f"{self.required_input_shape=}")
        if self.seq_length_is_variable:
            current_local_seq_length = query.shape[2]
            self.local_seq_length = current_local_seq_length
            self.global_seq_length = current_local_seq_length * self.world_size
            # update the required seqlen shapes
            self.required_query_shape = torch.Size([self.local_seq_length] + list(self.required_query_shape)[1:])
            self.required_key_value_shape = torch.Size([self.local_seq_length] +
                                                       list(self.required_key_value_shape)[1:])
            self.required_context_shape = torch.Size([self.global_seq_length] + list(self.required_context_shape)[1:])

        # make the blocks contiguous as early as possible to minimize fragmentation
        query = rearrange(query, "bs hc sl hs -> sl bs hc hs")  # .contiguous()
        key = rearrange(key, "bs hc sl hs -> sl bs hc hs")  # .contiguous()
        value = rearrange(value, "bs hc sl hs -> sl bs hc hs")  # .contiguous()

        # core attn like FA2 expects an unsharded `position_ids` - without which packed samples
        # will return loss=nan.
        #
        # XXX: need to figure out if we can do the same for SDPA - as it doesn't require this and
        # wants an attention mask, so possibly doing this for FA2 only?
        #
        # Ideally we would passing the original unsharded position_ids - but we have no way to pass
        # it here as HF Transformers drops unexpected keys in `batch` - so either we need to stash
        # it somewhere in UlyssesSPDataLoaderAdapter and retrieve it here or we could gather it once
        # per batch and stash it inside `module` arg - I already have a machinery to figure out
        # which layer number is being called below in the skip_all_but_last_attention_debug_mode
        # code where rotating_layer_counter is used - so we could calculate it on the first layer
        # and re-use on the remaining layers
        if "position_ids" in kwargs:
            position_ids_list = [torch.empty_like(kwargs["position_ids"]) for _ in range(self.world_size)]
            dist.all_gather(position_ids_list, kwargs["position_ids"], group=self.process_group)
            kwargs["position_ids"] = torch.cat(position_ids_list, dim=1)

        # please don't remove the white-space vertical alignment in the error message
        assert query.shape == self.required_query_shape, (
            f"[{dist.get_rank()}]: query input tensor does not match the required shape\n            "
            f" {self.required_query_shape}:\n {query.shape=}\n   {key.shape=}\n {value.shape=}")
        assert key.shape == value.shape == self.required_key_value_shape, (
            f"[{dist.get_rank()}]: key or value input tensor does not match the required shape\n            "
            f" {self.required_key_value_shape}:\n {query.shape=}\n   {key.shape=}\n {value.shape=}")

        # expects: [sl_l bs hc hs]
        query_layer, key_layer, value_layer = self._combine_local_sequences(query, key, value)
        # returns: [sl bs hc_l hs]

        query_layer = rearrange(query_layer, "sl bs hc_l hs -> bs hc_l sl hs").contiguous()
        key_layer = rearrange(key_layer, "sl bs hc_l hs -> bs hc_l sl hs").contiguous()
        value_layer = rearrange(value_layer, "sl bs hc_l hs -> bs hc_l sl hs").contiguous()

        # crucial in the case of MQA and some GQA cases we need to fix `module.num_key_value_groups`
        # XXX: could move this somewhere to do it only once per run
        if self.kv_replication_factor > 1:
            module.num_key_value_groups = query_layer.size(-3) // key_layer.size(-3)

        if not self.skip_all_but_last_attention_debug_mode:
            # expects: [bs hc_l sl hs]
            context_layer, attn_weights = self.attn(module, query_layer, key_layer, value_layer, attention_mask, *args,
                                                    **kwargs)
            # returns [bs sl hc_l hs]
        else:
            # we need this hack during development in order to be able to check memory fitting w/o
            # waiting for 3h to compute 1.5M seqlen attention, because it's quadratic in dense
            # attention, so we skip all but the last core attention call - we want the last one to
            # still get the memory usage approximately close to the real memory usage. of course
            # the loss will be wrong when we do that.
            self.rotating_layer_counter = (self.rotating_layer_counter + 1) % self.num_hidden_layers
            # we detect the last layer by module counting since we know how many layers there are
            if self.rotating_layer_counter % self.num_hidden_layers == 0:
                # do the real pass
                context_layer, attn_weights = self.attn(module, query_layer, key_layer, value_layer, attention_mask,
                                                        *args, **kwargs)
            else:
                # this feeds bogus data of the right shape - good enough for quick debug
                context_layer = rearrange(query_layer, "bs hc_l sl ... -> bs sl hc_l ...")
                attn_weights = None

        # [bs sl hc_l hs] -> [sl bs hc_l hs]'
        context_layer = rearrange(context_layer, "bs sl ... -> sl bs ...")
        context_layer = context_layer.reshape([*context_layer.shape[:2], -1])

        assert (
            context_layer.shape == self.required_context_shape
        ), f"The context shape {context_layer.shape} is not of the expected shape {self.required_context_shape}"

        # expects: [sl bs em_l]
        output = self._partition_global_sequence(context_layer)
        # returns: [sl_l bs em]

        output = rearrange(output, "sl_l bs ... -> bs sl_l ...")

        output = output.reshape([*output.shape[:2], -1])

        # expects [bs sl em]
        return output, attn_weights

    @classmethod
    def register_with_transformers(
        cls,
        model_name_or_path,
        core_attn_implementation,
        sequence_parallel_size,
        micro_batch_size,
        seq_length=None,
        seq_length_is_variable=True,
        # deprecated
        max_length=None,
    ):
        """
        Register "ulysses" attn_implementation with HF transformers and return mpu (Megatron-LM-style parallel state groups object).
        If sequence_parallel_size==1 do nothing and return None.

        Args:
        - model_name_or_path (object or str): model object, or HF hub model name, or model's local path
        - core_attn_implementation (str): which attention to use: flash_attention_2 or flash_attention_3 or sdpa
        - sequence_parallel_size (int): sequence parallelism dimension (if 1 it's disabled)
        - micro_batch_size (int): micro batch size
        - seq_length (int): set this argument if the sequence length is fixed in all batches
        - seq_length_is_variable (bool): whether global seqlen may change between batches an optimization flag - the default is `True`
        - max_length (int): actual global sequence length - this argument is deprecated - use `seq_length` instead

        """
        if sequence_parallel_size == 1:
            return None

        if max_length is not None:
            logger.warning(
                "The 'max_length` argument is deprecated and will be eventually removed, please use `seq_length` instead"
            )
            if seq_length is None and max_length is not None:
                seq_length = max_length
        if not seq_length_is_variable and seq_length is None:
            raise ValueError(
                "Either `seq_length_is_variable` needs to be `True` or `seq_length` needs to be set to an integer value of the fixed batch size length."
            )

        from transformers import AutoConfig
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu

        mpu.initialize_sequence_parallel(sequence_parallel_size=sequence_parallel_size)

        from transformers import PreTrainedModel
        if hasattr(model_name_or_path, "config") or isinstance(model_name_or_path, PreTrainedModel):
            # we already have the model (or a PEFT wrapper with config attribute)
            hf_model_config = model_name_or_path.config
        else:
            # if we don't have the model yet at this stage
            hf_model_config = AutoConfig.from_pretrained(model_name_or_path)

        supported_attn_implementation = ["flash_attention_2", "flash_attention_3", "sdpa"]
        if core_attn_implementation not in supported_attn_implementation:
            # notes on the excluded ones:
            # - eager: The problem is that `eager` wants an attention_mask and it creates the wrong attention mask it seems if we don't provide one - it's possible that we could somehow solve this, but it's also unlikely someone will want to use the slow eager attention with sequence parallelism
            # - flex_attention: haven't tried

            raise ValueError(
                f"{core_attn_implementation} attn_implementation isn't currently supported by Ulysses sequence"
                f" parallelism. Set core_attn_implementation arg to one of {supported_attn_implementation}.")

        if core_attn_implementation not in ALL_ATTENTION_FUNCTIONS:
            raise ValueError(
                f"{core_attn_implementation} is not a valid attn_implementation. The choices are {ALL_ATTENTION_FUNCTIONS.valid_keys()}"
            )
        core_attn_function = ALL_ATTENTION_FUNCTIONS[core_attn_implementation]

        if seq_length_is_variable:
            local_seq_length = None
            global_seq_length = None
        else:
            local_seq_length = seq_length // mpu.get_sequence_parallel_world_size()
            global_seq_length = seq_length

        uattn = UlyssesSPAttentionHF(
            attn=core_attn_function,
            batch_size=micro_batch_size,
            attn_head_count=hf_model_config.num_attention_heads,
            attn_head_size=getattr(hf_model_config, "head_dim",
                                   hf_model_config.hidden_size // hf_model_config.num_attention_heads),
            kv_head_count=hf_model_config.num_key_value_heads,
            num_hidden_layers=hf_model_config.num_hidden_layers,
            process_group=mpu.get_sequence_parallel_group(),
            seq_length_is_variable=seq_length_is_variable,
            local_seq_length=local_seq_length,
            global_seq_length=global_seq_length,
        )

        def uattn_wrapper(
            module: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor,
            *args,
            **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

            # We are relaying on position_ids for SP to work so attention_mask has to be None
            # the problem is that HF currently doesn't know anything about ALL_ATTENTION_FUNCTIONS["ulysses"] so it doesn't make a special case like for "flash_attention_2" and "sdpa" and it creates an attention mask on the fly and it breaks things.
            attention_mask = None

            attn_output, attn_weights = uattn(
                module,
                query,
                key,
                value,
                attention_mask,
                # XXX: fixme
                *args,
                **kwargs,
            )
            return attn_output, attn_weights

        # We don't do: ALL_ATTENTION_FUNCTIONS.register("ulysses", uattn_wrapper)
        # The problem with this approach is that we are missing on all the special use cases in HF Transformers that do things like: if self.config._attn_implementation == "flash_attention_2": ...
        # So instead we hack `ALL_ATTENTION_FUNCTIONS` to override all existing keys with our implementation, since it only gets used at the point of calling the attention and that's what we want, all other code branches relying on the original core `attn_implementation` will still be executed. This is what we called "Being John Malkovich"
        for key in ALL_ATTENTION_FUNCTIONS.keys():
            ALL_ATTENTION_FUNCTIONS[key] = uattn_wrapper

        return mpu


class UlyssesSPDataLoaderAdapter:

    def __init__(
        self,
        dl: DataLoader,
        sp_rank: int,
        sp_group,
        sp_world_size,
        device,
    ):
        """
        This a DataLoader adapter which wraps around any existing DataLoader. It is used in conjunction with Ulysses to perform batch sharding on the sequence dimension.

        It gathers 1 sample from each participating rank, using the DL it wraps, then shards each of them and sends back to the ranks. So that when dl->iter->next is called, we end up with:
        - rank 0: getting batch 0 shard 0
        - rank 1: getting batch 0 shard 1
        ...
        - rank n: getting batch 0 shard n
        which is used to compute the batch (from rank0) using all SP ranks.

        When the next iteration starts and dl->iter->next is called, we end up with:
        - rank 0: getting batch 1 shard 0
        - rank 1: getting batch 1 shard 1
        ...
        - rank n: getting batch 1 shard n
        which is used to compute a second batch (from rank1) using all SP ranks.

        This continues until SP iterations are performed. At this point we need to get more data and so the above repeats.

        The key thing to understand is that all SP ranks participate in processing a single DL sample. So instead of normal DataParallel we perform a sort of SP over DP.

        When SP number of iterations is completed it's an equivalent of performing a single iteration with normal DP.

        If more tokens need to be consumed per step use the gradient accumulation feature.

        Ulysses expects the following dict keys in each DL batch (`dl->iter->next`):
        - `input_ids`
        - `position_ids`
        - `labels`

        Additional entries can be present.

        The tensors are expected to be of shape: `[batch_size, seqlen, ...]`

        The sharding happens on the seqlen (1st) dimension for all tensors in the batch, any non-tensor entries get copied to all ranks.

        `attention_mask` isn't used by Ulysses, because it's typically too large when it's 4D, and position_ids is just 1D, therefore it's much much smaller and consumes little GPU memory.

        Arguments:
        - `dl`: an existing DataLoader object to wrap
        - `sp_rank`: SP rank
        - `sp_group`: SP group
        - `sp_world_size`: SP world size
        - `device`: cuda device

        Returns:
            Another DataLoader object
        """

        self.dl = dl
        self.sp_rank = sp_rank
        self.sp_group = sp_group
        self.sp_world_size = sp_world_size
        self.device = device

        self.iter = iter(dl)
        self.micro_batches: list[Any] = []

    def __len__(self):
        return len(self.dl) * self.sp_world_size

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.micro_batches) == 0:
            self.refill()

        return self.micro_batches.pop(0)

    def refill(self):
        # reset the iterator if StopIteration arrives, and re-raise it to allow multiple epochs to run
        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dl)
            raise StopIteration
        micro_batches = defaultdict(dict)
        # XXX: replace with more efficient all-to-all?

        # we have batches of variable seqlen so in order to do all_gather on batches - we need to know the exact length of each tensor on each rank
        seqlen = torch.tensor(batch["input_ids"].shape[1], dtype=torch.int64, device=self.device)
        seqlens = [torch.zeros(1, dtype=torch.int64, device=self.device) for _ in range(self.sp_world_size)]
        dist.all_gather(seqlens, seqlen, group=self.sp_group)
        seqlens = [x[0].item() for x in seqlens]

        for k in batch.keys():
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(self.device)
                if seqlen != batch[k].shape[1]:
                    raise ValueError(
                        f"{k}'s shape {batch[k].shape} must match input_ids's shape {batch['input_ids'].shape}")
                with torch.no_grad():
                    tensor_list = [
                        torch.zeros((batch[k].shape[0], seqlens[i]), dtype=batch[k].dtype, device=batch[k].device)
                        for i in range(self.sp_world_size)
                    ]
                    dist.all_gather(tensor_list, batch[k], group=self.sp_group)
            else:
                tensor_list = [None for _ in range(self.sp_world_size)]
                dist.all_gather_object(tensor_list, batch[k], group=self.sp_group)

            for rank, tensor in enumerate(tensor_list):
                micro_batches[rank][k] = tensor

        del tensor_list
        del batch

        for batch in micro_batches.values():
            seq_length = len(batch["input_ids"][0])

            if seq_length % self.sp_world_size != 0:
                raise ValueError(f"batch's seqlen={seq_length} isn't divisible by sp-size={self.sp_world_size}")
            chunk_len = seq_length // self.sp_world_size

            # because we have to gather logits from all sp ranks we have to do the loss function ourselves
            # therefore remove labels to avoid an attempt to calculate loss by transformers
            labels = batch.pop("labels")
            labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
            batch["shift_labels"] = labels[..., 1:].contiguous()
            # free up temp memory
            del labels

            # batch sharding
            for k in batch.keys():
                # leave non-tensors alone
                if not torch.is_tensor(batch[k]):
                    continue
                # at seqlen>10M and 32+ gpus this can take GBs of memory so keep the prefill buffer on cpu
                batch[k] = batch[k][:, chunk_len * self.sp_rank:chunk_len * (self.sp_rank + 1)].cpu()

            self.micro_batches.append(batch)


def sequence_tiled_compute(
    fn,
    seqlen,
    shards,
    kwargs_to_shard,
    kwargs_to_pass,
    grad_requiring_tensor_key,
    compute_params=None,
    output_unshard_dimension=1,
    output_reduction="mean",
):
    """
    This is a wrapper for SequenceTiledCompute which we need since torch.autograd.Function can't work with dicts of tensors (in backward it has to return a grad value and not a dict that may have a non-None grad value). It's also useful for setting default values which we can't do either in torch.autograd.Function.

    Args:
    - `fn`: the function to call on sharded inputs
    - `seqlen`: total seqlen of the seqlen dimension
    - `shards`: how many shards to use
    - `kwargs_to_shard`: this dict will be passed to `fn` as `**kwargs` after sharding on seqlen dimension
    - `kwargs_to_pass`: this dict will be passed to `fn` as is, as `**kwargs`
    - `grad_requiring_tensor_key`: which main key requires grads
    - `compute_params`: a list of weights engaged in the compute. Default: `None` (only needed when using DeepSpeed ZeRO)
    - `output_reduction`: None, "mean" or "sum": Default: "mean"
    - `output_unshard_dimension`: the dimension to concat the outputs on: Default: 1 (seqlen dim)

    Returns:
    - unsharded output with an optional reduction applied, depending on the `output_reduction` value:
        `None` - return the unsharded output tensor
        `"mean"` - apply mean
        `"sum"` - apply sum

    Please note that this implementation doesn't require DeepSpeed and can work without it. `compute_params` can remain `None` in such a case.

    """
    args_to_shard = kwargs_to_shard.values()
    keys_to_shard = list(kwargs_to_shard.keys())
    args_to_pass = kwargs_to_pass.values()
    keys_to_pass = list(kwargs_to_pass.keys())

    return SequenceTiledCompute.apply(
        fn,
        seqlen,
        shards,
        keys_to_shard,
        keys_to_pass,
        grad_requiring_tensor_key,
        compute_params,
        output_unshard_dimension,
        output_reduction,
        *args_to_shard,
        *args_to_pass,
    )


class SequenceTiledCompute(torch.autograd.Function):
    """
    A generic autograd function to perform a tiled compute.

    Please note this module re-computes `forward` in the `backward`. So the `forward` occurs twice each iteration. And if you're using activation checkpointing it then occurs trice.

    Please note that this implementation doesn't require DeepSpeed and can work without it. `compute_params` can remain `None` in such a case.

    For an easier to understand example see TiledMLP - which is the same as this autograd function but without the generalization code.
    """

    @staticmethod
    def forward(
        ctx,
        fn,
        seqlen,
        shards,
        keys_to_shard,
        keys_to_pass,
        grad_requiring_tensor_key,
        compute_params,
        output_unshard_dimension,
        output_reduction,
        *args,
    ) -> torch.Tensor:
        """
        for args and return values see `sequence_tiled_compute`'s doc

        Currently we assume that all kwargs_to_shard values have a shape of `[bs, seqlen, ...]` and we shard on seqlen dimension
        """
        ctx.fn = fn
        ctx.seqlen = seqlen
        ctx.shards = shards
        ctx.grad_requiring_tensor_key = grad_requiring_tensor_key
        ctx.compute_params = [p for p in compute_params if p.requires_grad]
        ctx.output_unshard_dimension = output_unshard_dimension
        ctx.output_reduction = output_reduction

        with torch.no_grad():
            args = list(args)
            ctx.total_args = len(args)
            ctx.grad_requiring_tensor_key_index = (keys_to_shard + keys_to_pass).index(grad_requiring_tensor_key)

            kwargs_to_shard = {k: args.pop(0) for k in keys_to_shard}
            kwargs_to_pass = {k: args.pop(0) for k in keys_to_pass}
            ctx.kwargs_to_shard = kwargs_to_shard
            ctx.kwargs_to_pass = kwargs_to_pass

        with torch.no_grad():
            shard_step = math.ceil(seqlen / shards)
            output_shards = []

            for i in range(shards):
                output = fn(
                    **{
                        k: v[:, i * shard_step:(i + 1) * shard_step]
                        for k, v in kwargs_to_shard.items()
                    },
                    **kwargs_to_pass,
                )
                output_shards.append(output)

            if output_unshard_dimension == 0:
                # this is just the shape=[1] loss use-case, not sure if it's generic enough
                output_unsharded = torch.cat([l.unsqueeze(0) for l in output_shards], dim=output_unshard_dimension)
            else:
                output_unsharded = torch.cat(output_shards, dim=output_unshard_dimension)  # .clone().detach()

            if output_reduction is None:
                return output_unsharded
            elif output_reduction == "mean":
                return output_unsharded.mean()
            elif output_reduction == "sum":
                return output_unsharded.sum()
            else:
                raise ValueError(f"unknown value {output_reduction}: valid values are: none/mean/sum")

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        fn = ctx.fn
        shards = ctx.shards
        kwargs_to_shard = ctx.kwargs_to_shard
        kwargs_to_pass = ctx.kwargs_to_pass
        output_reduction = ctx.output_reduction

        grad_requiring_tensor_key = ctx.grad_requiring_tensor_key
        grad_requiring_tensor_key_index = ctx.grad_requiring_tensor_key_index
        compute_params = ctx.compute_params
        output_unshard_dimension = ctx.output_unshard_dimension
        grad_requiring_tensor = kwargs_to_shard[grad_requiring_tensor_key]

        grad_requiring_tensor_requires_grad = grad_requiring_tensor.requires_grad
        grad_requiring_tensor = grad_requiring_tensor.detach()
        # detach() unsets `grad_requiring_tensor.requires_grad`, so restore it
        grad_requiring_tensor.requires_grad_(grad_requiring_tensor_requires_grad)

        incoming_grad = grads[0]
        # since we perform a reduction of outputs that doesn't get included in `autograd.backward` below we need to pre-adjust the incoming gradient. in the case of "sum" the gradient is 1.0, in the case of "mean" it's 1.0/num_elements, which in this case is 1/shards.
        if output_reduction == "mean":
            incoming_grad /= shards

        if grad_requiring_tensor.shape[0] == 1:
            grad_requiring_tensor_grad = torch.zeros_like(grad_requiring_tensor)
        else:
            grad_requiring_tensor_grad = torch.empty_like(grad_requiring_tensor)

        kwargs_to_shard_shards = {k: list(torch.chunk(v, chunks=shards, dim=1)) for k, v in kwargs_to_shard.items()}

        for i in range(shards):
            # when fn involves one or more model weights deepspeed will normally push a grad to
            # reduce per sub-module call, so since we only want it to add a grad for the last
            # shard's call, we signal to ZeRO not to add new gradients to reduce until the last
            # shard when all gradients have been accumulated. An example for such a call is
            # `model.lm_head(hidden_states)`
            if compute_params is not None:
                if i + 1 < shards:
                    for param in compute_params:
                        param.ds_grad_is_ready = False
                else:
                    # last shard, can add the grad
                    for param in compute_params:
                        param.ds_grad_is_ready = True

            kwargs_to_shard_shard = {k: v[i] for k, v in kwargs_to_shard_shards.items()}
            grad_requiring_tensor_shard = kwargs_to_shard_shard[grad_requiring_tensor_key]

            grad_requiring_tensor_shard.requires_grad_(grad_requiring_tensor_requires_grad)

            # if seqlen is not exactly divisible by shards the last step will be shorter than shard_step
            shard_step = kwargs_to_shard_shards[grad_requiring_tensor_key][i].shape[1]
            shard_offset = i * kwargs_to_shard_shards[grad_requiring_tensor_key][0].shape[1]

            if grad_requiring_tensor.shape[0] == 1:
                # on narrow the shard's stride is unaffected with dim0==1 (bs) so we use the most efficient `narrow` alias:
                # this will enable gradual population of the pre-allocated
                # `grad_requiring_tensor_shard.grad` during `torch.autograd.backward` calls
                grad_requiring_tensor_shard.grad = grad_requiring_tensor_grad.narrow(
                    1, shard_offset, shard_step).view_as(grad_requiring_tensor_shard)

            with torch.enable_grad():
                output = fn(**kwargs_to_shard_shard, **kwargs_to_pass)

            if output_unshard_dimension == 0:
                # loss use-case
                torch.autograd.backward(output, incoming_grad)
            else:
                incoming_grad_shard = (incoming_grad.narrow(1, shard_offset,
                                                            shard_step).view_as(grad_requiring_tensor_shard))
                torch.autograd.backward(output, incoming_grad_shard)

            if grad_requiring_tensor.shape[0] > 1:
                # this is less efficient than dim0==1 (bs) use case, due to a required copy to fix
                # the stride and needing a bit more memory for one shard's grad, since
                # narrow(dim=1, ...) while dim0>1 will lead to:
                # UserWarning: grad and param do not obey the gradient layout contract. This is not an error, but may impair performance.
                # when backward is called.
                grad_requiring_tensor_grad.narrow(1, shard_offset,
                                                  shard_step).view_as(grad_requiring_tensor_shard).copy_(
                                                      grad_requiring_tensor_shard.grad)

        # positional args
        grad_outputs = [None] * 9
        # inject the grad for the position of forward input that is grad-requiring
        arg_outputs = [None] * ctx.total_args
        arg_outputs[grad_requiring_tensor_key_index] = grad_requiring_tensor_grad

        return tuple(grad_outputs + arg_outputs)


class TiledMLP(torch.autograd.Function):
    """
    Perform a tiled MLP computation to massively reduce memory usage needed to compute MLP when using very long sequence lengths.

    Please note this module re-computes `forward` in the `backward`. So the `forward` occurs twice each iteration. And if you're using activation checkpointing it then occurs trice.

    For a general tiled compute implementation that can handle any `forward` see `SequenceTiledCompute`.

    Args:
    - fn: the function to call on sharded inputs
    - `self`: the MLP nn.Module object
    - `x`: the input to MLP.forward (`hidden_states`)
    - `shards`: how many shards to use
    - compute_params: a list of weights engaged in the compute Default: `None` (only needed when using DeepSpeed ZeRO)

    Returns:
    - the computed `hidden_states`

    Here is an example that monkey patches HF Transformers' LLamaMLP:

    def tiled_mlp_forward(self, x):
        bs, seqlen, hidden = x.shape
        num_shards = math.ceil(seqlen / hidden)
        # to avoid deadlocks get all ranks to agree on the same num_shards by using the max value
        tensor = torch.tensor(num_shards, device=x.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        num_shards = tensor.item()
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

        # this needs to be done before the model is instantiated
        from transformers.models.llama import modeling_llama
        modeling_llama.LlamaMLP.forward = tiled_mlp_forward
    """

    @staticmethod
    def forward(
        ctx,
        fn,
        self,
        x,
        shards,
        compute_params,
    ) -> torch.Tensor:
        ctx.fn = fn
        ctx.self = self
        ctx.shards = shards
        ctx.compute_params = [p for p in compute_params if p.requires_grad]
        ctx.save_for_backward(x)

        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size] (moe experts)
        x_shards = list(torch.chunk(x, chunks=shards, dim=-2))
        with torch.no_grad():
            output_shards = [fn(self, x_shard) for x_shard in x_shards]
        output_unsharded = torch.cat(output_shards, dim=-2)

        return output_unsharded

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        fn = ctx.fn
        (x, ) = ctx.saved_tensors
        self = ctx.self
        shards = ctx.shards
        compute_params = ctx.compute_params

        x_requires_grad = x.requires_grad
        x = x.detach()
        # detach() unsets `x.requires_grad`, so restore it
        x.requires_grad_(x_requires_grad)

        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size] (moe experts)
        hidden_size = x.shape[-1]
        x_shape_orig = x.shape

        # flatten bs+seqlen to avoid having stride issues when narrowing into seqlen w/ bs>1
        x = x.view(-1, hidden_size)
        incoming_grad = grads[0].view(-1, hidden_size)
        x_grad = torch.zeros_like(x)

        x_shards = list(torch.chunk(x, chunks=shards, dim=0))

        for i, x_shard in enumerate(x_shards):
            # Tell deepspeed not to add a new grad to its ipg bucket until the last shard is run
            # XXX: DDP, FSDP will need something similar to make it work
            if compute_params is not None:
                if i + 1 < shards:
                    for param in compute_params:
                        param.ds_grad_is_ready = False
                else:
                    # last shard, can add the grad
                    for param in compute_params:
                        param.ds_grad_is_ready = True

            x_shard.requires_grad_(x_requires_grad)

            # if seqlen is not exactly divisible by shards the last step will be shorter than shard_step
            shard_step = x_shards[i].shape[0]
            shard_offset = i * x_shards[0].shape[0]

            x_shard.grad = x_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)
            incoming_grad_shard = incoming_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)
            with torch.enable_grad():
                output = fn(self, x_shard)
            torch.autograd.backward(output, incoming_grad_shard)

        # unflatten
        x_grad = x_grad.view(x_shape_orig)

        return (None, None, x_grad, None, None)


class TiledFusedLogitsLoss(torch.autograd.Function):
    """
    Perform a tiled loss computation while not manifesting a full logits tensor to massively reduce memory usage.

    Args:
    - fn: the function to call on sharded inputs
    - `self`: the lm_head module object, often it will be `unwrapped_model.model.lm_head`
    - `x`: the input (typically `hidden_states`) - which gets sharded
    - `y`: the target (typically `labels` or `shift_labels`) - which gets sharded.
    - `mask`: an optional mask. It will be not passed to the `fn` if set to `None`. If not-`None` it'll be sharded with `x` and `y`
    - `shards`: how many shards to use
    - compute_params: a list of weights engaged in the compute Default: `None` (only needed when using DeepSpeed ZeRO)
    - output_reduction: "mean" or "sum". If the unmasked elements in `x` are of different sizes in different shards, it's recommended to use "sum" instead of "mean" and perform the balanced mean to the output. This would be the case if `x` is not evenly divisible by `shards` or if the mask may lead to a different number of unmasked elements.

    Returns:
    - the computed `loss`

    Note, that since this autograd function is typically the last one in the call stack, it performs `backward` inside `forward` and compensates for `output_reduction` artificially. This removes the need to re-run `forward` a second time inside `backward`

    For a generic tiled compute implementation that can handle many other types of `forward` see `SequenceTiledCompute`.

    An example:

        def loss_fn(self, x, y):
            logits = self.lm_head(x)
            return self.cross_entropy_loss(logits.view(-1, self.vocab_size), y.view(-1))

        x = hidden_states
        y = shift_labels
        mask = None
        shards = 2
        compute_params = [self.lm_head.weight]
        output_reduction = "mean"
        loss = TiledFusedLogitsLoss.apply(
            loss_fn,
            self,
            x,
            y,
            mask,
            shards,
            compute_params,
            output_reduction,
        )

    """

    @staticmethod
    def forward(
        ctx,
        fn,
        self,
        x,
        y,
        mask,
        shards,
        compute_params,
        output_reduction,
    ) -> torch.Tensor:

        if output_reduction not in ["mean", "sum"]:
            raise ValueError(f'unknown reduction {output_reduction}: valid values are: "mean"/"sum"')
        if x.dim() < 2:
            raise ValueError("x must be at least 2D [batch_size, seq_len, ...]")
        if y.dim() < 2:
            raise ValueError("y must be at least 2D [batch_size, seq_len, ...]")
        if x.shape[:2] != y.shape[:2]:
            raise ValueError("x and y batch/seq dims must match")
        if mask is not None:
            if mask.dim() != 2:
                raise ValueError(f"mask must be 2D [batch_size, seq_len], but got {mask.dim()}")
            if mask.shape != x.shape[:2]:
                raise ValueError(f"mask shape must match x and y batch/seq")

        compute_params = [p for p in compute_params if p.requires_grad]

        x_requires_grad = x.requires_grad
        x = x.detach().requires_grad_(x_requires_grad)

        bs, seqlen = x.shape[:2]

        # flatten bs+seqlen to avoid having stride issues when narrowing into seqlen w/ bs>1
        x = x.view(-1, *x.shape[2:])
        y = y.view(-1, *y.shape[2:])
        if mask is not None:
            mask = mask.view(-1)
        incoming_grad = torch.tensor(1.0, dtype=x.dtype, device=x.device)

        # we are faking the incoming gradient, and since we perform a reduction outside of `autograd.backward` below we need to pre-adjust the incoming gradient. in the case of "sum" the gradient is 1.0, in the case of "mean" it's 1.0/num_elements, which in this case is 1/shards.
        if output_reduction == "mean":
            incoming_grad /= shards

        # XXX: deal with the use case of running in inference mode, where we don't need backward
        x_grad = torch.zeros_like(x) if x_requires_grad else None
        x_shards = list(torch.chunk(x, chunks=shards, dim=0))
        y_shards = list(torch.chunk(y, chunks=shards, dim=0))
        if mask is not None:
            mask_shards = list(torch.chunk(mask, chunks=shards, dim=0))

        output_shards = []
        for i, (x_shard, y_shard) in enumerate(zip(x_shards, y_shards)):
            # Tell deepspeed not to add a new grad to its ipg bucket until the last shard is run
            # XXX: DDP, FSDP will need something similar to make it work
            if compute_params is not None:
                if i + 1 < shards:
                    for param in compute_params:
                        param.ds_grad_is_ready = False
                else:
                    # last shard, can add the grad
                    for param in compute_params:
                        param.ds_grad_is_ready = True

            x_shard.requires_grad_(x_requires_grad)

            # if seqlen is not exactly divisible by shards the last step will be shorter than shard_step
            shard_step = x_shards[i].shape[0]
            shard_offset = i * x_shards[0].shape[0]

            args = (self, x_shard, y_shard)
            if mask is not None:
                args += (mask_shards[i], )
            if x_grad is not None:
                x_shard.grad = x_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)
                with torch.enable_grad():
                    output = fn(*args)
                    output_shards.append(output)
                torch.autograd.backward(output, incoming_grad)
            else:
                output = fn(*args)
                output_shards.append(output)

        output_unsharded = torch.cat([l.unsqueeze(0) for l in output_shards], dim=0)

        if output_reduction == "mean":
            output = output_unsharded.mean()
        elif output_reduction == "sum":
            output = output_unsharded.sum()

        # unflatten
        if x_grad is not None:
            x_grad = x_grad.view(bs, seqlen, *x_grad.shape[1:])
            ctx.save_for_backward(x_grad.detach())

        return output

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        (x_grad, ) = ctx.saved_tensors
        # grads[0] should normally be 1.0 as it should be coming from loss.backward()
        if grads[0] != 1.0:
            x_grad *= grads[0]
        return (None, None, x_grad, None, None, None, None, None, None)


class AutogradComputeMLP(torch.autograd.Function):
    """
    This is a simplified example to override the normal MLP via an autograd function - then tiling can be added - this simplified version was useful to detect a leak in Deepspeed, so let's keep it.

    Here is an example of performing the monkey patching on LlamaMLP

    def mlp_forward_new(self, x):

        def mlp_forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return AutogradComputeMLP.apply(mlp_forward, self, x)

    from transformers.models.llama import modeling_llama
    modeling_llama.LlamaMLP.forward = mlp_forward_new
    """

    @staticmethod
    def forward(
        ctx,
        fn,
        self,
        x,
    ) -> torch.Tensor:
        ctx.fn = fn
        ctx.self = self
        ctx.save_for_backward(x)

        with torch.no_grad():
            return fn(self, x)

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        fn = ctx.fn
        (x, ) = ctx.saved_tensors
        self = ctx.self

        x1 = x.detach()
        x1.requires_grad = x.requires_grad
        with torch.enable_grad():
            output = fn(self, x1)

        torch.autograd.backward(output, grads[0])
        return (None, None, x1.grad, None)


###########################################################
### below are older versions that some might still want ###
###########################################################


class TiledLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, loss_fn, logits, vocab_size, shift_labels, shards) -> torch.Tensor:
        """

        This is a memory efficient loss autograd function that takes the existing logits and performs loss calculation in shards.

        This one is an SFT-aware version, therefore it takes care of special cases where the whole shard is made of -100 labels and which requires then a special care.

        Note: logits seqlen dimension doesn't have to be divisible by shards, the last shard will be shorter than the rest. The calculating of the number of shards is in the example.

        Here is an example of using it:

        def loss(self, batch) -> torch.Tensor:
            batch = to_device(batch, self.device)
            shift_labels = batch.pop("shift_labels")
            outputs = self.model(**batch, use_cache=False)
            logits = outputs.logits

            if all((shift_labels == -100).squeeze()):
                # this is the case where all labels in a micro-batch are -100 (very common for SFT if the seqlen is short) - CE returns `nan` in this case, so we don't want to call loss and instead create a differentiable loss `0` which will also set all the grads to `0` in `backward` - the effect of this is akin to a perfect score where the model needs no adjustment since grads will be all zeros.
                loss = (logits.sum() * 0.0).float()

            num_shards: Any = "auto"
            if num_shards == "auto":
                # parameterize to about 1GB fp32 logits shards
                slice_size_in_gb = 1
                size_in_gb = logits.numel() * 4 / 2**30  # fp32
                # the sp shard's seqlen sp shard can be easily not divisible by the derived number of chunked loss shards, so we use the uppper ceiling and allow the last chunk to be shorter than the rest
                num_shards = math.ceil(size_in_gb / slice_size_in_gb)
                # print(f"derived {num_shards} shards for size {size_in_gb}GB")
            if num_shards > 1:
                # if shards == 1 this will lead to a higher memory usage then calling the normal loss function, so don't do that.
                loss = TiledLoss.apply(
                    self.model_unwrapped.loss_function,
                    logits,
                    self.model_unwrapped.config.vocab_size,
                    shift_labels,
                    num_shards,
                )
            else:
                loss = self.model_unwrapped.loss_function(
                    logits=logits,
                    labels=None,
                    vocab_size=self.model_unwrapped.config.vocab_size,
                    shift_labels=shift_labels,
                )

            return loss


        """
        ctx.save_for_backward(logits, shift_labels)
        ctx.loss_fn = loss_fn
        ctx.vocab_size = vocab_size
        ctx.shards = shards

        with torch.no_grad():
            seqlen = shift_labels.shape[1]
            shard_step = math.ceil(seqlen / shards)
            loss_shards = []
            total_good_items = 0

            # since -100s are ignored we have to perform a weighted average on each loss slice as each slice may contribute a different number of non- -100 labels
            # if seqlen / shards != 0 - the last chunk is just shorter than the rest but no data is ignored
            for i in range(shards):
                # XXX: here and everywhere don't make a copy, pass the slice or perhaps narrow/view?
                shift_labels_shard = shift_labels[:, i * shard_step:(i + 1) * shard_step]
                if all((shift_labels_shard == -100).squeeze()):
                    continue  # ignore this shard
                loss_shard = loss_fn(
                    logits=logits[:, i * shard_step:(i + 1) * shard_step, :],
                    labels=None,
                    vocab_size=vocab_size,
                    shift_labels=shift_labels_shard,
                )
                good_items = sum((shift_labels_shard != -100).squeeze())
                loss_shards.append(loss_shard * good_items)
                total_good_items += good_items
            total_loss = torch.cat([l.unsqueeze(0) for l in loss_shards], dim=0).sum()
            weighted_loss = total_loss / total_good_items

        return weighted_loss

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        logits, shift_labels = ctx.saved_tensors
        loss_fn = ctx.loss_fn
        vocab_size = ctx.vocab_size
        shards = ctx.shards

        grad = grads[0]
        logits_grad = torch.zeros_like(logits)
        logits_shards = list(torch.chunk(logits, chunks=shards, dim=1))
        shift_labels_shards = list(torch.chunk(shift_labels, chunks=shards, dim=1))

        # if seqlen is not exactly divisible by shards the last step will be shorter than shard_step
        shard_step = logits_shards[0].shape[1]
        for i in range(shards):
            logits_shard = logits_shards.pop(0)
            shift_labels_shard = shift_labels_shards.pop(0)

            shard_offset = i * shard_step
            # this will enable gradual population of the pre-allocated `logits_shard.grad` during `torch.autograd.backward` calls
            logits_shard.grad = (logits_grad.narrow(1, shard_offset, shard_step).view_as(logits_shard))

            with torch.enable_grad():
                if all((shift_labels_shard == -100).squeeze()):
                    # fake loss calculation, since CE will return nan, but grads will be set
                    # a normal loss_fn upcasts logits to float so match it
                    loss_shard = (logits_shard.sum() * 0.0).float()
                else:
                    loss_shard = loss_fn(
                        logits=logits_shard.requires_grad_(),
                        labels=None,
                        vocab_size=vocab_size,
                        shift_labels=shift_labels_shard,
                    )

            torch.autograd.backward(loss_shard, grad)

        logits_grad /= shards

        # only logits (2nd arg) needs grads
        return None, logits_grad, None, None, None


# This is the original implementation/integration of UlyssesSP into the training loop, which was superseded by using UlyssesSPDataLoaderAdapter which did all the sharding and pull the shards from the DL
#
# There are 2 issues with this implementation:
# - it's complex and difficult to integrate into various training scenarios
# - it could lead to a huge number of tokens per step - e.g. 32 ranks of 15M seqlen -> 0.5B token step - which is very wasteful
#
# Therefore if you want to use UlyssesSP via UlyssesSPFwdLossBwdWithLogits with its fwd/loss/bwd for those don't want to use UlyssesSPDataLoaderAdapter - here is how it should be installed into the sub-trainer class:
# class SFTTrainer(Trainer):
# def sp_fwd_loss_bwd(self, batch) -> torch.Tensor:
#     batch = to_device(batch, self.device)
#
#     from arctic_training.trainer.trainer import UlyssesAttentionHFFwdLossBwdWithLogits
#     ulysses = UlyssesAttentionHFFwdLossBwdWithLogits(
#         model=self.model,
#         model_unwrapped=self.model_unwrapped,
#         device=self.device,
#         num_loss_logit_shards="auto",
#     )
#     return ulysses.sp_fwd_loss_bwd(batch)


class UlyssesSPFwdLossBwdWithLogits:

    def __init__(self, model, model_unwrapped, device, num_loss_logit_shards="auto", **kwargs):

        self.model = model
        self.model_unwrapped = model_unwrapped
        self.device = device
        self.num_loss_logit_shards = num_loss_logit_shards
        self.kwargs = kwargs

        from deepspeed.utils import groups

        self.sp_group = groups._get_sequence_parallel_group()
        self.sp_world_size = groups._get_sequence_parallel_world_size()
        self.sp_rank = groups._get_sequence_parallel_rank()

    def sp_fwd_loss_bwd(self, batch) -> torch.Tensor:

        see_memory_usage("entered sp_fwd_loss_bwd", force=True)

        # ensure shapes are correct
        if not (batch["input_ids"].shape == batch["position_ids"].shape == batch["labels"].shape):
            raise ValueError(
                f'Borked batch {batch["input_ids"].shape=} != {batch["position_ids"].shape=} !='
                f' {batch["labels"].shape=}) in DataLoader->iter->next, cannot continue with Ulysses Sequence'
                " parallelism")

        # gather DL batches into super-batches
        # Important: DL doesn't always yield max_length batches. Different ranks may have different seqlen and each could be <= max_length (but always divisible by 256)

        micro_batches: list[Any] = defaultdict(dict)
        # Efficient gathering of batch inputs across ranks:
        # The problem is that our DL doesn't guarantee the same seqlen on all ranks and may give, 3x 1024 and 1x 768 on 4 gpus for max_length 1024. so 3 options we have to be able to gather batches are:
        # 1. use all_gather_object - which allows different shapes - but potentially introducing an undesired overhead - 2x pickle calls
        # 2. use all_gather and change DL pad to make sure that all ranks always get the same input shape - this creates its own overhead since if we say have ranks with seqlen 512, 768, 1024, 1024 - now we will need to process 4x 1024 seqlens
        # 3. use all_gather and post gathering truncate tensors to their intended length - another overhead of allocating and truncating tensors
        # using approach (1) for now but might want to benchmark later the other 2 approaches

        # XXX: if using all_gather_object we can gather the whole batch at once and not per-key! so can drop the loop for that approach

        # we have batches of variable seqlen so in order to do all_gather on batches - we need to know the exact length of each tensor on each rank
        seqlen = torch.tensor(batch["input_ids"].shape[1], dtype=torch.int64, device=self.device)
        # print(seqlen)
        seqlens = [torch.zeros(1, dtype=torch.int64, device=self.device) for _ in range(self.sp_world_size)]
        dist.all_gather(seqlens, seqlen, group=self.sp_group)
        seqlens = [x[0].item() for x in seqlens]

        for k in batch.keys():
            batch[k] = batch[k].to(self.device)
            with torch.no_grad():
                tensor_list = [
                    torch.zeros((batch[k].shape[0], seqlens[i]), dtype=batch[k].dtype, device=batch[k].device)
                    for i in range(self.sp_world_size)
                ]
                dist.all_gather(tensor_list, batch[k], group=self.sp_group)

                # gathering on the data dimension
                # will be concatenating and later splitting again for the more general case
                # batch[k] = torch.cat(tensor_list, dim=1)
                for rank, tensor in enumerate(tensor_list):
                    micro_batches[rank][k] = tensor

        del tensor_list
        del batch

        # we need to chunk twice - each time on SP size level
        # - the first time is because we artificially made the seqlen SP-times longer
        # - the second time is because of the Ulysses algorithm

        see_memory_usage("after gathering", force=False)

        self.model.set_gradient_accumulation_boundary(False)

        losses = []
        for sub_step_id in range(self.sp_world_size):
            batch = micro_batches[sub_step_id]
            seq_length = len(batch["input_ids"][0])

            if seq_length % self.sp_world_size != 0:
                raise ValueError(
                    f"{sub_step_id=}: batch's seqlen={seq_length} isn't divisible by sp-size={self.sp_world_size}")
            chunk_len = int(seq_length / self.sp_world_size)

            # to enable the correct mean calculation across shards before sharding the micro batch:
            # 1. count the number of non- `-100`` elements per shard
            # 2. and subtract one more element because of label shifting
            non_skipped_items = {}
            for rank in range(self.sp_world_size):
                non_skipped = (batch["labels"][:, chunk_len * rank:chunk_len * (rank + 1)] != -100).sum().item()
                if non_skipped > 1:
                    non_skipped -= 1
                non_skipped_items[rank] = non_skipped

            # because we have to gather logits from all sp ranks we have to do the loss function ourselves
            # therefore remove labels to avoid an attempt to calculate loss by transformers
            labels = batch.pop("labels")
            labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
            batch["shift_labels"] = labels[..., 1:].contiguous()
            # free up temp memory
            del labels

            # batch sharding
            for k in batch.keys():
                batch[k] = batch[k][:, chunk_len * self.sp_rank:chunk_len * (self.sp_rank + 1)].to(self.device)

            shift_labels = batch.pop("shift_labels")

            outputs = self.forward(batch)
            loss = self.compute_loss(labels=None, shift_labels=shift_labels)

            # free up temp mem (e.g. outputs.logits are huge)
            del outputs

            # differentiable loss aggregation across ranks
            losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=self.sp_group)

            # since each shard may have a variable number of skipped elemented - need to calculate a weighted mean depending on each rank's contribution - this will also take care of loss=0 when all elements are -100 in a shard
            # XXX: not expecting a total of 0-non-skipped items for div
            loss = sum(losses_per_rank[rank] * non_skipped_items[rank]
                       for rank in range(self.sp_world_size)) / sum(non_skipped_items.values())

            self.backward()

            losses.append(loss.detach().item())

        self.model.set_gradient_accumulation_boundary(True)

        # for per-iteration reporting
        if len(losses) == 0:
            loss = float("nan")
        else:
            loss = sum(losses) / len(losses)

        return loss

    def forward(self, batch):
        # critical: the labels shouldn't be in batch
        outputs = self.model(**batch, use_cache=False)
        self.logits = outputs.logits
        return outputs

    def compute_loss(self, labels, shift_labels):
        if all((shift_labels == -100).squeeze()):
            # this is the case where all labels in a micro-batch are -100 (very common for SFT) - CE returns `nan` in this case, so we don't want to call loss and instead create a differentiable loss `0` which will also set all the grads to `0` in `backward` - the effect of this is akin to a perfect score where the model needs no adjustment since grads will be all zeros.
            # XXX: should this be float and not the original dtype?
            loss = (self.logits.sum() * 0.0).float()
        else:
            if self.num_loss_logit_shards == "auto":
                # parameterize to about 1GB fp32 logits shards
                slice_size_in_gb = 1  # XXX: make configurable?
                size_in_gb = self.logits.numel() * 4 / 2**30  # fp32
                # the sp shard's seqlen sp shard can be easily not divisible by the derived number of chunked loss shards, so we use the uppper ceiling and allow the last chunk to be shorter than the rest
                self.num_loss_logit_shards = math.ceil(size_in_gb / slice_size_in_gb)
                # print(f"derived {self.num_loss_logit_shards} shards for size {size_in_gb}GB")
            if self.num_loss_logit_shards > 1:
                loss = TiledLoss.apply(
                    self.model_unwrapped.loss_function,
                    self.logits,
                    self.model_unwrapped.config.vocab_size,
                    shift_labels,
                    self.num_loss_logit_shards,
                )
            else:
                # XXX: for some reason this fails with zero1
                loss = self.model_unwrapped.loss_function(
                    logits=self.logits,
                    labels=None,
                    vocab_size=self.model_unwrapped.config.vocab_size,
                    shift_labels=shift_labels,
                )

        self.loss = loss
        return loss

    def backward(self):
        self.model.backward(self.loss)
