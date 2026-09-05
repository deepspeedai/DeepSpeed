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

from collections import defaultdict, deque
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.sequence.layer import _DimZeroAllToAll
from deepspeed.runtime.sequence_parallel.ulysses_linear_attention import (
    CURRENT_FORWARD_METADATA,
    create_qwen35_linear_attention_registration,
    position_ids_to_document_ids,
    position_ids_to_packed_cu_seqlens,
    prepare_qwen35_linear_attention_cp,
)
from deepspeed.utils.logging import logger
from einops import rearrange
from packaging import version
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Any, Tuple
import deepspeed.comm as dist
from importlib import metadata as importlib_metadata
import inspect
import math
import re
import torch
import torch.distributed.nn

_ACTIVE_LINEAR_ATTENTION_REGISTRATION = None
_ACTIVE_ATTENTION_REGISTRATIONS = {}
_PACKED_FLEX_TRANSFORMERS_MIN_VERSION = "4.57.0"


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
        disable_in_eval (bool): whether to disable sequence parallelism during evaluation (default: False).
            When True, SP operations are bypassed during eval to avoid potential issues with frameworks
            like HF Trainer that may run eval with different data distribution.


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
        disable_in_eval: bool = False,
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
        self.disable_in_eval = disable_in_eval

        self.attn_head_size = attn_head_size
        self.attn_head_count = attn_head_count
        self.global_kv_head_count = kv_head_count

        self.num_hidden_layers = num_hidden_layers
        self.skip_all_but_last_attention_debug_mode = False
        self.rotating_layer_counter = 0  # used for dev work

        self.core_attn_implementation = None  # set by register_with_transformers
        self._flex_block_mask_cached = None  # cached BlockMask for flex_attention
        self._flex_block_mask_cache_key = None
        self._cached_local_position_ids = None
        self._cached_local_position_ids_version = None
        self._cached_full_position_ids = None

        self.local_q_head_count = attn_head_count // self.world_size

        # if we have 4 kv heads and sp 8, we need to replicate kv heads 2x
        self.kv_replication_factor = self.world_size // kv_head_count
        if self.kv_replication_factor > 1:
            self.local_kv_head_count = 1
        else:
            self.local_kv_head_count = kv_head_count // self.world_size

        transformers_version_min = "4.51.3"
        transformers_version_have = importlib_metadata.version("transformers")
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

    def _get_full_position_ids(self, local_position_ids: torch.LongTensor) -> torch.LongTensor:
        metadata = CURRENT_FORWARD_METADATA.get()
        if metadata is not None:
            return metadata.full_position_ids

        current_version = getattr(local_position_ids, "_version", None)
        if (self._cached_local_position_ids is local_position_ids
                and self._cached_local_position_ids_version == current_version):
            return self._cached_full_position_ids

        position_id_shards = [torch.empty_like(local_position_ids) for _ in range(self.world_size)]
        dist.all_gather(position_id_shards, local_position_ids.contiguous(), group=self.process_group)
        full_position_ids = torch.cat(position_id_shards, dim=-1)
        self._cached_local_position_ids = local_position_ids
        self._cached_local_position_ids_version = current_version
        self._cached_full_position_ids = full_position_ids
        return full_position_ids

    @staticmethod
    def _validate_flex_attention_pattern(module: torch.nn.Module, kwargs: dict) -> None:
        """Reject mask patterns that cannot be reconstructed after sequence all-to-all."""
        active_pattern_keys = []
        for key in ("sliding_window", "window_size", "attention_chunk_size", "chunk_size", "prefix_length"):
            value = kwargs.get(key)
            if value not in (None, False, 0):
                active_pattern_keys.append(key)

        config = getattr(module, "config", None)
        layer_types = getattr(config, "layer_types", None) or ()
        layer_idx = getattr(module, "layer_idx", None)
        layer_type = None
        if layer_idx is not None and 0 <= layer_idx < len(layer_types):
            layer_type = str(layer_types[layer_idx]).lower()
        if layer_type is not None and any(marker in layer_type for marker in ("sliding", "chunk", "prefix")):
            active_pattern_keys.append(f"layer_types[{layer_idx}]={layer_type}")
        elif not layer_types and config is not None:
            for key in ("sliding_window", "attention_chunk_size", "chunk_size"):
                value = getattr(config, key, None)
                if value not in (None, False, 0):
                    active_pattern_keys.append(f"config.{key}")

        if getattr(module, "is_causal", True) is False:
            active_pattern_keys.append("is_causal=False")
        if active_pattern_keys:
            patterns = ", ".join(active_pattern_keys)
            raise RuntimeError(
                "Ulysses flex_attention currently reconstructs only standard causal and packed document-causal "
                f"masks; unsupported attention pattern(s): {patterns}.")

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

        # Skip SP operations during eval if disable_in_eval is True
        # This avoids issues with frameworks like HF Trainer that may run eval with different data distribution
        if not module.training and self.disable_in_eval:
            return self.attn(module, query, key, value, attention_mask, *args, **kwargs)

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

        # All attention backends need unsharded position_ids after the all-to-all.
        # FA2 uses them for packed-sequence detection (flash_varlen_fn), sdpa/flex_attention
        # need them to be monotonically increasing so causal masking works correctly.
        # UlyssesSPDataLoaderAdapter ensures position_ids are in the batch before sharding,
        # so after gathering here they reconstruct to the correct global positions.
        assert "position_ids" in kwargs, (
            "Ulysses SP requires position_ids in every forward() call so that after all_gather "
            "causal masking works correctly. Without them each rank generates local [0..chunk_len-1] "
            "positions which, after gathering, look like packed sequences and break attention. "
            "For non-packed sequences: position_ids = torch.arange(seq_len) per sample. "
            "For packed sequences: position_ids must reset at document boundaries. "
            "Ensure your data collator or UlyssesSPDataLoaderAdapter includes position_ids.")
        full_position_ids = self._get_full_position_ids(kwargs["position_ids"])
        kwargs["position_ids"] = full_position_ids
        metadata = CURRENT_FORWARD_METADATA.get()
        if metadata is not None:
            global_cu_seqlens = metadata.global_cu_seqlens
            document_ids = metadata.document_ids
            is_packed = metadata.is_packed
            global_cu_seqlens_cpu = metadata.global_cu_seqlens_cpu
        else:
            document_ids = position_ids_to_document_ids(full_position_ids)
            global_cu_seqlens = position_ids_to_packed_cu_seqlens(full_position_ids)
            is_packed = bool((document_ids[:, 1:] != document_ids[:, :-1]).any().item())
            global_cu_seqlens_cpu = global_cu_seqlens.detach().cpu()
        if self.core_attn_implementation == "sdpa" and is_packed:
            raise RuntimeError(
                "Packed Ulysses SP is not supported with SDPA because SDPA requires a quadratic document-aware "
                "attention mask. Use flash_attention_2/3 or flex_attention for packed training.")

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

        # The model built the BlockMask for the local shard. Rebuild it for the full sequence
        # and preserve packed-document boundaries.
        if self.core_attn_implementation == "flex_attention":
            self._validate_flex_attention_pattern(module, kwargs)
            try:
                from torch.nn.attention.flex_attention import create_block_mask
            except (ImportError, AttributeError) as exc:
                raise RuntimeError("Ulysses flex_attention requires a PyTorch version exposing "
                                   "torch.nn.attention.flex_attention.create_block_mask.") from exc
            seq_len = query_layer.shape[2]
            batch_size = query_layer.shape[0]
            cache_key = (
                batch_size,
                seq_len,
                query_layer.device.type,
                query_layer.device.index,
                tuple(global_cu_seqlens_cpu.tolist()),
            )

            if self._flex_block_mask_cache_key != cache_key:
                if is_packed:
                    transformers_version_have = importlib_metadata.version("transformers")
                    if version.parse(transformers_version_have) < version.parse(_PACKED_FLEX_TRANSFORMERS_MIN_VERSION):
                        raise RuntimeError("Packed Ulysses flex_attention requires "
                                           f"transformers>={_PACKED_FLEX_TRANSFORMERS_MIN_VERSION}; found "
                                           f"transformers=={transformers_version_have}.")
                    try:
                        from transformers.masking_utils import create_causal_mask
                    except ImportError as exc:
                        raise RuntimeError(
                            "Packed Ulysses flex_attention requires transformers.masking_utils.create_causal_mask "
                            f"from transformers>={_PACKED_FLEX_TRANSFORMERS_MIN_VERSION}.") from exc

                    document_ids = document_ids.to(device=query_layer.device, dtype=torch.long).contiguous()
                    token_indices = torch.arange(seq_len, device=query_layer.device, dtype=torch.long)
                    token_indices = token_indices.unsqueeze(0).expand(batch_size, -1)
                    document_starts = torch.ones_like(document_ids, dtype=torch.bool)
                    document_starts[:, 1:] = document_ids[:, 1:] != document_ids[:, :-1]
                    latest_start = torch.where(document_starts, token_indices,
                                               torch.zeros_like(token_indices)).cummax(dim=1).values
                    canonical_position_ids = token_indices - latest_start

                    mask_kwargs = {
                        "config": module.config,
                        "attention_mask": None,
                        "past_key_values": None,
                        "position_ids": canonical_position_ids,
                    }
                    mask_parameters = inspect.signature(create_causal_mask).parameters
                    placeholder = query_layer.new_empty((batch_size, seq_len, 1))
                    if "input_embeds" in mask_parameters:
                        mask_kwargs["input_embeds"] = placeholder
                    elif "inputs_embeds" in mask_parameters:
                        mask_kwargs["inputs_embeds"] = placeholder
                    else:
                        raise RuntimeError(
                            "Unsupported transformers.masking_utils.create_causal_mask signature: missing "
                            "input_embeds/inputs_embeds.")
                    if "cache_position" in mask_parameters:
                        mask_kwargs["cache_position"] = torch.arange(
                            seq_len,
                            device=query_layer.device,
                            dtype=torch.long,
                        )
                    self._flex_block_mask_cached = create_causal_mask(**mask_kwargs)
                    if self._flex_block_mask_cached is None:
                        raise RuntimeError("Transformers did not construct a packed flex_attention BlockMask.")
                else:

                    def causal_mask(batch_idx, head_idx, q_idx, kv_idx):
                        return q_idx >= kv_idx

                    self._flex_block_mask_cached = create_block_mask(
                        mask_mod=causal_mask,
                        B=batch_size,
                        H=None,
                        Q_LEN=seq_len,
                        KV_LEN=seq_len,
                        device=query_layer.device,
                    )
            self._flex_block_mask_cache_key = cache_key

            attention_mask = self._flex_block_mask_cached

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
        disable_in_eval=False,
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
        - disable_in_eval (bool): whether to disable sequence parallelism during evaluation (default: False).
            When True, SP operations are bypassed during eval to avoid issues with frameworks
            like HF Trainer that may run eval with different data distribution.
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

        from transformers import PreTrainedModel
        if hasattr(model_name_or_path, "config") or isinstance(model_name_or_path, PreTrainedModel):
            # we already have the model (or a PEFT wrapper with config attribute)
            hf_model_config = model_name_or_path.config
        else:
            # if we don't have the model yet at this stage
            hf_model_config = AutoConfig.from_pretrained(model_name_or_path)

        model_attn_implementation = getattr(hf_model_config, "_attn_implementation", None)
        if model_attn_implementation is not None and model_attn_implementation != core_attn_implementation:
            raise ValueError(
                f"core_attn_implementation='{core_attn_implementation}' does not match "
                f"model config attn_implementation='{model_attn_implementation}'. "
                "Set both to the same value so sequence-parallel wrapper can intercept the active attention path.")

        # eager always materializes a 4D attention_mask (O(n²) memory) and cannot fall back
        # to is_causal=True like sdpa — so it's incompatible with SP which discards masks.
        unsupported_attn_implementation = ["eager", "paged|eager"]
        if core_attn_implementation in unsupported_attn_implementation:
            raise ValueError(
                f"{core_attn_implementation} attn_implementation isn't currently supported by Ulysses sequence"
                f" parallelism because it requires a 4D attention_mask (O(n²) memory)."
                f" Use any flash attention variant, 'flex_attention', 'sdpa',"
                f" or a hub-hosted kernel (e.g. 'kernels-community/flash-attn2').")

        # Hub kernels (e.g. kernels-community/flash-attn2) are registered lazily in transformers.
        # Ensure registration happens before validating against ALL_ATTENTION_FUNCTIONS.
        is_hub_kernel_attn = (isinstance(core_attn_implementation, str) and re.search(
            r"^[^/:]+/[^/:]+(?:@[^/:]+)?(?::[^/:]+)?$", core_attn_implementation) is not None)
        if is_hub_kernel_attn:
            try:
                from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
            except ImportError as e:
                raise ImportError("Hub kernel attention requires a transformers version exposing "
                                  "`transformers.modeling_flash_attention_utils.lazy_import_flash_attention`.") from e
            lazy_import_flash_attention(core_attn_implementation)

        if core_attn_implementation not in ALL_ATTENTION_FUNCTIONS:
            raise ValueError(
                f"{core_attn_implementation} is not a valid attn_implementation. The choices are {ALL_ATTENTION_FUNCTIONS.valid_keys()}"
            )
        core_attn_function = ALL_ATTENTION_FUNCTIONS[core_attn_implementation]

        if _ACTIVE_ATTENTION_REGISTRATIONS:
            raise RuntimeError(
                "Ulysses SP is already registered in this process. Call unregister_from_transformers() before "
                "registering another model.")

        arch_cfg = hf_model_config.get_text_config()
        prepared_linear_attention = prepare_qwen35_linear_attention_cp(
            model=model_name_or_path,
            hf_model_config=hf_model_config,
            arch_cfg=arch_cfg,
            core_attn_implementation=core_attn_implementation,
            disable_in_eval=disable_in_eval,
            micro_batch_size=micro_batch_size,
        )

        transformers_version_min = "4.51.3"
        transformers_version_have = importlib_metadata.version("transformers")
        if version.parse(transformers_version_have) < version.parse(transformers_version_min):
            raise ValueError(f"transformers>={transformers_version_min} is required, but you have "
                             f"transformers=={transformers_version_have}")
        if arch_cfg.num_attention_heads % sequence_parallel_size != 0:
            raise ValueError(f"Attention head count {arch_cfg.num_attention_heads} is not divisible by SP size "
                             f"{sequence_parallel_size}")
        if not (arch_cfg.num_key_value_heads % sequence_parallel_size == 0
                or sequence_parallel_size % arch_cfg.num_key_value_heads == 0):
            raise ValueError(
                f"KV attention head count {arch_cfg.num_key_value_heads} and SP size {sequence_parallel_size} "
                "must divide one another")
        if not seq_length_is_variable and seq_length % sequence_parallel_size != 0:
            raise ValueError(f"Sequence length {seq_length} is not divisible by SP size {sequence_parallel_size}")

        owned_module_ids = set()
        owned_config_ids = {id(arch_cfg)}
        if isinstance(model_name_or_path, torch.nn.Module):
            for module in model_name_or_path.modules():
                module_config = getattr(module, "config", None)
                module_name = type(module).__name__.lower()
                has_attention_projections = (all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj"))
                                             or "attention" in module_name or module_name.endswith("attn"))
                if id(module_config) in owned_config_ids and has_attention_projections:
                    owned_module_ids.add(id(module))
        if isinstance(prepared_linear_attention, dict):
            owned_module_ids.update(prepared_linear_attention["dense_attention_module_ids"])

        expected_model_type = str(getattr(arch_cfg, "model_type", "") or "").lower()
        expected_model_path = (model_name_or_path if isinstance(model_name_or_path, str) else None)
        expected_head_dim = getattr(arch_cfg, "head_dim", arch_cfg.hidden_size // arch_cfg.num_attention_heads)

        def owns_attention_module(module, query, key):
            if (linear_registration is not None and hasattr(linear_registration, "owns_dense_attention_module")
                    and linear_registration.owns_dense_attention_module(module)):
                return True
            if id(module) in owned_module_ids:
                return True
            module_config = getattr(module, "config", None)
            if expected_model_path is None:
                return False
            if module_config is None:
                return False
            module_model_type = str(getattr(module_config, "model_type", "") or "").lower()
            module_model_path = getattr(module_config, "_name_or_path", None)
            if module_model_type != expected_model_type or module_model_path != expected_model_path:
                return False
            return (query.ndim == 4 and key.ndim == 4 and query.shape[1] == arch_cfg.num_attention_heads
                    and key.shape[1] == arch_cfg.num_key_value_heads and query.shape[-1] == expected_head_dim)

        global _ACTIVE_LINEAR_ATTENTION_REGISTRATION
        linear_registration = None
        uattn_wrapper = None
        mpu.initialize_sequence_parallel(sequence_parallel_size=sequence_parallel_size)
        try:
            if seq_length_is_variable:
                local_seq_length = None
                global_seq_length = None
            else:
                local_seq_length = seq_length // mpu.get_sequence_parallel_world_size()
                global_seq_length = seq_length

            uattn = cls(
                attn=core_attn_function,
                batch_size=micro_batch_size,
                attn_head_count=arch_cfg.num_attention_heads,
                attn_head_size=expected_head_dim,
                kv_head_count=arch_cfg.num_key_value_heads,
                num_hidden_layers=arch_cfg.num_hidden_layers,
                process_group=mpu.get_sequence_parallel_group(),
                seq_length_is_variable=seq_length_is_variable,
                local_seq_length=local_seq_length,
                global_seq_length=global_seq_length,
                disable_in_eval=disable_in_eval,
            )
            uattn.core_attn_implementation = core_attn_implementation

            def uattn_wrapper(
                module: torch.nn.Module,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: torch.Tensor,
                *args,
                **kwargs,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                if not owns_attention_module(module, query, key):
                    return core_attn_function(module, query, key, value, attention_mask, *args, **kwargs)

                # Eval bypass must retain the model's original padding/custom mask.
                if disable_in_eval and not module.training:
                    return core_attn_function(module, query, key, value, attention_mask, *args, **kwargs)

                # Local dense masks cannot be used after sequence all-to-all. Flex BlockMask is
                # retained only as a signal that the model selected its flex path; forward()
                # rebuilds a canonical full-sequence document-causal mask.
                is_block_mask = False
                if core_attn_implementation == "flex_attention":
                    try:
                        from torch.nn.attention.flex_attention import BlockMask
                    except (ImportError, AttributeError) as exc:
                        raise RuntimeError(
                            "Ulysses flex_attention requires torch.nn.attention.flex_attention.BlockMask.") from exc
                    is_block_mask = isinstance(attention_mask, BlockMask)
                    if attention_mask is not None and not is_block_mask:
                        raise RuntimeError(
                            "Ulysses flex_attention cannot reconstruct an arbitrary dense/custom attention mask; "
                            "only standard causal or packed document-causal BlockMask inputs are supported.")

                if not is_block_mask:
                    attention_mask = None

                return uattn(module, query, key, value, attention_mask, *args, **kwargs)

            # Override the selected backend so Transformers still follows all of its backend-specific
            # setup branches. The wrapper delegates immediately for modules outside this registration.
            ALL_ATTENTION_FUNCTIONS[core_attn_implementation] = uattn_wrapper
            linear_registration = create_qwen35_linear_attention_registration(
                prepared_linear_attention,
                sp_group=mpu.get_sequence_parallel_group(),
                sp_world_size=mpu.get_sequence_parallel_world_size(),
            )
            if linear_registration is not None:
                linear_registration.install()
        except Exception:
            if uattn_wrapper is not None and ALL_ATTENTION_FUNCTIONS.get(core_attn_implementation) is uattn_wrapper:
                ALL_ATTENTION_FUNCTIONS[core_attn_implementation] = core_attn_function
            if linear_registration is not None:
                linear_registration.restore()
            mpu.destroy_sequence_parallel()
            raise

        _ACTIVE_ATTENTION_REGISTRATIONS[core_attn_implementation] = (core_attn_function, uattn_wrapper)
        _ACTIVE_LINEAR_ATTENTION_REGISTRATION = linear_registration

        return mpu

    @classmethod
    def unregister_from_transformers(cls, core_attn_implementation=None):
        """Restore Transformers and Qwen3.5 forwards modified by Ulysses registration."""
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu

        global _ACTIVE_LINEAR_ATTENTION_REGISTRATION
        if (core_attn_implementation is not None and core_attn_implementation not in _ACTIVE_ATTENTION_REGISTRATIONS):
            active = sorted(_ACTIVE_ATTENTION_REGISTRATIONS)
            raise ValueError(
                f"Ulysses SP backend {core_attn_implementation!r} is not registered; active backends: {active}.")

        implementations = ([core_attn_implementation]
                           if core_attn_implementation is not None else list(_ACTIVE_ATTENTION_REGISTRATIONS))
        for implementation in implementations:
            registration = _ACTIVE_ATTENTION_REGISTRATIONS.pop(implementation)
            original, wrapper = registration
            if ALL_ATTENTION_FUNCTIONS.get(implementation) is wrapper:
                ALL_ATTENTION_FUNCTIONS[implementation] = original

        if not _ACTIVE_ATTENTION_REGISTRATIONS:
            if _ACTIVE_LINEAR_ATTENTION_REGISTRATION is not None:
                _ACTIVE_LINEAR_ATTENTION_REGISTRATION.restore()
                _ACTIVE_LINEAR_ATTENTION_REGISTRATION = None
            mpu.destroy_sequence_parallel()


# ----------------------------------------------------------------------------
# Compatibility alias retained for existing callers and tests.
_position_ids_to_packed_cu_seqlens = position_ids_to_packed_cu_seqlens

_GLOBAL_CU_SEQLENS_KEYS = frozenset({
    "cu_seq_lens_q",
    "cu_seq_lens_k",
    "cu_seqlens",
    "cu_seqlens_q",
    "cu_seqlens_k",
})
_CU_SEQLENS_POSITION_PRIORITY = (
    "cu_seq_lens_q",
    "cu_seqlens_q",
    "cu_seqlens",
    "cu_seq_lens_k",
    "cu_seqlens_k",
)
_MULTIMODAL_BATCH_KEYS = frozenset({
    "images",
    "videos",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "image_sizes",
})


class UlyssesSPDataLoaderAdapter:

    def __init__(
        self,
        dl: DataLoader,
        sp_rank: int,
        sp_group,
        sp_world_size,
        device,
        pad_to_sp_multiple=True,
        pad_token_id=0,
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
        self.pad_to_sp_multiple = pad_to_sp_multiple
        self.pad_token_id = pad_token_id

        self.iter = iter(dl)
        self.micro_batches: deque[Any] = deque()

    def __len__(self):
        return len(self.dl) * self.sp_world_size

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.micro_batches) == 0:
            self.refill()

        return self.micro_batches.popleft()

    @staticmethod
    def _is_multimodal_key(key):
        return (key in _MULTIMODAL_BATCH_KEYS or key.startswith("pixel_") or key.startswith("image_")
                or key.startswith("video_") or key.startswith("vision_"))

    @staticmethod
    def _position_ids_from_seq_idx(seq_idx):
        if seq_idx.ndim != 2:
            raise ValueError(f"seq_idx must have shape [batch, sequence], got {tuple(seq_idx.shape)}")
        indices = torch.arange(seq_idx.shape[1], device=seq_idx.device, dtype=torch.long).expand_as(seq_idx)
        starts = torch.ones_like(seq_idx, dtype=torch.bool)
        starts[:, 1:] = seq_idx[:, 1:] != seq_idx[:, :-1]
        latest_start = torch.where(starts, indices, torch.zeros_like(indices)).cummax(dim=1).values
        return indices - latest_start

    @staticmethod
    def _position_ids_from_cu_seqlens(cu_seqlens, batch_size, seq_length, device):
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
        total_length = batch_size * seq_length
        if (cu_seqlens.ndim != 1 or cu_seqlens.numel() < batch_size + 1 or cu_seqlens[0].item() != 0
                or cu_seqlens[-1].item() != total_length or bool((cu_seqlens[1:] <= cu_seqlens[:-1]).any())):
            raise ValueError(
                "cu-seqlens used to generate position_ids must be a strictly increasing 1-D tensor from 0 to "
                f"batch_size * seq_length ({total_length}).")
        row_boundaries = torch.arange(
            0,
            total_length + 1,
            seq_length,
            device=device,
            dtype=torch.long,
        )
        if not bool(torch.isin(row_boundaries, cu_seqlens).all()):
            raise ValueError("cu-seqlens must contain every batch-row boundary before position_ids can be generated.")
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        starts = torch.repeat_interleave(cu_seqlens[:-1], lengths)
        flat_position_ids = torch.arange(total_length, device=device, dtype=torch.long) - starts
        return flat_position_ids.reshape(batch_size, seq_length)

    def _ensure_position_ids(self, batch):
        if "position_ids" in batch:
            position_ids = batch["position_ids"]
            if position_ids.ndim != 2 or position_ids.shape != batch["input_ids"].shape:
                raise ValueError(
                    "Text-only Ulysses SP requires position_ids with the same [batch, sequence] shape as input_ids; "
                    f"got {tuple(position_ids.shape)} and {tuple(batch['input_ids'].shape)}.")
            return

        batch_size, seq_length = batch["input_ids"].shape[:2]
        seq_idx = batch.get("seq_idx")
        if torch.is_tensor(seq_idx):
            if seq_idx.shape != batch["input_ids"].shape:
                raise ValueError(f"seq_idx must match input_ids shape, got {tuple(seq_idx.shape)} and "
                                 f"{tuple(batch['input_ids'].shape)}.")
            batch["position_ids"] = self._position_ids_from_seq_idx(seq_idx)
            return
        cu_seqlens = next(
            (batch[key] for key in _CU_SEQLENS_POSITION_PRIORITY if torch.is_tensor(batch.get(key))),
            None,
        )
        if cu_seqlens is not None:
            batch["position_ids"] = self._position_ids_from_cu_seqlens(
                cu_seqlens,
                batch_size,
                seq_length,
                batch["input_ids"].device,
            )
            return
        batch["position_ids"] = torch.arange(
            seq_length,
            device=batch["input_ids"].device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1).clone()

    def _gather_tensor(self, tensor):
        shapes = [None for _ in range(self.sp_world_size)]
        dist.all_gather_object(shapes, tuple(tensor.shape), group=self.sp_group)
        tensor_list = [torch.empty(shape, dtype=tensor.dtype, device=tensor.device) for shape in shapes]
        dist.all_gather(tensor_list, tensor.contiguous(), group=self.sp_group)
        return tensor_list

    def _pad_sequence_tensor(self, key, tensor, target_seq_length):
        pad_length = target_seq_length - tensor.shape[1]
        if pad_length <= 0:
            return tensor
        if key == "position_ids":
            increments = torch.arange(1, pad_length + 1, device=tensor.device, dtype=tensor.dtype)
            padding = tensor[:, -1:].expand(-1, pad_length) + increments.unsqueeze(0)
            return torch.cat((tensor, padding), dim=1)
        if key == "seq_idx":
            repeats = [1] * tensor.ndim
            repeats[1] = pad_length
            padding = tensor[:, -1:].repeat(*repeats)
            return torch.cat((tensor, padding), dim=1)

        if key in ("labels", "shift_labels"):
            fill_value = -100
        elif key == "input_ids":
            fill_value = self.pad_token_id
        else:
            fill_value = 0
        padded_shape = list(tensor.shape)
        padded_shape[1] = target_seq_length
        padded = torch.full(padded_shape, fill_value, dtype=tensor.dtype, device=tensor.device)
        padded[:, :tensor.shape[1]] = tensor
        return padded

    @staticmethod
    def _adjust_cu_seqlens_for_padding(batch, old_seq_length, new_seq_length):
        if old_seq_length == new_seq_length:
            return
        adjusted_cu_seqlens = []
        for key in _GLOBAL_CU_SEQLENS_KEYS:
            cu_seqlens = batch.get(key)
            if not torch.is_tensor(cu_seqlens):
                continue
            batch_size = batch["input_ids"].shape[0]
            expected_end = batch_size * old_seq_length
            if cu_seqlens.ndim != 1 or cu_seqlens[-1].item() != expected_end:
                raise ValueError(
                    f"{key} must be 1-D and end at the unpadded flattened sequence length {expected_end}.")
            rows = torch.div(cu_seqlens, old_seq_length, rounding_mode="floor")
            columns = torch.remainder(cu_seqlens, old_seq_length)
            adjusted = rows * new_seq_length + columns
            batch[key] = adjusted
            adjusted_cu_seqlens.append(adjusted)

        if adjusted_cu_seqlens:
            max_seqlen = max(int((cu[1:] - cu[:-1]).max().item()) for cu in adjusted_cu_seqlens)
            for key in ("max_seqlen", "max_seqlen_q", "max_seqlen_k", "max_length_q", "max_length_k"):
                value = batch.get(key)
                if torch.is_tensor(value):
                    batch[key] = value.new_tensor(max_seqlen)
                elif value is not None:
                    batch[key] = max_seqlen

    def refill(self):
        # reset the iterator if StopIteration arrives, and re-raise it to allow multiple epochs to run
        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dl)
            raise StopIteration
        if not isinstance(batch, dict) or "input_ids" not in batch or not torch.is_tensor(batch["input_ids"]):
            raise ValueError("Ulysses SP dataloader batches must be dictionaries containing tensor input_ids.")
        if batch["input_ids"].ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, sequence], got {tuple(batch['input_ids'].shape)}")
        multimodal_keys = [key for key, value in batch.items() if value is not None and self._is_multimodal_key(key)]
        if multimodal_keys:
            raise ValueError(
                "Ulysses SP currently supports text tokens only; multimodal/vision inputs require a separate "
                f"visual-token sharding contract. Unsupported keys: {sorted(multimodal_keys)}")
        if "labels" not in batch or not torch.is_tensor(batch["labels"]):
            raise ValueError("Ulysses SP dataloader batches must contain tensor labels.")
        if batch["labels"].shape[:2] != batch["input_ids"].shape:
            raise ValueError(
                f"labels must match input_ids [batch, sequence] shape, got {tuple(batch['labels'].shape)} and "
                f"{tuple(batch['input_ids'].shape)}.")

        self._ensure_position_ids(batch)
        micro_batches = defaultdict(dict)
        tensor_kinds = {}
        # XXX: replace with more efficient all-to-all?

        input_shape = batch["input_ids"].shape
        for k in batch.keys():
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(self.device)
                if k in _GLOBAL_CU_SEQLENS_KEYS:
                    tensor_kinds[k] = "global_metadata"
                elif batch[k].ndim >= 2 and batch[k].shape[:2] == input_shape[:2]:
                    tensor_kinds[k] = "sequence"
                else:
                    tensor_kinds[k] = "replicated"
                with torch.no_grad():
                    tensor_list = self._gather_tensor(batch[k])
            else:
                tensor_kinds[k] = "replicated"
                tensor_list = [None for _ in range(self.sp_world_size)]
                dist.all_gather_object(tensor_list, batch[k], group=self.sp_group)

            for rank, tensor in enumerate(tensor_list):
                micro_batches[rank][k] = tensor

        del tensor_list
        del batch

        for batch in micro_batches.values():
            seq_length = len(batch["input_ids"][0])

            if seq_length % self.sp_world_size != 0:
                if not self.pad_to_sp_multiple:
                    raise ValueError(f"batch's seqlen={seq_length} isn't divisible by sp-size={self.sp_world_size}")
                padded_seq_length = math.ceil(seq_length / self.sp_world_size) * self.sp_world_size
                self._adjust_cu_seqlens_for_padding(batch, seq_length, padded_seq_length)
                for key, kind in tensor_kinds.items():
                    if kind == "sequence" and key in batch:
                        batch[key] = self._pad_sequence_tensor(key, batch[key], padded_seq_length)
                seq_length = padded_seq_length
            chunk_len = seq_length // self.sp_world_size

            # because we have to gather logits from all sp ranks we have to do the loss function ourselves
            # therefore remove labels to avoid an attempt to calculate loss by transformers
            labels = batch.pop("labels")
            labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
            batch["shift_labels"] = labels[..., 1:].contiguous()
            tensor_kinds["shift_labels"] = "sequence"
            # free up temp memory
            del labels

            # batch sharding
            for k in batch.keys():
                if not torch.is_tensor(batch[k]):
                    continue
                if tensor_kinds[k] == "sequence":
                    batch[k] = batch[k][:, chunk_len * self.sp_rank:chunk_len * (self.sp_rank + 1)]
                # At seqlen>10M and 32+ GPUs this can take GBs, so keep queued batches on CPU.
                batch[k] = batch[k].cpu()

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
        # reshape rather than view: a caller may pass a non-contiguous x or incoming grad (a transposed or
        # channel-sliced activation), for which view cannot produce the flattened shape
        x = x.reshape(-1, hidden_size)
        incoming_grad = grads[0].reshape(-1, hidden_size)
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
