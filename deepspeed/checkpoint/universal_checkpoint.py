# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import torch
import types
from typing import List, Tuple, Union
from dataclasses import dataclass
from .constants import (FP32_WEIGHT_KEY, PARAM, VOCAB_TENSOR, CAT_DIM, PARAM_N_SUB_PARAMS, SUB_PARAM_SHAPE,
                        EP_IS_EXPERT_PARAM, EP_NUM_EXPERTS, DS_AUTOTP_UC_META, UNIVERSAL_CHECKPOINT_VERSION_KEY)


@dataclass
class SubparamShape:
    patterns: List[str]
    shape: Tuple[Union[Tuple[int], int]]
    partition_dim: int


def _get_param_uc_restore_meta(param):
    """Return the restore-facing view of AutoTP UC metadata for a parameter.

    AutoTP parameter metadata intentionally serves two separate consumers:
    - restore-time fields at the top level, consumed here by UC loading
    - conversion-time fields under `conversion`, consumed by
      `collect_autotp_universal_checkpoint_info()` in `layers.py`
    """
    return getattr(param, DS_AUTOTP_UC_META, None)


def _narrow_sub_params(full_view, partition_dim, sub_dim_sizes, shard_widths, tp_rank, tp_world_size, uc_version):
    """Take this rank's piece of every sub-parameter and concatenate them back together.

    ``shard_widths`` gives the per-rank width of each sub-parameter, so a fused attention
    weight is cut on key/value head boundaries. Metadata written before UCP version 0.4 carries
    no widths and can only have been split evenly.
    """
    assert sum(sub_dim_sizes) == full_view.shape[partition_dim], (
        f"Sub-parameter sizes {list(sub_dim_sizes)} sum to {sum(sub_dim_sizes)}, but dimension {partition_dim} "
        f"of the parameter is {full_view.shape[partition_dim]}. The metadata describes a different parameter.")

    if uc_version < 0.4:
        # An even split is the only layout recoverable without recorded widths. Assuming it for
        # an uneven sub-parameter shifts every offset and silently drops the trailing elements,
        # so refuse rather than restore wrong weights.
        uneven = [size for size in sub_dim_sizes if size % tp_world_size != 0]
        assert not uneven, (
            f"Sub-parameter sizes {uneven} are not divisible by tp_world_size {tp_world_size}, and this "
            "metadata records no sub_param_shard_widths, so its uneven layout cannot be reconstructed.")
        shard_widths = [[size // tp_world_size] * tp_world_size for size in sub_dim_sizes]

    assert len(shard_widths) == len(sub_dim_sizes), (
        f"Got {len(shard_widths)} shard width entries for {len(sub_dim_sizes)} sub-parameters.")

    offset = 0
    merged_chunks = []
    for sub_dim_size, widths in zip(sub_dim_sizes, shard_widths):
        assert len(widths) == tp_world_size, (
            f"Sub-parameter of size {sub_dim_size} has {len(widths)} shard widths, expected one per tp rank "
            f"({tp_world_size}).")
        assert sum(widths) == sub_dim_size, (
            f"Sub-parameter shard widths {list(widths)} sum to {sum(widths)}, expected {sub_dim_size}.")
        start = offset + sum(widths[:tp_rank])
        merged_chunks.append(full_view.narrow(partition_dim, start, widths[tp_rank]))
        offset += sub_dim_size

    slice_tensor = torch.cat(merged_chunks, dim=partition_dim)
    return slice_tensor.flatten()


def _resolve_autotp_partition(current_param, ckpt_dict, full_hp_param, tp_rank, tp_world_size):
    meta = _get_param_uc_restore_meta(current_param)
    if not meta:
        return None

    partition_dim = meta.get('partition_dim')
    logical_shape = meta.get('logical_shape')
    sub_param_shape = meta.get('sub_param_shape')
    sub_param_sizes = meta.get('sub_param_sizes')
    sub_param_shard_widths = meta.get('sub_param_shard_widths')
    partition_sizes = meta.get('partition_sizes')
    replicated = meta.get('replicated', False)
    uc_version = meta.get(UNIVERSAL_CHECKPOINT_VERSION_KEY, 0.0)

    # The layer could not describe how it split this parameter, so conversion refuses it. The
    # generic paths below would reassemble it as contiguous rank-ordered slices, which is not
    # how a fused layout was cut, so restoring here would silently corrupt the weight.
    unsupported_reason = meta.get('conversion', {}).get('unsupported_reason')
    if unsupported_reason:
        raise RuntimeError(f"Cannot restore a universal checkpoint into this AutoTP parameter: {unsupported_reason}")

    if replicated:
        assert partition_dim is None
        slice_tensor = full_hp_param
        return slice_tensor.flatten()

    if partition_dim is None:
        return None

    if logical_shape is None:
        return None

    full_view = full_hp_param.view(logical_shape)

    # sub_param_sizes holds the resolved physical width of each sub-parameter, whereas
    # sub_param_shape may be a logical view spec such as (3, -1) whose partition_dim entry is
    # the sub-parameter *count*. Reading that count as a width restores only a fraction of the
    # parameter, so only fall back to sub_param_shape for older metadata that lacks the sizes.
    sub_dim_sizes = None
    if sub_param_sizes is not None:
        sub_dim_sizes = sub_param_sizes
    elif sub_param_shape is not None:
        if hasattr(sub_param_shape, "shape") and hasattr(sub_param_shape, "partition_dim"):
            shape_spec = sub_param_shape.shape
            partition_dim = sub_param_shape.partition_dim
        else:
            shape_spec = sub_param_shape
        sub_dim_sizes = shape_spec[partition_dim]

    if sub_dim_sizes is not None:
        if not isinstance(sub_dim_sizes, (tuple, list)):
            sub_dim_sizes = (sub_dim_sizes, )

        return _narrow_sub_params(full_view, partition_dim, sub_dim_sizes, sub_param_shard_widths, tp_rank,
                                  tp_world_size, uc_version)

    if partition_sizes is not None:
        shard_offset = sum(partition_sizes[:tp_rank])
        shard_size = partition_sizes[tp_rank]
        slice_tensor = full_view.narrow(partition_dim, shard_offset, shard_size)
        return slice_tensor.flatten()

    # torch.chunk sizes every block as ceil(size / tp_world_size) and shrinks the last one,
    # while AutoTP hands one extra element to each of the first `size % tp_world_size` ranks.
    # The two disagree once the split is uneven (10/4 -> chunk [3, 3, 3, 1] vs AutoTP
    # [3, 3, 2, 2]), and chunk can even return fewer blocks than there are ranks.
    partition_dim_size = full_view.shape[partition_dim]
    assert partition_dim_size % tp_world_size == 0, (
        f"Dimension {partition_dim} of size {partition_dim_size} is not divisible by tp_world_size "
        f"{tp_world_size}, and this checkpoint records no partition_sizes, so its uneven layout cannot "
        "be reconstructed.")
    slice_tensor = full_view.chunk(tp_world_size, dim=partition_dim)[tp_rank]
    return slice_tensor.flatten()


def load_hp_checkpoint_state(self, folder, tp_rank, tp_world_size, ep_rank=0, ep_size=1):
    hp_mapping = self._hp_mapping
    hp_mapping.optim_fragment = {}

    hp_keys = []
    for file in os.listdir(folder):
        # We expect files named something like "exp_avg.pt", "exp_avg_sq.pt", "fp32.pt"
        pattern = r'(.+).pt'
        match = re.search(pattern, file)
        if match:
            hp_keys.append(match.group(1))

    step = None
    for key in hp_keys:
        ckpt_file = os.path.join(folder, f"{key}.pt")
        ckpt_dict = torch.load(ckpt_file, weights_only=False)

        if key == "step":
            step = ckpt_dict
            continue

        full_hp_param = ckpt_dict[PARAM]

        # EP-aware slicing for expert parameters saved in universal format.
        # Must happen BEFORE shape-match check so that after slicing,
        # full_hp_param.shape == self.shape triggers tp_rank=0, tp_world_size=1.
        is_expert_param = ckpt_dict.get(EP_IS_EXPERT_PARAM, False)
        if is_expert_param and ep_size > 1:
            ep_num_experts = ckpt_dict.get(EP_NUM_EXPERTS)
            assert ep_num_experts is not None, \
                f"Expert param in {ckpt_file} missing '{EP_NUM_EXPERTS}' metadata"
            assert full_hp_param.shape[0] == ep_num_experts, \
                f"Expert param dim 0 ({full_hp_param.shape[0]}) != {EP_NUM_EXPERTS} ({ep_num_experts})"
            assert ep_num_experts % ep_size == 0, \
                f"num_experts ({ep_num_experts}) not divisible by ep_size ({ep_size})"
            num_local = ep_num_experts // ep_size
            ep_start = ep_rank * num_local
            ep_end = ep_start + num_local
            full_hp_param = full_hp_param[ep_start:ep_end]

        # need to deal with slices that were averaged.
        # the opposite of averaging here becomes an exact copy of the first slice
        # I thought of 2 ways:
        # implementation a. find a way for a client to pass a dict with patterns
        # if any(re.search(pattern, folder) for pattern in WEIGHTS_TO_AVERAGE_PATTERNS):
        #     tp_rank = 0
        #     tp_world_size = 1
        # the other approach is to assume that the saved data is correct and if full_hp_param.shape ==
        # self.shape that means we automatically copy?
        # implementation b.
        # this version requires no additional data passed from the client
        # if the shapes already match it must be slices that were averaged - so we just hack around those
        # AutoTP restore metadata already pins down this rank's slice of the universal tensor,
        # including layouts where one rank owns the whole thing (e.g. a single KV head under MQA).
        # Collapsing the TP topology here would contradict the recorded per-rank widths.
        has_autotp_meta = _get_param_uc_restore_meta(self) is not None
        if full_hp_param.shape == self.shape and not has_autotp_meta:
            tp_rank = 0
            tp_world_size = 1

        # special case for word_embeddings weights which get padded differently depending on TP degree.
        # the converter to universal currently strips the original padding completely so the saved
        # weight is padding-free and we just need to add new padding depending on the target TP
        # degree
        # AutoTP restore metadata already describes the exact (possibly uneven) partition layout,
        # so the legacy tp-degree-derived vocab padding must not be applied on top of it.
        is_vocab_tensor = ckpt_dict.get(VOCAB_TENSOR, False) and not is_expert_param and not has_autotp_meta
        if is_vocab_tensor:
            # In the absence of data passed from the user wrt new padded vocab specific to tp degree
            # we can again derive that data by reverse engineering the target shapes like so:
            padded_target_vocab_size = self.shape[0] * tp_world_size
            assert padded_target_vocab_size >= full_hp_param.shape[0], \
                f'Vocab tensor padded size {padded_target_vocab_size} < loaded universal size {full_hp_param.shape[0]}'
            if padded_target_vocab_size > full_hp_param.shape[0]:
                padding_size = padded_target_vocab_size - full_hp_param.shape[0]
                full_hp_param = torch.nn.functional.pad(full_hp_param, (0, 0, 0, padding_size), "constant", 0)

        autotp_tp_hp_slice = _resolve_autotp_partition(self, ckpt_dict, full_hp_param, tp_rank, tp_world_size)
        if autotp_tp_hp_slice is not None:
            tp_hp_slice = autotp_tp_hp_slice
        else:
            full_param_numel = full_hp_param.numel()
            tp_slice_numel = self.numel()
            assert full_param_numel == tp_world_size * tp_slice_numel, \
                f'Loading {ckpt_file} full param numel {full_param_numel} != tensor slice numel {tp_slice_numel} * tp_world_size {tp_world_size}'

            #        print(f"{full_hp_param.shape=} {full_param_numel=} {folder=}")
            #        print(f"{dst_tensor.shape=} {dst_tensor.numel()=}{folder=}")

            sub_param_shape = ckpt_dict.get(SUB_PARAM_SHAPE, None)
            # since when we do many to 1 on tp we cat sometimes on dim=0 and other times on dim=1 we have to do exactly the same in reverse
            # special case is when a single parameter is effectively a container for multiple sub parameters
            # (more details at PARAM_N_SUB_PARAMS definition)
            chunk_dim = ckpt_dict.get(CAT_DIM, 0)
            n_sub_params = ckpt_dict.get(PARAM_N_SUB_PARAMS, 1)
            if sub_param_shape:
                partition_dim = sub_param_shape.partition_dim
                sub_dim_sizes = sub_param_shape.shape[partition_dim]
                if not isinstance(sub_dim_sizes, tuple):
                    sub_dim_sizes = (sub_dim_sizes, )

                partition_shape = [sum(d) if isinstance(d, tuple) else d for d in sub_param_shape.shape]
                full_hp_param = full_hp_param.view(partition_shape)

                offset = 0
                merged_chunks = []
                for sub_dim_size in sub_dim_sizes:
                    sub_params_tp_slice = full_hp_param.narrow(partition_dim,
                                                               offset, sub_dim_size).chunk(tp_world_size,
                                                                                           dim=partition_dim)[tp_rank]
                    merged_chunks.append(sub_params_tp_slice)
                    offset += sub_dim_size
                tp_hp_slice = torch.cat(merged_chunks, dim=partition_dim)

            elif n_sub_params > 1:
                sub_params = full_hp_param.chunk(n_sub_params, dim=chunk_dim)
                sub_params_tp_slice = [p.chunk(tp_world_size, dim=chunk_dim)[tp_rank] for p in sub_params]
                tp_hp_slice = torch.cat(sub_params_tp_slice, dim=chunk_dim)
            else:
                # this performs the opposite of cat when merging TP slices
                tp_hp_slice = full_hp_param.chunk(tp_world_size, chunk_dim)[tp_rank]

            tp_hp_slice = tp_hp_slice.flatten()

        lp_frag_address = hp_mapping.lp_fragment_address
        tp_hp_fragment = tp_hp_slice.narrow(0, lp_frag_address.start, lp_frag_address.numel)

        #        print(f"{key} SHAPE: {tp_hp_slice.shape=}")
        #        print(f"{key} SHAPE: {dst_tensor.shape=}")
        #        print(f"{key} SHAPE: {tp_hp_fragment.shape=}")

        if key == FP32_WEIGHT_KEY:
            dst_tensor = hp_mapping.get_hp_fragment()
            assert dst_tensor.numel() == lp_frag_address.numel, \
                f'Load checkpoint {key} dst numel {dst_tensor.numel()} != src numel {lp_frag_address.numel}'
            dst_tensor.data.copy_(tp_hp_fragment.data)
        else:
            assert tp_hp_fragment.numel() == lp_frag_address.numel, \
                f'Load checkpoint {key} dst numel {tp_hp_fragment.numel()} != src numel {lp_frag_address.numel}'

            hp_mapping.optim_fragment[key] = tp_hp_fragment.clone().detach()

    return step


def enable_universal_checkpoint(param_list):
    for param in param_list:
        param.load_hp_checkpoint_state = types.MethodType(load_hp_checkpoint_state, param)
