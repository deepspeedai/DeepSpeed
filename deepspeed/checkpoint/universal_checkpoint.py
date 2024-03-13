# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import torch
import types
from .constants import (FP32_WEIGHT_KEY, PARAM, VOCAB_TENSOR, CAT_DIM, PARAM_N_SUB_PARAMS)


def load_hp_checkpoint_state(self, folder, tp_rank, tp_world_size):
    hp_mapping = self._hp_mapping
    hp_mapping.optim_fragment = {}

    hp_keys = []
    for file in os.listdir(folder):
        # We expect files named something like "exp_avg.pt", "exp_avg_sq.pt", "fp32.pt"
        pattern = r'(.+).pt'
        match = re.search(pattern, file)
        if match:
            hp_keys.append(match.group(1))

    for key in hp_keys:
        ckpt_file = os.path.join(folder, f"{key}.pt")
        ckpt_dict = torch.load(ckpt_file)
        full_hp_param = ckpt_dict[PARAM]

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
        if full_hp_param.shape == self.shape:
            tp_rank = 0
            tp_world_size = 1

        # special case for word_embeddings weights which get padded differently depending on TP degree.
        # the converter to universal currently strips the original padding completely so the saved
        # weight is padding-free and we just need to add new padding depending on the target TP
        # degree
        is_vocab_tensor = ckpt_dict.get(VOCAB_TENSOR, False)
        if is_vocab_tensor:
            # In the absence of data passed from the user wrt new padded vocab specific to tp degree
            # we can again derive that data by reverse engineering the target shapes like so:
            padded_target_vocab_size = self.shape[0] * tp_world_size
            assert padded_target_vocab_size >= full_hp_param.shape[0], \
                f'Vocab tensor padded size {padded_target_vocab_size} < loaded universal size {full_hp_param.shape[0]}'
            if padded_target_vocab_size > full_hp_param.shape[0]:
                padding_size = padded_target_vocab_size - full_hp_param.shape[0]
                full_hp_param = torch.nn.functional.pad(full_hp_param, (0, 0, 0, padding_size), "constant", 0)

        full_param_numel = full_hp_param.numel()
        tp_slice_numel = self.numel()
        #        if key == FP32_WEIGHT_KEY and 'word_embeddings.weight' in folder:
        #            print_rank_0(f'{full_hp_param[:10]=}', force=True)


        assert full_param_numel == tp_world_size * tp_slice_numel, \
            f'Loading {ckpt_file} full param numel {full_param_numel} != tensor slice numel {tp_slice_numel} * tp_world_size {tp_world_size}'

        #        print(f"{full_hp_param.shape=} {full_param_numel=} {folder=}")
        #        print(f"{dst_tensor.shape=} {dst_tensor.numel()=}{folder=}")

        # since when we do many to 1 on tp we cat sometimes on dim=0 and other times on dim=1 we have to do exactly the same in reverse
        # special case is when a single parameter is effectively a container for multiple sub parameters
        # (more details at PARAM_N_SUB_PARAMS definition)
        chunk_dim = ckpt_dict.get(CAT_DIM, 0)
        n_sub_params = ckpt_dict.get(PARAM_N_SUB_PARAMS, 1)
        if n_sub_params > 1:
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


def enable_universal_checkpoint(param_list):
    for param in param_list:
        param.load_hp_checkpoint_state = types.MethodType(load_hp_checkpoint_state, param)
