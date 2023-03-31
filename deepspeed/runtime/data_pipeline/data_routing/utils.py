# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch


def bsh_decoder_gather(reserved_length, hidden_states, mask):
    # random-layer-token-drop
    rand_list = []
    part_hidden_states = []  #  batch, seq, hidden ## different from megatron
    for k in range(hidden_states.size(0)):
        B_tmp = torch.randperm(hidden_states.size(1), device=hidden_states.device)[:reserved_length]
        B = B_tmp.sort()[0]
        rand_list.append(B)
        part_hidden_states.append(hidden_states[k:k + 1, B, :])

    part_hidden_states = torch.cat(part_hidden_states, dim=0)
    part_mask = mask[:, :, :reserved_length, :reserved_length]
    return part_hidden_states, rand_list, part_mask


def bsh_decoder_scatter(hidden_states, part_hidden_states, rand_list):
    for k in range(hidden_states.size(0)):
        hidden_states[k, rand_list[k], :] = part_hidden_states[k, :, :]
    return hidden_states
