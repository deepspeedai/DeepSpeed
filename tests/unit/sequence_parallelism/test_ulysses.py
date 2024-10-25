# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed.comm as dist
from deepspeed import initialize
from transformers import AutoModel
from unit.common import DistributedTest
from deepspeed.sequence.layer import _SeqAllToAll
from unit.util import skip_on_arch
from unit.simple_model import *
from deepspeed.utils import groups
from deepspeed.module_inject.tp_shard import get_shard_size_list
#Use mesh device to create data and sequence parallel group


class TestUlyssesUtils(DistributedTest):
    world_size = 4

    def test_mesh_device_creation(self) -> None:
        skip_on_arch(min_arch=8)
        model = AutoModel.from_pretrained('bert-base-uncased')
        sp_size = 2
        dp_size = 2
        ds_engine, _, _, _ = initialize(
            model=model,
            config_params={
                "train_batch_size": 8,
                "data_parallel_size": dp_size,
                "sequence_parallel_size": sp_size
            },
        )
        assert ds_engine.seq_parallel_group is not None
        assert ds_engine.data_parallel_group is not None
        assert dist.get_world_size(group=ds_engine.seq_parallel_group) == sp_size
        assert dist.get_world_size(group=ds_engine.data_parallel_group) == dp_size
        assert dist.get_world_size() == sp_size * dp_size


#Sweep b,s,h,d to test all2all consistency
@pytest.mark.parametrize("d0", [2, 4])  #batch or sequence dimension
@pytest.mark.parametrize("d1", [4, 8])  #batch or sequence dimension
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [16, 32])
class TestUlyssesAll2All(DistributedTest):
    world_size = 4

    def test_alltoall_output_consistency(self, d0: int, d1: int, head_dim: int, num_heads: int) -> None:
        skip_on_arch(min_arch=8)
        model = AutoModel.from_pretrained('bert-base-uncased')
        ds_engine, _, _, _ = initialize(model=model, config_params={"train_batch_size": 8}, mesh_param=(2, 2))
        #4D tensor : b,s,h,d or s,b,h,d
        input_tensor = torch.randn(d0, d1, num_heads, head_dim, device=ds_engine.device)
        scatter_idx = 2
        batch_dim_idx = 0
        outputs = []
        seq_dims = [0]  #seq first API
        #TODO: Add support for batch first (that seq_dims=[0,1]) after PR for bs>1 issue with batch first is fixed
        ## See discussion in : https://github.com/microsoft/DeepSpeed/issues/5808
        for seq_dim in seq_dims:
            gather_idx = seq_dim
            #first all2all: sequence parallel to head parallel
            s2h_tensor = _SeqAllToAll.apply(ds_engine.seq_parallel_group, input_tensor, scatter_idx, gather_idx,
                                            batch_dim_idx)

            #No op
            # second all2all: head parallel to sequence parallel
            h2s_tensor = _SeqAllToAll.apply(ds_engine.seq_parallel_group, s2h_tensor, gather_idx, scatter_idx,
                                            batch_dim_idx)
            print(
                f'[{dist.get_rank()}] s={seq_dim} input: {input_tensor.shape} s2h: {s2h_tensor.shape} h2s_tensor: {h2s_tensor.shape}'
            )
            outputs.append(h2s_tensor)

        # Check outputs are the same as input
        for i in range(1, len(outputs)):
            assert torch.allclose(input_tensor, outputs[i]), f"Outputs differ for sequence dim {seq_dims[i]}"


@pytest.mark.parametrize("d0", [2, 4])  #batch or sequence dimension
@pytest.mark.parametrize("d1", [4, 8])  #batch or sequence dimension
@pytest.mark.parametrize("num_heads", [3, 7])
@pytest.mark.parametrize("head_dim", [16])
class TestUlyssesAll2All_odd(DistributedTest):
    world_size = 4

    def test_alltoall_output_consistency(self, d0: int, d1: int, head_dim: int, num_heads: int) -> None:

        data_parallel_size = 2
        seq_parallel_size = self.world_size // data_parallel_size
        skip_on_arch(min_arch=8)

        def seq_batch_heads_hash(d0, d1, h, offset_d0=0, offset_d1=0, offset_h=0):
            d0 += offset_d0
            d1 += offset_d1
            h += offset_h
            return d0 * 10 + h + d1 * 0.1

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        ds_engine, _, _, _ = initialize(model=model,
                                        config_params={"train_batch_size": 8},
                                        mesh_param=(data_parallel_size, seq_parallel_size))

        scatter_idx = 2
        outputs = []
        inputs = []
        batch_dims = [0, 1]
        seq_dims = [1, 0]

        for idx, seq_dim in enumerate(seq_dims):
            gather_idx = seq_dim
            batch_dim_idx = batch_dims[idx]

            #4D tensor : b,s,h,d or s,b,h,d
            #create a hash tensor from pos_id, head_id, and batch_id
            d0_indices = torch.arange(d0).reshape(-1, 1, 1, 1)
            d1_indices = torch.arange(d1).reshape(1, -1, 1, 1)
            h_indices = torch.arange(num_heads).reshape(1, 1, -1, 1)
            input_tensor = torch.randn(d0, d1, num_heads, head_dim, device=ds_engine.device)
            if batch_dim_idx == 1:  #seq_len_dim : 0(d0)
                input_tensor[:] = seq_batch_heads_hash(d0_indices, d1_indices, h_indices,
                                                       d0 * groups._get_sequence_parallel_rank(), 0)
            elif batch_dim_idx == 0:  #seq_len_dim : 1(d1)
                input_tensor[:] = seq_batch_heads_hash(d0_indices, d1_indices, h_indices, 0,
                                                       d1 * groups._get_sequence_parallel_rank())
            inputs.append(input_tensor)

            ### first all2all: sequence parallel to head parallel
            s2h_tensor = _SeqAllToAll.apply(ds_engine.seq_parallel_group, input_tensor, scatter_idx, gather_idx,
                                            batch_dim_idx)

            # s2h_tensor check for the first all2all: compare with the expected ground truth
            d0_indices = torch.arange(s2h_tensor.shape[0]).reshape(-1, 1, 1, 1)
            d1_indices = torch.arange(s2h_tensor.shape[1]).reshape(1, -1, 1, 1)
            h_indices = torch.arange(s2h_tensor.shape[2]).reshape(1, 1, -1, 1)
            shard_list = get_shard_size_list(num_heads, groups._get_sequence_parallel_world_size())
            head_offset = sum(shard_list[:groups._get_sequence_parallel_rank()])
            s2h_truth = torch.zeros_like(s2h_tensor)
            s2h_truth[:] = seq_batch_heads_hash(d0_indices, d1_indices, h_indices, 0, 0, head_offset)

            assert torch.allclose(s2h_truth,
                                  s2h_tensor), f"s2h_tensor differs from the expected for sequence dim: {seq_dim}"
            #No op
            ### second all2all: head parallel to sequence parallel
            h2s_tensor = _SeqAllToAll.apply(ds_engine.seq_parallel_group, s2h_tensor, gather_idx, scatter_idx,
                                            batch_dim_idx)
            print(
                f'[{dist.get_rank()}] s={seq_dim} input: {input_tensor.shape} s2h: {s2h_tensor.shape} h2s_tensor: {h2s_tensor.shape}'
            )
            outputs.append(h2s_tensor)

        # Check outputs for the second all2all
        for i in range(0, len(outputs)):
            assert torch.allclose(inputs[i],
                                  outputs[i]), f"[{dist.get_rank()}]Outputs differ for sequence dim {seq_dims[i]}"
