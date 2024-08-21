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
