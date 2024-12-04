# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import torch.nn.functional as F
import deepspeed.comm as dist
from deepspeed import initialize
from transformers import AutoModel
from unit.common import DistributedTest
from deepspeed.sequence.layer import _SeqAllToAll
from deepspeed.sequence.fpdt_layer import _FPDTGPUOffloadingAttentionImpl_, FPDT_InputConstruct
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


@pytest.mark.parametrize("d0", [4, 1])  #batch dimension
@pytest.mark.parametrize("d1", [2048, 8192])  #sequence dimension
@pytest.mark.parametrize("chunk_size", [128, 256])  #size of chunk
@pytest.mark.parametrize("num_heads", [8, 4])
@pytest.mark.parametrize("head_dim", [32])
class TestFPDTAttention(DistributedTest):

    def test_FPDT_attention_offloading_output_consistency(self, d0: int, d1: int, chunk_size: int, head_dim: int,
                                                          num_heads: int) -> None:
        skip_on_arch(min_arch=8)
        world_size = 2

        try:
            from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
        except ImportError:
            _flash_attn_forward = None
            _flash_attn_backward = None

        if _flash_attn_forward is None or _flash_attn_backward is None:
            pytest.skip("Flash Attention is not available.")

        model = AutoModel.from_pretrained('bert-base-uncased')
        ds_engine, _, _, _ = initialize(
            model=model,
            config_params={
                "train_batch_size": 8,
                "data_parallel_size": 1,
                "sequence_parallel_size": world_size
            },
        )
        #3D tensor : l, b, d
        dim = head_dim * num_heads

        seed = 42
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)

        input_tensor = torch.randn(d1, d0, dim, device=ds_engine.device, dtype=torch.half)  # l, b, d
        spg = ds_engine.seq_parallel_group

        dist.broadcast(input_tensor, src=0, group=spg)

        class args:

            def __init__(self):
                self.ds_sequence_parallel_fpdt_chunk_size = chunk_size

        fpdt_input_tensor = FPDT_InputConstruct(input_tensor.permute(1, 0, 2), None, None, None, None, args(),
                                                world_size, dist.get_rank()).generate()[0].permute(1, 0, 2)

        if dist.get_rank() == 0:
            qkv_linear_weight = torch.nn.Parameter(
                torch.empty(dim + 2 * dim, dim, device=dist.get_rank(), dtype=torch.half))
            torch.nn.init.normal_(qkv_linear_weight, mean=0.0, std=0.02)

            qkv_linear_bias = torch.nn.Parameter(torch.empty(dim + 2 * dim, device=dist.get_rank(), dtype=torch.half))
            torch.nn.init.normal_(qkv_linear_bias, mean=0.0, std=0.02)
        else:
            qkv_linear_weight = torch.nn.Parameter(
                torch.empty(dim + 2 * dim, dim, device=dist.get_rank(), dtype=torch.half))
            qkv_linear_bias = torch.nn.Parameter(torch.empty(dim + 2 * dim, device=dist.get_rank(), dtype=torch.half))

        dist.broadcast(qkv_linear_weight, src=0, group=spg)
        dist.broadcast(qkv_linear_bias, src=0, group=spg)

        num_chunks_attn = fpdt_input_tensor.shape[0] * dist.get_world_size(spg) // chunk_size
        fpdt_output = _FPDTGPUOffloadingAttentionImpl_.apply(fpdt_input_tensor, None, None, None, spg, 2, 0, dim, dim,
                                                             head_dim, dim, qkv_linear_weight, qkv_linear_bias, 0,
                                                             num_chunks_attn, True)

        # baseline
        qkv = torch.matmul(input_tensor, qkv_linear_weight.t()) + qkv_linear_bias
        q = qkv[:, :, :dim].contiguous().reshape(qkv.shape[0], qkv.shape[1], -1, head_dim).permute(1, 2, 0,
                                                                                                   3).contiguous()
        k = qkv[:, :, dim:dim * 2].contiguous().reshape(qkv.shape[0], qkv.shape[1], -1,
                                                        head_dim).permute(1, 2, 0, 3).contiguous()
        v = qkv[:, :, dim * 2:dim * 3].contiguous().reshape(qkv.shape[0], qkv.shape[1], -1,
                                                            head_dim).permute(1, 2, 0,
                                                                              3).contiguous()  # b, nhead, l, d

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dim, dtype=torch.half))

        causal_mask = torch.triu(torch.ones(d1, d1, device=ds_engine.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v).permute(0, 2, 1, 3)

        baseline_output_shuffled = FPDT_InputConstruct(output, None, None, None, None, args(), world_size,
                                                       dist.get_rank()).generate()[0]  # b, l, n, d

        assert torch.allclose(
            fpdt_output, baseline_output_shuffled, rtol=0.01, atol=0.1
        ), f"rank {dist.get_rank()}, sp size: {dist.get_world_size(spg)}, input_tensor: {input_tensor.shape}, fpdt_input_tensor: {fpdt_input_tensor.shape}, fpdt_output: {fpdt_output.shape},            baseline_output_shuffled: {baseline_output_shuffled.shape},{torch.max(torch.abs(fpdt_output - baseline_output_shuffled))}"
