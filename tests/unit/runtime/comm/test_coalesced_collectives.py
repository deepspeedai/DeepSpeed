# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
unit tests for coalesced collectives
"""

import torch
import deepspeed
import deepspeed.comm as dist
from deepspeed.runtime.comm.coalesced_collectives import reduce_scatter_coalesced, all_to_all_quant_reduce
from deepspeed.accelerator import get_accelerator
import pytest

from unit.common import DistributedTest


class TestReduceScatterCoalesced(DistributedTest):
    world_size = 2

    def test_single_input(self):
        input = torch.full((6, ), dist.get_rank(), dtype=torch.half, device=get_accelerator().current_device_name())

        (output, ) = reduce_scatter_coalesced([input], dist.get_world_group())

        assert output.shape == (3, )
        assert torch.allclose(output, torch.full_like(output, 0.5))

    def test_two_inputs(self):
        tensor_kwargs = {"device": get_accelerator().current_device_name(), "dtype": torch.half}
        inputs = [
            dist.get_rank() * torch.arange(0, 6, **tensor_kwargs),
            dist.get_rank() * torch.arange(6, 9, **tensor_kwargs),
        ]

        output1, output2 = reduce_scatter_coalesced(inputs, dist.get_world_group())

        if dist.get_rank() == 0:
            assert output1.shape == (3, )
            assert torch.allclose(output1, torch.arange(0, 3, **tensor_kwargs) / 2)
            assert output2.shape == (2, )
            assert torch.allclose(output2, torch.arange(6, 8, **tensor_kwargs) / 2)
        elif dist.get_rank() == 1:
            assert output1.shape == (3, )
            assert torch.allclose(output1, torch.arange(3, 6, **tensor_kwargs) / 2)
            assert output2.shape == (1, )
            assert torch.allclose(output2, torch.arange(8, 9, **tensor_kwargs) / 2)


class TestReduceScatterCoalescedTensorSmallerThanWorldSize(DistributedTest):
    world_size = 2

    def test(self):
        input = torch.zeros((1, ), dtype=torch.half, device=get_accelerator().current_device_name())

        (output, ) = reduce_scatter_coalesced([input], dist.get_world_group())

        if dist.get_rank() == 0:
            assert output.shape == (1, )
            assert torch.allclose(output, torch.zeros_like(output))
        elif dist.get_rank() == 1:
            assert output.shape == (0, )


# Currently we cannot test all_to_all_quant_reduce in non-fallback cases because we don't support multinodes tests.
class TestAllToAllQuantReduceFallback(DistributedTest):
    world_size = 2

    def test_1d_tensor(self):
        # case 1: 1D tensor
        input = torch.zeros((10, ), dtype=torch.half, device=get_accelerator().current_device_name())
        from deepspeed.ops.op_builder import QuantizerBuilder
        if not deepspeed.ops.__compatible_ops__[QuantizerBuilder.NAME]:
            pytest.skip("QuantizerBuilder is not implemented")
        output = all_to_all_quant_reduce([input], {})[0]

        if dist.get_rank() == 0:
            assert output.shape == (5, )
            assert torch.allclose(output, torch.zeros_like(output))
        elif dist.get_rank() == 1:
            assert output.shape == (5, )
            assert torch.allclose(output, torch.zeros_like(output))

    def test_non_divisible(self):
        # case 2: tensor size not divisible by global_world_size
        input = torch.zeros((7, 7), dtype=torch.half, device=get_accelerator().current_device_name())
        from deepspeed.ops.op_builder import QuantizerBuilder
        if not deepspeed.ops.__compatible_ops__[QuantizerBuilder.NAME]:
            pytest.skip("QuantizerBuilder is not implemented")
        output = all_to_all_quant_reduce([input], {})[0]

        if dist.get_rank() == 0:
            assert output.shape == (25, )
            assert torch.allclose(output, torch.zeros_like(output))
        elif dist.get_rank() == 1:
            assert output.shape == (24, )
            assert torch.allclose(output, torch.zeros_like(output))


class TestLocoQuantized(DistributedTest):

    world_size = 1

    @pytest.mark.parametrize("num_bits", [4, 8])
    @pytest.mark.parametrize("tensor_size", [(16, 16), (64, 64)])
    @pytest.mark.parametrize("devices_per_node", [4, 8])
    def test_loco_quantized_reduction(self, num_bits, tensor_size, devices_per_node):
        from deepspeed.ops.op_builder import QuantizerBuilder
        if not deepspeed.ops.__compatible_ops__[QuantizerBuilder.NAME]:
            pytest.skip("QuantizerBuilder is not implemented")

        quantizer_module = QuantizerBuilder().load()

        tensor = torch.randn(tensor_size, device='cuda', dtype=torch.half)

        num_nodes = 2  # Fake world size
        total_elements = tensor.numel()
        total_devices = devices_per_node * num_nodes
        num_groups = max(tensor.shape[0], tensor.shape[1], total_devices)

        # Initialize error_feedback tensor
        error_feedback = torch.randn(tensor_size, device=tensor.device, dtype=tensor.dtype)
        error_feedback_ori = error_feedback.clone()
        # Swizzle the original tensor
        tensor_reshaped = tensor.reshape(num_nodes, devices_per_node, total_elements // total_devices)
        swizzled_tensor = tensor_reshaped.permute(1, 0, 2).reshape(tensor.size())

        # Perform loco_swizzle_quant
        output, scales = quantizer_module.loco_swizzle_quant(tensor, error_feedback, 0.0, num_groups, num_bits,
                                                             quantizer_module.Symmetric, 1, num_nodes,
                                                             devices_per_node)

        # Compare swizzled_tensor with the output of loco_swizzle_quant
        dequantized = quantizer_module.dequantize(output, scales, scales.numel(), num_bits,
                                                  quantizer_module.Symmetric).view(tensor.size())

        assert torch.allclose(swizzled_tensor + error_feedback_ori, dequantized + error_feedback)

        # Calculate elements per group and groups per partition
        elements_per_group = total_elements // num_groups
        groups_per_partition = num_groups // devices_per_node

        # Reshape dequantized data to match the grouping in loco_quantized_reduction
        dequantized_reshaped = dequantized.view(devices_per_node, groups_per_partition, elements_per_group)

        # Perform reduction across devices_per_node dimension
        reduced_dequantized = dequantized_reshaped.cumsum(dim=0)[-1]
        # Initialize error_feedback tensor
        error_feedback = torch.randn(reduced_dequantized.shape, device=tensor.device, dtype=dequantized.dtype)
        error_feedback_ori = error_feedback.clone()

        # perform loco_quantized_reduction
        output, scales = quantizer_module.loco_quantized_reduction(output, scales, error_feedback, 0.0, num_groups,
                                                                   num_groups // devices_per_node, num_bits,
                                                                   quantizer_module.Symmetric, devices_per_node)

        dequantized_reduced = quantizer_module.dequantize(output, scales, scales.numel(), num_bits,
                                                          quantizer_module.Symmetric).view(error_feedback.size())

        assert torch.allclose(reduced_dequantized + error_feedback_ori, dequantized_reduced + error_feedback)
