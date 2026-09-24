# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from .builder import NPUOpBuilder


class NPUQuantizer:
    """
    Grouped symmetric quantization for the Zero++ quantized-communication
    paths (quantized weight all-gather in partition_parameters.py and the
    quantized gradient all-to-all in coalesced_collectives.py).

    Wire format, per group:
        scale   = 2**num_bits / (2 * max(|x|))   (1.0 when max == 0)
        params  = 1 / scale                      (fp32, the value that is sent)
        q       = trunc(x * scale), clamped to [-2**(b-1), 2**(b-1)-1]

    Truncation is toward zero: the reference implementation casts to an
    integer type, so a round-to-nearest "simplification" would silently
    change every quantized value. tests/unit/ops/quantizer/test_npu_quantizer.py
    pins this against an independent per-group reference.

    Known deviation: fp32/fp64 division on this accelerator is approximate
    (1 ulp off on ~5% of values), so params can differ from a correctly
    rounded computation by 1 ulp. This is bounded and safe: every rank in a
    run computes the same value, the 1-ulp scale error (~1e-7 relative) is
    five orders below the quantization error (~1e-2), and the wire format is
    transient (checkpoints store unquantized weights).

    Native operators: torch_npu's npu_quantize family rounds half-to-even
    and exposes no truncation mode, so no native op can honor the wire
    format today. If CANN later adds one, _quantize_groups is the single
    swap point.
    """

    Symmetric = 0
    Asymmetric = 1

    @staticmethod
    def _as_groups(tensor, groups):
        # The quantized tensor is processed as one flat run of per-group chunks.
        flat = tensor.contiguous().view(-1)
        assert flat.numel() % groups == 0, "numel must be divisible by groups"
        return flat.view(groups, flat.numel() // groups)

    @staticmethod
    def _quantize_groups(xg, num_bits):
        """Per-group symmetric quantization of an fp32 [groups, elems] tensor.

        Returns the quantized values (int32, in the representable range) and
        the stored per-group scale reciprocals. This is the shared math behind
        quantize, swizzle_quant, and quantized_reduction.
        """
        maxv = xg.abs().max(dim=1).values
        # scale = 2**num_bits / (2 * max), computed in fp32 like the reference.
        # Dividing by zero yields inf for empty groups; the mask below
        # restores the reference 1.0, so this form agrees bitwise with the
        # explicit where() (verified on 910B4).
        scale = (float(1 << num_bits)) / (2.0 * maxv)
        scale.masked_fill_(maxv == 0, 1.0)
        # The stored/sent value is the reciprocal of the scale; dequantize
        # multiplies it back, so the round trip is q * (1/scale) == q * scale.
        stored = scale.reciprocal().view(-1)
        # Truncate toward zero, then clamp into the representable range.
        # Chained in-place form: xg is always a fresh float() copy here.
        q_min = -(1 << (num_bits - 1))
        q_max = (1 << (num_bits - 1)) - 1
        xg.mul_(scale.view(-1, 1)).trunc_().clamp_(q_min, q_max)
        return xg, stored

    @staticmethod
    def _pack_int4(q_flat):
        """Pack values in [-8, 7] two per byte: even element in the low nibble,
        odd element in the high nibble (the reference layout)."""
        pairs = q_flat.to(torch.int64).reshape(-1, 2)
        low = pairs[:, 0] & 0x0F
        high = pairs[:, 1] & 0x0F
        return (low | (high << 4)).to(torch.int8)

    @staticmethod
    def _unpack_int4(packed):
        """Unpack int8 bytes into int64 nibbles, sign-extending each one."""
        p = packed.to(torch.int64)
        low = p & 0x0F
        high = (p >> 4) & 0x0F
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
        # even element is the low nibble: [b0_low, b0_high, b1_low, ...]
        return torch.stack([low, high], dim=-1).reshape(-1)

    @staticmethod
    def quantize(input_vals, groups, num_bits, quant_type):
        """Grouped symmetric int8 quantization.

        Args:
            input_vals: fp16 tensor (the hpZ path converts bf16 weights to fp16
                before calling in, so fp16 is the only dtype that appears here).
            groups: number of quantization groups; numel must divide evenly.
            num_bits: must be 8 (int4 uses a packed byte layout, not implemented).
            quant_type: must be Symmetric (the only mode the hpZ/qgZ paths use).

        Returns:
            (int8 tensor with input_vals' shape, fp32 params of shape [groups, 1])
        """
        assert quant_type == NPUQuantizer.Symmetric, "only Symmetric quantization is implemented"
        assert num_bits == 8, "only int8 is implemented (int4 uses a packed byte layout)"

        xg = NPUQuantizer._as_groups(input_vals, groups).float()  # fp16 -> fp32 is exact
        q, stored = NPUQuantizer._quantize_groups(xg, num_bits)
        return q.to(torch.int8).view(input_vals.shape), stored.view(groups, 1)

    @staticmethod
    def _dequantize_fp32(quantized_data, params, groups):
        q = NPUQuantizer._as_groups(quantized_data, groups).float()
        p = params.contiguous().view(-1).float()
        return q.mul_(p.view(groups, 1)).view(quantized_data.shape)

    @staticmethod
    def dequantize(quantized_data, params, groups, num_bits, quant_type):
        """Inverse of quantize; returns fp16, matching the hpZ/qgZ call sites.

        For int4 the input is the flat packed byte tensor and the output is
        the flat unpacked element tensor (2 elements per byte), which is what
        the qgZ path receives and expects.
        """
        assert quant_type == NPUQuantizer.Symmetric
        if num_bits == 4:
            assert quantized_data.dim() == 1, "the int4 wire format is a flat packed byte tensor"
            vals = NPUQuantizer._unpack_int4(quantized_data)
            out_elems = vals.numel()
            assert out_elems % groups == 0, "unpacked numel must be divisible by groups"
            q = vals.float().view(groups, out_elems // groups)
            p = params.contiguous().view(-1).float()
            return (q * p.view(groups, 1)).view(-1).half()
        assert num_bits == 8
        return NPUQuantizer._dequantize_fp32(quantized_data, params, groups).half()

    @staticmethod
    def dequantize_fp32(quantized_data, params, groups, num_bits, quant_type):
        """Same, but keeps fp32 (no fp16 rounding in the middle)."""
        assert quant_type == NPUQuantizer.Symmetric
        assert num_bits == 8
        return NPUQuantizer._dequantize_fp32(quantized_data, params, groups)

    @staticmethod
    def swizzle_quant(input_vals, groups, num_bits, quant_type, pipeline_size, nodes, devices_per_node):
        """Grouped quantization with the output groups permuted into the qgZ
        two-stage all-to-all layout.

        Input group g = (z * pipeline_size + y) * contiguous_groups + x, where
        z is the partition id, y the pipeline slice, and x the group within
        both, lands in output slot
        y * partitions + (z % devices_per_node) * nodes + z // devices_per_node.
        The permutation makes each destination's data one contiguous run, so
        the all-to-all can move it as whole chunks. Scales are stored in the
        same permuted order.

        Returns:
            (flat packed tensor (2 elements per byte for int4),
             fp32 scales of shape [groups, 1])
        """
        assert quant_type == NPUQuantizer.Symmetric, "only Symmetric quantization is implemented"
        assert num_bits in (4, 8), "swizzle_quant supports int4 and int8"
        partitions = nodes * devices_per_node
        numel = input_vals.numel()
        assert numel % groups == 0, "numel must be divisible by groups"
        assert groups % partitions == 0, "groups must cover all partitions"
        groups_per_partition = groups // partitions
        assert groups_per_partition % pipeline_size == 0, "partitions must split evenly into pipeline slices"
        contiguous_groups = groups_per_partition // pipeline_size
        assert numel // groups % (8 // num_bits) == 0, "group size must fit the packed layout"

        xg = NPUQuantizer._as_groups(input_vals, groups).float()
        q, stored = NPUQuantizer._quantize_groups(xg, num_bits)

        # output group index for each input group (the reference index math:
        # block_rank -> output_partition -> out_block_rank)
        group_idx = torch.arange(groups, device=input_vals.device)
        z = group_idx // contiguous_groups // pipeline_size  # partition id
        y = group_idx // contiguous_groups % pipeline_size  # pipeline slice
        x = group_idx % contiguous_groups
        slot = y * partitions + (z % devices_per_node) * nodes + z // devices_per_node
        out_group = slot * contiguous_groups + x
        perm = torch.argsort(out_group)  # input rows in output order

        q_perm = q.index_select(0, perm)
        stored_perm = stored.index_select(0, perm)
        if num_bits == 4:
            packed = NPUQuantizer._pack_int4(q_perm.reshape(-1))
        else:
            packed = q_perm.reshape(-1).to(torch.int8)
        return packed, stored_perm.view(groups, 1)

    @staticmethod
    def quantized_reduction(input_vals, input_scales, in_groups, out_groups, num_bits, quant_type, devices_per_node):
        """Dequantize, sum across the intra-node ranks, and re-quantize.

        input_vals holds the packed data of devices_per_node ranks back to
        back (the intra-node all-to-all result), each with its own scale run
        in input_scales. The per-rank sum is accumulated in fp16 and
        re-quantized with out_groups for the inter-node all-to-all.

        Returns:
            (flat packed tensor of one rank's worth of elements,
             fp32 scales of shape [out_groups, 1])
        """
        assert quant_type == NPUQuantizer.Symmetric, "only Symmetric quantization is implemented"
        assert num_bits in (4, 8), "quantized_reduction supports int4 and int8"
        pack = 8 // num_bits
        bytes_total = input_vals.numel()
        assert bytes_total % devices_per_node == 0, "data must split evenly across ranks"
        bytes_per_tensor = bytes_total // devices_per_node
        assert in_groups % devices_per_node == 0, "scales must split evenly across ranks"
        groups_per_tensor = in_groups // devices_per_node
        assert bytes_per_tensor % groups_per_tensor == 0, "group size must be whole bytes"
        elems_per_in_group = bytes_per_tensor // groups_per_tensor  # in packed bytes

        # per-element scale index within one rank's chunk (in unpacked elements)
        scale_idx = torch.arange(bytes_per_tensor * pack, device=input_vals.device) // (elems_per_in_group * pack)

        # accumulate the dequantized halves, like the reference accumulation dtype
        acc = None
        for j in range(devices_per_node):
            chunk = input_vals[j * bytes_per_tensor:(j + 1) * bytes_per_tensor]
            vals = NPUQuantizer._unpack_int4(chunk) if num_bits == 4 else chunk.to(torch.int64).reshape(-1)
            scales_j = input_scales[j * groups_per_tensor:(j + 1) * groups_per_tensor].view(-1).float()
            deq = (vals.float() * scales_j[scale_idx]).half()
            acc = deq if acc is None else acc + deq

        # re-quantize the summed values with out_groups
        assert bytes_per_tensor % out_groups == 0, "sum must split into out_groups"
        elems_per_out_group = bytes_per_tensor // out_groups * pack  # in unpacked elements
        q, stored = NPUQuantizer._quantize_groups(acc.reshape(out_groups, elems_per_out_group).float(), num_bits)
        if num_bits == 4:
            packed = NPUQuantizer._pack_int4(q.reshape(-1))
        else:
            packed = q.reshape(-1).to(torch.int8)
        return packed, stored.view(out_groups, 1)

    @staticmethod
    def dequantize_int4_to_half_experimental(data_in, scale_buffer, min_val_buffer, num_group, group_size):
        raise NotImplementedError("experimental fp_quantizer path is not needed for hpZ")

    @staticmethod
    def dequantize_int8_to_half_experimental(data_in, scale_buffer, min_val_buffer, num_group, group_size):
        raise NotImplementedError("experimental fp_quantizer path is not needed for hpZ")


class QuantizerBuilder(NPUOpBuilder):
    BUILD_VAR = "DS_BUILD_QUANTIZER"
    NAME = "quantizer"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.quantizer.{self.NAME}_op'

    def sources(self):
        # The quantizer is a pure-torch implementation; there is nothing to compile.
        return []

    def load(self, verbose=True):
        return NPUQuantizer

    def is_compatible(self, verbose=False):
        return True
