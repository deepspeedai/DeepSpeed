# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

try:
    import torch
except ImportError:
    pass

from .builder import NPUOpBuilder


class NPUQuantizer:
    """
    Grouped quantization for the Zero++ quantized-communication paths.
    Production uses symmetric int4/int8; asymmetric is exercised by the
    shared quantizer test.

    Wire format per group: scale = 2**num_bits / (2 * max(|x|)),
    params = 1 / scale, q = round(x * scale), with round-half-to-even.
    fp32/fp64 division can be 1 ULP off here; the error is bounded and the
    wire format is transient, so checkpoints stay unquantized.

    The dequantize_*_experimental methods are unrelated to that wire format:
    they serve the inference weight-quantization path, which is asymmetric
    with a per-group scale and minimum.

    The inference int4 path as a whole requires CANN >= 9.0.0: the shared
    Quantizer._compress_uint8_to_uint4 packs with a uint8 left shift
    (aclnnLeftShifts), which older CANN releases do not provide. The
    dequantizers below do not use that operator and are version-independent;
    on older CANN the failure happens during quantization, before they run.
    torch_npu exposes the CANN version through
    torch_npu.utils.get_cann_version() and
    torch_npu.npu.utils._is_gte_cann_version(), but this module does not gate
    on it because the failing step is in shared code.
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
        # Round half to even, then clamp into the representable range.
        # Chained in-place form: xg is always a fresh float() copy here.
        q_min = -(1 << (num_bits - 1))
        q_max = (1 << (num_bits - 1)) - 1
        xg.mul_(scale.view(-1, 1)).round_().clamp_(q_min, q_max)
        return xg, stored

    @staticmethod
    def _quantize_groups_asymmetric(xg, num_bits):
        """Per-group asymmetric quantization; used by the shared quantizer test."""
        maxv = xg.amax(dim=1, keepdim=True)
        minv = xg.amin(dim=1, keepdim=True)
        scale = (float(1 << num_bits)) / (maxv - minv)
        scale.masked_fill_(maxv == minv, 1.0)
        stored = scale.reciprocal().view(-1)
        q_min = -(1 << (num_bits - 1))
        q_max = (1 << (num_bits - 1)) - 1
        offset = q_min - minv * scale
        xg.mul_(scale).add_(offset).round_().clamp_(q_min, q_max)
        return xg, stored, offset.view(-1)

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
        """Quantize fp16 input to int4 or int8.

        Returns packed codes and per-group params ([scale reciprocal, zero
        point] for asymmetric). int4 requires an even last dimension.
        """
        assert quant_type in (NPUQuantizer.Symmetric, NPUQuantizer.Asymmetric)
        assert num_bits in (4, 8), "num_bits must be 4 or 8"

        xg = NPUQuantizer._as_groups(input_vals, groups).float()  # fp16 -> fp32 is exact
        if quant_type == NPUQuantizer.Symmetric:
            q, stored = NPUQuantizer._quantize_groups(xg, num_bits)
            params = stored.view(groups, 1)
        else:
            q, stored, offset = NPUQuantizer._quantize_groups_asymmetric(xg, num_bits)
            params = torch.stack((stored, offset), dim=1)
        if num_bits == 4:
            assert input_vals.shape[-1] % 2 == 0, "int4 requires an even last dimension"
            packed = NPUQuantizer._pack_int4(q.reshape(-1))
            out_shape = tuple(input_vals.shape[:-1]) + (input_vals.shape[-1] // 2, )
            return packed.view(out_shape), params
        return q.to(torch.int8).view(input_vals.shape), params

    @staticmethod
    def _dequantize_fp32(quantized_data, params, groups, quant_type=None):
        if quant_type is None:
            quant_type = NPUQuantizer.Symmetric
        q = NPUQuantizer._as_groups(quantized_data, groups).float()
        p = params.contiguous().float()
        if quant_type == NPUQuantizer.Symmetric:
            return q.mul_(p.view(-1).view(groups, 1)).view(quantized_data.shape)
        assert quant_type == NPUQuantizer.Asymmetric
        step = p[:, 0].view(groups, 1)
        offset = p[:, 1].view(groups, 1)
        return q.sub_(offset).mul_(step).view(quantized_data.shape)

    @staticmethod
    def dequantize(quantized_data, params, groups, num_bits, quant_type):
        """Inverse of quantize; int4 packed input has its last dimension doubled."""
        assert quant_type in (NPUQuantizer.Symmetric, NPUQuantizer.Asymmetric)
        if num_bits == 4:
            out_shape = None
            if quantized_data.dim() == 1:
                packed = quantized_data
            else:
                out_shape = list(quantized_data.shape)
                out_shape[-1] *= 2
                packed = quantized_data.reshape(-1)
            vals = NPUQuantizer._unpack_int4(packed)
            out_elems = vals.numel()
            assert out_elems % groups == 0, "unpacked numel must be divisible by groups"
            q = vals.float().view(groups, out_elems // groups)
            p = params.contiguous().float()
            if quant_type == NPUQuantizer.Symmetric:
                out = q.mul_(p.view(-1).view(groups, 1)).half()
            else:
                out = q.sub_(p[:, 1].view(groups, 1)).mul_(p[:, 0].view(groups, 1)).half()
            return out.view(-1) if out_shape is None else out.view(*out_shape)
        assert num_bits == 8
        return NPUQuantizer._dequantize_fp32(quantized_data, params, groups, quant_type).half()

    @staticmethod
    def dequantize_fp32(quantized_data, params, groups, num_bits, quant_type):
        """Same, but keeps fp32 (no fp16 rounding in the middle)."""
        assert quant_type in (NPUQuantizer.Symmetric, NPUQuantizer.Asymmetric)
        assert num_bits == 8
        return NPUQuantizer._dequantize_fp32(quantized_data, params, groups, quant_type)

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
    def _dequantize_inference_groups(values, scale_buffer, min_val_buffer, num_group, group_size):
        """Asymmetric inference dequantization: value / scale + min, in fp16.

        The inference weight-quantization path stores one scale and one minimum
        per group and expects fp16 output, matching DeQuantizer in
        deepspeed/inference/quantization/utils.py. The arithmetic is in-place
        to keep the peak memory close to one fp16 copy of the weights.
        """
        scale = scale_buffer.reshape(num_group, 1).to(torch.float16)
        min_val = min_val_buffer.reshape(num_group, 1).to(torch.float16)
        values = values.reshape(num_group, group_size).to(torch.float16)
        return values.div_(scale).add_(min_val)

    _int4_supported = None

    @staticmethod
    def _cann_supports_int4():
        """Return whether CANN provides aclnnLeftShifts for the int4 packer.

        The int4 inference path as a whole needs CANN >= 9.0.0. Returns True
        when the version cannot be determined, so the normal path reports its
        own error instead of being blocked by a failed probe.
        """
        cached = NPUQuantizer._int4_supported
        if cached is not None:
            return cached
        supported = True
        try:
            from torch_npu.npu.utils import _is_gte_cann_version
            supported = bool(_is_gte_cann_version('9.0.0', module='CANN'))
        except Exception:
            try:
                from torch_npu.utils import get_cann_version
                parts = []
                for token in get_cann_version().split('.'):
                    digits = ''.join(ch for ch in token if ch.isdigit())
                    parts.append(int(digits) if digits else 0)
                supported = tuple(parts[:2]) >= (9, 0)
            except Exception:
                supported = True
        NPUQuantizer._int4_supported = supported
        return supported

    @staticmethod
    def _check_int4_inference_support():
        if NPUQuantizer._cann_supports_int4():
            return
        try:
            from torch_npu.utils import get_cann_version
            version = get_cann_version()
        except Exception:
            version = 'unknown'
        raise RuntimeError(f"int4 inference quantization on Ascend requires CANN >= 9.0.0 "
                           f"for aclnnLeftShifts; detected CANN {version}. "
                           f"Upgrade CANN or use int8 quantization.")

    @staticmethod
    def dequantize_int4_to_half_experimental(data_in, scale_buffer, min_val_buffer, num_group, group_size):
        """Unpack the inference int4 wire format and dequantize to fp16.

        Each byte holds two unsigned 4-bit values: the high nibble is the even
        element and the low nibble the odd element. This is the inverse of
        Quantizer._compress_uint8_to_uint4 (inference/quantization/utils.py).
        The nibbles stay uint8 so the temporary is one byte per element.
        """
        NPUQuantizer._check_int4_inference_support()
        packed = data_in.reshape(-1)
        values = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=data_in.device)
        values[0::2] = packed >> 4
        values[1::2] = packed & 0x0F
        return NPUQuantizer._dequantize_inference_groups(values, scale_buffer, min_val_buffer, num_group, group_size)

    @staticmethod
    def dequantize_int8_to_half_experimental(data_in, scale_buffer, min_val_buffer, num_group, group_size):
        """Dequantize the inference int8 wire format to fp16."""
        return NPUQuantizer._dequantize_inference_groups(data_in, scale_buffer, min_val_buffer, num_group, group_size)


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
