# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

try:
    import torch
    import torch_supa_ext.deepspeed  # noqa: F401 — registers torch.ops.deepspeed
except ImportError:
    pass

from .builder import SUPAOpBuilder


class SUPAQuantizer:
    """
    Quantizer wrapper for Biren SUPA GPUs.
    Delegates to torch.ops.deepspeed quantization kernels.
    """

    Symmetric = 0
    Asymmetric = 1

    @staticmethod
    def _op(name):
        """Return torch.ops.deepspeed.<name>, raising clearly if not registered."""
        import torch  # ensure torch is available at runtime
        if not hasattr(torch.ops, 'deepspeed') or not hasattr(torch.ops.deepspeed, name):
            raise RuntimeError(f"torch.ops.deepspeed.{name} is not available. "
                               "Ensure torch_supa_ext is built with quantization support and imported before use.")
        return getattr(torch.ops.deepspeed, name)

    @staticmethod
    def ds_quantize_fp16(vals, groups, bits):
        return SUPAQuantizer._op('ds_quantize_fp16')(vals, groups, bits)

    @staticmethod
    def ds_quantize_fp32(vals, groups, bits):
        return SUPAQuantizer._op('ds_quantize_fp32')(vals, groups, bits)

    @staticmethod
    def ds_sr_quantize_fp16(vals, groups, bits):
        return SUPAQuantizer._op('ds_sr_quantize_fp16')(vals, groups, bits)

    @staticmethod
    def ds_sr_quantize_fp32(vals, groups, bits):
        return SUPAQuantizer._op('ds_sr_quantize_fp32')(vals, groups, bits)

    @staticmethod
    def ds_quantize_asym_fp16(vals, groups, bits):
        return SUPAQuantizer._op('ds_quantize_asym_fp16')(vals, groups, bits)

    @staticmethod
    def ds_quantize_asym_fp32(vals, groups, bits):
        return SUPAQuantizer._op('ds_quantize_asym_fp32')(vals, groups, bits)

    @staticmethod
    def ds_sr_quantize_asym_fp16(vals, groups, bits):
        return SUPAQuantizer._op('ds_sr_quantize_asym_fp16')(vals, groups, bits)

    @staticmethod
    def ds_sr_quantize_asym_fp32(vals, groups, bits):
        return SUPAQuantizer._op('ds_sr_quantize_asym_fp32')(vals, groups, bits)

    @staticmethod
    def quantize(input_vals, groups, num_bits, quant_type):
        return SUPAQuantizer._op('quantize')(input_vals, groups, num_bits, int(quant_type))

    @staticmethod
    def dequantize(quantized_data, params, groups, num_bits, quant_type):
        return SUPAQuantizer._op('dequantize')(quantized_data, params, groups, num_bits, int(quant_type))

    @staticmethod
    def dequantize_fp32(quantized_data, params, groups, num_bits, quant_type):
        return SUPAQuantizer._op('dequantize_fp32')(quantized_data, params, groups, num_bits, int(quant_type))

    @staticmethod
    def dequantize_int4_to_half_experimental(data_in, scale_buffer, min_val_buffer, num_group, group_size):
        return SUPAQuantizer._op('dequantize_int4_to_half_experimental')(data_in, scale_buffer, min_val_buffer,
                                                                         num_group, group_size)

    @staticmethod
    def dequantize_int8_to_half_experimental(data_in, scale_buffer, min_val_buffer, num_group, group_size):
        return SUPAQuantizer._op('dequantize_int8_to_half_experimental')(data_in, scale_buffer, min_val_buffer,
                                                                         num_group, group_size)

    @staticmethod
    def swizzle_quant(input_vals, groups, num_bits, quant_type, pipeline_size, nodes, devices_per_node):
        return SUPAQuantizer._op('swizzle_quant')(input_vals, groups, num_bits, int(quant_type), pipeline_size, nodes,
                                                  devices_per_node)

    @staticmethod
    def quantized_reduction(input_vals, input_scales, in_groups, out_groups, num_bits, quant_type, devices_per_node):
        return SUPAQuantizer._op('quantized_reduction')(input_vals, input_scales, in_groups, out_groups, num_bits,
                                                        int(quant_type), devices_per_node)

    @staticmethod
    def loco_swizzle_quant(input_vals, error_feedback, err_beta, groups, num_bits, quant_type, pipeline_size, nodes,
                           devices_per_node):
        return SUPAQuantizer._op('loco_swizzle_quant')(input_vals, error_feedback, err_beta, groups, num_bits,
                                                       int(quant_type), pipeline_size, nodes, devices_per_node)

    @staticmethod
    def loco_quantized_reduction(input_vals, input_scales, error_feedback, err_beta, in_groups, out_groups, num_bits,
                                 quant_type, devices_per_node):
        return SUPAQuantizer._op('loco_quantized_reduction')(input_vals, input_scales,
                                                             error_feedback, err_beta, in_groups, out_groups, num_bits,
                                                             int(quant_type), devices_per_node)


class QuantizerBuilder(SUPAOpBuilder):
    BUILD_VAR = "DS_BUILD_QUANTIZER"
    NAME = "quantizer"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.quantizer.{self.NAME}_op'

    def sources(self):
        return []

    def load(self, verbose=True):
        return SUPAQuantizer

    def is_compatible(self, verbose=False):
        return hasattr(torch.ops, 'deepspeed') and hasattr(torch.ops.deepspeed, 'ds_quantize_fp16')
