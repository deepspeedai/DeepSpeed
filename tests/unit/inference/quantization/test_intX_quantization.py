# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import numpy as np
import torch
import torch.nn as nn
from unit.common import DistributedTest
from deepspeed.accelerator import get_accelerator
from deepspeed.inference.quantization.quantization import _init_group_wise_weight_quantization
from deepspeed.inference.quantization.utils import Quantizer, DeQuantizer
from deepspeed.inference.quantization.layers import QuantizedLinear
from deepspeed.runtime.utils import required_torch_version
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers import AutoConfig, OPTConfig, AutoModel
import pytest
from collections import OrderedDict
from typing import Dict

device = get_accelerator().device_name() if get_accelerator().is_available() else 'cpu'

if not required_torch_version(min_version=1.11):
    pytest.skip("torch.Tensor.bitwise_left_shift in INT4 quantizer needs torch 1.11 or above.",
                allow_module_level=True)


def reset_random(seed=1234):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    get_accelerator().manual_seed_all(seed)


def quantization_test_helper(pre_quant_type: torch.dtype, num_bits: int):
    reset_random()
    num_group = 1024 * 32
    group_size = 64
    quantization_config = {'num_bits': num_bits, 'group_size': group_size, 'group_dim': 1, 'symmetric': False}

    quantizer = Quantizer(config=quantization_config)
    dequantizer = DeQuantizer(config=quantization_config, dtype=pre_quant_type)

    data = torch.randn(num_group, group_size, dtype=pre_quant_type, device=device)

    quantized_data, scale_buf, min_vals = quantizer.quantize(data)
    dequantized_data = dequantizer.dequantize(quantized_data, scale_buf, min_vals)

    max_diff = torch.max(torch.abs(data - dequantized_data))
    mean_diff = torch.mean(torch.abs(data - dequantized_data))

    # This threshold value is emperically selected.
    assert mean_diff < 0.15 and max_diff < 0.5, f'Numeric error exceed threshold, mean diff {mean_diff} (threshold 0.15), max diff {max_diff} (threshold 0.5)'


def zero3_post_init_quantization_test_helper(cpu_offload: bool, nvme_offload: bool, bits: int):
    import deepspeed
    from transformers.integrations.deepspeed import HfDeepSpeedConfig

    def get_zero3_ds_config(hf_config: OPTConfig, cpu_offload: bool, nvme_offload: bool, bits: int) -> Dict:
        GB = 1 << 30

        ds_config = {
            "fp16": {
                "enabled": True,
            },
            "zero_optimization": {
                "stage": 3,
                "stage3_prefetch_bucket_size": 2 * hf_config.hidden_size * hf_config.hidden_size,
                "stage3_param_persistence_threshold": hf_config.hidden_size,
                "stage3_max_live_parameters": 2 * hf_config.hidden_size * hf_config.hidden_size
            },
            "steps_per_print": 2000,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False,
            'weight_quantization': {
                'post_init_quant': {
                    'fc': {
                        'num_bits': bits,
                        'group_size': 32,
                        'group_dim': 1,
                        'symmetric': False
                    },
                    'self_attn.q_proj': {
                        'num_bits': bits,
                        'group_size': 32,
                        'group_dim': 1,
                        'symmetric': False
                    },
                    'self_attn.k_proj': {
                        'num_bits': bits,
                        'group_size': 32,
                        'group_dim': 1,
                        'symmetric': False
                    },
                    'self_attn.v_proj': {
                        'num_bits': bits,
                        'group_size': 32,
                        'group_dim': 1,
                        'symmetric': False
                    },
                    'self_attn.out_proj': {
                        'num_bits': bits,
                        'group_size': 32,
                        'group_dim': 1,
                        'symmetric': False
                    },
                    'lm_head': {
                        'num_bits': bits,
                        'group_size': 32,
                        'group_dim': 1,
                        'symmetric': False
                    },
                    'embed_tokens': {
                        'num_bits': bits,
                        'group_size': 32,
                        'group_dim': 1,
                        'symmetric': False
                    },
                }
            }
        }

        if cpu_offload:
            ds_config["zero_optimization"]["offload_param"] = dict(device="cpu", pin_memory=1)
        if nvme_offload:
            ds_config["zero_optimization"]["offload_param"] = dict(
                device="nvme",
                pin_memory=True,
                nvme_path='~/tmp_offload_dir',
                buffer_count=5,
                buffer_size=1 * GB,
            )
            ds_config["aio"] = {
                "block_size": 1048576,
                "queue_depth": 8,
                "thread_count": 1,
                "single_submit": False,
                "overlap_events": True,
            }

        return ds_config

    hf_config = AutoConfig.from_pretrained('facebook/opt-125m')
    ds_config = get_zero3_ds_config(hf_config=hf_config, cpu_offload=cpu_offload, nvme_offload=nvme_offload, bits=bits)

    input_ids = torch.ones(1, 16, dtype=torch.int32, device=device)
    attention_mask = torch.ones(1, 16, dtype=torch.float32, device=device)

    with torch.no_grad():
        ref_model = AutoModel.from_pretrained('facebook/opt-125m', torch_dtype=torch.float16).to(device)
        ref_model.eval()
        ref_output = ref_model(input_ids=input_ids, attention_mask=attention_mask)

    with torch.no_grad():
        dschf = HfDeepSpeedConfig(ds_config)
        model = AutoModel.from_pretrained('facebook/opt-125m', torch_dtype=torch.float16)
        model = model.eval()

        model = _init_group_wise_weight_quantization(model, ds_config)
        ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
        ds_engine.module.eval()
        model = ds_engine.module

    output = model(input_ids=input_ids, attention_mask=attention_mask)

    mean_diff = torch.mean(torch.abs(output.last_hidden_state - ref_output.last_hidden_state))

    # This threshold value is emperically selected.
    assert mean_diff < 0.4, f'Numeric error exceed threshold, relative error {mean_diff} (threshold 0.4)'


def zero3_quantized_initialization_test_helper(cpu_offload: bool, nvme_offload: bool, bits: int):
    import deepspeed
    from transformers.integrations.deepspeed import HfDeepSpeedConfig

    def get_zero3_ds_config(hf_config: OPTConfig, cpu_offload: bool, nvme_offload: bool, bits: int) -> Dict:
        GB = 1 << 30

        ds_config = {
            "fp16": {
                "enabled": True,
            },
            "zero_optimization": {
                "stage": 3,
                "stage3_prefetch_bucket_size": 2 * hf_config.hidden_size * hf_config.hidden_size,
                "stage3_param_persistence_threshold": hf_config.hidden_size,
                "stage3_max_live_parameters": 2 * hf_config.hidden_size * hf_config.hidden_size
            },
            "steps_per_print": 2000,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False,
            'weight_quantization': {
                'quantized_initialization': {
                    'num_bits': bits,
                    'group_size': 32,
                    'group_dim': 1,
                    'symmetric': False
                },
            }
        }

        if cpu_offload:
            ds_config["zero_optimization"]["offload_param"] = dict(device="cpu", pin_memory=1)
        if nvme_offload:
            ds_config["zero_optimization"]["offload_param"] = dict(
                device="nvme",
                pin_memory=True,
                nvme_path='~/tmp_offload_dir',
                buffer_count=5,
                buffer_size=1 * GB,
            )
            ds_config["aio"] = {
                "block_size": 1048576,
                "queue_depth": 8,
                "thread_count": 1,
                "single_submit": False,
                "overlap_events": True,
            }

        return ds_config

    hf_config = AutoConfig.from_pretrained('facebook/opt-125m')
    ds_config = get_zero3_ds_config(hf_config=hf_config, cpu_offload=cpu_offload, nvme_offload=nvme_offload, bits=bits)

    input_ids = torch.ones(1, 16, dtype=torch.int32, device=device)
    attention_mask = torch.ones(1, 16, dtype=torch.float32, device=device)

    with torch.no_grad():
        ref_model = AutoModel.from_pretrained('facebook/opt-125m', torch_dtype=torch.float16).to(device)
        ref_model.eval()
        ref_output = ref_model(input_ids=input_ids, attention_mask=attention_mask)

    with torch.no_grad():
        dschf = HfDeepSpeedConfig(ds_config)
        model = AutoModel.from_pretrained('facebook/opt-125m', torch_dtype=torch.float16)
        model = model.eval()
        ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
        ds_engine.module.eval()
        model = ds_engine.module

    output = model(input_ids=input_ids, attention_mask=attention_mask)

    mean_diff = torch.mean(torch.abs(output.last_hidden_state - ref_output.last_hidden_state))

    # This threshold value is emperically selected.
    assert mean_diff < 0.4, f'Numeric error exceed threshold, relative error {mean_diff} (threshold 0.4)'


@pytest.fixture(params=[4, 8], ids=["4bits", "8bits"])
def quantization_bits(request):
    return request.param


@pytest.fixture(params=[0, 1], ids=["0", "1"])
def group_dim(request):
    return request.param


class TestQuantizedInt(DistributedTest):

    def test_model_quantization(self, quantization_bits):
        reset_random()

        config = AutoConfig.from_pretrained('facebook/opt-125m')

        with torch.no_grad():
            model = OPTDecoderLayer(config).half().to(device)
            bits = quantization_bits

            ds_config = {
                'weight_quantization': {
                    'post_init_quant': {
                        'fc': {
                            'num_bits': bits,
                            'group_size': 64,
                            'group_dim': 0,
                            'symmetric': False
                        },
                        'self_attn.q_proj': {
                            'num_bits': bits,
                            'group_size': 64,
                            'group_dim': 0,
                            'symmetric': False
                        },
                        'self_attn.k_proj': {
                            'num_bits': bits,
                            'group_size': 64,
                            'group_dim': 0,
                            'symmetric': False
                        },
                        'self_attn.v_proj': {
                            'num_bits': bits,
                            'group_size': 64,
                            'group_dim': 0,
                            'symmetric': False
                        },
                        'self_attn.out_proj': {
                            'num_bits': bits,
                            'group_size': 64,
                            'group_dim': 0,
                            'symmetric': False
                        }
                    }
                }
            }

            model = _init_group_wise_weight_quantization(model, ds_config)

            assert type(model.fc1) is QuantizedLinear
            assert type(model.fc2) is QuantizedLinear
            assert type(model.self_attn.q_proj) is QuantizedLinear
            assert type(model.self_attn.k_proj) is QuantizedLinear
            assert type(model.self_attn.v_proj) is QuantizedLinear
            assert type(model.self_attn.out_proj) is QuantizedLinear

    @pytest.mark.skipif(device == 'cpu', reason='CPU does support FP16 GEMM')
    def test_quantized_linear(self, quantization_bits, group_dim):
        reset_random()

        layers = []

        for idx in range(5):
            layers.append(
                (f'layer_{idx}', nn.Linear(in_features=128, out_features=128, dtype=torch.float16, device=device)))

        input_tensor = torch.randn(32, 128, dtype=torch.float16, device=device)
        with torch.no_grad():
            model = nn.Sequential(OrderedDict(layers))

            ref_output = model(input_tensor)

            ds_config = {
                'weight_quantization': {
                    'post_init_quant': {
                        'layer': {
                            'num_bits': quantization_bits,
                            'group_size': 64,
                            'group_dim': group_dim,
                            'symmetric': False
                        }
                    }
                }
            }

            model = _init_group_wise_weight_quantization(model, ds_config)

            assert type(model.layer_0) is QuantizedLinear
            assert type(model.layer_1) is QuantizedLinear
            assert type(model.layer_2) is QuantizedLinear
            assert type(model.layer_3) is QuantizedLinear
            assert type(model.layer_4) is QuantizedLinear

            output = model(input_tensor)

            mean_diff = torch.mean(torch.abs(ref_output - output))

            # This threshold value is emperically selected.
            assert mean_diff < 0.15, f'Numeric error exceed threshold, mean diff {mean_diff}'

    def test_float_int4_quantization(self):
        reset_random()
        quantization_test_helper(torch.float32, 4)

    @pytest.mark.skipif(device == 'cpu', reason='CPU does support FP16 GEMM')
    def test_half_int4_quantization(self):
        reset_random()
        quantization_test_helper(torch.float16, 4)

    def test_float_int8_quantization(self):
        reset_random()
        quantization_test_helper(torch.float32, 8)

    def test_half_int8_quantization(self):
        reset_random()
        quantization_test_helper(torch.float16, 8)

    @pytest.mark.skipif(device == 'cpu', reason='CPU does support FP16 GEMM')
    def test_zero3_int4_post_init_quant(self, quantization_bits):
        reset_random()
        zero3_post_init_quantization_test_helper(cpu_offload=False, nvme_offload=False, bits=quantization_bits)

    @pytest.mark.skipif(device == 'cpu', reason='CPU does support FP16 GEMM')
    def test_zero3_int4_post_init_quant_cpu_offload(self, quantization_bits):
        reset_random()
        zero3_post_init_quantization_test_helper(cpu_offload=True, nvme_offload=False, bits=quantization_bits)

    @pytest.mark.skipif(device == 'cpu', reason='CPU does support FP16 GEMM')
    def test_zero3_int4_post_init_quant_nvme_offload(self):
        reset_random()
        zero3_post_init_quantization_test_helper(cpu_offload=False, nvme_offload=True, bits=4)

    @pytest.mark.skipif(device == 'cpu', reason='CPU does support FP16 GEMM')
    def test_zero3_int4_quantized_initialization(self, quantization_bits):
        reset_random()
        zero3_quantized_initialization_test_helper(cpu_offload=False, nvme_offload=False, bits=quantization_bits)

    @pytest.mark.skipif(device == 'cpu', reason='CPU does support FP16 GEMM')
    def test_zero3_int4_quantized_initialization_cpu_offload(self, quantization_bits):
        reset_random()
        zero3_quantized_initialization_test_helper(cpu_offload=True, nvme_offload=False, bits=quantization_bits)

    @pytest.mark.skipif(device == 'cpu', reason='CPU does support FP16 GEMM')
    def test_zero3_int4_quantized_initialization_nvme_offload(self):
        reset_random()
        zero3_quantized_initialization_test_helper(cpu_offload=False, nvme_offload=True, bits=4)
