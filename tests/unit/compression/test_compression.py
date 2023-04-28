# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import pytest
import random
import numpy as np
from unit.megatron_model import get_gpt2_model
from deepspeed.compression.compress import init_compression
from unit.modeling import BertConfig
from unit.modelingpreln import BertEncoder as BertEncoderPreln
from deepspeed.compression.basic_layer import LinearLayer_Compress, ColumnParallelLinear_Compress, RowParallelLinear_Compress
from deepspeed.compression.helper import convert_conv1d_to_linear
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest
from unit.util import required_minimum_torch_version, required_maximum_torch_version

pytestmark = pytest.mark.skipif(not required_minimum_torch_version(major_version=1, minor_version=5),
                                reason='Megatron-LM package requires Pytorch version 1.5 or above')


def reset_random(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    get_accelerator().manual_seed_all(seed)


def create_bert_model():
    hidden_size = 384
    num_layers = 2
    heads = 12
    dropout_ratio = 0.1
    bert_config = BertConfig(vocab_size_or_config_json_file=119547,
                             hidden_size=hidden_size,
                             num_hidden_layers=num_layers,
                             num_attention_heads=heads,
                             intermediate_size=hidden_size * 4,
                             hidden_act="gelu",
                             hidden_dropout_prob=dropout_ratio,
                             attention_probs_dropout_prob=dropout_ratio,
                             max_position_embeddings=512,
                             type_vocab_size=2,
                             initializer_range=0.2)

    weights = []
    biases = []

    for i in range(4):
        weights.append(torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size)))

    weights.append(torch.nn.Parameter(torch.Tensor(hidden_size)))
    weights.append(torch.nn.Parameter(torch.Tensor(hidden_size * 4, hidden_size)))
    weights.append(torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4)))
    weights.append(torch.nn.Parameter(torch.Tensor(hidden_size)))

    biases.append(torch.nn.Parameter(torch.Tensor(hidden_size)))
    for i in range(4):
        biases.append(torch.nn.Parameter(torch.Tensor(hidden_size)))
    biases.append(torch.nn.Parameter(torch.Tensor(hidden_size * 4)))
    biases.append(torch.nn.Parameter(torch.Tensor(hidden_size)))
    biases.append(torch.nn.Parameter(torch.Tensor(hidden_size)))

    return BertEncoderPreln(bert_config, weights, biases)


class Conv1D(torch.nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        self.weight = torch.nn.Parameter(w)
        self.bias = torch.nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf, )
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


def create_conv1d_model():
    nf = 128
    nx = 128

    return torch.nn.ModuleList([Conv1D(nf, nx) for i in range(4)])


class TestCompression(DistributedTest):

    def setup_method(self, method):
        reset_random()

    def get_ds_config(self):
        ds_config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Lamb",
                "params": {
                    "lr": 0.00015
                }
            },
            "fp16": {
                "enabled": True
            },
            "compression_training": {
                "weight_quantization": {
                    "shared_parameters": {
                        "enabled": True,
                        "quantizer_kernel": False,
                        "schedule_offset": 50,
                        "quantize_groups": 1,
                        "quantize_verbose": False,
                        "quantization_type": "asymmetric",
                        "rounding": "nearest",
                        "fp16_mixed_quantize": {
                            "enabled": False,
                            "quantize_change_ratio": 0.001
                        }
                    },
                    "different_groups": {
                        "wq1": {
                            "params": {
                                "start_bits": 12,
                                "target_bits": 8,
                                "quantization_period": 50
                            },
                            "modules": ["attention.self", "intermediate"]
                        },
                        "wq2": {
                            "params": {
                                "start_bits": 12,
                                "target_bits": 4,
                                "quantization_period": 50
                            },
                            "modules": ["attention.output"]
                        }
                    }
                },
                "activation_quantization": {
                    "shared_parameters": {
                        "enabled": True,
                        "quantization_type": "asymmetric",
                        "range_calibration": "dynamic",
                        "schedule_offset": 50
                    },
                    "different_groups": {
                        "aq1": {
                            "params": {
                                "bits": 8
                            },
                            "modules": ["attention.output"]
                        }
                    }
                },
                "sparse_pruning": {
                    "shared_parameters": {
                        "enabled": True,
                        "schedule_offset": 30,
                        "method": "l1"
                    },
                    "different_groups": {
                        "sp1": {
                            "params": {
                                "dense_ratio": 0.5
                            },
                            "modules": ["attention.self"]
                        }
                    }
                },
                "row_pruning": {
                    "shared_parameters": {
                        "enabled": True,
                        "schedule_offset": 20,
                        "method": "topk"
                    },
                    "different_groups": {
                        "rp1": {
                            "params": {
                                "dense_ratio": 0.5
                            },
                            "modules": ["intermediate.dense"],
                            "related_modules": [["layer.\\w+.output.dense"]]
                        }
                    }
                },
                "head_pruning": {
                    "shared_parameters": {
                        "enabled": True,
                        "schedule_offset": 10,
                        "method": "topk",
                        "num_heads": 12
                    },
                    "different_groups": {
                        "rp1": {
                            "params": {
                                "dense_ratio": 0.5
                            },
                            "modules": ["attention.output.dense"],
                            "related_modules": [["self.query", "self.key", "self.value"]]
                        }
                    }
                }
            }
        }

        return ds_config_dict

    def test_linear_layer_compress(self, tmpdir):
        model = create_bert_model()
        compressed_model = init_compression(model, self.get_ds_config())

        assert isinstance(compressed_model.layer[0].attention.self.query, LinearLayer_Compress)
        assert isinstance(compressed_model.layer[0].attention.self.key, LinearLayer_Compress)
        assert isinstance(compressed_model.layer[0].attention.self.value, LinearLayer_Compress)

    @pytest.mark.skip(reason="megatron-lm is currently broken so this test cannot be run.")
    def test_mpu_compress(self, tmpdir):
        if not required_maximum_torch_version(major_version=1, minor_version=13):
            pytest.skip("megatron not compatible with torch >1.13")
        from megatron import mpu
        args_defaults = {
            'num_layers': 2,
            'hidden_size': 128,
            'num_attention_heads': 8,
            'max_position_embeddings': 128,
        }

        model = get_gpt2_model(args_defaults)
        compressed_model = init_compression(model, self.get_ds_config(), mpu=mpu)

        assert isinstance(compressed_model.module.language_model.transformer.layers[0].attention.query_key_value,
                          ColumnParallelLinear_Compress)
        assert isinstance(compressed_model.module.language_model.transformer.layers[0].attention.dense,
                          RowParallelLinear_Compress)
        assert isinstance(compressed_model.module.language_model.transformer.layers[0].mlp.dense_h_to_4h,
                          ColumnParallelLinear_Compress)
        assert isinstance(compressed_model.module.language_model.transformer.layers[0].mlp.dense_4h_to_h,
                          RowParallelLinear_Compress)

    def test_conv1d_convertion(self, tmpdir):
        model = create_conv1d_model()
        compressed_model = convert_conv1d_to_linear(model, Conv1D)

        assert isinstance(compressed_model[0], torch.nn.Linear)
        assert isinstance(compressed_model[1], torch.nn.Linear)
        assert isinstance(compressed_model[2], torch.nn.Linear)
        assert isinstance(compressed_model[3], torch.nn.Linear)
