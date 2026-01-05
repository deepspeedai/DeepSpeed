# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.utils.torch import required_torch_version
from deepspeed.accelerator import get_accelerator

from unit.v1.compile.util import compare_loss
from unit.common import DistributedTest
from unit.util import bf16_required_version_check, skip_on_arch
import deepspeed
from deepspeed.ops.aio import AsyncIOBuilder

pytestmark = pytest.mark.skipif(not required_torch_version(min_version=2.1),
                                reason="Compile tests requires Pytorch version 2.1 or above")


class TestZeRO(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    @pytest.mark.parametrize('offload_device', [OffloadDeviceEnum.none, OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme])
    def test_compile_zero(self, tmpdir, zero_stage, dtype, offload_device):
        if dtype == torch.bfloat16:
            skip_on_arch(min_arch=8)
        if dtype == torch.bfloat16 and not bf16_required_version_check():
            pytest.skip(
                "DeepSpeed BFloat16 tests need NCCL >= 2.10.3, CUDA >=11.0, and HW support for BFloat16 to run correctly"
            )
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        if offload_device == OffloadDeviceEnum.nvme:
            if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
                pytest.skip('Skip tests since async-io is not compatible')
            if zero_stage != 3:
                pytest.skip(f"Nvme offload not supported for zero stage {zero_stage}")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }

        if offload_device == OffloadDeviceEnum.cpu:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": offload_device}
        elif offload_device == OffloadDeviceEnum.nvme:
            config_dict["zero_optimization"]["offload_optimizer"] = {
                "device": offload_device,
                "nvme_path": str(tmpdir)
            }
        if dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        elif dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        compare_loss(self, config_dict, dtype)


class TestDeepCompile(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize('zero_stage', [1, 3])
    @pytest.mark.parametrize('deepcompile', [True])  # deepcompile==False is included in test_compile_zero
    def test(self, zero_stage, dtype, deepcompile):
        if not required_torch_version(min_version=2.6):
            pytest.skip("DeepCompile requires PyTorch >= v2.6")
        if zero_stage == 3 and required_torch_version(min_version=2.9):
            pytest.skip("DeepCompile with ZeRO stage 3 is not currently supported on PyTorch >= 2.9")

        if dtype == torch.bfloat16:
            skip_on_arch(min_arch=8)
        if dtype == torch.bfloat16 and not bf16_required_version_check():
            pytest.skip(
                "DeepSpeed BFloat16 tests need NCCL >= 2.10.3, CUDA >=11.0, and HW support for BFloat16 to run correctly"
            )
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
            },
            "compile": {
                "deepcompile": deepcompile
            }
        }

        if dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        elif dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        # Need warmup steps
        compare_loss(self, config_dict, dtype, iteration=10)

    @pytest.mark.parametrize('dtype', [torch.float32])
    @pytest.mark.parametrize('zero_stage', [3])
    def test_padded_shard_handling(self, zero_stage, dtype):
        """Test that parameters with padding (uneven division) work correctly with DeepCompile"""
        if not required_torch_version(min_version=2.6):
            pytest.skip("DeepCompile requires PyTorch >= v2.6")
        if required_torch_version(min_version=2.9):
            pytest.skip("DeepCompile with ZeRO stage 3 is not supported on PyTorch >= 2.9")

        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        # Use a hidden dimension that requires padding when divided across ranks
        # With world_size=2, a hidden_dim of 13 creates parameters that need padding
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
            },
            "compile": {
                "deepcompile": True
            }
        }

        # This should work correctly with our padding-aware implementation
        # The test verifies that padded parameters are handled properly
        compare_loss(self, config_dict, dtype, iteration=1, hidden_dim_override=13)

    @pytest.mark.parametrize('dtype', [torch.float32])
    @pytest.mark.parametrize('zero_stage', [1, 3])
    def test_free_activation_mode(self, zero_stage, dtype):
        """Test that eagerly free activations work correctly and the threshold is configurable"""
        if not required_torch_version(min_version=2.6):
            pytest.skip("DeepCompile requires PyTorch >= v2.6")
        if zero_stage == 3 and required_torch_version(min_version=2.9):
            pytest.skip("DeepCompile with ZeRO stage 3 is not supported on PyTorch >= 2.9")

        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
            },
            "compile": {
                "deepcompile": True,
                "free_activation": True,
                "free_activation_threshold": 0,
            }
        }

        compare_loss(self, config_dict, dtype)

    @pytest.mark.parametrize('dtype', ["bfloat16", "float16"])
    @pytest.mark.parametrize('zero_stage', [3])
    def test_fusing_allgather_and_autocast(self, zero_stage, dtype):
        """Test that allgather and autocast can be correctly fused with DeepCompile"""
        if not required_torch_version(min_version=2.6):
            pytest.skip("DeepCompile requires PyTorch >= v2.6")
        if required_torch_version(min_version=2.9):
            pytest.skip("DeepCompile with ZeRO stage 3 is not supported on PyTorch >= 2.9")

        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "torch_autocast": {
                "enable": True,
                "dtype": dtype,
            },
            "zero_optimization": {
                "stage": zero_stage,
            },
            "compile": {
                "deepcompile": True
            }
        }

        compare_loss(self, config_dict, torch.float32)
