# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.utils import required_torch_version
from deepspeed.accelerator import get_accelerator

from unit.runtime.compile.util import compare_loss
from unit.common import DistributedTest
from unit.util import bf16_required_version_check

pytestmark = pytest.mark.skipif(not required_torch_version(min_version=2.1),
                                reason="Compile tests requires Pytorch version 2.1 or above")


class TestZeRO(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    @pytest.mark.parametrize('offload_device', [OffloadDeviceEnum.none, OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme])
    def test_compile_zero(self, tmpdir, zero_stage, dtype, offload_device):
        if dtype == torch.bfloat16 and not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        if offload_device == OffloadDeviceEnum.nvme:
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
            },
            "compile": {
                "enabled": True,
                "backend": "inductor"
            }
        }

        if get_accelerator().device_name() == 'hpu':
            config_dict['compile']['backend'] = 'hpu_backend'
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
