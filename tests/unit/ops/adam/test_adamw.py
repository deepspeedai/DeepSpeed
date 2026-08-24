# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
import torch
import pytest

from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.op_builder import FusedAdamBuilder
from unit.common import DistributedTest
from unit.simple_model import SimpleModel
from deepspeed.accelerator import get_accelerator

if torch.half not in get_accelerator().supported_dtypes():
    pytest.skip(f"fp16 not supported, valid dtype: {get_accelerator().supported_dtypes()}", allow_module_level=True)
# yapf: disable
#'optimizer, zero_offload, torch_adam, adam_w_mode, resulting_optimizer
adam_configs = [["AdamW", False, False, False, (FusedAdam, True)],
                ["AdamW", False, True,  False, (torch.optim.AdamW, None)],
                ["AdamW", True,  False, False, (DeepSpeedCPUAdam, True)],
                ["AdamW", True,  True,  False, (torch.optim.AdamW, None)],
                ["AdamW", False, False, True,  (FusedAdam, True)],
                ["AdamW", False, True,  True,  (torch.optim.AdamW, None)],
                ["AdamW", True,  False, True,  (DeepSpeedCPUAdam, True)],
                ["AdamW", True,  True,  True,  (torch.optim.AdamW, None)],
                ["Adam",  False, False, False, (FusedAdam, False)],
                ["Adam",  False, True,  False, (torch.optim.Adam, None)],
                ["Adam",  True,  False, False, (DeepSpeedCPUAdam, False)],
                ["Adam",  True,  True,  False, (torch.optim.Adam, None)],
                ["Adam",  False, False, True,  (FusedAdam, True)],
                ["Adam",  False, True,  True,  (torch.optim.AdamW, None)],
                ["Adam",  True,  False, True,  (DeepSpeedCPUAdam, True)],
                ["Adam",  True,  True,  True,  (torch.optim.AdamW, None)]]

@pytest.mark.parametrize(
    'optimizer, zero_offload, torch_adam, adam_w_mode, resulting_optimizer',
    adam_configs)
class TestAdamConfigs(DistributedTest):
    world_size = 1
    reuse_dist_env = True

    def test(self,
             optimizer,
             zero_offload,
             torch_adam,
             adam_w_mode,
             resulting_optimizer):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": optimizer,
                "params": {
                    "lr": 0.00015,
                    "torch_adam": torch_adam,
                    "adam_w_mode": adam_w_mode
                }
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2,
                "cpu_offload": zero_offload
            }
        }
        model = SimpleModel(10)
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              model_parameters=model.parameters())
        # get base optimizer under zero
        ds_optimizer = model.optimizer.optimizer
        opt_class, adam_w_mode = resulting_optimizer
        assert isinstance(ds_optimizer, opt_class)
        if adam_w_mode in [True, False]:
            assert ds_optimizer.adam_w_mode == adam_w_mode


@pytest.mark.parametrize('adam_w_mode', [True, False], ids=["adamw", "adam"])
@pytest.mark.parametrize('dtype', [torch.float, torch.bfloat16], ids=["fp32", "bf16"])
def test_fused_adam_matches_torch(adam_w_mode, dtype):
    if dtype not in get_accelerator().supported_dtypes():
        pytest.skip(f"{dtype} not supported on {get_accelerator().device_name()}")
    if not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME]:
        pytest.skip("FusedAdam is not compatible")

    device = get_accelerator().device_name()
    torch.manual_seed(0)
    ref_params = [torch.randn(1024, device=device, dtype=dtype, requires_grad=True) for _ in range(3)]
    ds_params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]
    optimizer_kwargs = dict(lr=1e-2, weight_decay=0.1)
    torch_optimizer = torch.optim.AdamW if adam_w_mode else torch.optim.Adam
    ref_optimizer = torch_optimizer(ref_params, **optimizer_kwargs)
    ds_optimizer = FusedAdam(ds_params, adam_w_mode=adam_w_mode, **optimizer_kwargs)

    for _ in range(5):
        for ref_param, ds_param in zip(ref_params, ds_params):
            grad = torch.randn_like(ref_param)
            ref_param.grad = grad.clone()
            ds_param.grad = grad.clone()
        ref_optimizer.step()
        ds_optimizer.step()

    # bf16 storage rounds differently depending on where intermediates are kept, so allow one ulp.
    atol = 1e-5 if dtype == torch.float else 2e-2
    for ref_param, ds_param in zip(ref_params, ds_params):
        torch.testing.assert_close(ds_param.float(), ref_param.float(), atol=atol, rtol=0)
