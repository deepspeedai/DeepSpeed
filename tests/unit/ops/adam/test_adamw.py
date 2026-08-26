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


def reference_adam_step(param, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps, weight_decay, adam_w_mode):
    """Adam/AdamW step with fp32 math and storage-dtype rounding, matching csrc/adam/multi_tensor_adam.cu."""
    dtype = param.dtype
    p, g, m, v = param.float(), grad.float(), exp_avg.float(), exp_avg_sq.float()
    if not adam_w_mode:
        g = g + weight_decay * p
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    denom = (v / (1 - beta2**step)).sqrt() + eps
    update = (m / (1 - beta1**step)) / denom
    if adam_w_mode:
        update = update + weight_decay * p
    p = p - lr * update
    param.copy_(p.to(dtype))
    exp_avg.copy_(m.to(dtype))
    exp_avg_sq.copy_(v.to(dtype))


@pytest.mark.parametrize('adam_w_mode', [True, False], ids=["adamw", "adam"])
@pytest.mark.parametrize('dtype', [torch.float, torch.bfloat16, torch.half], ids=["fp32", "bf16", "fp16"])
def test_fused_adam_matches_reference(adam_w_mode, dtype):
    if dtype not in get_accelerator().supported_dtypes():
        pytest.skip(f"{dtype} not supported on {get_accelerator().device_name()}")
    if not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME]:
        pytest.skip("FusedAdam is not compatible")

    device = get_accelerator().device_name()
    torch.manual_seed(0)
    lr, betas, eps, weight_decay = 1e-2, (0.9, 0.999), 1e-8, 0.1
    ds_params = [torch.nn.Parameter(torch.randn(1024, device=device, dtype=dtype)) for _ in range(3)]
    ref_params = [p.detach().clone() for p in ds_params]
    ref_exp_avgs = [torch.zeros_like(p) for p in ref_params]
    ref_exp_avg_sqs = [torch.zeros_like(p) for p in ref_params]
    ds_optimizer = FusedAdam(ds_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adam_w_mode=adam_w_mode)

    for step in range(1, 6):
        for ds_param, ref_param, exp_avg, exp_avg_sq in zip(ds_params, ref_params, ref_exp_avgs, ref_exp_avg_sqs):
            ds_param.grad = torch.randn_like(ds_param)
            reference_adam_step(ref_param, ds_param.grad, exp_avg, exp_avg_sq, step, lr, betas[0], betas[1], eps,
                                weight_decay, adam_w_mode)
        ds_optimizer.step()

    # fp32 operation order differs between implementations and accumulates over steps, so allow a
    # few ulps of the storage dtype at the scale of the tensor (per-element rtol is too strict near
    # zero). For this data (|param| ~ 3) that is ~3e-6 for fp32 (tighter than the 1e-5 it replaces)
    # and ~0.2 for bf16, against a measured implementation agreement of ~1e-6 and ~3e-5 respectively.
    for ds_param, ref_param in zip(ds_params, ref_params):
        atol = 8 * torch.finfo(dtype).eps * ref_param.abs().max().item()
        torch.testing.assert_close(ds_param.float(), ref_param.float(), rtol=0, atol=atol)
