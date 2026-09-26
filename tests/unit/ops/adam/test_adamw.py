# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
import torch
import pytest

from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam.fused_adam import multi_tensor_applier
from deepspeed.ops.op_builder import FusedAdamBuilder
from unit.common import DistributedTest
from unit.simple_model import SimpleModel
from deepspeed.accelerator import get_accelerator

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
# Skipping at module level would also hide the dtype-parametrized reference test below, which is
# what let the CPU adam_w_mode bug go unnoticed on runners without fp16 support.
@pytest.mark.skipif(torch.half not in get_accelerator().supported_dtypes(),
                    reason=f"fp16 not supported, valid dtype: {get_accelerator().supported_dtypes()}")
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


def reference_mixed_precision_adam_step(master,
                                        grad,
                                        exp_avg,
                                        exp_avg_sq,
                                        step,
                                        lr,
                                        beta1,
                                        beta2,
                                        eps,
                                        weight_decay,
                                        adam_w_mode,
                                        bias_correction,
                                        grad_scale):
    """Adam/AdamW step with low-precision gradients and fp32 state."""
    g = grad.float() / grad_scale
    if not adam_w_mode:
        g = g + weight_decay * master
    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    if bias_correction:
        next_m = exp_avg / (1 - beta1**step)
        next_v = exp_avg_sq / (1 - beta2**step)
    else:
        next_m = exp_avg
        next_v = exp_avg_sq
    update = next_m / (next_v.sqrt() + eps)
    if adam_w_mode:
        update = update + weight_decay * master
    master.add_(update, alpha=-lr)


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("adam_w_mode", [False, True], ids=["adam", "adamw"])
@pytest.mark.parametrize("bias_correction", [False, True], ids=["no_bias_correction", "bias_correction"])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("grad_scale", [1.0, 128.0])
def test_mixed_precision_fused_adam_op_matches_reference(dtype, adam_w_mode, bias_correction, weight_decay,
                                                         grad_scale):
    if dtype not in get_accelerator().supported_dtypes():
        pytest.skip(f"{dtype} not supported on {get_accelerator().device_name()}")
    if not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME]:
        pytest.skip("FusedAdam is not compatible")

    device = get_accelerator().device_name()
    torch.manual_seed(1234)
    lr, beta1, beta2, eps = 1e-2, 0.9, 0.999, 1e-8
    grad = torch.empty(1003, device=device, dtype=dtype)
    master = torch.randn(1003, device=device, dtype=torch.float32)
    exp_avg = torch.zeros_like(master)
    exp_avg_sq = torch.zeros_like(master)
    output = torch.empty_like(grad)
    ref_master = master.clone()
    ref_exp_avg = exp_avg.clone()
    ref_exp_avg_sq = exp_avg_sq.clone()
    fused_adam_op = FusedAdamBuilder().load()
    mixed_precision_op = fused_adam_op.multi_tensor_adam_mixed_precision
    dummy_overflow_buf = get_accelerator().IntTensor([0])

    for step in range(1, 6):
        grad.copy_(torch.randn_like(grad))
        reference_mixed_precision_adam_step(ref_master, grad, ref_exp_avg, ref_exp_avg_sq, step, lr, beta1, beta2,
                                            eps, weight_decay, adam_w_mode, bias_correction, grad_scale)
        multi_tensor_applier(mixed_precision_op, dummy_overflow_buf,
                             [[grad], [master], [exp_avg], [exp_avg_sq], [output]], lr, beta1, beta2, eps, step,
                             int(adam_w_mode), int(bias_correction), weight_decay, grad_scale)

    fp32_atol = 8 * torch.finfo(torch.float32).eps * ref_master.abs().max().item()
    output_atol = 8 * torch.finfo(dtype).eps * ref_master.abs().max().item()
    torch.testing.assert_close(master, ref_master, rtol=0, atol=fp32_atol)
    torch.testing.assert_close(exp_avg, ref_exp_avg, rtol=0, atol=fp32_atol)
    torch.testing.assert_close(exp_avg_sq, ref_exp_avg_sq, rtol=0, atol=fp32_atol)
    torch.testing.assert_close(output.float(), ref_master.to(dtype).float(), rtol=0, atol=output_atol)


@pytest.mark.parametrize(
    "invalid_case",
    [
        "scale_zero",
        "scale_negative",
        "scale_inf",
        "scale_nan",
        "wrong_list_count",
        "empty_lists",
        "fp32_gradient",
        "low_precision_master",
        "low_precision_exp_avg",
        "low_precision_exp_avg_sq",
        "output_dtype_mismatch",
        "non_contiguous",
        "different_numel",
        "different_shape",
        "different_device",
    ],
)
def test_mixed_precision_fused_adam_op_rejects_invalid_inputs_without_mutation(invalid_case):
    if torch.float16 not in get_accelerator().supported_dtypes():
        pytest.skip(f"fp16 not supported on {get_accelerator().device_name()}")
    if not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME]:
        pytest.skip("FusedAdam is not compatible")

    device = get_accelerator().device_name()
    grad = torch.randn(1003, device=device, dtype=torch.float16)
    master = torch.randn(1003, device=device, dtype=torch.float32)
    exp_avg = torch.randn_like(master)
    exp_avg_sq = torch.rand_like(master)
    output = torch.randn_like(grad)
    tensor_lists = [[grad], [master], [exp_avg], [exp_avg_sq], [output]]
    grad_scale = 128.0

    if invalid_case == "scale_zero":
        grad_scale = 0.0
    elif invalid_case == "scale_negative":
        grad_scale = -1.0
    elif invalid_case == "scale_inf":
        grad_scale = float("inf")
    elif invalid_case == "scale_nan":
        grad_scale = float("nan")
    elif invalid_case == "wrong_list_count":
        tensor_lists = tensor_lists[:-1]
    elif invalid_case == "empty_lists":
        tensor_lists = [[] for _ in range(5)]
    elif invalid_case == "fp32_gradient":
        tensor_lists[0][0] = grad.float()
        tensor_lists[4][0] = output.float()
    elif invalid_case == "low_precision_master":
        tensor_lists[1][0] = master.half()
    elif invalid_case == "low_precision_exp_avg":
        tensor_lists[2][0] = exp_avg.half()
    elif invalid_case == "low_precision_exp_avg_sq":
        tensor_lists[3][0] = exp_avg_sq.half()
    elif invalid_case == "output_dtype_mismatch":
        tensor_lists[4][0] = output.bfloat16()
    elif invalid_case == "non_contiguous":
        tensor_lists[0][0] = torch.randn(1003, 2, device=device, dtype=torch.float16)[:, 0]
    elif invalid_case == "different_numel":
        tensor_lists[1][0] = torch.randn(1004, device=device, dtype=torch.float32)
    elif invalid_case == "different_shape":
        tensor_lists[1][0] = master.view(17, 59)
    elif invalid_case == "different_device":
        if get_accelerator().device_count() < 2:
            pytest.skip("different-device validation requires two accelerator devices")
        tensor_lists[1][0] = master.to(f"{device}:1")

    supplied_tensors = [tensor for tensor_list in tensor_lists for tensor in tensor_list]
    snapshots = [tensor.clone() for tensor in supplied_tensors]
    fused_adam_op = FusedAdamBuilder().load()
    dummy_overflow_buf = get_accelerator().IntTensor([0])

    with pytest.raises(RuntimeError):
        multi_tensor_applier(fused_adam_op.multi_tensor_adam_mixed_precision, dummy_overflow_buf, tensor_lists, 1e-2,
                             0.9, 0.999, 1e-8, 1, 1, 1, 0.1, grad_scale)

    for tensor, snapshot in zip(supplied_tensors, snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0, equal_nan=True)


def test_fused_adam_mixed_precision_step_preserves_group_state():
    if not {torch.float16, torch.bfloat16}.issubset(get_accelerator().supported_dtypes()):
        pytest.skip(f"fp16 and bf16 not supported on {get_accelerator().device_name()}")
    if not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME]:
        pytest.skip("FusedAdam is not compatible")

    device = get_accelerator().device_name()
    torch.manual_seed(5678)
    masters = [torch.nn.Parameter(torch.randn(1003, device=device, dtype=torch.float32)) for _ in range(2)]
    grads = [torch.empty(1003, device=device, dtype=torch.float16),
             torch.empty(1003, device=device, dtype=torch.bfloat16)]
    outputs = [torch.empty_like(grad) for grad in grads]
    groups = [
        {
            "params": [masters[0]],
            "lr": 1e-2,
            "weight_decay": 0.0,
        },
        {
            "params": [masters[1]],
            "lr": 2e-3,
            "weight_decay": 0.1,
            "step": 3,
        },
    ]
    optimizer = FusedAdam(groups, betas=(0.8, 0.95), eps=1e-6, adam_w_mode=True, bias_correction=True)
    ref_masters = [master.detach().clone() for master in masters]
    ref_exp_avgs = [torch.zeros_like(master) for master in masters]
    ref_exp_avg_sqs = [torch.zeros_like(master) for master in masters]
    initial_steps = [0, 3]
    assert not optimizer.state
    assert optimizer._can_step_with_mixed_precision_grads(grads, outputs)

    for update_idx in range(1, 6):
        for group_idx, (grad, group) in enumerate(zip(grads, optimizer.param_groups)):
            grad.copy_(torch.randn_like(grad))
            reference_mixed_precision_adam_step(ref_masters[group_idx], grad, ref_exp_avgs[group_idx],
                                                ref_exp_avg_sqs[group_idx], initial_steps[group_idx] + update_idx,
                                                group["lr"], group["betas"][0], group["betas"][1], group["eps"],
                                                group["weight_decay"], True, group["bias_correction"], 128.0)
        optimizer._step_with_mixed_precision_grads(grads, outputs, 128.0)

    for group_idx, (master, output) in enumerate(zip(masters, outputs)):
        state = optimizer.state[master]
        assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
        assert state["step"] == initial_steps[group_idx] + 5
        assert master.dtype == torch.float32
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32
        fp32_atol = 8 * torch.finfo(torch.float32).eps * ref_masters[group_idx].abs().max().item()
        output_atol = 8 * torch.finfo(output.dtype).eps * ref_masters[group_idx].abs().max().item()
        torch.testing.assert_close(master, ref_masters[group_idx], rtol=0, atol=fp32_atol)
        torch.testing.assert_close(state["exp_avg"], ref_exp_avgs[group_idx], rtol=0, atol=fp32_atol)
        torch.testing.assert_close(state["exp_avg_sq"], ref_exp_avg_sqs[group_idx], rtol=0, atol=fp32_atol)
        torch.testing.assert_close(output.float(), ref_masters[group_idx].to(output.dtype).float(), rtol=0,
                                   atol=output_atol)


def test_fused_adam_missing_mixed_precision_symbol_falls_back():
    if torch.float16 not in get_accelerator().supported_dtypes():
        pytest.skip(f"fp16 not supported on {get_accelerator().device_name()}")
    if not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME]:
        pytest.skip("FusedAdam is not compatible")

    device = get_accelerator().device_name()
    master = torch.nn.Parameter(torch.randn(1003, device=device, dtype=torch.float32))
    grad = torch.randn(1003, device=device, dtype=torch.float16)
    output = torch.randn_like(grad)
    optimizer = FusedAdam([master])
    optimizer.multi_tensor_adam_mixed_precision = None
    master_snapshot = master.detach().clone()
    output_snapshot = output.clone()

    assert not optimizer._can_step_with_mixed_precision_grads([grad], [output])
    assert not optimizer.state
    torch.testing.assert_close(master, master_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(output, output_snapshot, rtol=0, atol=0)


def test_fused_adam_mixed_precision_rejects_invalid_scale_before_state_initialization():
    if torch.float16 not in get_accelerator().supported_dtypes():
        pytest.skip(f"fp16 not supported on {get_accelerator().device_name()}")
    if not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME]:
        pytest.skip("FusedAdam is not compatible")

    device = get_accelerator().device_name()
    master = torch.nn.Parameter(torch.randn(1003, device=device, dtype=torch.float32))
    grad = torch.randn(1003, device=device, dtype=torch.float16)
    output = torch.randn_like(grad)
    optimizer = FusedAdam([master])
    master_snapshot = master.detach().clone()
    output_snapshot = output.clone()

    with pytest.raises(RuntimeError):
        optimizer._step_with_mixed_precision_grads([grad], [output], 0.0)

    assert not optimizer.state
    torch.testing.assert_close(master, master_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(output, output_snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("invalid_state", ["low_precision", "wrong_shape"])
def test_fused_adam_mixed_precision_capability_rejects_invalid_state(invalid_state):
    if torch.float16 not in get_accelerator().supported_dtypes():
        pytest.skip(f"fp16 not supported on {get_accelerator().device_name()}")
    if not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME]:
        pytest.skip("FusedAdam is not compatible")

    device = get_accelerator().device_name()
    master = torch.nn.Parameter(torch.randn(1003, device=device, dtype=torch.float32))
    grad = torch.randn(1003, device=device, dtype=torch.float16)
    output = torch.randn_like(grad)
    optimizer = FusedAdam([master])
    optimizer.state[master] = {
        "step": 4,
        "exp_avg": torch.zeros_like(master),
        "exp_avg_sq": torch.zeros_like(master),
    }
    if invalid_state == "low_precision":
        optimizer.state[master]["exp_avg"] = optimizer.state[master]["exp_avg"].half()
    else:
        optimizer.state[master]["exp_avg"] = optimizer.state[master]["exp_avg"][:-1]
    state_snapshot = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in optimizer.state[master].items()
    }
    master_snapshot = master.detach().clone()
    output_snapshot = output.clone()

    assert not optimizer._can_step_with_mixed_precision_grads([grad], [output])
    assert optimizer.state[master].keys() == state_snapshot.keys()
    for key, value in optimizer.state[master].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, state_snapshot[key], rtol=0, atol=0)
        else:
            assert value == state_snapshot[key]
    torch.testing.assert_close(master, master_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(output, output_snapshot, rtol=0, atol=0)
