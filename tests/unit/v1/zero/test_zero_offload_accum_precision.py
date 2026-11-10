# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Accumulation precision of ZeRO-1/2 CPU offload across one optimizer window.

Each micro-step's gradient is exactly representable in bf16 (values (128 + k) / 128), so the
exact window sum is known. The only rounding comes from the accumulation dtype and the host
copy, which makes the error of each (stage, master dtype, accumulation dtype) cell directly
comparable.
"""

import os

import pytest
import torch

import deepspeed
from deepspeed.utils import safe_get_full_grad

from unit.common import DistributedTest
from deepspeed.accelerator import get_accelerator

MICRO_STEPS = 16
NUMEL = 16


class Weighted(torch.nn.Module):

    def __init__(self, numel):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(numel))

    def forward(self, c):
        return (self.w * c).sum()


def _config(stage, low_precision_master, accumulation_dtype):
    config = {
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_clipping": 0,
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": False
            },
        },
        "bf16": {
            "enabled": True,
            "bf16_master_weights_and_grads": low_precision_master,
        },
    }
    if accumulation_dtype is not None:
        config["data_types"] = {"grad_accum_dtype": accumulation_dtype}
    return config


def _window_error(stage, low_precision_master, accumulation_dtype):
    """Relative max error of the window's gradient sum against the exact sum, read before step()."""
    torch.manual_seed(0)
    model = Weighted(NUMEL)
    engine, _, _, _ = deepspeed.initialize(model=model,
                                           model_parameters=model.parameters(),
                                           config=_config(stage, low_precision_master, accumulation_dtype))
    try:
        generator = torch.Generator().manual_seed(1234)
        exact = torch.zeros(NUMEL, dtype=torch.float64)
        for step in range(MICRO_STEPS):
            k = torch.randint(0, 128, (NUMEL, ), generator=generator)
            c = ((128 + k).float() / 128).to(device=engine.device, dtype=torch.bfloat16)
            exact += c.double().cpu()
            engine.set_gradient_accumulation_boundary(step == MICRO_STEPS - 1)
            engine.backward(engine(c))
        grad = safe_get_full_grad(engine.module.w).detach().float().cpu().double()
        return ((grad - exact).abs().max() / exact.abs().max()).item()
    finally:
        engine.destroy()


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("low_precision_master", [False, True], ids=["fp32-master", "bf16-master"])
@pytest.mark.parametrize("accumulation_dtype", [None, "fp32"], ids=["bf16-accum", "fp32-accum"])
class TestZeroOffloadAccumPrecision(DistributedTest):
    world_size = 1

    def test_window_sum(self, stage, low_precision_master, accumulation_dtype):
        if not deepspeed.ops.__compatible_ops__[deepspeed.ops.op_builder.CPUAdamBuilder.NAME]:
            pytest.skip("CPUAdam is required for offloaded low-precision masters")
        if not get_accelerator().is_bf16_supported():
            pytest.skip("BF16 is not supported")
        error = _window_error(stage, low_precision_master, accumulation_dtype)
        out = os.environ.get("ACCUM_PRECISION_OUT")
        if out:
            with open(out, "a") as f:
                f.write(f"stage={stage} master={'bf16' if low_precision_master else 'fp32'} "
                        f"accum={accumulation_dtype or 'bf16'} rel_max_err={error:.3e}\n")
        if stage == 2 and not low_precision_master and accumulation_dtype == "fp32":
            # The supported case: the window sum is exact. Stage 1 rounds its window sum to the
            # 16-bit reduction dtype once (same error on the base commit); it is characterized, not required.
            assert error == 0.0, f"fp32 accumulation with fp32 masters lost precision: {error:.3e}"


def _set_grad_error(scenario, stage=2):
    """Drive safe_set_full_grad between two backwards in one window and return the window sum read at its end.

    Returns the value written by safe_get_full_grad minus the sum the caller would expect if the set value
    were kept and accumulated into: scenario "mid_window" sets while the GPU partition is resident (after the
    first backward of a window); "after_boundary" sets after a boundary freed the partition.
    """
    from deepspeed.utils import safe_set_full_grad
    torch.manual_seed(0)
    model = Weighted(NUMEL)
    engine, _, _, _ = deepspeed.initialize(model=model,
                                           model_parameters=model.parameters(),
                                           config=_config(stage, False, "fp32"))
    try:
        ones = torch.ones(NUMEL, device=engine.device, dtype=torch.bfloat16)
        c1 = (torch.arange(NUMEL, device=engine.device) % 7 + 128).float().div(128).to(torch.bfloat16)
        c2 = (torch.arange(NUMEL, device=engine.device) % 5 + 129).float().div(128).to(torch.bfloat16)
        value = torch.full((NUMEL, ), 0.5, device=engine.device, dtype=torch.float32)
        if scenario == "mid_window":
            engine.set_gradient_accumulation_boundary(False)
            engine.backward(engine(c1))
            safe_set_full_grad(engine.module.w, value)
            engine.set_gradient_accumulation_boundary(True)
            engine.backward(engine(c2))
            expected = value + c2.float()
        else:
            engine.set_gradient_accumulation_boundary(True)
            engine.backward(engine(c1))
            safe_set_full_grad(engine.module.w, value)
            engine.set_gradient_accumulation_boundary(True)
            engine.backward(engine(c2))
            expected = value + c2.float()
        grad = safe_get_full_grad(engine.module.w).detach().float().cpu()
        return (grad - expected.cpu()).abs().max().item(), (grad - (c1 + c2).float().cpu()).abs().max().item()
    finally:
        engine.destroy()


class TestZeroOffloadSetFullGrad(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("scenario", ["mid_window", "after_boundary"])
    def test_set_full_grad_between_backwards(self, scenario):
        if not deepspeed.ops.__compatible_ops__[deepspeed.ops.op_builder.CPUAdamBuilder.NAME]:
            pytest.skip("CPUAdam is required for offloaded low-precision masters")
        err_expected, err_overwritten = _set_grad_error(scenario)
        out = os.environ.get("ACCUM_PRECISION_OUT")
        if out:
            with open(out, "a") as f:
                f.write(f"set_full_grad scenario={scenario} err_if_kept={err_expected:.3e} "
                        f"err_if_overwritten={err_overwritten:.3e}\n")
        if scenario == "after_boundary":
            assert err_expected < 1e-6, "a value set after a freed boundary must be kept and accumulated"
        else:
            # Characterizes the resident case: the boundary overwrites the set value (not kept).
            assert err_overwritten < 1e-2, "mid-window set is no longer overwritten at the boundary"


def _retain_option_run(retain):
    """Three request-boundary backwards and one step; returns residency per boundary, grads, and final weight."""
    torch.manual_seed(0)
    model = Weighted(NUMEL)
    config = _config(2, False, "fp32")
    config["zero_optimization"]["retain_hbm_grad_partition"] = retain
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    try:
        generator = torch.Generator().manual_seed(99)
        exact = torch.zeros(NUMEL, dtype=torch.float64)
        resident = []  # True while the GPU partition is held
        for _ in range(3):
            k = torch.randint(0, 128, (NUMEL, ), generator=generator)
            c = ((128 + k).float() / 128).to(device=engine.device, dtype=torch.bfloat16)
            exact += c.double().cpu()
            engine.set_gradient_accumulation_boundary(True)
            engine.backward(engine(c))
            resident.append(engine.optimizer.averaged_gradients.get(0) is not None)
        grad = safe_get_full_grad(engine.module.w).detach().float().cpu().double()
        engine.step()
        weight = engine.module.w.detach().float().cpu().clone()
        return resident, (grad - exact).abs().max().item(), weight
    finally:
        engine.destroy()


class TestZeroOffloadReleaseOption(DistributedTest):
    world_size = 1

    def test_release_option_controls_residency_without_changing_results(self):
        if not deepspeed.ops.__compatible_ops__[deepspeed.ops.op_builder.CPUAdamBuilder.NAME]:
            pytest.skip("CPUAdam is required for offloaded low-precision masters")
        resident_released, err_released, w_released = _retain_option_run(False)
        resident_kept, err_kept, w_kept = _retain_option_run(True)
        # `resident` is True while the GPU partition is held. With the option on, every boundary frees it.
        assert resident_released == [False, False,
                                     False], f"partition stayed resident with the option on: {resident_released}"
        assert resident_kept == [True, True, True], f"partition was freed with the option off: {resident_kept}"
        assert err_kept == 0.0, f"resident path lost precision: {err_kept:.3e}"
        assert err_released == 0.0, f"released path lost precision: {err_released:.3e}"
        assert torch.equal(w_released, w_kept), "release option changed the optimizer result"
