# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
import os
import sys
import threading
import time
import torch
import numpy as np
import pytest
from cpuinfo import get_cpu_info

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.op_builder import CPUAdamBuilder, FusedAdamBuilder
from unit.common import DistributedTest

if not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
    pytest.skip("cpu-adam is not compatible", allow_module_level=True)

pytest.cpu_vendor = get_cpu_info().get("vendor_id_raw", "").lower()


def check_equal(first, second, atol=1e-2, verbose=False):
    x = first.detach().float().numpy()
    y = second.detach().float().numpy()
    print("ATOL", atol)
    if verbose:
        print("x = {}".format(x.flatten()))
        print("y = {}".format(y.flatten()))
        print('-' * 80)
    np.testing.assert_allclose(x, y, err_msg="param-update mismatch!", atol=atol)


def _compare_optimizers(model_size, param1, optimizer1, param2, optimizer2):
    for i in range(10):
        param1.grad = torch.randn(model_size, device=param1.device).to(param1.dtype)
        param2.grad = param1.grad.clone().detach().to(device=param2.device, dtype=param2.dtype)

        optimizer1.step()
        optimizer2.step()

    tolerance = param1.float().norm().detach().numpy() * 1e-2
    check_equal(param1.float().norm(), param2.float().cpu().norm(), atol=tolerance, verbose=True)


def _reference_adam_step(param, grad, exp_avg, exp_avg_sq, step, lr, betas, eps, weight_decay, adamw_mode,
                         bias_correction):
    beta1, beta2 = betas
    adjusted_grad = grad
    if weight_decay > 0 and not adamw_mode:
        adjusted_grad = grad.add(param, alpha=weight_decay)

    exp_avg.mul_(beta1).add_(adjusted_grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(adjusted_grad, adjusted_grad, value=1 - beta2)

    if bias_correction:
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 / math.sqrt(1 - beta2**step)
    else:
        bias_correction1 = 1
        bias_correction2 = 1

    if weight_decay > 0 and adamw_mode:
        param.add_(param, alpha=-lr * weight_decay)
    denominator = exp_avg_sq.sqrt().mul_(bias_correction2).add_(eps)
    param.addcdiv_(exp_avg, denominator, value=-lr / bias_correction1)


@pytest.mark.parametrize('model_size', [3, 4, 5, 15, 16, 17, 511, 512, 513, 524287, 524288, 524289])
@pytest.mark.parametrize('adamw_mode,weight_decay,bias_correction', [
    (True, 0.0, True),
    (True, 0.07, True),
    (False, 0.07, True),
    (True, 0.07, False),
])
def test_cpu_adam_strict_state_updates(model_size, adamw_mode, weight_decay, bias_correction):
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    generator = torch.Generator().manual_seed(model_size)
    initial = torch.rand(model_size, generator=generator).mul_(1.5).sub_(0.75)
    cpu_param = torch.nn.Parameter(initial.clone())
    ref_param = initial.clone()
    ref_exp_avg = torch.zeros_like(ref_param)
    ref_exp_avg_sq = torch.zeros_like(ref_param)
    lr = 7e-4
    betas = (0.85, 0.97)
    eps = 1e-7

    optimizer = DeepSpeedCPUAdam([cpu_param],
                                 lr=lr,
                                 betas=betas,
                                 eps=eps,
                                 weight_decay=weight_decay,
                                 adamw_mode=adamw_mode,
                                 bias_correction=bias_correction)

    for step in range(1, 6):
        grad = torch.rand(model_size, generator=generator).mul_(1.5).sub_(0.75)
        cpu_param.grad = grad
        optimizer.step()
        _reference_adam_step(ref_param, grad, ref_exp_avg, ref_exp_avg_sq, step, lr, betas, eps, weight_decay,
                             adamw_mode, bias_correction)

        state = optimizer.state[cpu_param]
        assert state['step'] == step
        torch.testing.assert_close(cpu_param, ref_param, rtol=3e-5, atol=2e-6)
        torch.testing.assert_close(state['exp_avg'], ref_exp_avg, rtol=3e-5, atol=2e-6)
        torch.testing.assert_close(state['exp_avg_sq'], ref_exp_avg_sq, rtol=3e-5, atol=2e-6)


def test_cpu_adam_step_releases_gil():
    """A CPU Adam step in another thread lets this thread run, as SuperOffload's worker thread needs."""
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    # Large enough that a step takes several ms, longer than the wakeups of this thread are apart.
    param = torch.nn.Parameter(torch.zeros(2**26))
    param.grad = torch.full_like(param, 0.5)
    optimizer = DeepSpeedCPUAdam([param])
    optimizer.step()
    steps = []
    done = threading.Event()

    def step():
        for _ in range(5):
            start = time.perf_counter()
            optimizer.step()
            steps.append((start, time.perf_counter()))
        done.set()

    thread = threading.Thread(target=step)
    thread.start()
    wakeups = []
    while not done.is_set():
        time.sleep(0.001)
        wakeups.append(time.perf_counter())
    thread.join()
    # Holding the GIL during a step would keep this thread from waking up until the step is done, so none of its
    # wakeups would fall inside a step. Being descheduled now and then only loses a few.
    inside = sum(any(a + 1e-3 < t < b - 1e-3 for a, b in steps) for t in wakeups)
    assert inside >= 10, f"{inside} wakeups during {len(steps)} steps of {sum(b - a for a, b in steps) * 1e3:.0f} ms"


def test_cpu_adam_rejects_amsgrad():
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    with pytest.raises(RuntimeError, match="AMSGrad"):
        DeepSpeedCPUAdam([torch.nn.Parameter(torch.zeros(4))], amsgrad=True)

    optimizer = DeepSpeedCPUAdam([torch.nn.Parameter(torch.zeros(4))])
    with pytest.raises(RuntimeError, match="AMSGrad"):
        optimizer.add_param_group({'params': [torch.nn.Parameter(torch.zeros(4))], 'amsgrad': True})
    assert len(optimizer.param_groups) == 1


def test_cpu_adam_steps_fp32_params_on_bf16_grads():
    """fp32 params step on bf16 gradients as on the same gradients in fp32, and a gradient dtype that has no kernel
    is rejected instead of being read as the params' dtype."""
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    op = CPUAdamBuilder().load()
    opt_id = DeepSpeedCPUAdam.optimizer_id
    DeepSpeedCPUAdam.optimizer_id += 1
    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
    n = 100003  # not a multiple of the SIMD block, so the scalar tail runs too
    torch.manual_seed(0)
    params = torch.randn(n)
    grads = [torch.randn(n).to(torch.bfloat16) for _ in range(3)]
    results = []
    for grad_dtype in (torch.bfloat16, torch.float):
        # A new native optimizer for each, as it carries beta**step over from the steps before.
        op.create_adam(opt_id, lr, beta1, beta2, eps, wd, True, False)
        p, m, v = params.clone(), torch.zeros(n), torch.zeros(n)
        for step, grad in enumerate(grads, 1):
            op.adam_update(opt_id, step, lr, beta1, beta2, eps, wd, True, p, grad.to(grad_dtype), m, v)
        results.append((p, m, v))
        op.destroy_adam(opt_id)
    for name, a, b in zip(("params", "exp_avg", "exp_avg_sq"), *results):
        assert torch.equal(a, b), name

    op.create_adam(opt_id, lr, beta1, beta2, eps, wd, True, False)
    try:
        with pytest.raises(RuntimeError, match="grad type"):
            op.adam_update(opt_id, 1, lr, beta1, beta2, eps, wd, True, params.clone(), grads[0].half(), torch.zeros(n),
                           torch.zeros(n))
    finally:
        op.destroy_adam(opt_id)


def test_cpu_adam_scales_grads_as_it_reads_them():
    """adam_update with grad_scale steps bit for bit as on the gradient scaled before, as unscale_and_clip_grads
    scales it, for fp32 and bf16 gradients of fp32 params."""
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    op = CPUAdamBuilder().load()
    opt_id = DeepSpeedCPUAdam.optimizer_id
    DeepSpeedCPUAdam.optimizer_id += 1
    lr, beta1, beta2, eps, wd, scale = 1e-3, 0.9, 0.999, 1e-8, 0.01, 0.37
    n = 100003  # not a multiple of the SIMD block, so the scalar tail runs too
    torch.manual_seed(0)
    params = torch.randn(n)
    for grad_dtype in (torch.bfloat16, torch.float):
        grads = [torch.randn(n).to(grad_dtype) for _ in range(3)]
        results = []
        for in_kernel in (True, False):
            # A new native optimizer for each, as it carries beta**step over from the steps before.
            op.create_adam(opt_id, lr, beta1, beta2, eps, wd, True, False)
            p, m, v = params.clone(), torch.zeros(n), torch.zeros(n)
            for step, grad in enumerate(grads, 1):
                if in_kernel:
                    op.adam_update(opt_id, step, lr, beta1, beta2, eps, wd, True, p, grad, m, v, scale)
                else:
                    op.adam_update(opt_id, step, lr, beta1, beta2, eps, wd, True, p, grad.float() * scale, m, v)
            results.append((p, m, v))
            op.destroy_adam(opt_id)
        for name, a, b in zip(("params", "exp_avg", "exp_avg_sq"), *results):
            assert torch.equal(a, b), f"{grad_dtype} {name}"


@pytest.mark.parametrize("earlier_steps", [0, 3], ids=["first_step", "later_step"])
@pytest.mark.parametrize("grad_dtype", [torch.bfloat16, torch.float], ids=["bf16", "fp32"])
def test_cpu_adam_rescale_step(earlier_steps, grad_dtype):
    """adam_rescale_step turns a step on grad * old_scale into the step on grad * new_scale from the same state, up to
    rounding: 7e-6 of the update for the params and 1e-6 of the largest moment (2.5e-7 and 1e-6 on the first step,
    where the moments of before were 0). Rescaling to the same scale leaves the moments as they are, and takes the
    update out of the params and adds it back."""
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    op = CPUAdamBuilder().load()
    opt_id = DeepSpeedCPUAdam.optimizer_id
    DeepSpeedCPUAdam.optimizer_id += 1
    lr, beta1, beta2, eps, wd, old_scale, new_scale = 1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 0.2
    n = 100003  # not a multiple of the SIMD block, so the scalar tail runs too
    torch.manual_seed(0)
    params = torch.randn(n) * 0.02
    earlier = [torch.randn(n) * 1e-3 for _ in range(earlier_steps)]
    grad = (torch.randn(n) * 1e-3).to(grad_dtype)
    step = earlier_steps + 1
    results = []
    for rescale in (True, False):
        op.create_adam(opt_id, lr, beta1, beta2, eps, wd, True, False)
        p, m, v = params.clone(), torch.zeros(n), torch.zeros(n)
        for t, g in enumerate(earlier, 1):
            op.adam_update(opt_id, t, lr, beta1, beta2, eps, wd, True, p, g, m, v)
        before = p.clone()
        if rescale:
            op.adam_update(opt_id, step, lr, beta1, beta2, eps, wd, True, p, grad, m, v, old_scale)
            stepped = (p.clone(), m.clone(), v.clone())
            op.adam_rescale_step(opt_id, step, lr, beta1, beta2, eps, wd, True, p, grad, m, v, old_scale, old_scale)
            assert torch.equal(m, stepped[1]) and torch.equal(v, stepped[2])
            torch.testing.assert_close(p, stepped[0], rtol=0, atol=1e-5 * (stepped[0] - before).abs().max().item())
            op.adam_rescale_step(opt_id, step, lr, beta1, beta2, eps, wd, True, p, grad, m, v, old_scale, new_scale)
        else:
            op.adam_update(opt_id, step, lr, beta1, beta2, eps, wd, True, p, grad, m, v, new_scale)
        results.append((p, m, v))
        op.destroy_adam(opt_id)
    (p, m, v), (p_ref, m_ref, v_ref) = results
    torch.testing.assert_close(p, p_ref, rtol=0, atol=1e-4 * (p_ref - before).abs().max().item())
    torch.testing.assert_close(m, m_ref, rtol=0, atol=2e-6 * m_ref.abs().max().item())
    torch.testing.assert_close(v, v_ref, rtol=0, atol=1e-5 * v_ref.abs().max().item())


def test_cpu_adam_rescale_step_rejects_l2_weight_decay():
    """Adam's L2 decay adds the params of before the step to the gradient, which a rescale cannot recover."""
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    op = CPUAdamBuilder().load()
    opt_id = DeepSpeedCPUAdam.optimizer_id
    DeepSpeedCPUAdam.optimizer_id += 1
    op.create_adam(opt_id, 1e-3, 0.9, 0.999, 1e-8, 0.01, False, False)
    try:
        p, g, m, v = torch.randn(64), torch.randn(64), torch.zeros(64), torch.zeros(64)
        op.adam_update(opt_id, 1, 1e-3, 0.9, 0.999, 1e-8, 0.01, True, p, g, m, v)
        with pytest.raises(RuntimeError, match="AdamW only"):
            op.adam_rescale_step(opt_id, 1, 1e-3, 0.9, 0.999, 1e-8, 0.01, True, p, g, m, v, 1.0, 0.5)
    finally:
        op.destroy_adam(opt_id)


@pytest.mark.parametrize('dtype', [torch.half, torch.bfloat16, torch.float], ids=["fp16", "bf16", "fp32"])
@pytest.mark.parametrize('model_size',
                         [
                             (64),
                             (22),
                             #(55),
                             (128),
                             (1024),
                             (1048576),
                         ]) # yapf: disable
class TestCPUAdam(DistributedTest):
    world_size = 1
    reuse_dist_env = True
    requires_cuda_env = False
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    @pytest.mark.skipif(not get_accelerator().is_available(), reason="only supported in CUDA environments.")
    @pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME],
                        reason="FusedAdam is not compatible")
    def test_fused_adam_equal(self, dtype, model_size):
        if dtype not in get_accelerator().supported_dtypes():
            pytest.skip(f"dtype {dtype} not supported in current accelerator")

        if ("amd" in pytest.cpu_vendor) and (dtype == torch.half):
            pytest.skip("cpu-adam with half precision not supported on AMD CPUs")

        from deepspeed.ops.adam import DeepSpeedCPUAdam

        cpu_data = torch.randn(model_size, device='cpu').to(dtype)
        cpu_param = torch.nn.Parameter(cpu_data)
        cuda_param = torch.nn.Parameter(cpu_data.to(get_accelerator().device_name()))

        # tolerance = cpu_param.float().norm().detach().numpy() * 1e-2
        # check_equal(cpu_param.float().norm(),
        #             cuda_param.float().cpu().norm(),
        #             atol=tolerance,
        #             verbose=True)

        cpu_optimizer = DeepSpeedCPUAdam([cpu_param])
        cuda_optimizer = FusedAdam([cuda_param])

        _compare_optimizers(model_size=model_size,
                            param1=cpu_param,
                            optimizer1=cpu_optimizer,
                            param2=cuda_param,
                            optimizer2=cuda_optimizer)

    def test_torch_adamw_equal(self, dtype, model_size):
        if get_accelerator().is_available():
            if dtype == torch.half:
                pytest.skip("torch.optim.AdamW with half precision inf/nan output.")
            if ("amd" in pytest.cpu_vendor) and (dtype == torch.half):
                pytest.skip("cpu-adam with half precision not supported on AMD CPUs")
            ref_param_device = get_accelerator().device_name()
        else:
            if dtype == torch.half:
                pytest.skip("torch.optim.AdamW with half precision only supported in CUDA environments.")
            ref_param_device = 'cpu'

        from deepspeed.ops.adam import DeepSpeedCPUAdam

        cpu_data = torch.randn(model_size, device='cpu').to(dtype)
        cpu_param = torch.nn.Parameter(cpu_data)
        ref_param = torch.nn.Parameter(cpu_data.to(ref_param_device))

        cpu_optimizer = DeepSpeedCPUAdam([cpu_param])
        ref_optimizer = torch.optim.AdamW([ref_param])

        _compare_optimizers(model_size=model_size,
                            param1=cpu_param,
                            optimizer1=cpu_optimizer,
                            param2=ref_param,
                            optimizer2=ref_optimizer)


class TestCPUAdamBf16OptimizerStates(DistributedTest):
    world_size = 1
    reuse_dist_env = True
    requires_cuda_env = False
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    @pytest.mark.parametrize('model_size', [64, 1024])
    def test_bf16_optimizer_states_dtype(self, model_size):
        """fp32_optimizer_states=False keeps the Adam moments in the bf16 parameter precision."""
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        param = torch.nn.Parameter(torch.randn(model_size, device='cpu', dtype=torch.bfloat16))
        optimizer = DeepSpeedCPUAdam([param], fp32_optimizer_states=False)
        param.grad = torch.randn(model_size, device='cpu', dtype=torch.bfloat16)
        optimizer.step()

        state = optimizer.state[param]
        assert state['exp_avg'].dtype == torch.bfloat16
        assert state['exp_avg_sq'].dtype == torch.bfloat16
        assert state['exp_avg'].device == torch.device('cpu')
        assert state['exp_avg_sq'].device == torch.device('cpu')

    @pytest.mark.parametrize('model_size', [64, 1024])
    def test_bf16_optimizer_states_match_fp32(self, model_size):
        """bf16 moments should track fp32 moments within bf16 tolerance over several steps."""
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        torch.manual_seed(0)
        base = torch.randn(model_size, device='cpu', dtype=torch.float32).to(torch.bfloat16)
        param_fp32_states = torch.nn.Parameter(base.clone())
        param_bf16_states = torch.nn.Parameter(base.clone())

        opt_fp32_states = DeepSpeedCPUAdam([param_fp32_states], fp32_optimizer_states=True)
        opt_bf16_states = DeepSpeedCPUAdam([param_bf16_states], fp32_optimizer_states=False)

        for _ in range(10):
            grad = torch.randn(model_size, device='cpu', dtype=torch.bfloat16)
            param_fp32_states.grad = grad.clone()
            param_bf16_states.grad = grad.clone()
            opt_fp32_states.step()
            opt_bf16_states.step()

        assert opt_fp32_states.state[param_fp32_states]['exp_avg'].dtype == torch.float32
        assert opt_bf16_states.state[param_bf16_states]['exp_avg'].dtype == torch.bfloat16

        # bf16 moments round every Adam update to an 8-bit mantissa, so over 10 steps they
        # diverge from fp32 moments more than the same-precision comparison in _compare_optimizers
        # (1e-2). A wider 5% band keeps this stable while still catching gross errors; the dtype
        # assertions above guard the precision itself. Norm comparison follows _compare_optimizers.
        tolerance = param_fp32_states.float().norm().detach().numpy() * 5e-2
        check_equal(param_fp32_states.float().norm(), param_bf16_states.float().norm(), atol=tolerance)


def _zenflow_adam_proc_worker(param, g0, g1, ea0, ea1, eq0, eq1, stale, ctrl, ready, affinity):
    op = CPUAdamBuilder().load()
    op.create_adam(0, 1e-3, 0.9, 0.999, 1e-8, 0.0, True, False)
    handle = op.zenflow_adam_create(0, affinity)
    op.zenflow_adam_register_group(handle, param, g0, g1, ea0, ea1, eq0, eq1, stale)
    ready.set()
    op.zenflow_adam_run_worker(handle, ctrl.data_ptr())  # blocks until the exit command
    op.zenflow_adam_destroy(handle)
    op.destroy_adam(0)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="cross-process ZenFlowAdam is Linux-only")
def test_zenflow_adam_cross_process():
    """The optimizer-process driver (shared-memory semaphore control + native worker, the
    production path for ZenFlow stage 1/2 overlap) must match a per-parameter adam_update
    reference bit-for-bit with alternating double buffers. Run as a plain test, not
    DistributedTest, so the pytest process (non-daemonic) can spawn the optimizer process."""
    import torch.multiprocessing as mp

    op = CPUAdamBuilder().load()
    if not hasattr(op, "zenflow_adam_ctrl_size"):
        pytest.skip("cross-process ZenFlowAdam not available in this build")

    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.0
    n = 100003  # non-SIMD-aligned, exercises the scalar tail
    affinity = list(range(min(4, os.cpu_count() or 1)))

    ctrl = torch.zeros(op.zenflow_adam_ctrl_size(), dtype=torch.uint8).share_memory_()
    op.zenflow_adam_ctrl_init(ctrl.data_ptr(), 1)

    torch.manual_seed(0)
    param = torch.randn(n).share_memory_()
    g = [torch.zeros(n).share_memory_(), torch.zeros(n).share_memory_()]
    ea = [torch.zeros(n).share_memory_(), torch.zeros(n).share_memory_()]
    eq = [torch.zeros(n).share_memory_(), torch.zeros(n).share_memory_()]
    stale = torch.zeros(n).share_memory_()

    op.create_adam(1, lr, beta1, beta2, eps, wd, True, False)
    p_ref = param.clone()
    ea_ref = [ea[0].clone(), ea[1].clone()]
    eq_ref = [eq[0].clone(), eq[1].clone()]
    st_ref = stale.clone()

    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    proc = ctx.Process(target=_zenflow_adam_proc_worker,
                       args=(param, g[0], g[1], ea[0], ea[1], eq[0], eq[1], stale, ctrl, ready, affinity))
    proc.start()
    try:
        assert ready.wait(timeout=60), "optimizer process did not start"
        # With no step submitted yet, a bounded wait must time out (return False) rather than
        # block -- this is what lets the training side notice a dead optimizer process.
        assert op.zenflow_adam_wait(ctrl.data_ptr(), 0.05) is False, "wait should time out when no step is pending"
        for step in range(1, 6):
            now = step & 1
            grad = torch.randn(n)
            g[now].copy_(grad)
            op.zenflow_adam_submit(ctrl.data_ptr(), now, step, [lr], [beta1], [beta2], [eps], [wd], [1])
            assert op.zenflow_adam_wait(ctrl.data_ptr(), 60.0), f"wait timed out step {step}"
            # Reference: single-tensor adam_update on the mirror, then snapshot the updated
            # param into the stale buffer -- exactly what the native worker does per group.
            op.adam_update(1, step, lr, beta1, beta2, eps, wd, True, p_ref, grad.clone(), ea_ref[now], eq_ref[now])
            st_ref.copy_(p_ref)
            assert torch.equal(param, p_ref), f"param mismatch step {step}"
            assert torch.equal(ea[now], ea_ref[now]), f"exp_avg mismatch step {step}"
            assert torch.equal(eq[now], eq_ref[now]), f"exp_avg_sq mismatch step {step}"
            assert torch.equal(stale, st_ref), f"stale mismatch step {step}"
        op.zenflow_adam_ctrl_exit(ctrl.data_ptr())
        proc.join(timeout=10)
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        op.destroy_adam(1)


class TestCPUAdamGPUError(DistributedTest):

    # The assertion under test can only fire for a param on a discrete accelerator;
    # on the cpu accelerator step() returns normally and the rank pool deadlocks.
    @pytest.mark.skipif(get_accelerator().device_name() == 'cpu', reason="requires a discrete accelerator device")
    def test_cpu_adam_gpu_error(self):
        model_size = 64
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        device = get_accelerator().device_name(0)  # 'cuda:0' or 'xpu:0'
        param = torch.nn.Parameter(torch.randn(model_size, device=device))
        optimizer = DeepSpeedCPUAdam([param])

        param.grad = torch.randn(model_size, device=device)
        with pytest.raises(AssertionError):
            optimizer.step()


class TestCPUAdamSubgroup(DistributedTest):
    world_size = 1
    reuse_dist_env = True
    requires_cuda_env = False
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    @pytest.mark.parametrize('dtype', [torch.half, torch.bfloat16], ids=["fp16", "bf16"])
    @pytest.mark.parametrize('model_size', [64, 128, 1024])
    def test_step_subgroup_basic(self, dtype, model_size):
        """Test basic functionality of step_subgroup method."""
        if ("amd" in pytest.cpu_vendor) and (dtype == torch.half):
            pytest.skip("cpu-adam with half precision not supported on AMD CPUs")

        from deepspeed.ops.adam import DeepSpeedCPUAdam

        # Create parameters
        cpu_data = torch.randn(model_size, device='cpu').to(dtype)
        param = torch.nn.Parameter(cpu_data)
        optimizer = DeepSpeedCPUAdam([param])

        # Set gradient
        param.grad = torch.randn(model_size, device='cpu').to(dtype)

        # Store initial parameter values
        initial_param = param.data.clone()

        # Test step_subgroup with subgroup_id=0
        subgroup_id = 0
        optimizer.step_subgroup(subgroup_id)

        # Verify parameter was updated
        assert not torch.equal(param.data, initial_param), "Parameters should be updated after step_subgroup"

        # Verify optimizer state was created for subgroup
        assert subgroup_id in optimizer.state, "Optimizer state should be created for subgroup"
        assert optimizer.state[subgroup_id]['step'] == 1, "Step count should be 1"
        assert 'exp_avg' in optimizer.state[subgroup_id], "exp_avg should be in state"
        assert 'exp_avg_sq' in optimizer.state[subgroup_id], "exp_avg_sq should be in state"

    @pytest.mark.parametrize('dtype', [torch.half, torch.bfloat16], ids=["fp16", "bf16"])
    def test_step_subgroup_multiple_calls(self, dtype):
        """Test multiple calls to step_subgroup increment step count correctly."""
        if ("amd" in pytest.cpu_vendor) and (dtype == torch.half):
            pytest.skip("cpu-adam with half precision not supported on AMD CPUs")

        from deepspeed.ops.adam import DeepSpeedCPUAdam

        model_size = 64
        cpu_data = torch.randn(model_size, device='cpu').to(dtype)
        param = torch.nn.Parameter(cpu_data)
        optimizer = DeepSpeedCPUAdam([param])

        subgroup_id = 0

        # Perform multiple steps
        for step in range(1, 4):
            param.grad = torch.randn(model_size, device='cpu').to(dtype)
            optimizer.step_subgroup(subgroup_id)

            # Verify step count increments
            assert optimizer.state[subgroup_id]['step'] == step, f"Step count should be {step}"

    @pytest.mark.parametrize('dtype', [torch.half, torch.bfloat16], ids=["fp16", "bf16"])
    def test_rollback_subgroup_basic(self, dtype):
        """Test basic functionality of rollback_subgroup method."""
        if ("amd" in pytest.cpu_vendor) and (dtype == torch.half):
            pytest.skip("cpu-adam with half precision not supported on AMD CPUs")

        from deepspeed.ops.adam import DeepSpeedCPUAdam

        model_size = 64
        cpu_data = torch.randn(model_size, device='cpu').to(dtype)
        param = torch.nn.Parameter(cpu_data)
        optimizer = DeepSpeedCPUAdam([param])

        subgroup_id = 0
        param.grad = torch.randn(model_size, device='cpu').to(dtype)

        # First, perform a step to initialize state
        optimizer.step_subgroup(subgroup_id)
        assert optimizer.state[subgroup_id]['step'] == 1

        # Store parameter state after step
        param_after_step = param.data.clone()
        exp_avg_after_step = optimizer.state[subgroup_id]['exp_avg'].clone()
        exp_avg_sq_after_step = optimizer.state[subgroup_id]['exp_avg_sq'].clone()

        # Now rollback
        optimizer.rollback_subgroup(subgroup_id)

        # Verify step count decremented
        assert optimizer.state[subgroup_id]['step'] == 0, "Step count should be decremented after rollback"

    def test_rollback_of_first_step_keeps_exp_avg_sq_non_negative(self):
        """exp_avg_sq is 0 before the first step. Rolling that step back must not leave it negative, because sqrt()
        of a negative value in the next step turns the parameter into NaN."""
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        model_size = 4096
        param = torch.nn.Parameter(torch.randn(model_size, device='cpu'))
        optimizer = DeepSpeedCPUAdam([param])
        param.grad = torch.randn(model_size, device='cpu')

        optimizer.step_subgroup(0)
        optimizer.rollback_subgroup(0)
        assert (optimizer.state[0]['exp_avg_sq'] >= 0).all()

        optimizer.step_subgroup(0)
        assert torch.isfinite(param.data).all()

    def test_rollback_subgroup_uninitialized_error(self):
        """Test that rollback_subgroup raises error for uninitialized subgroup."""
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        model_size = 64
        param = torch.nn.Parameter(torch.randn(model_size, device='cpu'))
        optimizer = DeepSpeedCPUAdam([param])

        # Try to rollback uninitialized subgroup
        with pytest.raises(RuntimeError, match="Cannot rollback optimizer state for sub_group_id 0"):
            optimizer.rollback_subgroup(0)

    def test_rollback_subgroup_zero_step_error(self):
        """Test that rollback_subgroup raises error when step count is already 0."""
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        model_size = 64
        param = torch.nn.Parameter(torch.randn(model_size, device='cpu'))
        optimizer = DeepSpeedCPUAdam([param])

        subgroup_id = 0
        param.grad = torch.randn(model_size, device='cpu')

        # Initialize state by doing one step
        optimizer.step_subgroup(subgroup_id)

        # Rollback once (step should become 0)
        optimizer.rollback_subgroup(subgroup_id)
        assert optimizer.state[subgroup_id]['step'] == 0

        # Try to rollback again - should raise error
        with pytest.raises(RuntimeError, match="Cannot rollback sub_group_id 0: step count is 0"):
            optimizer.rollback_subgroup(subgroup_id)

    @pytest.mark.parametrize('dtype', [torch.half, torch.bfloat16], ids=["fp16", "bf16"])
    def test_step_rollback_sequence(self, dtype):
        """Test sequence of step_subgroup and rollback_subgroup operations."""
        if ("amd" in pytest.cpu_vendor) and (dtype == torch.half):
            pytest.skip("cpu-adam with half precision not supported on AMD CPUs")

        from deepspeed.ops.adam import DeepSpeedCPUAdam

        model_size = 64
        cpu_data = torch.randn(model_size, device='cpu').to(dtype)
        param = torch.nn.Parameter(cpu_data)
        optimizer = DeepSpeedCPUAdam([param])

        subgroup_id = 0
        param.grad = torch.randn(model_size, device='cpu').to(dtype)

        # Perform multiple steps
        for step in range(1, 4):
            optimizer.step_subgroup(subgroup_id)
            assert optimizer.state[subgroup_id]['step'] == step

        # Rollback steps one by one
        for step in range(2, -1, -1):
            optimizer.rollback_subgroup(subgroup_id)
            assert optimizer.state[subgroup_id]['step'] == step

    def test_multiple_subgroups(self):
        """Test that different subgroups maintain independent state."""
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        model_size = 64
        param = torch.nn.Parameter(torch.randn(model_size, device='cpu'))
        optimizer = DeepSpeedCPUAdam([param])

        param.grad = torch.randn(model_size, device='cpu')

        # Step different subgroups
        optimizer.step_subgroup(0)
        optimizer.step_subgroup(1)
        optimizer.step_subgroup(0)  # Step subgroup 0 again

        # Verify independent step counts
        assert optimizer.state[0]['step'] == 2, "Subgroup 0 should have step count 2"
        assert optimizer.state[1]['step'] == 1, "Subgroup 1 should have step count 1"

        # Rollback subgroup 0 only
        optimizer.rollback_subgroup(0)
        assert optimizer.state[0]['step'] == 1, "Subgroup 0 step count should be decremented"
        assert optimizer.state[1]['step'] == 1, "Subgroup 1 step count should be unchanged"

    def test_step_subgroup_same_step_idempotent_across_subgroups(self):
        """Repeated same-step subgroup updates should remain bit-identical."""
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        model_size = 128
        steps = 4
        base = torch.randn(model_size, device='cpu', dtype=torch.float32)
        param_a = torch.nn.Parameter(base.clone())
        param_b = torch.nn.Parameter(base.clone())

        optimizer = DeepSpeedCPUAdam([param_a])
        for logical_step in range(1, steps + 1):
            grad = torch.randn(model_size, device='cpu', dtype=torch.float32)

            optimizer.param_groups[0]['params'] = [param_a]
            param_a.grad = grad.clone()
            optimizer.step_subgroup(0)

            optimizer.param_groups[0]['params'] = [param_b]
            param_b.grad = grad.clone()
            optimizer.step_subgroup(1)

            assert optimizer.state[0]['step'] == logical_step
            assert optimizer.state[1]['step'] == logical_step
            assert torch.equal(param_a.data, param_b.data)
            assert torch.equal(optimizer.state[0]['exp_avg'], optimizer.state[1]['exp_avg'])
            assert torch.equal(optimizer.state[0]['exp_avg_sq'], optimizer.state[1]['exp_avg_sq'])

    def test_step_same_step_idempotent_across_param_keys(self):
        """Repeated optimizer.step() with swapped param keys should be deterministic."""
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        model_size = 128
        steps = 4
        base = torch.randn(model_size, device='cpu', dtype=torch.float32)
        param_a = torch.nn.Parameter(base.clone())
        param_b = torch.nn.Parameter(base.clone())

        optimizer = DeepSpeedCPUAdam([param_a])
        for logical_step in range(1, steps + 1):
            grad = torch.randn(model_size, device='cpu', dtype=torch.float32)

            optimizer.param_groups[0]['params'] = [param_a]
            param_a.grad = grad.clone()
            optimizer.step()

            optimizer.param_groups[0]['params'] = [param_b]
            param_b.grad = grad.clone()
            optimizer.step()

            assert optimizer.state[param_a]['step'] == logical_step
            assert optimizer.state[param_b]['step'] == logical_step
            assert torch.equal(param_a.data, param_b.data)
            assert torch.equal(optimizer.state[param_a]['exp_avg'], optimizer.state[param_b]['exp_avg'])
            assert torch.equal(optimizer.state[param_a]['exp_avg_sq'], optimizer.state[param_b]['exp_avg_sq'])
