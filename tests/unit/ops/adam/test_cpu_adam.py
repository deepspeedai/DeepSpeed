# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

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

pytest.cpu_vendor = get_cpu_info()["vendor_id_raw"].lower()


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


class TestCPUAdamFusedMultiTensor(DistributedTest):
    """adam_update_multi (fused multi-tensor, used by ZenFlow overlap) must match a
    per-parameter sequence of adam_update bit-for-bit, and write the post-update
    parameter snapshot into the stale buffer."""
    world_size = 1
    reuse_dist_env = True
    requires_cuda_env = False
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    @pytest.mark.parametrize('dtype', [torch.half, torch.bfloat16, torch.float], ids=["fp16", "bf16", "fp32"])
    def test_multi_matches_single(self, dtype):
        if ("amd" in pytest.cpu_vendor) and (dtype == torch.half):
            pytest.skip("cpu-adam with half precision not supported on AMD CPUs")

        ds_opt_adam = CPUAdamBuilder().load()

        lr, beta1, beta2, eps, weight_decay = 1e-3, 0.9, 0.999, 1e-8, 0.0
        adamw_mode, bias_correction = True, True
        # Mixed sizes (including ones that don't divide the SIMD width) exercise both the
        # vectorized and scalar tails inside the fused C++ loop.
        sizes = [64, 22, 1024, 1048576]

        opt_single, opt_multi = 0, 1
        ds_opt_adam.create_adam(opt_single, lr, beta1, beta2, eps, weight_decay, adamw_mode, False)
        ds_opt_adam.create_adam(opt_multi, lr, beta1, beta2, eps, weight_decay, adamw_mode, False)

        torch.manual_seed(0)
        params_single = [torch.randn(n, dtype=dtype) for n in sizes]
        params_multi = [p.clone() for p in params_single]
        exp_avg_single = [torch.zeros(n, dtype=torch.float) for n in sizes]
        exp_avg_sq_single = [torch.zeros(n, dtype=torch.float) for n in sizes]
        exp_avg_multi = [torch.zeros(n, dtype=torch.float) for n in sizes]
        exp_avg_sq_multi = [torch.zeros(n, dtype=torch.float) for n in sizes]
        stale_multi = [torch.zeros(n, dtype=dtype) for n in sizes]

        try:
            for step in range(1, 6):
                grads = [torch.randn(n, dtype=dtype) for n in sizes]

                for i in range(len(sizes)):
                    ds_opt_adam.adam_update(opt_single, step, lr, beta1, beta2, eps, weight_decay, bias_correction,
                                            params_single[i], grads[i].clone(), exp_avg_single[i],
                                            exp_avg_sq_single[i])

                ds_opt_adam.adam_update_multi(opt_multi, step, lr, beta1, beta2, eps, weight_decay, bias_correction,
                                              params_multi, [g.clone() for g in grads], exp_avg_multi,
                                              exp_avg_sq_multi, stale_multi)

                for i in range(len(sizes)):
                    assert torch.equal(params_single[i], params_multi[i]), f"param mismatch at size {sizes[i]}"
                    assert torch.equal(exp_avg_single[i], exp_avg_multi[i]), f"exp_avg mismatch at size {sizes[i]}"
                    assert torch.equal(exp_avg_sq_single[i],
                                       exp_avg_sq_multi[i]), f"exp_avg_sq mismatch at size {sizes[i]}"
                    # stale must hold the post-update parameter snapshot
                    assert torch.equal(stale_multi[i], params_multi[i]), f"stale mismatch at size {sizes[i]}"
        finally:
            ds_opt_adam.destroy_adam(opt_single)
            ds_opt_adam.destroy_adam(opt_multi)

    @pytest.mark.parametrize('dtype', [torch.half, torch.bfloat16, torch.float], ids=["fp16", "bf16", "fp32"])
    def test_serial_matches_parallel(self, dtype):
        """The serial kernel path (parallel=False, used by ZenFlow's pinned thread pool)
        must match the OpenMP path (parallel=True) bit-for-bit."""
        if ("amd" in pytest.cpu_vendor) and (dtype == torch.half):
            pytest.skip("cpu-adam with half precision not supported on AMD CPUs")

        ds_opt_adam = CPUAdamBuilder().load()
        lr, beta1, beta2, eps, weight_decay = 1e-3, 0.9, 0.999, 1e-8, 0.0
        sizes = [64, 22, 1024, 1048576]

        opt_par, opt_ser = 3, 4
        ds_opt_adam.create_adam(opt_par, lr, beta1, beta2, eps, weight_decay, True, False)
        ds_opt_adam.create_adam(opt_ser, lr, beta1, beta2, eps, weight_decay, True, False)

        torch.manual_seed(0)
        params_par = [torch.randn(n, dtype=dtype) for n in sizes]
        params_ser = [p.clone() for p in params_par]
        ea_par = [torch.zeros(n) for n in sizes]
        eq_par = [torch.zeros(n) for n in sizes]
        ea_ser = [torch.zeros(n) for n in sizes]
        eq_ser = [torch.zeros(n) for n in sizes]

        try:
            for step in range(1, 4):
                grads = [torch.randn(n, dtype=dtype) for n in sizes]
                ds_opt_adam.adam_update_multi(opt_par,
                                              step,
                                              lr,
                                              beta1,
                                              beta2,
                                              eps,
                                              weight_decay,
                                              True,
                                              params_par, [g.clone() for g in grads],
                                              ea_par,
                                              eq_par, [],
                                              parallel=True)
                ds_opt_adam.adam_update_multi(opt_ser,
                                              step,
                                              lr,
                                              beta1,
                                              beta2,
                                              eps,
                                              weight_decay,
                                              True,
                                              params_ser, [g.clone() for g in grads],
                                              ea_ser,
                                              eq_ser, [],
                                              parallel=False)
                for i in range(len(sizes)):
                    assert torch.equal(params_par[i], params_ser[i]), f"param mismatch at size {sizes[i]}"
                    assert torch.equal(ea_par[i], ea_ser[i]), f"exp_avg mismatch at size {sizes[i]}"
                    assert torch.equal(eq_par[i], eq_ser[i]), f"exp_avg_sq mismatch at size {sizes[i]}"
        finally:
            ds_opt_adam.destroy_adam(opt_par)
            ds_opt_adam.destroy_adam(opt_ser)

    def test_multi_without_stale(self):
        """An empty stale list is allowed and simply skips the snapshot."""
        ds_opt_adam = CPUAdamBuilder().load()
        opt_id = 2
        ds_opt_adam.create_adam(opt_id, 1e-3, 0.9, 0.999, 1e-8, 0.0, True, False)
        try:
            params = [torch.randn(64, dtype=torch.float)]
            grads = [torch.randn(64, dtype=torch.float)]
            exp_avg = [torch.zeros(64, dtype=torch.float)]
            exp_avg_sq = [torch.zeros(64, dtype=torch.float)]
            before = params[0].clone()
            ds_opt_adam.adam_update_multi(opt_id, 1, 1e-3, 0.9, 0.999, 1e-8, 0.0, True, params, grads, exp_avg,
                                          exp_avg_sq, [])
            assert not torch.equal(params[0], before), "params should be updated even without stale buffers"
        finally:
            ds_opt_adam.destroy_adam(opt_id)


class TestZenFlowAdamNative(DistributedTest):
    """ZenFlowAdam (in-process background thread + pinned pool, sliced serial kernel)
    must produce the same update as the reference fused path, with the alternating
    double-buffered grads/moments that ZenFlow's overlap uses."""
    world_size = 1
    reuse_dist_env = True
    requires_cuda_env = False
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    @pytest.mark.parametrize('dtype', [torch.float, torch.bfloat16], ids=["fp32", "bf16"])
    def test_matches_reference(self, dtype):
        import os
        ds = CPUAdamBuilder().load()
        lr, beta1, beta2, eps, weight_decay = 1e-3, 0.9, 0.999, 1e-8, 0.0
        # Sizes that exercise the multi-thread slicing: smaller than the pool, not a
        # multiple of it, and large.
        sizes = [3, 1000, 100003]

        opt_zf, opt_ref = 5, 6
        ds.create_adam(opt_zf, lr, beta1, beta2, eps, weight_decay, True, False)
        ds.create_adam(opt_ref, lr, beta1, beta2, eps, weight_decay, True, False)

        affinity = list(range(min(4, os.cpu_count() or 1)))
        handle = ds.zenflow_adam_create(opt_zf, affinity)

        torch.manual_seed(0)
        # ZenFlowAdam state (double-buffered) and the reference mirror of it.
        params_zf = [torch.randn(n, dtype=dtype) for n in sizes]
        params_ref = [p.clone() for p in params_zf]
        grad = [[torch.zeros(n, dtype=dtype) for n in sizes] for _ in range(2)]
        ea = [[torch.zeros(n) for n in sizes] for _ in range(2)]
        eq = [[torch.zeros(n) for n in sizes] for _ in range(2)]
        stale = [torch.zeros(n, dtype=dtype) for n in sizes]
        ea_ref = [[t.clone() for t in ea[s]] for s in range(2)]
        eq_ref = [[t.clone() for t in eq[s]] for s in range(2)]
        stale_ref = [t.clone() for t in stale]

        for i in range(len(sizes)):
            ds.zenflow_adam_register_group(handle, params_zf[i], grad[0][i], grad[1][i], ea[0][i], ea[1][i], eq[0][i],
                                           eq[1][i], stale[i])

        try:
            for step in range(1, 6):
                now = step & 1
                grads = [torch.randn(n, dtype=dtype) for n in sizes]
                for i in range(len(sizes)):
                    grad[now][i].copy_(grads[i])

                ds.zenflow_adam_submit(handle, now, step, [lr] * len(sizes), [beta1] * len(sizes),
                                       [beta2] * len(sizes), [eps] * len(sizes), [weight_decay] * len(sizes),
                                       [1] * len(sizes))
                ds.zenflow_adam_wait(handle)

                ds.adam_update_multi(opt_ref, step, lr, beta1, beta2, eps, weight_decay, True, params_ref,
                                     [g.clone() for g in grads], ea_ref[now], eq_ref[now], stale_ref)

                for i in range(len(sizes)):
                    assert torch.equal(params_zf[i], params_ref[i]), f"param mismatch size {sizes[i]} step {step}"
                    assert torch.equal(ea[now][i], ea_ref[now][i]), f"exp_avg mismatch size {sizes[i]} step {step}"
                    assert torch.equal(eq[now][i], eq_ref[now][i]), f"exp_avg_sq mismatch size {sizes[i]}"
                    assert torch.equal(stale[i], stale_ref[i]), f"stale mismatch size {sizes[i]} step {step}"
        finally:
            ds.zenflow_adam_destroy(handle)
            ds.destroy_adam(opt_zf)
            ds.destroy_adam(opt_ref)

    def test_pipelined_submit_wait(self):
        """Mirror the engine's pipeline: warmup does submit-then-wait, steady state does
        wait-then-submit (each wait drains the *previous* submit), leaving one undrained
        completion that destroy() cleans up. Must not hang or desync."""
        import os
        ds = CPUAdamBuilder().load()
        lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.0
        n = 1024
        opt_id = 7
        ds.create_adam(opt_id, lr, beta1, beta2, eps, wd, True, False)
        handle = ds.zenflow_adam_create(opt_id, list(range(min(4, os.cpu_count() or 1))))

        param = torch.randn(n)
        g = [torch.zeros(n), torch.zeros(n)]
        ea = [torch.zeros(n), torch.zeros(n)]
        eq = [torch.zeros(n), torch.zeros(n)]
        stale = torch.zeros(n)
        ds.zenflow_adam_register_group(handle, param, g[0], g[1], ea[0], ea[1], eq[0], eq[1], stale)

        def submit(now, step):
            g[now].copy_(torch.randn(n))
            ds.zenflow_adam_submit(handle, now, step, [lr], [beta1], [beta2], [eps], [wd], [1])

        try:
            # warmup: submit then wait (no overlap)
            submit(1, 1)
            ds.zenflow_adam_wait(handle)
            # steady: the first post-warmup wait is skipped, so this round is submit-only,
            # and every later wait drains the submit from the previous round.
            submit(0, 2)
            for step in range(3, 8):
                ds.zenflow_adam_wait(handle)
                submit(step & 1, step)
            ds.zenflow_adam_wait(handle)  # drain the last submitted step
            assert torch.all(torch.isfinite(param))
        finally:
            ds.zenflow_adam_destroy(handle)
            ds.destroy_adam(opt_id)


class TestCPUAdamGPUError(DistributedTest):

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
