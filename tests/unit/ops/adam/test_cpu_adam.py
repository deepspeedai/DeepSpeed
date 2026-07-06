# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import sys
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
