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
