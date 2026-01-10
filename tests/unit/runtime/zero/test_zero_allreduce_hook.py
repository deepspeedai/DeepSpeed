# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
import deepspeed.comm as dist
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader
from deepspeed.accelerator import get_accelerator


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("contiguous_gradients", [True, False])
@pytest.mark.parametrize("reduce_bucket_size", [500000000, 10])
@pytest.mark.parametrize("reduce_scatter", [True, False])
@pytest.mark.parametrize("overlap_comm", [True, False])
@pytest.mark.parametrize("gradient_accumulation_steps", [1, 4])
class TestZeroAllReduceHook(DistributedTest):
    """Test _all_reduce_hook functionality for ZeRO stage 1, 2 and 3"""
    world_size = 4  # 4 processes to simulate 2 replica groups

    def test(self, zero_stage, contiguous_gradients, reduce_bucket_size, reduce_scatter, overlap_comm,
             gradient_accumulation_steps):
        """
        Test that _all_reduce_hook is called correctly and performs cross-replica gradient sync.

        Setup:
        - 4 processes split into 2 replica groups
        - Replica group 0: ranks [0, 1]
        - Replica group 1: ranks [2, 3]
        - Same initial parameters across all ranks
        - Different training data per replica group

        Verification:
        - Hook is called with gradient tensors
        - All ranks have identical model parameters after training (proves gradient sync works)
        """

        rank = dist.get_rank()

        # Create replica groups
        replica_group_0_ranks = [0, 1]
        replica_group_1_ranks = [2, 3]

        replica_group_0 = dist.new_group(ranks=replica_group_0_ranks)
        replica_group_1 = dist.new_group(ranks=replica_group_1_ranks)

        if rank in replica_group_0_ranks:
            replica_dp_group = replica_group_0
            replica_id = 0
        else:
            replica_dp_group = replica_group_1
            replica_id = 1

        # Create cross-replica groups for gradient synchronization
        # IMPORTANT: All ranks must call dist.new_group() for all groups!
        cross_replica_group_0 = dist.new_group(ranks=[0, 2])  # All 4 ranks must call this
        cross_replica_group_1 = dist.new_group(ranks=[1, 3])  # All 4 ranks must call this

        local_rank_in_replica = rank % 2
        if local_rank_in_replica == 0:
            cross_replica_group = cross_replica_group_0
        else:
            cross_replica_group = cross_replica_group_1

        # Create a custom MPU object to specify replica-specific DP group
        # This is crucial for stage 3 to ensure parameters are sharded correctly
        class ReplicaMPU:
            """Custom MPU that provides replica-specific data parallel group"""

            def __init__(self, dp_group):
                self._dp_group = dp_group

            def get_data_parallel_group(self):
                return self._dp_group

            def get_data_parallel_world_size(self):
                return dist.get_world_size(group=self._dp_group)

            def get_data_parallel_rank(self):
                return dist.get_rank(group=self._dp_group)

            def get_model_parallel_world_size(self):
                """Return 1 as we don't use model parallelism"""
                return 1

            def get_model_parallel_rank(self):
                """Return 0 as we don't use model parallelism"""
                return 0

            def get_model_parallel_group(self):
                """Return None as we don't use model parallelism"""
                return None

        replica_mpu = ReplicaMPU(replica_dp_group)

        # Track hook invocations
        hook_call_count = [0]
        hook_tensors = []

        def cross_replica_gradient_sync_hook(tensor):
            """Hook that averages gradients across replica groups"""
            hook_call_count[0] += 1
            hook_tensors.append(tensor.clone().detach())
            # Synchronize gradients across replica groups
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG, group=cross_replica_group)

        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": zero_stage,
                "contiguous_gradients": contiguous_gradients,
                "reduce_bucket_size": reduce_bucket_size,
                "reduce_scatter": reduce_scatter,
                "overlap_comm": overlap_comm,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.01
                }
            },
        }

        # Stage 3 specific configuration
        if zero_stage == 3:
            config_dict["zero_optimization"]["stage3_param_persistence_threshold"] = 0

        if get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        elif get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}

        hidden_dim = 10

        # Create model with same initial parameters for all ranks
        torch.manual_seed(42)  # Same seed for all ranks
        model = SimpleModel(hidden_dim=hidden_dim)

        # Pass the replica_mpu to deepspeed.initialize so that parameters
        # are sharded according to the replica-specific DP group from the start
        model, _, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
            dist_init_required=False,
            mpu=replica_mpu,
        )

        # Set the all_reduce_hook
        model.set_all_reduce_hook(cross_replica_gradient_sync_hook)

        # Create different data for different replica groups
        torch.manual_seed(42 + replica_id)  # Different seed for different replicas

        # Ensure we have enough samples for all steps
        # Each step consumes train_micro_batch_size_per_gpu samples
        num_steps = 3 * gradient_accumulation_steps
        train_batch_size = config_dict["train_micro_batch_size_per_gpu"]
        total_samples_needed = num_steps * train_batch_size
        data_loader = random_dataloader(
            model=model,
            total_samples=total_samples_needed + train_batch_size,  # Extra samples for safety
            hidden_dim=hidden_dim,
            device=model.device,
        )

        # Reset counters
        hook_call_count[0] = 0
        hook_tensors.clear()

        # Train for a few steps
        for step_id, batch in enumerate(data_loader):
            if step_id >= num_steps:
                break
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        # Verify hook was called
        assert hook_call_count[0] > 0, \
            f"Hook should be called for stage={zero_stage}, contiguous={contiguous_gradients}, bucket_size={reduce_bucket_size}, reduce_scatter={reduce_scatter}, overlap_comm={overlap_comm}"

        # Verify tensors were passed to hook
        assert len(hook_tensors) > 0, "Hook should receive gradient tensors"

        # Verify all tensors are valid
        non_empty_tensors = [t for t in hook_tensors if t.numel() > 0]
        assert len(non_empty_tensors) > 0, \
            f"At least some hook tensors should have elements. Got {len(hook_tensors)} total tensors, all empty."

        for tensor in non_empty_tensors:
            assert tensor is not None, "Hook tensor should not be None"
            assert tensor.device.type == get_accelerator().device_name(), \
                f"Tensor should be on {get_accelerator().device_name()}"

        # Synchronize before checking parameters
        dist.barrier()

        # Verify that all ranks have identical model parameters
        # This proves cross-replica gradient synchronization worked
        if zero_stage == 3:
            with deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=None):
                param_list = [p.data.clone() for p in model.parameters()]
        else:
            param_list = [p.data.clone() for p in model.parameters()]

        for param_idx, param in enumerate(param_list):
            gathered_params = [torch.zeros_like(param) for _ in range(self.world_size)]
            dist.all_gather(gathered_params, param)

            if rank == 0:
                for other_rank in range(1, self.world_size):
                    assert torch.allclose(gathered_params[0], gathered_params[other_rank], rtol=1e-3, atol=1e-5), \
                        f"Parameters differ between rank 0 and rank {other_rank} at param_idx={param_idx}. " \
                        f"Cross-replica gradient sync failed for stage={zero_stage}, contiguous={contiguous_gradients}, bucket_size={reduce_bucket_size}, reduce_scatter={reduce_scatter}, overlap_comm={overlap_comm}!"

        model.destroy()
