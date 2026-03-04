# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Integration tests for using ParallelState as mpu in deepspeed.initialize()

Tests the full workflow:
1. Initialize ParallelState with parallel configurations
2. Pass the ParallelState instance as mpu parameter to deepspeed.initialize()
3. Verify DeepSpeed Engine correctly uses the parallel state
"""

import pytest
import torch
import deepspeed
import deepspeed.comm as dist
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader

DTYPE = torch.float


class TestParallelStateAsMPU(DistributedTest):
    """Test ParallelState instance as mpu parameter in deepspeed.initialize()"""

    world_size = 4

    def _get_base_config(self):
        return {"train_micro_batch_size_per_gpu": 1, "optimizer": {"type": "Adam", "params": {"lr": 0.001}}}

    def _verify_mpu_integration(self, engine, mpu, expected_tp=1, expected_pp=1):
        assert engine.mpu == mpu
        assert mpu.get_tensor_model_parallel_world_size() == expected_tp
        assert mpu.get_pipeline_model_parallel_world_size() == expected_pp

        world_size = dist.get_world_size()
        expected_dp = world_size // (expected_tp * expected_pp)
        assert mpu.get_data_parallel_world_size() == expected_dp
        assert engine._config.world_size == expected_dp

    def _train_steps(self, engine, steps=3):
        data_loader = random_dataloader(model=engine,
                                        total_samples=10,
                                        hidden_dim=16,
                                        device=engine.device,
                                        dtype=DTYPE)
        for i, batch in enumerate(data_loader):
            if i >= steps:
                break
            loss = engine(batch[0], batch[1])
            assert loss is not None
            engine.backward(loss)
            engine.step()

    def test_basic_mpu_usage(self):
        """Test basic TP with ParallelState instance as mpu"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        state = ps.get_parallel_state_instance("test_basic")
        state.initialize_model_parallel(tensor_model_parallel_size=2)

        config = self._get_base_config()
        model = SimpleModel(hidden_dim=16)

        engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                       config=config,
                                                       mpu=state,
                                                       model_parameters=model.parameters())

        self._verify_mpu_integration(engine, state, expected_tp=2)
        assert optimizer is not None
        self._train_steps(engine)

    def test_config_driven_mpu(self):
        """Test mpu initialized from config with tensor_model_parallel_size"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        parallel_config = {
            "tensor_parallel": {
                "autotp_size": 2
            },
        }

        state = ps.initialize_parallel_state_from_config(parallel_config, name="config_driven_test")

        engine_config = self._get_base_config()
        model = SimpleModel(hidden_dim=16)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               config=engine_config,
                                               mpu=state,
                                               model_parameters=model.parameters())

        self._verify_mpu_integration(engine, state, expected_tp=2)
        self._train_steps(engine)

    def test_multi_instance_mpu(self):
        """Test multiple named instances as mpu (Actor-Critic scenario)"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        actor_state = ps.get_parallel_state_instance("actor")
        actor_state.initialize_model_parallel(tensor_model_parallel_size=2)

        critic_state = ps.get_parallel_state_instance("critic")
        critic_state.initialize_model_parallel(tensor_model_parallel_size=1)

        config = self._get_base_config()

        actor_model = SimpleModel(hidden_dim=16)
        actor_engine, _, _, _ = deepspeed.initialize(model=actor_model,
                                                     config=config,
                                                     mpu=actor_state,
                                                     model_parameters=actor_model.parameters())

        critic_model = SimpleModel(hidden_dim=16)
        critic_engine, _, _, _ = deepspeed.initialize(model=critic_model,
                                                      config=config,
                                                      mpu=critic_state,
                                                      model_parameters=critic_model.parameters())

        assert actor_state.get_tensor_model_parallel_world_size() == 2
        assert actor_engine.mpu == actor_state

        assert critic_state.get_tensor_model_parallel_world_size() == 1
        assert critic_engine.mpu == critic_state

        self._train_steps(actor_engine)
        self._train_steps(critic_engine)

    def test_mpu_with_zero_stage1(self):
        """Test mpu integration with ZeRO Stage 1"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        state = ps.get_parallel_state_instance("test_zero")
        state.initialize_model_parallel(tensor_model_parallel_size=2)

        config = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 1
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.001
                }
            }
        }

        model = SimpleModel(hidden_dim=16)
        engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                       config=config,
                                                       mpu=state,
                                                       model_parameters=model.parameters())

        assert optimizer is not None
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
        assert isinstance(optimizer, DeepSpeedZeroOptimizer)

        self._verify_mpu_integration(engine, state, expected_tp=2)
        self._train_steps(engine)

    def test_deepspeed_config_uses_mpu(self):
        """Test DeepSpeedConfig correctly uses mpu for world_size calculation"""
        from deepspeed.utils import parallel_state_deepspeed as ps
        from deepspeed.runtime.config import DeepSpeedConfig

        state = ps.get_parallel_state_instance("test_config")
        state.initialize_model_parallel(tensor_model_parallel_size=2)

        config_dict = self._get_base_config()
        ds_config = DeepSpeedConfig(config_dict, mpu=state)

        expected_dp = dist.get_world_size() // 2
        assert ds_config.world_size == expected_dp
        assert ds_config.world_size == state.get_data_parallel_world_size()

    def test_mpu_without_parallelism(self):
        """Test mpu with all parallelism dimensions = 1 (no parallelism)"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        state = ps.get_parallel_state_instance("test_no_parallel")
        state.initialize_model_parallel()

        config = self._get_base_config()
        model = SimpleModel(hidden_dim=16)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               config=config,
                                               mpu=state,
                                               model_parameters=model.parameters())

        assert state.get_tensor_model_parallel_world_size() == 1
        assert state.get_pipeline_model_parallel_world_size() == 1
        assert state.get_data_parallel_world_size() == dist.get_world_size()

        self._train_steps(engine)

    def test_mpu_with_different_orders(self):
        """Test mpu with custom parallel dimension order"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        state = ps.get_parallel_state_instance("test_order")
        state.initialize_model_parallel(tensor_model_parallel_size=2,
                                        expert_model_parallel_size=2,
                                        order="tp-ep-dp-pp")

        config = self._get_base_config()
        model = SimpleModel(hidden_dim=16)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               config=config,
                                               mpu=state,
                                               model_parameters=model.parameters())

        assert state.get_tensor_model_parallel_world_size() == 2
        assert state.get_expert_model_parallel_world_size() == 2

        # EP does not reduce the regular DP world size; DP = world_size / (TP * PP)
        expected_dp = dist.get_world_size() // 2
        assert state.get_data_parallel_world_size() == expected_dp


class TestParallelStateConfigPriority(DistributedTest):
    """Test configuration priority: params > config > defaults"""

    world_size = 4

    def test_param_overrides_config(self):
        """Function parameter should override nested config value"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        config = {
            "tensor_parallel": {
                "autotp_size": 2
            },
        }

        state = ps.initialize_parallel_state_from_config(
            config,
            name="param_override_test",
            tensor_model_parallel_size=1,
        )

        assert state.get_tensor_model_parallel_world_size() == 1
        assert state.get_data_parallel_world_size() == dist.get_world_size()

    def test_config_overrides_default(self):
        """Nested config value (tensor_parallel.autotp_size) should override default"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        config = {
            "tensor_parallel": {
                "autotp_size": 2
            },
        }

        state = ps.initialize_parallel_state_from_config(config, name="config_override_test")

        assert state.get_tensor_model_parallel_world_size() == 2
        assert state.get_data_parallel_world_size() == dist.get_world_size() // 2


class TestParallelStateValidation(DistributedTest):
    """Test validation and error handling"""

    world_size = 4

    def test_context_parallel_not_supported(self):
        """Test that CP > 1 raises NotImplementedError"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        with pytest.raises(NotImplementedError, match="does not support context_parallel_size"):
            ps.initialize_parallel_state_from_config({}, name="cp_test", context_parallel_size=2)

    def test_hierarchical_cp_not_supported(self):
        """Test that hierarchical CP raises NotImplementedError"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        with pytest.raises(NotImplementedError, match="does not support hierarchical_context_parallel_sizes"):
            ps.initialize_parallel_state_from_config({}, name="hcp_test", hierarchical_context_parallel_sizes=[2, 2])


class TestAllToAllGroupsWithMPU(DistributedTest):
    """Test All-to-All groups initialization with mpu"""

    world_size = 4

    def test_all_to_all_groups_with_mpu(self):
        """Test All-to-All groups work with mpu in initialize"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        state = ps.get_parallel_state_instance("test_all_to_all")
        state.initialize_model_parallel()

        config = {"train_micro_batch_size_per_gpu": 1, "optimizer": {"type": "Adam", "params": {"lr": 0.001}}}

        model = SimpleModel(hidden_dim=16)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               config=config,
                                               mpu=state,
                                               model_parameters=model.parameters())

        all_to_all_groups = state.initialize_all_to_all_groups()
        assert isinstance(all_to_all_groups, dict)
        assert len(all_to_all_groups) > 0

        data_loader = random_dataloader(model=engine,
                                        total_samples=10,
                                        hidden_dim=16,
                                        device=engine.device,
                                        dtype=DTYPE)
        for i, batch in enumerate(data_loader):
            if i >= 3:
                break
            loss = engine(batch[0], batch[1])
            assert loss is not None
            engine.backward(loss)
            engine.step()
