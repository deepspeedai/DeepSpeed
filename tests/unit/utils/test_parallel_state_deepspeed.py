# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Integration tests for using ParallelState as mpu in deepspeed.initialize()

Tests the full workflow:
1. Initialize parallel_state_deepspeed with parallel configurations
2. Pass it as mpu parameter to deepspeed.initialize()
3. Verify DeepSpeed Engine correctly uses the parallel state
"""

import pytest
import torch
import deepspeed
import deepspeed.comm as dist
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader


class TestParallelStateAsMPU(DistributedTest):
    """Test parallel_state_deepspeed as mpu parameter in deepspeed.initialize()"""

    world_size = 8

    def _get_base_config(self):
        """Get base DeepSpeed config"""
        return {"train_batch_size": 8, "optimizer": {"type": "Adam", "params": {"lr": 0.001}}}

    def _verify_mpu_integration(self, engine, mpu, expected_tp=1, expected_pp=1, expected_sp=1):
        """Verify mpu is correctly integrated in engine"""
        # 1. Engine holds mpu reference
        assert engine.mpu == mpu

        # 2. Parallel configuration is correct
        assert mpu.get_tensor_model_parallel_world_size() == expected_tp
        assert mpu.get_pipeline_model_parallel_world_size() == expected_pp

        # 3. Data parallel world size is correctly calculated
        world_size = dist.get_world_size()
        expected_dp = world_size // (expected_tp * expected_pp * expected_sp)
        assert mpu.get_data_parallel_world_size() == expected_dp

        # 4. Config uses mpu for world_size calculation
        assert engine.config.world_size == expected_dp

        return expected_dp

    def test_basic_mpu_usage(self):
        """Test basic mpu parameter usage with TP and PP"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        # Use named instance to avoid test interference
        state = ps.get_parallel_state_instance("test_basic")
        state.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)

        config = self._get_base_config()
        model = SimpleModel(hidden_dim=16)

        # Pass parallel_state module as mpu (the module provides compatibility layer)
        with ps.set_current_parallel_state("test_basic"):
            engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                           config=config,
                                                           mpu=ps,
                                                           model_parameters=model.parameters())

        # Verify integration
        with ps.set_current_parallel_state("test_basic"):
            self._verify_mpu_integration(engine, ps, expected_tp=2, expected_pp=2)

        # Verify optimizer is created
        assert optimizer is not None

        # Test training for 5 batches
        data_loader = random_dataloader(model=engine.module, total_samples=20, hidden_dim=16, device=engine.device)
        for i, batch in enumerate(data_loader):
            if i >= 5:
                break
            loss = engine(batch[0], batch[1])
            assert loss is not None
            engine.backward(loss)
            engine.step()

    def test_config_driven_mpu(self):
        """Test mpu initialized from config with sequence_parallel_size"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        config = {
            "train_batch_size": 8,
            "sequence_parallel_size": 2,
            "order": "tp-sp-dp-pp",  # Need to specify order when using sp
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.001
                }
            }
        }

        # Initialize from config
        ps.initialize_parallel_state_from_config(config, name="config_driven_test")

        model = SimpleModel(hidden_dim=16)

        # Set current instance
        ps.set_current_parallel_state("config_driven_test")

        engine, _, _, _ = deepspeed.initialize(model=model, config=config, mpu=ps, model_parameters=model.parameters())

        # Verify SP group is created
        with ps.set_current_parallel_state("config_driven_test"):
            sp_world_size = ps.get_sequence_parallel_world_size()
            assert sp_world_size == 2

        # Verify integration
        with ps.set_current_parallel_state("config_driven_test"):
            self._verify_mpu_integration(engine, ps, expected_sp=2)

        # Test training for 5 batches
        data_loader = random_dataloader(model=engine.module, total_samples=20, hidden_dim=16, device=engine.device)
        for i, batch in enumerate(data_loader):
            if i >= 5:
                break
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()

    def test_multi_instance_mpu(self):
        """Test multiple named instances as mpu (Actor-Critic scenario)"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        # Initialize Actor with TP=2
        actor_state = ps.get_parallel_state_instance("actor")
        actor_state.initialize_model_parallel(tensor_model_parallel_size=2)

        # Initialize Critic with TP=1 (no parallelism)
        critic_state = ps.get_parallel_state_instance("critic")
        critic_state.initialize_model_parallel(tensor_model_parallel_size=1)

        config = self._get_base_config()

        # Create Actor engine
        actor_model = SimpleModel(hidden_dim=16)
        with ps.set_current_parallel_state("actor"):
            actor_engine, _, _, _ = deepspeed.initialize(model=actor_model,
                                                         config=config,
                                                         mpu=ps,
                                                         model_parameters=actor_model.parameters())

        # Create Critic engine
        critic_model = SimpleModel(hidden_dim=16)
        with ps.set_current_parallel_state("critic"):
            critic_engine, _, _, _ = deepspeed.initialize(model=critic_model,
                                                          config=config,
                                                          mpu=ps,
                                                          model_parameters=critic_model.parameters())

        # Verify Actor uses TP=2
        with ps.set_current_parallel_state("actor"):
            assert ps.get_tensor_model_parallel_world_size() == 2
            assert actor_engine.mpu == ps

        # Verify Critic uses TP=1
        with ps.set_current_parallel_state("critic"):
            assert ps.get_tensor_model_parallel_world_size() == 1
            assert critic_engine.mpu == ps

        # Test training for 5 batches on both engines
        actor_loader = random_dataloader(model=actor_engine.module, total_samples=20, hidden_dim=16, device=actor_engine.device)
        critic_loader = random_dataloader(model=critic_engine.module, total_samples=20, hidden_dim=16, device=critic_engine.device)
        for i, (actor_batch, critic_batch) in enumerate(zip(actor_loader, critic_loader)):
            if i >= 5:
                break
            actor_loss = actor_engine(actor_batch[0], actor_batch[1])
            assert actor_loss is not None
            actor_engine.backward(actor_loss)
            actor_engine.step()

            critic_loss = critic_engine(critic_batch[0], critic_batch[1])
            assert critic_loss is not None
            critic_engine.backward(critic_loss)
            critic_engine.step()

    def test_mpu_with_zero_stage1(self):
        """Test mpu integration with ZeRO Stage 1"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        # Use named instance to avoid test interference
        state = ps.get_parallel_state_instance("test_zero")
        state.initialize_model_parallel(tensor_model_parallel_size=2)

        config = {
            "train_batch_size": 8,
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

        with ps.set_current_parallel_state("test_zero"):
            engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                           config=config,
                                                           mpu=ps,
                                                           model_parameters=model.parameters())

        # Verify ZeRO optimizer is created
        assert optimizer is not None
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
        assert isinstance(optimizer, DeepSpeedZeroOptimizer)

        # Verify mpu integration
        with ps.set_current_parallel_state("test_zero"):
            self._verify_mpu_integration(engine, ps, expected_tp=2)

        # Verify optimizer uses correct DP group
        assert optimizer.mpu == ps

        # Test training for 5 batches
        data_loader = random_dataloader(model=engine.module, total_samples=20, hidden_dim=16, device=engine.device)
        for i, batch in enumerate(data_loader):
            if i >= 5:
                break
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()

    def test_deepspeed_config_uses_mpu(self):
        """Test DeepSpeedConfig correctly uses mpu for world_size calculation"""
        from deepspeed.utils import parallel_state_deepspeed as ps
        from deepspeed.runtime.config import DeepSpeedConfig

        # Use named instance to avoid test interference
        state = ps.get_parallel_state_instance("test_config")
        state.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)

        config_dict = self._get_base_config()

        # Create DeepSpeedConfig with mpu
        with ps.set_current_parallel_state("test_config"):
            ds_config = DeepSpeedConfig(config_dict, mpu=ps)

        # Verify world_size calculation uses mpu
        expected_dp = dist.get_world_size() // (2 * 2)
        assert ds_config.world_size == expected_dp

        # Verify it matches mpu's calculation
        with ps.set_current_parallel_state("test_config"):
            assert ds_config.world_size == ps.get_data_parallel_world_size()

    def test_mpu_without_parallelism(self):
        """Test mpu with all parallelism dimensions = 1 (no parallelism)"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        # Use named instance to avoid test interference
        state = ps.get_parallel_state_instance("test_no_parallel")
        state.initialize_model_parallel()

        config = self._get_base_config()
        model = SimpleModel(hidden_dim=16)

        with ps.set_current_parallel_state("test_no_parallel"):
            engine, _, _, _ = deepspeed.initialize(model=model,
                                                   config=config,
                                                   mpu=ps,
                                                   model_parameters=model.parameters())

        # Verify all dimensions are 1
        with ps.set_current_parallel_state("test_no_parallel"):
            assert ps.get_tensor_model_parallel_world_size() == 1
            assert ps.get_pipeline_model_parallel_world_size() == 1

            # DP should equal world_size
            assert ps.get_data_parallel_world_size() == dist.get_world_size()

        # Test training for 5 batches
        data_loader = random_dataloader(model=engine.module, total_samples=20, hidden_dim=16, device=engine.device)
        for i, batch in enumerate(data_loader):
            if i >= 5:
                break
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()

    def test_mpu_with_different_orders(self):
        """Test mpu with different parallel dimension orders"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        # Use named instance to avoid test interference
        state = ps.get_parallel_state_instance("test_order")
        state.initialize_model_parallel(tensor_model_parallel_size=2,
                                        expert_model_parallel_size=2,
                                        order="tp-ep-dp-pp")

        config = self._get_base_config()
        model = SimpleModel(hidden_dim=16)

        with ps.set_current_parallel_state("test_order"):
            engine, _, _, _ = deepspeed.initialize(model=model,
                                                   config=config,
                                                   mpu=ps,
                                                   model_parameters=model.parameters())

        # Verify parallel configuration
        with ps.set_current_parallel_state("test_order"):
            assert ps.get_tensor_model_parallel_world_size() == 2
            assert ps.get_expert_model_parallel_world_size() == 2

            # Verify DP world_size: world_size / (tp * ep)
            expected_dp = dist.get_world_size() // (2 * 2)
            assert ps.get_data_parallel_world_size() == expected_dp


class TestParallelStateConfigPriority(DistributedTest):
    """Test configuration priority: params > config > defaults"""

    world_size = 4

    def test_param_overrides_config(self):
        """Function parameter should override config value"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        config = {
            "train_batch_size": 4,
            "sequence_parallel_size": 2,  # Config says 2
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.001
                }
            }
        }

        # Override with param: sp=1
        ps.initialize_parallel_state_from_config(
            config,
            name="param_override_test",
            sequence_parallel_size=1  # Parameter overrides config
        )

        model = SimpleModel(hidden_dim=16)

        with ps.set_current_parallel_state("param_override_test"):
            engine, _, _, _ = deepspeed.initialize(model=model,
                                                   config=config,
                                                   mpu=ps,
                                                   model_parameters=model.parameters())

        # With sp=1, SP group should not have special effect
        assert engine is not None
        assert engine.mpu == ps

    def test_config_overrides_default(self):
        """Config value should override default value"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        config = {
            "train_batch_size": 4,
            "sequence_parallel_size": 2,  # Override default (1)
            "order": "tp-sp-dp-pp",  # Need to specify order when using sp
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.001
                }
            }
        }

        # Don't pass sequence_parallel_size parameter
        ps.initialize_parallel_state_from_config(config, name="config_override_test")

        model = SimpleModel(hidden_dim=16)

        with ps.set_current_parallel_state("config_override_test"):
            engine, _, _, _ = deepspeed.initialize(model=model,
                                                   config=config,
                                                   mpu=ps,
                                                   model_parameters=model.parameters())

        # Verify SP is configured from config
        # Since sp_size = 2, SP group should be initialized
        with ps.set_current_parallel_state("config_override_test"):
            sp_world_size = ps.get_sequence_parallel_world_size()
            assert sp_world_size == 2


class TestParallelStateValidation(DistributedTest):
    """Test validation and error handling"""

    world_size = 4

    def test_context_parallel_not_supported(self):
        """Test that CP > 1 raises NotImplementedError"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        # CP > 1 should raise error via initialize_parallel_state_from_config
        with pytest.raises(NotImplementedError, match="does not support context_parallel_size"):
            ps.initialize_parallel_state_from_config({"context_parallel_size": 2}, name="cp_test")

    def test_hierarchical_cp_not_supported(self):
        """Test that hierarchical CP raises NotImplementedError"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        with pytest.raises(NotImplementedError, match="does not support hierarchical_context_parallel_sizes"):
            ps.initialize_parallel_state_from_config({"hierarchical_context_parallel_sizes": [2, 2]}, name="hcp_test")


class TestAllToAllGroupsWithMPU(DistributedTest):
    """Test All-to-All groups initialization with mpu"""

    world_size = 8

    def test_all_to_all_groups_with_mpu(self):
        """Test All-to-All groups work with mpu in initialize"""
        from deepspeed.utils import parallel_state_deepspeed as ps

        # Use named instance to avoid test interference
        state = ps.get_parallel_state_instance("test_all_to_all")
        state.initialize_model_parallel()

        config = {"train_batch_size": 8, "optimizer": {"type": "Adam", "params": {"lr": 0.001}}}

        model = SimpleModel(hidden_dim=16)

        with ps.set_current_parallel_state("test_all_to_all"):
            engine, _, _, _ = deepspeed.initialize(model=model,
                                                   config=config,
                                                   mpu=ps,
                                                   model_parameters=model.parameters())

        # Initialize All-to-All groups
        with ps.set_current_parallel_state("test_all_to_all"):
            all_to_all_groups = ps.initialize_all_to_all_groups()

        # Verify groups are created
        assert isinstance(all_to_all_groups, dict)
        assert len(all_to_all_groups) > 0

        # Test backward compatibility interface
        with ps.set_current_parallel_state("test_all_to_all"):
            compat_groups = ps._get_local_all_to_all_group()
            assert compat_groups == all_to_all_groups

        # Test training for 5 batches
        data_loader = random_dataloader(model=engine.module, total_samples=20, hidden_dim=16, device=engine.device)
        for i, batch in enumerate(data_loader):
            if i >= 5:
                break
            loss = engine(batch[0], batch[1])
            assert loss is not None
            engine.backward(loss)
            engine.step()
