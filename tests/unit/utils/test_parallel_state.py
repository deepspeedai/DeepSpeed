# SPDX-License-Identifier: Apache-2.0
# Copyright (c) DeepSpeed Team

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
from unit.common import DistributedTest, preferred_dtype
from unit.simple_model import SimpleModel, random_dataloader
from unit.util import torch_assert_close

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
        from deepspeed.utils import parallel_state_wrappers as ps

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
        from deepspeed.utils import parallel_state_wrappers as ps

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
        from deepspeed.utils import parallel_state_wrappers as ps

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
        from deepspeed.utils import parallel_state_wrappers as ps

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
        from deepspeed.utils import parallel_state_wrappers as ps
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
        from deepspeed.utils import parallel_state_wrappers as ps

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
        from deepspeed.utils import parallel_state_wrappers as ps

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
        from deepspeed.utils import parallel_state_wrappers as ps

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
        from deepspeed.utils import parallel_state_wrappers as ps

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
        from deepspeed.utils import parallel_state_wrappers as ps

        with pytest.raises(NotImplementedError, match="does not support context_parallel_size"):
            ps.initialize_parallel_state_from_config({}, name="cp_test", context_parallel_size=2)

    def test_hierarchical_cp_not_supported(self):
        """Test that hierarchical CP raises NotImplementedError"""
        from deepspeed.utils import parallel_state_wrappers as ps

        with pytest.raises(NotImplementedError, match="does not support hierarchical_context_parallel_sizes"):
            ps.initialize_parallel_state_from_config({}, name="hcp_test", hierarchical_context_parallel_sizes=[2, 2])


class TestAllToAllGroupsWithMPU(DistributedTest):
    """Test All-to-All groups initialization with mpu"""

    world_size = 4

    def test_all_to_all_groups_with_mpu(self):
        """Test All-to-All groups work with mpu in initialize"""
        from deepspeed.utils import parallel_state_wrappers as ps

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


@pytest.mark.parametrize("zero_stage", [0, 1, 2, 3])
@pytest.mark.parametrize("sp_size", [2])
class TestUlyssesSPWithParallelState(DistributedTest):
    world_size = 4

    def test_ulysses_sp_parallel_state(self, zero_stage, sp_size):
        """Compare loss using mpu=ParallelState to test parallel_state's MPU interface."""
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from deepspeed.utils.parallel_state_wrappers import get_parallel_state_instance
        from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF, UlyssesSPDataLoaderAdapter
        from deepspeed.runtime.utils import move_to_device
        from deepspeed.accelerator import get_accelerator

        model_name_or_path = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
        sequence_parallel_size = sp_size
        micro_batch_size = 1
        num_iterations = 10

        seed = 42
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)

        base_config = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": zero_stage
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            }
        }
        dtype = preferred_dtype()
        if dtype == torch.bfloat16:
            base_config["bf16"] = {"enabled": True}
        elif dtype == torch.float16:
            base_config["fp16"] = {"enabled": True, "loss_scale": 1.0}

        def collate_fn(batch):
            input_ids, position_ids = batch[0]
            return dict(input_ids=input_ids.unsqueeze(0),
                        position_ids=position_ids.unsqueeze(0),
                        labels=input_ids.unsqueeze(0))

        input_ids = torch.randint(1, 100, (num_iterations * 2, 6))
        position_ids = torch.arange(6).unsqueeze(0).expand(num_iterations * 2, -1)
        ds = torch.utils.data.TensorDataset(input_ids, position_ids)
        dl = torch.utils.data.DataLoader(ds, batch_size=micro_batch_size, collate_fn=collate_fn, shuffle=False)
        batches_full = list(dl)

        hf_model_config = AutoConfig.from_pretrained(model_name_or_path)
        core_attn_implementation = "sdpa"
        core_attn_function = ALL_ATTENTION_FUNCTIONS[core_attn_implementation]

        # No-SP model

        model_no_sp = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model_no_sp, _, _, _ = deepspeed.initialize(config=base_config,
                                                    model=model_no_sp,
                                                    model_parameters=model_no_sp.parameters(),
                                                    mpu=None)

        losses_no_sp = []
        for i in range(num_iterations):
            batch = move_to_device(batches_full[i], model_no_sp.device)
            loss = model_no_sp(**batch).loss
            model_no_sp.backward(loss)
            model_no_sp.step()
            losses_no_sp.append(loss.detach().cpu())

        # SP model
        #
        # UlyssesSPAttentionHF.register_with_transformers() creates and returns an SP-specific parallel state object.
        # Register explicitly here before UlyssesSPAttentionHF is adapted to the generic ParallelState.

        instance_name = "sp_test_psm"
        ps = get_parallel_state_instance(instance_name)
        ps.initialize_model_parallel(sequence_parallel_size=sequence_parallel_size, order="sp-dp")
        # Get SP group info from the initialized instance
        sp_group = ps.get_sequence_parallel_group()
        sp_world_size = ps.get_sequence_parallel_world_size()
        sp_rank = ps.get_sequence_parallel_rank()

        uattn_sp = UlyssesSPAttentionHF(attn=core_attn_function,
                                        batch_size=micro_batch_size,
                                        attn_head_count=hf_model_config.num_attention_heads,
                                        attn_head_size=getattr(
                                            hf_model_config, "head_dim",
                                            hf_model_config.hidden_size // hf_model_config.num_attention_heads),
                                        kv_head_count=hf_model_config.num_key_value_heads,
                                        num_hidden_layers=hf_model_config.num_hidden_layers,
                                        process_group=sp_group,
                                        seq_length_is_variable=True,
                                        local_seq_length=None,
                                        global_seq_length=None)

        def uattn_sp_wrapper(module, query, key, value, attention_mask, *args, **kwargs):
            return uattn_sp(module, query, key, value, None, *args, **kwargs)

        for key in list(ALL_ATTENTION_FUNCTIONS.keys()):
            ALL_ATTENTION_FUNCTIONS[key] = uattn_sp_wrapper

        config_sp = dict(base_config)
        config_sp["sequence_parallel_size"] = sequence_parallel_size
        config_sp["gradient_accumulation_steps"] = 2

        model_sp = AutoModelForCausalLM.from_pretrained(model_name_or_path)

        # Pass ps instance as mpu
        model_sp, _, _, _ = deepspeed.initialize(config=config_sp,
                                                 model=model_sp,
                                                 model_parameters=model_sp.parameters(),
                                                 mpu=ps)

        ds_sp = torch.utils.data.TensorDataset(input_ids, position_ids)
        dl_sp = torch.utils.data.DataLoader(ds_sp, batch_size=micro_batch_size, collate_fn=collate_fn, shuffle=False)
        dl_sp_sharded = UlyssesSPDataLoaderAdapter(dl_sp,
                                                   sp_rank=sp_rank,
                                                   sp_group=sp_group,
                                                   sp_world_size=sp_world_size,
                                                   device=model_sp.device)

        losses_sp = []
        loss_accum = []

        for i, batch_sp in enumerate(dl_sp_sharded):
            if i >= num_iterations * 2:
                break
            batch_sp = move_to_device(batch_sp, model_sp.device)
            outputs_sp = model_sp(**batch_sp)
            shift_labels = batch_sp["shift_labels"]
            loss_sp_local = model_sp.module.loss_function(logits=outputs_sp.logits,
                                                          labels=None,
                                                          shift_labels=shift_labels,
                                                          vocab_size=model_sp.module.config.vocab_size)
            losses_per_rank = torch.distributed.nn.functional.all_gather(loss_sp_local, group=sp_group)
            good_tokens = sum((shift_labels != -100).view(-1))
            good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)
            total_loss = sum(losses_per_rank[r] * good_tokens_per_rank[r] for r in range(sp_world_size))
            loss_sp = total_loss / sum(good_tokens_per_rank)
            model_sp.backward(loss_sp)
            model_sp.step()
            loss_accum.append(loss_sp.detach().cpu())
            if len(loss_accum) == 2:
                avg_loss = torch.stack(loss_accum).mean()
                losses_sp.append(avg_loss)
                loss_accum = []

        for i in range(num_iterations):
            torch_assert_close(losses_no_sp[i], losses_sp[i], rtol=1.6e-02, atol=1e-03)
