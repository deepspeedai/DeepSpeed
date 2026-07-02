# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
UlyssesPlus: UlyssesSPHF tests
"""

import deepspeed.runtime.sequence_parallel.ulysses_sp as ulysses_sp
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF, UlyssesSPDataLoaderAdapter
from deepspeed.runtime.utils import move_to_device
from deepspeed.utils import groups
from deepspeed.utils import safe_get_full_grad
from torch import tensor
from transformers import AutoModelForCausalLM
from unit.common import DistributedTest, preferred_dtype
from unit.util import torch_assert_equal, torch_assert_close, torch_assert_dicts_of_tensors_equal
import deepspeed
import deepspeed.comm as dist
import pytest
import sys
import torch
import types


def get_grad(param, zero_stage):
    return safe_get_full_grad(param)
    # z1 now has contiguous_gradients enabled by default so `param.grad is None` even under z1
    # if zero_stage == 1:
    #     return param.grad
    # else:
    #     return safe_get_full_grad(param)


class TestLinearAttentionCPHelpers:

    def test_position_ids_to_packed_cu_seqlens_single_sequence(self):
        position_ids = torch.tensor([[0, 1, 2, 3]])

        cu_seqlens = ulysses_sp._position_ids_to_packed_cu_seqlens(position_ids)

        torch_assert_equal(cu_seqlens, torch.tensor([0, 4], dtype=torch.long))

    def test_position_ids_to_packed_cu_seqlens_packed_sequence(self):
        position_ids = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2]])

        cu_seqlens = ulysses_sp._position_ids_to_packed_cu_seqlens(position_ids)

        torch_assert_equal(cu_seqlens, torch.tensor([0, 3, 5, 8], dtype=torch.long))

    def test_modeling_module_candidates_strip_text_suffix(self):
        cfg = types.SimpleNamespace(model_type="qwen3_5_text")

        candidates = list(ulysses_sp._modeling_module_candidates(cfg, cfg))

        assert "transformers.models.qwen3_5_text.modeling_qwen3_5_text" in candidates
        assert "transformers.models.qwen3_5.modeling_qwen3_5" in candidates

    def test_linear_attention_cp_noops_for_non_linear_config(self, monkeypatch):

        def fail_if_called(_name):
            raise AssertionError("FLA package version should not be probed for non-linear configs")

        monkeypatch.setattr(ulysses_sp.importlib_metadata, "version", fail_if_called)
        cfg = types.SimpleNamespace(model_type="llama", layer_types=["full_attention"])

        assert ulysses_sp._register_linear_attention_cp(cfg, cfg) == 0

    def test_linear_attention_cp_version_gate(self, monkeypatch):

        def fake_version(_name):
            return "0.4.1"

        monkeypatch.setattr(ulysses_sp.importlib_metadata, "version", fake_version)

        with pytest.raises(ImportError, match=">= 0.4.2"):
            ulysses_sp._load_linear_attention_cp_ops()

    def test_gated_delta_state_layout_kwargs_match_fla_version_signatures(self):

        def old_chunk_gated_delta_rule(transpose_state_layout=False, **kwargs):
            return transpose_state_layout, kwargs

        def new_chunk_gated_delta_rule(state_v_first=False, **kwargs):
            return state_v_first, kwargs

        assert ulysses_sp._gated_delta_state_layout_kwargs(old_chunk_gated_delta_rule) == {
            "transpose_state_layout": True
        }
        assert ulysses_sp._gated_delta_state_layout_kwargs(new_chunk_gated_delta_rule) == {"state_v_first": True}

    def test_linear_attention_cp_ignores_transformers_forward_flags(self, monkeypatch):

        class FakeNorm(torch.nn.Module):

            def forward(self, hidden_states, gate):
                return hidden_states

        class FakeGatedDeltaNet(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.in_proj_qkv = torch.nn.Linear(4, 12, bias=False)
                self.in_proj_z = torch.nn.Linear(4, 4, bias=False)
                self.in_proj_b = torch.nn.Linear(4, 1, bias=False)
                self.in_proj_a = torch.nn.Linear(4, 1, bias=False)
                self.conv1d = torch.nn.Conv1d(12, 12, kernel_size=1, groups=12)
                self.activation = "silu"
                self.num_v_heads = 1
                self.num_k_heads = 1
                self.head_k_dim = 4
                self.head_v_dim = 4
                self.conv_kernel_size = 1
                self.A_log = torch.nn.Parameter(torch.zeros(1))
                self.dt_bias = torch.nn.Parameter(torch.zeros(1))
                self.norm = FakeNorm()
                self.out_proj = torch.nn.Linear(4, 4, bias=False)

            def chunk_gated_delta_rule(self, query, key, value, g=None, beta=None, cp_context=None, **kwargs):
                return value, None

        def fake_causal_conv1d(x, weight=None, bias=None, activation=None, cp_context=None):
            return x

        monkeypatch.setattr(ulysses_sp, "_get_sequence_parallel_info", lambda: (None, 2, 0))
        monkeypatch.setattr(ulysses_sp, "_load_linear_attention_cp_ops", lambda: (None, fake_causal_conv1d))
        monkeypatch.setattr(ulysses_sp, "_build_linear_attention_cp_context", lambda **kwargs: object())

        layer = FakeGatedDeltaNet()
        hidden_states = torch.randn(1, 2, 4)

        output = ulysses_sp._gated_delta_cp_forward(
            layer,
            hidden_states,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )

        assert output.shape == hidden_states.shape

    def test_linear_attention_cp_patches_gdn_like_class(self, monkeypatch):
        modeling_module_name = "transformers.models.fake_linear.modeling_fake_linear"
        modeling_module = types.ModuleType(modeling_module_name)

        class FakeGatedDeltaNet(torch.nn.Module):

            def forward(self, hidden_states, cache_params=None, attention_mask=None, position_ids=None):
                return hidden_states

        modeling_module.FakeGatedDeltaNet = FakeGatedDeltaNet

        fla_module = types.ModuleType("fla")
        fla_ops_module = types.ModuleType("fla.ops")
        fla_cp_module = types.ModuleType("fla.ops.cp")
        fla_modules_module = types.ModuleType("fla.modules")
        fla_conv_module = types.ModuleType("fla.modules.conv")

        class FakeFLACPContext:
            pass

        def fake_build_cp_context(*args, **kwargs):
            return FakeFLACPContext()

        def fake_causal_conv1d(x, weight=None, bias=None, activation=None, cp_context=None):
            return x, None

        fla_cp_module.FLACPContext = FakeFLACPContext
        fla_cp_module.build_cp_context = fake_build_cp_context
        fla_conv_module.causal_conv1d = fake_causal_conv1d

        fla_module.ops = fla_ops_module
        fla_ops_module.cp = fla_cp_module
        fla_module.modules = fla_modules_module
        fla_modules_module.conv = fla_conv_module

        for name, module in (
            ("fla", fla_module),
            ("fla.ops", fla_ops_module),
            ("fla.ops.cp", fla_cp_module),
            ("fla.modules", fla_modules_module),
            ("fla.modules.conv", fla_conv_module),
            (modeling_module_name, modeling_module),
        ):
            monkeypatch.setitem(sys.modules, name, module)

        monkeypatch.setattr(ulysses_sp.importlib_metadata, "version", lambda _name: "0.5.0")

        cfg = types.SimpleNamespace(model_type="fake_linear", layer_types=["linear_attention"])
        original_forward = FakeGatedDeltaNet.forward

        try:
            installed = ulysses_sp._register_linear_attention_cp(cfg, cfg)
            installed_again = ulysses_sp._register_linear_attention_cp(cfg, cfg)

            assert installed == 1
            assert installed_again == 0
            assert FakeGatedDeltaNet.forward is ulysses_sp._gated_delta_cp_forward
            assert ulysses_sp._LINEAR_ATTENTION_CP_ORIGINAL_FORWARDS[FakeGatedDeltaNet] is original_forward
        finally:
            FakeGatedDeltaNet.forward = original_forward
            ulysses_sp._LINEAR_ATTENTION_CP_ORIGINAL_FORWARDS.pop(FakeGatedDeltaNet, None)


@pytest.mark.parametrize("zero_stage", [2, 3])
class TestUlyssesSPHF(DistributedTest):
    world_size = 2

    def test_ulysses_sp_hf(self, zero_stage):
        core_attn_implementation = "sdpa"
        model_name_or_path = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
        #model_name_or_path = 'Felladrin/Llama-160M-Chat-v1'
        #model_name_or_path = 'Felladrin/Llama-160M-Chat-v1'
        seq_length = 64
        sequence_parallel_size = self.world_size
        micro_batch_size = 1

        rank = dist.get_rank()

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": zero_stage,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "sequence_parallel_size": sequence_parallel_size,
        }

        dtype = preferred_dtype()
        if dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}
        elif dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "loss_scale": 1.0}

        # Part 1. Baseline: Setup
        def collate_fn(batch):
            input_ids, position_ids = batch[0]
            #print(f"{batch=}")
            return dict(input_ids=input_ids.unsqueeze(0),
                        position_ids=position_ids.unsqueeze(0),
                        labels=input_ids.unsqueeze(0))

        input_ids = tensor([[1, 10, 10, 10, 2, 2], [1, 20, 20, 20, 2, 2]], )
        position_ids = tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
        ds = torch.utils.data.TensorDataset(input_ids, position_ids)

        # 1. Baseline: DataLoader calibration
        dl_a = torch.utils.data.DataLoader(ds, batch_size=micro_batch_size, collate_fn=collate_fn)
        batch_a = next(iter(dl_a))
        #print(f"{rank=} {batch_a=}")
        expected_batch_a = {
            'input_ids': tensor([[1, 10, 10, 10, 2, 2]]),
            'position_ids': tensor([[0, 1, 2, 3, 4, 5]]),
            'labels': tensor([[1, 10, 10, 10, 2, 2]])
        }
        torch_assert_dicts_of_tensors_equal(batch_a, expected_batch_a)

        # 2. Baseline: Attention
        model_a = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model_a, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_a,
                                                model_parameters=model_a.parameters(),
                                                mpu=None)
        batch_a = move_to_device(batch_a, model_a.device)
        loss_a = model_a(**batch_a).loss
        model_a.backward(loss_a)
        #print(f"{loss_a=}")

        grad_a = get_grad(model_a.module.model.layers[0].self_attn.q_proj.weight, zero_stage)
        assert grad_a is not None
        #print(f"{grad_a}")

        # Part 2. Ulysses: Setup
        mpu = UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=model_name_or_path,
            core_attn_implementation=core_attn_implementation,
            sequence_parallel_size=sequence_parallel_size,
            micro_batch_size=micro_batch_size,
            seq_length=seq_length,
            seq_length_is_variable=True,
        )

        model_b = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                       attn_implementation=core_attn_implementation)
        model_b, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_b,
                                                model_parameters=model_b.parameters(),
                                                mpu=mpu)

        # 3. Ulysses: UlyssesSPDataLoaderAdapter test
        sp_group = groups._get_sequence_parallel_group()
        sp_world_size = groups._get_sequence_parallel_world_size()
        sp_rank = groups._get_sequence_parallel_rank()
        dl_a = torch.utils.data.DataLoader(ds, batch_size=micro_batch_size, collate_fn=collate_fn)
        dl_b = UlyssesSPDataLoaderAdapter(
            dl_a,
            sp_rank=sp_rank,
            sp_group=sp_group,
            sp_world_size=sp_world_size,
            device=model_b.device,
        )
        batch_b = next(iter(dl_b))

        expected_batch_b = [
            {
                'input_ids': tensor([[1, 10, 10]]),
                'position_ids': tensor([[0, 1, 2]]),
                'shift_labels': tensor([[10, 10, 10]]),
            },
            {
                'input_ids': tensor([[10, 2, 2]]),
                'position_ids': tensor([[3, 4, 5]]),
                'shift_labels': tensor([[2, 2, -100]]),
            },
        ]

        # here we expect each sample to be sharded in half, rank0 getting the first half and rank1 the other half
        #print(f"{sp_rank=} {batch_b=}")
        torch_assert_dicts_of_tensors_equal(batch_b, expected_batch_b[sp_rank])

        # 4. UlyssesSPAttentionHF test
        batch_b = move_to_device(batch_b, model_b.device)
        outputs = model_b(**batch_b)
        # HF doesn't calculate loss with shift_labels yet and requires us to do it manually (liger does that)
        shift_labels = batch_b["shift_labels"]
        loss_b = model_b.module.loss_function(
            logits=outputs.logits,
            labels=None,
            shift_labels=shift_labels,
            vocab_size=model_b.module.config.vocab_size,
        )
        # print(f"{sp_rank=} {loss_b=}")

        # differentiable weighted per-shard-loss aggregation across ranks
        losses_per_rank = torch.distributed.nn.functional.all_gather(loss_b, group=sp_group)
        good_tokens = sum((shift_labels != -100).view(-1))
        good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)
        total_loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(sp_world_size))
        total_good_tokens = sum(good_tokens_per_rank)
        loss_b = total_loss / total_good_tokens
        # print(f"{sp_rank=} gathered {loss_b=}")
        model_b.backward(loss_b)

        grad_b = get_grad(model_b.module.model.layers[0].self_attn.q_proj.weight, zero_stage)
        assert grad_b is not None
        #print(f"{grad_b}")

        # compare loss of A (non-Ulysses Attention) and B (Ulyssses Attention)
        torch_assert_equal(loss_a, loss_b)

        # - we are feeding the exact same sample to each rank of A
        # - for B we feed half the sample to each rank, but in total it's the same sample as each rank of A sees
        # thus we expect very similar grads (but not exact)
        if zero_stage in [1, 2]:
            # possibly some issue with z1/z2 as it requires higher tolerance than z3?
            torch_assert_close(grad_a, grad_b, rtol=1.6e-02, atol=1e-03)
        else:
            torch_assert_close(grad_a, grad_b)


class TestUlyssesSPHFPEFT(DistributedTest):
    world_size = 2

    def test_ulysses_sp_hf_with_peft_model(self):
        """Test that UlyssesSPAttentionHF.register_with_transformers works with PEFT models.

        PEFT models don't inherit from transformers.PreTrainedModel but have a config attribute.
        This test verifies the duck-typing check for the config attribute works correctly.
        """
        model_name_or_path = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
        seq_length = 64
        sequence_parallel_size = self.world_size
        micro_batch_size = 1

        # Create a mock PEFT model object that has config but doesn't inherit from PreTrainedModel
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_name_or_path)

        class MockPEFTModel:
            """Mock PEFT model that simulates PeftModel behavior"""

            def __init__(self, config):
                self.config = config

        mock_peft_model = MockPEFTModel(hf_config)

        # Test that register_with_transformers works with PEFT-like model object
        # This should not crash and should use the config attribute via duck-typing
        mpu = UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=mock_peft_model,
            core_attn_implementation="sdpa",
            sequence_parallel_size=sequence_parallel_size,
            micro_batch_size=micro_batch_size,
            seq_length=seq_length,
            seq_length_is_variable=True,
        )

        # Verify mpu is created successfully
        assert mpu is not None

        # Verify that the sequence parallel groups are initialized
        sp_group = mpu.get_sequence_parallel_group()
        assert sp_group is not None
        sp_world_size = mpu.get_sequence_parallel_world_size()
        assert sp_world_size == sequence_parallel_size


class TestUlyssesSPHFDisableInEval(DistributedTest):
    world_size = 2

    def test_disable_in_eval(self):
        """Test that disable_in_eval parameter controls SP behavior during evaluation.

        When disable_in_eval=True, SP operations should be bypassed during eval mode,
        allowing the user to pass full (non-sharded) sequences directly.
        This should produce the same output as a model without SP registered.
        """
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        model_name_or_path = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
        seq_length = 64
        sequence_parallel_size = self.world_size
        micro_batch_size = 1

        dtype = preferred_dtype()
        rank = dist.get_rank()

        # Full sequence input (not sharded) - this is what users would pass during eval
        # when they want to bypass SP and process sequences independently per rank
        input_ids = tensor([[1, 10, 10, 10, 2, 2]], device=f"cuda:{rank}")
        position_ids = tensor([[0, 1, 2, 3, 4, 5]], device=f"cuda:{rank}")

        # 1. Baseline: model without SP, processing full sequence
        model_baseline = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
        model_baseline = model_baseline.to(f"cuda:{rank}")
        model_baseline.eval()

        # Save original attention function for comparison
        original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

        with torch.no_grad():
            outputs_baseline = model_baseline(input_ids=input_ids, position_ids=position_ids)
            logits_baseline = outputs_baseline.logits.clone()

        del model_baseline

        # 2. Model with SP registered but disable_in_eval=True
        # In eval mode, SP is bypassed so full sequence can be passed directly
        mpu = UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=model_name_or_path,
            core_attn_implementation="sdpa",
            sequence_parallel_size=sequence_parallel_size,
            micro_batch_size=micro_batch_size,
            seq_length=seq_length,
            seq_length_is_variable=True,
            disable_in_eval=True,
        )

        # Verify that register_with_transformers actually registered the wrapper
        assert mpu is not None, "mpu should not be None when sequence_parallel_size > 1"
        assert ALL_ATTENTION_FUNCTIONS["sdpa"] is not original_sdpa, \
            "register_with_transformers should have replaced the attention function"

        model_sp = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
        model_sp = model_sp.to(f"cuda:{rank}")
        model_sp.eval()

        with torch.no_grad():
            outputs_sp = model_sp(input_ids=input_ids, position_ids=position_ids)
            logits_sp = outputs_sp.logits.clone()

        # Verify: with disable_in_eval=True, full sequence input should produce
        # the same output as baseline (SP is bypassed)
        torch_assert_equal(logits_baseline, logits_sp)


class TestUlyssesSPHFHubKernel(DistributedTest):
    world_size = 2

    def test_register_hub_kernel_attn(self, monkeypatch):
        """Test hub-kernel attention strings are registered before validation.

        This verifies that DeepSpeed can accept kernel-based attention implementations
        by triggering transformers' lazy registration path prior to checking
        ALL_ATTENTION_FUNCTIONS.
        """
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        model_name_or_path = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
        seq_length = 64
        sequence_parallel_size = self.world_size
        micro_batch_size = 1
        hub_attn_implementation = 'kernels-community/flash-attn2'

        called_with = []
        had_hub_key_before = hub_attn_implementation in ALL_ATTENTION_FUNCTIONS
        original_sdpa = ALL_ATTENTION_FUNCTIONS['sdpa']

        def _mock_lazy_import_flash_attention(implementation, attention_wrapper=None, allow_all_kernels=False):
            called_with.append(implementation)
            if implementation == hub_attn_implementation and implementation not in ALL_ATTENTION_FUNCTIONS:
                # Mimic transformers hub-kernel registration behavior.
                ALL_ATTENTION_FUNCTIONS.register(implementation, ALL_ATTENTION_FUNCTIONS['sdpa'])
            return (None, None, None, None), None

        monkeypatch.setattr(
            'transformers.modeling_flash_attention_utils.lazy_import_flash_attention',
            _mock_lazy_import_flash_attention,
        )

        try:
            mpu = UlyssesSPAttentionHF.register_with_transformers(
                model_name_or_path=model_name_or_path,
                core_attn_implementation=hub_attn_implementation,
                sequence_parallel_size=sequence_parallel_size,
                micro_batch_size=micro_batch_size,
                seq_length=seq_length,
                seq_length_is_variable=True,
            )
            assert ALL_ATTENTION_FUNCTIONS['sdpa'] is original_sdpa
            assert ALL_ATTENTION_FUNCTIONS[hub_attn_implementation] is not original_sdpa
        finally:
            if not had_hub_key_before and hub_attn_implementation in ALL_ATTENTION_FUNCTIONS:
                ALL_ATTENTION_FUNCTIONS.pop(hub_attn_implementation, None)

        assert mpu is not None
        assert called_with == [hub_attn_implementation]


class TestUlyssesSPHFAttnImplMismatch(DistributedTest):
    world_size = 2

    def test_register_with_mismatched_attn_impl_raises(self):
        from transformers import AutoConfig

        model_name_or_path = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
        seq_length = 64
        sequence_parallel_size = self.world_size
        micro_batch_size = 1

        hf_config = AutoConfig.from_pretrained(model_name_or_path)
        hf_config._attn_implementation = "sdpa"

        class MockModel:
            """Mock model wrapper exposing a transformers config attribute."""

            def __init__(self, config):
                self.config = config

        with pytest.raises(ValueError, match='does not match model config attn_implementation'):
            UlyssesSPAttentionHF.register_with_transformers(
                model_name_or_path=MockModel(hf_config),
                core_attn_implementation='flash_attention_2',
                sequence_parallel_size=sequence_parallel_size,
                micro_batch_size=micro_batch_size,
                seq_length=seq_length,
                seq_length_is_variable=True,
            )


@pytest.mark.parametrize("zero_stage", [2, 3])
class TestUlyssesSPHFFlexAttention(DistributedTest):
    """Separate class for flex_attention tests — requires non_daemonic_procs
    because torch.compile (used by flex_attention) creates unpicklable objects
    that break the default multiprocessing.Pool exception handling."""
    world_size = 2
    non_daemonic_procs = True

    def test_ulysses_sp_hf_flex_attention(self, zero_stage):
        core_attn_implementation = "flex_attention"
        # flex_attention's compiled kernel requires head_dim >= 16.
        # tiny-random-LlamaForCausalLM has head_dim=4, so we create a tiny model with head_dim=16.
        from transformers import LlamaConfig
        model_config = LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=32,
            max_position_embeddings=64,
        )  # head_dim = 32/2 = 16
        seq_length = 64
        sequence_parallel_size = self.world_size
        micro_batch_size = 1

        rank = dist.get_rank()

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": zero_stage,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "sequence_parallel_size": sequence_parallel_size,
        }

        dtype = preferred_dtype()
        if dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}
        elif dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "loss_scale": 1.0}

        # Part 1. Baseline: Setup
        def collate_fn(batch):
            input_ids, position_ids = batch[0]
            #print(f"{batch=}")
            return dict(input_ids=input_ids.unsqueeze(0),
                        position_ids=position_ids.unsqueeze(0),
                        labels=input_ids.unsqueeze(0))

        input_ids = tensor([[1, 10, 10, 10, 2, 2], [1, 20, 20, 20, 2, 2]])
        position_ids = tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
        ds = torch.utils.data.TensorDataset(input_ids, position_ids)

        # 1. Baseline: DataLoader calibration
        dl_a = torch.utils.data.DataLoader(ds, batch_size=micro_batch_size, collate_fn=collate_fn)
        batch_a = next(iter(dl_a))
        #print(f"{rank=} {batch_a=}")
        expected_batch_a = {
            'input_ids': tensor([[1, 10, 10, 10, 2, 2]]),
            'position_ids': tensor([[0, 1, 2, 3, 4, 5]]),
            'labels': tensor([[1, 10, 10, 10, 2, 2]])
        }
        torch_assert_dicts_of_tensors_equal(batch_a, expected_batch_a)

        # 2. Baseline: Attention
        torch.manual_seed(42)
        model_a = AutoModelForCausalLM.from_config(model_config, attn_implementation=core_attn_implementation)
        model_a, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_a,
                                                model_parameters=model_a.parameters(),
                                                mpu=None)
        batch_a = move_to_device(batch_a, model_a.device)
        loss_a = model_a(**batch_a).loss
        model_a.backward(loss_a)
        #print(f"{loss_a=}")

        grad_a = get_grad(model_a.module.model.layers[0].self_attn.q_proj.weight, zero_stage)
        assert grad_a is not None
        #print(f"{grad_a}")

        # Part 2. Ulysses: Setup
        mpu = UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=model_a.module,
            core_attn_implementation=core_attn_implementation,
            sequence_parallel_size=sequence_parallel_size,
            micro_batch_size=micro_batch_size,
            seq_length=seq_length,
            seq_length_is_variable=True,
        )

        torch.manual_seed(42)
        model_b = AutoModelForCausalLM.from_config(model_config, attn_implementation=core_attn_implementation)
        model_b, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_b,
                                                model_parameters=model_b.parameters(),
                                                mpu=mpu)

        # 3. Ulysses: UlyssesSPDataLoaderAdapter test
        sp_group = groups._get_sequence_parallel_group()
        sp_world_size = groups._get_sequence_parallel_world_size()
        sp_rank = groups._get_sequence_parallel_rank()
        dl_a = torch.utils.data.DataLoader(ds, batch_size=micro_batch_size, collate_fn=collate_fn)
        dl_b = UlyssesSPDataLoaderAdapter(
            dl_a,
            sp_rank=sp_rank,
            sp_group=sp_group,
            sp_world_size=sp_world_size,
            device=model_b.device,
        )
        batch_b = next(iter(dl_b))

        expected_batch_b = [
            {
                'input_ids': tensor([[1, 10, 10]]),
                'position_ids': tensor([[0, 1, 2]]),
                'shift_labels': tensor([[10, 10, 10]]),
            },
            {
                'input_ids': tensor([[10, 2, 2]]),
                'position_ids': tensor([[3, 4, 5]]),
                'shift_labels': tensor([[2, 2, -100]]),
            },
        ]

        # here we expect each sample to be sharded in half, rank0 getting the first half and rank1 the other half
        #print(f"{sp_rank=} {batch_b=}")
        torch_assert_dicts_of_tensors_equal(batch_b, expected_batch_b[sp_rank])

        # 4. UlyssesSPAttentionHF test
        batch_b = move_to_device(batch_b, model_b.device)
        outputs = model_b(**batch_b)
        # HF doesn't calculate loss with shift_labels yet and requires us to do it manually (liger does that)
        shift_labels = batch_b["shift_labels"]
        loss_b = model_b.module.loss_function(
            logits=outputs.logits,
            labels=None,
            shift_labels=shift_labels,
            vocab_size=model_b.module.config.vocab_size,
        )
        # print(f"{sp_rank=} {loss_b=}")

        # differentiable weighted per-shard-loss aggregation across ranks
        losses_per_rank = torch.distributed.nn.functional.all_gather(loss_b, group=sp_group)
        good_tokens = sum((shift_labels != -100).view(-1))
        good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)
        total_loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(sp_world_size))
        total_good_tokens = sum(good_tokens_per_rank)
        loss_b = total_loss / total_good_tokens
        # print(f"{sp_rank=} gathered {loss_b=}")
        model_b.backward(loss_b)

        grad_b = get_grad(model_b.module.model.layers[0].self_attn.q_proj.weight, zero_stage)
        assert grad_b is not None
        #print(f"{grad_b}")

        # compare loss of A (non-Ulysses Attention) and B (Ulyssses Attention)
        torch_assert_close(loss_a, loss_b, atol=1e-05, rtol=1e-05)

        # - we are feeding the exact same sample to each rank of A
        # - for B we feed half the sample to each rank, but in total it's the same sample as each rank of A sees
        # thus we expect very similar grads (but not exact)
        if zero_stage in [1, 2]:
            # possibly some issue with z1/z2 as it requires higher tolerance than z3?
            torch_assert_close(grad_a, grad_b, rtol=1.6e-02, atol=1e-03)
        else:
            torch_assert_close(grad_a, grad_b)
