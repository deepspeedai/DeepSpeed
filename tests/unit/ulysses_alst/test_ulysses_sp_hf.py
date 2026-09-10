# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
UlyssesPlus: UlyssesSPHF tests
"""

import deepspeed.runtime.sequence_parallel.ulysses_sp as ulysses_sp
import deepspeed.runtime.sequence_parallel.ulysses_linear_attention as linear_cp
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
            self.norm = TestLinearAttentionCPHelpers.FakeNorm()
            self.out_proj = torch.nn.Linear(4, 4, bias=False)
            self.original_forward_calls = 0

        def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
            self.original_forward_calls += 1
            return hidden_states

    class FakeDecoderLayer(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.block_type = "linear_attention"
            self.linear_attn = TestLinearAttentionCPHelpers.FakeGatedDeltaNet()

        def forward(self, hidden_states, position_ids=None, **kwargs):
            return self.linear_attn(hidden_states, **kwargs)

    class FakeTextModel(torch.nn.Module):

        def __init__(self, num_layers=2):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [TestLinearAttentionCPHelpers.FakeDecoderLayer() for _ in range(num_layers)])

        def forward(self, hidden_states, position_ids=None, **kwargs):
            for layer in self.layers:
                hidden_states = layer(hidden_states, position_ids=position_ids, **kwargs)
            return hidden_states

    class FakeModel(torch.nn.Module):

        def __init__(self, num_layers=2):
            super().__init__()
            self.text_model = TestLinearAttentionCPHelpers.FakeTextModel(num_layers)

    @staticmethod
    def fake_fla_ops(build_cp_context=None):

        def default_build_cp_context(**kwargs):
            return object()

        def fake_causal_conv1d(x, weight=None, bias=None, activation=None, cp_context=None):
            return x, None

        def fake_chunk_gated_delta_rule(query, key, value, g=None, beta=None, cp_context=None, **kwargs):
            return value, None

        return linear_cp._FLACPOps(
            build_cp_context or default_build_cp_context,
            fake_causal_conv1d,
            fake_chunk_gated_delta_rule,
        )

    def test_position_ids_to_packed_cu_seqlens(self):
        position_ids = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2]])

        cu_seqlens = linear_cp.position_ids_to_packed_cu_seqlens(position_ids)

        torch_assert_equal(cu_seqlens, torch.tensor([0, 3, 5, 8], dtype=torch.long))

    def test_position_ids_detect_nonzero_resets(self):
        position_ids = torch.tensor([[5, 6, 7, 5, 6]])

        document_ids = linear_cp.position_ids_to_document_ids(position_ids)
        cu_seqlens = linear_cp.position_ids_to_packed_cu_seqlens(position_ids)

        torch_assert_equal(document_ids, torch.tensor([[0, 0, 0, 1, 1]]))
        torch_assert_equal(cu_seqlens, torch.tensor([0, 3, 5]))

    def test_non_linear_kimi_config_does_not_probe_fla(self, monkeypatch):

        def fail_if_called():
            raise AssertionError("FLA should not be probed for a full-attention config")

        monkeypatch.setattr(linear_cp, "load_fla_cp_ops", fail_if_called)
        cfg = types.SimpleNamespace(model_type="kimi_k25", layer_types=["full_attention"])

        assert linear_cp.prepare_qwen35_linear_attention_cp(torch.nn.Identity(), cfg, cfg, "sdpa", False) is None

    def test_unsupported_linear_model_type_fails_without_global_patching(self):
        cfg = types.SimpleNamespace(model_type="future_linear", layer_types=["linear_attention"])

        with pytest.raises(RuntimeError, match="only Qwen3.5"):
            linear_cp.prepare_qwen35_linear_attention_cp(torch.nn.Identity(), cfg, cfg, "sdpa", False)

    def test_linear_attention_cp_version_gate(self, monkeypatch):
        monkeypatch.setattr(linear_cp.importlib_metadata, "version", lambda _name: "0.4.1")

        with pytest.raises(ImportError, match=">= 0.4.2"):
            linear_cp.load_fla_cp_ops()

    def test_current_style_qwen_layer_is_supported_and_patch_is_reversible(self, monkeypatch):
        model = self.FakeModel()
        cfg = types.SimpleNamespace(model_type="qwen3_5", layer_types=["linear_attention", "linear_attention"])
        monkeypatch.setattr(linear_cp, "load_fla_cp_ops", self.fake_fla_ops)
        class_forward = type(model.text_model.layers[0].linear_attn).forward

        prepared = linear_cp.prepare_qwen35_linear_attention_cp(model, cfg, cfg, "flash_attention_2", False)
        registration = linear_cp.create_qwen35_linear_attention_registration(prepared, object(), 2)
        registration.install()

        assert "forward" in model.text_model.layers[0].linear_attn.__dict__
        assert type(model.text_model.layers[0].linear_attn).forward is class_forward

        registration.restore()
        assert "forward" not in model.text_model.layers[0].linear_attn.__dict__
        assert type(model.text_model.layers[0].linear_attn).forward is class_forward

    def test_qwen35_moe_model_type_uses_capability_adapter(self, monkeypatch):
        model = self.FakeModel()
        cfg = types.SimpleNamespace(model_type="qwen3_5_moe_text", layer_types=["linear_attention"])
        monkeypatch.setattr(linear_cp, "load_fla_cp_ops", self.fake_fla_ops)

        prepared = linear_cp.prepare_qwen35_linear_attention_cp(
            model,
            cfg,
            cfg,
            "flash_attention_2",
            False,
            micro_batch_size=1,
        )

        assert len(prepared["linear_layers"]) == 2

    def test_linear_attention_rejects_micro_batch_greater_than_one(self, monkeypatch):
        model = self.FakeModel()
        cfg = types.SimpleNamespace(model_type="qwen3_5_text", layer_types=["linear_attention"])
        monkeypatch.setattr(linear_cp, "load_fla_cp_ops", self.fake_fla_ops)

        with pytest.raises(RuntimeError, match="micro_batch_size=1"):
            linear_cp.prepare_qwen35_linear_attention_cp(
                model,
                cfg,
                cfg,
                "flash_attention_2",
                False,
                micro_batch_size=2,
            )

    def test_multimodal_qwen35_is_rejected_before_patching(self, monkeypatch):
        model = self.FakeModel()
        text_cfg = types.SimpleNamespace(model_type="qwen3_5_text", layer_types=["linear_attention"])
        full_cfg = types.SimpleNamespace(
            model_type="qwen3_5",
            text_config=text_cfg,
            vision_config=types.SimpleNamespace(model_type="qwen3_5_vision"),
        )
        monkeypatch.setattr(linear_cp, "load_fla_cp_ops", self.fake_fla_ops)

        with pytest.raises(RuntimeError, match="multimodal Ulysses SP is not supported"):
            linear_cp.prepare_qwen35_linear_attention_cp(
                model,
                full_cfg,
                text_cfg,
                "flash_attention_2",
                False,
                micro_batch_size=1,
            )

    def test_metadata_is_built_once_and_shared_by_all_linear_layers(self, monkeypatch):
        model = self.FakeModel(num_layers=3)
        build_calls = []
        gather_calls = []

        def fake_build_cp_context(**kwargs):
            build_calls.append(kwargs)
            return object()

        def fake_all_gather(outputs, local, group=None):
            gather_calls.append(local.clone())
            outputs[0].copy_(local)
            outputs[1].copy_(torch.tensor([[2, 3]], dtype=local.dtype))

        registration = linear_cp.Qwen35LinearAttentionCPRegistration(
            model=model,
            text_model=model.text_model,
            decoder_layers=list(model.text_model.layers),
            linear_layers=[layer.linear_attn for layer in model.text_model.layers],
            fla_ops=self.fake_fla_ops(fake_build_cp_context),
            sp_group=object(),
            sp_world_size=2,
            core_attn_implementation="flash_attention_2",
            disable_in_eval=False,
        )
        monkeypatch.setattr(linear_cp.dist, "all_gather", fake_all_gather)
        registration.install()
        try:
            output = model.text_model(
                torch.randn(1, 2, 4),
                position_ids=torch.tensor([[0, 1]]),
            )
        finally:
            registration.restore()

        assert output.shape == (1, 2, 4)
        assert len(gather_calls) == 1
        assert len(build_calls) == 1

    def test_explicit_packed_metadata_takes_precedence(self, monkeypatch):
        model = self.FakeModel(num_layers=1)
        built = []

        def fake_build_cp_context(**kwargs):
            built.append(kwargs)
            return object()

        def fake_all_gather(outputs, local, group=None):
            outputs[0].copy_(local)
            outputs[1].copy_(torch.tensor([[2, 0]], dtype=local.dtype))

        registration = linear_cp.Qwen35LinearAttentionCPRegistration(
            model=model,
            text_model=model.text_model,
            decoder_layers=list(model.text_model.layers),
            linear_layers=[model.text_model.layers[0].linear_attn],
            fla_ops=self.fake_fla_ops(fake_build_cp_context),
            sp_group=object(),
            sp_world_size=2,
            core_attn_implementation="flash_attention_2",
            disable_in_eval=False,
        )
        monkeypatch.setattr(linear_cp.dist, "all_gather", fake_all_gather)

        metadata = registration._build_metadata(
            torch.tensor([[0, 1]]),
            explicit_cu_seqlens=torch.tensor([0, 3, 4]),
        )

        torch_assert_equal(metadata.global_cu_seqlens, torch.tensor([0, 3, 4]))
        torch_assert_equal(built[0]["cu_seqlens"], torch.tensor([0, 3, 4]))

    def test_conflicting_packed_metadata_is_rejected(self):
        model = self.FakeModel(num_layers=1)
        registration = linear_cp.Qwen35LinearAttentionCPRegistration(
            model=model,
            text_model=model.text_model,
            decoder_layers=list(model.text_model.layers),
            linear_layers=[model.text_model.layers[0].linear_attn],
            fla_ops=self.fake_fla_ops(),
            sp_group=object(),
            sp_world_size=1,
            core_attn_implementation="flash_attention_2",
            disable_in_eval=False,
        )

        with pytest.raises(RuntimeError, match="position_ids document boundaries do not match"):
            registration._build_metadata(
                torch.tensor([[0, 1, 2, 3]]),
                explicit_cu_seqlens=torch.tensor([0, 2, 4]),
            )

        with pytest.raises(RuntimeError, match="cu_seq_lens_q and cu_seq_lens_k"):
            registration._build_metadata(
                torch.tensor([[0, 1, 0, 1]]),
                explicit_cu_seqlens=torch.tensor([0, 2, 4]),
                explicit_cu_seqlens_k=torch.tensor([0, 3, 4]),
            )

    def test_packed_sdpa_fails_before_any_layer_runs(self, monkeypatch):
        model = self.FakeModel(num_layers=1)

        def fake_all_gather(outputs, local, group=None):
            outputs[0].copy_(local)
            outputs[1].copy_(torch.tensor([[0, 1]], dtype=local.dtype))

        registration = linear_cp.Qwen35LinearAttentionCPRegistration(
            model=model,
            text_model=model.text_model,
            decoder_layers=list(model.text_model.layers),
            linear_layers=[model.text_model.layers[0].linear_attn],
            fla_ops=self.fake_fla_ops(),
            sp_group=object(),
            sp_world_size=2,
            core_attn_implementation="sdpa",
            disable_in_eval=False,
        )
        monkeypatch.setattr(linear_cp.dist, "all_gather", fake_all_gather)

        with pytest.raises(RuntimeError, match="Packed Ulysses SP is not supported with SDPA"):
            registration._build_metadata(torch.tensor([[0, 1]]))

    def test_disable_in_eval_delegates_to_original_linear_forward(self):
        model = self.FakeModel(num_layers=1)
        registration = linear_cp.Qwen35LinearAttentionCPRegistration(
            model=model,
            text_model=model.text_model,
            decoder_layers=list(model.text_model.layers),
            linear_layers=[model.text_model.layers[0].linear_attn],
            fla_ops=self.fake_fla_ops(),
            sp_group=object(),
            sp_world_size=2,
            core_attn_implementation="flash_attention_2",
            disable_in_eval=True,
        )
        model.eval()
        registration.install()
        try:
            model.text_model(torch.randn(1, 4, 4), position_ids=torch.arange(4).unsqueeze(0))
        finally:
            registration.restore()

        assert model.text_model.layers[0].linear_attn.original_forward_calls == 1

    def test_linear_cp_preserves_parent_and_child_accelerate_hooks(self):
        accelerate_hooks = pytest.importorskip("accelerate.hooks")
        model = self.FakeModel(num_layers=1)
        linear_layer = model.text_model.layers[0].linear_attn

        class CountingHook(accelerate_hooks.ModelHook):

            def __init__(self):
                self.pre_calls = 0
                self.post_calls = 0

            def pre_forward(self, module, *args, **kwargs):
                self.pre_calls += 1
                return args, kwargs

            def post_forward(self, module, output):
                self.post_calls += 1
                return output

        parent_hook = CountingHook()
        child_hook = CountingHook()
        accelerate_hooks.add_hook_to_module(linear_layer, parent_hook)
        accelerate_hooks.add_hook_to_module(linear_layer.conv1d, child_hook)
        registration = linear_cp.Qwen35LinearAttentionCPRegistration(
            model=model,
            text_model=model.text_model,
            decoder_layers=list(model.text_model.layers),
            linear_layers=[linear_layer],
            fla_ops=self.fake_fla_ops(),
            sp_group=object(),
            sp_world_size=1,
            core_attn_implementation="flash_attention_2",
            disable_in_eval=False,
        )
        registration.install()
        try:
            model.text_model(torch.randn(1, 4, 4), position_ids=torch.arange(4).unsqueeze(0))
        finally:
            registration.restore()
            accelerate_hooks.remove_hook_from_module(linear_layer)
            accelerate_hooks.remove_hook_from_module(linear_layer.conv1d)

        assert parent_hook.pre_calls == parent_hook.post_calls == 1
        assert child_hook.pre_calls == child_hook.post_calls == 1

    def test_registration_rolls_back_attention_and_linear_patches(self, monkeypatch):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu

        original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
        destroyed = []

        class FailingLinearRegistration:
            restored = False

            def install(self):
                raise RuntimeError("linear installation failed")

            def restore(self):
                self.restored = True

        failing_registration = FailingLinearRegistration()
        arch_cfg = types.SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=64,
            num_hidden_layers=2,
        )
        config = types.SimpleNamespace(
            _attn_implementation="sdpa",
            get_text_config=lambda: arch_cfg,
        )
        model = types.SimpleNamespace(config=config)

        class FakeUlysses(UlyssesSPAttentionHF):

            def __init__(self, attn, **kwargs):
                torch.nn.Module.__init__(self)
                self.attn = attn

        monkeypatch.setattr(ulysses_sp, "_ACTIVE_ATTENTION_REGISTRATIONS", {})
        monkeypatch.setattr(ulysses_sp, "_ACTIVE_LINEAR_ATTENTION_REGISTRATION", None)
        monkeypatch.setattr(ulysses_sp, "prepare_qwen35_linear_attention_cp", lambda **kwargs: object())
        monkeypatch.setattr(
            ulysses_sp,
            "create_qwen35_linear_attention_registration",
            lambda *args, **kwargs: failing_registration,
        )
        monkeypatch.setattr(mpu, "initialize_sequence_parallel", lambda **kwargs: None)
        monkeypatch.setattr(mpu, "get_sequence_parallel_group", lambda: object())
        monkeypatch.setattr(mpu, "get_sequence_parallel_world_size", lambda: 2)
        monkeypatch.setattr(mpu, "destroy_sequence_parallel", lambda: destroyed.append(True))

        with pytest.raises(RuntimeError, match="linear installation failed"):
            FakeUlysses.register_with_transformers(
                model_name_or_path=model,
                core_attn_implementation="sdpa",
                sequence_parallel_size=2,
                micro_batch_size=1,
            )

        assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original_sdpa
        assert failing_registration.restored
        assert destroyed == [True]
        assert ulysses_sp._ACTIVE_ATTENTION_REGISTRATIONS == {}


class TestUlyssesSPHFBatchSizeTwo(DistributedTest):
    world_size = 2

    def test_dense_batch_size_two(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        rank = dist.get_rank()
        device = torch.device("cuda", rank)
        config = LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=64,
            use_cache=False,
        )
        input_ids = torch.tensor(
            [[1, 10, 11, 12, 13, 14, 15, 2], [1, 20, 21, 22, 23, 24, 25, 2]],
            device=device,
        )
        position_ids = torch.arange(input_ids.shape[1], device=device).expand(input_ids.shape[0], -1)

        torch.manual_seed(42)
        baseline = LlamaForCausalLM(config).to(device).eval()
        with torch.no_grad():
            expected = baseline(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits

        torch.manual_seed(42)
        candidate = LlamaForCausalLM(config).to(device).eval()
        UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=candidate,
            core_attn_implementation="sdpa",
            sequence_parallel_size=self.world_size,
            micro_batch_size=2,
            seq_length=input_ids.shape[1],
            seq_length_is_variable=False,
        )
        with torch.no_grad():
            actual = candidate(
                input_ids=input_ids.chunk(self.world_size, dim=1)[rank],
                position_ids=position_ids.chunk(self.world_size, dim=1)[rank],
                use_cache=False,
            ).logits

        torch_assert_close(expected.chunk(self.world_size, dim=1)[rank], actual)
        UlyssesSPAttentionHF.unregister_from_transformers("sdpa")


class TestUlyssesSPDataLoaderMetadata(DistributedTest):
    world_size = 2

    def test_global_cu_seqlens_survive_sharding_and_padding(self):
        import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu

        rank = dist.get_rank()
        device = torch.device("cuda", rank)
        full_input_ids = torch.arange(1, 8).unsqueeze(0)
        source_batch = {
            "input_ids": full_input_ids,
            "labels": full_input_ids.clone(),
            "cu_seq_lens_q": torch.tensor([0, 3, 7], dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor([0, 3, 7], dtype=torch.int32),
            "max_length_q": torch.tensor(4),
            "max_length_k": torch.tensor(4),
        }
        dataloader = torch.utils.data.DataLoader([source_batch], batch_size=1, collate_fn=lambda items: items[0])

        mpu.initialize_sequence_parallel(sequence_parallel_size=self.world_size)
        try:
            adapter = UlyssesSPDataLoaderAdapter(
                dataloader,
                sp_rank=mpu.get_sequence_parallel_rank(),
                sp_group=mpu.get_sequence_parallel_group(),
                sp_world_size=mpu.get_sequence_parallel_world_size(),
                device=device,
            )
            batch = next(adapter)
        finally:
            mpu.destroy_sequence_parallel()

        expected_positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]]).chunk(self.world_size, dim=1)[rank]
        assert batch["input_ids"].shape == (1, 4)
        torch_assert_equal(batch["position_ids"], expected_positions)
        torch_assert_equal(batch["cu_seq_lens_q"], torch.tensor([0, 3, 8], dtype=torch.int32))
        torch_assert_equal(batch["cu_seq_lens_k"], torch.tensor([0, 3, 8], dtype=torch.int32))
        torch_assert_equal(batch["max_length_q"], torch.tensor(5))
        torch_assert_equal(batch["max_length_k"], torch.tensor(5))


class TestUlyssesSPRegistrationLifecycle(DistributedTest):
    world_size = 2

    def test_scoped_wrapper_and_sequential_registration(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu

        rank = dist.get_rank()
        device = torch.device("cuda", rank)
        config = LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=64,
            use_cache=False,
        )
        input_ids = torch.tensor([[1, 10, 11, 12, 13, 14, 15, 2]], device=device)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

        torch.manual_seed(123)
        unrelated = LlamaForCausalLM(config).to(device).eval()
        with torch.no_grad():
            unrelated_expected = unrelated(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits

        for iteration in range(2):
            torch.manual_seed(123)
            candidate = LlamaForCausalLM(config).to(device).eval()
            UlyssesSPAttentionHF.register_with_transformers(
                model_name_or_path=candidate,
                core_attn_implementation="sdpa",
                sequence_parallel_size=self.world_size,
                micro_batch_size=1,
                seq_length=input_ids.shape[1],
                seq_length_is_variable=False,
            )
            if iteration == 0:
                with pytest.raises(ValueError, match="is not registered"):
                    UlyssesSPAttentionHF.unregister_from_transformers("flash_attention_2")
                with torch.no_grad():
                    unrelated_actual = unrelated(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        use_cache=False,
                    ).logits
                torch_assert_equal(unrelated_expected, unrelated_actual)

            with torch.no_grad():
                candidate_actual = candidate(
                    input_ids=input_ids.chunk(self.world_size, dim=1)[rank],
                    position_ids=position_ids.chunk(self.world_size, dim=1)[rank],
                    use_cache=False,
                ).logits
            torch_assert_close(unrelated_expected.chunk(self.world_size, dim=1)[rank], candidate_actual)
            UlyssesSPAttentionHF.unregister_from_transformers("sdpa")
            assert getattr(mpu, "_SEQUENCE_PARALLEL_GROUP") is None
            assert getattr(mpu, "_SEQUENCE_DATA_PARALLEL_GROUP") is None
            assert getattr(mpu, "_CREATED_SEQUENCE_PARALLEL_GROUPS") == []


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

    def test_disable_in_eval_preserves_padding_mask(self):
        model_name_or_path = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
        rank = dist.get_rank()
        device = torch.device("cuda", rank)
        input_ids = tensor(
            [[1, 10, 11, 12, 2, 2], [1, 20, 21, 2, 2, 2]],
            device=device,
        )
        attention_mask = tensor(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]],
            device=device,
        )
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        model_baseline = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device).eval()
        with torch.no_grad():
            expected = model_baseline(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits

        UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=model_name_or_path,
            core_attn_implementation="sdpa",
            sequence_parallel_size=self.world_size,
            micro_batch_size=2,
            seq_length_is_variable=True,
            disable_in_eval=True,
        )
        model_sp = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device).eval()
        with torch.no_grad():
            actual = model_sp(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits

        torch_assert_equal(expected, actual)
        UlyssesSPAttentionHF.unregister_from_transformers("sdpa")


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

    @pytest.mark.parametrize("packed", [False, True])
    def test_ulysses_sp_hf_flex_attention(self, zero_stage, packed):
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
            use_cache=not packed,
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
        positions = [0, 1, 2, 3, 0, 1] if packed else [0, 1, 2, 3, 4, 5]
        position_ids = tensor([positions, positions])
        ds = torch.utils.data.TensorDataset(input_ids, position_ids)

        # 1. Baseline: DataLoader calibration
        dl_a = torch.utils.data.DataLoader(ds, batch_size=micro_batch_size, collate_fn=collate_fn)
        batch_a = next(iter(dl_a))
        #print(f"{rank=} {batch_a=}")
        expected_batch_a = {
            'input_ids': tensor([[1, 10, 10, 10, 2, 2]]),
            'position_ids': tensor([positions]),
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
                'position_ids': tensor([positions[3:]]),
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
        torch_assert_close(loss_a, loss_b, atol=1e-02, rtol=2e-03)

        # - we are feeding the exact same sample to each rank of A
        # - for B we feed half the sample to each rank, but in total it's the same sample as each rank of A sees
        # thus we expect very similar grads (but not exact)
        # FlexAttention and Ulysses traverse different distributed BF16 reduction orders.
        # Direct full-vs-SP logits are exact; gradients differ only at low BF16 accumulation bits.
        torch_assert_close(grad_a, grad_b, rtol=2e-02, atol=2e-03)
