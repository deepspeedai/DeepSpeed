# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from copy import deepcopy
from types import SimpleNamespace

import deepspeed
import deepspeed.comm as dist
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from unit.common import DistributedTest, preferred_dtype
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.auto_tp import AutoTP
from deepspeed.module_inject.autotp_config import AutoTPConfig, AutoTPPresets, PartitionType
from deepspeed.module_inject.layers import DepthwiseConv1dLayer, LinearAllreduce, LinearLayer, SubParamLinearLayer
from deepspeed.runtime.tensor_parallel.config import TPTrainingConfig
from deepspeed.utils import groups


def skip_on_device():
    if get_accelerator().device_name() == "xpu":
        pytest.skip("XPU requires a higher version for test")


def assert_close_for_preferred_dtype(actual, expected):
    atol = 5e-3
    rtol = 2e-2
    if preferred_dtype() is torch.float32:
        atol = 1e-5
        rtol = 1e-5
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def make_mock_qwen35_config():
    return SimpleNamespace(
        model_type="qwen3_5_text",
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
    )


def load_real_qwen35_classes():
    try:
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    except ImportError:
        pytest.skip("transformers with Qwen3.5 support is required")
    return Qwen3_5TextConfig, Qwen3_5ForCausalLM


def make_real_qwen35_text_config():
    qwen35_text_config, _ = load_real_qwen35_classes()
    return qwen35_text_config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=4,
        linear_key_head_dim=8,
        linear_num_value_heads=4,
        linear_value_head_dim=8,
        attention_dropout=0.0,
        use_cache=False,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )


class MockLinearAttention(nn.Module):
    """GatedDeltaNet-style linear-attention submodule."""

    def __init__(self, hidden_dim):
        super().__init__()
        config = make_mock_qwen35_config()
        self.hidden_size = hidden_dim
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = 4
        self.layer_idx = 0
        self.activation = "silu"
        self.act = F.silu
        self.layer_norm_epsilon = 1e-6

        self.conv1d = nn.Conv1d(self.conv_dim,
                                self.conv_dim,
                                bias=False,
                                kernel_size=self.conv_kernel_size,
                                groups=self.conv_dim,
                                padding=self.conv_kernel_size - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, 2.0, self.num_v_heads)))
        self.norm = MockRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.in_proj_qkv = nn.Linear(hidden_dim, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_dim, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_dim, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_dim, self.num_v_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, hidden_dim, bias=False)
        self.causal_conv1d_fn = None
        self.causal_conv1d_update = mock_torch_causal_conv1d_update
        self.chunk_gated_delta_rule = mock_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = mock_recurrent_gated_delta_rule

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len]).transpose(1, 2)

        z = self.in_proj_z(x).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            repeat_factor = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        core_attn_out, _ = self.chunk_gated_delta_rule(query,
                                                       key,
                                                       value,
                                                       g=g,
                                                       beta=beta,
                                                       initial_state=None,
                                                       output_final_state=False,
                                                       use_qk_l2norm_in_kernel=True)
        core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))


class MockRMSNormGated(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def mock_torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)


def mock_chunk_gated_delta_rule(query,
                                key,
                                value,
                                g,
                                beta,
                                initial_state=None,
                                output_final_state=False,
                                use_qk_l2norm_in_kernel=True):
    del initial_state, use_qk_l2norm_in_kernel
    qk_mix = query + key
    core_attn_out = value + beta.unsqueeze(-1) * qk_mix + g.to(value.dtype).unsqueeze(-1)
    last_recurrent_state = core_attn_out[:, :, -1].contiguous() if output_final_state else None
    return core_attn_out, last_recurrent_state


def mock_recurrent_gated_delta_rule(query,
                                    key,
                                    value,
                                    g,
                                    beta,
                                    initial_state=None,
                                    output_final_state=False,
                                    use_qk_l2norm_in_kernel=True):
    del initial_state
    return mock_chunk_gated_delta_rule(query,
                                       key,
                                       value,
                                       g=g,
                                       beta=beta,
                                       initial_state=None,
                                       output_final_state=output_final_state,
                                       use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)


class MockFullAttention(nn.Module):
    """Standard multi-head attention submodule."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class MockGatedQProjAttention(nn.Module):
    """Full-attention block with a widened q_proj output."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.o_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x))


class MockMLP(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.down_proj = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class MockLinearAttnDecoderLayer(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.linear_attn = MockLinearAttention(hidden_dim)
        self.mlp = MockMLP(hidden_dim)

    def forward(self, x):
        return self.mlp(self.linear_attn(x))


class MockFullAttnDecoderLayer(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = MockFullAttention(hidden_dim)
        self.mlp = MockMLP(hidden_dim)

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class MockGatedQProjLayer(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = MockGatedQProjAttention(hidden_dim)

    def forward(self, x):
        return self.self_attn(x)


class MockQwen35HybridModel(nn.Module):
    """4-layer hybrid model with full attention every fourth layer."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.config = make_mock_qwen35_config()
        self.layers = nn.ModuleList([
            MockLinearAttnDecoderLayer(hidden_dim),
            MockLinearAttnDecoderLayer(hidden_dim),
            MockLinearAttnDecoderLayer(hidden_dim),
            MockFullAttnDecoderLayer(hidden_dim),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MockGatedQProjModel(nn.Module):
    """Single-layer model for widened q_proj sharding checks."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.config = make_mock_qwen35_config()
        self.layers = nn.ModuleList([MockGatedQProjLayer(hidden_dim)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestQwen35PresetPatterns:
    """Verify qwen3_5 preset resolution and pattern coverage."""

    def test_get_preset_returns_config(self):
        config = AutoTPPresets.get_preset("qwen3_5")
        assert config is not None
        assert isinstance(config, AutoTPConfig)

    def test_preset_has_eight_layer_specs(self):
        config = AutoTPPresets.get_preset("qwen3_5")
        assert len(config.layer_specs) == 8

    def test_self_attn_column_parallel(self):
        config = AutoTPPresets.get_preset("qwen3_5")
        for proj in ["q_proj", "k_proj", "v_proj"]:
            spec = config.find_matching_spec(f"model.layers.3.self_attn.{proj}.weight")
            assert spec is not None, f"self_attn.{proj} should match"
            assert spec.partition_type == PartitionType.COLUMN

    def test_self_attn_row_parallel(self):
        config = AutoTPPresets.get_preset("qwen3_5")
        spec = config.find_matching_spec("model.layers.3.self_attn.o_proj.weight")
        assert spec is not None
        assert spec.partition_type == PartitionType.ROW

    def test_mlp_matches_in_linear_attn_layer(self):
        config = AutoTPPresets.get_preset("qwen3_5")

        spec = config.find_matching_spec("model.layers.0.mlp.gate_proj.weight")
        assert spec is not None
        assert spec.partition_type == PartitionType.COLUMN

        spec = config.find_matching_spec("model.layers.0.mlp.up_proj.weight")
        assert spec is not None
        assert spec.partition_type == PartitionType.COLUMN

        spec = config.find_matching_spec("model.layers.0.mlp.down_proj.weight")
        assert spec is not None
        assert spec.partition_type == PartitionType.ROW

    def test_linear_attn_supported_linears_match(self):
        config = AutoTPPresets.get_preset("qwen3_5")

        qkv_spec = config.find_matching_spec("model.layers.0.linear_attn.in_proj_qkv.weight")
        assert qkv_spec is not None
        assert qkv_spec.partition_type == PartitionType.COLUMN
        assert qkv_spec.shape_resolver == "qwen3_5_linear_attn_qkv"

        z_spec = config.find_matching_spec("model.layers.0.linear_attn.in_proj_z.weight")
        assert z_spec is not None
        assert z_spec.partition_type == PartitionType.COLUMN

        a_spec = config.find_matching_spec("model.layers.0.linear_attn.in_proj_a.weight")
        assert a_spec is not None
        assert a_spec.partition_type == PartitionType.COLUMN

        b_spec = config.find_matching_spec("model.layers.0.linear_attn.in_proj_b.weight")
        assert b_spec is not None
        assert b_spec.partition_type == PartitionType.COLUMN

        out_spec = config.find_matching_spec("model.layers.0.linear_attn.out_proj.weight")
        assert out_spec is not None
        assert out_spec.partition_type == PartitionType.ROW

    def test_linear_attn_remaining_weights_not_matched(self):
        config = AutoTPPresets.get_preset("qwen3_5")
        unmatched_names = [
            "model.layers.0.linear_attn.conv1d.weight",
            "model.layers.0.linear_attn.dt_bias",
            "model.layers.0.linear_attn.A_log",
        ]
        for name in unmatched_names:
            assert config.find_matching_spec(name) is None, f"Preset should not match {name}"

    def test_moe_names_not_matched(self):
        config = AutoTPPresets.get_preset("qwen3_5")
        moe_names = [
            "model.layers.0.mlp.experts.gate_up_proj.weight",
            "model.layers.0.mlp.experts.down_proj.weight",
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.layers.0.mlp.shared_expert.down_proj.weight",
        ]
        for name in moe_names:
            assert config.find_matching_spec(name) is None, f"Dense preset should not match {name}"

    def test_multimodal_prefix_still_matches(self):
        config = AutoTPPresets.get_preset("qwen3_5")

        spec = config.find_matching_spec("model.language_model.layers.3.self_attn.q_proj.weight")
        assert spec is not None
        assert spec.partition_type == PartitionType.COLUMN

        spec = config.find_matching_spec("model.language_model.layers.0.mlp.down_proj.weight")
        assert spec is not None
        assert spec.partition_type == PartitionType.ROW

    def test_preset_via_get_partition_config_object(self):
        tp_config = TPTrainingConfig(autotp_size=2, preset_model="qwen3_5")
        config = tp_config.get_partition_config_object()
        assert config is not None
        assert config.tp_size == 2
        assert len(config.layer_specs) == 8


class TestQwen35MockHybridModel(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def _apply_preset(self, model, strict_mode=False):
        groups._init_tp_mesh_device(tensor_model_parallel_size=2)
        config = AutoTPPresets.get_preset("qwen3_5")
        config.strict_mode = strict_mode
        autotp = AutoTP(
            module=model,
            all_reduce_linears=[],
            prefix="",
            state_dict=None,
            linear_layer_setting=None,
            orig_layer_impl=None,
            keep_module_on_host=False,
            partition_config=config,
        )
        autotp.set_tensor_parallel_config(2, groups.get_tensor_model_parallel_group())
        autotp.update_linear_policies()
        autotp._replace_module(model)
        return model

    def test_preset_replaces_supported_layers(self):
        skip_on_device()
        model = MockQwen35HybridModel(hidden_dim=16)
        model = self._apply_preset(model)

        assert isinstance(model.layers[3].self_attn.q_proj, LinearLayer)
        assert isinstance(model.layers[3].self_attn.k_proj, LinearLayer)
        assert isinstance(model.layers[3].self_attn.v_proj, LinearLayer)
        assert isinstance(model.layers[3].self_attn.o_proj, LinearAllreduce)

        assert isinstance(model.layers[3].mlp.gate_proj, LinearLayer)
        assert isinstance(model.layers[3].mlp.up_proj, LinearLayer)
        assert isinstance(model.layers[3].mlp.down_proj, LinearAllreduce)

        for i in range(3):
            assert isinstance(model.layers[i].mlp.gate_proj, LinearLayer)
            assert isinstance(model.layers[i].mlp.up_proj, LinearLayer)
            assert isinstance(model.layers[i].mlp.down_proj, LinearAllreduce)

            assert isinstance(model.layers[i].linear_attn.in_proj_qkv, SubParamLinearLayer)
            assert isinstance(model.layers[i].linear_attn.conv1d, DepthwiseConv1dLayer)
            assert isinstance(model.layers[i].linear_attn.in_proj_z, LinearLayer)
            assert isinstance(model.layers[i].linear_attn.in_proj_a, LinearLayer)
            assert isinstance(model.layers[i].linear_attn.in_proj_b, LinearLayer)
            assert isinstance(model.layers[i].linear_attn.out_proj, LinearAllreduce)
            assert model.layers[i].linear_attn.num_k_heads == 1
            assert model.layers[i].linear_attn.num_v_heads == 1
            assert model.layers[i].linear_attn.key_dim == 4
            assert model.layers[i].linear_attn.value_dim == 4
            assert model.layers[i].linear_attn.conv_dim == 12
            assert model.layers[i].linear_attn.in_proj_qkv.weight.shape == (12, 16)
            assert model.layers[i].linear_attn.conv1d.weight.shape == (12, 1, 4)
            assert model.layers[i].linear_attn.in_proj_z.weight.shape == (4, 16)
            assert model.layers[i].linear_attn.in_proj_a.weight.shape == (1, 16)
            assert model.layers[i].linear_attn.in_proj_b.weight.shape == (1, 16)
            assert model.layers[i].linear_attn.dt_bias.shape == (1, )
            assert model.layers[i].linear_attn.A_log.shape == (1, )

    def test_strict_mode_accepts_full_linear_attn_block(self):
        skip_on_device()
        model = MockQwen35HybridModel(hidden_dim=16)
        model = self._apply_preset(model, strict_mode=True)
        assert isinstance(model.layers[0].linear_attn.in_proj_a, LinearLayer)
        assert isinstance(model.layers[0].linear_attn.conv1d, DepthwiseConv1dLayer)

    def test_linear_attn_supported_weights_shard_cleanly(self):
        skip_on_device()
        hidden_dim = 16
        device = get_accelerator().current_device_name()

        torch.manual_seed(1234)
        model = MockQwen35HybridModel(hidden_dim).to(device=device, dtype=preferred_dtype())
        baseline = deepcopy(model)
        model = self._apply_preset(model)

        torch.manual_seed(4321)
        inputs = torch.randn(2, 3, hidden_dim, dtype=preferred_dtype(), device=device)
        full_output = baseline(inputs)
        tp_output = model(inputs)
        assert_close_for_preferred_dtype(tp_output, full_output)

    def test_gated_q_proj_output_width_still_shards_cleanly(self):
        skip_on_device()
        hidden_dim = 16
        device = get_accelerator().current_device_name()

        torch.manual_seed(1234)
        model = MockGatedQProjModel(hidden_dim).to(device=device, dtype=preferred_dtype())
        baseline = deepcopy(model)
        model = self._apply_preset(model)

        q_proj = model.layers[0].self_attn.q_proj
        o_proj = model.layers[0].self_attn.o_proj

        assert isinstance(q_proj, LinearLayer)
        assert isinstance(o_proj, LinearAllreduce)
        assert q_proj.weight.shape == (hidden_dim, hidden_dim)

        torch.manual_seed(4321)
        inputs = torch.randn(2, hidden_dim, dtype=preferred_dtype(), device=device)
        full_output = baseline(inputs)
        tp_output = model(inputs)
        assert_close_for_preferred_dtype(tp_output, full_output)


class TestQwen35RealHFModel(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def _make_tp_config(self):
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2,
                "preset_model": "qwen3_5",
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-4,
                },
            },
            "zero_optimization": {
                "stage": 0,
            },
            "steps_per_print": 1,
        }
        if preferred_dtype() == torch.float16:
            config["fp16"] = {"enabled": True}
        elif preferred_dtype() == torch.bfloat16:
            config["bf16"] = {"enabled": True}
        return config

    def _seed_all(self, seed):
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)

    def _build_real_baseline_and_engine(self):
        skip_on_device()
        _, qwen35_for_causal_lm = load_real_qwen35_classes()

        config = make_real_qwen35_text_config()
        device = get_accelerator().current_device_name()

        self._seed_all(1234)
        baseline = qwen35_for_causal_lm(config).to(device=device, dtype=preferred_dtype())
        baseline.eval()

        model = deepcopy(baseline)
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=self._make_tp_config(),
        )
        engine.eval()
        return baseline, engine, config, device

    def test_real_qwen35_preset_replaces_supported_layers(self):
        _, engine, _, _ = self._build_real_baseline_and_engine()

        assert engine.autotp_size() == 2
        assert isinstance(engine.module.model.layers[0].linear_attn.in_proj_qkv, SubParamLinearLayer)
        assert isinstance(engine.module.model.layers[0].linear_attn.conv1d, DepthwiseConv1dLayer)
        assert isinstance(engine.module.model.layers[0].linear_attn.in_proj_z, LinearLayer)
        assert isinstance(engine.module.model.layers[0].linear_attn.in_proj_a, LinearLayer)
        assert isinstance(engine.module.model.layers[0].linear_attn.in_proj_b, LinearLayer)
        assert isinstance(engine.module.model.layers[0].linear_attn.out_proj, LinearAllreduce)
        assert isinstance(engine.module.model.layers[3].self_attn.q_proj, LinearLayer)
        assert isinstance(engine.module.model.layers[3].self_attn.o_proj, LinearAllreduce)
        assert engine.module.model.layers[0].linear_attn.num_k_heads == 2
        assert engine.module.model.layers[0].linear_attn.num_v_heads == 2
        assert engine.module.model.layers[0].linear_attn.key_dim == 16
        assert engine.module.model.layers[0].linear_attn.value_dim == 16
        assert engine.module.model.layers[0].linear_attn.conv_dim == 48
        assert engine.module.model.layers[0].linear_attn.dt_bias.shape == (2, )
        assert engine.module.model.layers[0].linear_attn.A_log.shape == (2, )

    def test_real_qwen35_first_forward_matches_baseline(self):
        baseline, engine, config, device = self._build_real_baseline_and_engine()

        self._seed_all(4321)
        input_ids = torch.randint(0, config.vocab_size, (1, 8), device=device)
        dist.broadcast(
            input_ids,
            src=groups.get_tensor_model_parallel_src_rank(),
            group=groups.get_tensor_model_parallel_group(),
        )
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            baseline_output = baseline(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                use_cache=False,
            )
            tp_output = engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                use_cache=False,
            )

        assert_close_for_preferred_dtype(tp_output.loss, baseline_output.loss)
        assert_close_for_preferred_dtype(tp_output.logits, baseline_output.logits)
