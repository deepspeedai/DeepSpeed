# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Shared fixtures and assertions for compact AutoEP tests."""

import copy

import deepspeed
import pytest
import torch
import torch.nn as nn

from deepspeed.accelerator import get_accelerator

UNSET = object()
UNSUPPORTED_LOAD_BALANCE_VALUES = [0, 0.0, 1e-3, 0.02, False, True, "1e-3", [1e-3], {"coeff": 1e-3}]


class MockHFConfig:
    model_type = "mixtral"
    num_local_experts = 4
    num_experts_per_tok = 2
    hidden_size = 64
    intermediate_size = 128


class MockMoEExperts(nn.Module):

    def __init__(self, num_experts=4, ffn_hidden=128, hidden_size=64, intermediate_size=None):
        super().__init__()
        if intermediate_size is not None:
            ffn_hidden = intermediate_size
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * ffn_hidden, hidden_size))
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_size, ffn_hidden))


class MockMoEBlock(nn.Module):

    def __init__(self, num_experts=4, ffn_hidden=128, hidden_size=64, intermediate_size=None):
        super().__init__()
        if intermediate_size is not None:
            ffn_hidden = intermediate_size
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = MockMoEExperts(num_experts, ffn_hidden, hidden_size, intermediate_size)
        self.top_k = 2

    def forward(self, x):
        original_shape = x.shape
        hidden_states = x.reshape(-1, original_shape[-1])
        scores = torch.softmax(self.gate(hidden_states), dim=-1)
        top_scores, top_indices = torch.topk(scores, k=self.top_k, dim=-1)
        top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(hidden_states)

        for expert_idx in range(self.gate.out_features):
            expert_mask = top_indices == expert_idx
            if not expert_mask.any():
                continue
            token_indices, route_indices = expert_mask.nonzero(as_tuple=True)
            expert_input = hidden_states[token_indices]
            gate_up = torch.matmul(expert_input, self.experts.gate_up_proj[expert_idx].transpose(0, 1))
            gate_part, up_part = gate_up.chunk(2, dim=-1)
            expert_output = torch.matmul(
                torch.nn.functional.silu(gate_part) * up_part, self.experts.down_proj[expert_idx].transpose(0, 1))
            output[token_indices] += expert_output * top_scores[token_indices, route_indices].unsqueeze(-1)

        return output.reshape(original_shape)


class MockDenseBlock(nn.Module):

    def __init__(self, hidden_size=64, ffn_hidden=128):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden, bias=False)
        self.down_proj = nn.Linear(ffn_hidden, hidden_size, bias=False)


class MockMoETransformer(nn.Module):

    def __init__(self, num_layers=2, num_experts=4, hidden_size=64, intermediate_size=128, moe_every_n=1):
        super().__init__()
        self.config = MockHFConfig()
        self.config.num_local_experts = num_experts
        self.config.hidden_size = hidden_size
        self.config.intermediate_size = intermediate_size
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            self._make_layer(layer_idx, num_experts, hidden_size, intermediate_size, moe_every_n)
            for layer_idx in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, 100)

    @staticmethod
    def _make_layer(layer_idx, num_experts, hidden_size, intermediate_size, moe_every_n):
        layer = nn.Module()
        layer.self_attn = nn.MultiheadAttention(hidden_size, 1, batch_first=True)
        if layer_idx % moe_every_n == 0:
            layer.mlp = MockMoEBlock(num_experts, intermediate_size, hidden_size)
        else:
            layer.mlp = MockDenseBlock(hidden_size, intermediate_size)
        layer.input_layernorm = nn.LayerNorm(hidden_size)
        layer.post_attention_layernorm = nn.LayerNorm(hidden_size)
        return layer

    def forward(self, x):
        for layer_module in self.model.layers:
            residual = x
            x = layer_module.input_layernorm(x)
            x, _ = layer_module.self_attn(x, x, x)
            x = residual + x
            residual = x
            x = layer_module.post_attention_layernorm(x)
            x = residual + layer_module.mlp(x)
        return self.lm_head(x)


def assert_load_balance_coeff_rejection_message(exc: BaseException, value: object) -> None:
    text = str(exc)
    for needle in ("load_balance_coeff", "expert_bias", "not supported", "null", "omit"):
        assert needle in text
    assert repr(value) in text


def mixed_precision_config():
    accelerator = get_accelerator()
    if accelerator.is_fp16_supported() and accelerator.device_name() != "cpu":
        return {"fp16": {"enabled": True, "initial_scale_power": 8}}
    if accelerator.is_bf16_supported():
        return {"bf16": {"enabled": True}}
    if accelerator.is_fp16_supported():
        return {"fp16": {"enabled": True, "initial_scale_power": 8}}
    pytest.skip("AutoEP tests require fp16 or bf16 support")


def make_autoep_config(zero_stage=0, ep_size=1, load_balance_coeff=UNSET, mixed_precision=True):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4
            },
        },
        "expert_parallel": {
            "enabled": True,
            "autoep_size": ep_size,
            "preset_model": "mixtral",
            "use_grouped_mm": False,
        },
        "zero_optimization": {
            "stage": zero_stage,
        },
    }
    if get_accelerator().device_name() == "cpu":
        config["optimizer"]["torch_adam"] = True
    if mixed_precision:
        config.update(mixed_precision_config())
    if load_balance_coeff is not UNSET:
        config["expert_parallel"]["load_balance_coeff"] = load_balance_coeff
    return config


def make_autoep_integration_config(zero_stage=0, ep_size=2):
    return make_autoep_config(zero_stage=zero_stage, ep_size=ep_size, mixed_precision=False)


def seed_everything(seed=1234):
    torch.manual_seed(seed)
    get_accelerator().manual_seed(seed)
    get_accelerator().manual_seed_all(seed)


def engine_input_dtype(engine):
    if engine.bfloat16_enabled():
        return torch.bfloat16
    if engine.fp16_enabled():
        return torch.float16
    return torch.float32


def init_autoep_engine(ep_size=1, zero_stage=0, load_balance_coeff=UNSET):
    seed_everything(42)
    engine, _, _, _ = deepspeed.initialize(
        model=MockMoETransformer(),
        config=make_autoep_config(zero_stage=zero_stage, ep_size=ep_size, load_balance_coeff=load_balance_coeff),
    )
    return engine


def run_training_steps(engine, num_steps=3, seq_len=8, hidden_dim=64):
    losses = []
    grad_norms = []
    for _ in range(num_steps):
        x = torch.randn(1, seq_len, hidden_dim, device=engine.device)
        loss = engine(x).mean()
        engine.backward(loss)

        total_norm = 0.0
        for param in engine.module.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.float().norm(2).item()**2
        grad_norms.append(total_norm**0.5)
        engine.step()
        losses.append(loss.item())
    return losses, grad_norms


def tiny_causal_lm_inputs():
    input_ids = torch.tensor([[1, 5, 7, 9, 11]], dtype=torch.long)
    return input_ids, input_ids.clone()


def state_matched_models(model_cls, config):
    native_model = model_cls(config)
    autoep_model = model_cls(config)
    autoep_model.load_state_dict(copy.deepcopy(native_model.state_dict()))
    return native_model, autoep_model


def replace_autoep_layers(model, preset_model, expected_count=1, **config_overrides):
    from deepspeed.module_inject.auto_ep import AutoEP
    from deepspeed.module_inject.auto_ep_config import parse_autoep_config

    config = {
        "enabled": True,
        "autoep_size": 1,
        "preset_model": preset_model,
        "use_grouped_mm": False,
        **config_overrides,
    }
    auto_ep = AutoEP(model, parse_autoep_config(config))
    specs = auto_ep.ep_parser()
    assert len(specs) == expected_count
    for spec in specs:
        auto_ep.replace_moe_layer(spec, ep_size=1, ep_rank=0)
    return specs


def assert_causal_lm_outputs_close(native_model,
                                   autoep_model,
                                   *,
                                   output_router_logits=False,
                                   compare_router_logits=False,
                                   compare_aux_loss=False,
                                   compare_logits=True,
                                   rtol=1e-5,
                                   atol=1e-6):
    input_ids, labels = tiny_causal_lm_inputs()
    native_model.eval()
    autoep_model.eval()
    with torch.no_grad():
        native_outputs = native_model(input_ids=input_ids, labels=labels, output_router_logits=output_router_logits)
        autoep_outputs = autoep_model(input_ids=input_ids, labels=labels, output_router_logits=output_router_logits)

    if compare_router_logits:
        assert autoep_outputs.router_logits
        torch.testing.assert_close(autoep_outputs.router_logits[0],
                                   native_outputs.router_logits[0],
                                   rtol=rtol,
                                   atol=atol)
    if compare_aux_loss:
        assert autoep_outputs.aux_loss is not None
        torch.testing.assert_close(autoep_outputs.aux_loss, native_outputs.aux_loss, rtol=rtol, atol=atol)
    if compare_logits:
        torch.testing.assert_close(autoep_outputs.logits, native_outputs.logits, rtol=rtol, atol=atol)
    torch.testing.assert_close(autoep_outputs.loss, native_outputs.loss, rtol=rtol, atol=atol)


def skip_unless_transformers_has(transformers, *names, min_version=None, reason="AutoEP coverage"):
    from packaging.version import Version

    if min_version is not None and Version(transformers.__version__) < Version(min_version):
        pytest.skip(f"{reason} requires Transformers >= {min_version}")
    missing = [name for name in names if not hasattr(transformers, name)]
    if missing:
        pytest.skip(f"Installed transformers does not expose required classes: {missing}")


def tiny_mixtral_config(transformers):
    return transformers.MixtralConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        output_router_logits=True,
        tie_word_embeddings=False,
        use_cache=False,
    )
