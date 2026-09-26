# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Accelerator-backed v1 HybridEngineRollout tests."""

from types import SimpleNamespace

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig


def test_continuous_generation_profile_on_accelerator():
    accelerator = get_accelerator()
    if not accelerator.is_available():
        pytest.skip("An accelerator is required for asynchronous profiling coverage")

    class CacheConfig(SimpleNamespace):

        def get_text_config(self, **_kwargs):
            return self

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = CacheConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            kwargs.pop("cache_position", None)
            kwargs.pop("position_ids", None)
            _, values = past_key_values.update(states, states, layer_idx=0, **kwargs)
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16), device=input_ids.device)
            logits[..., 7] = 1
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    device = torch.device(accelerator.device_name())
    model = CacheClassModel().to(device)
    rollout = HybridEngineRollout(
        SimpleNamespace(module=model),
        SimpleNamespace(pad_token_id=0, eos_token_id=2),
        cfg=HybridEngineRolloutConfig(enable_profiling=True),
    )
    request = RolloutRequest(
        torch.tensor([[1, 2, 3], [1, 2, 4]], device=device),
        torch.ones((2, 3), dtype=torch.long, device=device),
    )

    output = rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1))

    profile = rollout.get_last_profile()
    assert output.input_ids[:, 3:].cpu().tolist() == [[7, 7], [7, 7]]
    assert profile["num_prefill_forwards"] == 2
    assert profile["num_decode_forwards"] == 2
    assert profile["num_generated_tokens"] == 4
    assert profile["active_batch_size"] == 1
    assert profile["continuous_batch_size"] == 1
    assert profile["total_ms"] >= profile["generation_ms"]
