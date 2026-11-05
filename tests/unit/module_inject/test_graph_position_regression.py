# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""NL-1 regression: the graph path must pass explicit position tensors.

HF derives cache positions from the cache when absent; with a standard
full-attention model (LLaMA) + StaticCache that derivation D2H-syncs and
aborts stream capture ("operation not permitted when stream is
capturing", reproduced by the reviewer on PyTorch 2.9.1 / Transformers
4.57.1). Runs the graph path end to end on a tiny random LLaMA.
"""

import pytest
import torch

from deepspeed.accelerator import get_accelerator

from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from deepspeed.runtime.rollout.base import RolloutRecord, RolloutRequest, SamplingConfig

_TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


class _Shim:

    def __init__(self, m):
        self.module = m


@pytest.mark.skipif(get_accelerator().device_name() == "cpu", reason="non-CPU device required")
def test_graph_capture_standard_model():
    """Graph capture on a standard (non-hybrid) LLaMA must not sync-abort
    when HF would otherwise derive cache positions internally."""
    pytest.importorskip("transformers")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(_TINY_LLAMA)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(_TINY_LLAMA, dtype=torch.bfloat16).to("cuda")
    except Exception as e:  # offline / model unavailable
        pytest.skip(f"model {_TINY_LLAMA} unavailable: {e}")

    rollout = HybridEngineRollout(_Shim(model), tok, HybridEngineRolloutConfig(use_graph_capture=True))
    prompt = torch.randint(1000, 5000, (6, )).tolist()
    req = RolloutRequest(samples=[RolloutRecord(tokens=prompt, attention_mask=[1] * len(prompt))])
    out = rollout.generate(req, SamplingConfig(max_new_tokens=8, temperature=0.0))
    sample = out.samples[0]
    assert sample.tokens[:6] == prompt
    # A response ends at its EOS, so it may be shorter than max_new_tokens.
    assert 1 <= sample.response_length <= 8, "generation length mismatch"
    assert len(sample.tokens) == 6 + sample.response_length
