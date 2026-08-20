# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import os
import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder, OpBuilder
from deepspeed.runtime.hybrid_engine_graph import (DecodeGraphCache, decode_steps_from_generate_kwargs,
                                                   validate_cuda_graph_support)
from unit.common import DistributedTest

# ---------------------------------------------------------------------------
# Pure-logic tests: no accelerator or model needed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs,expected", [
    ({
        "max_new_tokens": 32,
        "min_new_tokens": 32
    }, 31),
    ({
        "max_new_tokens": 1,
        "min_new_tokens": 1
    }, 0),
    ({
        "max_new_tokens": 32
    }, None),
    ({
        "max_new_tokens": 32,
        "min_new_tokens": 8
    }, None),
    ({
        "max_length": 64
    }, None),
    ({}, None),
])
def test_decode_steps_only_for_pinned_length(kwargs, expected):
    # An open-ended length is unusable: a sequence that outruns its captured
    # graphs cannot fall back to eager without a stale sequence counter.
    assert decode_steps_from_generate_kwargs(kwargs) == expected


class _StubConfig:

    def __init__(self, release_inference_cache=False, inference_tp_size=1):
        self.release_inference_cache = release_inference_cache
        self.inference_tp_size = inference_tp_size


def test_validate_rejects_zero3():
    assert "ZeRO stage 3" in validate_cuda_graph_support(_StubConfig(), zero_stage=3)


def test_validate_rejects_released_cache():
    reason = validate_cuda_graph_support(_StubConfig(release_inference_cache=True), zero_stage=2)
    assert reason is not None


def test_validate_rejects_inference_tp():
    reason = validate_cuda_graph_support(_StubConfig(inference_tp_size=2), zero_stage=2)
    assert reason is not None


class _CountingForward:
    """Stands in for the wrapped module forward."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return "eager"


def _decode_kwargs():
    return {"input_ids": torch.zeros((2, 1), dtype=torch.long)}


def test_unknown_length_runs_eager():
    forward = _CountingForward()
    cache = DecodeGraphCache(forward, max_positions=8)
    cache.begin_sequence(None)

    assert cache(**_decode_kwargs()) == "eager"
    assert cache.captured_positions == 0
    assert forward.calls == 1


def test_length_beyond_max_positions_runs_eager():
    forward = _CountingForward()
    cache = DecodeGraphCache(forward, max_positions=4)
    cache.begin_sequence(16)

    assert cache(**_decode_kwargs()) == "eager"
    assert forward.calls == 1


def test_prompt_forward_never_uses_graphs():
    forward = _CountingForward()
    cache = DecodeGraphCache(forward, max_positions=8)
    cache.begin_sequence(4)

    prompt = {"input_ids": torch.zeros((2, 16), dtype=torch.long)}
    assert cache(**prompt) == "eager"
    assert cache.captured_positions == 0


def test_changing_generation_length_invalidates_graphs():
    cache = DecodeGraphCache(_CountingForward(), max_positions=64)
    cache.begin_sequence(8)
    # Populate the cache the way a capture pass would, without touching CUDA.
    cache._graphs[0] = object()
    cache._static_kwargs[0] = {}
    cache._static_outputs[0] = object()

    cache.begin_sequence(8)
    assert cache.captured_positions == 1, "same length must reuse captured graphs"

    cache.begin_sequence(16)
    assert cache.captured_positions == 0, "new length must discard stale graphs"


# ---------------------------------------------------------------------------
# End-to-end equivalence: needs a GPU and the inference kernels
# ---------------------------------------------------------------------------

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("This op had not been implemented on this system.", allow_module_level=True)

if OpBuilder.installed_rocm_version() != (0, 0):
    pytest.skip("skip inference tests on rocm for now", allow_module_level=True)


@pytest.mark.seq_inference
class TestHybridEngineCudaGraphEquivalence(DistributedTest):
    world_size = 1

    def test_graph_generation_matches_eager(self):
        from transformers import AutoModelForCausalLM

        model_name = "facebook/opt-125m"
        prompt_len, gen_len, bsz = 32, 16, 2
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = f"{get_accelerator().device_name()}:{local_rank}"

        config = {
            "train_batch_size": bsz,
            "train_micro_batch_size_per_gpu": bsz,
            "steps_per_print": 10**9,
            "zero_optimization": {
                "stage": 2
            },
            "fp16": {
                "enabled": True
            },
            "hybrid_engine": {
                "enabled": True,
                "max_out_tokens": prompt_len + gen_len,
                "enable_cuda_graph": True,
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-7
                }
            },
        }

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.half)
        model.config.use_cache = True
        engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        assert engine._decode_graphs is not None, "CUDA graphs should be active for this config"

        torch.manual_seed(0)
        input_ids = torch.randint(100, 1000, (bsz, prompt_len), device=device)
        gen_kwargs = dict(input_ids=input_ids,
                          attention_mask=torch.ones_like(input_ids),
                          max_new_tokens=gen_len,
                          min_new_tokens=gen_len,
                          do_sample=False,
                          pad_token_id=1,
                          synced_gpus=False)

        engine.eval()
        with torch.no_grad():
            graphed = engine.generate(**gen_kwargs).clone()
        assert engine._decode_graphs.captured_positions == gen_len - 1

        # Detach the graph dispatcher and repeat the same generation eagerly.
        engine.module.forward = engine._orig_module_forward
        engine._decode_graphs = None
        with torch.no_grad():
            eager = engine.generate(**gen_kwargs).clone()

        assert torch.equal(graphed, eager), "CUDA graph generation diverged from eager generation"
