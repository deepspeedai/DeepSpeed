# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Expert precision selection must preserve the base model and gradient contract."""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.moe.ep_experts import GroupedExperts


@pytest.mark.parametrize("counts", [[2, 1], [0, 3], [3, 0], [0, 0]])
def test_reference_experts_match_independent_linear_modules(counts):
    # Catches wrong gate/up/down orientation and lost empty-expert gradients.
    torch.manual_seed(42)
    experts = GroupedExperts(8, 16, 2, use_grouped_mm=False)
    references = []
    with torch.no_grad():
        for weight in experts.parameters():
            weight.normal_(std=0.1)
        for i in range(2):
            projections = [
                torch.nn.Linear(8, 16, bias=False),
                torch.nn.Linear(8, 16, bias=False),
                torch.nn.Linear(16, 8, bias=False)
            ]
            for linear, weight in zip(projections, (experts.w1, experts.w3, experts.w2)):
                linear.weight.copy_(weight[i])
            references.append(projections)
    x = torch.randn(sum(counts), 8, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual = experts(x, torch.tensor(counts))
    ref_outputs = []
    for rows, (gate, up, down) in zip(ref_x.split(counts), references):
        ref_outputs.append(down(torch.nn.functional.silu(gate(rows)) * up(rows)))
    expected = torch.cat(ref_outputs)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for i, projections in enumerate(references):
        for weight, linear in zip((experts.w1, experts.w3, experts.w2), projections):
            torch.testing.assert_close(weight.grad[i], linear.weight.grad)
            if counts[i] == 0:
                assert torch.count_nonzero(weight.grad[i]) == 0


def test_hifloat8_experts_fail_closed_on_cpu_without_changing_state():
    # Catches a silent BF16 fallback when the user explicitly requested HIF8.
    experts = GroupedExperts(8, 16, 2, use_grouped_mm=False)
    parameters = dict(experts.named_parameters())
    keys = tuple(experts.state_dict())
    experts.hifloat8_enabled = True
    with pytest.raises(RuntimeError, match="BF16 NPU inputs"):
        experts(torch.zeros(2, 8), torch.tensor([1, 1]))
    assert tuple(experts.state_dict()) == keys
    assert all(dict(experts.named_parameters())[name] is parameter for name, parameter in parameters.items())


@pytest.mark.parametrize("wrapper", [False, True])
def test_qwen35_autoep_preserves_native_router_and_text_outputs(wrapper):
    # Catches wrong wrapper paths, top-k/dtype drift, and post-softmax logit capture.
    import copy
    from transformers import (Qwen3_5MoeTextConfig, Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM,
                              Qwen3_5MoeForConditionalGeneration)
    from deepspeed.module_inject.auto_ep import AutoEP
    from deepspeed.module_inject.auto_ep_config import parse_autoep_config

    torch.manual_seed(42)
    config = Qwen3_5MoeTextConfig(vocab_size=128,
                                  hidden_size=128,
                                  num_hidden_layers=1,
                                  num_attention_heads=2,
                                  num_key_value_heads=1,
                                  head_dim=64,
                                  moe_intermediate_size=128,
                                  shared_expert_intermediate_size=128,
                                  num_experts=4,
                                  num_experts_per_tok=2,
                                  layer_types=["full_attention"],
                                  use_cache=False)
    cls = Qwen3_5MoeForCausalLM
    if wrapper:
        config = Qwen3_5MoeConfig(text_config=config.to_dict(),
                                  vision_config={
                                      "depth": 1,
                                      "hidden_size": 32,
                                      "intermediate_size": 64,
                                      "num_heads": 4,
                                      "out_hidden_size": 128,
                                      "num_position_embeddings": 16
                                  })
        cls = Qwen3_5MoeForConditionalGeneration
    config._attn_implementation = "eager"
    reference = cls(config).to(dtype=torch.bfloat16).eval()
    candidate = copy.deepcopy(reference)
    autoep = AutoEP(
        candidate,
        parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "preset_model": "qwen3_5_moe",
            "use_grouped_mm": False
        }))
    specs = autoep.ep_parser()
    assert len(specs) == 1
    path = "model.language_model.layers.0.mlp" if wrapper else "model.layers.0.mlp"
    assert specs[0].moe_module_name == path
    original = reference.get_submodule(path)
    autoep.replace_moe_layer(specs[0], ep_size=1, ep_rank=0)
    replaced = candidate.get_submodule(path)
    x = torch.randn(13, 128, dtype=torch.bfloat16)
    logits, scores, indices = original.gate(x)
    actual_scores, actual_indices, counts = replaced.router(x)
    assert actual_scores.dtype == scores.dtype == torch.bfloat16
    torch.testing.assert_close(actual_indices, indices, rtol=0, atol=0)
    torch.testing.assert_close(actual_scores, scores, rtol=0, atol=0)
    torch.testing.assert_close(replaced._cached_router_logits, logits, rtol=0, atol=0)
    assert counts.sum() == 26
    ids = torch.tensor([[1, 5, 7, 9, 11]])
    with torch.no_grad():
        expected = reference(input_ids=ids, labels=ids)
        actual = candidate(input_ids=ids, labels=ids)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0.005)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0.02, atol=0.01)


@pytest.mark.skipif(get_accelerator().device_name() != "npu", reason="requires an available NPU")
@pytest.mark.parametrize("counts", [[64, 64], [128, 0], [0, 128], [1, 127], [0, 0], [32, 32, 32, 32], [0, 127, 0, 1],
                                    [1, 0, 0, 127], [0, 0, 0, 128], [0, 0, 0, 0], [2] * 64])
@pytest.mark.parametrize("dim,hidden_dim", [(128, 128), (2048, 512)])
@pytest.mark.parametrize("hifloat8", [False, True])
def test_npu_hifloat8_expert_forward_and_gradients(counts, dim, hidden_dim, hifloat8):
    # Catches unsupported native dispatch, inaccurate dX/dW, and empty-expert leakage.
    import torch_npu
    from torch_npu.utils.hifloat8_train import get_hifloat8_op_counts, reset_hifloat8_op_counts

    torch_npu.npu.set_device(0)
    torch.manual_seed(42)
    baseline = GroupedExperts(dim, hidden_dim, len(counts), use_grouped_mm=False).to(device="npu",
                                                                                     dtype=torch.bfloat16)
    with torch.no_grad():
        for weight in baseline.parameters():
            weight.normal_(std=0.02)
    candidate = GroupedExperts(dim, hidden_dim, len(counts)).to(device="npu", dtype=torch.bfloat16)
    candidate.load_state_dict(baseline.state_dict())
    candidate.hifloat8_enabled = hifloat8
    x = torch.randn(sum(counts), dim, device="npu", dtype=torch.bfloat16, requires_grad=True)
    candidate_x = x.detach().clone().requires_grad_()
    groups = torch.tensor(counts, device="npu", dtype=torch.int64)
    reset_hifloat8_op_counts()
    reference = baseline(x, groups)
    actual = candidate(candidate_x, groups)
    grad = torch.randn_like(reference)
    reference.backward(grad)
    actual.backward(grad)
    torch_npu.npu.synchronize()
    pairs = [(actual, reference), (candidate_x.grad, x.grad)]
    pairs.extend((actual_w.grad, ref_w.grad) for actual_w, ref_w in zip(candidate.parameters(), baseline.parameters()))
    for actual_t, reference_t in pairs:
        assert torch.isfinite(actual_t).all()
        assert torch.isfinite(reference_t).all()
        if not reference_t.numel() or reference_t.float().norm() == 0:
            assert torch.count_nonzero(actual_t) == 0
            continue
        actual_flat, reference_flat = actual_t.float().flatten(), reference_t.float().flatten()
        assert torch.nn.functional.cosine_similarity(actual_flat, reference_flat, dim=0) >= 0.99
        assert (actual_flat - reference_flat).norm() / reference_flat.norm() <= 0.15
    for index, count in enumerate(counts):
        if count == 0:
            assert all(torch.count_nonzero(weight.grad[index]) == 0 for weight in candidate.parameters())
    if sum(counts) and hifloat8:
        counters = get_hifloat8_op_counts()
        assert counters.get("grouped_backward_dw", 0) == 3
        assert counters.get("grouped_backward_dx", 0) == 3
