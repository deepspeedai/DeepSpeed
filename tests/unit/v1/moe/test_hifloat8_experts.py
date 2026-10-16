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


def test_npu_expert_helper_preserves_upstream_activation(monkeypatch):
    """The NPU GEMM dispatch must use AutoEP's resolved expert activation."""
    # Pin the merge regression without NPU hardware: the stub implements the
    # native grouped GEMM contract while the helper supplies activation policy.
    import deepspeed.moe.ep_experts as ep_experts

    def fake_grouped_mm(lhs, weight, offsets, *, trans_b=False):
        weight = weight.transpose(-1, -2) if trans_b else weight
        starts = [0] + offsets.tolist()[:-1]
        return torch.cat([lhs[start:end] @ weight[i] for i, (start, end) in enumerate(zip(starts, offsets.tolist()))])

    torch.manual_seed(7)
    counts = torch.tensor([2, 1])
    x = torch.randn(4, 8, requires_grad=True)
    weights = [torch.randn(shape, requires_grad=True) for shape in ((2, 16, 8), (2, 8, 16), (2, 16, 8))]
    w1, w2, w3 = weights
    actual = ep_experts._run_experts_with_grouped_mm(w1,
                                                     w2,
                                                     w3,
                                                     x,
                                                     counts,
                                                     "swiglu_clamped",
                                                     limit=0.5,
                                                     grouped_mm=fake_grouped_mm)

    x_ref = x.detach().clone().requires_grad_()
    w1_ref, w2_ref, w3_ref = [w.detach().clone().requires_grad_() for w in weights]
    pieces = []
    start = 0
    for i, count in enumerate(counts.tolist()):
        rows = x_ref[start:start + count]
        gate = rows @ w1_ref[i].T
        up = rows @ w3_ref[i].T
        hidden = torch.nn.functional.silu(gate.clamp(max=0.5)) * up.clamp(min=-0.5, max=0.5)
        pieces.append(hidden @ w2_ref[i].T)
        start += count
    expected = torch.cat((torch.cat(pieces), x_ref.new_zeros((1, 8))))
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    for value, reference in ((x, x_ref), (w1, w1_ref), (w2, w2_ref), (w3, w3_ref)):
        torch.testing.assert_close(value.grad, reference.grad)


@pytest.mark.skipif(get_accelerator().device_name() != "npu", reason="requires an available NPU")
@pytest.mark.parametrize("counts", [[64, 64], [128, 0], [0, 128], [1, 127], [0, 0], [32, 32, 32, 32], [0, 127, 0, 1],
                                    [1, 0, 0, 127], [0, 0, 0, 128], [0, 0, 0, 0], [2] * 64])
@pytest.mark.parametrize("dim,hidden_dim", [(128, 128), (2048, 512)])
@pytest.mark.parametrize("backend", ["bf16", "torchao"])
def test_npu_hifloat8_expert_forward_and_gradients(counts, dim, hidden_dim, backend):
    # Catches unsupported native dispatch, inaccurate dX/dW, and empty-expert leakage.
    import torch_npu

    torch_npu.npu.set_device(0)
    torch.manual_seed(42)
    baseline = GroupedExperts(dim, hidden_dim, len(counts), use_grouped_mm=False).to(device="npu",
                                                                                     dtype=torch.bfloat16)
    with torch.no_grad():
        for weight in baseline.parameters():
            weight.normal_(std=0.02)
    candidate = GroupedExperts(dim, hidden_dim, len(counts)).to(device="npu", dtype=torch.bfloat16)
    candidate.load_state_dict(baseline.state_dict())
    if backend == "torchao":
        from torchao.quantization import quantize_
        from deepspeed.runtime.torchao import GroupedExpertsConfig

        provider = get_accelerator().get_hifloat8_provider()
        policy = provider.HiFloat8Config()
        quantize_(candidate, GroupedExpertsConfig(policy), filter_fn=lambda module, name: name == "")
    x = torch.randn(sum(counts), dim, device="npu", dtype=torch.bfloat16, requires_grad=True)
    candidate_x = x.detach().clone().requires_grad_()
    groups = torch.tensor(counts, device="npu", dtype=torch.int64)
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


@pytest.mark.skipif(get_accelerator().device_name() != "npu", reason="requires an available NPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("input_grad,weight_grad", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("counts", [[0, 8, 0, 4], [0, 0, 0, 0]])
def test_npu_native_grouped_mm_matches_matmul_with_partial_gradients(dtype, trans_b, input_grad, weight_grad, counts):
    # Catches saved-tensor ordering errors, packed dW orientation and empty-group leakage.
    from deepspeed.ops.npu.grouped_mm import grouped_mm

    torch.npu.set_device(0)
    torch.manual_seed(71)
    inner, columns = 64, 32
    shape = (len(counts), columns, inner) if trans_b else (len(counts), inner, columns)
    x = (torch.randn(sum(counts), inner, device="npu", dtype=dtype) * 0.1).requires_grad_(input_grad)
    weight = (torch.randn(shape, device="npu", dtype=dtype) * 0.1).requires_grad_(weight_grad)
    ref_x = x.detach().clone().requires_grad_(input_grad)
    ref_weight = weight.detach().clone().requires_grad_(weight_grad)
    offsets = torch.tensor(counts, device="npu", dtype=torch.int64).cumsum(0)
    actual = grouped_mm(x, weight, offsets, trans_b=trans_b)

    pieces = []
    for index, rows in enumerate(ref_x.split(counts)):
        right = ref_weight[index].T if trans_b else ref_weight[index]
        pieces.append(rows @ right)
    expected = torch.cat(pieces)
    grad = torch.randn_like(expected) * 0.1
    actual.backward(grad)
    expected.backward(grad)
    torch.npu.synchronize()

    pairs = [(actual, expected)]
    for value, reference, needs_grad in ((x, ref_x, input_grad), (weight, ref_weight, weight_grad)):
        if needs_grad:
            assert value.grad is not None and reference.grad is not None
            pairs.append((value.grad, reference.grad))
        else:
            assert value.grad is None and reference.grad is None
    rtol, atol = (0.02, 0.0005) if dtype == torch.bfloat16 else (0.003, 0.00005)
    for value, reference in pairs:
        assert value.dtype == reference.dtype == dtype
        assert bool(torch.isfinite(value).all())
        torch.testing.assert_close(value, reference, rtol=rtol, atol=atol)
    if weight_grad:
        for index, count in enumerate(counts):
            if count == 0:
                assert torch.count_nonzero(weight.grad[index]) == 0


@pytest.mark.skipif(get_accelerator().device_name() != "npu", reason="requires an available NPU")
def test_npu_provider_experts_preserve_bfloat16_residual_with_float16_compute():
    # A working-dtype leak promotes the residual and breaks the BF16-facing module contract.
    torch.npu.set_device(0)
    torch.manual_seed(73)
    experts = GroupedExperts(64, 64, 4).to(device="npu", dtype=torch.bfloat16)
    projection = torch.nn.Linear(64, 32).to(device="npu", dtype=torch.bfloat16)
    with torch.no_grad():
        for weight in experts.parameters():
            weight.normal_(std=0.02)
    parameters = dict(experts.named_parameters())
    state_keys = tuple(experts.state_dict())
    from torchao.quantization import quantize_
    from deepspeed.runtime.torchao import GroupedExpertsConfig

    provider = get_accelerator().get_hifloat8_provider()
    policy = provider.HiFloat8Config.from_dict({"compute_dtype": "float16"})
    quantize_(experts, GroupedExpertsConfig(policy), filter_fn=lambda module, name: name == "")
    counts = torch.tensor([0, 8, 0, 4], device="npu", dtype=torch.int64)
    x = torch.randn(15, 64, device="npu", dtype=torch.bfloat16, requires_grad=True)
    branch = experts(x, counts)
    residual = x + branch
    output = projection(residual)
    assert output.dtype == branch.dtype == residual.dtype == x.dtype
    assert torch.count_nonzero(branch[12:]) == 0
    output.float().square().mean().backward()
    torch.npu.synchronize()

    assert x.grad.dtype == torch.bfloat16 and bool(torch.isfinite(x.grad).all())
    assert tuple(experts.state_dict()) == state_keys
    for name, parameter in experts.named_parameters():
        assert parameter is parameters[name]
        assert parameter.dtype == parameter.grad.dtype == torch.bfloat16
        assert bool(torch.isfinite(parameter.grad).all())
        assert torch.count_nonzero(parameter.grad[0]) == torch.count_nonzero(parameter.grad[2]) == 0
    for parameter in projection.parameters():
        assert parameter.dtype == parameter.grad.dtype == torch.bfloat16
        assert bool(torch.isfinite(parameter.grad).all())


def test_grouped_override_rejects_explicit_fused_gate_up():
    # Upstream can request a fused gate/up path; a three-GEMM override cannot
    # silently ignore that public choice after this integration is merged.
    experts = GroupedExperts(8, 16, 2, use_grouped_mm=False)
    experts.gate_up_impl = "fused"

    def unexpected_kernel(*args, **kwargs):
        raise AssertionError("Unsupported execution must fail before a GEMM")

    experts.grouped_mm = unexpected_kernel
    with pytest.raises(RuntimeError, match="separate gate/up"):
        experts(torch.randn(2, 8), torch.tensor([1, 1]))
