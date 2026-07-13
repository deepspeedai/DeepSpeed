# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Standalone tests for AutoEP + AutoTP routed-assignment partitioning."""

import pytest
import torch

import deepspeed.comm as dist
from deepspeed.module_inject.auto_ep_layer import combine_from_routed
from deepspeed.moe.ep_tp_dispatch import (
    RoutedAssignmentPayload,
    assignment_ordinals_by_expert,
    assert_tp_payload_consistent,
    dispatch_counters,
    partition_assignments,
    restore_combined,
)
import deepspeed.moe.ep_tp_dispatch as dispatch
from unit.v1.moe.autoep_test_utils import run_cpu_gloo_test


def _payload():
    expert_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    token_indices = torch.tensor([0, 1, 2, 0, 3, 1, 2, 3, 4], dtype=torch.long)
    combine = torch.tensor([0.5, 0.25, 1.0, 0.5, 1.0, 0.75, 0.5, 0.25, 1.0])
    drop_mask = torch.zeros_like(expert_indices, dtype=torch.bool)
    pad_mask = torch.zeros_like(expert_indices, dtype=torch.bool)
    return RoutedAssignmentPayload(
        token_indices=token_indices,
        expert_indices=expert_indices,
        assignment_indices=assignment_ordinals_by_expert(expert_indices),
        capacity_slots=torch.arange(expert_indices.numel(), dtype=torch.long),
        combine_weights=combine,
        drop_mask=drop_mask,
        pad_mask=pad_mask,
        input_splits=[3, 2, 4],
        output_splits=[3, 2, 4],
        extra={
            "destination_ranks": expert_indices,
            "num_tokens": torch.tensor(5, dtype=torch.long),
        },
    )


def test_assignment_ordinals_are_stable_within_expert_segments():
    expert_indices = torch.tensor([0, 0, 2, 2, 2, 4], dtype=torch.long)
    assert assignment_ordinals_by_expert(expert_indices).tolist() == [0, 1, 0, 1, 2, 0]


def test_partition_assignments_splits_each_expert_once_across_tp_lanes():
    payload = _payload()
    local0, ctx0 = partition_assignments(payload, tp_group=None, tp_rank=0, tp_size=2)
    local1, ctx1 = partition_assignments(payload, tp_group=None, tp_rank=1, tp_size=2)

    observed = set(local0.capacity_slots.tolist()) | set(local1.capacity_slots.tolist())
    assert observed == set(range(payload.token_indices.numel()))
    assert set(local0.capacity_slots.tolist()).isdisjoint(set(local1.capacity_slots.tolist()))
    assert dispatch_counters(ctx0)["assignments_total"] == payload.token_indices.numel()
    assert dispatch_counters(ctx0)["assignments_local"] + dispatch_counters(
        ctx1)["assignments_local"] == payload.token_indices.numel()
    assert local0.input_splits == [2, 1, 2]
    assert local1.input_splits == [1, 1, 2]


def test_partition_excludes_padded_and_dropped_assignments_from_stats():
    payload = _payload()
    payload.drop_mask[1] = True
    payload.pad_mask[6] = True
    local0, ctx0 = partition_assignments(payload, tp_group=None, tp_rank=0, tp_size=1)

    assert local0.token_indices.numel() == payload.token_indices.numel() - 2
    counters = dispatch_counters(ctx0)
    assert counters["assignments_total"] == payload.token_indices.numel() - 2
    assert counters["padded"] == 1
    assert counters["dropped"] == 1


def test_restore_combined_sums_topk_assignments_by_original_token():
    payload = _payload()
    local, ctx = partition_assignments(payload, tp_group=None, tp_rank=0, tp_size=1)
    values = torch.arange(local.token_indices.numel() * 2, dtype=torch.float32).reshape(-1, 2)
    restored = restore_combined(values, ctx, tp_group=None)

    expected = torch.zeros(5, 2)
    for row, token, weight in zip(values, local.token_indices, local.combine_weights):
        expected[token] += row * weight
    assert torch.allclose(restored, expected)
    assert restored.dtype == values.dtype
    assert restored.device == values.device


def test_restore_combined_preserves_output_and_router_weight_gradients():
    payload = _payload()
    payload.combine_weights = payload.combine_weights.clone().requires_grad_(True)
    local, ctx = partition_assignments(payload, tp_group=None, tp_rank=0, tp_size=1)
    values = torch.arange(local.token_indices.numel() * 2, dtype=torch.float32).reshape(-1, 2).requires_grad_(True)

    restored = restore_combined(values, ctx, tp_group=None)
    restored.square().sum().backward()

    assert values.grad is not None
    assert values.grad.abs().sum().item() > 0
    assert payload.combine_weights.grad is not None
    assert payload.combine_weights.grad.abs().sum().item() > 0


def _tp_payload_for_backward_parity():
    top_k = 2
    token_indices_sorted = torch.tensor([0, 3, 5, 1, 2, 7, 4, 6], dtype=torch.long)
    expert_indices = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    top_scores = torch.tensor([[0.65, 0.35], [0.25, 0.75], [0.55, 0.45], [0.30, 0.70]], dtype=torch.float32)
    combine_weights = top_scores.reshape(-1).index_select(0, token_indices_sorted)
    drop_mask = torch.zeros_like(expert_indices, dtype=torch.bool)
    pad_mask = torch.zeros_like(expert_indices, dtype=torch.bool)
    return (
        RoutedAssignmentPayload(
            token_indices=(token_indices_sorted // top_k).to(torch.long),
            expert_indices=expert_indices,
            assignment_indices=assignment_ordinals_by_expert(expert_indices),
            capacity_slots=(token_indices_sorted % top_k).to(torch.long),
            combine_weights=combine_weights,
            drop_mask=drop_mask,
            pad_mask=pad_mask,
            input_splits=[2, 2, 2, 2],
            output_splits=[2, 2, 2, 2],
            extra={
                "destination_ranks": expert_indices,
                "num_tokens": torch.tensor(4, dtype=torch.long),
            },
        ),
        top_scores,
        token_indices_sorted,
    )


def _restore_combined_backward_parity_worker(rank, world_size, _shared_tmpdir):
    payload, top_scores, token_indices_sorted = _tp_payload_for_backward_parity()
    tp_group = dist.get_world_group()
    local, ctx = partition_assignments(payload, tp_group=tp_group, tp_rank=rank, tp_size=world_size)

    full_expert_output = torch.arange(payload.token_indices.numel() * 3, dtype=torch.float32).reshape(
        payload.token_indices.numel(), 3) / 11.0
    expected_expert_output = full_expert_output.clone().requires_grad_(True)
    expected_top_scores = top_scores.clone().requires_grad_(True)
    expected = combine_from_routed(expected_expert_output,
                                   top_scores=expected_top_scores,
                                   token_indices_sorted=token_indices_sorted,
                                   top_k=2,
                                   score_apply="post",
                                   combine_impl="weighted_sum",
                                   shape=(1, 4, 3))
    expected_loss = sum((expected * float(peer_rank + 1)).square().sum() for peer_rank in range(world_size))
    expected_loss.backward()
    expected_weight_grad = expected_top_scores.grad.reshape(-1).index_select(0, token_indices_sorted)

    actual_expert_output = full_expert_output.clone().requires_grad_(True)
    actual_top_scores = top_scores.clone().requires_grad_(True)
    payload.combine_weights = actual_top_scores.reshape(-1).index_select(0, token_indices_sorted)
    local_values = actual_expert_output.index_select(0, ctx.local_indices)
    restored = restore_combined(local_values, ctx, tp_group=tp_group)

    torch.testing.assert_close(restored.reshape(1, 4, 3), expected.detach(), rtol=0.0, atol=0.0)
    (restored * float(rank + 1)).square().sum().backward()

    actual_value_grad = actual_expert_output.grad.detach().clone()
    actual_top_score_grad = actual_top_scores.grad.detach().clone()
    dist.all_reduce(actual_value_grad, group=tp_group)
    dist.all_reduce(actual_top_score_grad, group=tp_group)

    torch.testing.assert_close(actual_value_grad, expected_expert_output.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_top_score_grad, expected_top_scores.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_top_score_grad.reshape(-1).index_select(0, token_indices_sorted),
                               expected_weight_grad,
                               rtol=1e-6,
                               atol=1e-6)


def test_restore_combined_tp_backward_matches_non_partitioned_combine(tmpdir):
    run_cpu_gloo_test(_restore_combined_backward_parity_worker, tmpdir, world_size=2)


def _restore_combined_topk_slot_order_worker(rank, world_size, _shared_tmpdir):
    payload = RoutedAssignmentPayload(
        token_indices=torch.tensor([0, 0, 0], dtype=torch.long),
        expert_indices=torch.tensor([0, 0, 0], dtype=torch.long),
        assignment_indices=torch.tensor([0, 1, 2], dtype=torch.long),
        capacity_slots=torch.tensor([0, 1, 2], dtype=torch.long),
        combine_weights=torch.ones(3, dtype=torch.float32),
        drop_mask=torch.zeros(3, dtype=torch.bool),
        pad_mask=torch.zeros(3, dtype=torch.bool),
        input_splits=[3],
        output_splits=[3],
        extra={
            "destination_ranks": torch.zeros(3, dtype=torch.long),
            "num_tokens": torch.tensor(1, dtype=torch.long),
        },
    )
    tp_group = dist.get_world_group()
    local, ctx = partition_assignments(payload, tp_group=tp_group, tp_rank=rank, tp_size=world_size)
    full_values = torch.tensor([[1.0e20], [1.0], [-1.0e20]], dtype=torch.float32)
    restored = restore_combined(full_values.index_select(0, ctx.local_indices), ctx, tp_group=tp_group)

    torch.testing.assert_close(restored, torch.zeros_like(restored), rtol=0.0, atol=0.0)


def test_restore_combined_tp_forward_uses_topk_slot_order(tmpdir):
    run_cpu_gloo_test(_restore_combined_topk_slot_order_worker, tmpdir, world_size=2)


def test_restore_coverage_assertion_detects_missing_assignment():
    payload = _payload()
    local, ctx = partition_assignments(payload, tp_group=None, tp_rank=0, tp_size=1)
    ctx.local_indices = ctx.local_indices[:-1]
    values = torch.ones((local.token_indices.numel() - 1, 2), dtype=torch.float32)

    restored = restore_combined(values, ctx, tp_group=None)
    assert restored.shape == (5, 2)

    with pytest.raises(RuntimeError, match="restore coverage mismatch"):
        restore_combined(values, ctx, tp_group=None, validate_coverage=True)


def test_tp_payload_consistency_detects_divergent_large_payload(monkeypatch):
    rows = 4097
    expert_indices = torch.zeros(rows, dtype=torch.long)
    payload = RoutedAssignmentPayload(
        token_indices=torch.arange(rows, dtype=torch.long),
        expert_indices=expert_indices,
        assignment_indices=torch.arange(rows, dtype=torch.long),
        capacity_slots=torch.arange(rows, dtype=torch.long),
        combine_weights=torch.ones(rows),
        drop_mask=torch.zeros(rows, dtype=torch.bool),
        pad_mask=torch.zeros(rows, dtype=torch.bool),
        input_splits=[rows],
        output_splits=[rows],
        extra={
            "destination_ranks": expert_indices,
            "num_tokens": torch.tensor(rows, dtype=torch.long),
        },
    )

    calls = []

    def fake_all_reduce(tensor, op=None, group=None):
        if not calls:
            tensor[3].add_(1)
        calls.append(op)

    monkeypatch.setattr(dispatch.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dispatch.dist, "all_reduce", fake_all_reduce)

    with pytest.raises(RuntimeError, match="routing decisions differ"):
        assert_tp_payload_consistent(payload, tp_group=object(), tp_size=2)
