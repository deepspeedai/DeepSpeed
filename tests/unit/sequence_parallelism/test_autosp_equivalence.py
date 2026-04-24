# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Numerical equivalence tests for AutoSP multimodal sequence parallelism.

Each test verifies that running the SP-wrapped path across N ranks produces
the same result as the equivalent single-device (non-SP) computation.

These tests require 2 GPUs and the NCCL backend.
Run with:

    deepspeed --num_gpus 2 --no_local_rank --module pytest tests/unit/sequence_parallelism/test_autosp_equivalence.py -v
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

import deepspeed.comm as dist
from deepspeed.sequence.autosp_vit import UlyssesSPViTAttention
from deepspeed.sequence.autosp_fusion import LlavaFusionAdapter

from unit.common import DistributedTest

# ---------------------------------------------------------------------------
# Shared identity attention — deterministic, easy to verify
# ---------------------------------------------------------------------------

_IMAGE_TOKEN_ID = -200


class _IdentityAttn(nn.Module):
    """Returns hidden_states unchanged so that gather-compute-scatter is a no-op."""

    def forward(self, hidden_states, **kwargs):
        return hidden_states


# ---------------------------------------------------------------------------
# UlyssesSPViTAttention equivalence
# ---------------------------------------------------------------------------


class TestViTSPEquivalence(DistributedTest):
    """SP-wrapped ViT attention with an identity inner module must reproduce
    the unsharded output on every rank."""

    world_size = 2

    @pytest.mark.parametrize("has_cls_token", [True, False])
    @pytest.mark.parametrize("num_patches", [8, 12])
    def test_output_equals_single_device(self, has_cls_token, num_patches):
        """Each rank's local output slice must match the corresponding slice of
        the single-device output."""
        sp_group = dist.new_group(ranks=list(range(self.world_size)))
        rank = dist.get_rank(sp_group)
        bs, hidden = 2, 32

        # --- Single-device reference ---
        # Build the full input (all ranks see the same RNG seed so the tensor
        # is identical everywhere).
        torch.manual_seed(42)
        if has_cls_token:
            full_input = torch.randn(bs, 1 + num_patches, hidden).cuda()
        else:
            full_input = torch.randn(bs, num_patches, hidden).cuda()

        identity = _IdentityAttn().cuda()
        # Single-device path is just identity — output == input.
        ref_out = identity(full_input)

        # --- SP path ---
        local_patches = num_patches // self.world_size
        if has_cls_token:
            cls = full_input[:, :1, :]
            patch_slice = full_input[:, 1 + rank * local_patches:1 + (rank + 1) * local_patches, :]
            local_input = torch.cat([cls, patch_slice], dim=1)
        else:
            local_input = full_input[:, rank * local_patches:(rank + 1) * local_patches, :]

        wrapper = UlyssesSPViTAttention(_IdentityAttn().cuda(), sp_group, has_cls_token=has_cls_token).cuda()
        sp_out = wrapper(local_input)

        # --- Compare ---
        # sp_out is the local slice; reconstruct what slice of ref_out it maps to.
        if has_cls_token:
            ref_slice = torch.cat(
                [ref_out[:, :1, :], ref_out[:, 1 + rank * local_patches:1 + (rank + 1) * local_patches, :]], dim=1)
        else:
            ref_slice = ref_out[:, rank * local_patches:(rank + 1) * local_patches, :]

        assert torch.allclose(sp_out, ref_slice,
                              atol=1e-5), (f"rank={rank} sp_out differs from reference: "
                                           f"max_diff={( sp_out - ref_slice).abs().max().item():.2e}")


# ---------------------------------------------------------------------------
# LlavaFusionAdapter equivalence
# ---------------------------------------------------------------------------


class TestLlavaFusionEquivalence(DistributedTest):
    """Verifies that the SP gather/scatter in LlavaFusionAdapter is a lossless
    round-trip: concatenating all ranks' output shards reproduces the full
    fused sequence that single-device splicing would produce."""

    world_size = 2

    def _build_inputs(self, bs, local_v, text_len, hidden, rank):
        """Build deterministic visual and text tensors identical on every rank."""
        torch.manual_seed(0)
        # Each rank holds a contiguous slice of the visual tokens.
        full_visual = torch.randn(bs, local_v * self.world_size, hidden).cuda()
        text = torch.randn(bs, text_len, hidden).cuda()
        ids = torch.zeros(bs, text_len, dtype=torch.long).cuda()
        ids[:, 1] = _IMAGE_TOKEN_ID  # one image placeholder at position 1
        local_visual = full_visual[:, rank * local_v:(rank + 1) * local_v, :]
        return full_visual, local_visual, text, ids

    def test_shards_reassemble_to_full_fused(self):
        """Gathering all ranks' output shards must equal the single-device
        fused sequence (modulo padding zeros)."""
        sp_group = dist.new_group(ranks=list(range(self.world_size)))
        rank = dist.get_rank(sp_group)

        bs, local_v, text_len, hidden = 1, 4, 6, 8
        full_visual, local_visual, text, ids = self._build_inputs(bs, local_v, text_len, hidden, rank)

        # --- SP path: each rank gets one shard ---
        adapter = LlavaFusionAdapter(nn.Identity(), sp_group, image_token_id=_IMAGE_TOKEN_ID).cuda()
        local_out = adapter(local_visual, text, ids)  # [bs, local_fused, hidden]

        # Gather all shards onto every rank so we can compare globally.
        gathered = [torch.zeros_like(local_out) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_out, group=sp_group)
        full_sp_out = torch.cat(gathered, dim=1)  # [bs, padded_fused, hidden]

        # --- Single-device reference ---
        # Simulate what a non-SP LlavaFusionAdapter would produce: project the
        # full visual tensor (identity here) and splice once.
        ref_adapter = LlavaFusionAdapter(nn.Identity(), sp_group, image_token_id=_IMAGE_TOKEN_ID).cuda()
        # Call _splice_visual_into_text directly so we bypass the SP scatter.
        ref_fused = ref_adapter._splice_visual_into_text(text, full_visual, ids)

        # Pad reference to the same padded length.
        fused_len = ref_fused.shape[1]
        pad = (self.world_size - fused_len % self.world_size) % self.world_size
        if pad > 0:
            ref_fused = F.pad(ref_fused, (0, 0, 0, pad))

        assert torch.allclose(full_sp_out, ref_fused,
                              atol=1e-5), (f"rank={rank} reassembled SP output differs from reference: "
                                           f"max_diff={( full_sp_out - ref_fused).abs().max().item():.2e}")

    def test_no_image_token_passthrough(self):
        """When there are no image placeholders the SP fused output must equal
        the sharded text after padding/scatter (all-text path)."""
        sp_group = dist.new_group(ranks=list(range(self.world_size)))
        rank = dist.get_rank(sp_group)

        bs, local_v, text_len, hidden = 1, 2, 8, 4
        torch.manual_seed(1)
        local_visual = torch.randn(bs, local_v, hidden).cuda()
        text = torch.randn(bs, text_len, hidden).cuda()
        ids = torch.zeros(bs, text_len, dtype=torch.long).cuda()  # no image placeholder

        adapter = LlavaFusionAdapter(nn.Identity(), sp_group, image_token_id=_IMAGE_TOKEN_ID).cuda()
        local_out = adapter(local_visual, text, ids)

        # Gather shards and strip the padding slice from visual gather.
        gathered = [torch.zeros_like(local_out) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_out, group=sp_group)
        full_sp_out = torch.cat(gathered, dim=1)

        # Expected: when there is no image token, the visual tokens are ignored.
        # So the fused output should just be the text tokens.
        ref_fused = text
        pad = (self.world_size - ref_fused.shape[1] % self.world_size) % self.world_size
        if pad > 0:
            ref_fused = F.pad(ref_fused, (0, 0, 0, pad))

        assert torch.allclose(full_sp_out, ref_fused,
                              atol=1e-5), (f"rank={rank} no-image path differs from reference: "
                                           f"max_diff={( full_sp_out - ref_fused).abs().max().item():.2e}")
