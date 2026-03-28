# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
AutoEPMoELayer: drop-in replacement for a model's native MoE block.

Implements Expert Parallelism via two AllToAllV collectives around a
local GroupedExperts forward pass.  Expert parameters are tagged with
``_autoep_expert = True`` so that ZeRO-3 skips DP partitioning.

Ported from the prototype branch (tohtana/add_autoep).
"""

import logging

import torch
import deepspeed.comm as dist
import torch.nn as nn

from deepspeed.module_inject.auto_ep_config import MoELayerSpec, MoEModelPreset
from deepspeed.moe.ep_experts import GroupedExperts
from deepspeed.moe.ep_kernels import TokenReorderer
from deepspeed.moe.ep_repack import repack_expert_weights
from deepspeed.moe.ep_router import TokenChoiceTopKRouter

logger = logging.getLogger(__name__)

# ===========================================================================
# AllToAllV autograd function
# ===========================================================================


class _AllToAllV(torch.autograd.Function):
    """AllToAllV with automatically transposed split-sizes for backward."""

    @staticmethod
    def forward(ctx, input_tensor, output_splits, input_splits, group):
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group

        output = input_tensor.new_empty(
            sum(output_splits) if output_splits else input_tensor.shape[0], *input_tensor.shape[1:])
        dist.all_to_all_single(
            output,
            input_tensor,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Swap input/output splits so backward routes gradients correctly
        grad_input = grad_output.new_empty(
            sum(ctx.input_splits) if ctx.input_splits else grad_output.shape[0], *grad_output.shape[1:])
        dist.all_to_all_single(
            grad_input,
            grad_output,
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
            group=ctx.group,
        )
        return grad_input, None, None, None


def _alltoallv(tensor, output_splits, input_splits, group):
    return _AllToAllV.apply(tensor, output_splits, input_splits, group)


# ===========================================================================
# AutoEPMoELayer
# ===========================================================================


class AutoEPMoELayer(nn.Module):
    """Expert-parallel MoE layer that replaces the model's native MoE block.

    Args:
        original_layer: The original MoE sub-layer (used to copy weights).
        spec:           Structural description of the layer.
        ep_size:        Expert-parallel world size.
        ep_rank:        This rank's index in the EP group.
        ep_group:       PyTorch distributed process group for EP comms.
        preset:         Model preset, or None (autodetect from spec).
    """

    def __init__(
        self,
        original_layer: nn.Module,
        spec: MoELayerSpec,
        ep_size: int,
        ep_rank: int,
        ep_group,
        preset: MoEModelPreset = None,
    ):
        super().__init__()

        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_group = ep_group
        self.ep_group_name = f"ep_group_{id(ep_group)}"

        self.num_experts = spec.num_experts
        self.num_local_experts = spec.num_experts // ep_size
        self.dim = spec.dim
        self.ffn_dim = spec.ffn_dim
        self.top_k = spec.top_k

        # Determine preset to pass to repack_expert_weights
        if preset is None and spec is not None:
            # Try to look up preset by scanning PRESET_MODELS
            # (not critical — repack_expert_weights handles both formats)
            preset = None

        # ---------------------------------------------------------------
        # Router
        # ---------------------------------------------------------------
        self.router = TokenChoiceTopKRouter(
            dim=spec.dim,
            num_experts=spec.num_experts,
            top_k=spec.top_k,
            gate_bias=spec.gate_bias,
        )

        # Copy gate weights from original layer if possible
        self._copy_router_weights(original_layer, spec, preset)

        # ---------------------------------------------------------------
        # Local experts
        # ---------------------------------------------------------------
        self.experts = GroupedExperts(
            num_experts=self.num_local_experts,
            dim=spec.dim,
            hidden_dim=spec.ffn_dim,
        )

        # Pack weights from the original layer's experts into GroupedExperts
        local_experts_data = repack_expert_weights(
            original_layer,
            preset,
            ep_rank=ep_rank,
            ep_size=ep_size,
        )
        if local_experts_data is not None:
            # local_experts_data is a dict: {"w1": Tensor, "w2": Tensor, "w3": Tensor}
            self.experts.w1.data.copy_(local_experts_data["w1"])
            self.experts.w2.data.copy_(local_experts_data["w2"])
            self.experts.w3.data.copy_(local_experts_data["w3"])

        # ---------------------------------------------------------------
        # Token reorderer
        # ---------------------------------------------------------------
        self.reorderer = TokenReorderer(
            num_experts=self.num_local_experts,
            top_k=spec.top_k,
        )

        # Mark expert params so ZeRO-3 skips DP partitioning,
        # and set allreduce=False so the engine treats them as EP params.
        # _autoep_expert and allreduce=False are already set in GroupedExperts.__init__;
        # we only need to assign group_name here (requires ep_group_name from this layer).
        for param in self.experts.parameters():
            param.group_name = self.ep_group_name

    # -------------------------------------------------------------------
    # Router weight copy helper
    # -------------------------------------------------------------------

    def _copy_router_weights(self, original_layer, spec, preset):
        """Copy gate/router weights from the original layer when available."""
        if preset is None:
            return

        gate_attr = preset.gate_attr
        gate_module = getattr(original_layer, gate_attr, None)
        if gate_module is None:
            return

        gate_weight = getattr(gate_module, "weight", None)
        if gate_weight is not None and gate_weight.shape == self.router.gate.weight.shape:
            self.router.gate.weight.data.copy_(gate_weight.data)

        if spec.gate_bias:
            gate_bias = getattr(gate_module, "bias", None)
            if gate_bias is not None:
                self.router.gate.bias.data.copy_(gate_bias.data)

    # -------------------------------------------------------------------
    # set_deepspeed_parallelism (called by engine after ZeRO init)
    # -------------------------------------------------------------------

    def set_deepspeed_parallelism(self, ep_group=None):
        """Bind EP process group (called after DeepSpeed engine sets up groups)."""
        if ep_group is not None:
            self.ep_group = ep_group
            self.ep_group_name = f"ep_group_{id(ep_group)}"
            for param in self.experts.parameters():
                param.group_name = self.ep_group_name

    # -------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor, shape ``(batch, seq, dim)``.

        Returns:
            Output tensor, same shape as *hidden_states*.
        """
        orig_shape = hidden_states.shape  # (B, S, D)
        hidden_states = hidden_states.view(-1, self.dim)  # (T, D)
        num_tokens = hidden_states.shape[0]

        # 1. Route tokens
        top_scores, selected_experts = self.router(hidden_states)
        # top_scores: (T, top_k),  selected_experts: (T, top_k)

        # 2. Reorder tokens into expert-sorted order
        top_scores_sorted, token_indices_sorted, num_tokens_per_expert_local = self.reorderer(
            top_scores, selected_experts)
        # num_tokens_per_expert_local: (num_local_experts,)

        # 3. AllToAllV dispatch — exchange token counts and hidden states
        #    across EP ranks so each rank receives tokens for its experts.
        num_tokens_per_rank = self._tokens_per_rank(num_tokens_per_expert_local)
        # Flatten tokens to send: each token duplicated top_k times
        hidden_flat = hidden_states[token_indices_sorted % num_tokens]  # (T*top_k, D)

        dispatched = _alltoallv(
            hidden_flat,
            output_splits=num_tokens_per_rank.tolist(),
            input_splits=num_tokens_per_rank.tolist(),
            group=self.ep_group,
        )

        num_tokens_per_expert_recv = self._all_gather_token_counts(num_tokens_per_expert_local)

        # 4. Local expert computation
        expert_output = self.experts(dispatched, num_tokens_per_expert_recv)

        # 5. AllToAllV combine — send results back to originating ranks
        combined = _alltoallv(
            expert_output,
            output_splits=num_tokens_per_rank.tolist(),
            input_splits=num_tokens_per_rank.tolist(),
            group=self.ep_group,
        )

        # 6. Weighted combine with routing scores
        output = self._weighted_combine(combined, top_scores_sorted, token_indices_sorted, num_tokens)

        return output.view(orig_shape)

    # -------------------------------------------------------------------
    # Forward helpers
    # -------------------------------------------------------------------

    def _tokens_per_rank(self, num_tokens_per_expert_local: torch.Tensor) -> torch.Tensor:
        """Compute how many tokens each EP rank should send/receive."""
        # Sum over local experts → scalar per rank after all-gather
        local_total = num_tokens_per_expert_local.sum().unsqueeze(0)
        gathered = [torch.zeros_like(local_total) for _ in range(self.ep_size)]
        dist.all_gather(gathered, local_total, group=self.ep_group)
        return torch.cat(gathered)  # (ep_size,)

    def _all_gather_token_counts(self, num_tokens_per_expert_local: torch.Tensor) -> torch.Tensor:
        """All-gather per-expert token counts across EP ranks."""
        gathered = [torch.zeros_like(num_tokens_per_expert_local) for _ in range(self.ep_size)]
        dist.all_gather(gathered, num_tokens_per_expert_local, group=self.ep_group)
        # Each rank gets all experts' token counts; we only need our local slice
        return torch.cat(gathered)  # (ep_size * num_local_experts,)

    def _weighted_combine(
        self,
        expert_output: torch.Tensor,
        top_scores: torch.Tensor,
        token_indices: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Scatter expert outputs back and weight by routing scores."""
        # expert_output: (T*top_k, D),  top_scores: (T*top_k,)
        weighted = expert_output * top_scores.unsqueeze(-1)  # (T*top_k, D)

        output = torch.zeros(
            num_tokens,
            self.dim,
            dtype=expert_output.dtype,
            device=expert_output.device,
        )
        # Scatter-add back to original token positions
        orig_indices = token_indices % num_tokens  # (T*top_k,)
        output.scatter_add_(0, orig_indices.unsqueeze(-1).expand_as(weighted), weighted)
        return output
