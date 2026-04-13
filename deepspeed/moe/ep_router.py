# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Token-choice top-K router for expert parallelism.

Ported from TorchTitan's TokenChoiceTopKRouter with adaptations for DeepSpeed.
This module is self-contained: no imports from deepspeed.module_inject
or deepspeed.runtime.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenChoiceTopKRouter(nn.Module):
    """Token-choice top-K routing for Mixture of Experts.

    Each token is routed to top-K experts based on router scores.
    Optionally supports node-limited (group-limited) routing where experts
    are divided into groups (e.g., by node), and only ``num_limited_groups``
    groups are considered before selecting top_k experts. This reduces
    cross-node communication in distributed settings.

    Args:
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each MoE layer.
        num_expert_groups (int | None): Number of expert groups for
            node-limited routing. If None, standard top-k routing is used.
            Must be a divisor of num_experts.
        num_limited_groups (int | None): Number of groups to select in
            node-limited routing. Required when num_expert_groups is set.
        top_k (int): Number of experts each token will be routed to.
        score_func (str): ``"softmax"`` or ``"sigmoid"`` scoring function.
        route_norm (bool): Whether to normalize routing scores.
        route_scale (float): Scaling factor applied to routing scores.
        gate_bias (bool): Whether to include a bias term in the gate linear.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        num_expert_groups: int | None,
        num_limited_groups: int | None,
        top_k: int,
        score_func: str,
        route_norm: bool,
        route_scale: float,
        gate_bias: bool,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=gate_bias)
        self.num_experts = num_experts
        self.num_expert_groups = num_expert_groups
        self.num_limited_groups = num_limited_groups
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        # Trainable expert score correction bias (e.g. DeepSeek-V3/Moonlight noaux_tc).
        # Separate from the dynamic load-balancing expert_bias passed in forward().
        self.e_score_correction_bias = None

    # ------------------------------------------------------------------
    # Node-limited (group-limited) routing
    # ------------------------------------------------------------------

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice: torch.Tensor,
    ) -> torch.Tensor:
        """Select ``num_limited_groups`` groups based on group scores and
        mask out experts in non-selected groups.

        Args:
            scores_for_choice: Router scores with optional expert_bias,
                shape ``(T, num_experts)``.

        Returns:
            Masked scores of the same shape, with non-selected group
            entries set to ``-inf``.
        """
        if self.num_limited_groups is None:
            raise ValueError("num_limited_groups must be set when num_expert_groups is set")
        assert self.num_expert_groups is not None
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(f"num_experts ({self.num_experts}) must be divisible by "
                             f"num_expert_groups ({self.num_expert_groups})")

        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")

        scores_grouped = scores_for_choice.view(-1, self.num_expert_groups, experts_per_group)
        # Score each group by the sum of its top-2 expert scores
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)

        # Select top groups
        _, group_idx = torch.topk(group_scores, k=self.num_limited_groups, dim=-1, sorted=False)

        # Build mask: True = masked out (non-selected groups)
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, False)

        scores_for_choice = scores_grouped.masked_fill(group_mask.unsqueeze(-1),
                                                       float("-inf")).view(-1, self.num_experts)

        return scores_for_choice

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        expert_bias: torch.Tensor | None = None,
    ) -> tuple:
        """
        Args:
            x: Input tensor of shape ``(T, dim)``.
            expert_bias: Optional bias tensor of shape ``(num_experts,)``
                used for load balancing.

        Returns:
            Tuple of:
                - top_scores ``(T, top_k)``: routing weights for selected experts.
                - selected_experts ``(T, top_k)``: expert indices per token.
                - num_tokens_per_expert ``(num_experts,)``: histogram of token counts.
        """
        # Gate projection -> (T, num_experts)
        scores = self.gate(x)

        # Scoring in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function: {self.score_func}")

        scores_for_choice = (scores if expert_bias is None else scores + expert_bias)

        # Apply pre-trained score correction bias (e.g. DeepSeek-V3 noaux_tc routing)
        if self.e_score_correction_bias is not None:
            scores_for_choice = scores_for_choice + self.e_score_correction_bias.unsqueeze(0)

        # Apply node-limited routing if configured
        if self.num_expert_groups is not None:
            scores_for_choice = self._get_node_limited_routing_scores(scores_for_choice)

        # Select top-k experts per token
        _, selected_experts_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)

        # Gather original (unbiased) scores for selected experts
        top_scores = scores.gather(dim=1, index=selected_experts_indices)

        # Optional normalization
        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator

        top_scores = top_scores * self.route_scale

        # Count tokens per expert
        # histc requires float input on CPU, so cast indices
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1).float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_scores, selected_experts_indices, num_tokens_per_expert
