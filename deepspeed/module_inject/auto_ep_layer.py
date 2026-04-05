# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP MoE Layer: drop-in replacement for HF MoE blocks with EP support.

Contains AutoEPMoELayer, compute_split_plan, _AllToAllV, and helper functions.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
import torch.nn as nn
import deepspeed.comm as dist
from deepspeed.module_inject.auto_ep_config import AutoEPConfig, MoELayerSpec
from deepspeed.utils import logger
from deepspeed.moe.ep_router import TokenChoiceTopKRouter
from deepspeed.moe.ep_count import count_tokens_per_expert
from deepspeed.moe.ep_experts import GroupedExperts
from deepspeed.moe.ep_kernels import TokenReorderer
from deepspeed.moe.ep_repack import repack_expert_weights

# ---------------------------------------------------------------------------
# Named tuples
# ---------------------------------------------------------------------------


class RouterOutput(NamedTuple):
    top_scores: torch.Tensor  # [T, K]
    selected_experts: torch.Tensor  # [T, K]
    num_tokens_per_expert: torch.Tensor  # [E_global]


class SplitPlan(NamedTuple):
    input_splits: list[int]  # len=ep_size
    output_splits: list[int]  # len=ep_size
    local_counts: torch.Tensor  # [E_local]
    local_counts_by_source: torch.Tensor  # [ep_size, E_local]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def resolve_score_apply_mode(
    spec: MoELayerSpec,
    config_override: Literal["auto", "pre", "post"],
) -> Literal["pre", "post"]:
    """Resolve score-application mode from config override or preset default."""
    if config_override != "auto":
        return config_override
    return spec.score_apply


def apply_scores_before_experts_if_enabled(
    routed_input: torch.Tensor,
    top_scores: torch.Tensor,
    score_apply: Literal["pre", "post"],
) -> torch.Tensor:
    """Pre-multiply token representations by router scores before expert compute."""
    if score_apply == "pre":
        return (routed_input.to(torch.float32) * top_scores.reshape(-1, 1)).to(routed_input.dtype)
    return routed_input


def compute_split_plan(
    selected_experts: torch.Tensor,  # [T, K]
    num_experts: int,
    ep_size: int,
    num_local_experts: int,
    ep_group: dist.ProcessGroup | None,
) -> SplitPlan:
    """Compute AllToAllV split sizes for token dispatch/combine.

    Returns SplitPlan with input_splits, output_splits, local_counts, and
    local_counts_by_source.
    """
    T_K = selected_experts.numel()

    if ep_size == 1:
        # No dispatch needed - all tokens stay local
        num_tokens_per_expert = count_tokens_per_expert(
            selected_experts,
            num_experts,
            out_dtype=torch.int32,
        )
        return SplitPlan(
            input_splits=[T_K],
            output_splits=[T_K],
            local_counts=num_tokens_per_expert,
            local_counts_by_source=num_tokens_per_expert.view(1, num_local_experts),
        )

    # Count tokens per expert globally
    num_tokens_per_expert = count_tokens_per_expert(
        selected_experts,
        num_experts,
        out_dtype=torch.int32,
    )

    # Reshape to [ep_size, num_local_experts] to get per-rank counts
    count_matrix = num_tokens_per_expert.view(ep_size, num_local_experts)

    # input_splits: how many tokens THIS rank sends to each destination rank
    input_splits = count_matrix.sum(dim=1).cpu().tolist()

    # Exchange counts with all ranks to get output_splits
    # Each rank tells every other rank how many tokens it will send
    local_counts_tensor = count_matrix.sum(dim=1).clone()  # [ep_size]
    remote_counts_tensor = torch.zeros_like(local_counts_tensor)

    dist.all_to_all_single(
        remote_counts_tensor,
        local_counts_tensor,
        group=ep_group,
    )
    output_splits = remote_counts_tensor.cpu().tolist()

    # local_counts: how many tokens this rank will process for each local expert
    # After receiving tokens, we need per-expert counts for this rank
    local_expert_counts = count_matrix[:, :].clone()  # [ep_size, E_local]

    # Exchange the detailed per-expert counts
    # Each rank needs to know, for its local experts, how many tokens come from each source
    local_expert_counts_flat = local_expert_counts.view(-1).contiguous()  # [ep_size * E_local]
    received_counts_flat = torch.zeros_like(local_expert_counts_flat)

    dist.all_to_all_single(
        received_counts_flat,
        local_expert_counts_flat,
        group=ep_group,
    )

    # Sum over source ranks to get total per local expert
    received_counts = received_counts_flat.view(ep_size, num_local_experts)
    local_counts = received_counts.sum(dim=0)  # [E_local]

    return SplitPlan(
        input_splits=input_splits,
        output_splits=output_splits,
        local_counts=local_counts,
        local_counts_by_source=received_counts,
    )


class _AllToAllV(torch.autograd.Function):
    """Autograd-compatible all-to-all with variable split sizes."""

    @staticmethod
    def forward(ctx, group, x, input_splits, output_splits):
        ctx.group = group
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits

        output_size = sum(output_splits)
        output = torch.empty(
            (output_size, x.shape[1]),
            dtype=x.dtype,
            device=x.device,
        )

        dist.all_to_all_single(
            output,
            x.contiguous(),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_out):
        # Reverse the splits for backward
        grad_out = grad_out.contiguous()
        input_size = sum(ctx.input_splits)
        grad_input = torch.empty(
            (input_size, grad_out.shape[1]),
            dtype=grad_out.dtype,
            device=grad_out.device,
        )

        dist.all_to_all_single(
            grad_input,
            grad_out,
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
            group=ctx.group,
        )
        return None, grad_input, None, None


def permute_by_local_expert(
    tokens: torch.Tensor,
    local_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Reorder tokens so they are grouped contiguously by local expert ID.

    Uses TorchTitan's Triton kernel for permutation index generation.

    Returns:
        tokens_permuted: [N_padded, H] (alignment-padded)
        permuted_indices: [N_padded] (maps padded positions -> original positions)
        aligned_counts: [E_local] aligned token counts per expert (for expert computation)
        n_tokens: original token count before padding (for unpermute)
    """
    from deepspeed.moe.ep_kernels import generate_permute_indices, TOKEN_GROUP_ALIGN_SIZE_M

    if local_counts.ndim == 1:
        # [E_local]: already aggregated over sources (ep_degree=1)
        ep_degree = 1
        num_local_experts = local_counts.shape[0]
        local_counts_flat = local_counts
    elif local_counts.ndim == 2:
        # [ep_size, E_local]: preserve per-source layout for correct regrouping
        ep_degree, num_local_experts = local_counts.shape
        local_counts_flat = local_counts.reshape(-1)
    else:
        raise ValueError(
            f"local_counts must have shape [E_local] or [ep_degree, E_local], got {tuple(local_counts.shape)}")

    n_tokens = tokens.shape[0]
    alignment = TOKEN_GROUP_ALIGN_SIZE_M

    # Compute padded max length
    x_padded_per_expert = n_tokens + num_local_experts * alignment
    padded_max_len = ((x_padded_per_expert + alignment - 1) // alignment) * alignment

    # Use the pure-PyTorch path for host tensors. The CPU accelerator reports
    # CPU tensors as "on accelerator", but Triton still requires a GPU driver.
    use_cpu = tokens.device.type == "cpu"
    counts_for_permute = local_counts_flat.cpu() if use_cpu else local_counts_flat
    with torch.no_grad():
        permuted_indices, m_sizes, _offsets = generate_permute_indices(
            counts_for_permute,
            num_local_experts,
            ep_degree,
            padded_max_len,
            alignment,
            use_cpu=use_cpu,
        )
    if not use_cpu:
        permuted_indices = permuted_indices.to(tokens.device)
        m_sizes = m_sizes.to(tokens.device)

    # Add padding row for out-of-bounds indices (index n_tokens -> zero row)
    tokens_padded = torch.vstack((tokens, tokens.new_zeros((tokens.shape[-1], ))))
    tokens_permuted = tokens_padded[permuted_indices, :]

    return tokens_permuted, permuted_indices, m_sizes, n_tokens


def unpermute_by_local_expert(
    expert_output: torch.Tensor,
    permuted_indices: torch.Tensor,
    n_tokens: int,
) -> torch.Tensor:
    """Reverse permute_by_local_expert: restore original token order and strip padding.

    Args:
        expert_output: [N_padded, H] from expert computation
        permuted_indices: [N_padded] index mapping from permute_by_local_expert
        n_tokens: original token count before alignment padding
    """
    # Scatter expert outputs back to original positions.
    # permuted_indices values range 0..n_tokens, where n_tokens is the zero-padding row.
    out_unpermuted = expert_output.new_zeros((n_tokens + 1, expert_output.shape[-1]))
    out_unpermuted[permuted_indices, :] = expert_output
    # Strip the zero-padding row to get [n_tokens, H]
    return out_unpermuted[:-1]


def combine_from_routed(
        expert_output: torch.Tensor,  # [N, H]
        top_scores: torch.Tensor,  # [T, K]
        token_indices_sorted: torch.Tensor,  # [N]
        top_k: int,
        score_apply: Literal["pre", "post"],
        shape: tuple[int, int, int],  # (B, S, H)
) -> torch.Tensor:
    """Scatter-add expert outputs back to original token positions."""
    bsz, seqlen, hdim = shape
    T = bsz * seqlen

    # Create output tensor
    output = torch.zeros(T * top_k, hdim, dtype=expert_output.dtype, device=expert_output.device)

    # Place expert outputs back in unsorted order
    output[token_indices_sorted] = expert_output

    # Reshape to [T, K, H]
    output = output.reshape(T, top_k, hdim)

    if score_apply == "post":
        # Apply scores during combine
        output = (torch.bmm(
            top_scores.reshape(-1, 1, top_k).float(),
            output.float(),
        ).to(expert_output.dtype).squeeze(1))
    else:
        # Scores already applied pre-experts, just sum over top_k
        output = output.sum(dim=1)

    return output.reshape(bsz, seqlen, hdim)


# ---------------------------------------------------------------------------
# AutoEPMoELayer
# ---------------------------------------------------------------------------


class AutoEPMoELayer(nn.Module):
    """Drop-in replacement for HF MoE blocks with Expert Parallelism support."""

    _is_autoep_layer = True  # Marker for AutoTP skip handshake

    def __init__(
        self,
        spec: MoELayerSpec,
        source_module: nn.Module,
        ep_size: int,
        ep_rank: int,
        config: AutoEPConfig,
    ) -> None:
        super().__init__()

        self.model_family = spec.model_family
        self.return_router_logits = spec.return_router_logits
        self.router_logits_capture_target = spec.router_logits_capture_target
        self.router_logits_capture_index = spec.router_logits_capture_index
        self.top_k = spec.top_k
        self.score_apply = resolve_score_apply_mode(spec, config.score_apply)
        route_norm = spec.route_norm if config.route_norm is None else config.route_norm
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_experts = spec.num_experts
        self.num_local_experts = spec.num_experts // ep_size
        self.hidden_size = spec.hidden_size
        self.ep_group_name = f"ep_size_{ep_size}"
        self.ep_group = None  # Set by set_deepspeed_parallelism()

        # Router: copy gate weights from source
        source_gate = getattr(source_module, spec.router_name)
        self.router = TokenChoiceTopKRouter(
            dim=spec.hidden_size,
            num_experts=spec.num_experts,
            num_expert_groups=config.num_expert_groups,
            num_limited_groups=config.num_limited_groups,
            top_k=spec.top_k,
            score_func=spec.score_func,
            route_norm=route_norm,
            route_scale=config.route_scale,
            gate_bias=spec.gate_bias,
        )
        # Copy gate weights
        self.router.gate.weight.data.copy_(source_gate.weight.data)
        if spec.gate_bias and getattr(source_gate, 'bias', None) is not None:
            self.router.gate.bias.data.copy_(source_gate.bias.data)

        # Alias router under the name OutputRecorder expects (layer_name if provided),
        # but only when OutputRecorder captures from the router child and the alias is safe.
        alias_target = spec.router_logits_capture_layer_name or spec.router_name
        if spec.router_logits_capture_target == "router" and alias_target != "router":
            if "." in alias_target or alias_target in ("experts", "shared_experts") or hasattr(self, alias_target):
                logger.warning(f"Skipping router alias '{alias_target}' to avoid name collision.")
            else:
                setattr(self, alias_target, self.router)

        # Experts: extract local expert weights
        w1, w2, w3 = repack_expert_weights(
            experts_source=getattr(source_module, spec.experts_name),
            spec=spec,
            ep_rank=ep_rank,
            ep_size=ep_size,
        )
        self.experts = GroupedExperts(
            dim=spec.hidden_size,
            hidden_dim=spec.ffn_hidden_size,
            num_experts=self.num_local_experts,
            use_grouped_mm=config.use_grouped_mm,
        )
        self.experts.w1.data.copy_(w1)
        self.experts.w2.data.copy_(w2)
        self.experts.w3.data.copy_(w3)

        self.reorderer = TokenReorderer(num_experts=self.num_experts, top_k=self.top_k)
        self.shared_experts = getattr(source_module, spec.shared_experts_name,
                                      None) if spec.has_shared_experts else None

        # Mark expert params for EDP gradient reduction
        for param in self.experts.parameters():
            param.allreduce = False
            param.group_name = self.ep_group_name

        # Mark shared expert and router params for global DP reduction
        for param in self.router.parameters():
            param.allreduce = True
        if self.shared_experts is not None:
            for param in self.shared_experts.parameters():
                param.allreduce = True

        # Load balancing buffers
        self.load_balance_coeff = config.load_balance_coeff
        buf_device = source_gate.weight.device
        if self.load_balance_coeff is not None:
            self.register_buffer(
                "expert_bias",
                torch.zeros(spec.num_experts, dtype=torch.float32, device=buf_device),
                persistent=True,
            )
        else:
            self.expert_bias = None
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(spec.num_experts, dtype=torch.float32, device=buf_device),
            persistent=False,
        )

        # Router-logit cache
        self._cached_router_logits = None
        self._register_logit_hook()

    def _register_logit_hook(self):
        """Register a forward hook that caches gate logits for OutputRecorder capture."""
        if self.router_logits_capture_target != "router":
            return

        def hook_fn(module, input, output):
            x = input[0]  # [T, H]
            logits = module.gate(x)  # [T, E_global]
            # Apply activation for HF semantic parity
            if self.router.score_func == "softmax":
                logits = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
            elif self.router.score_func == "sigmoid":
                logits = torch.sigmoid(logits.float()).to(logits.dtype)
            self._cached_router_logits = logits

        self.router.register_forward_hook(hook_fn)

    def set_deepspeed_parallelism(
        self,
        use_data_before_expert_parallel_: bool = False,
    ) -> None:
        """Bind EP group handle to this module."""
        from deepspeed.utils import groups
        from deepspeed.utils.bwc import bwc_pipeline_parallel_world_size

        if self.ep_group_name not in groups._get_expert_parallel_group_dict():
            mp_size = max(
                getattr(groups, '_get_model_parallel_world_size', lambda: 1)(),
                getattr(groups, '_get_sequence_parallel_world_size', lambda: 1)(),
            )
            mp_mode = "tp" if getattr(groups, '_get_model_parallel_world_size', lambda: 1)() > 1 else "sp"
            pp_size = 1 if groups.mpu is None else bwc_pipeline_parallel_world_size(groups.mpu)
            groups._create_expert_and_data_parallel(
                expert_parallel_size_=self.ep_size,
                mp_size=mp_size,
                pp_size=pp_size,
                mp_mode=mp_mode,
                use_data_before_expert_parallel_=use_data_before_expert_parallel_,
            )
        self.ep_group = groups._get_expert_parallel_group(self.ep_group_name)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hidden_states: [B, S, H]

        Returns:
            [B, S, H] or ([B, S, H], [T, E]) if return_router_logits
        """
        bsz, seqlen, hdim = hidden_states.shape
        x = hidden_states.reshape(-1, hdim)  # [T, H]

        # Router
        ro: RouterOutput = RouterOutput(*self.router(x, self.expert_bias))

        # Accumulate expert utilization
        with torch.no_grad():
            self.tokens_per_expert.add_(ro.num_tokens_per_expert)

        # Reorder tokens by expert
        top_scores_sorted, token_indices_sorted, _ = self.reorderer(ro.top_scores, ro.selected_experts)

        routed_input = x[token_indices_sorted // self.top_k]  # [N, H]
        routed_input = apply_scores_before_experts_if_enabled(routed_input,
                                                              top_scores_sorted,
                                                              score_apply=self.score_apply)

        if self.ep_size == 1:
            # No AllToAll needed - local computation only
            local_counts = count_tokens_per_expert(
                ro.selected_experts,
                self.num_local_experts,
                out_dtype=torch.int32,
            )

            routed_input_permuted, perm_indices, aligned_counts, n_tokens = permute_by_local_expert(
                routed_input, local_counts)
            expert_output = self.experts(routed_input_permuted, aligned_counts)
            expert_output = unpermute_by_local_expert(expert_output, perm_indices, n_tokens)
        else:
            # EP dispatch/compute/combine
            plan = compute_split_plan(
                selected_experts=ro.selected_experts,
                num_experts=self.num_experts,
                ep_size=self.ep_size,
                num_local_experts=self.num_local_experts,
                ep_group=self.ep_group,
            )

            routed_input = _AllToAllV.apply(self.ep_group, routed_input, plan.input_splits, plan.output_splits)

            routed_input, perm_indices, aligned_counts, n_tokens = permute_by_local_expert(
                routed_input, plan.local_counts_by_source)
            expert_output = self.experts(routed_input, aligned_counts)
            expert_output = unpermute_by_local_expert(expert_output, perm_indices, n_tokens)

            expert_output = _AllToAllV.apply(self.ep_group, expert_output, plan.output_splits, plan.input_splits)

        output = combine_from_routed(
            expert_output,
            top_scores=ro.top_scores,
            token_indices_sorted=token_indices_sorted,
            top_k=self.top_k,
            score_apply=self.score_apply,
            shape=(bsz, seqlen, hdim),
        )

        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)

        if self.return_router_logits:
            logits = self._cached_router_logits
            self._cached_router_logits = None
            return output, logits

        self._cached_router_logits = None
        return output
