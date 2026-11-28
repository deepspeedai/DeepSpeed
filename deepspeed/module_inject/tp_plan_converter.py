# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import re
from typing import List, Dict, Optional, Any
from .autotp_config import TPLayerSpec, PartitionType

SUPPORTED_STYLES = {
    "colwise", "colwise_rep", "colwise_gather_output", "rowwise", "replicated", "replicated_with_grad_allreduce",
    "embedding_rowwise"
}
# `colwise_rep` was renamed to `colwise_gather_output` in huggingface/transformers#42809.
# `embedding_rowwise` is injected for every tied-embedding model since huggingface/transformers#47579.


class TPPlanConverter:
    """Convert HuggingFace tp_plan format to DeepSpeed TPLayerSpec format."""

    @staticmethod
    def supplement_replicated_qk_norm(hf_tp_plan: Dict[str, str], model: Optional[Any] = None) -> Dict[str, str]:
        """Ensure replicated_with_grad_allreduce is set for per-head q_norm/k_norm/v_norm.

        In transformers < 5.3, models with per-head Q/K norm (such as Qwen3, Qwen3-MoE, Gemma3,
        Exaone4) declared column-wise sharding for self_attn.q_proj / k_proj in their
        base_model_tp_plan, but had no entries for self_attn.q_norm or self_attn.k_norm.
        Without grad_allreduce hooks, each tensor-parallel rank updates whole norm weights
        from partial attention gradients, causing ranks to drift apart during training.
        """
        if not hf_tp_plan:
            return hf_tp_plan

        if model is not None:
            if not hasattr(model, "named_parameters"):
                return hf_tp_plan
            model_params = [name for name, _ in model.named_parameters()]
        else:
            model_params = None

        has_q_norm = model_params is None or any("q_norm" in name for name in model_params)
        has_k_norm = model_params is None or any("k_norm" in name for name in model_params)
        has_v_norm = any("v_norm" in name for name in model_params) if model_params is not None else False

        proj_to_norm = {
            "q_proj": ("q_norm", has_q_norm),
            "k_proj": ("k_norm", has_k_norm),
            "v_proj": ("v_norm", has_v_norm),
        }

        has_wildcard_colwise = {}
        for proj in proj_to_norm:
            has_wildcard_colwise[proj] = any(
                "*" in pattern and partition.lower() == "colwise" and pattern.endswith((f".{proj}", f".{proj}.weight"))
                for pattern, partition in hf_tp_plan.items()) or any(
                    "*" in pattern and partition.lower() == "replicated_with_grad_allreduce" and pattern.endswith(
                        (f".{proj_to_norm[proj][0]}", f".{proj_to_norm[proj][0]}.weight"))
                    for pattern, partition in hf_tp_plan.items())

        additions = {}
        for pattern, partition in list(hf_tp_plan.items()):
            part_lower = partition.lower()
            if part_lower not in ("colwise", "colwise_rep", "colwise_gather_output"):
                continue

            for proj, (norm, has_norm) in proj_to_norm.items():
                if not has_norm:
                    continue
                for suffix in (f".{proj}", f".{proj}.weight"):
                    if pattern.endswith(suffix):
                        prefix = pattern[:-len(suffix)]
                        norm_key = f"{prefix}.{norm}"
                        if part_lower == "colwise":
                            if norm_key not in hf_tp_plan and norm_key not in additions:
                                additions[norm_key] = "replicated_with_grad_allreduce"
                        elif part_lower in ("colwise_rep", "colwise_gather_output"):
                            if has_wildcard_colwise[proj] or "*" not in pattern:
                                if norm_key not in hf_tp_plan:
                                    additions[norm_key] = "replicated"

        if model_params is not None:
            for param_name in model_params:
                for proj, (norm, has_norm) in proj_to_norm.items():
                    if not has_norm or norm not in param_name:
                        continue
                    proj_param_name = param_name.replace(norm, proj)
                    matching_patterns = []
                    for pattern, partition in hf_tp_plan.items():
                        regex = pattern.replace('.', r'\.').replace('*', r'.*')
                        if re.match(r".*" + regex + r"(\.weight)?$", proj_param_name):
                            matching_patterns.append((pattern, partition))
                    if not matching_patterns:
                        continue
                    best_pattern, best_partition = matching_patterns[0]
                    best_style = best_partition.lower()

                    norm_base = param_name[:-len(".weight")] if param_name.endswith(".weight") else param_name
                    norm_base = norm_base[:-len(".bias")] if norm_base.endswith(".bias") else norm_base

                    if best_style in ("colwise_rep", "colwise_gather_output"):
                        if norm_base not in hf_tp_plan:
                            additions[norm_base] = "replicated"

        hf_tp_plan.update(additions)
        return hf_tp_plan

    @staticmethod
    def convert(hf_tp_plan: Dict[str, str], model: Optional[Any] = None) -> Optional[List[TPLayerSpec]]:
        """Convert HF tp_plan to DeepSpeed layer specs.

        A style outside SUPPORTED_STYLES raises ValueError and invalidates the whole plan, so a
        style newly introduced upstream fails loudly and gets a deliberate decision rather than
        being dropped on the quiet. Converting only the recognized entries could shard one half
        of a column/row pair, and silently discarding the plan would send models like Llama4 or
        Qwen3 down the heuristic path, which shards by name patterns alone and has no notion of
        the modules the plan deliberately excluded. On Llama4 that path wraps the MoE router,
        whose forward returns a tuple, and breaks the model.

        Supported styles that must stay whole become SKIP specs, which keeps those layers
        untouched on purpose while the rest of the plan is still applied: `embedding_rowwise`,
        because this SKIP spec only opts the embedding out of the generic pattern-matched
        partitioning path, `replicated`, which keeps the layer whole without gradient all-reduce,
        and `replicated_with_grad_allreduce`, which additionally sums the gradient across the group.

        Returns None only for an empty plan, so the caller can fall back to the existing AutoTP
        path for models that give us nothing to work with.
        """
        if not hf_tp_plan:
            return None

        if model is not None:
            hf_tp_plan = dict(hf_tp_plan)
            TPPlanConverter.supplement_replicated_qk_norm(hf_tp_plan, model)

        unsupported = {style for style in hf_tp_plan.values() if style.lower() not in SUPPORTED_STYLES}
        if unsupported:
            raise ValueError(f"HuggingFace tp_plan contains unsupported partition style(s): {sorted(unsupported)}. "
                             "Applying only the supported entries could shard one half of a column/row pair, so the "
                             "plan is rejected as a whole. Provide an explicit 'tensor_parallel.partition_config' "
                             "for this model instead.")

        layer_specs = []

        for pattern, partition in hf_tp_plan.items():
            regex_pattern = TPPlanConverter._wildcard_to_regex(pattern)
            partition_style = partition.lower()
            gather_output = False
            grad_allreduce = False

            if partition_style in ("colwise", "colwise_rep", "colwise_gather_output"):
                partition_type = PartitionType.COLUMN
                gather_output = partition_style != "colwise"
            elif partition_style == "rowwise":
                partition_type = PartitionType.ROW
            elif partition_style in ("embedding_rowwise", "replicated"):
                # The SKIP spec here only means "don't shard this via the generic
                # pattern-matched path". A tied embedding whose lm_head becomes
                # vocab-parallel is still detected and jointly vocab-sharded by
                # AutoTP._create_vocab_parallel_layer, independent of this spec.
                partition_type = PartitionType.SKIP
            else:  # replicated_with_grad_allreduce
                # The parameter stays whole; only its gradient needs summing across the group.
                partition_type = PartitionType.SKIP
                grad_allreduce = True

            # Only add .weight suffix if not already present
            if not regex_pattern.endswith(r"\.weight"):
                regex_pattern += r"\.weight$"
            else:
                regex_pattern += r"$"

            layer_specs.append(
                TPLayerSpec(
                    patterns=[regex_pattern],
                    partition_type=partition_type,
                    gather_output=gather_output,
                    grad_allreduce=grad_allreduce,
                ))

        return layer_specs

    @staticmethod
    def _wildcard_to_regex(pattern: str) -> str:
        regex = pattern.replace('.', r'\.')
        regex = regex.replace('*', r'.*')
        return ".*" + regex
