# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Dict, Optional
from .autotp_config import TPLayerSpec, PartitionType

SUPPORTED_STYLES = {"colwise", "colwise_rep", "colwise_gather_output", "rowwise", "replicated_with_grad_allreduce"}
# `colwise_rep` was renamed to `colwise_gather_output` in huggingface/transformers#42809.


class TPPlanConverter:
    """Convert HuggingFace tp_plan format to DeepSpeed TPLayerSpec format."""

    @staticmethod
    def convert(hf_tp_plan: Dict[str, str]) -> Optional[List[TPLayerSpec]]:
        """Convert HF tp_plan to DeepSpeed layer specs.

        Entries whose style is not supported are converted to SKIP specs instead of invalidating
        the whole plan. Discarding the plan used to send models like Llama4 or Qwen3 down the
        heuristic path, which shards by name patterns alone and has no notion of the modules the
        plan deliberately excluded — on Llama4 it wraps the MoE router, whose forward returns a
        tuple, and breaks the model. A SKIP spec keeps such layers untouched on purpose while the
        supported entries are still applied.

        Returns None only when no entry is convertible, so the caller can fall back to the
        existing AutoTP path for models whose plan gives us nothing to work with.
        """
        if not hf_tp_plan:
            return None

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
            else:  # replicated_with_grad_allreduce, the only other supported style
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
