# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Dict
from .autotp_config import TPLayerSpec, PartitionType


class TPPlanConverter:
    """Convert HuggingFace tp_plan format to DeepSpeed TPLayerSpec format."""

    @staticmethod
    def convert(hf_tp_plan: Dict[str, str]) -> List[TPLayerSpec]:
        layer_specs = []

        for pattern, partition in hf_tp_plan.items():
            regex_pattern = TPPlanConverter._wildcard_to_regex(pattern)

            if partition.lower() == "colwise":
                partition_type = PartitionType.COLUMN
            elif partition.lower() == "rowwise":
                partition_type = PartitionType.ROW
            else:
                # TODO: HF tp_plan supports additional partition types that are not yet handled:
                #   colwise_rep, local_colwise, local_rowwise, local_packed_rowwise,
                #   gather, sequence_parallel. Add support as needed.
                raise ValueError(f"Unsupported partition type '{partition}'. "
                                 f"Currently only 'colwise' and 'rowwise' are supported.")

            # Only add .weight suffix if not already present
            if not regex_pattern.endswith(r"\.weight"):
                regex_pattern += r"\.weight$"
            else:
                regex_pattern += r"$"

            layer_specs.append(TPLayerSpec(
                patterns=[regex_pattern],
                partition_type=partition_type,
            ))

        return layer_specs

    @staticmethod
    def _wildcard_to_regex(pattern: str) -> str:
        regex = pattern.replace('.', r'\.')
        regex = regex.replace('*', r'.*')
        return ".*" + regex
