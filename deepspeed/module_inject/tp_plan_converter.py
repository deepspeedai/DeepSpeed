# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import logging
from typing import List, Dict, Optional
from .autotp_config import TPLayerSpec, PartitionType

logger = logging.getLogger(__name__)

SUPPORTED_STYLES = {"colwise", "rowwise"}


class TPPlanConverter:
    """Convert HuggingFace tp_plan format to DeepSpeed TPLayerSpec format."""

    @staticmethod
    def convert(hf_tp_plan: Dict[str, str]) -> Optional[List[TPLayerSpec]]:
        """Convert HF tp_plan to DeepSpeed layer specs.

        Returns None if the plan contains any unsupported partition styles,
        allowing the caller to fall back to the existing AutoTP path.
        """
        unsupported = {style for style in hf_tp_plan.values() if style.lower() not in SUPPORTED_STYLES}
        if unsupported:
            logger.warning(
                "HuggingFace tp_plan contains unsupported partition style(s): %s. "
                "Falling back to AutoTP preset-based partitioning.", sorted(unsupported))
            return None

        layer_specs = []

        for pattern, partition in hf_tp_plan.items():
            regex_pattern = TPPlanConverter._wildcard_to_regex(pattern)

            if partition.lower() == "colwise":
                partition_type = PartitionType.COLUMN
            elif partition.lower() == "rowwise":
                partition_type = PartitionType.ROW

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
