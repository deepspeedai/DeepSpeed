# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Shape helpers for fused gate/up expert tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FusedLayout = Literal["gate_up_first", "hidden_first"]


@dataclass(frozen=True)
class FusedExpertLayout:
    layout: FusedLayout
    hidden_size: int
    ffn_hidden_size: int
    needs_transpose: bool


def classify_fused_gate_up_layout(
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
) -> FusedExpertLayout | None:
    """Classify fused gate/up expert weights from raw tensor shapes."""
    if len(w1_shape) != 3 or len(w2_shape) != 3:
        return None

    if w1_shape[1] % 2 == 0 and w2_shape[1:] == (w1_shape[2], w1_shape[1] // 2):
        return FusedExpertLayout(
            layout="gate_up_first",
            hidden_size=w1_shape[2],
            ffn_hidden_size=w1_shape[1] // 2,
            needs_transpose=False,
        )

    if w1_shape[2] % 2 == 0 and w2_shape[1:] == (w1_shape[2] // 2, w1_shape[1]):
        return FusedExpertLayout(
            layout="hidden_first",
            hidden_size=w1_shape[1],
            ffn_hidden_size=w1_shape[2] // 2,
            needs_transpose=True,
        )

    return None
