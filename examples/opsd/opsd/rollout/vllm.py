# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Backward-compatible re-export from :mod:`deepspeed.runtime.rollout`.

The vLLM rollout engine lives in
:class:`deepspeed.runtime.rollout.VLLMRollout`.  This module re-exports it
(and :func:`stitch_rollout`) so that existing ``from opsd.rollout.vllm import
...`` statements keep working.
"""

from deepspeed.runtime.rollout.vllm_rollout import VLLMRollout, stitch_rollout

__all__ = ["VLLMRollout", "stitch_rollout"]
