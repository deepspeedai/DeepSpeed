# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""On-Policy Distillation (OPSD) training on DeepSpeed.

A student model generates rollouts; a frozen teacher scores them; the student
is updated by a per-token divergence (forward-KL / reverse-KL / JSD) computed
against the teacher's distribution on the student's own samples.

Supports two rollout engines selected via config:
    * ``hybrid_engine`` — DeepSpeed's built-in train+infer engine (ZeRO-3 safe)
    * ``vllm``          — vLLM running on a disjoint set of GPUs with NCCL
                          weight sync from the trainer each step
"""

__version__ = "0.1.0"
