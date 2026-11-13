# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Training changes for specific model families.

One subpackage per model family, named after the model's folder in ``transformers.models`` (for
example ``glm_moe_dsa`` for GLM-5.2). A subpackage holds whatever that family needs to train fast
or to fit in memory: kernels, replacement modules, or plain PyTorch code. ``import deepspeed`` does
not import these subpackages, so their optional dependencies are only needed by the models that
use them. Each subpackage documents the function to call on a built model before
``deepspeed.initialize``.
"""
