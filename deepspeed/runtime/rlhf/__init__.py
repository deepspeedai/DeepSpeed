# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""deepspeed.runtime.rlhf — Reinforcement Learning from Human Feedback runtime.

Sub-modules
-----------
config   : Training, rollout, distillation, and data configuration dataclasses.
losses   : Per-token KL / JSD divergence losses with sequence-axis chunking.
utils    : Shared tensor / masking helpers.
trainer  : Algorithm-specific training loops (OPSD, GRPO, …).
"""

from deepspeed.runtime.rlhf.config import (  # noqa: F401
    OPSDConfig, StudentConfig, TeacherConfig, RolloutConfig, DistillationConfig, TrainingConfig, DataConfig,
)
from deepspeed.runtime.rlhf.losses import (  # noqa: F401
    chunked_distillation_loss, streamed_distillation_loss, per_token_logprobs,
)
from deepspeed.runtime.rlhf.utils import (  # noqa: F401
    build_response_mask, shift_for_next_token_prediction,
)
