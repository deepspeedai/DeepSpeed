# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.sequence.autosp_detector import detect_model_sp_info, SPModelInfo
from deepspeed.sequence.autosp_vit import UlyssesSPViTAttention
from deepspeed.sequence.autosp_fusion import (ModalityFusionSPAdapter, LlavaFusionAdapter, InternVLFusionAdapter,
                                              Qwen2VLFusionAdapter)
from deepspeed.sequence.auto_sp import auto_wrap_model_for_sp
