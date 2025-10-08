# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# Apache-2.0 License Copyright (c) UChicago Argonne LLC, operator of Argonne National Laboratory.

# DeepSpeed Team

from deepspeed.runtime.config_utils import DeepSpeedConfigObject
import copy

DATASTATES_CHECKPOINTING = "datastates_ckpt"
DATASTATES_CHECKPOINTING_ENABLED = False


class DeepSpeedDataStatesConfig(DeepSpeedConfigObject):

    def __init__(self, param_dict):
        super(DeepSpeedDataStatesConfig, self).__init__()

        self.enabled = param_dict.get(DATASTATES_CHECKPOINTING, DATASTATES_CHECKPOINTING_ENABLED) is not False
        self.config = copy.deepcopy(param_dict.get(DATASTATES_CHECKPOINTING, None))
