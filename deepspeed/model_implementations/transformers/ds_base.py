# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch.nn as nn


class DeepSpeedTransformerBase(nn.module):

    def __init__(self):
        pass

    # this would be the new clean base class that will replace DeepSpeedTransformerInference.
    # we currently don't know how this will look like but keeping it here as a placeholder.
