# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch


def is_torch_elastic_compatible():
    '''
        Helper to lookup torch version. Elastic training is
        introduced in 1.11.x
    '''
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    if TORCH_MAJOR == 1 and TORCH_MINOR >= 11:
        return True
    else:
        return False
