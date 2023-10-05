# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from . import adam
from . import adagrad
from . import lamb
from . import lion
#from ..git_version_info_installed import installed_ops as __installed_ops__
#if __installed_ops__['sparse_attn']:
from . import sparse_attention
from . import transformer

from .transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig

from ..git_version_info import compatible_ops as __compatible_ops__
