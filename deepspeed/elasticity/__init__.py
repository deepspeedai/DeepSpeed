# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .elasticity import compute_elastic_config, elasticity_enabled, ensure_immutable_elastic_config
from .utils import is_torch_elastic_compatible
from .constants import ENABLED, ENABLED_DEFAULT, ELASTICITY
if is_torch_elastic_compatible():
    from .elastic_agent import DSElasticAgent
