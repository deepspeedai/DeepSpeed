# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .containers import HFGPT2LayerPolicy
from .containers import HFBertLayerPolicy
from .containers import BLOOMLayerPolicy
from .containers import HFGPTJLayerPolicy
from .containers import HFGPTNEOLayerPolicy
from .containers import GPTNEOXLayerPolicy
from .containers import HFOPTLayerPolicy
from .containers import MegatronLayerPolicy
from .containers import HFDistilBertLayerPolicy
from .containers import HFCLIPLayerPolicy
from .containers import LLAMALayerPolicy
from .containers import UNetPolicy
from .containers import VAEPolicy

# transformer-based policies
replace_policies = [
    HFBertLayerPolicy, HFGPTNEOLayerPolicy, GPTNEOXLayerPolicy, HFGPTJLayerPolicy, MegatronLayerPolicy,
    HFGPT2LayerPolicy, BLOOMLayerPolicy, HFOPTLayerPolicy, HFCLIPLayerPolicy, HFDistilBertLayerPolicy, LLAMALayerPolicy
]

# non-transformer-based policies
generic_policies = [UNetPolicy, VAEPolicy]
