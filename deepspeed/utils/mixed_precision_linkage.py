# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from deepspeed.runtime.zero.mixed_precision_linkage import *
from deepspeed.utils import logger

logger.warning_once(
    "deepspeed.utils.mixed_precision_linkage is deprecated, please use deepspeed.runtime.zero.mixed_precision_linkage"
)
