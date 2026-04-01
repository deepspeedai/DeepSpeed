# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from deepspeed.runtime.zero.tensor_fragment import *
from deepspeed.utils import logger

logger.warning_once("deepspeed.utils.tensor_fragment is deprecated, please use deepspeed.runtime.zero.tensor_fragment")
