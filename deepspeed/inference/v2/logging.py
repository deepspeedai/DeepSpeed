# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import logging

from deepspeed.utils.logging import LoggerFactory

inf_logger = None


def inference_logger(level: int = logging.INFO) -> logging.Logger:
    """
    Create the inference logger. NOTE: Logging is not cost free. On a 3960X,
    there is a cost of about 6 us per call to a no-op logger, so this should
    be used during setup only and not during the inference loop.

    Args:
        level (int, optional): The logging level. Defaults to logging.INFO.
    """
    global inf_logger
    if inf_logger is None:
        inf_logger = LoggerFactory.create_logger(name="DS-Inference", level=level)
        inf_logger.debug("Inference logger created.")
    return inf_logger
