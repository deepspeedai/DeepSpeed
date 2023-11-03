# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from enum import Enum


class SchedulingResult(Enum):

    Success = 0
    """
    The proposed batch is valid and can be scheduled.
    """

    EngineSequenceLimitExceeded = 1
    """
    The proposed batch would would overflow the number of concurrent sequences the engine may support.
    """

    BatchSequenceLimitExceeded = 2
    """
    The proposed batch contains more sequences than the engine was configured
    to support in a single forwardp
    """

    BatchTokenLimitExceeded = 3
    """
    The proposed batch contains more tokens than the engine was configured
    to support in a single forward.
    """

    KVCacheLimitExceeded = 4
    """
    The proposed batch would require more KV cache to be allocated than the engine
    currently has available.
    """

    SequenceTokenLimitExceeded = 5
    """
    The proposed batch contains a sequence that is longer than the engine/model can support.
    """


class SchedulingError(RuntimeError):

    result: SchedulingResult
    """
    The failed result of the scheduling check. Guaranteed to not be SchedulingResult.Success.
    """

    def __init__(self, result: SchedulingResult) -> None:
        self.result = result
        super().__init__(f"Batch scheduling failed with result {result}")
