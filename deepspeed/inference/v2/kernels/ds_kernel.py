# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import ABC, abstractmethod


class DSKernelBase(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        If necessary trigger compilation and warmup
        Autotuning of the kernel would happen at this stage to
        eliminate any potential hangs that might occur mid-deployment
        Validate that the desired run configuration is compatible.

        It is not necessary to call super on this method.
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        However the kernel needs to be called, it can be called here. Auto-tuning
        should never be performed here.

        All inputs/outputs should be passed as arguments to this function. No allocations
        should be performed here.
        """
        raise NotImplementedError()
