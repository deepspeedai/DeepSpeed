# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .transport import PipelineTransport
from . import p2p


class NcclTransport(PipelineTransport):
    """Pipeline transport backed by NCCL point-to-point communication.

    Wraps the existing :mod:`deepspeed.runtime.pipe.p2p` module which uses
    :mod:`deepspeed.comm` (NCCL backend) for inter-stage tensor transfer.

    This transport requires all stages to reside within the same CUDA process
    group and share the same NCCL communicator.
    """

    def __init__(self):
        self._initialized = False

    def send(self, tensor, dest_stage):
        """Send a tensor via NCCL p2p to the destination stage.

        Delegates to :func:`p2p.send`.
        """
        if not self._initialized:
            raise RuntimeError("NcclTransport not initialized. Call initialize() first.")
        p2p.send(tensor, dest_stage)

    def recv(self, tensor, src_stage):
        """Receive a tensor via NCCL p2p from the source stage.

        Delegates to :func:`p2p.recv`.
        """
        if not self._initialized:
            raise RuntimeError("NcclTransport not initialized. Call initialize() first.")
        p2p.recv(tensor, src_stage)

    def initialize(self, topology):
        """Initialize NCCL process groups for inter-stage communication.

        Delegates to :func:`p2p.init_process_groups`.
        """
        if self._initialized:
            return
        p2p.init_process_groups(topology)
        self._initialized = True

    def shutdown(self):
        """NCCL transport shutdown is handled by the distributed runtime.

        NCCL process groups are destroyed when the process exits. No explicit
        cleanup is required.
        """
        self._initialized = False
