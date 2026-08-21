# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import ABC, abstractmethod


class PipelineTransport(ABC):
    """Abstract interface for data transfer between adjacent pipeline stages.

    Transport backends handle the message-passing between consecutive stages
    (stage N -> stage N+1 for activations, stage N+1 -> stage N for gradients).
    The protocol is unidirectional point-to-point between adjacent stage IDs.

    Sub-classes:
        :class:`NcclTransport` – NCCL send/recv for GPU-GPU communication.
        ``TcpTransport`` (future) – TCP sockets for cross-platform transfer.
        ``SharedMemoryTransport`` (future) – SHM for CPU-CPU on the same node.
    """

    @abstractmethod
    def send(self, tensor, dest_stage):
        """Send a tensor to the given destination stage.

        Args:
            tensor (torch.Tensor): The tensor to send (must be on accelerator device for NCCL).
            dest_stage (int): The destination stage ID.
        """
        pass

    @abstractmethod
    def recv(self, tensor, src_stage):
        """Receive a tensor from the given source stage.

        For NCCL-backed transports the received data is written into
        ``tensor`` in-place and ``None`` is returned. For pure-Python
        transports (TCP, shared memory, Ray object store) the receiving
        side allocates and returns a new tensor; the ``tensor`` parameter
        is ignored.

        Args:
            tensor (torch.Tensor): Pre-allocated buffer for in-place
                receive (NCCL), or a dummy tensor (other transports).
            src_stage (int): The source stage ID.

        Returns:
            torch.Tensor or None: The received tensor for non-NCCL
            transports; ``None`` for NCCL in-place receives.
        """
        pass

    @abstractmethod
    def initialize(self, topology):
        """Initialize the transport layer with pipeline topology.

        Called once after the process grid is set up. For NCCL this creates
        the adjacent process groups. For TCP this binds sockets.

        Args:
            topology: The pipeline topology object providing stage-to-rank mapping.
        """
        pass

    @abstractmethod
    def shutdown(self):
        """Tear down the transport layer.

        Close sockets, destroy process groups, etc. Called during cleanup.
        """
        pass

    def is_available(self, src_stage, dest_stage):
        """Check if transport between two stages is supported.

        The default implementation always returns ``True``.

        Args:
            src_stage (int): Source stage ID.
            dest_stage (int): Destination stage ID.

        Returns:
            bool: ``True`` if the transport can handle communication between these stages.
        """
        return True
