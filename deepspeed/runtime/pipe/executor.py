# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import ABC, abstractmethod

from . import schedule


class PipelineExecutor(ABC):
    """Abstract interface for pipeline stage execution backends.

    Each pipeline instruction from :class:`~schedule.PipeSchedule` maps to a method
    on the executor. Implementations decide *how* each instruction is carried out
    (locally in-process, or on a remote Ray actor).

    The executor owns the pipeline buffers and is responsible for forwarding
    activation/gradient communication to the :class:`PipelineTransport`.

    Life-cycle:
        1. ``start_batch(pipe_schedule)`` – allocate/reset buffers for a new batch.
        2. **Instruction methods** – called sequentially per the schedule.
        3. ``end_batch()`` – finalize the batch (optional).

    Sub-classes:
        :class:`ProcessGroupExecutor` – in-process execution (existing behaviour).
        ``RayActorExecutor`` (future) – per-stage Ray actor execution.
    """

    def __init__(self, transport):
        """Initialize the executor with a transport backend.

        Args:
            transport (:class:`PipelineTransport`): The transport layer for
                inter-stage communication.
        """
        self._transport = transport

    # ------------------------------------------------------------------
    # Batch lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def start_batch(self, pipe_schedule):
        """Prepare buffers and state at the beginning of a batch.

        Args:
            pipe_schedule (:class:`~schedule.PipeSchedule`): The schedule that
                will be executed for this batch.
        """
        pass

    @abstractmethod
    def end_batch(self):
        """Clean up after a batch completes.

        Optional hook – default implementation is a no-op.
        """
        pass

    # ------------------------------------------------------------------
    # Stage topology
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def stage_id(self):
        """int: The pipeline stage index of this executor."""
        pass

    @property
    @abstractmethod
    def num_stages(self):
        """int: Total number of pipeline stages."""
        pass

    @property
    @abstractmethod
    def is_first_stage(self):
        """bool: ``True`` if this executor is the first stage in the pipeline."""
        pass

    @property
    @abstractmethod
    def is_last_stage(self):
        """bool: ``True`` if this executor is the last stage in the pipeline."""
        pass

    # ------------------------------------------------------------------
    # Instruction execution methods
    # ------------------------------------------------------------------

    @abstractmethod
    def forward_pass(self, buffer_id):
        """Execute a forward pass on the micro-batch in the given buffer.

        Args:
            buffer_id (int): Index of the pipeline buffer containing inputs.
        """
        pass

    @abstractmethod
    def backward_pass(self, buffer_id):
        """Execute a backward pass on the micro-batch in the given buffer.

        Args:
            buffer_id (int): Index of the pipeline buffer containing outputs.
        """
        pass

    @abstractmethod
    def load_micro_batch(self, buffer_id):
        """Load the next micro-batch of data into the pipeline buffer.

        The first stage loads inputs; the last stage loads labels.
        Intermediate stages are a no-op.

        Args:
            buffer_id (int): Index of the pipeline buffer to fill.
        """
        pass

    @abstractmethod
    def send_activations(self, buffer_id):
        """Send activations from this stage to the next stage.

        Args:
            buffer_id (int): Index of the pipeline buffer containing outputs.
        """
        pass

    @abstractmethod
    def recv_activations(self, buffer_id):
        """Receive activations from the previous stage.

        Args:
            buffer_id (int): Index of the pipeline buffer to store inputs.
        """
        pass

    @abstractmethod
    def send_grads(self, buffer_id):
        """Send gradients to the previous stage.

        Args:
            buffer_id (int): Index of the pipeline buffer containing input grads.
        """
        pass

    @abstractmethod
    def recv_grads(self, buffer_id):
        """Receive gradients from the next stage.

        Args:
            buffer_id (int): Index of the pipeline buffer to store output grads.
        """
        pass

    @abstractmethod
    def optimizer_step(self, lr_kwargs=None):
        """Perform one optimizer step.

        Args:
            lr_kwargs (dict, optional): Learning rate overrides.
        """
        pass

    @abstractmethod
    def reduce_grads(self):
        """Reduce gradients across data-parallel ranks within this stage."""
        pass

    @abstractmethod
    def reduce_tied_grads(self):
        """Reduce gradients of tied weights across pipeline stages."""
        pass

    # ------------------------------------------------------------------
    # Instruction dispatch map
    # ------------------------------------------------------------------
    # Maps schedule.PipeInstruction subclasses to executor methods.
    # Used by PipelineEngine._exec_schedule to avoid if/elif chains.

    @property
    def instruction_map(self):
        """Mapping from schedule instruction types to executor methods.

        Returns:
            dict: ``{type(PipeInstruction): callable}``
        """
        return {
            schedule.OptimizerStep: self.optimizer_step,
            schedule.ReduceGrads: self.reduce_grads,
            schedule.ReduceTiedGrads: self.reduce_tied_grads,
            schedule.LoadMicroBatch: self.load_micro_batch,
            schedule.ForwardPass: self.forward_pass,
            schedule.BackwardPass: self.backward_pass,
            schedule.SendActivation: self.send_activations,
            schedule.RecvActivation: self.recv_activations,
            schedule.SendGrad: self.send_grads,
            schedule.RecvGrad: self.recv_grads,
        }
