# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .executor import PipelineExecutor


class ProcessGroupExecutor(PipelineExecutor):
    """Pipeline executor that runs all stages in-process.

    This executor delegates to the existing :class:`PipelineEngine` execution
    methods, which use NCCL process groups for inter-stage communication.
    It is a thin adapter that makes the existing in-process pipeline engine
    conform to the :class:`PipelineExecutor` interface.

    Args:
        engine (:class:`PipelineEngine`): The pipeline engine that owns the
            model, optimizer, and pipeline buffers.
        transport (:class:`PipelineTransport`): The transport backend
            (typically :class:`NcclTransport`).
    """

    def __init__(self, engine, transport):
        super().__init__(transport)
        self._engine = engine

    # ------------------------------------------------------------------
    # Stage topology – delegates to engine
    # ------------------------------------------------------------------

    @property
    def stage_id(self):
        return self._engine.stage_id

    @property
    def num_stages(self):
        return self._engine.num_stages

    @property
    def is_first_stage(self):
        return self._engine.is_first_stage()

    @property
    def is_last_stage(self):
        return self._engine.is_last_stage()

    # ------------------------------------------------------------------
    # Batch lifecycle
    # ------------------------------------------------------------------

    def start_batch(self, pipe_schedule):
        """Reserve pipeline buffers and reset forward outputs.

        Delegates to :meth:`PipelineEngine._reserve_pipe_buffers` and clears
        ``fwd_outputs``.
        """
        self._engine._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        self._engine.fwd_outputs = []

    def end_batch(self):
        """No-op: in-process executor has no batch-level teardown."""
        pass

    # ------------------------------------------------------------------
    # Instruction execution – delegates to engine's existing _exec_* methods
    # ------------------------------------------------------------------

    def forward_pass(self, buffer_id):
        self._engine._exec_forward_pass(buffer_id)

    def backward_pass(self, buffer_id):
        self._engine._exec_backward_pass(buffer_id)

    def load_micro_batch(self, buffer_id):
        self._engine._exec_load_micro_batch(buffer_id)

    def send_activations(self, buffer_id):
        self._engine._exec_send_activations(buffer_id)

    def recv_activations(self, buffer_id):
        self._engine._exec_recv_activations(buffer_id)

    def send_grads(self, buffer_id):
        self._engine._exec_send_grads(buffer_id)

    def recv_grads(self, buffer_id):
        self._engine._exec_recv_grads(buffer_id)

    def optimizer_step(self, lr_kwargs=None):
        self._engine._exec_optimizer_step(lr_kwargs)

    def reduce_grads(self):
        self._engine._exec_reduce_grads()

    def reduce_tied_grads(self):
        self._engine._exec_reduce_tied_grads()
