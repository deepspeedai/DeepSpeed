# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""DualPipeV pipeline parallelism (https://github.com/deepseek-ai/DualPipe).

Each pipeline rank holds two stages and micro-batches travel out through the phase-0
stages and back through the phase-1 stages, so rank 0 both loads the data and computes
the loss. :class:`DualPipeVModule` builds the two stages; :class:`DualPipeVEngine` runs
:class:`~deepspeed.runtime.pipe.schedule.DualPipeVSchedule` on top of the regular
:class:`~deepspeed.runtime.pipe.engine.PipelineEngine`.
"""

from contextlib import contextmanager

import torch
from deepspeed import comm as dist

from .engine import PipelineEngine
from .module import PipelineModule
from . import schedule

# Sentinel for the parent engine's __init__, which warms up p2p with is_last_stage() in
# its topological sense before this engine has any phase state.
_INIT = object()


def _as_list(tensors):
    return [tensors] if torch.is_tensor(tensors) else list(tensors)


class DualPipeVModule(PipelineModule):
    """A :class:`PipelineModule` cut into ``2 * num_stages`` stages for DualPipeV.

    Rank ``r`` builds stage ``r`` (phase 0) and stage ``2 * num_stages - 1 - r`` (phase 1).
    ``num_stages`` (or the ``pipe`` dimension of ``topology``) is the number of pipeline
    ranks, as for :class:`PipelineModule`. Layers tied between the first and the last stage
    share one module on rank 0, so no gradient exchange is needed for them.

    Only one phase is exposed through ``forward()`` at a time.
    """

    def _partition_layers(self, method='uniform'):
        self.num_stages *= 2
        super()._partition_layers(method)
        mirror = self.num_stages - 1 - self.stage_id
        self.phase_bounds = [(self._local_start, self._local_stop), (self.parts[mirror], self.parts[mirror + 1])]

    def stage_owner(self, layer_idx):
        """The pipeline rank that owns ``layer_idx``."""
        stage = super().stage_owner(layer_idx)
        return min(stage, self.num_stages - 1 - stage)

    def _build(self):
        self.phase_funcs = []
        self.phase_checkpointable = [[], []]
        for start, stop in self.phase_bounds:
            self._set_bounds(start, stop)
            super()._build()
            self.phase_funcs.append(self.forward_funcs)
            self.forward_funcs = []
        self.activate_phase(0)

    def activate_phase(self, phase):
        """Expose the layers of ``phase`` as this module's forward pass."""
        self.active_phase = phase
        self.forward_funcs = self.phase_funcs[phase]
        self.is_checkpointable_results = self.phase_checkpointable[phase]
        self._local_start, self._local_stop = self.phase_bounds[phase]

    def _precompute_checkpointable_values(self):
        for phase in (0, 1):
            self.activate_phase(phase)
            # the parent caches per interval
            # the cache must not carry over between phases
            self.is_checkpointable_results_interval = None
            super()._precompute_checkpointable_values()
            self.phase_checkpointable[phase] = self.is_checkpointable_results

    def save_state_dict(self, *args, **kwargs):
        for phase in (0, 1):
            self.activate_phase(phase)
            super().save_state_dict(*args, **kwargs)

    def load_state_dir(self, *args, **kwargs):
        for phase in (0, 1):
            self.activate_phase(phase)
            super().load_state_dir(*args, **kwargs)

    def compile(self, *args, **kwargs):
        for phase in (0, 1):
            self.activate_phase(phase)
            super().compile(*args, **kwargs)


class DualPipeVEngine(PipelineEngine):
    """Runs a :class:`DualPipeVModule` with the DualPipeV schedule.
    """

    def __init__(self, *args, **kwargs):
        self._active_phase = _INIT
        super().__init__(*args, **kwargs)
        assert isinstance(self.module, DualPipeVModule), "model must base DualPipeVModule"
        assert self.micro_batches >= 2 * self.num_stages, \
            f"DualPipeV needs gradient_accumulation_steps >= 2 * {self.num_stages}, got {self.micro_batches}"
        assert not (self.is_pipe_partitioned or self.is_grad_partitioned), \
            "DualPipeV does not support pipe_partitioned or grad_partitioned"
        assert not self.dynamic_shape, "DualPipeV exchanges activation shapes once and does not support dynamic_shape"
        assert not (self.has_attention_mask or self.has_bool_tensors), "DualPipeV cannot send bool tensors"

        self._active_phase = None
        self._p2p_ops = []
        self._first_send = [True, True]
        self._recv_specs = [None, None]
        # received output gradients and computed losses, per buffer
        self.pipe_buffers['grads'] = []
        self.pipe_buffers['losses'] = []

    def is_last_stage(self):
        """Rank 0 holds the last stage, but only its phase-1 half is the loss stage."""
        if self._active_phase is _INIT:
            return super().is_last_stage()
        return self.stage_id == 0 and self._active_phase != 0

    def _is_loss_stage(self, phase):
        return self.stage_id == 0 and phase == 1

    def _is_last_rank(self):
        return self.stage_id == self.num_stages - 1

    def _train_schedule(self):
        return schedule.DualPipeVSchedule(micro_batches=self.micro_batches,
                                          stages=self.num_stages,
                                          stage_id=self.stage_id)

    def _inference_schedule(self, micro_batches):
        return schedule.DualPipeVSchedule(micro_batches=micro_batches,
                                          stages=self.num_stages,
                                          stage_id=self.stage_id,
                                          forward_only=True)

    def _loss_stage_global_rank(self):
        return self.grid.stage_to_global(0)

    def eval_batch(self,
                   data_iter,
                   return_logits=False,
                   compute_loss=True,
                   reduce_output='avg',
                   bcast_loss=True,
                   num_micro_batches=None):
        # Reject a too-small micro-batch count before the parent swaps in the eval data iterator.
        self._inference_schedule(self.micro_batches if num_micro_batches is None else num_micro_batches)
        return super().eval_batch(data_iter, return_logits, compute_loss, reduce_output, bcast_loss, num_micro_batches)

    def reset_activation_shape(self):
        super().reset_activation_shape()
        self._first_send = [True, True]
        self._recv_specs = [None, None]

    @contextmanager
    def _phase(self, phase):
        self._active_phase = phase
        self.module.activate_phase(phase)
        try:
            yield
        finally:
            self._active_phase = None

    def _downstream_stage(self, phase):
        return self.stage_id + 1 if phase == 0 else self.stage_id - 1

    def _upstream_stage(self, phase):
        return self.stage_id - 1 if phase == 0 else self.stage_id + 1

    def _queue_p2p(self, op, tensors, stage):
        peer = self.grid.stage_to_global(stage)
        for tensor in tensors:
            self._p2p_ops.append(dist.P2POp(op, tensor, peer))

    def _exec_commit_p2p(self):
        if not self._p2p_ops:
            return
        for req in dist.batch_isend_irecv(self._p2p_ops):
            req.wait()
        self._p2p_ops = []

    def _phase1_buffer(self, buffer_id):
        """The phase-1 slot of the micro-batch held in phase-0 slot ``buffer_id``."""
        return buffer_id + self.micro_batches

    def _exec_load_micro_batch(self, buffer_id):
        super()._exec_load_micro_batch(buffer_id)
        self.pipe_buffers['labels'][self._phase1_buffer(buffer_id)] = self.pipe_buffers['labels'][buffer_id]
        self.pipe_buffers['labels'][buffer_id] = None

    def _exec_forward_pass(self, buffer_id, phase):
        with self._phase(phase):
            super()._exec_forward_pass(buffer_id)
        outputs = self.pipe_buffers['outputs'][buffer_id]
        if self._is_loss_stage(phase):
            if torch.is_grad_enabled():
                self.pipe_buffers['losses'][buffer_id] = self.loss
        elif self._is_last_rank() and phase == 0:
            inputs = [t.detach().requires_grad_(t.is_floating_point()) for t in _as_list(outputs)]
            inputs = inputs[0] if torch.is_tensor(outputs) else tuple(inputs)
            self.pipe_buffers['inputs'][self._phase1_buffer(buffer_id)] = inputs
        if not torch.is_grad_enabled():
            # No backward will run. The outputs of the loss stage and of the turn-around are not
            # sent either, so nothing else frees them.
            self.pipe_buffers['inputs'][buffer_id] = None
            self.pipe_buffers['labels'][buffer_id] = None
            if self._is_loss_stage(phase) or (self._is_last_rank() and phase == 0):
                self.pipe_buffers['outputs'][buffer_id] = None

    def _exec_backward_pass(self, buffer_id, phase):
        if self._is_loss_stage(phase):
            self.loss = self.pipe_buffers['losses'][buffer_id]
            self.pipe_buffers['losses'][buffer_id] = None
        elif self._is_last_rank() and phase == 0:
            inputs = self.pipe_buffers['inputs'][self._phase1_buffer(buffer_id)]
            self.pipe_buffers['inputs'][self._phase1_buffer(buffer_id)] = None
            grads = [t.grad for t in _as_list(inputs) if t.is_floating_point()]
            self.grad_layer = grads[0] if torch.is_tensor(inputs) else grads
        else:
            self.grad_layer = self.pipe_buffers['grads'][buffer_id]
            self.pipe_buffers['grads'][buffer_id] = None
        with self._phase(phase):
            super()._exec_backward_pass(buffer_id)
        # The first stage sends no gradients, so nothing else frees its inputs, and the loss stage
        # keeps its outputs and labels; the parent relies on cyclic buffers overwriting them.
        self.pipe_buffers['outputs'][buffer_id] = None
        if self.is_first_stage() and phase == 0:
            self.pipe_buffers['inputs'][buffer_id] = None
        if self._is_loss_stage(phase):
            self.pipe_buffers['labels'][buffer_id] = None

    def _exec_send_activations(self, buffer_id, phase):
        outputs = self.pipe_buffers['outputs'][buffer_id]
        stage = self._downstream_stage(phase)
        if self._first_send[phase]:
            self._first_send[phase] = False
            self._send_tensor_meta(outputs, stage)
        self._queue_p2p(dist.isend, _as_list(outputs), stage)
        if not torch.is_grad_enabled():
            self.pipe_buffers['outputs'][buffer_id] = None

    def _exec_recv_activations(self, buffer_id, phase):
        stage = self._upstream_stage(phase)
        if self._recv_specs[phase] is None:
            buffers = self._recv_tensor_meta(stage)
            self._recv_specs[phase] = (torch.is_tensor(buffers), [(t.shape, t.dtype) for t in _as_list(buffers)])
        is_tensor, specs = self._recv_specs[phase]

        recvd = [
            torch.empty(shape, dtype=dtype, device=self.device, requires_grad=dtype.is_floating_point)
            for shape, dtype in specs
        ]
        self._queue_p2p(dist.irecv, recvd, stage)
        self.pipe_buffers['inputs'][buffer_id] = recvd[0] if is_tensor else tuple(recvd)

    def _exec_send_grads(self, buffer_id, phase):
        inputs = self.pipe_buffers['inputs'][buffer_id]
        grads = [t.grad for t in _as_list(inputs) if t.is_floating_point()]
        assert all(g is not None for g in grads)
        self._queue_p2p(dist.isend, grads, self._upstream_stage(phase))
        self.pipe_buffers['inputs'][buffer_id] = None

    def _exec_recv_grads(self, buffer_id, phase):
        outputs = self.pipe_buffers['outputs'][buffer_id]
        grads = [torch.empty_like(t) for t in _as_list(outputs) if t.is_floating_point()]
        self._queue_p2p(dist.irecv, grads, self._downstream_stage(phase))
        self.pipe_buffers['grads'][buffer_id] = grads[0] if torch.is_tensor(outputs) else grads

    _INSTRUCTION_MAP = {
        **PipelineEngine._INSTRUCTION_MAP,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
        schedule.CommitP2P: _exec_commit_p2p,
    }
