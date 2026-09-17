# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

from ..executor import PipelineExecutor
from .placement import RayTopology
from .stage_actor import StageActor


class RayActorExecutor(PipelineExecutor):
    """Pipeline executor that dispatches instructions to per-stage Ray actors.

    Each pipeline stage runs as a :class:`StageActor` Ray actor on its own
    GPU (or CPU). The driver orchestrates the schedule by calling remote
    methods on actors and coordinating inter-stage tensor transfers via the
    transport backend.

    This executor enables heterogeneous resource allocation: each stage can
    request different GPU types, CPU counts, or custom resources through
    :class:`RayTopology` placement group bundles.

    Args:
        engine: The :class:`PipelineEngine` instance (provides model, optimizer config).
        transport: The :class:`PipelineTransport` for inter-stage communication.
        topology: Optional :class:`RayTopology` for custom resource placement.
    """

    def __init__(self, engine, transport, topology=None):
        if not HAS_RAY:
            raise ImportError("RayActorExecutor requires Ray. Install with: pip install ray")

        super().__init__(transport)
        self._engine = engine
        self._topology = topology if topology is not None else RayTopology(num_stages=engine.num_stages)
        self._actors = {}
        self._initialized = False

    # ------------------------------------------------------------------
    # Actor lifecycle
    # ------------------------------------------------------------------

    def _initialize_actors(self):
        """Create a StageActor for the current pipeline stage.

        Each Ray driver hosts only its own stage's actor. The engine's
        PipelineModule already has layers partitioned for the current
        stage via ``_local_start``/``_local_stop``, so we pass the
        model directly.

        In a multi-stage Ray deployment, each stage's driver calls this
        independently. Cross-stage communication uses the transport layer
        with Ray object store references.
        """
        if self._initialized:
            return

        self._topology.initialize()
        stage_id = self._engine.stage_id

        options = self._topology.get_stage_options(stage_id)
        self._actors[stage_id] = StageActor.options(**options).remote(
            stage_id=stage_id,
            num_stages=self._topology.num_stages,
            model=self._engine.module,
            optimizer=self._engine.optimizer,
        )

        if hasattr(self._transport, 'set_actor_handles'):
            self._transport.set_actor_handles(self._actors, stage_id)

        self._initialized = True

    def shutdown(self):
        """Kill all actors and remove the placement group."""
        for actor in self._actors.values():
            ray.kill(actor)
        self._actors.clear()
        self._topology.shutdown()
        self._initialized = False

    # ------------------------------------------------------------------
    # Stage topology — delegates to engine
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
    # Helper
    # ------------------------------------------------------------------

    def _get_actor(self, stage_id=None):
        """Get the StageActor handle for a given stage.

        Args:
            stage_id (int, optional): Stage index. Defaults to current stage.

        Returns:
            StageActor handle.
        """
        if stage_id is None:
            stage_id = self._engine.stage_id
        if stage_id not in self._actors:
            raise RuntimeError(f"No actor for stage {stage_id}. Actors: {list(self._actors.keys())}")
        return self._actors[stage_id]

    # ------------------------------------------------------------------
    # Batch lifecycle
    # ------------------------------------------------------------------

    def start_batch(self, pipe_schedule):
        """Reserve pipeline buffers on all actors."""
        num_buffers = pipe_schedule.num_pipe_buffers()
        futures = [actor.reserve_buffers.remote(num_buffers) for actor in self._actors.values()]
        ray.get(futures)

    def end_batch(self):
        """No-op for Ray executor."""
        pass

    # ------------------------------------------------------------------
    # Instruction execution methods
    # ------------------------------------------------------------------

    def forward_pass(self, buffer_id):
        actor = self._get_actor()
        ray.get(actor.forward_pass.remote(buffer_id))

    def backward_pass(self, buffer_id):
        actor = self._get_actor()
        ray.get(actor.backward_pass.remote(buffer_id))

    def load_micro_batch(self, buffer_id):
        actor = self._get_actor()
        # Delegate data loading to the engine which handles first/last stages
        self._engine._exec_load_micro_batch(buffer_id)

        # For Ray executor, pass the loaded data to the actor
        if self.is_first_stage:
            inputs = self._engine.pipe_buffers['inputs'][buffer_id]
            if inputs is not None:
                ray.get(actor.load_micro_batch.remote(buffer_id, inputs=inputs))
        elif self.is_last_stage:
            labels = self._engine.pipe_buffers['labels'][buffer_id]
            if labels is not None:
                ray.get(actor.load_micro_batch.remote(buffer_id, labels=labels))

    def send_activations(self, buffer_id):
        src_actor = self._get_actor()
        tensors = ray.get(src_actor.get_activations.remote(buffer_id))
        dest = self._engine.stage_id + 1
        if dest < self._engine.num_stages:
            self._transport.send(tensors, dest_stage=dest)

    def recv_activations(self, buffer_id):
        src = self._engine.stage_id - 1
        if src < 0:
            return
        # Receive via transport (blocking)
        dummy = torch.zeros(1)
        tensors = self._transport.recv(dummy, src_stage=src)
        dest_actor = self._get_actor()
        ray.get(dest_actor.set_inputs.remote(buffer_id, tensors))

    def send_grads(self, buffer_id):
        actor = self._get_actor()
        grads = ray.get(actor.get_input_grads.remote(buffer_id))
        if grads is not None:
            dest = self._engine.stage_id - 1
            if dest >= 0:
                self._transport.send(grads, dest_stage=dest)

    def recv_grads(self, buffer_id):
        src = self._engine.stage_id + 1
        if src >= self._engine.num_stages:
            return
        dummy = torch.zeros(1)
        grads = self._transport.recv(dummy, src_stage=src)
        if grads is not None:
            actor = self._get_actor()
            ray.get(actor.set_output_grads.remote(buffer_id, grads))

    def optimizer_step(self, lr_kwargs=None):
        futures = [actor.optimizer_step.remote(lr_kwargs) for actor in self._actors.values()]
        ray.get(futures)

    def reduce_grads(self):
        actor = self._get_actor()
        ray.get(actor.reduce_grads.remote())

    def reduce_tied_grads(self):
        actor = self._get_actor()
        ray.get(actor.reduce_tied_grads.remote())
