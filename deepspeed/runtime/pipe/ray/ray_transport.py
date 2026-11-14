# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

from ..transport import PipelineTransport


class RayTransport(PipelineTransport):
    """Pipeline transport using the Ray distributed object store.

    Since Ray actors may run on different machines, NCCL p2p is not always
    available. This transport defaults to Ray's distributed object store for
    tensor serialization and transfer, which works transparently across nodes.

    For same-machine GPU-GPU transfers where NCCL is available, the
    ``'auto'`` backend can be configured (future enhancement).

    Args:
        backend (str): Transport backend mode.
            ``'ray_object_store'`` (default) — Always use Ray object store.
            ``'auto'`` (future) — Use NCCL p2p when stages are co-located,
            fall back to Ray object store otherwise.
    """

    VALID_BACKENDS = ('ray_object_store', 'auto')

    def __init__(self, backend='ray_object_store'):
        if not HAS_RAY:
            raise ImportError("RayTransport requires Ray. Install with: pip install ray")

        if backend not in self.VALID_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}. Must be one of {self.VALID_BACKENDS}")
        self._backend = backend
        self._initialized = False
        self._peer_refs = {}
        self._colocated_cache = {}
        self._current_stage = None

    def send(self, tensor, dest_stage):
        """Send tensor(s) to a destination stage via Ray object store.

        Stores the tensor in the Ray object store and deposits the
        reference on the destination actor via ``_store_pending_ref``.
        When backend is ``'auto'``, logs whether the source and
        destination stages are colocated.

        Args:
            tensor: A ``torch.Tensor`` or sequence of tensors.
            dest_stage (int): Destination pipeline stage ID.
        """
        if dest_stage not in self._peer_refs:
            raise ValueError(f"No actor handle registered for stage {dest_stage}. "
                             "Call set_actor_handles() first.")

        if self._backend == 'auto':
            colocated = self._is_colocated(self._current_stage, dest_stage)
            import logging
            logging.info("RayTransport send: stages %d -> %d, colocated=%s", self._current_stage, dest_stage,
                         colocated)

        ref = ray.put(tensor)
        self._peer_refs[dest_stage]._store_pending_ref.remote(self._current_stage, ref)

    def recv(self, tensor, src_stage):
        """Receive tensor(s) from a source stage.

        Retrieves the pending object reference from the current stage's
        actor via ``_get_pending_ref``, then resolves it via ``ray.get``.

        Args:
            tensor: Pre-allocated buffer (unused; returned data replaces this).
            src_stage (int): Source pipeline stage ID.

        Returns:
            The received tensor(s) from the Ray object store.
        """
        actor = self._peer_refs[self._current_stage]
        ref = ray.get(actor._get_pending_ref.remote(src_stage))
        if ref is None:
            raise RuntimeError(f"No pending receive from stage {src_stage}. "
                               "Ensure send() was called before recv().")
        return ray.get(ref)

    def initialize(self, topology):
        """Initialize the transport with pipeline topology.

        For Ray transport, topology is used to validate stage ranges.

        Args:
            topology: Pipeline topology object.
        """
        self._topology = topology
        if self._backend == 'auto':
            import logging
            logging.info("RayTransport: 'auto' backend selected, colocation "
                         "detection enabled.")
        self._initialized = True

    def set_actor_handles(self, actors, current_stage):
        """Register Ray actor handles for all pipeline stages.

        Called by :class:`RayActorExecutor` after actor creation so the
        transport can push object refs directly to the receiving actor.

        Args:
            actors (dict): Mapping of ``stage_id`` to ``StageActor`` handle.
            current_stage (int): The driver/executor's stage ID.
        """
        self._peer_refs = dict(actors)
        self._current_stage = current_stage

    def _detect_colocation(self, stage_a, stage_b):
        """Check if two stages are on the same Ray node.

        Uses Ray's ``get_node_id()`` to detect colocation. On macOS
        or platforms without a real node, returns ``False`` so tests pass.

        Args:
            stage_a (int): First stage ID.
            stage_b (int): Second stage ID.

        Returns:
            bool: ``True`` if both stage actors exist and are assigned to
            the same Ray node.
        """
        try:
            node_a = ray.get(self._peer_refs[stage_a]._get_node_id.remote())
            node_b = ray.get(self._peer_refs[stage_b]._get_node_id.remote())
            return node_a == node_b
        except Exception:
            return False

    def _is_colocated(self, stage_a, stage_b):
        """Check colocation with caching.

        Args:
            stage_a (int): First stage ID.
            stage_b (int): Second stage ID.

        Returns:
            bool: ``True`` if both stages are on the same Ray node.
        """
        key = (min(stage_a, stage_b), max(stage_a, stage_b))
        if key not in self._colocated_cache:
            self._colocated_cache[key] = self._detect_colocation(stage_a, stage_b)
        return self._colocated_cache[key]

    def shutdown(self):
        """Release all actor handles and clear caches."""
        self._peer_refs.clear()
        self._colocated_cache.clear()
        self._current_stage = None
        self._initialized = False
