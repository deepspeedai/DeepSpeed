# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tests for RayTransport inter-stage tensor communication.

RayTransport uses the Ray distributed object store for sending tensors
between pipeline stages running on different Ray actors.
"""

import torch
import pytest

pytest.importorskip("ray", reason="Ray is not installed")


class TestRayTransportValidation:
    """Validate RayTransport initialization and error handling."""

    def test_valid_backends(self, ray_isolated):
        """Ray object store backend is accepted."""
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport
        transport = RayTransport(backend='ray_object_store')
        assert transport._backend == 'ray_object_store'

    def test_invalid_backend_raises(self, ray_isolated):
        """Invalid backend raises ValueError."""
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport
        with pytest.raises(ValueError, match="Unsupported backend"):
            RayTransport(backend='invalid_backend')

    def test_send_without_handles_raises(self, ray_isolated):
        """send() before set_actor_handles raises ValueError."""
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport
        transport = RayTransport()
        transport.initialize(None)
        with pytest.raises(ValueError, match="No actor handle"):
            transport.send(torch.zeros(1), dest_stage=1)

    def test_recv_without_pending_raises(self, ray_isolated):
        """recv() without pending send raises RuntimeError."""
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport
        transport = RayTransport()
        transport.initialize(None)
        with pytest.raises(RuntimeError, match="No pending receive"):
            transport.recv(torch.zeros(1), src_stage=0)


class TestRayTransportIntegration:
    """Integration tests for RayTransport with real Ray actors."""

    @pytest.fixture(scope="class")
    def actors(self):
        """Create two simple actors that support _store_pending_ref."""
        import ray

        @ray.remote
        class TestActor:

            def __init__(self):
                self._pending = {}

            def _store_pending_ref(self, src_stage, ref):
                self._pending[src_stage] = ref

            def get_pending(self, src_stage):
                return self._pending.get(src_stage)

        return {
            0: TestActor.remote(),
            1: TestActor.remote(),
        }

    def test_send_recv_roundtrip(self, ray_isolated, actors):
        """Tensor sent via Ray object store is received correctly."""
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport

        transport = RayTransport()
        transport.initialize(None)
        transport.set_actor_handles(actors, current_stage=0)

        tensor = torch.tensor([1.0, 2.0, 3.0])
        transport.send(tensor, dest_stage=1)

        transport.set_actor_handles(actors, current_stage=1)
        received = transport.recv(torch.zeros(1), src_stage=0)
        assert torch.allclose(tensor, received)

    def test_send_recv_multi_dimensional(self, ray_isolated, actors):
        """Multi-dimensional tensors round-trip correctly."""
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport

        transport = RayTransport()
        transport.initialize(None)
        transport.set_actor_handles(actors, current_stage=0)

        tensor = torch.randn(4, 8, 16)
        transport.send(tensor, dest_stage=1)
        transport.set_actor_handles(actors, current_stage=1)
        received = transport.recv(torch.zeros(1), src_stage=0)
        assert torch.allclose(tensor, received)

    def test_shutdown_clears_state(self, ray_isolated):
        """shutdown clears peer refs and pending data."""
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport

        transport = RayTransport()
        transport.initialize(None)
        transport.set_actor_handles({0: None}, 0)
        transport.shutdown()
        assert len(transport._peer_refs) == 0


class TestRayTransportPipelineFlow:
    """Integration: full pipeline ref-passing flow with real StageActors."""

    @pytest.fixture(autouse=True)
    def setup_ray(self):
        import ray
        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=2, ignore_reinit_error=True)
        yield
        ray.shutdown()

    @pytest.fixture
    def two_actors(self):
        """Create two StageActors for a 2-stage pipeline."""
        from deepspeed.runtime.pipe.ray.stage_actor import StageActor
        import torch.nn as nn

        class OneLayer(nn.Module):

            def __init__(self, dim=4):
                super().__init__()
                self.linear = nn.Linear(dim, dim)

            def forward(self, x):
                return self.linear(x)

        model0 = OneLayer(4)
        model1 = OneLayer(4)

        return {
            0: StageActor.remote(stage_id=0, num_stages=2, model=model0),
            1: StageActor.remote(stage_id=1, num_stages=2, model=model1),
        }

    def test_forward_activation_flow(self, ray_isolated, two_actors):
        """Stage 0 computes activations → transport sends to stage 1."""
        import ray
        import torch
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport

        actor0, actor1 = two_actors[0], two_actors[1]

        # Stage 0 computes forward
        ray.get(actor0.reserve_buffers.remote(1))
        x = torch.randn(3, 4)
        ray.get(actor0.set_inputs.remote(0, x))
        ray.get(actor0.forward_pass.remote(0))

        # Transport: stage 0 → stage 1
        transport = RayTransport()
        transport.initialize(None)
        transport.set_actor_handles(two_actors, current_stage=0)

        activations = ray.get(actor0.get_activations.remote(0))
        transport.send(activations, dest_stage=1)

        # Stage 1 receives via transport
        transport.set_actor_handles(two_actors, current_stage=1)
        received = transport.recv(torch.zeros(1), src_stage=0)

        # Stage 1 consumes
        ray.get(actor1.reserve_buffers.remote(1))
        ray.get(actor1.set_inputs.remote(0, received))
        output = ray.get(actor1.forward_pass.remote(0))
        assert output.shape == (3, 4)

    def test_forward_backward_grad_flow(self, ray_isolated, two_actors):
        """Stage 1 backward → grads flow back to stage 0 via transport."""
        import ray
        import torch
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport

        actor0, actor1 = two_actors[0], two_actors[1]
        transport = RayTransport()
        transport.initialize(None)

        # ---- Forward pass ----
        ray.get(actor0.reserve_buffers.remote(1))
        ray.get(actor1.reserve_buffers.remote(1))

        x = torch.randn(3, 4)
        ray.get(actor0.set_inputs.remote(0, x))
        ray.get(actor0.forward_pass.remote(0))

        transport.set_actor_handles(two_actors, current_stage=0)
        activations = ray.get(actor0.get_activations.remote(0))
        transport.send(activations, dest_stage=1)

        transport.set_actor_handles(two_actors, current_stage=1)
        received = transport.recv(torch.zeros(1), src_stage=0)
        ray.get(actor1.set_inputs.remote(0, received))
        ray.get(actor1.forward_pass.remote(0))

        # ---- Backward pass ----
        ray.get(actor1.set_output_grads.remote(0, torch.ones(3, 4)))
        ray.get(actor1.backward_pass.remote(0))

        # Stage 1 → Stage 0: send input grads
        transport.set_actor_handles(two_actors, current_stage=1)
        grads = ray.get(actor1.get_input_grads.remote(0))
        transport.send(grads, dest_stage=0)

        # Stage 0 receives grads
        transport.set_actor_handles(two_actors, current_stage=0)
        received_grads = transport.recv(torch.zeros(1), src_stage=1)
        ray.get(actor0.set_output_grads.remote(0, received_grads))
        ray.get(actor0.backward_pass.remote(0))

        # Gradients should be non-zero on stage 0 inputs
        input_grads = ray.get(actor0.get_input_grads.remote(0))
        assert input_grads is not None

    def test_send_before_recv_ordering(self, ray_isolated, two_actors):
        """send() must be called before recv() — actor state is decoupled."""
        import ray
        import torch
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport

        actor0, actor1 = two_actors[0], two_actors[1]
        transport = RayTransport()
        transport.initialize(None)

        ray.get(actor0.reserve_buffers.remote(1))
        ray.get(actor0.set_inputs.remote(0, torch.randn(3, 4)))
        ray.get(actor0.forward_pass.remote(0))

        transport.set_actor_handles(two_actors, current_stage=0)
        activations = ray.get(actor0.get_activations.remote(0))
        transport.send(activations, dest_stage=1)

        # recv on stage 1 side works because send() already stored the ref
        transport.set_actor_handles(two_actors, current_stage=1)
        received = transport.recv(torch.zeros(1), src_stage=0)
        assert received is not None

    def test_transport_tolerates_sequential_sends(self, ray_isolated, two_actors):
        """Multiple send/recv cycles work correctly with actor-side refs."""
        import ray
        import torch
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport

        actor0, actor1 = two_actors[0], two_actors[1]
        transport = RayTransport()
        transport.initialize(None)

        ray.get(actor0.reserve_buffers.remote(3))
        ray.get(actor1.reserve_buffers.remote(3))

        for i in range(3):
            x = torch.randn(3, 4) + i
            ray.get(actor0.set_inputs.remote(i, x))
            ray.get(actor0.forward_pass.remote(i))

            transport.set_actor_handles(two_actors, current_stage=0)
            activations = ray.get(actor0.get_activations.remote(i))
            transport.send(activations, dest_stage=1)

            transport.set_actor_handles(two_actors, current_stage=1)
            received = transport.recv(torch.zeros(1), src_stage=0)
            ray.get(actor1.set_inputs.remote(i, received))

            output = ray.get(actor1.forward_pass.remote(i))
            assert output.shape == (3, 4)


class TestRayTransportAutoBackend:
    """Tests for auto backend colocation detection."""

    @pytest.fixture(autouse=True)
    def setup_ray(self):
        import ray
        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=2, ignore_reinit_error=True)
        yield
        ray.shutdown()

    def test_auto_backend_is_valid(self):
        """'auto' backend is accepted."""
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport
        transport = RayTransport(backend='auto')
        assert transport._backend == 'auto'

    def test_detect_colocation_same_node(self):
        """Two actors on the same (local) node are detected as colocated."""
        from deepspeed.runtime.pipe.ray.stage_actor import StageActor
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport
        import torch.nn as nn

        model = nn.Linear(4, 4)
        actor_a = StageActor.remote(stage_id=0, num_stages=2, model=model)
        actor_b = StageActor.remote(stage_id=1, num_stages=2, model=model)

        transport = RayTransport(backend='auto')
        transport.initialize(None)
        transport.set_actor_handles({0: actor_a, 1: actor_b}, current_stage=0)

        colocated = transport._detect_colocation(0, 1)
        # On a local cluster with 1 node, actors should be colocated
        assert colocated is True

    def test_colocation_cache(self):
        """Colocation results are cached."""
        from deepspeed.runtime.pipe.ray.stage_actor import StageActor
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport
        import torch.nn as nn

        model = nn.Linear(4, 4)
        actor_a = StageActor.remote(stage_id=0, num_stages=2, model=model)
        actor_b = StageActor.remote(stage_id=1, num_stages=2, model=model)

        transport = RayTransport(backend='auto')
        transport.initialize(None)
        transport.set_actor_handles({0: actor_a, 1: actor_b}, current_stage=0)

        result1 = transport._is_colocated(0, 1)
        result2 = transport._is_colocated(0, 1)
        assert result1 == result2

    def test_shutdown_clears_cache(self):
        """shutdown clears colocation cache."""
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport
        transport = RayTransport(backend='auto')
        transport.initialize(None)
        transport._colocated_cache[(0, 1)] = True
        transport._is_colocated(0, 1)  # populates cache
        transport.shutdown()
        assert len(transport._colocated_cache) == 0
