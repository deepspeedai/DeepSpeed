# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tests for StageActor Ray remote class.

StageActor wraps a single pipeline stage's model layers as a Ray actor.
Tests verify forward/backward pass, buffer management, and state checkpointing.
"""

import torch
import torch.nn as nn
import pytest

pytest.importorskip("ray", reason="Ray is not installed")


class SimpleTwoLayer(nn.Module):
    """Two-layer model for testing StageActor."""

    def __init__(self, input_dim=4, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestStageActorUnit:
    """Unit tests for StageActor remote calls."""

    def _create_actor(self, model, optimizer=None):
        from deepspeed.runtime.pipe.ray.stage_actor import StageActor
        return StageActor.remote(stage_id=0, num_stages=2, model=model, optimizer=optimizer)

    def test_create_actor(self, ray_isolated):
        """StageActor can be created and returns correct stage info."""
        import ray
        model = SimpleTwoLayer()
        actor = self._create_actor(model)

        stage_id = ray.get(actor.get_stage_id.remote())
        num_stages = ray.get(actor.get_num_stages.remote())
        is_first = ray.get(actor.is_first_stage.remote())
        is_last = ray.get(actor.is_last_stage.remote())

        assert stage_id == 0
        assert num_stages == 2
        assert is_first is True
        assert is_last is False

    def test_reserve_buffers(self, ray_isolated):
        """reserve_buffers allocates the correct number of slots."""
        import ray
        model = SimpleTwoLayer()
        actor = self._create_actor(model)

        ray.get(actor.reserve_buffers.remote(3))
        # Forward pass should use buffer 0
        ray.get(actor.set_inputs.remote(0, torch.randn(2, 4)))
        ray.get(actor.forward_pass.remote(0))

        outputs = ray.get(actor.get_activations.remote(0))
        assert outputs is not None
        assert outputs.shape == (2, 8)

    def test_forward_pass(self, ray_isolated):
        """forward_pass runs model on buffered input."""
        import ray
        model = SimpleTwoLayer()
        actor = self._create_actor(model)

        ray.get(actor.reserve_buffers.remote(1))
        x = torch.randn(3, 4)
        ray.get(actor.set_inputs.remote(0, x))

        output = ray.get(actor.forward_pass.remote(0))
        assert output.shape == (3, 8)

        # Compare with direct model call
        expected = model(x)
        assert torch.allclose(output, expected)

    def test_backward_pass(self, ray_isolated):
        """backward_pass computes gradients on model parameters."""
        import ray
        model = SimpleTwoLayer()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        actor = self._create_actor(model, optimizer)

        ray.get(actor.reserve_buffers.remote(1))
        x = torch.randn(3, 4)
        ray.get(actor.set_inputs.remote(0, x))
        ray.get(actor.forward_pass.remote(0))

        # Set output gradients
        grad = torch.ones(3, 8)
        ray.get(actor.set_output_grads.remote(0, grad))
        ray.get(actor.backward_pass.remote(0))

        # Gradients should be non-zero
        grads = ray.get(actor.get_input_grads.remote(0))
        assert grads is not None

    def test_optimizer_step(self, ray_isolated):
        """optimizer_step updates model parameters."""
        import ray
        model = SimpleTwoLayer()
        model_copy = SimpleTwoLayer()
        model_copy.load_state_dict(model.state_dict())

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        actor = self._create_actor(model, optimizer)

        ray.get(actor.reserve_buffers.remote(1))
        x = torch.randn(3, 4)
        ray.get(actor.set_inputs.remote(0, x))
        ray.get(actor.forward_pass.remote(0))
        ray.get(actor.set_output_grads.remote(0, torch.ones(3, 8)))
        ray.get(actor.backward_pass.remote(0))
        ray.get(actor.optimizer_step.remote())

        # Parameters should have changed
        state = ray.get(actor.get_model_state.remote())
        for name, param in model_copy.named_parameters():
            assert not torch.allclose(param.data, state[name])

    def test_load_micro_batch_first_stage(self, ray_isolated):
        """First stage loads inputs."""
        import ray
        from deepspeed.runtime.pipe.ray.stage_actor import StageActor
        model = SimpleTwoLayer()
        actor = StageActor.remote(stage_id=0, num_stages=2, model=model)

        ray.get(actor.reserve_buffers.remote(1))
        x = torch.randn(2, 4)
        ray.get(actor.load_micro_batch.remote(0, inputs=x))
        assert ray.get(actor.is_first_stage.remote())

    def test_load_micro_batch_last_stage(self, ray_isolated):
        """Last stage loads labels."""
        import ray
        from deepspeed.runtime.pipe.ray.stage_actor import StageActor

        model = SimpleTwoLayer()
        actor = StageActor.remote(stage_id=1, num_stages=2, model=model)

        ray.get(actor.reserve_buffers.remote(1))
        labels = torch.randint(0, 8, (2, ))
        ray.get(actor.load_micro_batch.remote(0, labels=labels))
        assert ray.get(actor.is_last_stage.remote())

    def test_get_activations_before_forward_raises(self, ray_isolated):
        """get_activations raises before forward pass."""
        import ray
        model = SimpleTwoLayer()
        actor = self._create_actor(model)

        ray.get(actor.reserve_buffers.remote(1))
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(actor.get_activations.remote(0))

    def test_get_set_state(self, ray_isolated):
        """get_model_state and load_model_state round-trip correctly."""
        import ray
        model = SimpleTwoLayer()
        actor = self._create_actor(model)

        state = ray.get(actor.get_model_state.remote())
        assert isinstance(state, dict)
        assert "fc1.weight" in state

        ray.get(actor.load_model_state.remote(state))
        restored = ray.get(actor.get_model_state.remote())
        for key in state:
            assert torch.allclose(state[key], restored[key])

    def test_optimizer_state_roundtrip(self, ray_isolated):
        """get_optimizer_state and load_optimizer_state round-trip."""
        import ray
        model = SimpleTwoLayer()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        actor = self._create_actor(model, optimizer)

        # Take one step to populate optimizer state
        ray.get(actor.reserve_buffers.remote(1))
        ray.get(actor.set_inputs.remote(0, torch.randn(3, 4)))
        ray.get(actor.forward_pass.remote(0))
        ray.get(actor.set_output_grads.remote(0, torch.ones(3, 8)))
        ray.get(actor.backward_pass.remote(0))
        ray.get(actor.optimizer_step.remote())

        opt_state = ray.get(actor.get_optimizer_state.remote())
        assert isinstance(opt_state, dict)
        assert "state" in opt_state


class TestStageActorRefMethods:
    """Tests for _store_pending_ref and _get_pending_ref (transport integration)."""

    @pytest.fixture(autouse=True)
    def setup_ray(self):
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=1, ignore_reinit_error=True)
        yield

    def _create_actor(self, stage_id=0, num_stages=2):
        from deepspeed.runtime.pipe.ray.stage_actor import StageActor
        model = SimpleTwoLayer()
        return StageActor.remote(stage_id=stage_id, num_stages=num_stages, model=model)

    def test_store_and_get_pending_ref(self, ray_isolated):
        """Store a ref and retrieve it — basic round-trip."""
        import ray
        actor = self._create_actor()

        tensor = torch.randn(4, 8)
        ref = ray.put(tensor)
        ray.get(actor._store_pending_ref.remote(src_stage=0, ref=ref))

        retrieved = ray.get(actor._get_pending_ref.remote(src_stage=0))
        data = ray.get(retrieved)
        assert torch.allclose(tensor, data)

    def test_get_pending_ref_consumes_entry(self, ray_isolated):
        """get_pending_ref removes the entry — second call returns None."""
        import ray
        actor = self._create_actor()

        ref = ray.put(torch.ones(2, 4))
        ray.get(actor._store_pending_ref.remote(src_stage=1, ref=ref))

        first = ray.get(actor._get_pending_ref.remote(src_stage=1))
        assert first is not None

        second = ray.get(actor._get_pending_ref.remote(src_stage=1))
        assert second is None

    def test_get_pending_ref_unused_src_returns_none(self, ray_isolated):
        """get_pending_ref for a stage that never stored returns None."""
        import ray
        actor = self._create_actor()

        result = ray.get(actor._get_pending_ref.remote(src_stage=99))
        assert result is None

    def test_multiple_stage_refs_independent(self, ray_isolated):
        """Refs from different source stages are stored independently."""
        import ray
        actor = self._create_actor()

        ref0 = ray.put(torch.tensor([1.0]))
        ref1 = ray.put(torch.tensor([2.0]))
        ray.get(actor._store_pending_ref.remote(src_stage=0, ref=ref0))
        ray.get(actor._store_pending_ref.remote(src_stage=1, ref=ref1))

        r0 = ray.get(actor._get_pending_ref.remote(src_stage=0))
        r1 = ray.get(actor._get_pending_ref.remote(src_stage=1))
        assert torch.allclose(ray.get(r0), torch.tensor([1.0]))
        assert torch.allclose(ray.get(r1), torch.tensor([2.0]))

    def test_store_overwrites_previous_pending(self, ray_isolated):
        """Storing a second ref for the same src_stage overwrites the first."""
        import ray
        actor = self._create_actor()

        ref1 = ray.put(torch.tensor([1.0]))
        ref2 = ray.put(torch.tensor([2.0]))
        ray.get(actor._store_pending_ref.remote(src_stage=0, ref=ref1))
        ray.get(actor._store_pending_ref.remote(src_stage=0, ref=ref2))

        retrieved = ray.get(actor._get_pending_ref.remote(src_stage=0))
        assert torch.allclose(ray.get(retrieved), torch.tensor([2.0]))

    def test_pending_ref_persists_across_buffers(self, ray_isolated):
        """Ref storage survives buffer allocation operations."""
        import ray
        actor = self._create_actor()

        ref = ray.put(torch.randn(3, 4))
        ray.get(actor._store_pending_ref.remote(src_stage=0, ref=ref))

        # Allocate buffers — should not interfere with pending refs
        ray.get(actor.reserve_buffers.remote(4))

        retrieved = ray.get(actor._get_pending_ref.remote(src_stage=0))
        assert torch.allclose(ray.get(retrieved), ray.get(ref))

    def test_empty_string_key(self, ray_isolated):
        """Empty string as src_stage key is treated as any other key."""
        import ray
        actor = self._create_actor()

        ref = ray.put(torch.tensor([42.0]))
        ray.get(actor._store_pending_ref.remote(src_stage="", ref=ref))

        retrieved = ray.get(actor._get_pending_ref.remote(src_stage=""))
        assert torch.allclose(ray.get(retrieved), torch.tensor([42.0]))
