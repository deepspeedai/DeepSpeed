# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tests for RayActorExecutor pipeline instruction dispatch.

RayActorExecutor dispatches schedule instructions to per-stage StageActor
Ray actors. Tests verify correct dispatch via mock actors and integration
with real Ray actors.
"""

import torch
import torch.nn as nn
import pytest

pytest.importorskip("ray", reason="Ray is not installed")


class TestRayActorExecutorInterface:
    """Verify RayActorExecutor implements the PipelineExecutor interface."""

    def test_inherits_pipeline_executor(self, ray_isolated):
        """RayActorExecutor is a PipelineExecutor subclass."""
        from deepspeed.runtime.pipe.executor import PipelineExecutor
        from deepspeed.runtime.pipe.ray.ray_executor import RayActorExecutor
        assert issubclass(RayActorExecutor, PipelineExecutor)

    def test_instruction_map_has_all_entries(self, ray_isolated):
        """RayActorExecutor instruction_map has all 10 instruction types."""
        from deepspeed.runtime.pipe import schedule
        from deepspeed.runtime.pipe.ray.ray_executor import RayActorExecutor
        from deepspeed.runtime.pipe.ray.ray_transport import RayTransport

        transport = RayTransport()
        executor = RayActorExecutor.__new__(RayActorExecutor)
        PipelineExecutor = type(executor).__bases__[0]
        PipelineExecutor.__init__(executor, transport)

        imap = executor.instruction_map
        assert len(imap) == 10
        assert schedule.ForwardPass in imap
        assert schedule.BackwardPass in imap
        assert schedule.OptimizerStep in imap


class TestRayActorExecutorIntegration:
    """Integration tests for RayActorExecutor with real StageActors."""

    @pytest.fixture(scope="function")
    def simple_model(self):

        class TwoStageModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(8, 2)

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        return TwoStageModel()

    def test_actor_creation(self, ray_isolated, simple_model):
        """StageActors can be created within a placement group."""
        import ray
        from deepspeed.runtime.pipe.ray.stage_actor import StageActor

        actor = StageActor.remote(stage_id=0, num_stages=2, model=simple_model)
        sid = ray.get(actor.get_stage_id.remote())
        assert sid == 0

    def test_two_stage_forward_flow(self, ray_isolated, simple_model):
        """Activations flow from stage 0 to stage 1 via Ray object store."""
        import ray
        from deepspeed.runtime.pipe.ray.stage_actor import StageActor

        actor0 = StageActor.remote(stage_id=0, num_stages=2, model=simple_model)
        actor1 = StageActor.remote(stage_id=1, num_stages=2, model=simple_model)

        ray.get(actor0.reserve_buffers.remote(1))
        x = torch.randn(3, 4)
        ray.get(actor0.set_inputs.remote(0, x))
        ray.get(actor0.forward_pass.remote(0))

        activations = ray.get(actor0.get_activations.remote(0))
        ray.get(actor1.reserve_buffers.remote(1))
        ray.get(actor1.set_inputs.remote(0, activations))

        output = ray.get(actor1.forward_pass.remote(0))
        assert output.shape == (3, 2)

    def test_two_stage_backward_flow(self, ray_isolated, simple_model):
        """Gradients flow from stage 1 to stage 0."""
        import ray
        from deepspeed.runtime.pipe.ray.stage_actor import StageActor

        model0 = simple_model
        opt0 = torch.optim.SGD(model0.parameters(), lr=0.01)

        class CopyModel(nn.Module):

            def __init__(self, source):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(8, 2)
                self.load_state_dict(source.state_dict())

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        model1 = CopyModel(simple_model)
        opt1 = torch.optim.SGD(model1.parameters(), lr=0.01)

        actor0 = StageActor.remote(stage_id=0, num_stages=2, model=model0, optimizer=opt0)
        actor1 = StageActor.remote(stage_id=1, num_stages=2, model=model1, optimizer=opt1)

        ray.get(actor0.reserve_buffers.remote(1))
        ray.get(actor1.reserve_buffers.remote(1))

        x = torch.randn(3, 4)
        ray.get(actor0.set_inputs.remote(0, x))
        ray.get(actor0.forward_pass.remote(0))
        activations = ray.get(actor0.get_activations.remote(0))
        ray.get(actor1.set_inputs.remote(0, activations))
        ray.get(actor1.forward_pass.remote(0))

        ray.get(actor1.set_output_grads.remote(0, torch.ones(3, 2)))
        ray.get(actor1.backward_pass.remote(0))
        grads = ray.get(actor1.get_input_grads.remote(0))
        ray.get(actor0.set_output_grads.remote(0, grads))
        ray.get(actor0.backward_pass.remote(0))

        input_grads = ray.get(actor0.get_input_grads.remote(0))
        assert input_grads is not None
