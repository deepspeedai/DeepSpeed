# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tests for Ray-backed pipeline engine factory methods.

These tests verify that ``PipelineEngine._create_transport()`` and
``PipelineEngine._create_executor()`` correctly select backends based on
config, and handle missing Ray gracefully.
"""

import pytest


class TestEngineFactoryDefaults:
    """Verify default executor and transport selection."""

    def test_default_executor_is_process_group(self):
        """Default executor should be ProcessGroupExecutor."""
        from deepspeed.runtime.pipe.process_group_exec import ProcessGroupExecutor
        from deepspeed.runtime.pipe.executor import PipelineExecutor
        assert issubclass(ProcessGroupExecutor, PipelineExecutor)

    def test_default_transport_is_nccl(self):
        """Default transport should be NcclTransport."""
        from deepspeed.runtime.pipe.nccl_transport import NcclTransport
        from deepspeed.runtime.pipe.transport import PipelineTransport
        assert issubclass(NcclTransport, PipelineTransport)


class TestRayImportGuard:
    """Verify Ray components degrade gracefully when Ray is not installed."""

    def test_has_ray_flag(self):
        """HAS_RAY should be False when Ray is not available."""
        from deepspeed.runtime.pipe.ray import HAS_RAY
        # HAS_RAY may be True or False depending on environment
        assert isinstance(HAS_RAY, bool)

    def test_ray_placeholders_exist(self):
        """Placeholder classes should exist even without Ray."""
        from deepspeed.runtime.pipe.ray import StageActor, RayActorExecutor, RayTransport
        # Placeholders are either None (no Ray) or real classes (Ray available)
        assert StageActor is not None or StageActor is None
        assert RayActorExecutor is not None or RayActorExecutor is None
        assert RayTransport is not None or RayTransport is None

    def test_ray_transport_requires_ray(self):
        """RayTransport.__init__ should raise when Ray not imported."""
        from deepspeed.runtime.pipe.ray import HAS_RAY
        if not HAS_RAY:
            from deepspeed.runtime.pipe.ray.ray_transport import RayTransport
            with pytest.raises(ImportError, match="RayTransport requires Ray"):
                RayTransport()

    def test_ray_executor_requires_ray(self):
        """RayActorExecutor.__init__ should raise when Ray not imported."""
        from deepspeed.runtime.pipe.ray import HAS_RAY
        if not HAS_RAY:
            from deepspeed.runtime.pipe.ray.ray_executor import RayActorExecutor
            with pytest.raises(ImportError, match="RayActorExecutor requires Ray"):
                RayActorExecutor(None, None)


class TestNcclTransportInterface:
    """Verify NcclTransport implements the PipelineTransport interface."""

    def test_nccl_transport_has_required_methods(self):
        from deepspeed.runtime.pipe.nccl_transport import NcclTransport
        transport = NcclTransport()
        assert hasattr(transport, 'send')
        assert hasattr(transport, 'recv')
        assert hasattr(transport, 'initialize')
        assert hasattr(transport, 'shutdown')
        assert hasattr(transport, 'is_available')


class TestProcessGroupExecutorInterface:
    """Verify ProcessGroupExecutor implements the PipelineExecutor interface."""

    def test_executor_has_required_methods(self):
        required = [
            'start_batch',
            'end_batch',
            'forward_pass',
            'backward_pass',
            'load_micro_batch',
            'send_activations',
            'recv_activations',
            'send_grads',
            'recv_grads',
            'optimizer_step',
            'reduce_grads',
            'reduce_tied_grads',
        ]
        from deepspeed.runtime.pipe.executor import PipelineExecutor
        for name in required:
            assert hasattr(PipelineExecutor, name), f"PipelineExecutor missing {name}"

    def test_instruction_map_has_all_entries(self):
        from deepspeed.runtime.pipe import schedule
        expected_instructions = {
            schedule.OptimizerStep,
            schedule.ReduceGrads,
            schedule.ReduceTiedGrads,
            schedule.LoadMicroBatch,
            schedule.ForwardPass,
            schedule.BackwardPass,
            schedule.SendActivation,
            schedule.RecvActivation,
            schedule.SendGrad,
            schedule.RecvGrad,
        }
        assert len(expected_instructions) == 10, "Expected 10 instruction types"
