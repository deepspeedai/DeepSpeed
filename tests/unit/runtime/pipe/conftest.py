# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Pytest fixtures for Ray-based pipeline parallelism tests.

Provides Ray cluster lifecycle management and shared model fixtures.
All fixtures gracefully degrade when Ray is not installed — Ray-dependent
tests should use ``pytest.importorskip("ray")`` at module level.
"""

import pytest
import torch
import torch.nn as nn

# ------------------------------------------------------------------
# Ray availability guard
# ------------------------------------------------------------------

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

# ------------------------------------------------------------------
# Session-scoped Ray cluster
# ------------------------------------------------------------------


@pytest.fixture(scope="session")
def ray_cluster(tmp_path_factory):
    """Start a local Ray cluster for the test session.

    The cluster runs entirely in-process with a configurable number of
    CPUs. Tests in the session share this cluster to avoid repeated
    init/shutdown overhead.

    Yields the Ray address string, then shuts down the cluster when
    all tests complete.
    """
    if not HAS_RAY:
        pytest.skip("Ray is not installed")

    if not ray.is_initialized():
        ray.init(
            num_cpus=4,
            num_gpus=0,  # CPU-only by default; GPU tests override via ray_local fixture
            ignore_reinit_error=True,
            _temp_dir=str(tmp_path_factory.mktemp("ray")),
        )

    yield ray.get_runtime_context().gcs_address

    if ray.is_initialized():
        ray.shutdown()


# ------------------------------------------------------------------
# Function-scoped Ray session
# ------------------------------------------------------------------


@pytest.fixture(scope="function")
def ray_session(ray_cluster):
    """Ensure Ray is initialized for the current test.

    This fixture reuses the session-scoped cluster. If a test previously
    called ``ray.shutdown()``, this re-initializes without restarting
    the cluster.

    Yields ``None``, then performs no cleanup (the session fixture
    handles shutdown).
    """
    if not ray.is_initialized():
        ray.init(
            num_cpus=1,
            num_gpus=0,
            ignore_reinit_error=True,
            _temp_dir="/tmp/ray",
        )
    yield
    # No teardown — session fixture handles shutdown


# ------------------------------------------------------------------
# Per-test Ray isolate: init/teardown for tests that need it
# ------------------------------------------------------------------


@pytest.fixture(scope="function")
def ray_isolated(tmp_path):
    """Start a completely fresh Ray instance for a single test.

    Use this fixture when a test calls ``ray.shutdown()`` or modifies
    global Ray state that would interfere with other tests.

    Initializes Ray with 1 CPU, then shuts down after the test.
    """
    if not HAS_RAY:
        pytest.skip("Ray is not installed")

    if ray.is_initialized():
        ray.shutdown()

    ray.init(
        num_cpus=1,
        num_gpus=0,
        ignore_reinit_error=True,
        _temp_dir=str(tmp_path),
    )
    yield
    ray.shutdown()


# ------------------------------------------------------------------
# Ray-local GPU fixture (for tests on machines with GPUs)
# ------------------------------------------------------------------


@pytest.fixture(scope="function")
def ray_local_gpu():
    """Initialize Ray with GPU support for a single test.

    Skips if no GPU is available. The fixture isolates its Ray instance
    so GPU allocation does not interfere with other tests.
    """
    if not HAS_RAY:
        pytest.skip("Ray is not installed")

    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("CUDA is not available")

    if ray.is_initialized():
        ray.shutdown()

    ray.init(num_cpus=1, num_gpus=1, ignore_reinit_error=True)
    yield
    ray.shutdown()


# ------------------------------------------------------------------
# Shared model fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def simple_two_layer():
    """A simple two-layer MLP for StageActor tests.

    Returns an uninitialized ``nn.Module`` with:
        fc1: 4→8 (Linear+ReLU) → fc2: 8→2 (Linear)
    """
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    # Initialize weights deterministically
    with torch.no_grad():
        for param in model.parameters():
            nn.init.ones_(param)
    return model


@pytest.fixture(scope="function")
def simple_two_layer_fresh():
    """A fresh copy of simple_two_layer per test function.

    Ensures tests do not share mutated model state.
    """
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    with torch.no_grad():
        for param in model.parameters():
            nn.init.ones_(param)
    return model


@pytest.fixture(scope="function")
def sgd_optimizer(simple_two_layer_fresh):
    """SGD optimizer for the simple model."""
    return torch.optim.SGD(simple_two_layer_fresh.parameters(), lr=0.01)


# ------------------------------------------------------------------
# Ray cluster with custom resource labels (heterogeneous placement)
# ------------------------------------------------------------------


@pytest.fixture(scope="session")
def ray_heterogeneous_cluster(tmp_path_factory):
    """Start a Ray cluster with custom resource labels for heterogeneous tests.

    Adds custom resources ``"accelerator_type_a"`` and ``"accelerator_type_b"``
    to simulate multi-accelerator deployments. Tests can use these labels
    in placement group bundles to verify heterogeneous resource allocation.
    """
    if not HAS_RAY:
        pytest.skip("Ray is not installed")

    if not ray.is_initialized():
        ray.init(
            num_cpus=8,
            num_gpus=0,
            resources={
                "accelerator_type_a": 4,
                "accelerator_type_b": 4,
            },
            ignore_reinit_error=True,
            _temp_dir=str(tmp_path_factory.mktemp("ray_hetero")),
        )

    yield

    if ray.is_initialized():
        ray.shutdown()


# ------------------------------------------------------------------
# Pytest configuration hooks
# ------------------------------------------------------------------


def pytest_configure(config):
    """Register custom markers for Ray pipeline tests."""
    config.addinivalue_line("markers", "ray_gpu: tests that require Ray with GPU backing")


def pytest_collection_modifyitems(config, items):
    """Auto-skip Ray GPU tests when CUDA is unavailable."""
    skip_gpu = pytest.mark.skip(reason="Ray GPU tests require CUDA")
    for item in items:
        if "ray_gpu" in item.keywords:
            if not torch.cuda.is_available():  #ignore-cuda
                item.add_marker(skip_gpu)
