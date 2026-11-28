# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .placement import RayTopology

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

if HAS_RAY:
    from .stage_actor import StageActor
    from .ray_executor import RayActorExecutor
    from .ray_transport import RayTransport
else:
    StageActor = None
    RayActorExecutor = None
    RayTransport = None
