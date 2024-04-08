# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


def bwc_tensor_model_parallel_rank(mpu=None):
    """Backwards-compatible way of querying the tensor model parallel rank from
    an ``mpu`` object.

    *Tensor* model parallelism means that tensors are physically split across
    processes. This contrasts with *pipeline* model parallelism, in which the
    layers are partitioned but tensors left intact.

    The API for tensor model parallelism has changed across versions and this
    helper provides a best-effort implementation across versions of ``mpu``
    objects.  The preferred mechanism is
    ``mpu.get_tensor_model_parallel_rank()``.

    This should "just work" with both Megatron-LM and DeepSpeed's pipeline
    parallelism.

    Args:
        mpu (model parallel unit, optional): The tensor model parallel rank.
            If ``mpu=None``, returns 0. Defaults to ``None``.

    Returns:
        int: the rank
    """
    if mpu is None:
        # No model parallelism in easy :)
        return 0

    if hasattr(mpu, 'get_tensor_model_parallel_rank'):
        # New Megatron and DeepSpeed convention (post pipeline-parallelism release)
        return mpu.get_tensor_model_parallel_rank()
    elif hasattr(mpu, 'get_slice_parallel_rank'):
        # Some DeepSpeed + pipeline parallelism versions
        return mpu.get_slice_parallel_rank()
    else:
        # Deprecated Megatron and DeepSpeed convention
        return mpu.get_model_parallel_rank()


def bwc_tensor_model_parallel_world_size(mpu=None):
    """Backwards-compatible way of querying the tensor model parallel world size.
       Similar to bwc_tensor_model_parallel_rank.
    """
    if mpu is None:
        return 1

    if hasattr(mpu, 'get_tensor_model_parallel_world_size'):
        # New Megatron and DeepSpeed convention (post pipeline-parallelism release)
        return mpu.get_tensor_model_parallel_world_size()
    elif hasattr(mpu, 'get_slice_parallel_world_size'):
        # Some DeepSpeed + pipeline parallelism versions
        return mpu.get_slice_parallel_world_size()
    else:
        # Deprecated Megatron and DeepSpeed convention
        return mpu.get_model_parallel_world_size()


def bwc_tensor_model_parallel_group(mpu=None):
    """Backwards-compatible way of querying the tensor model parallel group.
       Similar to bwc_tensor_model_parallel_rank.
    """
    if mpu is None:
        return None

    if hasattr(mpu, 'get_tensor_model_parallel_group'):
        # New Megatron and DeepSpeed convention (post pipeline-parallelism release)
        return mpu.get_tensor_model_parallel_group()
    elif hasattr(mpu, 'get_slice_parallel_group'):
        # Some DeepSpeed + pipeline parallelism versions
        return mpu.get_slice_parallel_group()
    else:
        # Deprecated Megatron and DeepSpeed convention
        return mpu.get_model_parallel_group()


def bwc_pipeline_parallel_world_size(mpu=None):
    """Backwards-compatible way of querying the pipeline parallel world size."""
    world_size = 1
    if mpu is not None:
        if hasattr(mpu, 'get_pipeline_model_parallel_world_size'):
            # New Megatron and DeepSpeed convention (post pipeline-parallelism release)
            world_size = mpu.get_pipeline_model_parallel_world_size()
        elif hasattr(mpu, 'get_pipe_parallel_world_size'):
            # DeepSpeed Topology
            world_size = mpu.get_pipe_parallel_world_size()
    return world_size


def bwc_pipeline_parallel_group(mpu=None):
    """Backwards-compatible way of querying the pipeline parallel group."""
    if mpu is None:
        return None
    if hasattr(mpu, 'get_pipeline_model_parallel_group'):
        # Megatron
        return mpu.get_pipeline_model_parallel_group()
    elif hasattr(mpu, 'get_pipe_parallel_group'):
        # DeepSpeed Topology
        return mpu.get_pipe_parallel_group()
    assert False, 'mpu does not support pipeline parallel group'
