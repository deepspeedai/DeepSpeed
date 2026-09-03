# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from packaging import version as pkg_version


def check_tb_availability():
    try:
        # torch.utils.tensorboard will fail if `tensorboard` is not available,
        # see their docs for more details: https://pytorch.org/docs/1.8.0/tensorboard.html
        import tensorboard  # noqa: F401 # type: ignore
    except ImportError:
        print('If you want to use tensorboard logging, please `pip install tensorboard`')
        raise


def check_wandb_availability():
    try:
        import wandb  # noqa: F401 # type: ignore
    except ImportError:
        print(
            'If you want to use wandb logging, please `pip install wandb` and follow the instructions at https://docs.wandb.ai/quickstart'
        )
        raise


def check_comet_availability():
    try:
        import comet_ml
        comet_version = pkg_version.parse(comet_ml.__version__)
        if comet_version < pkg_version.Version("3.41.0"):
            raise ImportError("`comet_ml` must have at least version 3.41.0")
    except ImportError:
        print('If you want to use comet logging, please `pip install "comet_ml>=3.41.0"`')
        raise


def _json_safe_int(value, default):
    if value is None:
        return default
    return int(value)


def _parallel_int(getter, default):
    # TP/DP groups are only created when those parallelisms are configured.
    try:
        return _json_safe_int(getter(), default)
    except (AssertionError, RuntimeError, ValueError, TypeError):
        return default


def _offload_device(offload_cfg):
    if offload_cfg is None:
        return None
    device = getattr(offload_cfg, "device", None)
    if device is None:
        return None
    device_name = device.value if hasattr(device, "value") else str(device)
    if device_name == 'none':
        return None
    return device_name


def _pipeline_parallel_rank(engine):
    if not getattr(engine, "pipeline_parallelism", False) and getattr(engine, "mpu", None) is None:
        return 0
    mpu = getattr(engine, "mpu", None)
    if mpu is None:
        return 0
    if hasattr(mpu, "get_pipeline_model_parallel_rank"):
        return int(mpu.get_pipeline_model_parallel_rank())
    if hasattr(mpu, "get_pipe_parallel_rank"):
        return int(mpu.get_pipe_parallel_rank())
    return 0


def collect_monitor_config(engine):
    """Build a JSON-safe snapshot of ZeRO / precision / parallelism settings."""
    from deepspeed.utils import groups
    from deepspeed.utils.bwc import bwc_pipeline_parallel_world_size
    import deepspeed.comm as dist

    world_size_default = dist.get_world_size() if dist.is_initialized() else 1
    rank_default = dist.get_rank() if dist.is_initialized() else 0

    return {
        'zero_stage': int(engine.zero_optimization_stage()),
        'offload_optimizer': _offload_device(engine.zero_offload_optimizer()),
        'offload_param': _offload_device(engine.zero_offload_param()),
        'fp16': bool(engine.fp16_enabled()),
        'bf16': bool(engine.bfloat16_enabled()),
        'train_batch_size': int(engine.train_batch_size()),
        'train_micro_batch_size_per_gpu': int(engine.train_micro_batch_size_per_gpu()),
        'gradient_accumulation_steps': int(engine.gradient_accumulation_steps()),
        'data_parallel_world_size': _parallel_int(groups.get_data_parallel_world_size, world_size_default),
        'tensor_parallel_world_size': _parallel_int(groups.get_tensor_model_parallel_world_size, 1),
        'pipeline_parallel_world_size': int(bwc_pipeline_parallel_world_size(engine.mpu)),
        'sequence_parallel_world_size': int(groups._get_sequence_parallel_world_size()),
        'data_parallel_rank': _parallel_int(groups.get_data_parallel_rank, rank_default),
        'tensor_parallel_rank': _parallel_int(groups.get_tensor_model_parallel_rank, 0),
        'pipeline_parallel_rank': _pipeline_parallel_rank(engine),
        'sequence_parallel_rank': int(groups._get_sequence_parallel_rank()),
    }
