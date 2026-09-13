# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

# By default, PyTorch's CUDA availability check (cudaGetDeviceCount/cuInit)
# creates a CUDA context, which poisons fork()-based multiprocessing once
# DeepSpeed probes op compatibility at import time. Opt into PyTorch's
# NVML-based availability check so importing DeepSpeed never creates a CUDA
# context, before importing torch or anything that may query CUDA.
# setdefault() preserves an explicit user setting. See issue #7918.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

import argparse
import sys
import types
import json
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from packaging import version as pkg_version

# Skip Triton import for AMD due to pytorch-triton-rocm module breaking device API in DeepSpeed
if not (hasattr(torch.version, 'hip') and torch.version.hip is not None):
    try:
        import triton  # noqa: F401 # type: ignore
        HAS_TRITON = True
    except ImportError:
        HAS_TRITON = False
else:
    HAS_TRITON = False

from . import ops
from . import module_inject

from .accelerator import get_accelerator
from .constants import TORCH_DISTRIBUTED_DEFAULT_PORT
from .runtime.engine import DeepSpeedEngine, DeepSpeedOptimizerCallable, DeepSpeedSchedulerCallable
from .runtime.engine import ADAM_OPTIMIZER, LAMB_OPTIMIZER, MUON_OPTIMIZER
from .runtime.base_optimizer import DeepSpeedOptimizer
from .runtime.dataloader import DeepSpeedDataLoader
from .runtime.hybrid_engine import DeepSpeedHybridEngine
from .runtime.pipe.engine import PipelineEngine
from .inference.engine import InferenceEngine
from .inference.config import DeepSpeedInferenceConfig
from .runtime.lr_schedules import add_tuning_arguments
from .runtime.config import DeepSpeedConfig, DeepSpeedConfigError
from .runtime.activation_checkpointing import checkpointing
from .ops.transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig
from .module_inject import replace_transformer_layer, revert_transformer_layer, set_autotp_mode

from .utils import log_dist, OnDevice, logger
from .comm.comm import init_distributed

from .runtime import zero, domino
from .runtime.compiler import is_compile_supported

from .pipe import PipelineModule

from .git_version_info import version, git_hash, git_branch
from .runtime.tensor_parallel.init_utils import (load_ds_config, merge_tp_model_init_into_config,
                                                 record_tp_model_init_args)


def _parse_version(version_str):
    '''Parse a version string and extract the major, minor, and patch versions.'''
    ver = pkg_version.parse(version_str)
    return ver.major, ver.minor, ver.micro


# Export version information
__version__ = version
__version_major__, __version_minor__, __version_patch__ = _parse_version(__version__)
__git_hash__ = git_hash
__git_branch__ = git_branch

# Set to torch's distributed package or deepspeed.comm based inside DeepSpeedEngine init
dist = None

# Projections whose *output* dimension is blocked by heads, because the per-head split is on
# dim 0 of the weight. Standard attention blocks Q/K/V as `[num_heads * head_dim, hidden]`. MLA
# blocks its two up-projections instead: `q_b_proj` is `[num_heads * (qk_nope + qk_rope), rank]`
# and `kv_b_proj` is `[num_heads * (qk_nope + v_head_dim), rank]`, which is the split GLM-5's
# "Muon Split" applies. The output projection is deliberately absent everywhere: its head
# structure is on the input dimension, so splitting dim 0 would cut across the wrong axis, and
# with the usual hidden == num_heads * head_dim it still divides evenly, i.e. silently wrong.
_QUERY_HEAD_LEAVES = ("q_proj", "query", "wq")
_KV_HEAD_LEAVES = ("k_proj", "key", "wk", "v_proj", "value", "wv")
# MLA up-projections. Without a q_lora_rank there is no q_a/q_b pair and the query
# up-projection is a plain `q_proj` (DeepSeek-V2-Lite), so `q_proj` can be either kind and
# is resolved by shape rather than by which config fields happen to exist.
_MLA_Q_LEAVES = ("q_b_proj", )
_MLA_KV_LEAVES = ("kv_b_proj", )
# A single matrix holding Q, K and V, or an MLA down-projection that mixes latent and rope
# components. Neither splits into uniform heads, so leave them on the full-matrix path.
_FUSED_QKV_LEAVES = ("qkv_proj", "query_key_value", "c_attn", "in_proj_qkv", "wqkv")
_MLA_DOWN_LEAVES = ("q_a_proj", "kv_a_proj", "kv_a_proj_with_mqa")

QUERY, KV, MLA_Q, MLA_KV, NOT_HEAD_BLOCKED = "query", "kv", "mla_q", "mla_kv", "not-head-blocked"


def _layer_shape(param: torch.Tensor):
    """The parameter's shape as a layer, rather than as a ZeRO-3 partition.

    Under ``deepspeed.zero.Init`` a partitioned parameter's data is a flat placeholder -
    ``torch.Size([0])`` on the ranks that do not hold it - and the shape it has as a layer is
    recorded as ``ds_shape``. Reading ``param.shape`` there sees a 1-D tensor for every
    parameter in the model.
    """
    ds_shape = getattr(param, "ds_shape", None)
    return tuple(param.shape) if ds_shape is None else tuple(ds_shape)


def _per_head_muon_meta(model: torch.nn.Module):
    """The head-count reader and the config the widths come from, built once per model.

    AutoTPMeta.from_model_config is the repo's single source of truth for these counts: it
    descends into text_config and probes the several spellings models use (num_heads, n_head,
    attention_heads, ...) rather than assuming one attribute name. It is loop-invariant, so it
    is built here rather than per parameter.
    """
    model_config = getattr(model, "config", None)
    if model_config is None:
        return None, None

    from .module_inject.tp_shard import AutoTPMeta

    meta = AutoTPMeta.from_model_config(model_config)
    if meta.num_attention_heads is None:
        return None, None
    return meta, getattr(model_config, "text_config", model_config)


def _leaf_module_name(param_name: str) -> str:
    """`model.layers.0.self_attn.q_proj.weight` -> `q_proj`.

    Matching the leaf rather than the whole path keeps generic names from matching by accident:
    `dense` appears in both `attention.output.dense` and `intermediate.dense`, and an MLP matrix
    tagged with a head count would be split on a dimension that has no heads in it.
    """
    parts = param_name.split(".")
    return parts[-2].lower() if len(parts) >= 2 else parts[-1].lower()


def _classify_leaf(leaf: str):
    """Which kind of attention matrix this leaf name claims to be, or None if it claims none.

    A name is a claim, not a layout. What the leaf resolves to is decided later, by the shape.
    """
    if any(leaf.startswith(k) for k in _FUSED_QKV_LEAVES) or any(leaf.startswith(k) for k in _MLA_DOWN_LEAVES):
        return NOT_HEAD_BLOCKED
    if any(leaf.startswith(k) for k in _MLA_Q_LEAVES):
        return MLA_Q
    if any(leaf.startswith(k) for k in _MLA_KV_LEAVES):
        return MLA_KV
    if any(leaf.startswith(k) for k in _KV_HEAD_LEAVES):
        return KV
    if any(leaf.startswith(k) for k in _QUERY_HEAD_LEAVES):
        return QUERY
    return None


def _standard_head_dim(text_config, num_attention_heads: int):
    """`head_dim`, or the value it is defined as when a config leaves it out."""
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is not None:
        return head_dim
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is not None and num_attention_heads:
        quotient, remainder = divmod(hidden_size, num_attention_heads)
        if remainder == 0:
            return quotient
    return None


def _geometry_candidates(kind, meta, text_config):
    """Every (heads, per-head width) this leaf could plausibly have, each labelled.

    Candidates are collected rather than chosen. `q_proj` is the case that matters: on an MLA
    model without a q_lora_rank it is the query up-projection, and on an ordinary model it is
    the standard query projection, and a config can carry the fields for both. Deciding by the
    order the branches are written makes the outcome depend on which fields happen to exist;
    collecting both and letting the shape confirm one makes it depend on the model.
    """
    num_attention_heads = meta.num_attention_heads
    num_kv_heads = meta.num_kv_heads or num_attention_heads
    head_dim = _standard_head_dim(text_config, num_attention_heads)
    qk_nope = getattr(text_config, "qk_nope_head_dim", None)
    qk_rope = getattr(text_config, "qk_rope_head_dim", None)
    v_head_dim = getattr(text_config, "v_head_dim", None)

    candidates = []
    if kind in (QUERY, MLA_Q) and qk_nope is not None and qk_rope is not None:
        candidates.append((num_attention_heads, qk_nope + qk_rope, "mla-q"))
    if kind == MLA_KV and qk_nope is not None and v_head_dim is not None:
        candidates.append((num_attention_heads, qk_nope + v_head_dim, "mla-kv"))
    if kind == QUERY and head_dim is not None:
        candidates.append((num_attention_heads, head_dim, "head-dim"))
    if kind == KV and head_dim is not None:
        candidates.append((num_kv_heads, head_dim, "head-dim"))
    return [c for c in candidates if c[0] and c[0] >= 1 and c[1] and c[1] >= 1]


def _confirm(param: torch.Tensor, candidates):
    """Resolve candidates against the shape. Returns (num_heads or None, reason).

    Only an exact `rows == heads * width` confirms a candidate. Several candidates can confirm
    at once; that is only an ambiguity if they disagree on the head count, which is the whole
    output, so agreeing candidates are not a conflict.
    """
    shape = _layer_shape(param)
    if len(shape) != 2:
        return None, "not-2d"
    if not candidates:
        return None, "no-candidate-geometry"

    rows = shape[0]
    exact = [c for c in candidates if rows == c[0] * c[1]]
    if not exact:
        return None, "width-mismatch"

    head_counts = {c[0] for c in exact}
    if len(head_counts) > 1:
        return None, "ambiguous:" + "/".join(f"{c[2]}={c[0]}" for c in sorted(exact, key=lambda c: c[2]))
    return exact[0][0], exact[0][2]


def _attention_head_count(param_name: str, param: torch.Tensor, model: torch.nn.Module):
    """Heads this projection splits into on dim 0, or None to leave it on the full-matrix path.

    Takes the model for callers holding one parameter. `set_optimizer_flags` builds the reader
    once and calls `_resolve_attention_head_count` directly, since it is loop-invariant.
    """
    meta, text_config = _per_head_muon_meta(model)
    if meta is None:
        return None
    num_heads, _ = _resolve_attention_head_count(param_name, param, meta, text_config)
    return num_heads


def _resolve_attention_head_count(param_name: str, param: torch.Tensor, meta, text_config):
    """As above, with the reason, so callers can report why a parameter was not tagged."""
    kind = _classify_leaf(_leaf_module_name(param_name))
    if kind is None:
        return None, "not-attention"
    if kind == NOT_HEAD_BLOCKED:
        return None, NOT_HEAD_BLOCKED
    return _confirm(param, _geometry_candidates(kind, meta, text_config))


def _report_per_head_tagging(tagged: dict, skipped: dict) -> None:
    """Report what an explicit `per_head_muon: true` actually did.

    An opt-in that silently does nothing is the failure this guards. The systemic case is
    tensor parallelism: the config describes the whole model while each rank holds a shard, so
    every projection fails its width check and the feature is off model-wide while the user
    believes it is on. That is an error rather than a warning, because there is no partial
    result to keep.
    """
    if not tagged:
        raise ValueError("per_head_muon is enabled but no attention projection could be tagged. Per-head "
                         "Newton-Schulz is therefore inactive for every parameter. Likely causes: the model "
                         "arrives already sharded by an external tensor-parallel implementation, so each "
                         "rank holds a shard whose width no longer matches the config; the architecture's "
                         "attention layout is not recognized; or the model has no attention projections. "
                         "AutoTP is not one of the causes - it partitions after this runs, and the counts "
                         "are re-resolved against the shards afterwards. Unset per_head_muon to train "
                         f"without it. Leaves examined: {dict(sorted(skipped.items())) or 'none'}")

    unrecognized = {
        leaf: reason
        for leaf, reason in skipped.items() if reason not in (NOT_HEAD_BLOCKED, "not-attention")
    }
    if unrecognized:
        logger.warning(
            "per_head_muon: %s matched an attention name but no candidate geometry confirmed them; "
            "they stay on the full-matrix path", dict(sorted(unrecognized.items())))
    logger.info("per_head_muon: tagged %s", dict(sorted(tagged.items())))


def _layer_shape(param: torch.Tensor):
    """The parameter's shape as a layer, rather than as a ZeRO-3 partition.

    Under ``deepspeed.zero.Init`` a partitioned parameter's data is a flat placeholder -
    ``torch.Size([0])`` on the ranks that do not hold it - and the shape it has as a layer is
    recorded as ``ds_shape``. Reading ``param.shape`` there sees a 1-D tensor for every
    parameter in the model.
    """
    ds_shape = getattr(param, "ds_shape", None)
    return tuple(param.shape) if ds_shape is None else tuple(ds_shape)


def set_optimizer_flags(config_class: DeepSpeedConfig, model: torch.nn.Module) -> None:
    if config_class.optimizer_name == MUON_OPTIMIZER:
        per_head = bool((config_class.optimizer_params or {}).get("per_head_muon", False))
        meta, text_config = _per_head_muon_meta(model) if per_head else (None, None)
        tagged: dict = {}
        skipped: dict = {}

        for name, p in model.named_parameters():
            # Muon is defined on matrices, so the test is on the layer's shape. `zero.Init`
            # makes every parameter report as 1-D, which would switch Muon off for the whole
            # model without anything saying so.
            if len(_layer_shape(p)) >= 2 and not any(keyword in name.lower() for keyword in ("embed", "lm_head")):
                setattr(p, "use_muon", True)
            else:
                setattr(p, "use_muon", False)

            num_heads = None
            if per_head and p.use_muon and meta is not None:
                num_heads, reason = _resolve_attention_head_count(name, p, meta, text_config)
                leaf = _leaf_module_name(name)
                if num_heads is not None:
                    tagged[leaf] = f"{num_heads} heads of {_layer_shape(p)[0] // num_heads} ({reason})"
                else:
                    skipped[leaf] = reason
            setattr(p, "muon_num_heads", num_heads)
            # The width, not the count, is what survives a column-parallel split; see
            # `resolve_per_head_muon_after_sharding`.
            setattr(p, "muon_head_dim", _layer_shape(p)[0] // num_heads if num_heads else None)

        if per_head:
            _report_per_head_tagging(tagged, skipped)


def resolve_per_head_muon_after_sharding(model: torch.nn.Module) -> None:
    """Re-derive head counts from the shapes the parameters actually have.

    `set_optimizer_flags` runs before the engine partitions the model, so the count it records
    is the model's, not the rank's. Column-parallel tensor parallelism splits an attention
    projection on dim 0, which is the axis the heads are on, so after the split the per-head
    width is unchanged and the head count is not. Nothing catches that on its own: with tp=2 a
    tag of 8 heads lands on a shard holding 4 heads' worth of rows, `out_features % num_heads`
    still divides, and Newton-Schulz runs on half of each head.

    Re-deriving the count from the width is not just a repair. A column-parallel shard holds
    whole heads, so per-head Newton-Schulz on the shard is exactly the corresponding blocks of
    per-head Newton-Schulz on the whole matrix - the split is along the same axis the batch is
    taken over. A shard whose rows are not a multiple of the width does not hold whole heads,
    and is dropped rather than guessed at.
    """
    tagged, dropped = {}, {}
    for name, p in model.named_parameters():
        head_dim = getattr(p, "muon_head_dim", None)
        if head_dim is None:
            continue
        leaf = _leaf_module_name(name)
        rows = _layer_shape(p)[0]
        if rows % head_dim:
            setattr(p, "muon_num_heads", None)
            dropped[leaf] = f"{rows} rows do not divide into heads of {head_dim}"
            continue
        setattr(p, "muon_num_heads", rows // head_dim)
        tagged[leaf] = f"{rows // head_dim} heads of {head_dim}"

    if not tagged and not dropped:
        return
    if dropped:
        logger.warning("per_head_muon: %s are sharded across head boundaries and stay on the full-matrix "
                       "path", dict(sorted(dropped.items())))
    if not tagged:
        raise ValueError("per_head_muon is enabled but every tagged projection is sharded across head "
                         "boundaries, so per-head Newton-Schulz is inactive for all of them. Unset "
                         f"per_head_muon to train without it. Parameters examined: {dict(sorted(dropped.items()))}")
    logger.info("per_head_muon: after sharding, tagged %s", dict(sorted(tagged.items())))


def initialize(
    args: Any = None,
    model: torch.nn.Module = None,
    optimizer: Optional[Union[Optimizer, DeepSpeedOptimizerCallable]] = None,
    model_parameters: Optional[torch.nn.Module] = None,
    training_data: Optional[torch.utils.data.Dataset] = None,
    lr_scheduler: Optional[Union[_LRScheduler, DeepSpeedSchedulerCallable]] = None,
    distributed_port: int = TORCH_DISTRIBUTED_DEFAULT_PORT,
    mpu: Any = None,
    dist_init_required: Optional[bool] = None,
    collate_fn: Optional[Callable] = None,
    config: Optional[Union[str, Dict[str, Any]]] = None,
    mesh_param: Any = None,
    config_params: Optional[Union[str, Dict[str, Any]]] = None
) -> Tuple[DeepSpeedEngine, Optional[Union[Optimizer, DeepSpeedOptimizer]], Optional[DeepSpeedDataLoader], Any]:
    """Initialize the DeepSpeed Engine.

    Arguments:
        args: an object containing local_rank and deepspeed_config fields.
            This is optional if `config` is passed.

        model: Required: nn.module class before apply any wrappers

        optimizer: Optional: a user defined Optimizer or Callable that returns an Optimizer object.
            This overrides any optimizer definition in the DeepSpeed json config.

        model_parameters: Optional: An iterable of torch.Tensors or dicts.
            Specifies what Tensors should be optimized.

        training_data: Optional: Dataset of type torch.utils.data.Dataset

        lr_scheduler: Optional: Learning Rate Scheduler Object or a Callable that takes an Optimizer and returns a Scheduler object.
            The scheduler object should define a get_lr(), step(), state_dict(), and load_state_dict() methods

        distributed_port: Optional: Master node (rank 0)'s free port that needs to be used for communication during distributed training

        mpu: Optional: A model parallelism unit object that implements
            get_{model,data}_parallel_{rank,group,world_size}()

        dist_init_required: Optional: None will auto-initialize torch distributed if needed,
            otherwise the user can force it to be initialized or not via boolean.

        collate_fn: Optional: Merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.

        config: Optional: Instead of requiring args.deepspeed_config you can pass your deepspeed config
            as an argument instead, as a path or a dictionary.

        config_params: Optional: Same as `config`, kept for backwards compatibility.

    Returns:
        A tuple of ``engine``, ``optimizer``, ``training_dataloader``, ``lr_scheduler``

        * ``engine``: DeepSpeed runtime engine which wraps the client model for distributed training.

        * ``optimizer``: Wrapped optimizer if a user defined ``optimizer`` is supplied, or if
          optimizer is specified in json config else ``None``.

        * ``training_dataloader``: DeepSpeed dataloader if ``training_data`` was supplied,
          otherwise ``None``.

        * ``lr_scheduler``: Wrapped lr scheduler if user ``lr_scheduler`` is passed, or
          if ``lr_scheduler`` specified in JSON configuration. Otherwise ``None``.
    """
    log_dist("DeepSpeed info: version={}, git-hash={}, git-branch={}".format(__version__, __git_hash__,
                                                                             __git_branch__),
             ranks=[0])

    # Disable zero.Init context if it's currently enabled
    zero.partition_parameters.shutdown_init_context()

    assert model is not None, "deepspeed.initialize requires a model"

    global dist
    from deepspeed import comm as dist
    dist_backend = get_accelerator().communication_backend_name()
    dist.init_distributed(dist_backend=dist_backend,
                          distributed_port=distributed_port,
                          dist_init_required=dist_init_required)

    ##TODO: combine reuse mpu as mesh device and vice versa
    # Set config using config_params for backwards compat
    if config is None and config_params is not None:
        config = config_params

    # Check for deepscale_config for backwards compat
    if hasattr(args, "deepscale_config") and args.deepscale_config is not None:
        logger.warning("************ --deepscale_config is deprecated, please use --deepspeed_config ************")
        if hasattr(args, "deepspeed_config"):
            assert (args.deepspeed_config
                    is None), "Not sure how to proceed, we were given both a deepscale_config and deepspeed_config"
        args.deepspeed_config = args.deepscale_config
        args.deepscale_config = None

    # Check that we have only one config passed
    if hasattr(args, "deepspeed_config") and args.deepspeed_config is not None:
        assert config is None, "Not sure how to proceed, we were given deepspeed configs in the deepspeed arguments and deepspeed.initialize() function call"
        config = args.deepspeed_config
    assert config is not None, "DeepSpeed requires --deepspeed_config to specify configuration file"

    if not isinstance(config, dict):
        config = load_ds_config(config)

    mesh_device = None
    if mesh_param:
        logger.info(f"mesh_param to Initialize mesh device: {mesh_param}")
        mesh_device = dist.initialize_mesh_device(mesh_param, ("data_parallel", "sequence_parallel"))
    #if config file has sequence parallelize and data parallelize, then use them to initialize mesh device
    else:
        if "sequence_parallel_size" in config and "data_parallel_size" in config:
            logger.info(f"config to Initialize mesh device: {config}")
            mesh_device = dist.initialize_mesh_device((config["data_parallel_size"], config["sequence_parallel_size"]), \
            ("data_parallel", "sequence_parallel"))

    merge_tp_model_init_into_config(config, mpu, mesh_param, dist)

    autotp_size = config.get("tensor_parallel", {}).get("autotp_size", 0)
    if autotp_size and autotp_size > 0:
        set_autotp_mode(training=True)
    if not isinstance(model, PipelineModule):
        config_class = DeepSpeedConfig(config, mpu, mesh_device=mesh_device)
        set_optimizer_flags(config_class, model)
        if config_class.hybrid_engine.enabled:
            engine = DeepSpeedHybridEngine(args=args,
                                           model=model,
                                           optimizer=optimizer,
                                           model_parameters=model_parameters,
                                           training_data=training_data,
                                           lr_scheduler=lr_scheduler,
                                           mpu=mpu,
                                           dist_init_required=dist_init_required,
                                           collate_fn=collate_fn,
                                           config=config,
                                           config_class=config_class)
        else:
            engine = DeepSpeedEngine(args=args,
                                     model=model,
                                     optimizer=optimizer,
                                     model_parameters=model_parameters,
                                     training_data=training_data,
                                     lr_scheduler=lr_scheduler,
                                     mpu=mpu,
                                     dist_init_required=dist_init_required,
                                     collate_fn=collate_fn,
                                     config=config,
                                     mesh_device=mesh_device,
                                     config_class=config_class)
    else:
        assert mpu is None, "mpu must be None with pipeline parallelism"
        mpu = model.mpu()
        config_class = DeepSpeedConfig(config, mpu)
        set_optimizer_flags(config_class, model)
        engine = PipelineEngine(args=args,
                                model=model,
                                optimizer=optimizer,
                                model_parameters=model_parameters,
                                training_data=training_data,
                                lr_scheduler=lr_scheduler,
                                mpu=mpu,
                                dist_init_required=dist_init_required,
                                collate_fn=collate_fn,
                                config=config,
                                config_class=config_class)

    # Restore zero.Init context if necessary
    zero.partition_parameters.restore_init_context()

    return_items = [
        engine,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler,
    ]
    return tuple(return_items)


def _add_core_arguments(parser):
    r"""Helper (internal) function to update an argument parser with an argument group of the core DeepSpeed arguments.
        The core set of DeepSpeed arguments include the following:
        1) --deepspeed: boolean flag to enable DeepSpeed
        2) --deepspeed_config <json file path>: path of a json configuration file to configure DeepSpeed runtime.

        This is a helper function to the public add_config_arguments()

    Arguments:
        parser: argument parser
    Return:
        parser: Updated Parser
    """
    group = parser.add_argument_group('DeepSpeed', 'DeepSpeed configurations')

    group.add_argument('--deepspeed',
                       default=False,
                       action='store_true',
                       help='Enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)')

    group.add_argument('--deepspeed_config', default=None, type=str, help='DeepSpeed json configuration file.')

    group.add_argument('--deepscale',
                       default=False,
                       action='store_true',
                       help='Deprecated enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)')

    group.add_argument('--deepscale_config',
                       default=None,
                       type=str,
                       help='Deprecated DeepSpeed json configuration file.')

    return parser


def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    r"""Update the argument parser to enabling parsing of DeepSpeed command line arguments.
        The set of DeepSpeed arguments include the following:
        1) --deepspeed: boolean flag to enable DeepSpeed
        2) --deepspeed_config <json file path>: path of a json configuration file to configure DeepSpeed runtime.

    Arguments:
        parser: argument parser
    Return:
        parser: Updated Parser
    """
    parser = _add_core_arguments(parser)

    return parser


def default_inference_config() -> Dict[str, Any]:
    """
        Return a default DeepSpeed inference configuration dictionary.
    """
    return DeepSpeedInferenceConfig().dict()


def init_inference(model: torch.nn.Module,
                   config: Optional[Union[str, Dict[str, Any]]] = None,
                   **kwargs: Any) -> InferenceEngine:
    """Initialize the DeepSpeed InferenceEngine.

    Description: all four cases are valid and supported in DS init_inference() API.

    # Case 1: user provides no config and no kwargs. Default config will be used.

    .. code-block:: python

        generator.model = deepspeed.init_inference(generator.model)
        string = generator("DeepSpeed is")
        print(string)

    # Case 2: user provides a config and no kwargs. User supplied config will be used.

    .. code-block:: python

        generator.model = deepspeed.init_inference(generator.model, config=config)
        string = generator("DeepSpeed is")
        print(string)

    # Case 3: user provides no config and uses keyword arguments (kwargs) only.

    .. code-block:: python

        generator.model = deepspeed.init_inference(generator.model,
                                                    tensor_parallel={"tp_size": world_size},
                                                    dtype=torch.half,
                                                    replace_with_kernel_inject=True)
        string = generator("DeepSpeed is")
        print(string)

    # Case 4: user provides config and keyword arguments (kwargs). Both config and kwargs are merged and kwargs take precedence.

    .. code-block:: python

        generator.model = deepspeed.init_inference(generator.model, config={"dtype": torch.half}, replace_with_kernel_inject=True)
        string = generator("DeepSpeed is")
        print(string)

    Arguments:
        model: Required: original nn.module object without any wrappers

        config: Optional: instead of arguments, you can pass in a DS inference config dict or path to JSON file

    Returns:
        A deepspeed.InferenceEngine wrapped model.
    """
    log_dist("DeepSpeed info: version={}, git-hash={}, git-branch={}".format(__version__, __git_hash__,
                                                                             __git_branch__),
             ranks=[0])

    # Load config_dict from config first
    if config is None:
        config = {}
    if isinstance(config, str):
        with open(config, "r") as f:
            config_dict = json.load(f)
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError(f"'config' argument expected string or dictionary, got {type(config)}")

    # Update with values from kwargs, ensuring no conflicting overlap between config and kwargs
    overlap_keys = set(config_dict.keys()).intersection(kwargs.keys())
    # If there is overlap, error out if values are different
    for key in overlap_keys:
        if config_dict[key] != kwargs[key]:
            raise ValueError(f"Conflicting argument '{key}' in 'config':{config_dict[key]} and kwargs:{kwargs[key]}")
    config_dict.update(kwargs)

    ds_inference_config = DeepSpeedInferenceConfig(**config_dict)

    engine = InferenceEngine(model, config=ds_inference_config)

    return engine


def tp_model_init(model: torch.nn.Module,
                  tp_size: int,
                  dtype: torch.dtype,
                  config: Optional[Union[str, Dict[str, Any]]] = None,
                  **kwargs: Any) -> torch.nn.Module:
    """
    Record tensor-parallel initialization arguments for training.

    Note (compatibility and initialization behavior):
    AutoTP sharding is applied during ``deepspeed.initialize(...)``. This
    function exists for backward compatibility and only records TP arguments so
    they can be validated and merged with the DeepSpeed config at initialization.
    When you use both (i.e., calling ``set_autotp_mode(training=True)`` and
    ``deepspeed.tp_model_init`` while also passing the config to
    ``deepspeed.initialize``), DeepSpeed merges the settings at initialization.
    Conflicting settings raise an error. The table below summarizes the behavior
    across input combinations.

    Inputs:
    - TPI: tp_model_init was called? (Y/N)
    - TPG: tp_model_init provided tp_group? (Y/N)
    - CFG: tensor_parallel in DeepSpeed config? (Y/N)
    - MPU: mpu passed to deepspeed.initialize()? (Y/N)

    | TPI | TPG | CFG | MPU | Outcome                               | Notes |
    |-----|-----|-----|-----|----------------------------------------|-------|
    | N   | N   | N   | N   | Error                                  | No TP intent; nothing to initialize |
    | N   | N   | N   | Y   | No AutoTP                              | mpu may be used for other MP, but TP not enabled |
    | N   | N   | Y   | N   | Init AutoTP from config                | Use config; need TP group via config-driven init |
    | N   | N   | Y   | Y   | Init AutoTP from config                | mpu used to build TP group |
    | Y   | N   | N   | N   | Error                                  | No TP group source |
    | Y   | N   | N   | Y   | Init AutoTP from tp_model_init         | Use recorded args + mpu for TP group |
    | Y   | N   | Y   | N   | Init AutoTP from config                | Fill missing from TPI; error on mismatches; need TP group source |
    | Y   | N   | Y   | Y   | Init AutoTP from config                | Fill missing from TPI; error on mismatches |
    | Y   | Y   | N   | N   | Init AutoTP from tp_model_init         | Use recorded tp_group; config absent |
    | Y   | Y   | N   | Y   | Error                                  | tp_group + mpu conflict |
    | Y   | Y   | Y   | N   | Init AutoTP from config                | Error on mismatches; use tp_group from TPI; reject mpu |
    | Y   | Y   | Y   | Y   | Error                                  | tp_group + mpu conflict |

    Field-level merge rules when both tp_model_init and config exist:
    - Canonical source: config
    - Allowed: fill missing config fields from tp_model_init
    - Error on mismatch: autotp_size, dtype, tp_group size or identity

    Extra checks:
    - If tp_group is provided, reject mpu.
    - If tp_group is not provided, require mpu (or another TP group source).
    - If tensor_parallel is absent and only tp_model_init was called, require
      a TP group source (direct tp_group or mpu).

    Args:
        model (torch.nn.Module): The model to be initialized.
        tp_size (int): The tensor parallelism size.
        dtype (torch.dtype): The data type to be used for the model.

    Returns:
        torch.nn.Module: The original model (no sharding applied here).
    """
    if hasattr(model, 'ds_autotp_parsed'):
        logger.warning("ds_autotp_parsed' attribute already exists in the model; tp_model_init is now record-only.")

    tp_group = kwargs.get("tp_group")
    record_tp_model_init_args(tp_size=tp_size, dtype=dtype, tp_group=tp_group, dist_module=dist)

    # Keep AutoTP training mode active for backward compatibility.
    set_autotp_mode(training=True)

    return model
