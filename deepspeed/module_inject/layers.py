# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import logging
import torch
import re
from deepspeed import comm as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.tp_shard import AutoTPMeta, get_shard_size_list
from deepspeed.utils.logging import log_dist_once
from deepspeed.runtime.zero.utils import is_zero_param
from abc import ABC, abstractmethod
from typing import Iterable, Any, Optional, List, Tuple, Dict
from .fusedqkv_utils import (shard_value_with_share_qk, prepare_tp_fused_qkvw, fused_qkv_subparam_sizes,
                             set_fused_qkv_shard_state)
from deepspeed.runtime.tensor_parallel import AUTOTP_MODE
from deepspeed.checkpoint.constants import (DS_AUTOTP_UC_META, UNIVERSAL_CHECKPOINT_VERSION_KEY,
                                            UNIVERSAL_CHECKPOINT_VERSION_VALUE)
from copy import deepcopy
from typing import Union

__all__ = [
    "TensorParallel_Layer", "LinearAllreduce", "LinearLayer", "LmHeadLinearAllreduce", "Yuan_LinearAllreduce",
    "Yuan_LinearLayer", "GateUpPack_LinearLayer", "Conv_LinearALlreduce", "fused_LinearLayer", "conv_LinearLayer",
    "SubParamLinearLayer", "SubParamLinearAllreduce"
]

DEEPSPEED_AUTOTP_MODE = AUTOTP_MODE.INFERENCE
DS_IS_REPLACED_MODULE = 'ds_is_replaced_module'
DS_TENSOR_MODEL_PARALLEL = 'tensor_model_parallel'


def _normalize_uc_shape(value):
    return tuple(value) if value is not None else None


def _build_param_uc_conversion_meta(*,
                                    partition_type,
                                    partition_dim=None,
                                    sub_param_shape=None,
                                    sub_param_shard_widths=None,
                                    original_shape=None,
                                    is_bias=False,
                                    replicated=False,
                                    unsupported_reason=None):
    """Build the conversion-facing subset of parameter UC metadata.

    This is the only schema that should flow into model-level
    `UNIVERSAL_CHECKPOINT_INFO` via `collect_autotp_universal_checkpoint_info()`.
    """
    return {
        'partition_type': partition_type,
        'partition_dim': partition_dim,
        'sub_param_shape': _normalize_uc_shape(sub_param_shape),
        'sub_param_shard_widths': sub_param_shard_widths,
        'original_shape': _normalize_uc_shape(original_shape),
        'is_bias': is_bias,
        'replicated': replicated,
        'unsupported_reason': unsupported_reason,
    }


def _build_param_uc_restore_meta(*,
                                 partition_type,
                                 partition_dim=None,
                                 logical_shape=None,
                                 output_shape=None,
                                 sub_param_shape=None,
                                 sub_param_sizes=None,
                                 sub_param_shard_widths=None,
                                 partition_sizes=None,
                                 target_partition_shape=None,
                                 original_shape=None,
                                 is_bias=False,
                                 replicated=False,
                                 unsupported_reason=None):
    """Build the restore-facing parameter UC metadata.

    Restore metadata stays on the parameter object and may include details that
    are intentionally omitted from model-level conversion schema.
    """
    return {
        UNIVERSAL_CHECKPOINT_VERSION_KEY:
        UNIVERSAL_CHECKPOINT_VERSION_VALUE,
        'partition_type':
        partition_type,
        'partition_dim':
        partition_dim,
        'logical_shape':
        _normalize_uc_shape(logical_shape),
        'output_shape':
        _normalize_uc_shape(output_shape),
        'sub_param_shape':
        _normalize_uc_shape(sub_param_shape),
        'sub_param_sizes':
        _normalize_uc_shape(sub_param_sizes),
        'sub_param_shard_widths':
        sub_param_shard_widths,
        'partition_sizes':
        _normalize_uc_shape(partition_sizes),
        'target_partition_shape':
        _normalize_uc_shape(target_partition_shape),
        'original_shape':
        _normalize_uc_shape(original_shape),
        'is_bias':
        is_bias,
        'replicated':
        replicated,
        'conversion':
        _build_param_uc_conversion_meta(partition_type=partition_type,
                                        partition_dim=partition_dim,
                                        sub_param_shape=sub_param_shape,
                                        sub_param_shard_widths=sub_param_shard_widths,
                                        original_shape=original_shape,
                                        is_bias=is_bias,
                                        replicated=replicated,
                                        unsupported_reason=unsupported_reason),
    }


def get_auto_tp_mode():
    global DEEPSPEED_AUTOTP_MODE
    return DEEPSPEED_AUTOTP_MODE


def is_autotp_training_mode():
    global DEEPSPEED_AUTOTP_MODE
    return DEEPSPEED_AUTOTP_MODE == AUTOTP_MODE.TRAINING


def set_autotp_mode(training=False):
    """
    Set the DEEPSPEED_AUTOTP_MODE based on the training flag
    """
    global DEEPSPEED_AUTOTP_MODE
    if training:
        DEEPSPEED_AUTOTP_MODE = AUTOTP_MODE.TRAINING
    else:
        DEEPSPEED_AUTOTP_MODE = AUTOTP_MODE.INFERENCE


def add_bias(input, bias):
    if bias is None:
        return input
    if is_autotp_training_mode():
        # Training mode - avoid inplace to ensure correct autograd
        input = input + bias
        return input
    else:
        input += bias
        return input


class RowParallel(torch.autograd.Function):
    """
    A custom autograd function for performing row-wise parallelism.
    """

    @staticmethod
    def symbolic(graph, input):
        """Symbolic function for tracing."""
        return input

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: torch.Tensor, is_inference_mode: bool) -> torch.Tensor:
        """
        Forward pass.
        """
        ctx.group = group
        if group == None:
            return input
        if is_inference_mode:
            dist.inference_all_reduce(input, group=group)
        else:
            dist.all_reduce(input.contiguous(), group=group)
        return input

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None]:
        """
        Backward pass.
        """
        return None, grad_output, None


class AsyncColumnParallel(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: torch.Tensor, weight, bias) -> torch.Tensor:
        """
        Forward pass.
        """
        ctx.use_bias = bias is not None
        ctx.group = group
        output = torch.matmul(input, weight.transpose(-1, -2))
        if bias is not None:
            output = add_bias(output, bias)

        ctx.save_for_backward(input, weight)

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, torch.Tensor]:

        input, weight = ctx.saved_tensors
        grad_input = grad_output.matmul(weight)
        handle = dist.all_reduce(grad_input.contiguous(), group=ctx.group, async_op=True)
        grad_weight = grad_output.view(-1, grad_output.shape[-1]).t().matmul(input.view(-1, input.shape[-1]))
        grad_bias = grad_output.sum(0) if ctx.use_bias else None
        handle.wait()
        return None, grad_input, grad_weight, grad_bias


class ColumnParallel(torch.autograd.Function):
    """
    Custom autograd function for column-wise parallelism.
    """

    @staticmethod
    def symbolic(graph, input):
        """Symbolic function for tracing."""
        return dist.all_reduce(input.contiguous(), dist.get_tensor_model_parallel_group())

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        ctx.group = group
        return input

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, torch.Tensor]:
        """
        Backward pass.
        """
        if ctx.group == None:
            return None, grad_output

        dist.all_reduce(grad_output.contiguous(), group=ctx.group)
        return None, grad_output


class GatherFromTensorParallelRegion(torch.autograd.Function):
    """Gather last-dimension shards while keeping the output replicated."""

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: torch.Tensor, partition_sizes: Tuple[int,
                                                                                                ...]) -> torch.Tensor:
        """Gather the shards of a column parallel output described by ``partition_sizes``.

        The widths were resolved when the layer was built, so they need neither an extra
        collective nor a second lookup of the tensor parallel globals. Uneven shards are zero
        padded to a common width, which keeps the uniform (and faster)
        ``all_gather_into_tensor`` collective usable, and are then trimmed back.
        """
        ctx.group = group
        ctx.partition_sizes = partition_sizes
        ctx.tp_index = 0

        tp_world_size = len(partition_sizes)
        if group is None or tp_world_size == 1:
            return input

        ctx.tp_index = dist.get_rank(group=group)
        local_size = partition_sizes[ctx.tp_index]
        assert local_size == input.shape[-1], (
            f"Rank {ctx.tp_index} produced {input.shape[-1]} output features, but the partition "
            f"scheme {partition_sizes} frozen at construction expects {local_size}.")

        max_partition_size = max(partition_sizes)
        if local_size == max_partition_size:
            input_padded = input.contiguous()
        else:
            padded_shape = (*input.shape[:-1], max_partition_size)
            input_padded = input.new_zeros(padded_shape)
            input_padded[..., :local_size].copy_(input)

        buffer = input.new_empty((tp_world_size * input_padded.shape[0], *input_padded.shape[1:]))
        dist.all_gather_into_tensor(buffer, input_padded, group=group)

        shards = buffer.view(tp_world_size, *input_padded.shape)
        return torch.cat([shards[i].narrow(-1, 0, size) for i, size in enumerate(partition_sizes)], dim=-1)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None]:
        shard_offset = sum(ctx.partition_sizes[:ctx.tp_index])
        shard_size = ctx.partition_sizes[ctx.tp_index]
        grad_input = grad_output.narrow(-1, shard_offset, shard_size).contiguous()
        return None, grad_input, None


class TensorParallel_Layer(nn.Module, ABC):
    """
    A base class for model layers with  tensor parallelism support.
    This class is designed to be extended by specific layers that require distributed
    operations and parameter gather/partitioning during inference or training.

    Attributes:
        mode (str): The mode of operation[INFERENCE or TRAINING], default is "INFERENCE".
        mp_group (Optional[dist.ProcessGroup]): The process group used for model parallelism.
        tp_world_size (int): The world size of tensor parallelism, i.e., the number of parallel workers.
        tp_index (int): The rank (ID) of the current worker in tensor parallelism.
        support_training (bool): Flag indicating whether the layer supports training (default: False).
        name (Optional[str]): The name of the layer, if provided.
    """
    ##### Initialize Parameter List #####

    # keep_module_on_host determines whether to keep the module on the host.
    # Checkpoints are first loaded to the host (sometimes directly from disk to avoid filling host memory),
    # so an additional copy is unnecessary.
    keep_module_on_host: bool = False

    ##### Runtime Parameter List #####
    tp_overlap_comm: bool = False
    """ Whether to overlap communication with computation. Currently, only allreduce supports overlap. """

    def __init__(self, mp_group: Optional[dist.ProcessGroup], **kwargs: Any):
        """
        Initializes the TensorParallel_Layer with optional model parallelism group and layer name.

        Args:
            mp_group (Optional[dist.ProcessGroup]): The process group for model parallelism.
                                                    If None, no model parallelism is set.
        """
        super().__init__()
        self.support_training: bool = False
        self.defer_collectives_to_compiler: bool = False
        self.mp_group = mp_group
        if mp_group is not None:
            self.tp_world_size: int = dist.get_world_size(self.mp_group)
            self.tp_index: int = dist.get_rank(self.mp_group)
        else:
            self.tp_world_size: int = 1
            self.tp_index: int = 0

        # backward compatibility
        self.world_size = self.tp_world_size
        self.rank = self.tp_index

        self.name = getattr(self, 'name', None)
        if kwargs.get('name') is not None:
            self.name = kwargs.get('name')  # Set the layer name if provided.

        # Per-model TP metadata threaded from AutoTP; defaults for layers built outside it
        # (e.g. the from_weights back-compat constructor).
        self.tp_meta: AutoTPMeta = kwargs.get('tp_meta') or AutoTPMeta()

    @classmethod
    def set_keep_module_on_host(cls, value: bool):
        """
        Set the static variable keep_module_on_host.

        Args:
            value (bool): The new value for keep_module_on_host.
        """
        cls.keep_module_on_host = value

    @abstractmethod
    def forward(self, input):
        """
        Forward pass method. Must be implemented by subclasses to define layer-specific operations.
        """
        pass

    @abstractmethod
    def gather_params(self, params_list):
        """
        Gathers parameters across devices for distributed training. Must be implemented by subclasses in "TRAINING" mode.
        """
        pass

    @abstractmethod
    def _tp_partition(self, params_list: List[torch.Tensor]):
        """
        Partitions the parameters for tensor parallelism.
        It is necessary to ensure that this function only involves the logic of params partitioning.
        """
        pass

    def config_requires_grad(self, weight):
        if weight is not None:
            if self.is_training_mode():
                if weight.requires_grad is None:
                    weight.requires_grad = True
            else:
                weight.requires_grad = False

    def _is_bias_param(self, param):
        bias = getattr(self, 'bias', None)
        return bias is not None and param is bias

    def config_tp_params(self, weight):
        """
        Configures the weight tensor for training with tensor parallelism. This includes enabling gradients
        and associating necessary methods for parameter gathering and partitioning.

        Args:
            weight (Optional[torch.Tensor]): The weight tensor to configure for tensor parallelism.
                                              If None, no action is taken.
        """
        # # The RNG states have already been synchronized in init_inference.
        if self.is_training_mode():
            assert self.support_training, "No implementation of backward."
        if weight is not None:
            self.config_requires_grad(weight)
            weight.gather_params = self.gather_params
            weight._tp_partition = self._tp_partition
            setattr(weight, DS_TENSOR_MODEL_PARALLEL, True)
            setattr(weight, DS_IS_REPLACED_MODULE, True)

    @staticmethod
    def _shape_before_zero3_partition(param):
        """Shape of ``param`` as it was before ZeRO-3 partitioned it.

        ZeRO-3 replaces a partitioned parameter's local data with an empty 1-D
        tensor, so ``param.shape`` no longer describes the layer and indexing it
        raises. ZeRO-3 records the pre-partition shape as ``ds_shape``, which is
        what the universal-checkpoint metadata needs.
        """
        ds_shape = getattr(param, 'ds_shape', None)
        return tuple(param.shape) if ds_shape is None else tuple(ds_shape)

    def _set_param_uc_meta(self,
                           param,
                           *,
                           partition_type,
                           partition_dim=None,
                           logical_shape=None,
                           output_shape=None,
                           sub_param_shape=None,
                           sub_param_sizes=None,
                           sub_param_shard_widths=None,
                           partition_sizes=None,
                           target_partition_shape=None,
                           original_shape=None,
                           is_bias=False,
                           replicated=False,
                           unsupported_reason=None):
        if param is None:
            return
        setattr(
            param, DS_AUTOTP_UC_META,
            _build_param_uc_restore_meta(partition_type=partition_type,
                                         partition_dim=partition_dim,
                                         logical_shape=logical_shape,
                                         output_shape=output_shape,
                                         sub_param_shape=sub_param_shape,
                                         sub_param_sizes=sub_param_sizes,
                                         sub_param_shard_widths=sub_param_shard_widths,
                                         partition_sizes=partition_sizes,
                                         target_partition_shape=target_partition_shape,
                                         original_shape=original_shape,
                                         is_bias=is_bias,
                                         replicated=replicated,
                                         unsupported_reason=unsupported_reason))

    def _mark_uc_metadata(self):
        return

    def _should_materialize_tp_partition(self):
        # AutoTP partitioning should only materialize parameters when an actual
        # TP process group is present. Metadata-only construction with
        # mp_group=None should not touch device placement.
        return self.mp_group is not None

    def is_training_mode(self):
        global DEEPSPEED_AUTOTP_MODE
        return DEEPSPEED_AUTOTP_MODE == AUTOTP_MODE.TRAINING

    def _freeze_partition_sizes(self, total_size):
        """Resolve the tensor parallel split of this layer once, while the layer is built.

        The split depends on this model's kv-head/grain metadata (``self.tp_meta``), so it is
        resolved here and every later consumer -- the forward gather, the parameter gather and
        the checkpoint metadata -- reads the cached value rather than re-deriving it.
        """
        self._partition_sizes = tuple(get_shard_size_list(total_size, self.tp_world_size, self.tp_meta, self.name))
        return self._partition_sizes

    @torch.no_grad()
    def _all_gather_shards(self, shard, partition_sizes, dim):
        """Reassemble a parameter from its tensor parallel shards along ``dim``.

        ``partition_sizes`` is derived locally from the same deterministic split used to
        create the shards, so no extra collective is needed to discover the remote sizes.
        Uneven shards are zero padded to a common size, which keeps the uniform (and
        faster) ``all_gather_into_tensor`` collective usable, and are then trimmed back.
        """
        world_size = len(partition_sizes)
        assert partition_sizes[self.tp_index] == shard.shape[dim], (
            f"Rank {self.tp_index} holds {shard.shape[dim]} elements along dim {dim} of "
            f"{self.name}, but the partition scheme expects {partition_sizes[self.tp_index]}.")

        max_size = max(partition_sizes)
        padded_shape = list(shard.shape)
        padded_shape[dim] = max_size
        if shard.shape[dim] == max_size:
            padded = shard.contiguous()
        else:
            padded = shard.new_zeros(padded_shape)
            padded.narrow(dim, 0, shard.shape[dim]).copy_(shard)

        buffer = shard.new_empty((world_size * padded_shape[0], *padded_shape[1:]))
        dist.all_gather_into_tensor(buffer, padded, group=self.mp_group)

        if dim == 0 and min(partition_sizes) == max_size:
            # Shards are uniform and concatenated along dim 0, so the flat buffer is the result.
            return buffer

        shards = buffer.view(world_size, *padded_shape)
        return torch.cat([shards[i].narrow(dim, 0, size) for i, size in enumerate(partition_sizes)], dim=dim)

    def __deepcopy__(self, memo):
        # This function is designed for
        # 'mp_group' (a 'ProcessGroup') cannot be pickled during deepcopy in some usage.
        cls = self.__class__
        new_obj = cls.__new__(cls)

        for key, value in vars(self).items():
            if key == 'mp_group':
                new_obj.mp_group = self.mp_group
            else:
                setattr(new_obj, key, deepcopy(value, memo))

        memo[id(self)] = new_obj
        return new_obj

    def extra_repr(self):
        out_features, in_features = None, None
        if self.weight is not None:
            out_features, in_features = self.weight.ds_shape[-2:] if is_zero_param(
                self.weight) else self.weight.shape[-2:]
        dtype = self.weight.dtype if self.weight is not None else None
        return "in_features={}, out_features={}, bias={}, dtype={}".format(in_features, out_features, self.bias
                                                                           is not None, dtype)

    def move(self, tensor):
        # TODO: consider the timing of deletion
        # to save host resources when DP > 1。

        # keep_module_on_host is used to keep the module on the host. Checkpoints are loaded to the host first (in some
        # cases it can be done from the disk even to prevent filling host's memory), thus no need to create a new copy.
        if tensor.is_meta:
            # Keep tensor in meta device if tensor is meta.
            return tensor
        else:
            device = 'cpu' if self.__class__.keep_module_on_host else get_accelerator().current_device_name()
            return_new_copy = not self.__class__.keep_module_on_host

            # Using new tensors help in freeing memory (after split for example) was done before by calling clone().
            # Using copy=True instead of clone() will help in case of cpu --> cpu.
            # Otherwise to() will not create a new copy for the view of the full tensor, and it will not be de-referenced.
            cloned_tensor = tensor.to(device, copy=return_new_copy)

            if return_new_copy:
                # free the memory of the original tensor to reduce memory peak
                # Equivalent to directly deleting the tensor reference outside the function.
                # see https://github.com/microsoft/DeepSpeed/pull/4353
                tensor.data = torch.empty(0, device=tensor.device)
            return cloned_tensor


def configure_tensor_parallel_runtime(config):
    runtime_keys = ['tp_overlap_comm']
    for key in runtime_keys:
        if hasattr(config, key):
            setattr(TensorParallel_Layer, key, getattr(config, key))


def _get_param_uc_conversion_meta(param: torch.Tensor) -> Optional[Dict[str, Any]]:
    """Return the conversion-facing view of AutoTP UC metadata for a parameter.

    AutoTP keeps a single parameter-level metadata object with two roles:
    - top-level fields: restore-time details consumed by `universal_checkpoint.py`
    - `conversion`: conversion-time details consumed by
      `collect_autotp_universal_checkpoint_info()` and then aggregated into
      model-level `UNIVERSAL_CHECKPOINT_INFO` for `ds_to_universal.py`
    """
    meta = getattr(param, DS_AUTOTP_UC_META, None)
    if not meta:
        return None
    return meta.get('conversion', None)


def collect_autotp_universal_checkpoint_info(model: nn.Module) -> Dict[str, Any]:
    """Collect the model-level conversion schema for AutoTP universal checkpoints.

    The returned `UNIVERSAL_CHECKPOINT_INFO` is intentionally limited to the
    pattern/schema data needed during checkpoint conversion. It does not include
    restore-time per-parameter details such as `sub_param_sizes` or
    `target_partition_shape`, which stay on the parameter metadata object.
    """
    from deepspeed.checkpoint.constants import (AUTOTP_UNSUPPORTED_PARAMETER_PATTERNS, ORIGINAL_VOCAB_SIZE,
                                                PARAMETER_WITH_ROW_PARALLELISM_PATTERNS, PARAMETER_WITH_SUB_PARAMS,
                                                SUB_PARAM_SHARD_WIDTHS, TP_REPLICATED_PARAMETER_PATTERNS,
                                                UNIVERSAL_CHECKPOINT_VERSION_KEY, UNIVERSAL_CHECKPOINT_VERSION_VALUE,
                                                VOCABULARY_PARAMETER_PATTERNS)

    row_parallel_patterns = []
    sub_param_shard_widths = {}
    replicated_patterns = []
    vocabulary_patterns = []
    parameter_with_sub_params = []
    unsupported_parameter_patterns = {}
    original_vocab_size = None

    # Tied parameters are reachable under several module attributes, but the optimizer -- and
    # therefore the checkpoint -- only knows the first of those names. Publishing a pattern for
    # an alias would describe a parameter that has no slices to convert.
    canonical_names = {id(param): name for name, param in model.named_parameters()}

    for module_name, module in model.named_modules():
        marker = getattr(module, "_mark_uc_metadata", None)
        if marker is not None:
            marker()

        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if canonical_names.get(id(param), full_name) != full_name:
                continue
            pattern = rf"^{re.escape(full_name)}$"

            conversion_meta = _get_param_uc_conversion_meta(param)
            if not conversion_meta:
                # AutoTP left this parameter untouched, so it is identical across TP
                # ranks. Classify it as TP-replicated; otherwise it falls through to
                # the converter's default dim-0 concat and is wrongly expanded (e.g.
                # LayerNorm/RMSNorm weights [H] -> [H * tp_degree]).
                replicated_patterns.append(pattern)
                continue

            unsupported_reason = conversion_meta.get('unsupported_reason')
            if unsupported_reason:
                unsupported_parameter_patterns[pattern] = unsupported_reason
                continue

            if conversion_meta.get('replicated'):
                replicated_patterns.append(pattern)

            if conversion_meta.get('partition_type') == 'row' and not conversion_meta.get('is_bias', False):
                row_parallel_patterns.append(pattern)

            original_shape = conversion_meta.get('original_shape')
            partition_dim = conversion_meta.get('partition_dim')
            if (original_shape and len(original_shape) == 2 and partition_dim == 0
                    and ('embed' in full_name or 'lm_head' in full_name)):
                vocabulary_patterns.append(pattern)
                if original_vocab_size is None:
                    original_vocab_size = original_shape[0]

            sub_param_shape = conversion_meta.get('sub_param_shape')
            if sub_param_shape is not None and partition_dim is not None:
                shard_widths = conversion_meta.get('sub_param_shard_widths')
                published_shape = list(sub_param_shape)
                if shard_widths is not None:
                    # sub_param_shape is a view spec whose partition_dim entry may be the
                    # sub-parameter *count*, as in (3, -1). Publishing that count invites a
                    # reader to take it for a width, which is how a fused weight ends up merged
                    # from a single narrow slice. The recorded widths already carry the real
                    # per-sub-parameter sizes, so publish those and leave no room for the
                    # ambiguity.
                    published_shape[partition_dim] = tuple(sum(widths) for widths in shard_widths)
                    sub_param_shard_widths[pattern] = [list(widths) for widths in shard_widths]
                parameter_with_sub_params.append({
                    'patterns': [pattern],
                    'shape': published_shape,
                    'partition_dim': partition_dim,
                })

    uc_info = {
        UNIVERSAL_CHECKPOINT_VERSION_KEY: UNIVERSAL_CHECKPOINT_VERSION_VALUE,
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS: sorted(set(row_parallel_patterns)),
        TP_REPLICATED_PARAMETER_PATTERNS: sorted(set(replicated_patterns)),
        VOCABULARY_PARAMETER_PATTERNS: sorted(set(vocabulary_patterns)),
        PARAMETER_WITH_SUB_PARAMS: parameter_with_sub_params,
        AUTOTP_UNSUPPORTED_PARAMETER_PATTERNS: unsupported_parameter_patterns,
    }
    if sub_param_shard_widths:
        uc_info[SUB_PARAM_SHARD_WIDTHS] = sub_param_shard_widths
    if original_vocab_size is not None:
        uc_info[ORIGINAL_VOCAB_SIZE] = original_vocab_size
    return uc_info


class GatherReplacedLayerParams:
    """
    A context manager for gathering parameters of a replaced layer, enabling partitioning and gathering functionality
    based on the configuration of the model.
    """

    def __init__(self,
                 params: Union[Iterable[torch.Tensor], torch.Tensor],
                 module: torch.nn.Module,
                 enabled: bool = True):
        """
        Initialize the context manager to handle parameter gathering and partitioning for a replaced layer.

        Args:
            params (Iterable or torch.Tensor): A collection or single parameter to manage.
            module (torch.nn.Module): The module that these parameters belong to.
            enabled (bool): Flag indicating whether the parameter management is enabled (default: True).
        """
        self.enabled = enabled
        self.module = module
        if not enabled:
            return

        # Ensure params is a list, whether it's a single param or iterable (e.g., model.parameters())
        if isinstance(params, Iterable) and not isinstance(params, torch.Tensor):
            self.params: List[torch.Tensor] = list(params)  # Convert generators to a list for multiple iterations
        else:
            self.params: List[torch.Tensor] = [params]  # Wrap single parameter in a list for uniform processing

        # Check if the parameters belong to a replaced layer (indicated by a specific attribute)
        if not any(self._is_replaced_module_weight(p) for p in params):
            self.enabled = False
            return

    def _is_replaced_module_weight(self, param: torch.Tensor) -> bool:
        """
        Helper function to determine if a parameter belongs to a replaced module.

        Args:
            param (torch.Tensor): The parameter to check.

        Returns:
            bool: True if the parameter belongs to a replaced module, False otherwise.
        """
        return getattr(param, DS_IS_REPLACED_MODULE, False)

    def __enter__(self) -> None:
        """
        Enter the context manager. If enabled, gather parameters for the replaced module.
        """
        if self.enabled:
            self.params[0].gather_params(self.params)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Exit the context manager. If enabled, partition the parameters for the replaced module.
        """
        #TODO : Check whether there are any missing attributes.
        if self.enabled:
            self.params[0]._tp_partition(self.params)


class LinearAllreduce(TensorParallel_Layer):

    def __init__(self, module, mp_group, **kwargs):
        super(LinearAllreduce, self).__init__(mp_group, **kwargs)
        self.weight = module.weight
        self.bias = module.bias
        self._orig_weight_shape = self._shape_before_zero3_partition(module.weight)
        self._freeze_partition_sizes(self._orig_weight_shape[1])

        if self._should_materialize_tp_partition():
            self._tp_partition([self.weight, self.bias])
        self.support_training = True
        self.config_tp_params(self.weight)
        if self.bias is not None:
            # bias here is not tp params
            self.config_requires_grad(self.bias)
        self._mark_uc_metadata()

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if not self.defer_collectives_to_compiler:
            output = RowParallel.apply(self.mp_group, output, not self.is_training_mode())
        if self.bias is not None:
            output = add_bias(output, self.bias)
        return output

    @torch.no_grad()
    def gather_params(self, params_list):
        # Row parallelism only shards the weight; the bias is replicated across ranks.
        weight = params_list[0]
        if weight is None:
            return

        if self.mp_group is None or self.tp_world_size == 1:
            weight.data = weight.data.contiguous()
            return

        weight.data = self._all_gather_shards(weight, self._partition_sizes, dim=1).contiguous()

    @torch.no_grad()
    def _tp_partition(self, params_list):
        # Row parallelism shards the weight's input dimension; the bias stays replicated.
        self.uneven_partition(params_list)

        bias = params_list[1] if len(params_list) > 1 else None
        if bias is not None and self.is_training_mode():
            # Training materializes the replicated bias on the target device.
            bias.data = self.move(bias).detach()

    def uneven_partition(self, params_list):
        for idx, param in enumerate(params_list):
            if param is None or idx > 0:
                # don't slipt bias
                return
            _partition = params_list[idx].split(self._partition_sizes, dim=1)[self.tp_index]

            _partition = self.move(_partition).detach()
            params_list[idx].data = _partition

    def _mark_uc_metadata(self):
        self._set_param_uc_meta(self.weight,
                                partition_type='row',
                                partition_dim=1,
                                logical_shape=self._orig_weight_shape,
                                output_shape=(self._orig_weight_shape[0], ),
                                partition_sizes=self._partition_sizes,
                                target_partition_shape=tuple(self.weight.shape),
                                original_shape=self._orig_weight_shape)
        if self.bias is not None:
            bias_shape = self._shape_before_zero3_partition(self.bias)
            self._set_param_uc_meta(self.bias,
                                    partition_type='row',
                                    partition_dim=None,
                                    logical_shape=bias_shape,
                                    output_shape=bias_shape,
                                    original_shape=bias_shape,
                                    is_bias=True,
                                    replicated=True)


#remove kwargs from partition.
class LinearLayer(TensorParallel_Layer):

    def __init__(self, module, mp_group=None, skip_partition=False, gather_output=False, **kwargs):
        super(LinearLayer, self).__init__(mp_group, **kwargs)
        self.weight = module.weight
        self.bias = module.bias
        self.gather_output = gather_output
        self._orig_weight_shape = self._shape_before_zero3_partition(module.weight)
        self._orig_bias_shape = self._shape_before_zero3_partition(module.bias) if self.bias is not None else None
        self._freeze_partition_sizes(self._orig_weight_shape[0])
        if not skip_partition and self._should_materialize_tp_partition():
            self._tp_partition([self.weight, self.bias])
        self.support_training = True
        self.config_tp_params(self.weight)
        if self.bias is not None:
            self.config_tp_params(self.bias)
        self._mark_uc_metadata()

    def forward(self, input):
        if not self.__class__.tp_overlap_comm:
            if getattr(self, 'mp_group', None) is not None and not self.defer_collectives_to_compiler:
                input = ColumnParallel.apply(self.mp_group, input)
            output = torch.matmul(input, self.weight.transpose(-1, -2))
            if self.bias is not None:
                output = add_bias(output, self.bias)
        else:
            output = AsyncColumnParallel.apply(self.mp_group, input, self.weight, self.bias)

        if self.gather_output:
            if self.defer_collectives_to_compiler:
                # The gather changes the activation's width, so downstream ops (e.g. a depthwise
                # conv sized for the full width) only trace correctly if it happens inline. The
                # custom op is graph-capturable, unlike GatherFromTensorParallelRegion, which
                # reads gathered shard sizes back into Python. Pass the frozen shard widths so the
                # compiled custom op supports uneven TP partitions too.
                output = torch.ops.autotp.gather_from_tp_region(output, self._partition_sizes)
            else:
                output = GatherFromTensorParallelRegion.apply(self.mp_group, output, self._partition_sizes)

        return output

    @torch.no_grad()
    def gather_params(self, params_list):
        for idx, param in enumerate(params_list):
            if param is None:
                continue

            if self.mp_group is None or self.tp_world_size == 1:
                params_list[idx].data = param.data.contiguous()
                continue

            # Column parallelism shards dim 0 of both the weight and the bias, so gathering
            # along dim 0 restores the original shape.
            params_list[idx].data = self._all_gather_shards(param, self._partition_sizes, dim=0).contiguous()

    @torch.no_grad()
    def _tp_partition(self, params_list):
        self.uneven_partition(params_list)

    def uneven_partition(self, params_list):

        for idx, param in enumerate(params_list):
            if param is None:
                #split bias if provide
                return
            _partition = params_list[idx].split(self._partition_sizes, dim=0)[self.tp_index]

            _partition = self.move(_partition).detach()

            params_list[idx].data = _partition

    def _mark_uc_metadata(self):
        original_out_dim = self._orig_weight_shape[0]
        self._set_param_uc_meta(self.weight,
                                partition_type='column',
                                partition_dim=0,
                                logical_shape=self._orig_weight_shape,
                                output_shape=(original_out_dim, ),
                                partition_sizes=self._partition_sizes,
                                target_partition_shape=tuple(self.weight.shape),
                                original_shape=self._orig_weight_shape)
        if self.bias is not None:
            self._set_param_uc_meta(self.bias,
                                    partition_type='column',
                                    partition_dim=0,
                                    logical_shape=self._orig_bias_shape,
                                    output_shape=self._orig_bias_shape,
                                    partition_sizes=self._partition_sizes,
                                    target_partition_shape=tuple(self.bias.shape),
                                    original_shape=self._orig_bias_shape,
                                    is_bias=True)

    # for bwc
    @classmethod
    def from_weights(cls, weight_shape=None, dtype=torch.half, weight=None, bias=None, gather_output=False):
        if weight is not None:
            in_features = weight.shape[1]
            out_features = weight.shape[0]
            linear = nn.Linear(in_features, out_features, bias=(bias is not None))
            linear.weight.data = weight
            if bias is not None:
                linear.bias.data = bias
        else:
            in_features = weight_shape[1]
            out_features = weight_shape[0]
            linear = nn.Linear(in_features, out_features, bias=(bias is not None))
        return cls(linear, skip_partition=True, gather_output=gather_output)


class SubParamColumnParallel(LinearLayer):
    """Column-parallel layer whose shard concatenates one piece of every sub-parameter.

    ``LinearLayer`` assumes a rank owns one contiguous block of the output dimension, so its
    gather and its checkpoint metadata would reassemble these layers in rank order and shuffle
    the sub-parameters. Layers that cut a fused weight per sub-parameter mix in this class to
    describe that layout instead.
    """

    # Used when a subclass reports no sub-parameter sizes, to say why its layout cannot be
    # described. Subclasses that can hit that case override it with a specific explanation.
    _unsupported_uc_reason = "its tensor parallel layout cannot be described per sub-parameter"

    def _freeze_partition_sizes(self, total_size):
        """Resolve the sub-parameter layout once, while the layer is built.

        Subclasses describe their split by setting ``_subparam_layout_spec`` before the base
        constructor runs. Its sizes are ``None`` for fused layouts that interleave or replicate
        blocks rather than splitting them per sub-parameter; those have no such description.
        """
        super()._freeze_partition_sizes(total_size)
        subparam_sizes, shard_name = self._subparam_layout_spec
        self._subparam_sizes = subparam_sizes
        self._subparam_shard_widths = None
        if subparam_sizes is not None:
            self._subparam_shard_widths = _subparam_shard_widths(subparam_sizes, self.tp_world_size, self.tp_meta,
                                                                 shard_name)

    def _subparam_shape_spec(self, logical_shape):
        shape_spec = list(logical_shape)
        shape_spec[0] = tuple(self._subparam_sizes)
        return tuple(shape_spec)

    @torch.no_grad()
    def _tp_partition(self, params_list):
        """Cut every sub-parameter at the widths frozen when the layer was built.

        The helpers a subclass would otherwise call re-derive the split from the tp_shard
        globals, which a second AutoTP model overwrites. Repartitioning after a gather would
        then cut the weight differently than the frozen widths that the gather and the
        checkpoint metadata describe.
        """
        if self._subparam_sizes is None:
            return self._tp_partition_unsupported_layout(params_list)

        for idx, param in enumerate(params_list):
            if param is None:
                continue
            _partition = _partition_logical_tensor(param.data,
                                                   0,
                                                   self.tp_world_size,
                                                   self.tp_index,
                                                   self._subparam_shard_widths,
                                                   subparam_sizes=self._subparam_sizes)
            params_list[idx].data = self.move(_partition).detach()

    @torch.no_grad()
    def _tp_partition_unsupported_layout(self, params_list):
        """Cut a fused layout that has no per-sub-parameter description."""
        raise RuntimeError(self._unsupported_uc_reason)

    @torch.no_grad()
    def gather_params(self, params_list):
        if self._subparam_sizes is None:
            # The inherited gather concatenates the shards in rank order, which is not how this
            # layout was split, so it would hand back a silently wrong weight to consolidate.
            raise RuntimeError(self._unsupported_uc_reason)

        for idx, param in enumerate(params_list):
            if param is None:
                continue
            logical_shape = self._orig_bias_shape if self._is_bias_param(param) else self._orig_weight_shape
            full_view = _gather_logical_tensor(param,
                                               logical_shape,
                                               0,
                                               self.mp_group,
                                               self.tp_world_size,
                                               self._subparam_shard_widths,
                                               subparam_sizes=self._subparam_sizes)
            params_list[idx].data = full_view.reshape(logical_shape).contiguous()

    def _mark_uc_metadata(self):
        if self._subparam_sizes is None:
            # Publishing a plain column layout here would make the converter reassemble the
            # parameter in rank order, which is not how it was split. Record why instead, so
            # conversion reports it rather than writing a silently wrong checkpoint.
            self._set_param_uc_meta(self.weight,
                                    partition_type='column',
                                    partition_dim=0,
                                    logical_shape=self._orig_weight_shape,
                                    original_shape=self._orig_weight_shape,
                                    unsupported_reason=self._unsupported_uc_reason)
            if self.bias is not None:
                self._set_param_uc_meta(self.bias,
                                        partition_type='column',
                                        partition_dim=0,
                                        logical_shape=self._orig_bias_shape,
                                        original_shape=self._orig_bias_shape,
                                        is_bias=True,
                                        unsupported_reason=self._unsupported_uc_reason)
            return

        self._set_param_uc_meta(self.weight,
                                partition_type='column',
                                partition_dim=0,
                                logical_shape=self._orig_weight_shape,
                                output_shape=(self._orig_weight_shape[0], ),
                                sub_param_shape=self._subparam_shape_spec(self._orig_weight_shape),
                                sub_param_sizes=self._subparam_sizes,
                                sub_param_shard_widths=self._subparam_shard_widths,
                                target_partition_shape=tuple(self.weight.shape),
                                original_shape=self._orig_weight_shape)
        if self.bias is not None:
            self._set_param_uc_meta(self.bias,
                                    partition_type='column',
                                    partition_dim=0,
                                    logical_shape=self._orig_bias_shape,
                                    output_shape=self._orig_bias_shape,
                                    sub_param_shape=self._subparam_shape_spec(self._orig_bias_shape),
                                    sub_param_sizes=self._subparam_sizes,
                                    sub_param_shard_widths=self._subparam_shard_widths,
                                    target_partition_shape=tuple(self.bias.shape),
                                    original_shape=self._orig_bias_shape,
                                    is_bias=True)


class FusedModuleWrapper:

    def __init__(self, fused_module: nn.Module):
        self.fused_module = fused_module

    def __getattr__(self, module):
        return self.fused_module


class fused_LinearLayer(SubParamColumnParallel):

    _unsupported_uc_reason = ("its fused qkv layout interleaves or replicates blocks across tensor parallel "
                              "ranks, so the original parameter cannot be reassembled from the shards")

    def __init__(self, module, mp_group, skip_partition=False, **kwargs):
        assert kwargs.get('fused_module') is not None, "'fused_module' is required but not provided"
        # Use the warp class to avoid module circular references.
        self.fused_module = FusedModuleWrapper(kwargs.get('fused_module'))
        # prepare_tp_fused_qkvw takes its own shard sizes without a layer name, so the widths
        # describing its split must be resolved the same way.
        self._subparam_layout_spec = (fused_qkv_subparam_sizes(kwargs.get('fused_module'), tuple(module.weight.shape),
                                                               kwargs['tp_meta']), None)
        super().__init__(module, mp_group, skip_partition, **kwargs)

    def _freeze_partition_sizes(self, total_size):
        super()._freeze_partition_sizes(total_size)
        if self._subparam_shard_widths is not None:
            set_fused_qkv_shard_state(self.fused_module.module, self._subparam_shard_widths, self.tp_index)

    @torch.no_grad()
    def _tp_partition_unsupported_layout(self, params_list):
        # These layouts interleave or replicate blocks, so only the original helper knows how
        # to cut them. They cannot be gathered, so they are never repartitioned after a gather.
        for idx, param in enumerate(params_list):
            if param is None:
                return

            _partition = prepare_tp_fused_qkvw(self.fused_module.module, param, self.tp_world_size, self.tp_index,
                                               self.tp_meta)

            _partition = self.move(_partition).detach()

            params_list[idx].data = _partition


class conv_LinearLayer(LinearLayer):

    @torch.no_grad()
    def _tp_partition(self, params_list):
        weight = None
        bias = None
        if len(params_list) == 1:
            weight = params_list[0]
        elif len(params_list) == 2:
            weight, bias = params_list[0], params_list[1]
        _partition = weight.data.split(get_shard_size_list(weight.shape[0], self.tp_world_size, self.tp_meta,
                                                           self.name),
                                       dim=1)[self.tp_index]
        _partition = self.move(_partition).detach()
        weight.data = _partition

        if bias is not None:
            _partition = bias.data.split(get_shard_size_list(weight.shape[1], self.tp_world_size, self.tp_meta,
                                                             self.name),
                                         dim=0)[self.tp_index]
            _partition = self.move(_partition).detach()

            bias.data = _partition


#override the subclasses related to weight splitting.
class Yuan_LinearAllreduce(LinearAllreduce):

    _unsupported_uc_reason = ("Yuan shared-QK tensor parallelism selects noncontiguous head groups that universal "
                              "checkpoint conversion cannot currently describe")

    #Yuan2
    @torch.no_grad()
    def _tp_partition(self, params_list):
        weight, bias = shard_value_with_share_qk(params_list[0].data, params_list[1], self.tp_index,
                                                 self.tp_world_size, False, self.tp_meta)
        params_list[0].data = weight
        if bias is not None:
            params_list[1].data = bias

    @torch.no_grad()
    def gather_params(self, params_list):
        raise RuntimeError(self._unsupported_uc_reason)

    def _mark_uc_metadata(self):
        self._set_param_uc_meta(self.weight,
                                partition_type='row',
                                partition_dim=1,
                                logical_shape=self._orig_weight_shape,
                                original_shape=self._orig_weight_shape,
                                unsupported_reason=self._unsupported_uc_reason)
        if self.bias is not None:
            bias_shape = tuple(self.bias.shape)
            self._set_param_uc_meta(self.bias,
                                    partition_type='row',
                                    logical_shape=bias_shape,
                                    original_shape=bias_shape,
                                    is_bias=True,
                                    unsupported_reason=self._unsupported_uc_reason)


class Yuan_LinearLayer(LinearLayer):
    _unsupported_uc_reason = ("Yuan shared-QK tensor parallelism selects noncontiguous head groups that universal "
                              "checkpoint conversion cannot currently describe")

    #Yuan2
    @torch.no_grad()
    def _tp_partition(self, params_list):
        weight, bias = shard_value_with_share_qk(params_list[0].data, params_list[1], self.tp_index,
                                                 self.tp_world_size, True, self.tp_meta)
        params_list[0].data = self.move(weight).detach()
        if bias is not None:
            params_list[1].data = self.move(bias).detach()

    @torch.no_grad()
    def gather_params(self, params_list):
        raise RuntimeError(self._unsupported_uc_reason)

    def _mark_uc_metadata(self):
        self._set_param_uc_meta(self.weight,
                                partition_type='column',
                                partition_dim=0,
                                logical_shape=self._orig_weight_shape,
                                original_shape=self._orig_weight_shape,
                                unsupported_reason=self._unsupported_uc_reason)
        if self.bias is not None:
            self._set_param_uc_meta(self.bias,
                                    partition_type='column',
                                    partition_dim=0,
                                    logical_shape=self._orig_bias_shape,
                                    original_shape=self._orig_bias_shape,
                                    is_bias=True,
                                    unsupported_reason=self._unsupported_uc_reason)


class GateUpPack_LinearLayer(SubParamColumnParallel):
    # chatGLM2, chatGLM2

    def __init__(self, module, mp_group=None, **kwargs):
        # shard_chunk_mlp splits the gate and the up halves separately, under the "mlp" name.
        half = tuple(module.weight.shape)[0] // 2
        self._subparam_layout_spec = ((half, half), "mlp")
        super().__init__(module, mp_group, **kwargs)


class Conv_LinearALlreduce(LinearAllreduce):

    @torch.no_grad()
    def _tp_partition(self, params_list):
        for idx, param in enumerate(params_list):
            if param is None:
                return
            param.data = param.data.transpose(-1, -2).contiguous()

            _partition = param.split(get_shard_size_list(param.shape[0], self.tp_world_size, self.tp_meta, self.name),
                                     dim=1)[self.tp_index]

            _partition = self.move(_partition).detach()

            params_list[idx].data = _partition


#override the subclasses related to fwd/bwd.
class LmHeadLinearAllreduce(LinearAllreduce):

    def __init__(self, module, mp_group, **kwargs):
        # set the fixed name before partition
        self.name = "lm_head"

        # In some tied_embedding cases, only the lm head is sharded, while the word embedding is not.
        # Reinitialization is used to decouple them and prevent the word embedding from being sharded.
        # This should also be effective for cases where both are sharded in tied_embedding scenarios.

        # TODO: Training scenario-related tests, is it necessary to re-implement the vocab parallel module?
        module.weight = nn.Parameter(module.weight.clone().detach())
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias = nn.Parameter(module.bias.clone().detach())
        super().__init__(module, mp_group, **kwargs)

    def forward(self, input):
        # The weight columns were cut with the sizes frozen at construction, so the input has to
        # be cut the same way. Recomputing the split here would read the tp_shard globals that a
        # later model overwrites, and the row-parallel all-reduce would hide the misalignment.
        input_shard_sizes = self._partition_sizes
        assert sum(input_shard_sizes) == input.shape[-1], (
            f"lm_head was partitioned for an input of {sum(input_shard_sizes)} features, but got "
            f"{input.shape[-1]}.")
        input_shard_size = input_shard_sizes[self.tp_index]
        input_shard_offset = sum(input_shard_sizes[0:self.tp_index])
        output = torch.matmul(input[:, :, input_shard_offset:input_shard_offset + input_shard_size],
                              self.weight.transpose(-1, -2))
        if self.mp_group is not None:
            dist.inference_all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output = add_bias(output, self.bias)
        return output


class TensorParallelConv2d(nn.Module):

    def __init__(self, conv, rank, world_size, shard_by_oc):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.shard_by_oc = shard_by_oc
        self.shard_weights(conv)

    # Split along the input/output channel depending on whether it is the last conv layer.
    def shard_weights(self, conv):
        if self.shard_by_oc:
            total_size = conv.weight.shape[0]
        else:
            total_size = conv.weight.shape[1]
        bias_data = None
        cols_per_rank = [0]
        for i in range(self.world_size - 1, -1, -1):
            cols = total_size // self.world_size
            if i < total_size % self.world_size:
                cols += 1
            cols_per_rank.append(cols_per_rank[-1] + cols)
        weight_data = conv.weight.data
        if self.shard_by_oc:
            # not last conv layer, split output channel
            weight_data = weight_data[cols_per_rank[self.rank]:cols_per_rank[self.rank + 1]]
            if conv.bias is not None:
                bias_data = conv.bias.data[cols_per_rank[self.rank]:cols_per_rank[self.rank + 1]]
        else:
            # last conv layer, split input channel
            weight_data = weight_data[:, cols_per_rank[self.rank]:cols_per_rank[self.rank + 1]]
            if conv.bias is not None:
                bias_data = conv.bias.data / float(self.world_size)
        self.conv = nn.Conv2d(weight_data.shape[1], weight_data.shape[0], conv.kernel_size, conv.stride, conv.padding,
                              conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode)
        self.conv.weight = torch.nn.Parameter(weight_data)
        if conv.bias is not None:
            self.conv.bias = torch.nn.Parameter(bias_data)
        del conv

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class TensorParallelOcShardConv2d(TensorParallelConv2d):

    def __init__(self, conv, rank, world_size):
        super().__init__(conv, rank, world_size, True)


class TensorParallelIcShardConv2d(TensorParallelConv2d):

    def __init__(self, conv, rank, world_size):
        super().__init__(conv, rank, world_size, False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.conv(input)
        if self.world_size > 1:
            dist.inference_all_reduce(out)
        return out


class Normalize(nn.Module):

    def __init__(self, dim=None, dtype=torch.float, eps=1e-5, weight=None, bias=None):
        super(Normalize, self).__init__()
        if weight is not None:
            self.weight = weight
            self.bias = bias
        else:
            self.norm = nn.LayerNorm(dim, eps=eps).to(dtype).to(get_accelerator().current_device_name())
            self.weight = self.norm.weight
            self.bias = self.norm.bias

        self.eps = eps

    def forward(self, input):
        return nn.functional.layer_norm(input, input.shape[-1:], self.weight, self.bias, eps=self.eps)


class EmbeddingLayer(nn.Module):

    def __init__(self, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        super(EmbeddingLayer, self).__init__()
        if weight is None:
            self.weight = Parameter(
                torch.empty(weight_shape[0],
                            weight_shape[1],
                            dtype=dtype,
                            device=get_accelerator().current_device_name()))
        else:
            self.weight = weight

    def forward(self, input):
        return F.embedding(input, self.weight)


class OPTEmbedding(EmbeddingLayer):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, weight_shape=None, weight=None, bias=None):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(weight_shape, weight=weight)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0, position_ids: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


def _shape_prod(values):
    result = 1
    for val in values:
        result *= val
    return result


def _normalize_shape_spec(shape):
    if isinstance(shape, list):
        return tuple(_normalize_shape_spec(item) for item in shape)
    if isinstance(shape, tuple):
        return tuple(_normalize_shape_spec(item) if isinstance(item, list) else item for item in shape)
    return shape


def _infer_subparam_logical_shapes(weight_shape, shape, partition_dim, name=None):
    shape = _normalize_shape_spec(shape)
    if not isinstance(shape, tuple):
        raise ValueError("AutoTP shape must be a tuple for sub-parameter partitioning.")
    if partition_dim < 0 or partition_dim >= len(shape):
        raise ValueError(f"AutoTP partition_dim {partition_dim} is out of range for shape length {len(shape)}.")

    layer_label = f"AutoTP layer '{name}'" if name else "AutoTP layer"
    partition_elem = shape[partition_dim]
    subparam_sizes = None
    num_subparams = None

    if isinstance(partition_elem, tuple):
        if len(partition_elem) == 0:
            raise ValueError(f"{layer_label} sub-parameter size tuple cannot be empty.")
        if any(isinstance(val, tuple) for val in partition_elem):
            raise ValueError(f"{layer_label} supports only 1-level nesting at partition_dim.")
        if any((not isinstance(val, int)) or val <= 0 for val in partition_elem):
            raise ValueError(f"{layer_label} sub-parameter sizes must be positive integers.")
        subparam_sizes = tuple(int(val) for val in partition_elem)
        partition_dim_size = sum(subparam_sizes)
    elif isinstance(partition_elem, int):
        if partition_elem == -1:
            partition_dim_size = None
        elif partition_elem > 0:
            num_subparams = partition_elem
            partition_dim_size = None
        else:
            raise ValueError(f"{layer_label} partition_dim spec must be positive integer or -1.")
    else:
        raise ValueError(f"{layer_label} partition_dim spec must be int or tuple.")

    logical_dims = []
    for idx, dim in enumerate(shape):
        if idx == partition_dim:
            logical_dims.append(partition_dim_size)
            continue
        if isinstance(dim, tuple):
            raise ValueError(f"{layer_label} nested tuple only allowed at partition_dim={partition_dim}.")
        if isinstance(dim, int):
            if dim == -1:
                logical_dims.append(None)
            elif dim > 0:
                logical_dims.append(dim)
            else:
                raise ValueError(f"{layer_label} shape dimensions must be positive integers or -1.")
        else:
            raise ValueError(f"{layer_label} shape dimensions must be integers.")

    total_numel = _shape_prod(weight_shape)
    known_product = _shape_prod([dim for dim in logical_dims if dim is not None])
    unknown_indices = [idx for idx, dim in enumerate(logical_dims) if dim is None]

    if len(unknown_indices) == 0:
        if known_product != total_numel:
            raise ValueError(f"{layer_label} shape product {known_product} != weight numel {total_numel}.")
    elif len(unknown_indices) == 1:
        inferred = total_numel // known_product
        if inferred * known_product != total_numel:
            raise ValueError(f"{layer_label} cannot infer shape for weight with numel {total_numel}.")
        logical_dims[unknown_indices[0]] = inferred
    else:
        if len(shape) == len(weight_shape):
            for idx in unknown_indices:
                logical_dims[idx] = weight_shape[idx]
            if _shape_prod(logical_dims) != total_numel:
                raise ValueError(
                    f"{layer_label} shape product {_shape_prod(logical_dims)} != weight numel {total_numel}.")
        else:
            raise ValueError(f"{layer_label} shape has multiple inferred dims and is ambiguous for weight.")

    logical_shape = tuple(logical_dims)
    if logical_shape[-1] != weight_shape[-1]:
        raise ValueError(
            f"{layer_label} shape last dim {logical_shape[-1]} must match weight input dim {weight_shape[-1]}.")

    output_shape = logical_shape[:-1]
    if len(output_shape) == 0:
        raise ValueError(f"{layer_label} shape must include at least one output dimension.")
    if _shape_prod(output_shape) != weight_shape[0]:
        raise ValueError(
            f"{layer_label} output shape product {_shape_prod(output_shape)} != weight output dim {weight_shape[0]}.")

    partition_dim_size = logical_shape[partition_dim]
    if partition_dim_size is None or partition_dim_size <= 0:
        raise ValueError(f"{layer_label} partition_dim size must be a positive integer.")

    if num_subparams is not None:
        if partition_dim_size % num_subparams != 0:
            raise ValueError(
                f"{layer_label} partition_dim size {partition_dim_size} not divisible by sub-param count {num_subparams}."
            )
        subparam_sizes = tuple([partition_dim_size // num_subparams] * num_subparams)

    if subparam_sizes is not None and sum(subparam_sizes) != partition_dim_size:
        raise ValueError(
            f"{layer_label} sub-parameter sizes sum {sum(subparam_sizes)} != partition_dim size {partition_dim_size}.")

    bias_partition_dim = partition_dim if partition_dim < len(output_shape) else None
    return logical_shape, output_shape, subparam_sizes, bias_partition_dim


def _bias_subparam_shape_spec(output_shape, bias_partition_dim, subparam_sizes):
    """View spec describing how a partitioned bias is laid out over its sub-parameters.

    The weight's spec cannot be reused: it carries the input dimension the bias does not have,
    and its partition_dim entry may be a sub-parameter *count* rather than a width. This spec is
    fully resolved, so a reader never has to infer a dimension.
    """
    if bias_partition_dim is None:
        return None
    shape_spec = list(output_shape)
    if subparam_sizes is not None:
        shape_spec[bias_partition_dim] = tuple(subparam_sizes)
    return tuple(shape_spec)


def _subparam_shard_widths(subparam_sizes, tp_world_size, meta: AutoTPMeta, name=None):
    """Per-rank width of each sub-parameter, as one list per sub-parameter.

    Sub-parameters follow the same deterministic split as ordinary layers, so a fused
    attention weight lands on key/value head boundaries instead of being cut inside a head.
    """
    widths = []
    for size in subparam_sizes:
        per_rank = get_shard_size_list(size, tp_world_size, meta, name)
        if min(per_rank) == 0:
            # Those ranks contribute zeros to the row-parallel all-reduce, so the result stays
            # correct and this matches how separate q/k/v projections already behave. Serving
            # them would mean replicating heads, which AutoTP does not do. The layer name is
            # left out so the whole model reports this once rather than once per layer.
            log_dist_once(
                f"AutoTP: a sub-parameter of width {size} splits across tp_size {tp_world_size} as "
                f"{per_rank}, so some ranks hold none of it and stay idle for this weight. This "
                f"happens when there are fewer attention heads than tensor parallel ranks.",
                ranks=[0],
                level=logging.WARNING)
        widths.append(per_rank)
    return widths


def _check_shard_widths(shard_widths, subparam_sizes, tp_world_size):
    """Validate a layer's per-rank widths against the sub-parameters they describe.

    Partition and gather have to agree on where every sub-parameter was cut, so both take the
    widths the layer recorded rather than deriving their own, and this checks that what they
    were handed actually tiles the sub-parameters.
    """
    assert len(shard_widths) == len(subparam_sizes), (
        f"Got {len(shard_widths)} shard width entries for {len(subparam_sizes)} sub-parameters.")
    for size, per_rank in zip(subparam_sizes, shard_widths):
        assert len(per_rank) == tp_world_size, (
            f"Sub-parameter of size {size} has {len(per_rank)} shard widths, expected one per tp rank "
            f"({tp_world_size}).")
        assert sum(per_rank) == size, (
            f"Sub-parameter shard widths {list(per_rank)} sum to {sum(per_rank)}, expected {size}.")
    return shard_widths


def _partition_logical_tensor(tensor, partition_dim, tp_world_size, tp_index, shard_widths, subparam_sizes=None):
    if tp_world_size == 1:
        return tensor
    sizes = tuple(subparam_sizes) if subparam_sizes else (tensor.shape[partition_dim], )
    widths = _check_shard_widths(shard_widths, sizes, tp_world_size)
    sub_params = torch.split(tensor, sizes, dim=partition_dim)
    partitioned = [sp.split(w, dim=partition_dim)[tp_index] for sp, w in zip(sub_params, widths)]
    if len(partitioned) == 1:
        return partitioned[0]
    return torch.cat(partitioned, dim=partition_dim)


def _all_gather_along_dim(tensor, partition_dim, mp_group, tp_world_size):
    if mp_group is None or tp_world_size == 1:
        return tensor
    perm = [partition_dim] + [idx for idx in range(tensor.dim()) if idx != partition_dim]
    inv_perm = [0] * len(perm)
    for idx, dim in enumerate(perm):
        inv_perm[dim] = idx
    tensor_perm = tensor.permute(perm).contiguous()
    output = torch.empty((tp_world_size * tensor_perm.shape[0], *tensor_perm.shape[1:]),
                         dtype=tensor.dtype,
                         device=tensor.device)
    dist.all_gather_into_tensor(output, tensor_perm, group=mp_group)
    return output.permute(inv_perm).contiguous()


def _all_gather_uneven_along_dim(tensor, partition_dim, mp_group, shard_widths):
    """Gather shards of differing widths along ``partition_dim``.

    Shards are zero padded to a common width so the uniform (and faster)
    ``all_gather_into_tensor`` collective stays usable, and are then trimmed back.
    """
    tp_world_size = len(shard_widths)
    if mp_group is None or tp_world_size == 1:
        return tensor
    if min(shard_widths) == max(shard_widths):
        return _all_gather_along_dim(tensor, partition_dim, mp_group, tp_world_size)

    max_width = max(shard_widths)
    padded_shape = list(tensor.shape)
    padded_shape[partition_dim] = max_width
    padded = tensor.new_zeros(padded_shape)
    padded.narrow(partition_dim, 0, tensor.shape[partition_dim]).copy_(tensor)

    gathered = _all_gather_along_dim(padded, partition_dim, mp_group, tp_world_size)
    trimmed = [gathered.narrow(partition_dim, i * max_width, w) for i, w in enumerate(shard_widths)]
    return torch.cat(trimmed, dim=partition_dim)


def _gather_logical_tensor(tensor,
                           logical_shape,
                           partition_dim,
                           mp_group,
                           tp_world_size,
                           shard_widths,
                           subparam_sizes=None):
    if mp_group is None or tp_world_size == 1:
        return tensor.reshape(logical_shape)
    sizes = tuple(subparam_sizes) if subparam_sizes else (logical_shape[partition_dim], )
    widths = _check_shard_widths(shard_widths, sizes, tp_world_size)
    # The local shard holds one piece of every sub-parameter, so its own widths decide how to
    # unpack it before each piece is gathered back to its full size.
    tp_index = dist.get_rank(group=mp_group)
    local_sizes = [per_rank[tp_index] for per_rank in widths]

    partitioned_shape = list(logical_shape)
    partitioned_shape[partition_dim] = sum(local_sizes)
    tensor_view = tensor.reshape(partitioned_shape)

    sub_params = torch.split(tensor_view, local_sizes, dim=partition_dim)
    gathered = [
        _all_gather_uneven_along_dim(sp, partition_dim, mp_group, per_rank)
        for sp, per_rank in zip(sub_params, widths)
    ]
    if len(gathered) == 1:
        return gathered[0]
    return torch.cat(gathered, dim=partition_dim)


class SubParamLinearLayer(TensorParallel_Layer):
    """
    Column-parallel linear layer with sub-parameter support.

    Handles cases where weights contain multiple logical sub-parameters
    that need to be partitioned separately (e.g., fused QKV, chunked MLP, GQA).

    The `shape` parameter controls how the weight is viewed and partitioned:
    - (3, -1) with partition_dim=0: 3 equal sub-params, partition each at dim 0
    - ((q, k, v), -1) with partition_dim=0: 3 unequal sub-params (1-level nesting)
    """

    def __init__(self, module, mp_group, shape, partition_dim=0, **kwargs):
        super(SubParamLinearLayer, self).__init__(mp_group, **kwargs)
        self.weight = module.weight
        self.bias = module.bias
        self.shape = shape
        self.partition_dim = partition_dim

        self._orig_weight_shape = tuple(module.weight.shape)
        self._orig_bias_shape = tuple(module.bias.shape) if self.bias is not None else None
        (self._logical_shape, self._output_shape, self._subparam_sizes,
         self._bias_partition_dim) = _infer_subparam_logical_shapes(self._orig_weight_shape, self.shape,
                                                                    self.partition_dim, self.name)
        # Resolve the per-rank widths once, for the same reason _freeze_partition_sizes does:
        # the split depends on this model's tp_meta.
        self._subparam_shard_widths = _subparam_shard_widths(
            self._subparam_sizes or (self._logical_shape[self.partition_dim], ), self.tp_world_size, self.tp_meta,
            self.name)
        self._bias_shape_spec = _bias_subparam_shape_spec(self._output_shape, self._bias_partition_dim,
                                                          self._subparam_sizes)
        if self.bias is not None and self.bias.numel() != _shape_prod(self._output_shape):
            raise ValueError(f"AutoTP layer '{self.name}' bias size {self.bias.numel()} does not match output shape "
                             f"{self._output_shape}.")

        if self._should_materialize_tp_partition():
            self._tp_partition([self.weight, self.bias])
        self.support_training = True
        self.config_tp_params(self.weight)
        if self.bias is not None:
            self.config_tp_params(self.bias)
        self._mark_uc_metadata()

    def forward(self, input):
        if getattr(self, 'mp_group', None) is not None and not self.defer_collectives_to_compiler:
            input = ColumnParallel.apply(self.mp_group, input)
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.bias is not None:
            output = add_bias(output, self.bias)
        return output

    @torch.no_grad()
    def gather_params(self, params_list):
        """Gather partitioned parameters back to full size."""
        for idx, param in enumerate(params_list):
            if param is None:
                continue
            if self._is_bias_param(param):
                if self._bias_partition_dim is None:
                    params_list[idx].data = param.data
                else:
                    full_bias_view = _gather_logical_tensor(param,
                                                            self._output_shape,
                                                            self._bias_partition_dim,
                                                            self.mp_group,
                                                            self.tp_world_size,
                                                            self._subparam_shard_widths,
                                                            subparam_sizes=self._subparam_sizes)
                    params_list[idx].data = full_bias_view.reshape(self._orig_bias_shape)
                continue

            full_view = _gather_logical_tensor(param,
                                               self._logical_shape,
                                               self.partition_dim,
                                               self.mp_group,
                                               self.tp_world_size,
                                               self._subparam_shard_widths,
                                               subparam_sizes=self._subparam_sizes)
            params_list[idx].data = full_view.reshape(self._orig_weight_shape)

    @torch.no_grad()
    def _tp_partition(self, params_list):
        for idx, param in enumerate(params_list):
            if param is None:
                continue
            if self._is_bias_param(param):
                if self._bias_partition_dim is None:
                    params_list[idx].data = self.move(param).detach()
                else:
                    bias_view = param.reshape(self._output_shape)
                    bias_partitioned = _partition_logical_tensor(bias_view,
                                                                 self._bias_partition_dim,
                                                                 self.tp_world_size,
                                                                 self.tp_index,
                                                                 self._subparam_shard_widths,
                                                                 subparam_sizes=self._subparam_sizes)
                    params_list[idx].data = self.move(bias_partitioned.reshape(-1)).detach()
                continue

            weight_view = param.reshape(self._logical_shape)
            partitioned_view = _partition_logical_tensor(weight_view,
                                                         self.partition_dim,
                                                         self.tp_world_size,
                                                         self.tp_index,
                                                         self._subparam_shard_widths,
                                                         subparam_sizes=self._subparam_sizes)
            leading_size = _shape_prod(partitioned_view.shape[:-1])
            params_list[idx].data = self.move(partitioned_view.reshape(leading_size,
                                                                       partitioned_view.shape[-1])).detach()

    def _mark_uc_metadata(self):
        self._set_param_uc_meta(self.weight,
                                partition_type='column',
                                partition_dim=self.partition_dim,
                                logical_shape=self._logical_shape,
                                output_shape=self._output_shape,
                                sub_param_shape=self.shape,
                                sub_param_sizes=self._subparam_sizes,
                                sub_param_shard_widths=self._subparam_shard_widths,
                                target_partition_shape=self.weight.shape,
                                original_shape=self._orig_weight_shape)
        if self.bias is not None:
            self._set_param_uc_meta(
                self.bias,
                partition_type='column',
                partition_dim=self._bias_partition_dim,
                logical_shape=self._output_shape,
                output_shape=self._output_shape,
                sub_param_shape=self._bias_shape_spec,
                sub_param_sizes=self._subparam_sizes if self._bias_partition_dim is not None else None,
                sub_param_shard_widths=self._subparam_shard_widths if self._bias_partition_dim is not None else None,
                target_partition_shape=self.bias.shape,
                original_shape=self._orig_bias_shape,
                is_bias=True,
                replicated=self._bias_partition_dim is None)


class SubParamLinearAllreduce(TensorParallel_Layer):
    """
    Row-parallel linear layer with sub-parameter support (AllReduce after forward).

    Handles cases where weights contain multiple logical sub-parameters
    that need to be partitioned separately.
    """

    def __init__(self, module, mp_group, shape, partition_dim=1, **kwargs):
        super(SubParamLinearAllreduce, self).__init__(mp_group, **kwargs)
        self.weight = module.weight
        self.bias = module.bias
        self.shape = shape
        self.partition_dim = partition_dim

        self._orig_weight_shape = tuple(module.weight.shape)
        self._orig_bias_shape = tuple(module.bias.shape) if self.bias is not None else None
        (self._logical_shape, self._output_shape, self._subparam_sizes,
         self._bias_partition_dim) = _infer_subparam_logical_shapes(self._orig_weight_shape, self.shape,
                                                                    self.partition_dim, self.name)
        # Resolve the per-rank widths once, for the same reason _freeze_partition_sizes does:
        # the split depends on this model's tp_meta.
        self._subparam_shard_widths = _subparam_shard_widths(
            self._subparam_sizes or (self._logical_shape[self.partition_dim], ), self.tp_world_size, self.tp_meta,
            self.name)

        if self._should_materialize_tp_partition():
            self._tp_partition([self.weight, self.bias])
        self.support_training = True
        self.config_tp_params(self.weight)
        if self.bias is not None:
            self.config_requires_grad(self.bias)
        self._mark_uc_metadata()

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if not self.defer_collectives_to_compiler:
            output = RowParallel.apply(self.mp_group, output, not self.is_training_mode())
        if self.bias is not None:
            output = add_bias(output, self.bias)
        return output

    @torch.no_grad()
    def gather_params(self, params_list):
        """Gather partitioned parameters back to full size."""
        for idx, param in enumerate(params_list):
            if param is None:
                continue
            if self._is_bias_param(param):
                # don't gather bias for row parallel
                continue
            full_view = _gather_logical_tensor(param,
                                               self._logical_shape,
                                               self.partition_dim,
                                               self.mp_group,
                                               self.tp_world_size,
                                               self._subparam_shard_widths,
                                               subparam_sizes=self._subparam_sizes)
            params_list[idx].data = full_view.reshape(self._orig_weight_shape)

    @torch.no_grad()
    def _tp_partition(self, params_list):
        for idx, param in enumerate(params_list):
            if param is None:
                continue
            if self._is_bias_param(param):
                # Bias is not partitioned for row parallel (it's applied after all-reduce)
                params_list[idx].data = self.move(param).detach()
                continue

            weight_view = param.reshape(self._logical_shape)
            partitioned_view = _partition_logical_tensor(weight_view,
                                                         self.partition_dim,
                                                         self.tp_world_size,
                                                         self.tp_index,
                                                         self._subparam_shard_widths,
                                                         subparam_sizes=self._subparam_sizes)
            leading_size = _shape_prod(partitioned_view.shape[:-1])
            params_list[idx].data = self.move(partitioned_view.reshape(leading_size,
                                                                       partitioned_view.shape[-1])).detach()

    def _mark_uc_metadata(self):
        self._set_param_uc_meta(self.weight,
                                partition_type='row',
                                partition_dim=self.partition_dim,
                                logical_shape=self._logical_shape,
                                output_shape=self._output_shape,
                                sub_param_shape=self.shape,
                                sub_param_sizes=self._subparam_sizes,
                                sub_param_shard_widths=self._subparam_shard_widths,
                                target_partition_shape=self.weight.shape,
                                original_shape=self._orig_weight_shape)
        if self.bias is not None:
            self._set_param_uc_meta(self.bias,
                                    partition_type='row',
                                    partition_dim=None,
                                    logical_shape=self._orig_bias_shape,
                                    output_shape=self._orig_bias_shape,
                                    original_shape=self._orig_bias_shape,
                                    target_partition_shape=self.bias.shape,
                                    is_bias=True,
                                    replicated=True)


class RMSNormalize(nn.Module):

    def __init__(self, dim=None, dtype=torch.float, eps=1e-5, weight=None):
        super(RMSNormalize, self).__init__()
        if weight is not None:
            self.weight = weight
        else:
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=get_accelerator().current_device_name()))

        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return hidden_states * self.weight
