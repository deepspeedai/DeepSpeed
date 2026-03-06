# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed import comm as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.tp_shard import get_shard_size, get_shard_size_list
from deepspeed.runtime.zero.utils import is_zero_param
from abc import ABC, abstractmethod
from typing import Iterable, Any, Optional, List, Tuple
from .fusedqkv_utils import shard_value_with_share_qk, shard_chunk_mlp, prepare_tp_fused_qkvw
from deepspeed.runtime.tensor_parallel import AUTOTP_MODE
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

    def is_training_mode(self):
        global DEEPSPEED_AUTOTP_MODE
        return DEEPSPEED_AUTOTP_MODE == AUTOTP_MODE.TRAINING

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

        self._tp_partition([self.weight, self.bias])
        self.support_training = True
        self.config_tp_params(self.weight)
        if self.bias is not None:
            # bias here is not tp params
            self.config_requires_grad(self.bias)

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        output = RowParallel.apply(self.mp_group, output, not self.is_training_mode())
        if self.bias is not None:
            output = add_bias(output, self.bias)
        return output

    @torch.no_grad()
    def gather_params(self, params_list):

        for idx, param in enumerate(params_list):
            if param is None or idx > 0:
                # don't gather bias
                return
            params_list[idx].data_partition = param.data
            param = param.transpose(0, 1).contiguous()

            output_param = torch.empty(self.tp_world_size * param.shape[0],
                                       param.shape[1],
                                       dtype=param.dtype,
                                       device=param.device)
            dist.all_gather_into_tensor(output_param, param, group=self.mp_group)
            params_list[idx].data = output_param.transpose(0, 1).contiguous()
        return

    @torch.no_grad()
    def _tp_partition(self, params_list):

        if not self.is_training_mode():
            self.uneven_partition(params_list)
            return

        else:
            for idx, param in enumerate(params_list):
                if param is None:
                    # don't slipt bias
                    return
                if idx > 0:  # move bias to device at initialization
                    _partition = self.move(param).detach()
                    params_list[idx].data = _partition
                    return

                _partition = torch.chunk(param, self.tp_world_size, dim=-1)[self.tp_index]

                _partition = self.move(_partition).detach()

                params_list[idx].data = _partition

    def uneven_partition(self, params_list):
        for idx, param in enumerate(params_list):
            if param is None or idx > 0:
                # don't slipt bias
                return
            assert self.name is not None, "The module name must be provided in the initialization."
            _partition = params_list[idx].split(get_shard_size_list(params_list[idx].shape[1], self.tp_world_size,
                                                                    self.name),
                                                dim=1)[self.tp_index]

            _partition = self.move(_partition).detach()
            params_list[idx].data = _partition


#remove kwargs from partition.
class LinearLayer(TensorParallel_Layer):

    def __init__(self, module, mp_group=None, skip_partition=False, **kwargs):
        super(LinearLayer, self).__init__(mp_group, **kwargs)
        self.weight = module.weight
        self.bias = module.bias
        if not skip_partition:
            self._tp_partition([self.weight, self.bias])
        self.support_training = True
        self.config_tp_params(self.weight)
        if self.bias is not None:
            self.config_tp_params(self.bias)

    def forward(self, input):
        if not self.__class__.tp_overlap_comm:
            if getattr(self, 'mp_group', None) is not None:
                input = ColumnParallel.apply(self.mp_group, input)
            output = torch.matmul(input, self.weight.transpose(-1, -2))
            if self.bias is not None:
                output = add_bias(output, self.bias)
        else:
            output = AsyncColumnParallel.apply(self.mp_group, input, self.weight, self.bias)

        return output

    @torch.no_grad()
    def gather_params(self, params_list):
        #  Does not support uneven shard.
        for idx, param in enumerate(params_list):

            params_list[idx].data_partition = param.data
            output_param = torch.empty((self.tp_world_size * param.shape[0], *param.shape[1:]),
                                       dtype=param.dtype,
                                       device=param.device)
            dist.all_gather_into_tensor(output_param, param, group=self.mp_group)
            params_list[idx].data = output_param.contiguous()

    @torch.no_grad()
    def _tp_partition(self, params_list):

        if not self.is_training_mode():
            self.uneven_partition(params_list)
            return
        for idx, param in enumerate(params_list):
            if param is None:
                return
            #split bias if provide
            _partition = torch.chunk(param, self.tp_world_size, dim=0)[self.tp_index]

            _partition = self.move(_partition).detach()

            params_list[idx].data = _partition

    def uneven_partition(self, params_list):

        for idx, param in enumerate(params_list):
            if param is None:
                #split bias if provide
                return
            assert self.name is not None, "The module name must be provided in the initialization."
            _partition = params_list[idx].split(get_shard_size_list(params_list[idx].shape[0], self.tp_world_size,
                                                                    self.name),
                                                dim=0)[self.tp_index]

            _partition = self.move(_partition).detach()

            params_list[idx].data = _partition

    # for bwc
    @classmethod
    def from_weights(cls, weight_shape=None, dtype=torch.half, weight=None, bias=None):
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
        return cls(linear, skip_partition=True)


class FusedModuleWrapper:

    def __init__(self, fused_module: nn.Module):
        self.fused_module = fused_module

    def __getattr__(self, module):
        return self.fused_module


class fused_LinearLayer(LinearLayer):

    def __init__(self, module, mp_group, skip_partition=False, **kwargs):
        assert kwargs.get('fused_module') is not None, "'fused_module' is required but not provided"
        # Use the warp class to avoid module circular references.
        self.fused_module = FusedModuleWrapper(kwargs.get('fused_module'))
        super().__init__(module, mp_group, skip_partition, **kwargs)

    @torch.no_grad()
    def _tp_partition(self, params_list):
        for idx, param in enumerate(params_list):
            if param is None:
                return

            _partition = prepare_tp_fused_qkvw(self.fused_module.module, param, self.tp_world_size, self.tp_index)

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
        _partition = weight.data.split(get_shard_size_list(weight.shape[0], self.tp_world_size, self.name),
                                       dim=1)[self.tp_index]
        _partition = self.move(_partition).detach()
        weight.data = _partition

        if bias is not None:
            _partition = bias.data.split(get_shard_size_list(weight.shape[1], self.tp_world_size, self.name),
                                         dim=0)[self.tp_index]
            _partition = self.move(_partition).detach()

            bias.data = _partition


#override the subclasses related to weight splitting.
class Yuan_LinearAllreduce(LinearAllreduce):

    #Yuan2
    @torch.no_grad()
    def _tp_partition(self, params_list):
        weight, bias = shard_value_with_share_qk(params_list[0].data, params_list[1], self.tp_index,
                                                 self.tp_world_size, False)
        params_list[0].data = weight
        if bias is not None:
            params_list[1].data = bias


class Yuan_LinearLayer(LinearLayer):
    #Yuan2
    @torch.no_grad()
    def _tp_partition(self, params_list):
        weight, bias = shard_value_with_share_qk(params_list[0].data, params_list[1], self.tp_index,
                                                 self.tp_world_size, True)
        params_list[0].data = self.move(weight).detach()
        if bias is not None:
            params_list[1].data = self.move(bias).detach()


class GateUpPack_LinearLayer(LinearLayer):
    # chatGLM2, chatGLM2
    @torch.no_grad()
    def _tp_partition(self, params_list):
        weight, bias = shard_chunk_mlp(params_list[0].data, params_list[1], self.tp_index, self.tp_world_size)
        params_list[0].data = self.move(weight).detach()
        if bias is not None:
            params_list[1].data = self.move(bias).detach()


class Conv_LinearALlreduce(LinearAllreduce):

    @torch.no_grad()
    def _tp_partition(self, params_list):
        for idx, param in enumerate(params_list):
            if param is None:
                return
            param.data = param.data.transpose(-1, -2).contiguous()

            _partition = param.split(get_shard_size_list(param.shape[0], self.tp_world_size, self.name),
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
        input_shard_size = get_shard_size(input.shape[-1], self.tp_world_size, "lm_head")
        input_shard_offset = sum(get_shard_size_list(input.shape[-1], self.tp_world_size, "lm_head")[0:self.tp_index])
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


def _partition_logical_tensor(tensor, partition_dim, tp_world_size, tp_index, name=None, subparam_sizes=None):
    if tp_world_size == 1:
        return tensor
    layer_label = f"AutoTP layer '{name}'" if name else "AutoTP layer"
    if subparam_sizes:
        for size in subparam_sizes:
            if size % tp_world_size != 0:
                raise ValueError(f"{layer_label} sub-parameter size {size} not divisible by tp_size {tp_world_size}.")
        sub_params = torch.split(tensor, subparam_sizes, dim=partition_dim)
        partitioned_sub_params = [torch.chunk(sp, tp_world_size, dim=partition_dim)[tp_index] for sp in sub_params]
        return torch.cat(partitioned_sub_params, dim=partition_dim)
    if tensor.shape[partition_dim] % tp_world_size != 0:
        raise ValueError(
            f"{layer_label} partition_dim size {tensor.shape[partition_dim]} not divisible by tp_size {tp_world_size}."
        )
    return torch.chunk(tensor, tp_world_size, dim=partition_dim)[tp_index]


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


def _gather_logical_tensor(tensor,
                           logical_shape,
                           partition_dim,
                           mp_group,
                           tp_world_size,
                           name=None,
                           subparam_sizes=None):
    if mp_group is None or tp_world_size == 1:
        return tensor.reshape(logical_shape)
    layer_label = f"AutoTP layer '{name}'" if name else "AutoTP layer"
    if logical_shape[partition_dim] % tp_world_size != 0:
        raise ValueError(
            f"{layer_label} partition_dim size {logical_shape[partition_dim]} not divisible by tp_size {tp_world_size}."
        )
    partitioned_shape = list(logical_shape)
    partitioned_shape[partition_dim] = logical_shape[partition_dim] // tp_world_size
    tensor_view = tensor.reshape(partitioned_shape)

    if subparam_sizes:
        for size in subparam_sizes:
            if size % tp_world_size != 0:
                raise ValueError(f"{layer_label} sub-parameter size {size} not divisible by tp_size {tp_world_size}.")
        partitioned_sizes = [size // tp_world_size for size in subparam_sizes]
        sub_params = torch.split(tensor_view, partitioned_sizes, dim=partition_dim)
        gathered_sub_params = [_all_gather_along_dim(sp, partition_dim, mp_group, tp_world_size) for sp in sub_params]
        return torch.cat(gathered_sub_params, dim=partition_dim)
    return _all_gather_along_dim(tensor_view, partition_dim, mp_group, tp_world_size)


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
        if self.bias is not None and self.bias.numel() != _shape_prod(self._output_shape):
            raise ValueError(f"AutoTP layer '{self.name}' bias size {self.bias.numel()} does not match output shape "
                             f"{self._output_shape}.")

        self._tp_partition([self.weight, self.bias])
        self.support_training = True
        self.config_tp_params(self.weight)
        if self.bias is not None:
            self.config_tp_params(self.bias)

    def forward(self, input):
        if getattr(self, 'mp_group', None) is not None:
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
            params_list[idx].data_partition = param.data
            if idx == 0:
                full_view = _gather_logical_tensor(param,
                                                   self._logical_shape,
                                                   self.partition_dim,
                                                   self.mp_group,
                                                   self.tp_world_size,
                                                   name=self.name,
                                                   subparam_sizes=self._subparam_sizes)
                params_list[idx].data = full_view.reshape(self._orig_weight_shape)
            else:
                if self._bias_partition_dim is None:
                    params_list[idx].data = param.data
                else:
                    full_bias_view = _gather_logical_tensor(param,
                                                            self._output_shape,
                                                            self._bias_partition_dim,
                                                            self.mp_group,
                                                            self.tp_world_size,
                                                            name=self.name,
                                                            subparam_sizes=self._subparam_sizes)
                    params_list[idx].data = full_bias_view.reshape(self._orig_bias_shape)

    @torch.no_grad()
    def _tp_partition(self, params_list):
        weight = params_list[0]
        if weight is None:
            return

        weight_view = weight.reshape(self._logical_shape)
        partitioned_view = _partition_logical_tensor(weight_view,
                                                     self.partition_dim,
                                                     self.tp_world_size,
                                                     self.tp_index,
                                                     name=self.name,
                                                     subparam_sizes=self._subparam_sizes)
        params_list[0].data = self.move(partitioned_view.reshape(-1, partitioned_view.shape[-1])).detach()

        if params_list[1] is not None:
            if self._bias_partition_dim is None:
                params_list[1].data = self.move(params_list[1]).detach()
            else:
                bias_view = params_list[1].reshape(self._output_shape)
                bias_partitioned = _partition_logical_tensor(bias_view,
                                                             self._bias_partition_dim,
                                                             self.tp_world_size,
                                                             self.tp_index,
                                                             name=self.name,
                                                             subparam_sizes=self._subparam_sizes)
                params_list[1].data = self.move(bias_partitioned.reshape(-1)).detach()


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

        self._tp_partition([self.weight, self.bias])
        self.support_training = True
        self.config_tp_params(self.weight)
        if self.bias is not None:
            self.config_requires_grad(self.bias)

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        output = RowParallel.apply(self.mp_group, output, not self.is_training_mode())
        if self.bias is not None:
            output = add_bias(output, self.bias)
        return output

    @torch.no_grad()
    def gather_params(self, params_list):
        """Gather partitioned parameters back to full size."""
        for idx, param in enumerate(params_list):
            if param is None or idx > 0:
                # don't gather bias for row parallel
                return
            params_list[idx].data_partition = param.data
            full_view = _gather_logical_tensor(param,
                                               self._logical_shape,
                                               self.partition_dim,
                                               self.mp_group,
                                               self.tp_world_size,
                                               name=self.name,
                                               subparam_sizes=self._subparam_sizes)
            params_list[idx].data = full_view.reshape(self._orig_weight_shape)

    @torch.no_grad()
    def _tp_partition(self, params_list):
        weight = params_list[0]
        if weight is None:
            return

        weight_view = weight.reshape(self._logical_shape)
        partitioned_view = _partition_logical_tensor(weight_view,
                                                     self.partition_dim,
                                                     self.tp_world_size,
                                                     self.tp_index,
                                                     name=self.name,
                                                     subparam_sizes=self._subparam_sizes)
        params_list[0].data = self.move(partitioned_view.reshape(-1, partitioned_view.shape[-1])).detach()

        # Bias is not partitioned for row parallel (it's applied after all-reduce)
        if params_list[1] is not None:
            params_list[1].data = self.move(params_list[1]).detach()


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
