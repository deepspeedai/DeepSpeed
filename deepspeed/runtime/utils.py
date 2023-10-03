# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Copyright NVIDIA/Megatron

Helper functions and classes from multiple sources.
"""

from collections.abc import Iterable
from deepspeed.moe.utils import is_moe_param
import os
import psutil
import gc
from math import sqrt
from bisect import bisect_left
from packaging import version as pkg_version

import torch
from deepspeed import comm as dist

try:
    from torch._six import inf
except ModuleNotFoundError:
    from torch import inf

from deepspeed.utils import groups, logger
from deepspeed.runtime.constants import PIPE_REPLICATED
from numpy import prod
from deepspeed.accelerator import get_accelerator

from deepspeed.module_inject.policy import transpose
from torch.nn import functional as F

torch_memory_reserved = get_accelerator().memory_reserved
torch_max_memory_reserved = get_accelerator().max_memory_reserved


class DummyOptim():
    """
    Dummy optimizer presents model parameters as a param group, this is
    primarily used to allow ZeRO-3 without an optimizer
    """

    def __init__(self, params):
        self.param_groups = []
        self.param_groups.append({'params': params})


def noop_decorator(func):
    return func


class noop_context(object):

    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def ensure_directory_exists(filename):
    """Create the directory path to ``filename`` if it does not already exist.

    Args:
        filename (str): A file path.
    """
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)


def set_random_seed(seed):
    """Set the random seed for common PRNGs used during training: random, numpy, and torch.

    Args:
        seed (int): the seed to use
    """
    import numpy
    import random
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def is_model_parallel_parameter(p) -> bool:
    if hasattr(p, 'model_parallel') and p.model_parallel:
        return True

    if hasattr(p, 'tensor_model_parallel') and p.tensor_model_parallel:
        return True

    return False


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


def copy_to_device(item, device, criterion_func):
    """
    Return a copy of tensor on specified device.
    Works on individual tensors, and tensors contained/nested in lists, tuples, and dicts.
    Parameters:
        item: tensor to copy or (possibly nested) container of tensors to copy.
        device: target device
        criterion_func: Function to restrict copy operation to items meet criterion

    Returns:
        None
    """
    if criterion_func(item):
        return item.to(device)
    elif isinstance(item, list):
        return [copy_to_device(v, device, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([copy_to_device(v, device, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: copy_to_device(v, device, criterion_func) for k, v in item.items()}
    else:
        return item


def move_to_device(item, device, criterion_func):
    """
    Move tensor on to specified device by changing the storage.
    Works on individual tensors, and tensors contained/nested in lists, tuples, and dicts.
    Parameters:
        item: tensor to move or (possibly nested) container of tensors to move.
        device: target device
        criterion_func: Function to restrict move operation to items meet criterion

    Returns:
        None
    """
    if criterion_func(item):
        device_copy = item.to(device)
        item.data = device_copy.data
        return item
    elif isinstance(item, list):
        return [move_to_device(v, device, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([move_to_device(v, device, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: move_to_device(v, device, criterion_func) for k, v in item.items()}
    else:
        return item


class CheckOverflow(object):
    '''Checks for overflow in gradient across parallel process'''

    def __init__(self, param_groups=None, mpu=None, zero_reduce_scatter=False, deepspeed=None):
        self.mpu = mpu
        self.params = [] if param_groups else None
        self.zero_reduce_scatter = zero_reduce_scatter
        self.deepspeed = deepspeed
        self.has_moe_params = False
        if param_groups:
            for group in param_groups:
                for param in group:
                    self.params.append(param)
                    if is_moe_param(param):
                        self.has_moe_params = True

    def check_using_norm(self, norm_group, reduce_overflow=True):
        # TODO: I don't think reduce_overflow is needed if mpu is None
        overflow = -1 in norm_group
        overflow_gpu = get_accelerator().FloatTensor([overflow])
        if self.has_moe_params:
            # In this case, we need to do an all_reduce across
            # the expert_parallel_group, so that if there was
            # an overflow due to expert weights, we detect it

            # Only need to check groups.get_largest_expert_parallel_group()
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=groups._get_max_expert_parallel_group())
        if self.mpu is not None:
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.mpu.get_model_parallel_group())
        elif reduce_overflow:
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX)
            dist.barrier()
        overflow = overflow_gpu[0].item()
        return bool(overflow)

    def check(self, param_groups=None):
        params = []
        has_moe_params = False
        if param_groups is None:
            params = self.params
            has_moe_params = self.has_moe_params
        else:
            assert param_groups is not None, \
                "self.params and param_groups both cannot be none"

            for group in param_groups:
                for param in group:
                    params.append(param)
                    if is_moe_param(param):
                        has_moe_params = True

        return self.has_overflow(params, has_moe_params=has_moe_params)

    # `params` is a list / generator of torch.Variable
    def has_overflow_serial(self, params):
        for i, p in enumerate(params):
            if p.grad is not None and self._has_inf_or_nan(p.grad.data, i):
                return True
        return False

    def has_overflow(self, params, has_moe_params=None):
        if has_moe_params is None:
            has_moe_params = self.has_moe_params
        overflow = self.has_overflow_serial(params)
        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        overflow_gpu = get_accelerator().ByteTensor([overflow])
        # deepspeed.comm.all_reduce(overflow_gpu,
        #                             op=deepspeed.comm.ReduceOp.MAX,
        #                             group=mpu.get_model_parallel_group())
        if has_moe_params:
            # All reduce this across expert_parallel_group, so that if an expert
            # overflows, we detect it here
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=groups._get_max_expert_parallel_group())
        if self.zero_reduce_scatter:
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=dist.get_world_group())
        elif self.mpu is not None:
            if self.deepspeed is not None:
                using_pipeline = hasattr(self.deepspeed, 'pipeline_enable_backward_allreduce')
                if (using_pipeline and self.deepspeed.pipeline_enable_backward_allreduce is False) or (
                        not using_pipeline and self.deepspeed.enable_backward_allreduce is False):
                    dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.mpu.get_data_parallel_group())
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.mpu.get_model_parallel_group())
        elif self.deepspeed is not None and self.deepspeed.enable_backward_allreduce is False:
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=dist.get_world_group())

        overflow = overflow_gpu[0].item()
        return bool(overflow)

    # `x` is a torch.Tensor
    @staticmethod
    def _has_inf_or_nan(x, i):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False


def _handle_overflow(cpu_sum, x, i):
    import math
    rank = dist.get_rank()
    if rank == 0:
        t_i = -1
        for v_i, v in enumerate(x.data.contiguous().view(-1)):
            if not math.isfinite(float(v)):
                t_i = v_i
                break
        logger.info(f"rank {rank} detected overflow {cpu_sum} in tensor {i}:{t_i} shape {x.shape}")


def get_global_norm(norm_list):
    """ Compute total from a list of norms
    """
    total_norm = 0.0
    for norm in norm_list:
        total_norm += norm**2.0
    # logger.info(f'norm_list = {norm_list} global = {sqrt(total_norm)}')
    return sqrt(total_norm)


def clip_grad_norm_(parameters, max_norm, norm_type=2, mpu=None):
    """Clips gradient norm of an iterable of parameters.

    This has been adapted from Nvidia megatron. We add norm averaging
    to consider MoE params when calculating norm as they will result
    in different norms across different ranks.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        # Take max across all GPUs.
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0
        for p in parameters:
            if mpu is not None:
                if (mpu.get_model_parallel_rank() == 0) or is_model_parallel_parameter(p):
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item()**norm_type
            else:
                param_norm = p.grad.data.float().norm(norm_type)
                total_norm += param_norm.item()**norm_type

        # Sum across all model parallel GPUs.
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

    # Need to average total_norm across different GPUs due to the presence of moe params
    pg = groups._get_data_parallel_group()
    scaled_norm = total_norm * 1.0 / float(dist.get_world_size(group=pg))

    scaled_norm_tensor = get_accelerator().FloatTensor([float(scaled_norm)])
    dist.all_reduce(scaled_norm_tensor, group=pg)
    total_norm = scaled_norm_tensor.item()

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm


def get_grad_norm(parameters, norm_type=2, mpu=None):
    """Get grad norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        # Take max across all GPUs.
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.
        tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=mpu)
        for p in parameters:
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue

            # Filter to avoid over-counting replicated tensors from tensor
            # model parallelism
            if (tensor_mp_rank > 0) and not is_model_parallel_parameter(p):
                continue

            param_norm = p.grad.data.float().norm(norm_type)
            total_norm += param_norm.item()**norm_type

        # Sum across all model parallel GPUs.
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

    if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm


def get_grad_zeros(parameters, mpu=None):
    """Compute the number of grads with zero values.

    This is adapted from get_grad_norm

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized

    Returns:
        Total number of params with zero values (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_zeros = 0.
    tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=mpu)
    for p in parameters:
        # Pipeline parallelism may replicate parameters. Avoid multi-counting.
        if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
            continue

        # Filter to avoid over-counting replicated tensors from tensor
        # model parallelism
        if (tensor_mp_rank > 0) and not is_model_parallel_parameter(p):
            continue

        count_zeros = p.grad.numel() - torch.count_nonzero(p.grad)
        total_zeros += count_zeros.item()

    # Sum across all model parallel GPUs.
    total_zeros_cuda = get_accelerator().FloatTensor([float(total_zeros)])
    if mpu is not None:
        dist.all_reduce(total_zeros_cuda, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
    total_zeros = total_zeros_cuda[0].item()

    return total_zeros


def get_weight_norm(parameters, norm_type=2, mpu=None):
    """Get norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
        -1 if the norm value is NaN or Inf.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in parameters)
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        # Take max across all GPUs.
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.
        tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=mpu)
        for p in parameters:
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue

            # Filter to avoid over-counting replicated tensors from tensor
            # model parallelism
            if (tensor_mp_rank > 0) and not is_model_parallel_parameter(p):
                continue

            param_norm = p.data.float().norm(norm_type)
            total_norm += param_norm**norm_type

        # Sum across all model parallel GPUs.
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

    if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm


def prefix_sum_inc(weights):
    """ Compute an inclusive prefix sum.

    Example:
        >>> prefix_sum_inc([3,4,5])
        [3, 7, 12]
    """
    weights_ = [w for w in weights]
    for x in range(1, len(weights_)):
        weights_[x] += weights_[x - 1]
    return weights_


def partition_uniform(num_items, num_parts):
    import numpy
    parts = [0] * (num_parts + 1)
    # First check for the trivial edge case
    if num_items <= num_parts:
        for p in range(num_parts + 1):
            parts[p] = min(p, num_items)
        return parts

    chunksize = num_items // num_parts
    residual = num_items - (chunksize * num_parts)

    parts = numpy.arange(0, (num_parts + 1) * chunksize, chunksize)

    for i in range(residual):
        parts[i + 1:] += 1
    parts = parts.tolist()

    return parts


def _lprobe(weights, num_parts, bottleneck):
    num_items = len(weights)
    total_weight = weights[-1]

    # initialize partitioning
    parts = [0] * (num_parts + 1)
    for p in range(1, num_parts + 1):
        parts[p] = num_items

    bsum = bottleneck  # running sum of target weight for pth partition
    chunksize = num_items // num_parts
    step = chunksize
    for p in range(1, num_parts):
        # Jump to the next bucket
        while (step < num_items) and (weights[step] < bsum):
            step += chunksize

        # Find the end index of partition p
        parts[p] = bisect_left(weights, bsum, lo=step - chunksize, hi=min(step, num_items))
        # Nothing more to partition, return early
        if parts[p] == num_items:
            # See if the current partition is overweight.
            part_size = weights[-1] - weights[parts[p - 1]]
            return parts, part_size < bottleneck

        # Next partition target
        bsum = weights[parts[p] - 1] + bottleneck

    return parts, bsum >= total_weight


def _rb_partition_balanced(weights, num_parts, eps):
    total_weight = weights[-1]
    lower = total_weight / num_parts  # best case heaviest partition
    upper = total_weight  # worst case heaviest partition

    # Do a binary search for the best partitioning
    while upper > lower + eps:
        mid = lower + ((upper - lower) / 2)
        parts, success = _lprobe(weights, num_parts, mid)
        if success:
            upper = mid
        else:
            lower = mid + eps
    return upper


def partition_balanced(weights, num_parts, eps=1e-3):
    num_items = len(weights)
    # First check for the trivial edge case
    if num_items <= num_parts:
        return partition_uniform(num_items, num_parts)

    weights_ = prefix_sum_inc(weights)

    # Find the smallest bottleneck (weight of heaviest partition)
    bottleneck = _rb_partition_balanced(weights_, num_parts, eps=eps)

    # Now compute that partitioning
    parts, success = _lprobe(weights_, num_parts, bottleneck)
    assert success

    return parts


class PartitionedTensor:

    def __init__(self, tensor, group, partition_meta=None):
        super().__init__()

        self.group = group
        self.num_parts = dist.get_world_size(group=self.group)
        self.rank = dist.get_rank(group=self.group)

        self.orig_size = list(tensor.size())
        self.orig_device = tensor.device
        self.local_data, self.partition = self._partition_tensor(tensor)

    @classmethod
    def from_meta(cls, meta, local_part, group, device=get_accelerator().device_name()):
        assert meta.dtype == torch.long
        dummy = torch.ones(dist.get_world_size(group=group))
        part_obj = cls(tensor=dummy, group=group)

        meta = meta.tolist()

        # [N, list0, ..., listN-1]
        part_obj.orig_size = meta[1:(1 + meta[0])]
        meta = meta[1 + meta[0]:]

        part_obj.orig_device = device
        part_obj.local_data = local_part.detach()

        part_obj.group = group

        # Partition is encoded like the rowptr of a CSR matrix:
        # [num_parts, rank, 0, part_1, ..., part_num_parts]
        # TODO: support shuffle between different partition granularities
        assert part_obj.num_parts == meta[0]
        assert part_obj.rank == meta[1]
        part_obj.partition = meta[2:]  # length num_parts+1

        return part_obj

    def _partition_tensor(self, tensor):
        partition = partition_uniform(num_items=tensor.numel(), num_parts=self.num_parts)
        start = partition[self.rank]
        length = partition[self.rank + 1] - start
        tensor_part = tensor.detach().contiguous().view(-1).narrow(0, start=start, length=length).clone()

        return tensor_part, partition

    def full(self, device=None):
        if device is None:
            device = self.orig_device

        # Allocate the full tensor as a flat buffer.
        full_numel = prod(self.full_size())
        flat_tensor = torch.zeros([full_numel], dtype=self.local_data.dtype, device=device)

        # Prepare all-gather buffer
        partition_tensors = []
        for part_id in range(self.num_parts):
            part_size = self.partition[part_id + 1] - self.partition[part_id]
            buf = flat_tensor.narrow(0, start=self.partition[part_id], length=part_size)
            if part_id == self.rank:
                buf.copy_(self.local_data)
            partition_tensors.append(buf)

        # Collect the full tensor
        dist.all_gather(partition_tensors, partition_tensors[self.rank], group=self.group)

        for i in range(len(partition_tensors)):
            partition_tensors[i].data = torch.zeros(1)
            partition_tensors[i] = None

        return flat_tensor.view(self.full_size()).clone().detach()

    def to_meta(self):
        """Returns a torch.LongTensor that encodes partitioning information.

        Can be used along with ``data()`` to serialize a ``PartitionedTensor`` for
        communication.

        Returns:
            torch.LongTensor: a tensor encoding the meta-information for the partitioning
        """
        meta = []
        meta.append(len(self.orig_size))
        meta += list(self.orig_size)
        meta.append(self.num_parts)
        meta.append(self.rank)
        meta += self.partition
        return torch.LongTensor(data=meta).to(self.orig_device)

    def data(self):
        return self.local_data

    def local_size(self):
        return self.local_data.size()

    def full_size(self):
        return self.orig_size


mem_alloced = 0
mem_cached = 0


def memory_status(msg, print_rank=-1, reset_max=False):
    global mem_alloced, mem_cached

    rank = dist.get_rank()
    if print_rank != -1 and rank != print_rank:
        return

    get_accelerator().synchronize()

    if reset_max:
        get_accelerator().reset_max_memory_cached()
        get_accelerator().reset_max_memory_allocated()

    new_alloced = get_accelerator().memory_allocated()
    new_cached = get_accelerator().memory_cached()

    delta_alloced = new_alloced - mem_alloced
    delta_cached = new_cached - mem_cached

    mem_cached = new_cached
    mem_alloced = new_alloced

    max_alloced = get_accelerator().max_memory_allocated()
    max_cached = get_accelerator().max_memory_cached()

    # convert to GB for printing
    new_alloced /= 1024**3
    new_cached /= 1024**3
    delta_alloced /= 1024**3
    delta_cached /= 1024**3
    max_alloced /= 1024**3
    max_cached /= 1024**3

    print(
        f'RANK={rank} MEMSTATS', msg, f'device={get_accelerator().current_device_name()} '
        f'current alloc={new_alloced:0.4f}GB (delta={delta_alloced:0.4f}GB max={max_alloced:0.4f}GB) '
        f'current cache={new_cached:0.4f}GB (delta={delta_cached:0.4f}GB max={max_cached:0.4f}GB)')


def get_ma_status():
    if dist.is_initialized() and not dist.get_rank() == 0:
        return 0
    return get_accelerator().memory_allocated()


def empty_cache():
    get_accelerator().empty_cache()
    get_accelerator().reset_peak_memory_stats()


def see_memory_usage(message, force=False):
    if not force:
        return
    if dist.is_initialized() and not dist.get_rank() == 0:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    logger.info(message)
    logger.info(f"MA {round(get_accelerator().memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        Max_MA {round(get_accelerator().max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        CA {round(torch_memory_reserved() / (1024 * 1024 * 1024),2)} GB \
        Max_CA {round(torch_max_memory_reserved() / (1024 * 1024 * 1024))} GB ")

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    logger.info(f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')

    # get the peak memory to report correct data, so reset the counter for the next call
    get_accelerator().reset_peak_memory_stats()


def call_to_str(base, *args, **kwargs):
    """Construct a string representation of a call.

    Args:
        base (str): name of the call
        args (tuple, optional): args to ``base``
        kwargs (dict, optional): kwargs supplied to ``base``

    Returns:
        str: A string representation of base(*args, **kwargs)
    """
    name = f'{base}('
    if args:
        name += ', '.join(repr(arg) for arg in args)
        if kwargs:
            name += ', '
    if kwargs:
        name += ', '.join(f'{key}={repr(arg)}' for key, arg in kwargs.items())
    name += ')'
    return name


def get_only_unique_item(items):
    item_set = set(items)
    if len(item_set) != 1:
        raise RuntimeError(f"expected there to be only one unique element in {items}")
    unique_item, = item_set

    return unique_item


def clip_gradients(parameters, max_norm=1.0, global_grad_norm=None, mpu=None, eps=1e-6):
    """Clip the gradient of a list of parameters.
    Args:
        parameters: List of parameters whose .grad will be clipped.
        global_grad_norm (float, optional): Precomputed gradient norm. Defaults to None.
        mpu (optional): model parallelism unit. Defaults to None.
        eps (float, optional): epsilon value added to grad norm. Defaults to 1e-6
    Returns:
        float: the global gradient norm
    """
    if global_grad_norm is None:
        global_grad_norm = get_grad_norm(parameters, mpu=mpu)
    clip_coef = max_norm / (global_grad_norm + eps)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
    return global_grad_norm


def get_global_norm_of_tensors(input_tensors, norm_type=2, mpu=None):
    """Get norm of an iterable of tensors.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Taken from Nvidia Megatron.

    Arguments:
        input_tensors (Iterable[Tensor]): an iterable of Tensors will have norm computed
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the tensors (viewed as a single vector).
    """

    assert isinstance(input_tensors, Iterable), f'expected Iterable type not {type(input_tensors)}'
    assert all([torch.is_tensor(t) for t in input_tensors]), f'expected list of only tensors'

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(t.data.abs().max() for t in input_tensors)
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
            total_norm = total_norm_cuda[0].item()
    else:
        total_norm = sum([t.data.float().norm(norm_type).item()**norm_type for t in input_tensors])
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

    if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm


def clip_tensors_by_global_norm(input_tensors, max_norm=1.0, global_norm=None, mpu=None, eps=1e-6):
    """Clip list of tensors by global norm.
    Args:
        input_tensors: List of tensors to be clipped
        global_norm (float, optional): Precomputed norm. Defaults to None.
        mpu (optional): model parallelism unit. Defaults to None.
        eps (float, optional): epsilon value added to grad norm. Defaults to 1e-6
    Returns:
        float: the global norm
    """
    if global_norm is None:
        global_norm = get_global_norm_of_tensors(input_tensors, mpu=mpu)

    clip_coef = max_norm / (global_norm + eps)

    if clip_coef < 1:
        for t in input_tensors:
            t.detach().mul_(clip_coef)

    return global_norm


def align_dense_tensors(tensor_list, alignment):
    num_elements = sum(t.numel() for t in tensor_list)
    remaining = num_elements % alignment

    if remaining:
        elements_to_add = alignment - remaining
        pad_tensor = torch.zeros(elements_to_add, device=tensor_list[0].device, dtype=tensor_list[0].dtype)
        padded_tensor_list = tensor_list + [pad_tensor]
    else:
        padded_tensor_list = tensor_list

    return padded_tensor_list


def all_gather_dp_groups(partitioned_param_groups, dp_process_group, start_alignment_factor, allgather_bucket_size):
    for group_id, partitioned_params in enumerate(partitioned_param_groups):
        # Sequential AllGather Best of both worlds
        partition_id = dist.get_rank(group=dp_process_group[group_id])
        dp_world_size = dist.get_world_size(group=dp_process_group[group_id])

        if dp_world_size == 1:
            # no groups share optimizer states
            # pipeline parallel with bf16 will default call this even if dp size = 1.
            continue
        num_shards = max(1, partitioned_params[partition_id].numel() * dp_world_size // allgather_bucket_size)

        shard_size = partitioned_params[partition_id].numel() // num_shards

        # Enforce nccl/rccl alignment of start location of each shard
        shard_size = shard_size - (shard_size % start_alignment_factor)

        num_elements = shard_size

        assert shard_size * num_shards <= partitioned_params[partition_id].numel()

        for shard_id in range(num_shards):

            if shard_id == (num_shards - 1):
                num_elements = partitioned_params[partition_id].numel() - shard_id * shard_size

            shard_list = []
            for dp_id in range(dp_world_size):
                curr_shard = partitioned_params[dp_id].narrow(0, shard_id * shard_size, num_elements).detach()
                shard_list.append(curr_shard)

            dist.all_gather(shard_list, shard_list[partition_id], dp_process_group[group_id])


class TLinear(torch.nn.Linear):

    def __init__(self, orig_layer, name=""):
        self.name = name
        super().__init__(orig_layer.weight.shape[1], orig_layer.weight.shape[0], bias=(orig_layer.bias is not None))
        self.weight.data = transpose(orig_layer.weight.data)
        self.bias = orig_layer.bias
        self._fwd_func = self._fwd_bias_add if self.bias is not None else self._fwd

    def _fwd(self, input):
        return F.linear(input, self.weight)

    def _fwd_bias_add(self, input):
        return F.linear(input, self.weight, bias=self.bias)

    def forward(self, input):
        return self._fwd_func(input)


def get_inactive_params(param_list):
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    return [param for param in param_list if (hasattr(param, 'ds_id') and \
                            param.ds_status == ZeroParamStatus.NOT_AVAILABLE)]


def required_torch_version(min_version=None, max_version=None):
    assert min_version or max_version, "Must provide a min_version or max_version argument"

    torch_version = pkg_version.parse(torch.__version__)

    if min_version and pkg_version.parse(str(min_version)) > torch_version:
        return False

    if max_version and pkg_version.parse(str(max_version)) < torch_version:
        return False

    return True
