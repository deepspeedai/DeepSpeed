'''
Copyright 2019 The Microsoft DeepSpeed Team

Copyright NVIDIA/Megatron

Helper functions and classes from multiple sources.
'''

import os
import psutil
import gc
from math import ceil
from math import floor
from bisect import bisect_left, bisect_right

import torch
import torch.distributed as dist
from torch._six import inf
import torch.distributed as dist

from deepspeed.utils import logger
from numpy import prod


def noop_decorator(func):
    return func


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


def move_to_device(item, device):
    """
    Move tensor onto device. Works on individual tensors, and tensors contained/nested in lists, tuples, and dicts.
    Parameters:
        item: tensor to move or (possibly nested) container of tensors to move.
        device: target device

    Returns:
        None
    """
    if torch.is_tensor(item):
        return item.to(device)
    elif isinstance(item, list):
        return [move_to_device(v, device) for v in item]
    elif isinstance(item, tuple):
        return tuple([move_to_device(v, device) for v in item])
    elif isinstance(item, dict):
        return {k: move_to_device(v, device) for k, v in item.items()}
    else:
        return item


class CheckOverflow(object):
    '''Checks for overflow in gradient across parallel process'''
    def __init__(self,
                 param_groups=None,
                 mpu=None,
                 zero_reduce_scatter=False,
                 deepspeed=None):
        self.mpu = mpu
        self.params = [] if param_groups else None
        self.zero_reduce_scatter = zero_reduce_scatter
        self.deepspeed = deepspeed
        if param_groups:
            for group in param_groups:
                for param in group:
                    self.params.append(param)

    def check_using_norm(self, norm_group, reduce_overflow=True):
        # TODO: I don't think reduce_overflow is needed if mpu is None
        overflow = -1 in norm_group

        if self.mpu is not None:
            overflow_gpu = torch.cuda.ByteTensor([overflow])
            torch.distributed.all_reduce(overflow_gpu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=self.mpu.get_model_parallel_group())
            overflow = overflow_gpu[0].item()
        elif reduce_overflow:
            cuda_overflow = torch.cuda.FloatTensor([overflow])
            dist.all_reduce(cuda_overflow, op=torch.distributed.ReduceOp.MAX)
            dist.barrier()
            overflow = cuda_overflow[0].item()

        return bool(overflow)

    def check(self, param_groups=None):
        params = []
        if param_groups is None:
            params = self.params
        else:
            assert param_groups is not None, \
                "self.params and param_groups both cannot be none"

            for group in param_groups:
                for param in group:
                    params.append(param)

        return self.has_overflow(params)

    # `params` is a list / generator of torch.Variable
    def has_overflow_serial(self, params):
        for i, p in enumerate(params):
            if p.grad is not None and self._has_inf_or_nan(p.grad.data, i):
                return True
        return False

    def has_overflow(self, params):
        overflow = self.has_overflow_serial(params)
        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        overflow_gpu = torch.cuda.ByteTensor([overflow])
        # torch.distributed.all_reduce(overflow_gpu,
        #                             op=torch.distributed.ReduceOp.MAX,
        #                             group=mpu.get_model_parallel_group())
        if self.zero_reduce_scatter:
            torch.distributed.all_reduce(overflow_gpu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=torch.distributed.group.WORLD)
        elif self.mpu is not None:
            if self.deepspeed is not None:
                using_pipeline = hasattr(self.deepspeed,
                                         'pipeline_enable_backward_allreduce')
                if (using_pipeline
                        and self.deepspeed.pipeline_enable_backward_allreduce is False
                    ) or (not using_pipeline
                          and self.deepspeed.enable_backward_allreduce is False):
                    torch.distributed.all_reduce(
                        overflow_gpu,
                        op=torch.distributed.ReduceOp.MAX,
                        group=self.mpu.get_data_parallel_group())
            torch.distributed.all_reduce(overflow_gpu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=self.mpu.get_model_parallel_group())
        elif self.deepspeed is not None and self.deepspeed.enable_backward_allreduce is False:
            torch.distributed.all_reduce(overflow_gpu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=torch.distributed.group.WORLD)

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
    rank = torch.distributed.get_rank()
    if rank == 0:
        t_i = -1
        for v_i, v in enumerate(x.data.contiguous().view(-1)):
            if not math.isfinite(float(v)):
                t_i = v_i
                break
        logger.info(
            f"rank {rank} detected overflow {cpu_sum} in tensor {i}:{t_i} shape {x.shape}"
        )


def get_grad_norm(parameters, norm_type=2, mpu=None):
    """Clips gradient norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

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

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # Take max across all GPUs.
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.
        for p in parameters:
            if mpu is not None:
                if (mpu.get_model_parallel_rank() == 0
                    ) or is_model_parallel_parameter(p):
                    param_norm = p.grad.data.float().norm(norm_type)
                    total_norm += param_norm.item()**norm_type
            else:
                param_norm = p.grad.data.float().norm(norm_type)
                total_norm += param_norm.item()**norm_type

        # Sum across all model parallel GPUs.
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

    if total_norm == float(
            'inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm


def get_weight_norm(parameters, norm_type=2, mpu=None):
    """Clips gradient norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

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

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in parameters)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # Take max across all GPUs.
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.
        for p in parameters:
            if mpu is not None:
                if (mpu.get_model_parallel_rank() == 0
                    ) or is_model_parallel_parameter(p):
                    try:
                        param_norm = float(torch.norm(p, norm_type, dtype=torch.float32))
                    except TypeError as err:
                        param_norm = float(torch.norm(p.float(), norm_type))

                    #param_norm = p.data.float().norm(norm_type)
                    total_norm += param_norm**norm_type
            else:
                try:
                    param_norm = float(torch.norm(p, norm_type, dtype=torch.float32))
                except TypeError as err:
                    param_norm = float(torch.norm(p.float(), norm_type))
                #param_norm = p.data.float().norm(norm_type)
                total_norm += param_norm**norm_type

        # Sum across all model parallel GPUs.
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

    if total_norm == float(
            'inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm


def is_model_parallel_parameter(p):
    return hasattr(p, 'model_parallel') and p.model_parallel


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
    parts = [0] * (num_parts + 1)
    # First check for the trivial edge case
    if num_items <= num_parts:
        for p in range(num_parts + 1):
            parts[p] = min(p, num_items)
        return parts

    chunksize = floor(num_items / num_parts)
    for p in range(num_parts):
        parts[p] = min(chunksize * p, num_items)
    parts[num_parts] = num_items
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
        parts[p] = bisect_left(weights,
                               bsum,
                               lo=step - chunksize,
                               hi=min(step,
                                      num_items))
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
    def from_meta(cls, meta, local_part, group, device='cuda'):
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
        tensor_part = tensor.detach().contiguous().view(-1).narrow(
            0,
            start=start,
            length=length).clone()

        return tensor_part, partition

    def full(self, device=None):
        if device is None:
            device = self.orig_device

        # Allocate the full tensor as a flat buffer.
        full_numel = prod(self.full_size())
        flat_tensor = torch.zeros([full_numel],
                                  dtype=self.local_data.dtype,
                                  device=device)

        # Prepare all-gather buffer
        partition_tensors = []
        for part_id in range(self.num_parts):
            part_size = self.partition[part_id + 1] - self.partition[part_id]
            buf = flat_tensor.narrow(0, start=self.partition[part_id], length=part_size)
            if part_id == self.rank:
                buf.copy_(self.local_data)
            partition_tensors.append(buf)

        # Collect the full tensor
        dist.all_gather(partition_tensors,
                        partition_tensors[self.rank],
                        group=self.group)

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

    torch.cuda.synchronize()

    if reset_max:
        torch.cuda.reset_max_memory_cached()
        torch.cuda.reset_max_memory_allocated()

    new_alloced = torch.cuda.memory_allocated()
    new_cached = torch.cuda.memory_cached()

    delta_alloced = new_alloced - mem_alloced
    delta_cached = new_cached - mem_cached

    mem_cached = new_cached
    mem_alloced = new_alloced

    max_alloced = torch.cuda.max_memory_allocated()
    max_cached = torch.cuda.max_memory_cached()

    # convert to GB for printing
    new_alloced /= 1024**3
    new_cached /= 1024**3
    delta_alloced /= 1024**3
    delta_cached /= 1024**3
    max_alloced /= 1024**3
    max_cached /= 1024**3

    print(
        f'RANK={rank} MEMSTATS',
        msg,
        f'device={torch.cuda.current_device()} '
        f'current alloc={new_alloced:0.4f}GB (delta={delta_alloced:0.4f}GB max={max_alloced:0.4f}GB) '
        f'current cache={new_cached:0.4f}GB (delta={delta_cached:0.4f}GB max={max_cached:0.4f}GB)'
    )


def see_memory_usage(message, force=False):
    if not force:
        return
    if torch.distributed.is_initialized() and not torch.distributed.get_rank() == 0:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    logger.info(message)
    logger.info(
        f"MA {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        Max_MA {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        CA {round(torch.cuda.memory_cached() / (1024 * 1024 * 1024),2)} GB \
        Max_CA {round(torch.cuda.max_memory_cached() / (1024 * 1024 * 1024))} GB ")

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    logger.info(
        f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')

    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()


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
