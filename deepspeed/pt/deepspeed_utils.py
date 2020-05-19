'''
Copyright 2019 The Microsoft DeepSpeed Team

Copyright NVIDIA/Megatron

Helper functions and classes from multiple sources.
'''

import torch
from torch._six import inf


class CheckOverflow(object):
    '''Checks for overflow in gradient across parallel process'''
    def __init__(self, param_groups=None, mpu=None, zero_reduce_scatter=False):
        self.mpu = mpu
        self.params = [] if param_groups else None
        self.zero_reduce_scatter = zero_reduce_scatter
        if param_groups:
            for group in param_groups:
                for param in group:
                    self.params.append(param)

    def check_using_norm(self, norm_group):
        overflow = -1 in norm_group

        if self.mpu is not None:
            overflow_gpu = torch.cuda.ByteTensor([overflow])
            torch.distributed.all_reduce(overflow_gpu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=self.mpu.get_model_parallel_group())
            overflow = overflow_gpu[0].item()

        return bool(overflow)

    def check(self, param_groups=None):

        #TODO: what's the equivalent here? do we need this?
        # for group in self.fp32_from_fp32_groups:
        #     for param in group:
        #         params.append(param)

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
        #torch.distributed.all_reduce(overflow_gpu,
        #                             op=torch.distributed.ReduceOp.MAX,
        #                             group=mpu.get_model_parallel_group())
        if self.zero_reduce_scatter:
            torch.distributed.all_reduce(overflow_gpu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=torch.distributed.group.WORLD)
        elif self.mpu is not None:
            torch.distributed.all_reduce(overflow_gpu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=self.mpu.get_model_parallel_group())
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
                _handle_overflow(cpu_sum, x, i)
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
        print(
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
                if (mpu.get_model_parallel_rank() == 0) or (hasattr(p,
                                                                    'model_parallel')
                                                            and p.model_parallel):
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
                if (mpu.get_model_parallel_rank() == 0) or (hasattr(p,
                                                                    'model_parallel')
                                                            and p.model_parallel):
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


def see_memory_usage(message):
    return
    if torch.distributed.is_initialized() and not torch.distributed.get_rank() == 0:
        return

    # Print message except when distributed but not rank 0
    print(message, flush=True)
    print("Memory Allocated ",
          torch.cuda.memory_allocated() / (1024 * 1024 * 1024),
          "GigaBytes",
          flush=True)
    print("Max Memory Allocated ",
          torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),
          "GigaBytes",
          flush=True)
    print("Cache Allocated ",
          torch.cuda.memory_cached() / (1024 * 1024 * 1024),
          "GigaBytes",
          flush=True)
    print("Max cache Allocated ",
          torch.cuda.max_memory_cached() / (1024 * 1024 * 1024),
          "GigaBytes",
          flush=True)
    print(" ", flush=True)
