# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
from typing import Any, Tuple, Dict
import statistics

import torch
from torch.fx import GraphModule, Interpreter
from torch.fx.node import map_aggregate

try:
    from torch.utils._pytree import tree_all, tree_leaves
    from torch._subclasses.fake_tensor import unset_fake_temporarily, is_fake
except ImportError:
    # Unsupported torch version
    pass

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from ..util import is_comm_op, is_release_node, get_deepcompile_handle


def _all_real_if_tensor(args):
    return tree_all(lambda x: not torch.is_tensor(x) or not is_fake(x), args)


def _to(v, device):
    if torch.is_tensor(v):
        with unset_fake_temporarily():
            return v.to(device)
    return v


def _args_to_key(v):

    def _tensor_to_key(v) -> str:
        if torch.is_tensor(v):
            if v.numel() == 1:
                try:
                    return f"{v.dtype}{v.device}{v.item()}"
                except Exception as e:
                    return f"{v.dtype}{v.device}ptr{v.data_ptr()}"
            else:
                return f"{v.dtype}{v.device}{v.shape}"
        return str(v)

    return map_aggregate(v, _tensor_to_key)


def _node_size(out):
    return sum([v.element_size() * v.numel() for v in tree_leaves(out) if torch.is_tensor(v)])


def _get_mem_usage_out_of_torch():

    adjust = 0
    try:
        import pynvml
        pynvml.nvmlInit()

        current_dev_id = get_accelerator().current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(current_dev_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        torch_alloc = get_accelerator().memory_allocated()
        adjust = info.used - torch_alloc
    except:
        # pynvml not available
        pass

    return adjust


# https://pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html
class ProfilingInterpreter(Interpreter):

    def __init__(self, gm: GraphModule, iteration: int = 10, warmup: int = 5, debug_log=False):
        super().__init__(gm)

        self.nz3 = get_deepcompile_handle()

        assert iteration > 0
        assert warmup >= 0
        self.iteration = iteration
        self.warmup = warmup
        self.device = torch.device(get_accelerator().current_device())
        self.cache: Dict[Tuple, Any] = {}
        self.distributed = dist.is_initialized()
        self.allgather_mem: Dict[int, int] = {}
        self.debug_log = debug_log
        self.mem_usage_out_of_torch = 0

    def run(self, *args) -> Any:
        """Run the graph with profiling enabled.

        args: inputs to the graph. Tensors in the inpusts must be real tensors, not fake tensors. args can contain ds parameters.
        returns: The output of the graph. Tensor in the output is real tensors.
        """
        return_val = None
        try:
            assert _all_real_if_tensor(args), "Inputs must be real tensors"
            self.nz3.enable_profiling(True)

            with unset_fake_temporarily():
                with get_accelerator().random().fork_rng(devices=[self.device]):
                    self.mem_usage_out_of_torch = _get_mem_usage_out_of_torch()
                    return_val = super().run(*args)
        except Exception as e:
            msg = e.msg if "msg" in dir(e) else str(e)
            print(f"Profiling error {msg}")
        finally:
            self.nz3.clear_all_gathered_params()
            self.nz3.enable_profiling(False)
        return return_val

    def run_node(self, n: torch.fx.Node) -> Any:

        if n.op in {"placeholder", "output"}:
            n.meta["device_time"] = 0.0
            n.meta["wall_time"] = 0.0
            n.meta["alloc_mem"] = 0
            n.meta["max_mem"] = 0
            n.meta["tensor_size"] = _node_size(n)
            return super().run_node(n)

        args, kwargs = self.fetch_args_kwargs_from_env(n)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)

        partitioned_params = {}

        def rebuild_param_if_necessary(v):
            if hasattr(v, "ds_id"):
                v.all_gather(param_list=[v])
                if hasattr(v, "ds_target_dtype"):
                    casted = v.to(v.ds_target_dtype)
                    partitioned_params[id(casted)] = v
                    return casted
            return v

        args = map_aggregate(args, lambda x: rebuild_param_if_necessary(x))

        args = map_aggregate(args, lambda x: _to(x, self.device))
        kwargs = map_aggregate(kwargs, lambda x: _to(x, self.device))

        cache_key = (n.target, _args_to_key(args), _args_to_key(kwargs))
        cache_hit = cache_key in self.cache

        cache_hit_flag = torch.tensor([0 if cache_hit else 1], device=self.device, dtype=torch.int)
        if self.distributed:
            dist.all_reduce(cache_hit_flag, dist.ReduceOp.SUM)
        cache_hit = cache_hit_flag.item() == 0

        if cache_hit:
            device_time, wall_time, alloc_mem, max_mem, tensor_size = self.cache[cache_key]
            n.meta["device_time"] = device_time
            n.meta["wall_time"] = wall_time
            n.meta["alloc_mem"] = alloc_mem
            n.meta["max_mem"] = max_mem
            n.meta["tensor_size"] = tensor_size

        is_release_op = is_release_node(n)
        run_only_once = cache_hit or is_release_op
        iteration = 1 if run_only_once else self.iteration
        accelerator = get_accelerator()
        start_events = [accelerator.Event(enable_timing=True) for _ in range(iteration)]
        end_events = [accelerator.Event(enable_timing=True) for _ in range(iteration)]

        get_accelerator().reset_peak_memory_stats()
        alloc_mem_start = get_accelerator().memory_allocated()
        max_mem_start = get_accelerator().max_memory_allocated()

        if not run_only_once:
            for i in range(self.warmup):
                out = getattr(self, n.op)(n.target, args, kwargs)

        if is_comm_op(n):
            assert self.distributed, f"Distributed environment is not initialized but comm operator {n.name} {n.target} is used."
            dist.barrier()

        start = time.time()
        for i in range(iteration):
            start_events[i].record()
            out = getattr(self, n.op)(n.target, args, kwargs)
            end_events[i].record()
        accelerator.synchronize()
        walltime_sum = time.time() - start

        if is_comm_op(n):
            dist.barrier()

        alloc_mem = get_accelerator().memory_allocated() - alloc_mem_start + self.mem_usage_out_of_torch
        max_memory = get_accelerator().max_memory_allocated() - max_mem_start + self.mem_usage_out_of_torch
        tensor_size = _node_size(out)

        def partition_param_if_necessary(v):
            if id(v) in partitioned_params:
                v = partitioned_params[id(v)]
            if hasattr(v, "ds_id") and not v.ds_persist:
                v.partition(param_list=[v], has_been_updated=False)
            return v

        args = map_aggregate(args, lambda x: partition_param_if_necessary(x))

        if not cache_hit:
            device_time = statistics.mean([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
            wall_time = walltime_sum / iteration * 1000

            with unset_fake_temporarily():
                vals_to_bcast = torch.tensor([device_time, wall_time, alloc_mem, max_memory, tensor_size],
                                             device=self.device)
                if self.distributed:
                    dist.all_reduce(vals_to_bcast, dist.ReduceOp.AVG)
                n.meta["device_time"] = vals_to_bcast[0].item()
                n.meta["wall_time"] = vals_to_bcast[1].item()
                n.meta["alloc_mem"] = int(vals_to_bcast[2].item())
                n.meta["max_mem"] = int(vals_to_bcast[3].item())
                n.meta["tensor_size"] = int(vals_to_bcast[4].item())
                self.cache[cache_key] = (n.meta["device_time"], n.meta["wall_time"], n.meta["alloc_mem"],
                                         n.meta["max_mem"], n.meta["tensor_size"])

            if is_release_op:
                n.meta["alloc_mem"] = -self.allgather_mem.get(args[2], 0)

            if dist.get_rank() == 0 and self.debug_log:
                print(
                    f"{n.target} {n.meta['device_time']:.2f}ms {n.meta['wall_time']:.2f}ms alloc_mem={n.meta['alloc_mem'] / 1024 / 1024:.2f}MB max_mem={n.meta['max_mem'] / 1024 / 1024:.2f}MB tensor_size={n.meta['tensor_size']}"
                )

        if n.target == torch.ops.dc.allgather_param.default:
            out = args[0]
            assert hasattr(out, "ds_id")
            if not out.ds_persist:
                self.nz3.invalidate_gathered_param(args[2])
            if "dtype" in n.kwargs:
                setattr(out, "ds_target_dtype", n.kwargs["dtype"])
            self.allgather_mem[out.ds_id] = n.meta["alloc_mem"]

        return out


class MemoryProfilingInterpreter(Interpreter):

    def __init__(self, gm: GraphModule, debug_log=False):
        super().__init__(gm)
        self.nz3 = get_deepcompile_handle()
        self.device = torch.device(get_accelerator().current_device())
        self.mem_record = []
        self.last_alloc = get_accelerator().memory_allocated()

        self.node_counter = 0
        self.node_num = len(gm.graph.nodes)
        self.debug_log = debug_log

    def run(self, *args) -> Any:
        return_val = None
        try:
            assert _all_real_if_tensor(args), "Inputs must be real tensors"
            self.nz3.enable_profiling(True)
            self.mem_usage_out_of_torch = _get_mem_usage_out_of_torch()

            with unset_fake_temporarily():
                with get_accelerator().random().fork_rng(devices=[self.device]):
                    return_val = super().run(*args)
        except Exception as e:
            print(f"MemoryProfiling error {e}")
        finally:
            self.nz3.enable_profiling(False)

        return return_val

    def run_node(self, n: torch.fx.Node) -> Any:
        get_accelerator().reset_peak_memory_stats()

        if n.op in {"placeholder", "output"}:
            ret = super().run_node(n)
        else:
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            args = map_aggregate(args, lambda x: _to(x, self.device))
            kwargs = map_aggregate(kwargs, lambda x: _to(x, self.device))
            ret = getattr(self, n.op)(n.target, args, kwargs)

            del args, kwargs

        current_alloc = get_accelerator().memory_allocated() + self.mem_usage_out_of_torch
        max_alloc = get_accelerator().max_memory_allocated() + self.mem_usage_out_of_torch
        vals_to_bcast = torch.tensor([current_alloc, max_alloc], device=self.device, dtype=torch.int64)
        dist.all_reduce(vals_to_bcast, dist.ReduceOp.MAX)
        current_alloc = vals_to_bcast[0].item()
        max_alloc = vals_to_bcast[1].item()

        self.mem_record.append((n.name, current_alloc, current_alloc - self.last_alloc, max_alloc))

        self.node_counter += 1
        if self.debug_log and dist.get_rank() == 0:
            print(
                f"Mem prof Node {self.node_counter}/{self.node_num} {n.name} memory {current_alloc / 1024 / 1024:.2f}MB delta {(current_alloc - self.last_alloc) / 1024 / 1024:.2f}MB"
            )

        self.last_alloc = current_alloc

        return ret

    def dump(self, path):
        import pandas as pd
        df = pd.DataFrame(self.mem_record, columns=["node", "memory", "delta", "max_mem"])
        df.to_csv(path, index=False)
