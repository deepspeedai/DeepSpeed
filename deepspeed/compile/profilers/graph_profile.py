# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
from typing import Any, Tuple, Dict
import statistics

import torch
from torch.fx import Graph, GraphModule, Interpreter
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
        # A pinned host tensor inside a graph is an offloaded value: the op reading it expects it on
        # the host, and copying it to the device would both undo the offload and hide from the
        # profile the memory the offload just gave back.
        if v.device.type == "cpu" and v.is_pinned():
            return v
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


_PROFILE_META_DEFAULTS = {
    "device_time": 0.0,
    "wall_time": 0.0,
    "tensor_size": 0,
    "alloc_mem": 0,
    "max_mem": 0,
}
_PROFILE_INCOMPLETE_ATTR = "_deepcompile_profile_incomplete"
_PROFILE_INCOMPLETE_META_KEY = "deepcompile_profile_incomplete"


def _mark_profile_incomplete(graph: Graph):
    setattr(graph, _PROFILE_INCOMPLETE_ATTR, True)
    for node in graph.nodes:
        node.meta[_PROFILE_INCOMPLETE_META_KEY] = True


def is_profile_incomplete(graph: Graph):
    if graph is None:
        return False
    if getattr(graph, _PROFILE_INCOMPLETE_ATTR, False):
        return True
    return any(node.meta.get(_PROFILE_INCOMPLETE_META_KEY, False) for node in graph.nodes)


def _has_missing_profile_metadata(graph: Graph):
    return any(key not in node.meta for node in graph.nodes for key in _PROFILE_META_DEFAULTS)


def _backfill_missing_profile_metadata(graph: Graph, profile_complete: bool = True):
    if not profile_complete or _has_missing_profile_metadata(graph):
        _mark_profile_incomplete(graph)
    for node in graph.nodes:
        for key, default in _PROFILE_META_DEFAULTS.items():
            node.meta.setdefault(key, default)


class ProfileAborted(RuntimeError):
    """Raised on every rank once profiling has failed on any of them."""


def _abort_if_any_rank_failed(error, device, distributed=True):
    """Agree across ranks on whether profiling can go on, and stop everywhere if it cannot.

    Profiling issues collectives node by node. A rank that hits an error and simply stops taking
    part leaves every other rank waiting on a collective that will never be matched, which surfaces
    half an hour later as an NCCL watchdog timeout and a killed job rather than as the error that
    started it. Ranks therefore vote, and all of them raise together into the caller's handler,
    which already knows how to carry on with an incomplete profile.

    This covers a rank failing while it runs a node, which is where profiling runs out of memory.
    It cannot cover a rank failing inside a collective, or while gathering a node's arguments,
    because by then the other ranks are already waiting in a call this one will never make.
    """
    if not distributed or not dist.is_initialized():
        if error is not None:
            raise ProfileAborted(str(error)) from error
        return

    with unset_fake_temporarily():
        failed_flag = torch.tensor([1 if error is not None else 0], device=device, dtype=torch.int)
        dist.all_reduce(failed_flag, dist.ReduceOp.MAX)

    if failed_flag.item() == 0:
        return
    if error is not None:
        raise ProfileAborted(str(error)) from error
    raise ProfileAborted("profiling failed on another rank")


def _run_warmup_for_profile(call_fn, warmup):
    for _ in range(warmup):
        warmup_out = call_fn()
        del warmup_out


def _run_repeatedly_for_profile(call_fn, iteration, start_events, end_events):
    out = None
    for i in range(iteration):
        start_events[i].record()
        out = call_fn()
        end_events[i].record()
        if i + 1 < iteration:
            del out
            out = None

    return out


def _get_mem_usage_out_of_torch():

    adjust = 0
    try:
        import pynvml
        pynvml.nvmlInit()

        current_dev_id = get_accelerator().current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(current_dev_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # Reserved, not allocated. This number is meant to be the memory *other* things hold, and
        # it is added to every node's reading. Subtracting only what torch has handed out counts
        # the allocator's own cache as somebody else's usage: with expandable_segments the cache is
        # most of the card while live tensors are a fraction of it, which inflated a measured
        # 125GiB peak to a reported 239GiB and left the planner reading it as hopeless.
        torch_held = get_accelerator().memory_reserved()
        adjust = max(0, info.used - torch_held)
    except Exception:
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
        profile_complete = True
        try:
            assert _all_real_if_tensor(args), "Inputs must be real tensors"
            self.nz3.enable_profiling(True)

            with unset_fake_temporarily():
                with get_accelerator().random().fork_rng(devices=[self.device]):
                    self.mem_usage_out_of_torch = _get_mem_usage_out_of_torch()
                    return_val = super().run(*args)
        except Exception as e:
            profile_complete = False
            msg = e.msg if "msg" in dir(e) else str(e)
            if not self.distributed or dist.get_rank() == 0:
                print(f"DeepCompile profiling failed; using default profile metadata for incomplete nodes: {msg}")
        finally:
            try:
                self.nz3.clear_all_gathered_params()
            finally:
                try:
                    self.nz3.enable_profiling(False)
                finally:
                    _backfill_missing_profile_metadata(self.graph, profile_complete=profile_complete)
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

        # Running the node is where profiling runs out of memory in practice, and it sits between
        # two collectives every rank must reach. Record a failure and keep going to the vote at the
        # end of the node rather than unwinding past those collectives and stranding the others.
        error = None
        out = None
        alloc_mem = 0
        max_memory = 0
        tensor_size = 0
        try:
            is_release_op = is_release_node(n)
            run_only_once = cache_hit or is_release_op
            iteration = 1 if run_only_once else self.iteration
            accelerator = get_accelerator()
            start_events = [accelerator.Event(enable_timing=True) for _ in range(iteration)]
            end_events = [accelerator.Event(enable_timing=True) for _ in range(iteration)]

            get_accelerator().reset_peak_memory_stats()
            alloc_mem_start = get_accelerator().memory_allocated()
            max_mem_start = get_accelerator().max_memory_allocated()

            def run_target():
                return getattr(self, n.op)(n.target, args, kwargs)

            warmup = 0 if run_only_once else self.warmup
            _run_warmup_for_profile(run_target, warmup)

            if is_comm_op(n):
                assert self.distributed, f"Distributed environment is not initialized but comm operator {n.name} {n.target} is used."
                dist.barrier()

            start = time.time()
            out = _run_repeatedly_for_profile(run_target, iteration, start_events, end_events)
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
        except Exception as e:
            error = e

        if not cache_hit:
            if error is None:
                device_time = statistics.mean([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
                wall_time = walltime_sum / iteration * 1000
            else:
                # Nothing worth averaging in, but this rank still has to take part in the reduce.
                device_time = 0.0
                wall_time = 0.0

            with unset_fake_temporarily():
                vals_to_bcast = torch.tensor([device_time, wall_time, alloc_mem, max_memory, tensor_size],
                                             device=self.device)
                if self.distributed:
                    dist.all_reduce(vals_to_bcast, dist.ReduceOp.AVG)
                if error is None:
                    n.meta["device_time"] = vals_to_bcast[0].item()
                    n.meta["wall_time"] = vals_to_bcast[1].item()
                    n.meta["alloc_mem"] = int(vals_to_bcast[2].item())
                    n.meta["max_mem"] = int(vals_to_bcast[3].item())
                    n.meta["tensor_size"] = int(vals_to_bcast[4].item())
                    self.cache[cache_key] = (n.meta["device_time"], n.meta["wall_time"], n.meta["alloc_mem"],
                                             n.meta["max_mem"], n.meta["tensor_size"])

        _abort_if_any_rank_failed(error, self.device, self.distributed)

        if not cache_hit:
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
        self.profile_complete = True

        self.node_counter = 0
        self.node_num = len(gm.graph.nodes)
        self.debug_log = debug_log

    def run(self, *args) -> Any:
        return_val = None
        self.profile_complete = True
        try:
            assert _all_real_if_tensor(args), "Inputs must be real tensors"
            self.nz3.enable_profiling(True)
            self.mem_usage_out_of_torch = _get_mem_usage_out_of_torch()

            with unset_fake_temporarily():
                with get_accelerator().random().fork_rng(devices=[self.device]):
                    return_val = super().run(*args)
        except Exception as e:
            self.profile_complete = False
            self.mem_record.clear()
            print(f"MemoryProfiling error {e}")
        finally:
            try:
                self.nz3.clear_all_gathered_params()
            finally:
                self.nz3.enable_profiling(False)

        return return_val

    def run_node(self, n: torch.fx.Node) -> Any:
        get_accelerator().reset_peak_memory_stats()

        ret = None
        error = None
        try:
            if n.op in {"placeholder", "output"}:
                ret = super().run_node(n)
            else:
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                args = map_aggregate(args, lambda x: _to(x, self.device))
                kwargs = map_aggregate(kwargs, lambda x: _to(x, self.device))
                ret = getattr(self, n.op)(n.target, args, kwargs)

                del args, kwargs
        except Exception as e:
            # Leaving the node loop here would skip the collectives below, which every rank has to
            # reach. Carry the error to the vote instead; running out of memory is the usual reason
            # to land here, and reading the allocator counters still works afterwards.
            error = e

        current_alloc = get_accelerator().memory_allocated() + self.mem_usage_out_of_torch
        max_alloc = get_accelerator().max_memory_allocated() + self.mem_usage_out_of_torch
        vals_to_bcast = torch.tensor([current_alloc, max_alloc], device=self.device, dtype=torch.int64)
        dist.all_reduce(vals_to_bcast, dist.ReduceOp.MAX)
        current_alloc = vals_to_bcast[0].item()
        max_alloc = vals_to_bcast[1].item()

        _abort_if_any_rank_failed(error, self.device)

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
