# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
import os
from typing import List, Tuple

import torch
from torch.fx import Graph, GraphModule

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.offload_states import _make_offload_state_key

try:
    from torch._subclasses.fake_tensor import unset_fake_temporarily
except ImportError:
    # Unsupported torch version
    pass

try:
    from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
except ImportError:
    # Without the effects registry the ops survive inductor only via the DCE patch in inductor.py.
    _register_effectful_op = None

from ..profilers import ProfilingResult
from ..graph_param import DSGraphParamManager
from ..fx import move_primals_to_head
from .contract import CAP_OPT_STATES_EVICTED, PassContract

import deepspeed.comm as dist

NAME = "offload_adam_states"
NAME_SYNC = "offload_adam_states_sync"
NAME_FOR_INIT = "offload_adam_states_for_init"
# All three act on optimizer state that only init_z3 registers, and DeepSpeed supports a single
# offload target per run, so none of them may share a schedule with offload_parameters or with the
# ZeRO-1/2 reduce passes.
_INCOMPATIBLE = frozenset({"offload_parameters", "zero1_compile", "zero2_compile"})
# move_opt_states plans from the profiled per-node peaks, which describe the run only if the
# optimizer state was already off the accelerator when the profilers ran.
CONTRACT = PassContract(requires=frozenset({CAP_OPT_STATES_EVICTED}), conflicts_with=_INCOMPATIBLE)
# The synchronous variant needs no such profile: it offloads everything at the head of the first
# graph and reloads at the tail of the last, so it takes no requires. It replaces move_opt_states
# rather than running alongside it.
CONTRACT_SYNC = PassContract(conflicts_with=_INCOMPATIBLE | {NAME})
CONTRACT_FOR_INIT = PassContract(provides=frozenset({CAP_OPT_STATES_EVICTED}), conflicts_with=_INCOMPATIBLE)


def print_r0(msg):
    if dist.get_rank() == 0:
        print(msg)


MARGIN = 0.2

copy_stream = None
offload_event = None
reload_event = None

max_memory = 0


def lazy_init():
    global copy_stream
    global offload_event
    global reload_event

    if copy_stream is None:

        copy_stream = get_accelerator().Stream()
        offload_event = get_accelerator().Event()
        reload_event = get_accelerator().Event()


optimizer = None
device = None
nz3 = None


def move_key(state, key, key_event=None):
    # Already offloaded: return before touching state[key], which no longer exists.
    if key not in state:
        return
    offload_buf_key = _make_offload_state_key(key)
    if offload_buf_key not in state:
        state[offload_buf_key] = get_accelerator().pin_memory(torch.empty_like(state[key], device="cpu"))

    with get_accelerator().stream(copy_stream):
        state[offload_buf_key].copy_(state[key], non_blocking=True)
        # Callers free state[key] without waiting, so hold the block until the copy is done.
        if state[key].device.type != "cpu":
            state[key].record_stream(copy_stream)

    if key_event is None:
        offload_event.record(stream=copy_stream)
    else:
        key_event.record(stream=copy_stream)


def _alloc_reload_buffer(like_tensor, compute_stream):
    # Reuse the activation blocks backward just freed, but wait for the compute stream first:
    # the allocator reissues them while its kernels may still read, and a mid-backward reload
    # writing from copy_stream would overwrite a live activation (seen as NaN losses).
    buf = torch.empty_like(like_tensor, device=device)
    copy_stream.wait_stream(compute_stream)
    return buf


def move_back_key(state, key, key_event=None):
    # record_stream holds the buffer until the copy lands; later compute reads are already
    # ordered before the next offload by the launch op's wait_stream.
    buf = _alloc_reload_buffer(state[_make_offload_state_key(key)], get_accelerator().current_stream())
    with get_accelerator().stream(copy_stream):
        buf.copy_(state[_make_offload_state_key(key)], non_blocking=True)
    buf.record_stream(copy_stream)
    state[key] = buf

    if key_event is None:
        reload_event.record(stream=copy_stream)
    else:
        key_event.record(stream=copy_stream)


def move_hp_param(src_tensor, dest_buf, key_event=None):
    with get_accelerator().stream(copy_stream):
        dest_buf.copy_(src_tensor, non_blocking=True)
        # The .data rebind below drops the GPU storage the copy is still reading; hold it.
        # Already-offloaded tensors have no GPU storage and cannot take record_stream.
        if src_tensor.device.type != "cpu":
            src_tensor.record_stream(copy_stream)
        src_tensor.data = dest_buf

    if key_event is None:
        reload_event.record(stream=copy_stream)
    else:
        key_event.record(stream=copy_stream)


def move_back_hp_param(src_tensor, dest_buf, key_event=None):
    # Same allocation and ownership discipline as move_back_key.
    buf = _alloc_reload_buffer(src_tensor, get_accelerator().current_stream())
    with get_accelerator().stream(copy_stream):
        buf.copy_(src_tensor, non_blocking=True)
    buf.record_stream(copy_stream)
    dest_buf.data = buf

    if key_event is None:
        reload_event.record(stream=copy_stream)
    else:
        key_event.record(stream=copy_stream)


def offload_adam_states_sync():

    with unset_fake_temporarily():

        if not hasattr(optimizer, "hp_params_pin_buffers"):
            optimizer.hp_params_pin_buffers = [
                get_accelerator().pin_memory(torch.empty_like(t, device="cpu"))
                for t in optimizer.fp32_partitioned_groups_flat
            ]

        for i, (k, state) in enumerate(optimizer.state.items()):
            if "exp_avg" in state:
                move_key(state, "exp_avg")
            if "exp_avg_sq" in state:
                move_key(state, "exp_avg_sq")

        for _, state in optimizer.state.items():
            if "exp_avg" in state:
                del state["exp_avg"]
            if "exp_avg_sq" in state:
                del state["exp_avg_sq"]

        for src_tensor, dest_buf in zip(optimizer.fp32_partitioned_groups_flat, optimizer.hp_params_pin_buffers):
            move_hp_param(src_tensor, dest_buf)

        get_accelerator().synchronize()


def reload_adam_states_sync():

    with unset_fake_temporarily():

        for _, state in optimizer.state.items():
            if _make_offload_state_key("exp_avg") in state:
                move_back_key(state, "exp_avg")
            if _make_offload_state_key("exp_avg_sq") in state:
                move_back_key(state, "exp_avg_sq")

        for src, dest in zip(optimizer.hp_params_pin_buffers, optimizer.fp32_partitioned_groups_flat):
            move_back_hp_param(src, dest)

        get_accelerator().synchronize()


def sync_offload_states(event=None):
    if nz3.is_profiling():
        offload_adam_states_sync()
    else:
        if event is None:
            offload_event.wait(copy_stream)
        else:
            event.wait(copy_stream)


def sync_reload_states(event=None):
    if nz3.is_profiling():
        reload_adam_states_sync()
    else:
        if event is None:
            reload_event.wait(copy_stream)
        else:
            event.wait(copy_stream)


# This work used to be inserted as Python closures, which inductor cannot compile or cache
# (no importable qualified name, no schema, no Meta kernel). The dc.* ops below carry only an
# int index into this registry, so the task tuples holding live tensors never cross the op
# boundary; the anchor tensor exists only to give the dispatcher something to route on.
_op_task_registry = []
_offload_ops_lib = None

# Rank-local op execution counts. Reloads are skipped while profiling, so a nonzero reload
# count proves the ops ran in the compiled graph -- the only cheap detector for the silent
# failure where dead-code elimination drops them and training just keeps the states resident.
_offload_op_stats = {"launches": 0, "reloads": 0}


def get_offload_op_stats():
    return dict(_offload_op_stats)


def reset_offload_op_stats():
    for key in _offload_op_stats:
        _offload_op_stats[key] = 0


def _register_op_task(task) -> int:
    _op_task_registry.append(task)
    return len(_op_task_registry) - 1


def _offload_opt_launch_impl(anchor, idx):
    _offload_op_stats["launches"] += 1
    task = _op_task_registry[idx]
    # The optimizer step just wrote these states on the compute stream; order the reads after it.
    copy_stream.wait_stream(get_accelerator().current_stream())
    if task[2] == "hp_param":
        move_hp_param(task[1][0], task[1][1])
    else:
        assert task[1] in optimizer.state, f"State {task[1]} not found in optimizer"
        state = optimizer.state[task[1]]
        move_key(state, task[2])
        # Safe now: move_key's record_stream keeps the block alive until the copy completes.
        if task[2] in state:
            del state[task[2]]


def _reload_opt_impl(anchor, idx):
    if nz3.is_profiling():
        return

    _offload_op_stats["reloads"] += 1
    task = _op_task_registry[idx]
    if task[2] == "hp_param":
        move_back_hp_param(task[1][1], task[1][0])
    else:
        state = optimizer.state[task[1]]
        move_back_key(state, task[2])


# Re-armed each time the pass runs (i.e. once per compile phase) and cleared on first execution.
_empty_cache_pending = False


def _opt_empty_cache_impl(anchor):
    # Once per compile phase is enough to return the freed segments; per step costs +28%.
    global _empty_cache_pending
    if not _empty_cache_pending:
        return
    _empty_cache_pending = False
    get_accelerator().empty_cache()


def _reload_copy_stream_sync_impl(anchor):
    # Inserted at the end of backward: the optimizer step reads the reloaded states as soon as
    # the graph returns. Draining the whole stream also covers any offload still in flight.
    copy_stream.synchronize()


_OFFLOAD_OP_SPECS = [
    ("offload_opt_launch", "offload_opt_launch(Tensor anchor, int idx) -> ()", _offload_opt_launch_impl),
    ("reload_opt", "reload_opt(Tensor anchor, int idx) -> ()", _reload_opt_impl),
    ("opt_empty_cache", "opt_empty_cache(Tensor anchor) -> ()", _opt_empty_cache_impl),
    ("reload_copy_stream_sync", "reload_copy_stream_sync(Tensor anchor) -> ()", _reload_copy_stream_sync_impl),
]


def register_offload_ops():
    global _offload_ops_lib
    if _offload_ops_lib is None:
        # FRAGMENT, not DEF: the compiled extension creates the "dc" namespace with TORCH_LIBRARY,
        # and a namespace may only be created once per process. A FRAGMENT extends the namespace
        # without claiming it, and none of the names below are also defined by the extension, so
        # the two can be set up in either order.
        lib = torch.library.Library("dc", "FRAGMENT")
        for name, schema, impl in _OFFLOAD_OP_SPECS:
            lib.define(schema)
            lib.impl(name, impl, "CompositeExplicitAutograd")
            lib.impl(name, lambda *args: None, "Meta")

        # Nothing consumes these ops' output, so two dead-code eliminations would drop them: FX's,
        # guarded by _side_effectful_functions, and inductor's scheduler DCE, which keys off the
        # schema instead. The ORDERED effect covers the second and pins the ops in program order,
        # which reload-before-sync depends on.
        for name, _, _ in _OFFLOAD_OP_SPECS:
            overload = getattr(torch.ops.dc, name).default
            torch.fx.node._side_effectful_functions.add(overload)
            if _register_effectful_op is not None:
                _register_effectful_op(overload, _EffectType.ORDERED)

        # The ops deregister if the library object is garbage collected.
        _offload_ops_lib = lib


def _find_graph_anchor(graph: Graph):
    for node in graph.nodes:
        if node.op == 'placeholder' and isinstance(node.meta.get("val"), torch.Tensor):
            return node
    # A non-tensor anchor would violate the op schemas at runtime; fail loudly instead.
    raise AssertionError("no tensor placeholder found to anchor the offload ops on")


def update_max_memory(name):

    global max_memory
    mem = get_accelerator().max_memory_allocated()
    max_memory = max(max_memory, mem)


offload_tasks = []
offload_tasks_scheduled = []
# Entries of offload_tasks_scheduled that already have launch nodes (graph breaks run the
# pass once per forward graph, and each must only insert its own share).
offload_tasks_inserted = 0
reload_tasks_remaining = []
total_reload_mem = 0


def offload_opt_states_inc(graph: Graph, graph_id: int, graph_order: List[Tuple[int, bool]],
                           profiling_results: ProfilingResult, mem_budget: float, param_manager: DSGraphParamManager,
                           bwd: bool) -> Graph:
    global _empty_cache_pending, offload_tasks_inserted, reload_tasks_remaining, total_reload_mem

    to_remove = []
    for node in graph.nodes:
        if node.op == 'call_function' and \
            node.target in [offload_adam_states_sync, sync_offload_states, reload_adam_states_sync, sync_reload_states, update_max_memory]:
            to_remove.append(node)

    for node in to_remove:
        graph.erase_node(node)

    register_offload_ops()
    anchor = _find_graph_anchor(graph)

    accelerator = get_accelerator()
    budget_override = os.environ.get("DS_DC_OFFLOAD_OPT_BUDGET_GB")
    if budget_override is not None:
        # Test hook: pretend the device has this much memory, to force or suppress offloading.
        total_mem = float(budget_override) * 1e9
    else:
        total_mem = accelerator.total_memory() * (1 - MARGIN)
    print_r0(f"offload_opt_states_inc start graph {graph_id} bwd={bwd} max_memory={max_memory} total_mem={total_mem}")

    mem = profiling_results[graph_id].bwd_mem if bwd else profiling_results[graph_id].fwd_mem
    mem_dict = {name: peak for name, alloc_mem, delta, peak in mem}

    current_peak_mem = 0
    peak_mem = {}

    ordered_node = reversed(graph.nodes) if bwd else graph.nodes
    for node in ordered_node:
        # Nodes with no profiled entry inherit the running peak instead of raising.
        if node.name in mem_dict and mem_dict[node.name] > current_peak_mem:
            current_peak_mem = mem_dict[node.name]
        peak_mem[node.name] = current_peak_mem

    if not bwd:
        is_first_graph = graph_id == graph_order[0][0]

        # At the beginning of the first graph, we schedule offload tasks to launch all offloading
        if is_first_graph:
            # Module state survives compile phases; reset so re-running does not double-append.
            offload_tasks.clear()
            offload_tasks_scheduled.clear()
            offload_tasks_inserted = 0
            _op_task_registry.clear()
            total_reload_mem = 0

            with unset_fake_temporarily():
                offload_adam_states_sync()
                reload_adam_states_sync()
                sync_reload_states()

            for i, ((k, state), hp_param, hp_param_cpu) in enumerate(
                    zip(optimizer.state.items(), optimizer.fp32_partitioned_groups_flat,
                        optimizer.hp_params_pin_buffers)):

                if _make_offload_state_key("exp_avg") in state:
                    key = _make_offload_state_key("exp_avg")
                    offload_tasks.append(
                        (i, k, "exp_avg", state[key].numel() * state[key].element_size(), state[key].dtype))

                if _make_offload_state_key("exp_avg_sq") in state:
                    key = _make_offload_state_key("exp_avg_sq")
                    offload_tasks.append(
                        (i, k, "exp_avg_sq", state[key].numel() * state[key].element_size(), state[key].dtype))

                offload_tasks.append((i, (hp_param, hp_param_cpu), "hp_param",
                                      hp_param.numel() * hp_param.element_size(), hp_param.dtype))

        for node in graph.nodes:
            if node.name not in peak_mem \
                    or node.op == 'placeholder' \
                    or "offload_opt_" in node.name:
                continue

            to_offload = []
            optim_size = sum([task[3] for task in offload_tasks])

            # Peaks were profiled with the states already emptied, so residency adds on top.
            while total_mem - peak_mem[node.name] - optim_size < 0:
                if len(offload_tasks) == 0:
                    break

                task = offload_tasks.pop(0)
                to_offload.append(task)
                optim_size = sum([task[3] for task in offload_tasks])

            # No sync node needed: the launch op frees the state, gated by record_stream.
            for task in to_offload:
                print_r0(f"Scheduling offload of optimizer state {task[0]}_{task[2]}")
                offload_tasks_scheduled.append(task)

        # Only newly scheduled tasks get launch nodes; earlier graphs carry their own share.
        new_tasks = offload_tasks_scheduled[offload_tasks_inserted:]
        for node in graph.nodes:
            if node.op != 'placeholder':
                print_r0(f"Inserting {len(new_tasks)} offload tasks before {node.name}")
                for task in new_tasks:
                    name = f"offload_opt_{task[0]}_{task[2]}"
                    with graph.inserting_before(node):
                        graph.create_node('call_function',
                                          torch.ops.dc.offload_opt_launch.default, (anchor, _register_op_task(task)),
                                          {},
                                          name=name)
                break
        offload_tasks_inserted = len(offload_tasks_scheduled)

        print_r0(f"offload_opt_states_inc finish graph {graph_id}")
    else:

        graph_order_with_backward = [g[0] for g in graph_order if g[1]]
        is_first_graph = graph_id == graph_order_with_backward[-1]
        is_last_graph = graph_id == graph_order_with_backward[0]

        if is_first_graph:
            _empty_cache_pending = True
            inserted_sync = False
            for node in graph.nodes:
                if node.op != 'placeholder' and not inserted_sync:
                    with graph.inserting_before(node):
                        graph.create_node('call_function',
                                          torch.ops.dc.opt_empty_cache.default, (anchor, ), {},
                                          name="empty_cache")

                    inserted_sync = True
        if is_first_graph:
            # Reset once per backward, not per graph: later graphs continue from the remainder.
            reload_tasks_remaining = copy.copy(offload_tasks_scheduled)

        for node in graph.nodes:
            if node.name not in peak_mem \
                or node.op == 'placeholder' \
                or node.op == 'output':
                continue

            if len(reload_tasks_remaining) > 0:
                task = reload_tasks_remaining[0]
                next_reload_mem = task[3]

                insert_pos = node
                while total_mem > peak_mem[node.name] + total_reload_mem + next_reload_mem:
                    expected_mem = peak_mem[node.name] + total_reload_mem
                    print_r0(
                        f" Inserting reload_opt reload_opt_{task[0]}_{task[2]} after {insert_pos.name} next_inc={next_reload_mem} peak_mem[{node.name}]={peak_mem[node.name]} inc_total={total_reload_mem} expected_mem={expected_mem}"
                    )

                    with graph.inserting_after(insert_pos):
                        insert_pos = graph.create_node('call_function',
                                                       torch.ops.dc.reload_opt.default,
                                                       (anchor, _register_op_task(task)), {},
                                                       name=f"reload_opt_{task[0]}_{task[2]}")

                    total_reload_mem += next_reload_mem
                    reload_tasks_remaining.pop(0)
                    if len(reload_tasks_remaining) == 0:
                        break

                    task = reload_tasks_remaining[0]
                    next_reload_mem = task[3]

        if is_last_graph:
            for node in graph.nodes:
                if node.op == 'output':
                    for task in reload_tasks_remaining:
                        with graph.inserting_before(node):
                            graph.create_node('call_function',
                                              torch.ops.dc.reload_opt.default, (anchor, _register_op_task(task)), {},
                                              name=f"reload_opt_{task[0]}_{task[2]}")

                    with graph.inserting_before(node):
                        graph.create_node('call_function',
                                          torch.ops.dc.reload_copy_stream_sync.default, (anchor, ), {},
                                          name="reload_copy_stream_sync")

        print_r0(
            f"offload_opt_states_inc graph {graph_id} graph_order {graph_order} bwd is_first_graph {is_first_graph} is_last_graph {is_last_graph}"
        )

    return graph


def add_record_max_mem_nodes(graph: Graph):

    nodes = list(graph.nodes)
    for node in nodes:
        if node.op == "output" or node.op == "placeholder":
            continue

        with graph.inserting_after(node):
            name = f"update_max_memory_{node.name}"
            graph.create_node('call_function', update_max_memory, (name, ), {}, name=name)


def insert_offload_opt_states(graph: Graph, graph_id: int, graph_order: List[Tuple[int, bool]],
                              profiling_results: ProfilingResult, mem_budget: float,
                              param_manager: DSGraphParamManager, bwd: bool) -> Graph:

    if bwd:
        graph_order_with_backward = [g[0] for g in graph_order if g[1]]
        is_last_graph = graph_id == graph_order_with_backward[0]

        inserted_reload = False
        for node in graph.nodes:
            if node.op == 'output' and not inserted_reload and is_last_graph:
                with graph.inserting_before(node):
                    graph.create_node('call_function', reload_adam_states_sync, (), {}, name="reload_opt")
                inserted_reload = True

    else:
        is_first_graph = graph_id == graph_order[0][0]

        graph = move_primals_to_head(graph)

        inserted_offload = False
        for node in graph.nodes:
            if node.op != 'placeholder' and not inserted_offload and is_first_graph:
                print_r0(f"Inserting offload_opt before {node.name}")
                with graph.inserting_before(node):
                    graph.create_node('call_function', offload_adam_states_sync, (), {}, name="offload_opt")
                inserted_offload = True

    add_record_max_mem_nodes(graph)

    return graph


def move_opt_states(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                    create_inputs_fn, mem_budget: float, param_manager: DSGraphParamManager, bwd: bool) -> GraphModule:
    gm.graph = offload_opt_states_inc(gm.graph, graph_id, graph_order, profiling_results, mem_budget, param_manager,
                                      bwd)
    return gm


def move_opt_states_sync(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                         create_inputs_fn, mem_budget: float, param_manager: DSGraphParamManager,
                         bwd: bool) -> GraphModule:
    gm.graph = insert_offload_opt_states(gm.graph, graph_id, graph_order, profiling_results, mem_budget, param_manager,
                                         bwd)
    return gm


def offload_adam_states_for_init(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]],
                                 profiling_results, create_inputs_fn, mem_budget: float,
                                 param_manager: DSGraphParamManager, bwd: bool) -> GraphModule:
    if not bwd and graph_id == graph_order[0][0]:
        with unset_fake_temporarily():
            offload_adam_states_sync()
    # returns None, and profiling will be skipped


def init_offload_opt_states(adam_optimizer, _nz3):
    lazy_init()
    register_offload_ops()

    global optimizer
    optimizer = adam_optimizer
    global device
    device = torch.device(get_accelerator().current_device())
    global nz3
    nz3 = _nz3
