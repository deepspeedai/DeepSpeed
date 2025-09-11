# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
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

from ..profilers import ProfilingResult
from ..graph_param import DSGraphParamManager
from ..fx import move_primals_to_head

import deepspeed.comm as dist

NAME = "offload_adam_states"


def print_r0(msg):
    if dist.get_rank() == 0:
        print(msg)


MARGIN = 0.2

copy_stream = None
offload_event = None
reload_event = None

offload_key_events = {}
reload_key_events = {}

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
    offload_buf_key = _make_offload_state_key(key)
    if offload_buf_key not in state:
        state[offload_buf_key] = get_accelerator().pin_memory(torch.empty_like(state[key], device="cpu"))

    if key not in state:
        return

    with get_accelerator().stream(copy_stream):
        state[offload_buf_key].copy_(state[key], non_blocking=True)

    if key_event is None:
        offload_event.record(stream=copy_stream)
    else:
        key_event.record(stream=copy_stream)


def move_back_key(state, key, key_event=None):

    with get_accelerator().stream(copy_stream):
        state[key] = torch.empty_like(state[_make_offload_state_key(key)], device=device)
        state[key].copy_(state[_make_offload_state_key(key)], non_blocking=True)

    if key_event is None:
        reload_event.record(stream=copy_stream)
    else:
        key_event.record(stream=copy_stream)


def move_hp_param(src_tensor, dest_buf, key_event=None):
    with get_accelerator().stream(copy_stream):
        dest_buf.copy_(src_tensor, non_blocking=True)
        src_tensor.data = dest_buf

    if key_event is None:
        reload_event.record(stream=copy_stream)
    else:
        key_event.record(stream=copy_stream)


def move_back_hp_param(src_tensor, dest_buf, key_event=None):
    with get_accelerator().stream(copy_stream):
        dest_buf.data = torch.empty_like(src_tensor, device=device)
        dest_buf.copy_(src_tensor, non_blocking=True)

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
        # print_r0("Reloading Adam states")

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


def make_offload_task(task):

    def run_offload_task():
        # if not nz3.is_profiling():
        # print_r0(f"run_offload_task {task[0]} {task[2]} {task[3]} {task[4]}")

        if offload_key_events.get(task[1]) is None:
            offload_key_events[task[1]] = get_accelerator().Event()

        if task[2] == "hp_param":
            move_hp_param(task[1][0], task[1][1], offload_key_events[task[1][0]])
        else:
            assert task[1] in optimizer.state, f"State {task[1]} not found in optimizer"
            state = optimizer.state[task[1]]
            # if offload_key_events.get(task[1]) is None:
            #     offload_key_events[task[1]] = get_accelerator().Event()
            move_key(state, task[2], offload_key_events[task[1]])

    return run_offload_task


def make_offload_sync(task):

    def run_offload_sync():
        # if not nz3.is_profiling():
        event = offload_key_events[task[1]]
        event.synchronize()

        if task[2] != "hp_param":
            state = optimizer.state[task[1]]
            key = task[2]
            if key in state:
                del state[key]
        # print_r0(f"run_offload_sync {task[0]} {task[2]} alloc_mem={get_accelerator().memory_allocated()}")

    return run_offload_sync


def make_reload_task(task):

    def run_reload_task():
        if not nz3.is_profiling():
            if reload_key_events.get(task[1]) is None:
                reload_key_events[task[1]] = get_accelerator().Event()

            if task[2] == "hp_param":
                move_back_hp_param(task[1][1], task[1][0], reload_key_events[task[1]])
            else:
                state = optimizer.state[task[1]]
                # print_r0(f"run_reload_task {task[0]} {task[2]} {task[3]} {task[4]}")
                move_back_key(state, task[2], reload_key_events[task[1]])

    return run_reload_task


def update_max_memory(name):

    global max_memory
    mem = get_accelerator().max_memory_allocated()
    max_memory = max(max_memory, mem)


def empty_cache():
    get_accelerator().empty_cache()


offload_tasks = []
offload_tasks_remaining = []
offload_tasks_scheduled = []
reload_task_remaining = []
total_reload_mem = 0


def offload_opt_states_inc(graph: Graph, graph_id: int, graph_order: List[Tuple[int, bool]],
                           profiling_results: ProfilingResult, mem_budget: float, param_manager: DSGraphParamManager,
                           bwd: bool) -> Graph:

    to_remove = []
    for node in graph.nodes:
        if node.op == 'call_function' and \
            node.target in [offload_adam_states_sync, sync_offload_states, reload_adam_states_sync, sync_reload_states, update_max_memory]:
            to_remove.append(node)

    for node in to_remove:
        graph.erase_node(node)

    accelerator = get_accelerator()
    total_mem = accelerator.total_memory() * (1 - MARGIN)
    print_r0(f"offload_opt_states_inc start graph {graph_id} bwd={bwd} max_memory={max_memory} total_mem={total_mem}")

    mem = profiling_results[graph_id].bwd_mem if bwd else profiling_results[graph_id].fwd_mem
    mem_dict = {name: peak for name, alloc_mem, delta, peak in mem}

    current_peak_mem = 0
    peak_mem = {}

    ordered_node = reversed(graph.nodes) if bwd else graph.nodes
    for node in ordered_node:
        # print(f"Node: {node.name} mem: {mem_dict[node.name]}")
        if mem_dict[node.name] > current_peak_mem:
            current_peak_mem = mem_dict[node.name]
        peak_mem[node.name] = current_peak_mem

    # fwd_max_mem = max(m[3] for m in prof.fwd_mem)
    # bwd_max_mem = max(m[3] for m in prof.bwd_mem) if len(prof.bwd_mem) > 0 else 0
    # peak_mem = max(peak_mem, fwd_max_mem, bwd_max_mem)

    global offload_tasks_remaining, reload_tasks_remaining, offload_tasks_scheduled

    if not bwd:
        is_first_graph = graph_id == graph_order[0][0]
        # print_r0(
        #     f"offload_opt_states_inc start graph {graph_id} graph_order {graph_order} fwd is_first_graph {is_first_graph}"
        # )

        # At the beginning of the first graph, we schedule offload tasks to launch all offloading
        if is_first_graph:
            # print_r0(
            #     f"offload_opt_states_inc fwd before reload graph {graph_id} allocated_mem={get_accelerator().memory_allocated()}"
            # )

            with unset_fake_temporarily():
                offload_adam_states_sync()
                reload_adam_states_sync()
                sync_reload_states()

            reload_size = 0

            for i, ((k, state), hp_param, hp_param_cpu) in enumerate(
                    zip(optimizer.state.items(), optimizer.fp32_partitioned_groups_flat,
                        optimizer.hp_params_pin_buffers)):
                # print_r0(
                # f"Checking key for offloading {i} {k.shape} has_key {_make_offload_state_key('exp_avg') in state}")

                if _make_offload_state_key("exp_avg") in state:
                    key = _make_offload_state_key("exp_avg")
                    size = state[key].numel() * state[key].element_size()

                    # if total_mem < max_memory + reload_size + size:
                    offload_tasks.append(
                        (i, k, "exp_avg", state[key].numel() * state[key].element_size(), state[key].dtype))
                    # print_r0(
                    #     f"Offloading task {i} exp_avg reload_size={reload_size} size={size} estimated_mem={max_memory + reload_size + size}"
                    # )

                if _make_offload_state_key("exp_avg_sq") in state:
                    key = _make_offload_state_key("exp_avg_sq")
                    size = state[key].numel() * state[key].element_size()

                    # if total_mem < max_memory + reload_size + size:
                    offload_tasks.append(
                        (i, k, "exp_avg_sq", state[key].numel() * state[key].element_size(), state[key].dtype))
                    # print_r0(
                    #     f"Offloading task {i} exp_avg_sq reload_size={reload_size} size={size} estimated_mem={max_memory + reload_size + size}"
                    # )

                hp_param_size = hp_param.numel() * hp_param.element_size()
                # if total_mem < max_memory + reload_size + hp_param_size:
                offload_tasks.append((i, (hp_param, hp_param_cpu), "hp_param",
                                      hp_param.numel() * hp_param.element_size(), hp_param.dtype))
                # print_r0(
                #     f"Offloading task {i} hp_param reload_size={reload_size} size={hp_param_size} estimated_mem={max_memory + reload_size + hp_param_size}"
                # )

        # print_r0(f"offload_opt_states_inc fwd graph {graph_id} allocated_mem={get_accelerator().memory_allocated()}")

        for node in graph.nodes:
            # print_r0(f"checking sync node insert node: {node.name}")

            if node.name not in peak_mem \
                    or node.op == 'placeholder' \
                    or "offload_opt_" in node.name:
                continue

            to_offload = []
            optim_size = sum([task[3] for task in offload_tasks])

            # print_r0(
            #     f" optim_size: {optim_size} total_mem: {total_mem} peak_mem: {peak_mem[node.name]} available: {total_mem - peak_mem[node.name] - optim_size} #tasks={len(offload_tasks)}"
            # )
            while total_mem - peak_mem[node.name] - optim_size < 0:
                if len(offload_tasks) == 0:
                    break

                task = offload_tasks.pop(0)
                to_offload.append(task)
                optim_size = sum([task[3] for task in offload_tasks])
                # print_r0(
                #     f" scheduled task {task[0]} {task[2]} {task[3]} optim_size: {optim_size} peak_mem: {peak_mem[node.name]} available: {total_mem - peak_mem[node.name] - optim_size} #tasks={len(offload_tasks)}"
                # )

            for task in to_offload:
                with graph.inserting_before(node):
                    graph.create_node('call_function',
                                      make_offload_sync(task), (), {},
                                      name=f"offload_opt_sync_{task[0]}_{task[2]}")
                print_r0(f"Inserting fwd offload_opt_sync_{task[0]}_{task[2]}")
                offload_tasks_scheduled.append(task)

        for node in graph.nodes:
            # print(f"Node: {node.name} mem: {mem_dict[node.name]}")
            if node.op != 'placeholder':
                print_r0(f"Inserting all offload tasks before {node.name}")
                for task in offload_tasks_scheduled:
                    name = f"offload_opt_{task[0]}_{task[2]}"
                    with graph.inserting_before(node):
                        offload_node = graph.create_node('call_function', make_offload_task(task), (), {}, name=name)
                break

        # print_r0(f"offload_opt_states_inc finish graph {graph_id} fwd graph {graph}")
        print_r0(f"offload_opt_states_inc finish graph {graph_id}")
    else:

        graph_order_with_backward = [g[0] for g in graph_order if g[1]]
        is_first_graph = graph_id == graph_order_with_backward[-1]
        is_last_graph = graph_id == graph_order_with_backward[0]

        # print_r0(
        #     f"offload_opt_states_inc bwd graph {graph_id} graph_order_with_backward {graph_order_with_backward} is_first_graph {is_first_graph} is_last_graph {is_last_graph}"
        # )

        if is_first_graph:
            inserted_sync = False
            for node in graph.nodes:
                if node.op != 'placeholder' and not inserted_sync:
                    # print(f"Inserting offload_sync before {node.name}")
                    with graph.inserting_before(node):
                        graph.create_node('call_function', empty_cache, (), {}, name="empty_cache")

                    inserted_sync = True
        reload_tasks_remaining = copy.copy(offload_tasks_scheduled)

        global total_reload_mem
        for node in graph.nodes:
            if node.name not in peak_mem \
                or node.op == 'placeholder' \
                or node.op == 'output' \
                or "offload_opt_sync_" in node.name:
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
                                                       make_reload_task(task), (), {},
                                                       name=f"reload_opt_{task[0]}_{task[2]}")

                    total_reload_mem += next_reload_mem
                    reload_tasks_remaining.pop(0)
                    if len(reload_tasks_remaining) == 0:
                        break

                    task = reload_tasks_remaining[0]
                    next_reload_mem = task[3]

            # prev_node = node

        if is_last_graph:
            for node in graph.nodes:
                # print(f"Node: {node.name} mem: {mem_dict[node.name]}")
                if node.op == 'output':
                    for task in reload_tasks_remaining:
                        with graph.inserting_before(node):
                            graph.create_node('call_function',
                                              make_reload_task(task), (), {},
                                              name=f"reload_opt_{task[0]}_{task[2]}")

                    sync_fn = lambda: copy_stream.synchronize()
                    with graph.inserting_before(node):
                        graph.create_node('call_function', sync_fn, (), {}, name="sync_offload_copy_stream")

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
            # print(f"Node: {node.name} mem: {mem_dict[node.name]}")
            if node.op == 'output' and not inserted_reload and is_last_graph:
                # print(f"Inserting reload_opt before {node.name}")
                with graph.inserting_before(node):
                    graph.create_node('call_function', reload_adam_states_sync, (), {}, name="reload_opt")
                inserted_reload = True

        # add_record_max_mem_nodes(graph)

    else:
        is_first_graph = graph_id == graph_order[0][0]

        graph = move_primals_to_head(graph)

        inserted_offload = False
        for node in graph.nodes:
            # print(f"Node: {node.name} mem: {mem_dict[node.name]}")
            if node.op != 'placeholder' and not inserted_offload and is_first_graph:
                print(f"Inserting offload_opt before {node.name}")
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

    global optimizer
    optimizer = adam_optimizer
    global device
    device = torch.device(get_accelerator().current_device())
    global nz3
    nz3 = _nz3
