# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import gc
from typing import List, Dict, Tuple
import _operator

import torch
from torch.fx import Graph, Node, GraphModule

from ..util import get_input_nodes, get_param_nodes, get_index_by_graph_id, get_deepcompile_handle, get_real_uses, is_cast_op
from ..fx import add_postprocess, _make_node_meta, get_output_node, move_primals_to_head
from ..profilers.graph_profile import ProfilingInterpreter
from ..list_schedule import fast_free_schedule

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

NAME = "zero3_compile"


def add_allgather(graph_id: int, graph: Graph, node: Node, ds_id: int, dtype: torch.dtype):
    new_ag_node = add_postprocess(graph,
                                  node,
                                  torch.ops.dc.allgather_param.default,
                                  extra_args=[graph_id, ds_id],
                                  extra_kwargs={"dtype": dtype},
                                  name=f"allgather_ds_param_{node.target}_{ds_id}",
                                  meta=_make_node_meta(node, ds_id, True))
    new_ag_node.meta["val"] = node.meta["val"].to(dtype)

    # Set the previous node back to output
    # We don't want to change the output node to allgather
    output_node = get_output_node(graph)
    output_node.replace_input_with(new_ag_node, node)

    # Add wait as well
    new_wait_node = add_postprocess(graph,
                                    new_ag_node,
                                    torch.ops.dc.wait_allgather.default,
                                    extra_args=[graph_id, ds_id],
                                    name=f"wait_allgather_ds_param__{node.target}_{ds_id}",
                                    meta=_make_node_meta(node, ds_id, False))
    new_wait_node.meta["val"] = new_ag_node.meta["val"]

    return new_ag_node


def add_release(graph_id: int, graph: Graph, node: Node, release_node: Node, ds_id: int, n_users: int):
    new_node = add_postprocess(graph,
                               node,
                               torch.ops.dc.release_param.default,
                               extra_args=[graph_id, ds_id, n_users],
                               name=f"release_ds_param_{release_node.target}_{node.name}_{ds_id}",
                               meta=_make_node_meta(node, ds_id, False))
    new_node.meta["val"] = None


def add_reduce(graph_id: int, graph: Graph, grad_node: Node, param_name: str, ds_id: int):
    new_node = add_postprocess(graph,
                               grad_node,
                               torch.ops.dc.reduce_grad.default,
                               extra_args=[graph_id, ds_id],
                               name=f"reduce_ds_param_{param_name}",
                               meta=_make_node_meta(grad_node, ds_id, True))
    new_node.meta["val"] = None


def add_gather_and_release(graph_id: int, graph: Graph, param_manager, param_nodes: List[Node]) -> Graph:

    node_to_uses = get_real_uses(graph)
    for pn in param_nodes:
        if len(pn.users) == 0:
            continue

        # If the only use of the parameter is a type-cast to a smaller type, fuse it with all-gather.
        fuse_typecast = False
        target_dtype = param_manager.params[pn.name].dtype
        if len([user for user in pn.users if user.op != "output"]) == 1:
            typecast_node = next(iter(pn.users))

            is_cast, casted_dtype = is_cast_op(typecast_node)
            if is_cast and casted_dtype.itemsize < target_dtype.itemsize:
                fuse_typecast = True
                target_dtype = casted_dtype

        add_allgather(graph_id, graph, pn, param_manager.ds_ids[pn.name], target_dtype)
        if fuse_typecast:
            users = node_to_uses[typecast_node]
            wait_node = typecast_node.args[0]
            for user in list(typecast_node.users.keys()):
                if user.op == "output":
                    wait_node.meta["original_output_name"] = typecast_node.name
                user.replace_input_with(typecast_node, wait_node)
            graph.erase_node(typecast_node)
        else:
            users = node_to_uses[pn]

        ds_id = param_manager.ds_ids[pn.name]
        for user in users:
            # release_param() only accepts tensors as its first argument. If
            # `user` is a tuple, we should release the param after any of
            # operator.getitem of that tuple.
            #
            # Since no torch op takes a tuple as an input, we simply walk
            # through users of `user` and check if there is any call to
            # operator.getitem.
            for secondary_user in user.users:
                if secondary_user.op == "call_function" and secondary_user.target == _operator.getitem:
                    add_release(graph_id, graph, secondary_user, pn, ds_id, len(users))
                    break
            else:
                add_release(graph_id, graph, user, pn, ds_id, len(users))

    return move_primals_to_head(graph)


def add_gather_and_reduce(graph_id: int, graph: Graph, param_manager, param_nodes_bw: List[Node],
                          param_name_to_grad: Dict[str, Node]) -> Graph:

    add_gather_and_release(graph_id, graph, param_manager, param_nodes_bw)

    for param_name in param_manager.param_names:
        if param_name_to_grad[param_name] is None:
            continue
        add_reduce(graph_id, graph, param_name_to_grad[param_name], param_name, param_manager.ds_ids[param_name])

    return move_primals_to_head(graph)


def add_z3_gather_release_fw(gm: GraphModule,
                             graph_id: int,
                             graph_order: List[Tuple[int, bool]],
                             profiling_results,
                             create_inputs_fn,
                             param_manager,
                             debug_log=False) -> GraphModule:

    nz3 = get_deepcompile_handle()

    real_inputs = create_inputs_fn()
    param_indices = profiling_results[graph_id].param_indices

    gm.graph = add_gather_and_release(graph_id, gm.graph, param_manager[graph_id],
                                      get_param_nodes(gm.graph, param_indices))

    nz3.register_graph_z3(graph_id, [v[1] for v in param_indices])  # Need this before profiling

    profiler = ProfilingInterpreter(gm, debug_log=debug_log)
    profiler.run(*real_inputs)
    del profiler
    gc.collect()
    get_accelerator().empty_cache()

    rank = dist.get_rank()
    graph_index = get_index_by_graph_id(graph_order, graph_id)
    if rank == 0 and debug_log:
        print(f"Fwd before scheduling graph {graph_index} graph_id={graph_id} {gm.graph}")

    for n in gm.graph.nodes:
        is_ds_param = n.name in param_manager[graph_id].ds_ids
        if "val" in n.meta and is_ds_param:
            # Used for Inductor's validation
            n.meta["val"] = torch.empty([0], dtype=n.meta['val'].dtype, device=n.meta['val'].device)

    gm.graph = fast_free_schedule(
        gm.graph,
        get_accelerator().available_memory(),
        0,  # unused
        debug_log=debug_log)

    if rank == 0 and debug_log:
        print(f"Fwd after scheduling graph {graph_index} graph_id={graph_id} {gm.graph}")

    return gm


def add_z3_gather_release_bw(gm: GraphModule,
                             graph_id: int,
                             graph_order: List[Tuple[int, bool]],
                             profiling_results,
                             create_inputs_fn,
                             param_manager,
                             debug_log=False) -> GraphModule:

    param_nodes_bw, param_name_to_grad = param_manager[graph_id].get_bwd_mapping(gm.graph)
    gm.graph = add_gather_and_reduce(graph_id, gm.graph, param_manager[graph_id], param_nodes_bw, param_name_to_grad)

    input_nodes = get_input_nodes(gm.graph)
    real_inputs = create_inputs_fn()
    assert len(input_nodes) == len(real_inputs), f"Expected {len(real_inputs)} inputs, got {len(input_nodes)}"

    real_outputs = ProfilingInterpreter(gm, debug_log=debug_log).run(*real_inputs)

    del real_outputs
    gc.collect()
    get_accelerator().empty_cache()

    rank = dist.get_rank()
    graph_index = get_index_by_graph_id(graph_order, graph_id)
    if rank == 0 and debug_log:
        print(f"Bwd before scheduling graph {graph_index} graph_id={graph_id} {gm.graph}")

    gm.graph = fast_free_schedule(
        gm.graph,
        get_accelerator().available_memory(),
        0,  # unused
        debug_log=debug_log)

    with gm.graph.inserting_before(get_output_node(gm.graph)):
        gm.graph.create_node("call_function", torch.ops.dc.end_backward.default, (graph_id, ))

    return gm


def add_z3_gather_release(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                          create_inputs_fn, mem_budget: float, param_manager, bwd: bool) -> GraphModule:
    if bwd:
        return add_z3_gather_release_bw(gm,
                                        graph_id,
                                        graph_order,
                                        profiling_results,
                                        create_inputs_fn,
                                        param_manager,
                                        debug_log=False)
    return add_z3_gather_release_fw(gm,
                                    graph_id,
                                    graph_order,
                                    profiling_results,
                                    create_inputs_fn,
                                    param_manager,
                                    debug_log=False)
