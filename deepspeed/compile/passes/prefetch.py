# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Tuple

import torch
from torch.fx import Graph, Node, GraphModule

from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist

from ..profilers.comm_profile import create_predictor
from ..graph_param import DSGraphParamManager

NAME = "prefetch"

FUSE_FACTOR = 0.8
MARGIN = 0.1
MAX_FUSE_SIZE = 1e9
MAX_BUFFERED_SIZE = 4e9

run_prefetch_pass = False


def print_rank_0(message):
    if dist.get_rank() == 0:
        print(message)


def get_ds_id(node: Node):
    assert node.target == torch.ops.dc.allgather_param.default
    return node.args[2]


def schedule_prefetch(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                      create_inputs_fn, mem_budget: float, param_manager: DSGraphParamManager,
                      bwd: bool) -> GraphModule:

    max_mem = get_accelerator().total_memory() * (1 - MARGIN)
    vals_to_bcast = torch.tensor([max_mem], device=torch.device(get_accelerator().current_device()))
    dist.all_reduce(vals_to_bcast, dist.ReduceOp.MIN)
    max_mem = vals_to_bcast[0].item()

    mem = profiling_results[graph_id].bwd_mem if bwd else profiling_results[graph_id].fwd_mem
    op_time = profiling_results[graph_id].bwd_time if bwd else profiling_results[graph_id].fwd_time
    tensor_sizes = profiling_results[graph_id].bwd_tensor_sizes if bwd else profiling_results[graph_id].fwd_tensor_sizes

    mem_dict = {name: (alloc_mem, peak) for name, alloc_mem, delta, peak in mem}
    time_dict = {name: (device_time, wall_time) for name, device_time, wall_time in op_time}
    tensor_size_dict = {name: size for name, size in tensor_sizes}

    graph = gm.graph
    total_param_size = sum(
        [tensor_size_dict[n.name] for n in graph.nodes if n.target == torch.ops.dc.allgather_param.default])

    print_rank_0(
        f"schedule_prefetch graph_id={graph_id} max_mem={max_mem} available_memory={get_accelerator().available_memory()} memory_allocated={get_accelerator().memory_allocated()} max_allocated={get_accelerator().max_memory_allocated()} total_param_size={total_param_size} margin={MARGIN}"
    )

    # Fill missing values
    prev_mem = 0
    prev_peak = 0
    for node in graph.nodes:
        if node.name in mem_dict:
            prev_mem = mem_dict[node.name][0]
            prev_peak = mem_dict[node.name][1]
        else:
            print_rank_0(f"node {node.name} not in mem_dict")
            mem_dict[node.name] = (prev_mem, prev_peak)

    comm_predictor = create_predictor()

    order_rev = list(reversed(graph.nodes))
    new_order_rev = []
    prefetch_ags = []
    prefetch_ag_groups = []
    ag_tensor_size_sum = 0
    for i, node in enumerate(order_rev):
        # print_rank_0(
        #     f"Checking node reverse order {node.name} {node.target} ag_tensor_size_sum={ag_tensor_size_sum} max_mem={max_mem}"
        # )

        if node.op != "placeholder":
            assert i < len(order_rev) - 1
            assert node.name in mem_dict
            next_node = order_rev[i + 1]
            next_alloc_mem, next_peak = mem_dict[next_node.name]

            # Free up memory
            while next_peak + ag_tensor_size_sum > max_mem or ag_tensor_size_sum > MAX_BUFFERED_SIZE:
                if len(prefetch_ag_groups) > 0:
                    # launch prefetch
                    fused_ag_nodes = prefetch_ag_groups.pop(0)
                    total_ag_tensor_size = sum([tensor_size_dict[ag_node.name] for ag_node in fused_ag_nodes])
                    ag_tensor_size_sum -= total_ag_tensor_size
                    new_order_rev.append(fused_ag_nodes)
                    assert len(fused_ag_nodes) > 0
                    # print_rank_0(
                    #     f"Free up memory fused_ag_nodes={fused_ag_nodes} next_alloc_mem={next_alloc_mem} total_ag_tensor_size={total_ag_tensor_size} ag_tensor_size_sum={ag_tensor_size_sum} max_mem={max_mem}"
                    # )
                elif len(prefetch_ags) > 0:
                    prefetch_ag_groups.append(prefetch_ags)
                    prefetch_ags = []
                    # print_rank_0(
                    #     f"Free up memory prefetch_ags={prefetch_ag_groups} next_alloc_mem={next_alloc_mem} ag_tensor_size_sum={ag_tensor_size_sum} max_mem={max_mem}"
                    # )
                else:
                    break

            if node.target == torch.ops.dc.allgather_param.default:

                current_ag_size = sum([tensor_size_dict[ag_node.name] for ag_node in prefetch_ags])
                pred_time_current = comm_predictor(current_ag_size)
                pred_time_next = comm_predictor(tensor_size_dict[node.name])
                pred_time_fused = comm_predictor(current_ag_size + tensor_size_dict[node.name])

                do_fuse = max(pred_time_current, pred_time_next) * 1.2 > pred_time_fused and (
                    current_ag_size + tensor_size_dict[node.name]) < MAX_FUSE_SIZE
                # print_rank_0(
                #     f"found allgather_param do_fuse={do_fuse} current_ag_size={current_ag_size} tensor_size_dict[node.name]={tensor_size_dict[node.name]} pred_time_current={pred_time_current} pred_time_next={pred_time_next} pred_time_fused={pred_time_fused}"
                # )

                if len(prefetch_ags) > 0 and not do_fuse:
                    # stop fusing here
                    prefetch_ag_groups.append(prefetch_ags)
                    prefetch_ags = []
                #     print_rank_0(
                #         f"stop fusing prefetch_ags={prefetch_ag_groups} ag_tensor_size_sum={ag_tensor_size_sum}")
                # else:
                #     print_rank_0(
                #         f"continue fusing ag_tensor_size_sum={ag_tensor_size_sum} ag_size={tensor_size_dict[node.name]} prefetch_ags={prefetch_ags} prefetch_ag_groups={prefetch_ag_groups}"
                #     )
                prefetch_ags.append(node)
                ag_tensor_size_sum += tensor_size_dict[node.name]

        new_order_rev.append(node)

        if (node.op != "placeholder"
                and node.target != torch.ops.dc.reload_parameter) and order_rev[i + 1].op == "placeholder":
            for ag_group in prefetch_ag_groups:
                assert len(ag_group) > 0
                new_order_rev.append(ag_group)
                total_ag_tensor_size = sum([tensor_size_dict[ag_node.name] for ag_node in ag_group])
                ag_tensor_size_sum -= total_ag_tensor_size
            if len(prefetch_ags) > 0:
                new_order_rev.append(prefetch_ags)
                ag_tensor_size_sum -= sum([tensor_size_dict[ag_node.name] for ag_node in prefetch_ags])
            assert ag_tensor_size_sum == 0

        # print_rank_0(
        #     f"node={node} next_alloc_mem={next_alloc_mem} pending_ags={len(prefetch_ags)} ag_tensor_size_sum={ag_tensor_size_sum}"
        # )

        assert ag_tensor_size_sum >= 0

    new_graph = Graph()
    env = {}
    for node in reversed(new_order_rev):
        if isinstance(node, Node):
            #print(f"reconstruct {node.name} {node.target}")
            new_node = new_graph.node_copy(node, lambda n: env[n.name])
            env[node.name] = new_node
        else:
            param_nodes = [ag_node.args[0] for ag_node in node]
            param_nodes_copy = [env[param_node.name] for param_node in param_nodes]

            ds_ids = [get_ds_id(ag_node) for ag_node in node]
            new_graph.call_function(torch.ops.dc.prefetch_params_fused.default,
                                    args=(graph_id, param_nodes_copy, ds_ids))
    new_graph.lint()
    gm.graph = new_graph

    return gm
