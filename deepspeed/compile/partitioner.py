# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# This file was copied from PyTorch and modified for DeepSpeed.

from typing import Tuple, List
import operator

import torch
from torch.fx import GraphModule, Graph, Node

try:
    from torch._functorch.partitioners import is_sym_node, _is_primal, _is_fwd_seed_offset, _extract_fwd_bwd_outputs, _extract_graph_with_inputs_outputs, _extract_fwd_bwd_modules, has_recomputable_ops, min_cut_rematerialization_partition, choose_saved_values_set
except ImportError:
    pass

from .util import get_no_copy_ops

_recompute_ops = {torch.ops.aten.t.default}


def _find_recompute_nodes(graph: Graph, ds_param_node: Node) -> List[Node]:
    """
    Given a graph and a node that represents a parameter that was allgathered,
    find all nodes that use the parameter and require recomputation.
    """
    no_copy_ops = get_no_copy_ops()
    recompute_nodes = set()
    for node in graph.nodes:
        if node.target in no_copy_ops:
            if ds_param_node in node.args:
                recompute_nodes.add(node)
            if any(a in recompute_nodes for a in node.args):
                recompute_nodes.add(node)

    return recompute_nodes


def _get_values_from_ds_params(joint_graph, param_indices):
    primal_inputs = list(filter(_is_primal, joint_graph.nodes))
    ds_param_inputs = [primal_inputs[arg_idx] for arg_idx, _, _ in param_indices]

    no_copy_ops = get_no_copy_ops()

    ds_param_inputs = set(ds_param_inputs)
    ds_param_users = {}

    for node in joint_graph.nodes:
        if node.target in no_copy_ops and any((a in ds_param_inputs or a in ds_param_users) for a in node.args):
            for a in node.args:
                if a in ds_param_inputs:
                    ds_param_users[node] = a
                elif a in ds_param_users:
                    ds_param_users[node] = ds_param_users[a]

    return ds_param_users


def get_wrapped_choose_saved_values_set(param_indices: List[Tuple[int, int, torch.Size]]):

    def ds_choose_saved_values_set(joint_graph: torch.fx.Graph, node_info, memory_budget=1) -> List[Node]:
        saved_values = choose_saved_values_set(joint_graph, node_info, memory_budget)
        ds_param_users = _get_values_from_ds_params(joint_graph, param_indices)

        new_saved_values = []
        for v in saved_values:
            if v in ds_param_users:
                ds_val = ds_param_users[v]
                if ds_val not in new_saved_values:
                    new_saved_values.append(ds_val)
            else:
                new_saved_values.append(v)

        return new_saved_values

    return ds_choose_saved_values_set


def get_wrapped_partitioner(param_indices: List[Tuple[int, int, torch.Size]]):

    def partition_recompute_ds_params(joint_module: GraphModule, _joint_inputs, *,
                                      num_fwd_outputs) -> Tuple[GraphModule, GraphModule]:
        """
        This is basically the same as the default_partition function, but
        it doesn't save the gathered params and values computed from them.
        """
        if has_recomputable_ops(joint_module):
            return min_cut_rematerialization_partition(joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs)

        primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
        fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_module.graph.nodes))
        inputs = primal_inputs + fwd_seed_offset_inputs
        fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
        forward_only_graph = _extract_graph_with_inputs_outputs(joint_module.graph, inputs, fwd_outputs, "forward")
        forward_node_names = {node.name for node in forward_only_graph.nodes if node.op != "output"}
        saved_values = []
        saved_sym_nodes = []

        fwd_inputs = list(filter(_is_primal, forward_only_graph.nodes))
        ds_param_inputs = [fwd_inputs[arg_idx] for arg_idx, _, _ in param_indices]
        ds_param_input_names = {node.name for node in ds_param_inputs}

        ds_param_recompute_nodes = set()

        for node in joint_module.graph.nodes:
            if node.name not in forward_node_names:
                continue

            if is_sym_node(node):
                # Symints must be kept separate from tensors so that PythonFunction only calls
                # save_for_backward on tensors and stashes symints in autograd .ctx
                saved_sym_nodes.append(node)
            elif "tensor_meta" not in node.meta and node.op == "call_function":
                # Since we can't save tuple of tensor values, we need to flatten out what we're saving
                users = node.users
                assert all(user.target == operator.getitem for user in users)
                saved_values.extend(users)
            else:
                backward_usages = [n for n in node.users if n.name not in forward_node_names]

                if "tensor_meta" in node.meta and all(is_sym_node(n) for n in backward_usages):
                    # If we have a tensor in the forward, where only its sizes/strides are needed in the backward,
                    # and not the actual tensor data,
                    # then it will be a lot cheaper to save only the sizes/strides, and not the actual tensor.
                    #
                    # Note that saving the tensor could also cause compilation problems:
                    # If the user mutated an input in the forward and uses its sizes/strides in the backward,
                    # then we would be obligated to clone the input before saving it to appease autograd.
                    # (This is how we originally found this bug).
                    saved_sym_nodes.extend(backward_usages)

                    if node.name in ds_param_input_names:
                        saved_values.append(node)
                        recompute_nodes = _find_recompute_nodes(joint_module.graph, node)
                        recompute_nodes = [n for n in recompute_nodes if n.name in forward_node_names]
                        for recompute_node in recompute_nodes:
                            ds_param_recompute_nodes.add(recompute_node)

                        if len(recompute_nodes) > 0:
                            saved_values.append(node)
                else:
                    if node not in ds_param_recompute_nodes:
                        saved_values.append(node)
        saved_values = list(dict.fromkeys(saved_values).keys())
        saved_sym_nodes = list(dict.fromkeys(saved_sym_nodes).keys())

        f_gm, b_gm = _extract_fwd_bwd_modules(
            joint_module,
            saved_values,
            saved_sym_nodes=saved_sym_nodes,
            num_fwd_outputs=num_fwd_outputs,
        )

        return f_gm, b_gm

    return partition_recompute_ds_params
