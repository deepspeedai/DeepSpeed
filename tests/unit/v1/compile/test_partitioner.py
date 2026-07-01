# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch.fx import Graph, GraphModule
from torch.utils.checkpoint import CheckpointPolicy

import deepspeed.compile.partitioner as partitioner


def test_recompute_param_aliases_leaves_unrelated_activations_to_min_cut(monkeypatch):
    monkeypatch.setattr(partitioner, "get_no_copy_ops", lambda: {torch.ops.aten.view.default})

    graph = Graph()
    param = graph.placeholder("param")
    activation = graph.placeholder("activation")
    param_alias = graph.call_function(torch.ops.aten.view.default, args=(param, [4]))
    activation_node = graph.call_function(torch.ops.aten.neg.default, args=(activation, ))
    graph.output((param_alias, activation_node))
    gm = GraphModule(torch.nn.Module(), graph)

    partitioner._recompute_param_aliases(gm.graph, [(0, 1, torch.Size([4]))])

    assert param_alias.meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE
    assert "recompute" not in activation_node.meta
    assert all(node.meta["ac_graph_id"] == 1 for node in gm.graph.nodes)


def test_recompute_param_aliases_preserves_existing_non_param_policy(monkeypatch):
    monkeypatch.setattr(partitioner, "get_no_copy_ops", lambda: {torch.ops.aten.view.default})

    graph = Graph()
    param = graph.placeholder("param")
    activation = graph.placeholder("activation")
    param_alias = graph.call_function(torch.ops.aten.view.default, args=(param, [4]))
    activation_node = graph.call_function(torch.ops.aten.neg.default, args=(activation, ))
    activation_node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
    graph.output((param_alias, activation_node))
    gm = GraphModule(torch.nn.Module(), graph)

    partitioner._recompute_param_aliases(gm.graph, [(0, 1, torch.Size([4]))])

    assert param_alias.meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE
    assert activation_node.meta["recompute"] == CheckpointPolicy.MUST_SAVE
