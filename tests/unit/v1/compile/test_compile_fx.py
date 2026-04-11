# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
from torch.fx import Graph

from deepspeed.compile.fx import add_end_backward, replace_reduce_outputs_with_none, get_output_node
from deepspeed.compile.util import get_deepcompile_handle, is_deepcompile_supported


@pytest.mark.skipif(not is_deepcompile_supported(), reason="DeepCompile requires CUDA and supported PyTorch")
def test_end_backward_depends_on_all_reduce_nodes():
    get_deepcompile_handle()

    graph = Graph()
    grad = graph.placeholder("grad")
    reduce_a = graph.create_node("call_function", torch.ops.dc.reduce_grad.default, (grad, 7, 11), name="reduce_a")
    reduce_b = graph.create_node("call_function", torch.ops.dc.reduce_grad.default, (grad, 7, 12), name="reduce_b")
    graph.output((grad, ))

    add_end_backward(graph, 7)
    replace_reduce_outputs_with_none(graph)
    graph.lint()

    end_backward = next(n for n in graph.nodes if n.target == torch.ops.dc.end_backward.default)
    deps, graph_id = end_backward.args
    output_node = get_output_node(graph)

    assert graph_id == 7
    assert list(deps) == [reduce_a, reduce_b]
    assert end_backward in reduce_a.users
    assert end_backward in reduce_b.users
    assert output_node.args == ((grad, ), )
