# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.fx import Graph, GraphModule

_DC_SCHEMAS = (
    "wait_allgather(Tensor(a) a, int graph_id, int id) -> Tensor(a)",
    "release_param(Tensor(a) a, int graph_id, int id, int n_users) -> Tensor(a)",
    "offload_tensor(Tensor a, int graph_id, int id) -> Tensor",
    "reload_tensor(Tensor a, int graph_id, int id) -> Tensor",
    "wait_offload(Tensor a, int graph_id, int id) -> Tensor",
    "wait_reload(Tensor a, int graph_id, int id) -> Tensor",
)


def _register_missing_dc_ops():
    library = torch.library.Library("dc", "FRAGMENT")
    for schema in _DC_SCHEMAS:
        name = schema.split("(", 1)[0]
        try:
            getattr(torch.ops.dc, name).default
        except AttributeError:
            library.define(schema)
    return library


def test_release_preserves_saved_output_metadata():
    library = _register_missing_dc_ops()
    from deepspeed.compile import util
    from deepspeed.compile.passes import offload_activation
    from deepspeed.compile.passes.zero3_compile import add_release

    try:
        graph = Graph()
        param = graph.placeholder("param")
        consumer = graph.call_function(torch.ops.aten.slice.Tensor, (param, 0, 1, 8, 2))
        base = torch.arange(10, dtype=torch.float32)
        view = base[1:8:2]
        assert view.stride() != (1, )
        assert view.storage_offset() != 0
        assert view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
        consumer.meta["val"] = view
        output = graph.output((consumer, ))

        add_release(0, graph, consumer, param, 7, 1)
        graph.lint()

        releases = [node for node in graph.nodes if node.target == torch.ops.dc.release_param.default]
        assert len(releases) == 1
        release = releases[0]
        nodes = list(graph.nodes)
        assert nodes.index(release) == nodes.index(consumer) + 1
        assert release.args == (consumer, 0, 7, 1)
        assert output.args == ((release, ), )
        assert tuple(consumer.users) == (release, )
        assert tuple(release.users) == (output, )

        assert release.meta["val"] is consumer.meta["val"]
        assert isinstance(release.meta["val"], torch.Tensor)
        assert release.meta["val"].shape == view.shape
        assert release.meta["val"].dtype == view.dtype
        assert release.meta["val"].device == view.device
        assert release.meta["val"].stride() == view.stride()
        assert release.meta["val"].storage_offset() == view.storage_offset()
        assert release.meta["val"].untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
        assert offload_activation._is_floating_point(release)

        meta_view = torch.empty_strided(view.shape, view.stride(), dtype=view.dtype, device="meta")
        consumer.meta["val"] = meta_view
        release.meta["val"] = meta_view
        forward = GraphModule(torch.nn.Module(), graph)

        backward_graph = Graph()
        saved = backward_graph.placeholder(consumer.name)
        saved.meta["val"] = meta_view
        backward_user = backward_graph.call_function(torch.ops.aten.sin.default, (saved, ))
        backward_user.meta["val"] = meta_view
        backward_graph.output((backward_user, ))
        backward = GraphModule(torch.nn.Module(), backward_graph)
        profile = {0: SimpleNamespace(num_fwd_outputs=0, fwd_mem=[], bwd_mem=[], bwd_time=[])}

        util.get_no_copy_ops.cache_clear()
        offload_activation._offload_plans.clear()
        offload_activation._h2d_bytes_per_sec = 1e10
        with patch("deepspeed.compile.util.get_deepcompile_handle", return_value=None), \
                patch.dict("os.environ", {"DS_DC_OFFLOAD_ACT_MIN_SIZE_MB": "0", "DS_DC_OFFLOAD_ACT_BUDGET_GB": "1"}):
            offload_activation._offload_everything_fwd(forward, 0, profile, {})
            offload_activation._reload_activation_bwd(backward, 0, profile)

        assert list(offload_activation._offload_plans[0]) == [consumer.name]
        offloads = [node for node in graph.nodes if node.target == torch.ops.dc.offload_tensor.default]
        reloads = [node for node in backward_graph.nodes if node.target == torch.ops.dc.reload_tensor.default]
        assert len(offloads) == 1
        assert len(reloads) == 1
        assert reloads[0].args[0] is saved
        assert reloads[0].args[2] == offloads[0].args[2]

        offload_activation._bring_back(graph, consumer.name)
        graph.lint()
        assert output.args == ((release, ), )
        assert tuple(release.users) == (output, )
        assert not any(node.target in (torch.ops.dc.offload_tensor.default, torch.ops.dc.wait_offload.default)
                       for node in graph.nodes)
    finally:
        util.get_no_copy_ops.cache_clear()
        offload_activation._offload_plans.clear()
        offload_activation.reset_offload_activation_stats()
        offload_activation._h2d_bytes_per_sec = None
        library._destroy()
