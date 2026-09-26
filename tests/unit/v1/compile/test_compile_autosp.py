# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import operator
import re
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch.fx import Graph, GraphModule

from deepspeed.utils.torch import required_torch_version
from deepspeed.accelerator import get_accelerator
from deepspeed.compile import constants

from unit.v1.compile.util import compare_sp_loss, create_gm_nodes, find_sym_seq_node
from unit.common import DistributedTest
from unit.util import bf16_required_version_check, skip_on_arch

pytestmark = pytest.mark.skipif(not required_torch_version(min_version=2.9),
                                reason="AutoSP tests require PyTorch >= 2.9")

# Fixed sp_size injected into mocks.
_SP_SIZE = 2


def _create_sdpa_graph(seq_len, num_heads=2, num_kv_heads=None):
    num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    graph = Graph()
    inputs = []
    for name, heads in (("query", num_heads), ("key", num_kv_heads), ("value", num_kv_heads)):
        node = graph.placeholder(name)
        node.meta["example_value"] = torch.empty(1, heads, seq_len, 8)
        inputs.append(node)
    sdpa = graph.call_function(F.scaled_dot_product_attention, args=tuple(inputs))
    graph.output(sdpa)
    return GraphModule({}, graph)


class TestAutoSPCompile(DistributedTest):
    world_size = 4
    non_daemonic_procs = True

    @pytest.mark.sequential
    @pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize('zero_stage', [0, 1])
    @pytest.mark.parametrize('sp_size', [2, 4])
    def test(self, zero_stage, dtype, sp_size):
        if dtype == torch.bfloat16:
            skip_on_arch(min_arch=8)
        if dtype == torch.bfloat16 and not bf16_required_version_check():
            pytest.skip(
                "DeepSpeed BFloat16 tests need NCCL >= 2.10.3, CUDA >=11.0, and HW support for BFloat16 to run correctly"
            )
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        dp_size = self.world_size // sp_size

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": dp_size,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
            },
            "compile": {
                "deepcompile": True,
                "passes": ["autosp"]
            },
            "sequence_parallel_size": sp_size,
            "gradient_clipping": 1.0,
        }

        if dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        compare_sp_loss(self, config_dict, sp_size)


# Plain pytest classes — no distributed runtime needed because these functions
# perform pure IR-level graph rewrites; sp_size and get_rank are mocked.


class TestSDPANodesCompile:

    @pytest.mark.parametrize('seq_len', [64, 128, 256])
    def test(self, seq_len):
        from deepspeed.compile.util import get_sdpa_nodes

        # This DeepSpeed 0.19.3 patch was historically validated with Transformers 4.51.3, 5.13.0,
        # and main@e9c987a3eceb1cd130f30cb5e1f1fac542e4fcfe (5.15.0.dev0); future revisions are not guaranteed.
        gm = _create_sdpa_graph(seq_len)
        sdpa_nodes = get_sdpa_nodes(gm)

        assert len(sdpa_nodes) >= 1, f"Expected at least 1 SDPA node, got {len(sdpa_nodes)}"
        for node in sdpa_nodes:
            assert node.target == F.scaled_dot_product_attention


class TestInputIdCompile:

    @pytest.mark.sequential
    @pytest.mark.parametrize('seq_len', [64, 128, 256])
    def test(self, seq_len):
        from deepspeed.compile.util import get_input_id_node

        gm, _ = create_gm_nodes(seq_len=seq_len)
        node = get_input_id_node(gm)

        assert node.op == "placeholder"
        tensor_dict = node.meta.get("tensor_dict", {})
        assert tensor_dict.get("tag") == constants.AUTOSP_INPUT_ID_KEY


class TestLabelIdCompile:

    @pytest.mark.sequential
    @pytest.mark.parametrize('seq_len', [64, 128, 256])
    def test(self, seq_len):
        from deepspeed.compile.util import get_label_id_node

        gm, _ = create_gm_nodes(seq_len=seq_len)
        node = get_label_id_node(gm)

        assert node.op == "placeholder"
        tensor_dict = node.meta.get("tensor_dict", {})
        assert tensor_dict.get("tag") == constants.AUTOSP_LABEL_ID_KEY


class TestPositionIdCompile:

    @pytest.mark.sequential
    @pytest.mark.parametrize('seq_len', [64, 128, 256])
    def test(self, seq_len):
        from deepspeed.compile.util import get_position_id_node

        gm, _ = create_gm_nodes(seq_len=seq_len)
        node = get_position_id_node(gm)

        assert node is not None, "position_id node not found in graph"
        assert node.op == "placeholder"
        tensor_dict = node.meta.get("tensor_dict", {})
        assert tensor_dict.get("tag") == constants.AUTOSP_POSITION_ID_KEY


class TestShardOffsetsCompile:

    @pytest.mark.sequential
    @pytest.mark.parametrize('seq_len', [64, 128, 256])
    def test(self, seq_len):
        import deepspeed.comm as _dist
        from deepspeed.compile.custom_ops import sp_dp_registry as _registry
        from deepspeed.compile.util import create_shard_offsets

        gm, _ = create_gm_nodes(seq_len=seq_len)
        sym_seq_node = find_sym_seq_node(gm)
        assert sym_seq_node is not None, "Symbolic sequence-length node not found in graph"

        with patch.object(_registry, 'sp_size', return_value=_SP_SIZE), \
             patch.object(_dist, 'get_rank', return_value=0):
            start_node, end_node = create_shard_offsets(gm, sym_seq_node)

        # create_shard_offsets emits: chunk = seq // sp_size; start = rank * chunk; end = start + chunk.
        # Verify the three-node chain has the right operators and wiring.
        chunk_size_node = start_node.args[1]  # start = rank * chunk  →  chunk is arg[1]

        assert chunk_size_node.target == operator.floordiv
        assert chunk_size_node.args[0] is sym_seq_node
        assert chunk_size_node.args[1] == _SP_SIZE

        assert start_node.target == operator.mul
        assert start_node.args[0] == 0  # rank 0 baked in at transform time
        assert start_node.args[1] is chunk_size_node

        assert end_node.target == operator.add
        assert end_node.args[0] is start_node
        assert end_node.args[1] is chunk_size_node


class TestSymSliceCompile:

    @pytest.mark.sequential
    @pytest.mark.parametrize('seq_len', [64, 128, 256])
    def test(self, seq_len):
        import deepspeed.comm as _dist
        from deepspeed.compile.custom_ops import sp_dp_registry as _registry
        from deepspeed.compile.util import create_symbolic_slice_indices

        gm, _ = create_gm_nodes(seq_len=seq_len)
        sym_seq_node = find_sym_seq_node(gm)
        assert sym_seq_node is not None, "Symbolic sequence-length node not found in graph"

        with patch.object(_registry, 'sp_size', return_value=_SP_SIZE), \
             patch.object(_dist, 'get_rank', return_value=0):
            slice_all, slice_range = create_symbolic_slice_indices(gm, sym_seq_node)

        # slice_all = slice(None, None, None) — selects the batch dimension unchanged
        assert slice_all.target == slice
        assert slice_all.args == (None, None, None)

        # slice_range selects [start, end) along the sequence dim, where start and
        # end come from create_shard_offsets (mul and add nodes respectively).
        assert slice_range.target == slice
        start_arg, end_arg, step_arg = slice_range.args
        assert step_arg is None

        # start = rank * chunk  →  verify the full shard-offset wiring
        chunk_size_node = start_arg.args[1]
        assert start_arg.target == operator.mul
        assert start_arg.args[0] == 0  # rank 0 baked in at transform time
        assert chunk_size_node.target == operator.floordiv
        assert chunk_size_node.args[0] is sym_seq_node
        assert chunk_size_node.args[1] == _SP_SIZE

        # end = start + chunk
        assert end_arg.target == operator.add
        assert end_arg.args[0] is start_arg
        assert end_arg.args[1] is chunk_size_node


class TestShardTensorCompile:

    @pytest.mark.sequential
    @pytest.mark.parametrize('seq_len', [64, 128, 256])
    def test(self, seq_len):
        import deepspeed.comm as _dist
        from deepspeed.compile.custom_ops import sp_dp_registry as _registry
        from deepspeed.compile.util import shard_tensor_node, get_input_id_node

        gm, _ = create_gm_nodes(seq_len=seq_len)
        input_ids_node = get_input_id_node(gm)
        original_users = set(input_ids_node.users.keys())
        assert len(original_users) > 0, "input_ids_node must have users before sharding"

        with patch.object(_registry, 'sp_size', return_value=_SP_SIZE), \
             patch.object(_dist, 'get_rank', return_value=0):
            shard_tensor_node(gm, input_ids_node)

        getitem_nodes = [n for n in gm.graph.nodes if n.target == operator.getitem and n.args[0] is input_ids_node]
        assert len(getitem_nodes) == 1, f"Expected 1 slice node after sharding, got {len(getitem_nodes)}"
        sliced_node = getitem_nodes[0]

        # After sharding, the raw node must only feed the slice; all downstream
        # consumers are rewired to sliced_node by replace_node_users.
        assert set(input_ids_node.users.keys()) == {sliced_node}

        for user in original_users:
            assert input_ids_node not in user.all_input_nodes, \
                f"User '{user.name}' still references the unsharded input_ids_node"
            assert sliced_node in user.all_input_nodes, \
                f"User '{user.name}' does not reference the sliced node"

    @pytest.mark.sequential
    def test_preserves_topological_order_when_sym_placeholder_follows_input(self):
        import deepspeed.comm as _dist
        from deepspeed.compile.custom_ops import sp_dp_registry as _registry
        from deepspeed.compile.fx import find_node_by_name, get_node_shape_meta
        from deepspeed.compile.util import shard_tensor_node, get_input_id_node

        # Regression test for the torch 2.9 bf16 trace where the SymInt
        # placeholder can appear after input_ids. shard_tensor_node must still
        # produce a lint-clean graph instead of inserting getitem before its
        # symbolic slice dependencies.
        gm, _ = create_gm_nodes(seq_len=64)
        input_ids_node = get_input_id_node(gm)
        seq_symint = get_node_shape_meta(input_ids_node).shape[1]
        sym_seq_node = find_node_by_name(gm, str(seq_symint))
        assert sym_seq_node is not None, "Symbolic sequence-length node not found in graph"

        nodes = list(gm.graph.nodes)
        input_idx = nodes.index(input_ids_node)
        sym_idx = nodes.index(sym_seq_node)
        assert sym_idx < input_idx, "Expected source graph to place the symbolic placeholder before input_ids"

        # Reorder placeholders to mirror the torch 2.9 bf16 trace where the symbolic
        # sequence placeholder can appear after input_ids.
        reordered_nodes = nodes[:]
        reordered_nodes.pop(input_idx)
        reordered_nodes.insert(sym_idx, input_ids_node)
        reordered_nodes.pop(sym_idx + 1)
        reordered_nodes.insert(input_idx, sym_seq_node)

        reordered_graph = Graph()
        env = {}
        for node in reordered_nodes:
            new_node = reordered_graph.node_copy(node, lambda n: env[n])
            new_node.meta = node.meta.copy()
            env[node] = new_node
        reordered_graph.lint()

        reordered_gm = GraphModule(gm, reordered_graph)
        reordered_input_ids = get_input_id_node(reordered_gm)

        with patch.object(_registry, 'sp_size', return_value=_SP_SIZE), \
             patch.object(_dist, 'get_rank', return_value=0):
            shard_tensor_node(reordered_gm, reordered_input_ids)

        reordered_gm.graph.lint()


def _create_shard_offsets_graph():
    """Graph mapping the (dynamic) sequence length to this rank's (start, end) shard offsets."""
    import deepspeed.comm as _dist
    from deepspeed.compile.custom_ops import sp_dp_registry as _registry
    from deepspeed.compile.util import create_shard_offsets

    graph = Graph()
    seq_len = graph.placeholder("seq_len")
    output = graph.output(seq_len)
    gm = GraphModule({}, graph)
    with patch.object(_registry, "sp_size", return_value=_SP_SIZE), \
         patch.object(_dist, "get_rank", return_value=0):
        start, end = create_shard_offsets(gm, seq_len)
    output.args = ((start, end), )
    gm.recompile()
    return gm


class TestAutoSPDivisibilityValidation:
    """AutoSP splits the sequence across SP ranks and the heads in the all-to-all, so both must divide evenly."""

    @pytest.mark.parametrize("seq_len, expected", [(16, (0, 8)), (32, (0, 16))])
    def test_shard_offsets_for_divisible_sequence_length(self, seq_len, expected):
        assert _create_shard_offsets_graph()(seq_len) == expected

    def test_shard_offsets_reject_non_divisible_sequence_length(self):
        # The sequence length is dynamic, so the check has to run with the graph rather than at compile time.
        gm = _create_shard_offsets_graph()
        with pytest.raises(RuntimeError, match="sequence length to be divisible by sequence_parallel_size"):
            gm(15)

    @pytest.mark.parametrize("num_heads, num_kv_heads, role, bad_heads", [(3, 3, "query", 3), (4, 1, "key", 1)],
                             ids=["mha", "gqa"])
    def test_rejects_non_divisible_attention_heads(self, num_heads, num_kv_heads, role, bad_heads):
        from deepspeed.compile.custom_ops import sp_dp_registry as _registry
        from deepspeed.compile.passes.sp_compile import pass_insert_attention_all_to_all

        gm = _create_sdpa_graph(seq_len=8, num_heads=num_heads, num_kv_heads=num_kv_heads)
        with patch.object(_registry, "sp_size", return_value=_SP_SIZE):
            with pytest.raises(ValueError, match=re.escape(f"number of {role} heads ({bad_heads}) to be divisible")):
                pass_insert_attention_all_to_all(gm, ())

    def test_accepts_grouped_query_attention_with_divisible_heads(self):
        from deepspeed.compile.custom_ops import sp_dp_registry as _registry
        from deepspeed.compile.passes.sp_compile import pass_insert_attention_all_to_all

        gm = _create_sdpa_graph(seq_len=8, num_heads=4, num_kv_heads=2)
        with patch.object(_registry, "sp_size", return_value=_SP_SIZE):
            pass_insert_attention_all_to_all(gm, ())

        a2a_nodes = [n for n in gm.graph.nodes if n.target == torch.ops.autosp.all_to_all.default]
        assert len(a2a_nodes) == 4

    def test_all_to_all_rejects_non_divisible_heads_before_the_collective(self):
        import importlib
        import deepspeed.comm as _dist

        a2a_module = importlib.import_module("deepspeed.compile.custom_ops.all_to_all")

        def fail_collective(*args, **kwargs):
            raise AssertionError("the collective must not run for a non-divisible head count")

        with patch.object(a2a_module, "is_setup", return_value=True), \
             patch.object(a2a_module, "sp_size", return_value=_SP_SIZE), \
             patch.object(a2a_module, "get_group", return_value=None), \
             patch.object(_dist, "get_rank", return_value=0), \
             patch.object(_dist, "all_to_all_single", side_effect=fail_collective):
            with pytest.raises(ValueError, match=re.escape("number of attention heads (3)")):
                torch.ops.autosp.all_to_all(torch.empty(1, 3, 4, 8), 1, 2, "q")
