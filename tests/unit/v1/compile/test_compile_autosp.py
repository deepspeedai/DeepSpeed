# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import operator
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


def _create_sdpa_graph(seq_len, num_heads=2, mask_rank=None):
    graph = Graph()
    inputs = []
    for name in ("query", "key", "value"):
        node = graph.placeholder(name)
        node.meta["example_value"] = torch.empty(1, num_heads, seq_len, 8)
        inputs.append(node)
    kwargs = {}
    if mask_rank is not None:
        mask = graph.placeholder("attention_mask")
        mask_shape = (seq_len, seq_len * _SP_SIZE)
        if mask_rank == 4:
            mask_shape = (1, 1) + mask_shape
        mask.meta["example_value"] = torch.empty(mask_shape)
        kwargs["attn_mask"] = mask
    sdpa = graph.call_function(F.scaled_dot_product_attention, args=tuple(inputs), kwargs=kwargs)
    graph.output(sdpa)
    return GraphModule({}, graph)


def _create_causal_loss_graph(seq_len=16, ignore_index=-100, shift_labels=True):

    class CausalLoss(torch.nn.Module):

        def forward(self, input_ids, labels):
            logits = F.one_hot(input_ids, num_classes=32).float()
            targets = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous() if shift_labels else labels
            return F.cross_entropy(logits.view(-1, 32), targets.view(-1), ignore_index=ignore_index)

    torch._dynamo.reset()
    input_ids = torch.randint(0, 32, (2, seq_len))
    labels = input_ids.clone()
    input_ids.tag = constants.AUTOSP_INPUT_ID_KEY
    labels.tag = constants.AUTOSP_LABEL_ID_KEY
    torch._dynamo.decorators.mark_dynamic(input_ids, 1)
    torch._dynamo.decorators.mark_dynamic(labels, 1)

    captured_gm = [None]

    def capture(gm, example_inputs):
        captured_gm[0] = gm
        return gm

    compiled = torch.compile(CausalLoss(), backend=capture, dynamic=True)
    compiled(input_ids, labels)
    return captured_gm[0]


def _create_sequence_first_graph(seq_len=16):

    class SequenceFirst(torch.nn.Module):

        def forward(self, input_ids, labels):
            return input_ids.float().sum() + labels.float().sum()

    torch._dynamo.reset()
    input_ids = torch.ones(seq_len, 2, dtype=torch.long)
    labels = input_ids.clone()
    input_ids.tag = (constants.AUTOSP_INPUT_ID_KEY, 0)
    labels.tag = (constants.AUTOSP_LABEL_ID_KEY, 0)
    torch._dynamo.decorators.mark_dynamic(input_ids, 0)
    torch._dynamo.decorators.mark_dynamic(labels, 0)

    captured_gm = [None]

    def capture(gm, example_inputs):
        captured_gm[0] = gm
        return gm

    compiled = torch.compile(SequenceFirst(), backend=capture, dynamic=True)
    compiled(input_ids, labels)
    return captured_gm[0]


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
        assert_node = next(node for node in gm.graph.nodes if node.target == torch._assert)
        divisible_node = assert_node.args[0]
        remainder_node = divisible_node.args[0]

        assert divisible_node.target == operator.eq
        assert divisible_node.args[1] == 0
        assert remainder_node.target == operator.mod
        assert remainder_node.args == (sym_seq_node, _SP_SIZE)

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


class TestAutoSPValidation:

    def test_reads_tensor_shape_metadata_without_boolean_conversion(self):
        from deepspeed.compile.fx import get_node_shape_meta

        node = Graph().placeholder("input")
        value = torch.empty(2)
        node.meta["val"] = value

        assert get_node_shape_meta(node) is value

    def test_prepare_inputs_preserves_non_default_sequence_dimension(self):
        from deepspeed.compile.passes.sp_compile import prepare_autosp_inputs

        input_ids = torch.ones(8, 2, dtype=torch.long)
        labels = input_ids.clone()
        position_ids = input_ids.clone()
        with patch.object(torch._dynamo.decorators, "mark_dynamic"):
            prepare_autosp_inputs(input_ids, labels, position_ids, seq_dim=0)

        assert input_ids.tag == (constants.AUTOSP_INPUT_ID_KEY, 0)
        assert labels.tag == (constants.AUTOSP_LABEL_ID_KEY, 0)
        assert position_ids.tag == (constants.AUTOSP_POSITION_ID_KEY, 0)

    @pytest.mark.sequential
    def test_shards_non_default_sequence_dimension(self):
        import deepspeed.comm as _dist
        from deepspeed.compile.custom_ops import sp_dp_registry
        from deepspeed.compile.passes.sp_compile import pass_shard_input_ids
        from deepspeed.compile.util import get_input_id_node

        gm = _create_sequence_first_graph()
        input_node = get_input_id_node(gm)
        with patch.object(sp_dp_registry, "sp_size", return_value=_SP_SIZE), \
             patch.object(_dist, "get_rank", return_value=0):
            pass_shard_input_ids(gm, ())

        shard = next(node for node in gm.graph.nodes if node.target == operator.getitem and node.args[0] is input_node)
        indices = shard.args[1]
        assert indices[0].target == slice
        assert indices[0].args[0] is not None
        assert indices[1].target == slice
        assert indices[1].args == (None, None, None)

    @pytest.mark.sequential
    def test_rejects_non_divisible_sequence_length(self):
        import deepspeed.comm as _dist
        from deepspeed.compile.custom_ops import sp_dp_registry
        from deepspeed.compile.passes.sp_compile import pass_canonicalize, pass_shard_input_ids

        gm = _create_sequence_first_graph()
        with patch.object(sp_dp_registry, "sp_size", return_value=_SP_SIZE), \
             patch.object(_dist, "get_rank", return_value=0):
            pass_shard_input_ids(gm, ())
            pass_canonicalize(gm, ())

        input_ids = torch.ones(15, 2, dtype=torch.long)
        labels = input_ids.clone()
        with pytest.raises(AssertionError, match="sequence length must be divisible"):
            gm(15, 2, input_ids, 15, labels)

    def test_rejects_changed_mesh(self):
        from deepspeed.compile.custom_ops import sp_dp_registry

        registry = {"SP_SIZE": 2, "DP_SIZE": 2, "is_reg": True}
        with patch.object(sp_dp_registry, "GROUP_REGISTRY", registry), \
             patch.object(sp_dp_registry.dist, "get_world_size", return_value=4):
            with pytest.raises(RuntimeError, match="already initialized"):
                sp_dp_registry.populate_registry(4, 1)

    def test_rejects_partial_mesh(self):
        from deepspeed.compile.custom_ops import sp_dp_registry

        with patch.object(sp_dp_registry.dist, "get_world_size", return_value=4):
            with pytest.raises(ValueError, match="must cover"):
                sp_dp_registry.populate_registry(2, 1)

    def test_all_ignored_loss_shard_contributes_zero(self):
        import deepspeed.comm as _dist
        from deepspeed.compile.custom_ops import sp_dp_registry
        from deepspeed.compile.custom_ops.all_to_all import aggregate_loss

        loss = torch.tensor(float("nan"), requires_grad=True)
        valid_tokens = torch.tensor(0)
        registry = {0: object(), "SP_SIZE": 2, "DP_SIZE": 1, "is_reg": True}
        with patch.object(sp_dp_registry, "GROUP_REGISTRY", registry), \
             patch.object(_dist, "get_rank", return_value=0), \
             patch.object(_dist, "all_reduce"):
            global_loss, weight = aggregate_loss(loss, valid_tokens)
            global_loss.backward()

        assert global_loss.item() == 0
        assert weight.item() == 0
        assert loss.grad.item() == 0

    def test_rejects_non_divisible_attention_heads(self):
        from deepspeed.compile.custom_ops import sp_dp_registry
        from deepspeed.compile.passes.sp_compile import pass_insert_attention_all_to_all

        gm = _create_sdpa_graph(seq_len=8, num_heads=3)
        with patch.object(sp_dp_registry, "sp_size", return_value=_SP_SIZE):
            with pytest.raises(ValueError, match="query head count"):
                pass_insert_attention_all_to_all(gm, ())

    def test_gathers_local_attention_mask_query_dimension(self):
        from deepspeed.compile.custom_ops import sp_dp_registry
        from deepspeed.compile.passes.sp_compile import pass_insert_attention_all_to_all

        gm = _create_sdpa_graph(seq_len=8, mask_rank=4)
        with patch.object(sp_dp_registry, "sp_size", return_value=_SP_SIZE):
            pass_insert_attention_all_to_all(gm, ())

        sdpa_node = next(node for node in gm.graph.nodes if node.target == F.scaled_dot_product_attention)
        mask_node = sdpa_node.kwargs["attn_mask"]
        assert mask_node.target == torch.ops.autosp.all_gather_sequence.default
        assert mask_node.args[1] == -2

    def test_gathers_rank_two_attention_mask_query_dimension(self):
        from deepspeed.compile.custom_ops import sp_dp_registry
        from deepspeed.compile.passes.sp_compile import pass_insert_attention_all_to_all

        gm = _create_sdpa_graph(seq_len=8, mask_rank=2)
        with patch.object(sp_dp_registry, "sp_size", return_value=_SP_SIZE):
            pass_insert_attention_all_to_all(gm, ())

        sdpa_node = next(node for node in gm.graph.nodes if node.target == F.scaled_dot_product_attention)
        assert sdpa_node.kwargs["attn_mask"].target == torch.ops.autosp.all_gather_sequence.default

    @pytest.mark.sequential
    def test_shards_labels_after_causal_shift(self):
        import deepspeed.comm as _dist
        from deepspeed.compile.custom_ops import sp_dp_registry
        from deepspeed.compile.passes.sp_compile import pass_shard_label_ids
        from deepspeed.compile.util import get_label_id_node

        gm = _create_causal_loss_graph(seq_len=64)
        label_node = get_label_id_node(gm)
        with patch.object(sp_dp_registry, "sp_size", return_value=_SP_SIZE), \
             patch.object(_dist, "get_rank", return_value=0):
            pass_shard_label_ids(gm, ())

        raw_label_slices = [
            node for node in gm.graph.nodes if node.target == operator.getitem and node.args[0] is label_node
        ]
        assert not raw_label_slices
        assert any(node.target == torch.ops.autosp.aggregate_loss.default for node in gm.graph.nodes)
        gm.graph.lint()

    @pytest.mark.sequential
    def test_shards_direct_cross_entropy_labels(self):
        import deepspeed.comm as _dist
        from deepspeed.compile.custom_ops import sp_dp_registry
        from deepspeed.compile.passes.sp_compile import pass_shard_label_ids
        from deepspeed.compile.util import get_label_id_node

        gm = _create_causal_loss_graph(seq_len=64, shift_labels=False)
        label_node = get_label_id_node(gm)
        with patch.object(sp_dp_registry, "sp_size", return_value=_SP_SIZE), \
             patch.object(_dist, "get_rank", return_value=0):
            pass_shard_label_ids(gm, ())

        raw_label_slices = [
            node for node in gm.graph.nodes if node.target == operator.getitem and node.args[0] is label_node
        ]
        assert len(raw_label_slices) == 1
        assert not any(node.target == torch.ops.autosp.aggregate_loss.default for node in gm.graph.nodes)
        gm.graph.lint()

    @pytest.mark.sequential
    def test_causal_loss_uses_configured_ignore_index(self):
        import deepspeed.comm as _dist
        from deepspeed.compile.custom_ops import sp_dp_registry
        from deepspeed.compile.passes.sp_compile import pass_shard_label_ids

        gm = _create_causal_loss_graph(seq_len=64, ignore_index=-1)
        with patch.object(sp_dp_registry, "sp_size", return_value=_SP_SIZE), \
             patch.object(_dist, "get_rank", return_value=0):
            pass_shard_label_ids(gm, ())

        loss_node = next(node for node in gm.graph.nodes if node.target is F.cross_entropy)
        ignore_index = loss_node.kwargs["ignore_index"] if "ignore_index" in loss_node.kwargs else loss_node.args[4]
        valid_mask = next(node for node in gm.graph.nodes if node.target == operator.ne)
        assert valid_mask.args[1] is ignore_index
