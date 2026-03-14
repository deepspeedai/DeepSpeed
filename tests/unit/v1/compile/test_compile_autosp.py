# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import operator
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from deepspeed.utils.torch import required_torch_version
from deepspeed.accelerator import get_accelerator
from deepspeed.compile import constants

from unit.v1.compile.util import compare_sp_loss, create_gm_nodes, find_sym_seq_node
from unit.common import DistributedTest
from unit.util import bf16_required_version_check, skip_on_arch

pytestmark = pytest.mark.skipif(not required_torch_version(min_version=2.6),
                                reason="AutoSP tests require PyTorch >= 2.6")

# Fixed sp_size injected into mocks.
_SP_SIZE = 2


class TestAutoSPCompile(DistributedTest):
    world_size = 4
    non_daemonic_procs = True

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

        gm, _ = create_gm_nodes(seq_len=seq_len)
        sdpa_nodes = get_sdpa_nodes(gm)

        assert len(sdpa_nodes) >= 1, f"Expected at least 1 SDPA node, got {len(sdpa_nodes)}"
        for node in sdpa_nodes:
            assert node.target == F.scaled_dot_product_attention


class TestInputIdCompile:

    @pytest.mark.parametrize('seq_len', [64, 128, 256])
    def test(self, seq_len):
        from deepspeed.compile.util import get_input_id_node

        gm, _ = create_gm_nodes(seq_len=seq_len)
        node = get_input_id_node(gm)

        assert node.op == "placeholder"
        tensor_dict = node.meta.get("tensor_dict", {})
        assert tensor_dict.get("tag") == constants.AUTOSP_INPUT_ID_KEY


class TestLabelIdCompile:

    @pytest.mark.parametrize('seq_len', [64, 128, 256])
    def test(self, seq_len):
        from deepspeed.compile.util import get_label_id_node

        gm, _ = create_gm_nodes(seq_len=seq_len)
        node = get_label_id_node(gm)

        assert node.op == "placeholder"
        tensor_dict = node.meta.get("tensor_dict", {})
        assert tensor_dict.get("tag") == constants.AUTOSP_LABEL_ID_KEY


class TestPositionIdCompile:

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
