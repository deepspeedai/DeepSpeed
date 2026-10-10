# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
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


class _SeqDimModel(torch.nn.Module):
    """Causal single-layer transformer whose inputs carry the sequence on `seq_dim` of a 2-D [seq, batch] layout."""

    def __init__(self, seq_dim, vocab_size=64, hidden=32, num_heads=2):
        super().__init__()
        self.seq_dim = seq_dim
        self.num_heads = num_heads
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.pos_embed = torch.nn.Embedding(1024, hidden)
        self.qkv = torch.nn.Linear(hidden, 3 * hidden)
        self.out = torch.nn.Linear(hidden, hidden)
        self.head = torch.nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, labels, position_ids):
        if self.seq_dim == 1:
            input_ids, labels, position_ids = input_ids.t(), labels.t(), position_ids.t()
        seq_len, batch = input_ids.shape
        x = self.embed(input_ids) + self.pos_embed(position_ids)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def to_heads(t):
            return t.view(seq_len, batch, self.num_heads, -1).permute(1, 2, 0, 3)

        attn = F.scaled_dot_product_attention(to_heads(q), to_heads(k), to_heads(v), is_causal=True)
        x = x + self.out(attn.permute(2, 0, 1, 3).reshape(seq_len, batch, -1))
        logits = self.head(x)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


class TestAutoSPSeqDim(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.sequential
    @pytest.mark.parametrize("seq_dim", [0, 1])
    def test_matches_full_sequence_reference(self, seq_dim):
        import deepspeed
        import deepspeed.comm as dist
        from deepspeed.compile.passes.sp_compile import prepare_autosp_inputs

        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        sp_size = self.world_size
        seq_len, batch, vocab_size = 16, 2, 64
        device = get_accelerator().current_device_name()

        torch.manual_seed(0)
        reference = _SeqDimModel(seq_dim, vocab_size=vocab_size).to(device)
        config = {
            "train_micro_batch_size_per_gpu": batch,
            "train_batch_size": batch,
            "zero_optimization": {
                "stage": 0
            },
            "compile": {
                "deepcompile": True,
                "passes": ["autosp"]
            },
            "sequence_parallel_size": sp_size,
        }
        engine, _, _, _ = deepspeed.initialize(config=config, model=copy.deepcopy(reference))
        engine.compile()

        # Only forward losses are compared: they pin which dimension is sharded without
        # depending on how gradients are reduced across the SP group.
        for step in range(3):
            torch.manual_seed(100 + step)
            input_ids = torch.randint(0, vocab_size, (seq_len, batch), device=device)
            labels = torch.randint(0, vocab_size, (seq_len, batch), device=device)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(1).expand(seq_len, batch)
            if seq_dim == 1:
                input_ids, labels, position_ids = input_ids.t(), labels.t(), position_ids.t()
            input_ids, labels, position_ids = (t.contiguous() for t in (input_ids, labels, position_ids))

            with torch.no_grad():
                reference_loss = reference(input_ids, labels, position_ids)

            autosp_inputs = prepare_autosp_inputs(input_ids.clone(),
                                                  labels.clone(),
                                                  position_ids.clone(),
                                                  seq_dim=seq_dim)
            loss = engine(*autosp_inputs[:3])

            # The SP group is the whole world. Every rank holds the mean over an equally sized
            # shard of the sequence, so their average is the full-sequence mean.
            global_loss = loss.detach().clone()
            dist.all_reduce(global_loss)
            global_loss /= sp_size
            torch.testing.assert_close(global_loss, reference_loss, rtol=1e-4, atol=1e-5)

        engine.destroy()


class _CausalLMModel(torch.nn.Module):
    """Single-layer causal LM that shifts its labels inside the loss, like Hugging Face causal LMs."""

    def __init__(self, reduction, vocab_size=64, hidden=32, num_heads=2):
        super().__init__()
        self.reduction = reduction
        self.num_heads = num_heads
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.pos_embed = torch.nn.Embedding(1024, hidden)
        self.qkv = torch.nn.Linear(hidden, 3 * hidden)
        self.out = torch.nn.Linear(hidden, hidden)
        self.head = torch.nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, labels, position_ids):
        batch, seq_len = input_ids.shape
        x = self.embed(input_ids) + self.pos_embed(position_ids)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def to_heads(t):
            return t.view(batch, seq_len, self.num_heads, -1).transpose(1, 2)

        attn = F.scaled_dot_product_attention(to_heads(q), to_heads(k), to_heads(v), is_causal=True)
        x = x + self.out(attn.transpose(1, 2).reshape(batch, seq_len, -1))
        logits = self.head(x)
        shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
        return F.cross_entropy(logits.view(-1, logits.size(-1)),
                               shift_labels.view(-1),
                               ignore_index=-100,
                               reduction=self.reduction)


class TestAutoSPCausalLoss(DistributedTest):
    """The loss and parameter updates of AutoSP match training on the full, unsharded sequence."""
    world_size = 4
    non_daemonic_procs = True

    @pytest.mark.sequential
    @pytest.mark.parametrize("zero_stage", [0, 1])
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_matches_full_sequence_reference(self, zero_stage, reduction):
        import copy
        import deepspeed
        import deepspeed.comm as dist
        from deepspeed.compile.passes.sp_compile import prepare_autosp_inputs

        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        sp_size = 2
        dp_size = self.world_size // sp_size
        dp_rank = dist.get_rank() // sp_size
        seq_len, batch, vocab_size, lr = 16, 2, 64, 0.1
        device = get_accelerator().current_device_name()

        def make_batch(step, rank):
            generator = torch.Generator().manual_seed(1000 * step + rank)
            input_ids = torch.randint(0, vocab_size, (batch, seq_len), generator=generator)
            labels = torch.randint(0, vocab_size, (batch, seq_len), generator=generator)
            # Ignore most of the second half so the two shards hold different numbers of valid tokens.
            labels[:, seq_len // 2 + 2:] = -100
            labels[0, -1] = 7
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, seq_len)
            return [t.to(device).contiguous() for t in (input_ids, labels, position_ids)]

        torch.manual_seed(0)
        reference = _CausalLMModel(reduction, vocab_size=vocab_size).to(device)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=lr)
        model = copy.deepcopy(reference)
        config = {
            "train_micro_batch_size_per_gpu": batch,
            "zero_optimization": {
                "stage": zero_stage
            },
            "zero_allow_untested_optimizer": True,
            # Clipping would hide a wrongly scaled gradient.
            "gradient_clipping": 0.0,
            "compile": {
                "deepcompile": True,
                "passes": ["autosp"]
            },
            "sequence_parallel_size": sp_size,
        }
        engine, _, _, _ = deepspeed.initialize(config=config,
                                               model=model,
                                               optimizer=torch.optim.SGD(model.parameters(), lr=lr))
        engine.compile()

        for step in range(2):
            # The reference trains on the full sequences of every DP group and averages over DP.
            reference_optimizer.zero_grad()
            for rank in range(dp_size):
                reference_loss = reference(*make_batch(step, rank))
                (reference_loss / dp_size).backward()
                if rank == dp_rank:
                    expected_loss = reference_loss.detach()
            reference_optimizer.step()

            loss = engine(*prepare_autosp_inputs(*make_batch(step, dp_rank))[:3])
            engine.backward(loss)
            engine.step()

            torch.testing.assert_close(loss.detach(), expected_loss, rtol=1e-4, atol=1e-5)
            for (name, param), expected in zip(engine.module.named_parameters(), reference.parameters()):
                torch.testing.assert_close(param.detach(),
                                           expected.detach(),
                                           rtol=1e-4,
                                           atol=1e-5,
                                           msg=lambda m, name=name: f"{name}: {m}")

        engine.destroy()


class TestAutoSPMeshRegistry(DistributedTest):
    """Re-registering the mesh an engine already uses keeps the existing process groups."""
    world_size = 4

    def test_same_mesh_is_reused(self, monkeypatch):
        import deepspeed.comm as dist
        from deepspeed.compile.custom_ops import sp_dp_registry

        monkeypatch.setattr(sp_dp_registry, "GROUP_REGISTRY", {})
        sp_dp_registry.populate_registry(2, 2)
        group = sp_dp_registry.get_group(dist.get_rank() // 2)
        sp_dp_registry.populate_registry(2, 2)

        assert sp_dp_registry.get_group(dist.get_rank() // 2) is group
        assert dist.get_world_size(group) == 2


# Plain pytest classes — distributed state is mocked where needed, and the remaining
# tests perform pure IR-level graph rewrites.


class TestAutoSPMeshValidation:
    """The SP/DP registry is process-global and read by compiled graphs at runtime."""

    def test_rejects_changed_mesh(self, monkeypatch):
        from deepspeed.compile.custom_ops import sp_dp_registry

        registry = {"SP_SIZE": 2, "DP_SIZE": 2, "is_reg": True}
        monkeypatch.setattr(sp_dp_registry, "GROUP_REGISTRY", registry)
        monkeypatch.setattr(sp_dp_registry.dist, "get_world_size", lambda *args, **kwargs: 4)

        with pytest.raises(RuntimeError, match="already initialized"):
            sp_dp_registry.populate_registry(4, 1)
        assert (sp_dp_registry.sp_size(), sp_dp_registry.dp_size()) == (2, 2)

    @pytest.mark.parametrize("sp_size, dp_size", [(2, 1), (1, 2)])
    def test_rejects_mesh_not_covering_world(self, monkeypatch, sp_size, dp_size):
        from deepspeed.compile.custom_ops import sp_dp_registry

        monkeypatch.setattr(sp_dp_registry, "GROUP_REGISTRY", {})
        monkeypatch.setattr(sp_dp_registry.dist, "get_world_size", lambda *args, **kwargs: 4)

        def fail_new_group(ranks):
            raise AssertionError(f"process group {ranks} created for an invalid mesh")

        monkeypatch.setattr(sp_dp_registry.dist, "new_group", fail_new_group)

        with pytest.raises(ValueError, match="must cover"):
            sp_dp_registry.populate_registry(sp_size, dp_size)
        assert not sp_dp_registry.is_setup()

    @pytest.mark.parametrize("sp_size, dp_size", [(0, 4), (-2, -2)])
    def test_rejects_non_positive_mesh_dimensions(self, monkeypatch, sp_size, dp_size):
        from deepspeed.compile.custom_ops import sp_dp_registry

        monkeypatch.setattr(sp_dp_registry, "GROUP_REGISTRY", {})
        monkeypatch.setattr(sp_dp_registry.dist, "get_world_size", lambda *args, **kwargs: 4)

        with pytest.raises(ValueError, match="must be positive"):
            sp_dp_registry.populate_registry(sp_size, dp_size)
        assert not sp_dp_registry.is_setup()


class TestAutoSPEngineBackward(DistributedTest):
    """AutoSP never initializes the ZeRO DeepCompile runtime, so backward must not enter its hooks."""
    world_size = 1

    def test_backward_skips_zero_deepcompile_hooks(self, monkeypatch):
        import deepspeed
        import deepspeed.compile.util as dc_util
        from unit.simple_model import SimpleModel, random_dataloader

        config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "zero_optimization": {
                "stage": 0
            },
            "compile": {
                "deepcompile": True,
                "passes": ["autosp"]
            },
        }
        model = SimpleModel(hidden_dim=8)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        # Emulate the state after engine.compile() installed the AutoSP backend, without compiling.
        engine._set_deepcompile_active(True)
        assert engine.is_deepcompile_active()

        def fail_native_handle():
            raise RuntimeError("AutoSP backward must not load the ZeRO DeepCompile runtime")

        monkeypatch.setattr(dc_util, "get_deepcompile_handle", fail_native_handle)
        monkeypatch.setattr(dc_util, "post_backward_hooks", [fail_native_handle])

        batch = next(
            iter(
                random_dataloader(model=engine, total_samples=1, hidden_dim=8, device=engine.device,
                                  dtype=torch.float)))
        loss = engine(batch[0], batch[1])
        engine.backward(loss)
        engine.step()


# Plain pytest classes — distributed state is mocked where needed, and the remaining
# tests perform pure IR-level graph rewrites.


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
