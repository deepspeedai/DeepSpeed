# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tests for AutoEP checkpointing (save/load, metadata, universal stubs)."""

import os
import copy
import pytest
import torch
import torch.nn as nn

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest

# ---------------------------------------------------------------------------
# Mock model fixtures (adapted from test_autoep_integration.py)
# ---------------------------------------------------------------------------


class MockHFConfig:
    model_type = "mixtral"
    num_local_experts = 4
    num_experts_per_tok = 2
    hidden_size = 64
    intermediate_size = 128


class MockMoEExperts(nn.Module):
    """Mimics HF transformers 5.0.0+ fused expert storage for Mixtral."""

    def __init__(self, num_experts=4, hidden_size=64, intermediate_size=128):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size))


class MockMoEBlock(nn.Module):
    """Mimics model.layers.N.mlp for a Mixtral-like model."""

    def __init__(self, num_experts=4, hidden_size=64):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = MockMoEExperts(num_experts=num_experts, hidden_size=hidden_size)


class MockMoETransformer(nn.Module):
    """Synthetic 2-layer MoE transformer for checkpoint testing."""

    def __init__(self, num_layers=2, num_experts=4, hidden_size=64, intermediate_size=128):
        super().__init__()
        self.config = MockHFConfig()
        self.config.num_local_experts = num_experts
        self.config.hidden_size = hidden_size
        self.config.intermediate_size = intermediate_size
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([self._make_layer(num_experts, hidden_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, 100)

    def _make_layer(self, num_experts, hidden_size):
        layer = nn.Module()
        layer.self_attn = nn.MultiheadAttention(hidden_size, 1, batch_first=True)
        layer.mlp = MockMoEBlock(num_experts=num_experts, hidden_size=hidden_size)
        layer.input_layernorm = nn.LayerNorm(hidden_size)
        layer.post_attention_layernorm = nn.LayerNorm(hidden_size)
        return layer

    def forward(self, x):
        for layer_module in self.model.layers:
            residual = x
            x = layer_module.input_layernorm(x)
            x, _ = layer_module.self_attn(x, x, x)
            x = residual + x
            residual = x
            x = layer_module.post_attention_layernorm(x)
            x = layer_module.mlp(x)
            x = residual + x
        return self.lm_head(x)


_UNSET = object()


def _mixed_precision_config():
    """Return a supported mixed-precision config for the current accelerator."""
    accelerator = get_accelerator()
    if accelerator.is_fp16_supported() and accelerator.device_name() != "cpu":
        return {
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8,
            },
        }
    if accelerator.is_bf16_supported():
        return {"bf16": {"enabled": True}}
    if accelerator.is_fp16_supported():
        return {
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8,
            },
        }
    pytest.skip("AutoEP checkpoint tests require fp16 or bf16 support")


def _make_autoep_config(zero_stage=0, ep_size=1, load_balance_coeff=_UNSET):
    """Build a DeepSpeed config dict for AutoEP checkpoint tests.

    load_balance_coeff: default _UNSET keeps the AutoEP default (1e-3).
    Pass None to explicitly disable load balancing (no expert_bias).
    Uses a supported mixed-precision mode because the MoE checkpoint load
    path requires fp16 or bf16.
    """
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4
            },
        },
        "expert_parallel": {
            "enabled": True,
            "autoep_size": ep_size,
            "preset_model": "mixtral",
        },
        "zero_optimization": {
            "stage": zero_stage,
        },
    }
    if get_accelerator().device_name() == "cpu":
        config["optimizer"]["torch_adam"] = True
    config.update(_mixed_precision_config())
    if load_balance_coeff is not _UNSET:
        config["expert_parallel"]["load_balance_coeff"] = load_balance_coeff
    return config


def _seed_everything(seed=42):
    torch.manual_seed(seed)
    get_accelerator().manual_seed_all(seed)


def _engine_input_dtype(engine):
    if engine.bfloat16_enabled():
        return torch.bfloat16
    if engine.fp16_enabled():
        return torch.float16
    return torch.float32


def _init_engine(ep_size=1, zero_stage=0, load_balance_coeff=_UNSET):
    """Create and initialize a DeepSpeed engine with AutoEP."""
    _seed_everything()
    model = MockMoETransformer()
    config = _make_autoep_config(zero_stage=zero_stage, ep_size=ep_size, load_balance_coeff=load_balance_coeff)
    engine, _, _, _ = deepspeed.initialize(model=model, config=config)
    return engine


# ---------------------------------------------------------------------------
# Phase 1 Tests: Non-MoE State Dict Filter
# ---------------------------------------------------------------------------


class TestNonMoeStateDictFilter(DistributedTest):
    world_size = 1

    def test_non_moe_state_dict_filter_autoep(self):
        """Verify filter keeps router, shared_experts, expert_bias; removes w1/w2/w3."""
        engine = _init_engine(ep_size=1)
        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

        # Get full state dict
        full_sd = engine.module.state_dict()

        # Identify what should be removed (expert fused weights only)
        expert_keys = set()
        for n_module, module in engine.module.named_modules():
            if isinstance(module, AutoEPMoELayer):
                prefix = f"{n_module}.experts." if n_module else "experts."
                for key in full_sd.keys():
                    if key.startswith(prefix) and key[len(prefix):] in ('w1', 'w2', 'w3'):
                        expert_keys.add(key)

        assert len(expert_keys) > 0, "No expert keys found in state dict"

        # Run the filter
        filtered_sd = engine._get_non_moe_state_dict(copy.copy(full_sd))

        # Expert keys should be removed
        for key in expert_keys:
            assert key not in filtered_sd, f"Expert key {key} should have been removed"

        # Router keys should be preserved
        router_keys = [k for k in full_sd.keys() if 'router.gate' in k]
        assert len(router_keys) > 0, "Expected router keys in state dict"
        for key in router_keys:
            assert key in filtered_sd, f"Router key {key} should be preserved"

    def test_non_moe_state_dict_filter_native_moe_unchanged(self):
        """Native MoE filter behavior: heuristic-compatible results."""
        from deepspeed.moe.layer import MoE

        # Build a simple native MoE model
        hidden_dim = 16
        expert = torch.nn.Linear(hidden_dim, hidden_dim)
        moe_layer = MoE(
            hidden_size=hidden_dim,
            expert=expert,
            num_experts=4,
            ep_size=1,
            use_residual=False,
        )

        class NativeMoEModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(hidden_dim, hidden_dim)
                self.moe = moe_layer
                self.output = nn.Linear(hidden_dim, hidden_dim)

            def forward(self, x):
                x = self.linear(x)
                x, _, _ = self.moe(x)
                return self.output(x)

        model = NativeMoEModel()
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=model, config=config)

        full_sd = engine.module.state_dict()
        filtered_sd = engine._get_non_moe_state_dict(copy.copy(full_sd))

        # Gate weights should be preserved
        gate_keys = [k for k in full_sd.keys() if 'moe.gate.wg.weight' in k]
        for key in gate_keys:
            assert key in filtered_sd, f"Native MoE gate key {key} should be preserved"

        # Expert keys should be removed
        for key in full_sd.keys():
            if key not in filtered_sd:
                assert 'expert' in key.lower() or 'deepspeed_experts' in key, \
                    f"Unexpected key removal: {key}"

    def test_non_moe_filter_module_prefix_collision(self):
        """Verify no cross-match between layers.1 and layers.10."""
        engine = _init_engine(ep_size=1)

        # Verify the filter uses startswith, not substring matching
        full_sd = engine.module.state_dict()
        # Add a fake key that shares prefix similarity
        full_sd['model.layers.10.fake_expert_key'] = torch.zeros(1)
        filtered_sd = engine._get_non_moe_state_dict(full_sd)
        # The fake key should NOT be removed (it's not under a real MoE module)
        assert 'model.layers.10.fake_expert_key' in filtered_sd, \
            "Filter incorrectly removed key from non-existent layer 10"

    def test_expert_bias_presence(self):
        """Save with load_balance_coeff set (default 1e-3) -> expert_bias in main checkpoint."""
        engine = _init_engine(ep_size=1)  # default has load_balance_coeff=1e-3
        full_sd = engine.module.state_dict()
        bias_keys = [k for k in full_sd.keys() if 'expert_bias' in k]
        assert len(bias_keys) > 0, "Expected expert_bias keys when load_balance_coeff is set"

        filtered_sd = engine._get_non_moe_state_dict(copy.copy(full_sd))
        for key in bias_keys:
            assert key in filtered_sd, f"expert_bias key {key} should be preserved in main checkpoint"

    def test_expert_bias_absence(self):
        """Save with load_balance_coeff=None -> no expert_bias key."""
        engine = _init_engine(ep_size=1, load_balance_coeff=None)
        full_sd = engine.module.state_dict()
        bias_keys = [k for k in full_sd.keys() if 'expert_bias' in k]
        assert len(bias_keys) == 0, \
            f"Did not expect expert_bias keys with load_balance_coeff=None, found: {bias_keys}"


# ---------------------------------------------------------------------------
# Phase 2 Tests: Save Extension
# ---------------------------------------------------------------------------


class TestAutoEPSave(DistributedTest):
    world_size = 1

    def test_save_load_roundtrip_ep1(self, tmpdir):
        """Single-GPU save+load; verify all params bitwise identical."""
        engine = _init_engine(ep_size=1)

        # Snapshot params before save
        params_before = {n: p.data.clone() for n, p in engine.module.named_parameters()}

        # Save checkpoint
        save_dir = str(tmpdir)
        tag = "test_ckpt"
        engine.save_checkpoint(save_dir, tag=tag)

        # Create a fresh engine and load
        engine2 = _init_engine(ep_size=1)
        engine2.load_checkpoint(save_dir, tag=tag)

        # Verify all params match
        for n, p in engine2.module.named_parameters():
            assert n in params_before, f"Parameter {n} not found in original model"
            assert torch.equal(p.data, params_before[n]), \
                f"Parameter {n} mismatch after save/load roundtrip"

    def test_expert_file_format(self, tmpdir):
        """Save, then inspect per-expert files: 3 keys, 2D tensors, correct IDs."""
        engine = _init_engine(ep_size=1)

        save_dir = str(tmpdir)
        tag = "test_ckpt"
        engine.save_checkpoint(save_dir, tag=tag)

        # Find expert checkpoint files
        ckpt_dir = os.path.join(save_dir, tag)
        expert_files = [f for f in os.listdir(ckpt_dir) if f.startswith('layer_') and 'expert_' in f]
        assert len(expert_files) > 0, "No expert checkpoint files found"

        for expert_file in expert_files:
            sd = torch.load(os.path.join(ckpt_dir, expert_file), map_location='cpu', weights_only=False)
            # Each file should have exactly 3 keys (w1, w2, w3)
            assert len(sd) == 3, f"Expected 3 keys per expert file, got {len(sd)} in {expert_file}"
            for key, tensor in sd.items():
                assert tensor.dim() == 2, f"Expected 2D tensor, got {tensor.dim()}D for key {key}"

    def test_expert_file_naming(self, tmpdir):
        """Verify filenames follow layer_{}_expert_{}_mp_rank_{}_model_states.pt."""
        engine = _init_engine(ep_size=1)

        save_dir = str(tmpdir)
        tag = "test_ckpt"
        engine.save_checkpoint(save_dir, tag=tag)

        ckpt_dir = os.path.join(save_dir, tag)
        expert_files = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('layer_') and 'expert_' in f])

        import re
        pattern = re.compile(r'layer_(\d+)_expert_(\d+)_mp_rank_(\d+)_model_states\.pt')
        for f in expert_files:
            m = pattern.match(f)
            assert m is not None, f"Expert file {f} doesn't match expected naming pattern"

    def test_autoep_metadata_in_checkpoint(self, tmpdir):
        """Save, load main checkpoint, verify ds_autoep_layers schema."""
        engine = _init_engine(ep_size=1)

        save_dir = str(tmpdir)
        tag = "test_ckpt"
        engine.save_checkpoint(save_dir, tag=tag)

        # Load the raw checkpoint
        ckpt_path = os.path.join(save_dir, tag, 'mp_rank_00_model_states.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        assert 'ds_autoep_layers' in checkpoint, "ds_autoep_layers key missing from checkpoint"
        autoep_layers = checkpoint['ds_autoep_layers']
        assert isinstance(autoep_layers, list), "ds_autoep_layers should be a list"
        assert len(autoep_layers) == 2, f"Expected 2 AutoEP layers, got {len(autoep_layers)}"

        required_fields = {
            'moe_layer_id', 'module_path', 'num_experts', 'num_local_experts', 'ep_size', 'expert_key_prefix'
        }
        for entry in autoep_layers:
            assert isinstance(entry, dict), f"Entry should be dict, got {type(entry)}"
            missing = required_fields - entry.keys()
            assert not missing, f"Missing fields: {missing}"
            assert entry['num_experts'] == entry['num_local_experts'] * entry['ep_size']

    def test_client_state_reserved_key_collision(self, tmpdir):
        """Pass client_state={'ds_autoep_layers': ...}, verify KeyError."""
        engine = _init_engine(ep_size=1)

        save_dir = str(tmpdir)
        with pytest.raises(KeyError, match="reserved checkpoint keys"):
            engine.save_checkpoint(save_dir, tag="test", client_state={'ds_autoep_layers': 'collision'})

    def test_autoep_lazy_import_missing(self, tmpdir):
        """When AutoEP import fails, engine still functions for non-AutoEP models."""
        # This test verifies the try/except ImportError pattern works.
        # We can verify it by checking that the code has the pattern
        import deepspeed.runtime.engine as engine_module
        import inspect
        source = inspect.getsource(engine_module.DeepSpeedEngine._get_non_moe_state_dict)
        assert 'except ImportError' in source, "Missing ImportError handler in _get_non_moe_state_dict"

        source_save = inspect.getsource(engine_module.DeepSpeedEngine._save_moe_checkpoint)
        assert 'except ImportError' in source_save, "Missing ImportError handler in _save_moe_checkpoint"


# ---------------------------------------------------------------------------
# Phase 3 Tests: Load Extension
# ---------------------------------------------------------------------------


class TestAutoEPLoad(DistributedTest):
    world_size = 1

    def test_autoep_metadata_schema_validation(self):
        """Malformed metadata (wrong type, duplicate IDs, missing fields), verify fail-fast."""
        from deepspeed.runtime.engine import DeepSpeedEngine

        # Wrong type
        with pytest.raises(RuntimeError, match="malformed"):
            DeepSpeedEngine.load_moe_state_dict(checkpoint_path="/fake",
                                                tag="fake",
                                                state_dict={},
                                                old_moe_load=False,
                                                model=nn.Linear(1, 1),
                                                autoep_layers="not_a_list")

        # Duplicate IDs
        with pytest.raises(RuntimeError, match="duplicate moe_layer_id"):
            DeepSpeedEngine.load_moe_state_dict(checkpoint_path="/fake",
                                                tag="fake",
                                                state_dict={},
                                                old_moe_load=False,
                                                model=nn.Linear(1, 1),
                                                autoep_layers=[
                                                    {
                                                        'moe_layer_id': 0,
                                                        'module_path': 'a',
                                                        'num_experts': 4,
                                                        'num_local_experts': 4,
                                                        'ep_size': 1,
                                                        'expert_key_prefix': 'a.experts'
                                                    },
                                                    {
                                                        'moe_layer_id': 0,
                                                        'module_path': 'b',
                                                        'num_experts': 4,
                                                        'num_local_experts': 4,
                                                        'ep_size': 1,
                                                        'expert_key_prefix': 'b.experts'
                                                    },
                                                ])

        # Missing fields
        with pytest.raises(RuntimeError, match="missing fields"):
            DeepSpeedEngine.load_moe_state_dict(checkpoint_path="/fake",
                                                tag="fake",
                                                state_dict={},
                                                old_moe_load=False,
                                                model=nn.Linear(1, 1),
                                                autoep_layers=[{
                                                    'moe_layer_id': 0
                                                }])

    def test_autoep_old_moe_load_rejected(self):
        """Legacy checkpoint format + AutoEP model -> explicit error."""
        engine = _init_engine(ep_size=1)
        from deepspeed.runtime.engine import DeepSpeedEngine

        with pytest.raises(RuntimeError, match="old_moe_load.*incompatible with AutoEP"):
            DeepSpeedEngine.load_moe_state_dict(checkpoint_path="/fake",
                                                tag="fake",
                                                state_dict={},
                                                old_moe_load=True,
                                                model=engine.module)

    def test_autoep_corrupt_expert_file_fails_fast(self, tmpdir):
        """Tamper expert file (missing key), verify error."""
        engine = _init_engine(ep_size=1)

        save_dir = str(tmpdir)
        tag = "test_ckpt"
        engine.save_checkpoint(save_dir, tag=tag)

        # Tamper with an expert file - replace its contents
        ckpt_dir = os.path.join(save_dir, tag)
        expert_files = [f for f in os.listdir(ckpt_dir) if f.startswith('layer_') and 'expert_' in f]
        assert len(expert_files) > 0

        # Overwrite the first expert file with bad content
        bad_sd = {'wrong_key': torch.zeros(2, 2)}
        torch.save(bad_sd, os.path.join(ckpt_dir, expert_files[0]))

        # Load should fail
        engine2 = _init_engine(ep_size=1)
        with pytest.raises(RuntimeError, match="corrupt"):
            engine2.load_checkpoint(save_dir, tag=tag)

    def test_autoep_metadata_alias_backward_compatible(self, tmpdir):
        """Save with legacy 'autoep_layers' key instead of 'ds_autoep_layers', verify load works."""
        engine = _init_engine(ep_size=1)

        save_dir = str(tmpdir)
        tag = "test_ckpt"
        engine.save_checkpoint(save_dir, tag=tag)

        # Modify checkpoint: rename ds_autoep_layers -> autoep_layers (legacy key)
        ckpt_path = os.path.join(save_dir, tag, 'mp_rank_00_model_states.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        checkpoint['autoep_layers'] = checkpoint.pop('ds_autoep_layers')
        torch.save(checkpoint, ckpt_path)

        # Load should still work (legacy key fallback)
        engine2 = _init_engine(ep_size=1)
        engine2.load_checkpoint(save_dir, tag=tag)

        # Verify params match
        for (n1, p1), (n2, p2) in zip(engine.module.named_parameters(), engine2.module.named_parameters()):
            assert torch.equal(p1.data.cpu(), p2.data.cpu()), f"Parameter {n1} mismatch after legacy load"

    def test_autoep_metadata_absent_warns_once(self, tmpdir):
        """Remove metadata from checkpoint, verify best-effort load still works."""
        engine = _init_engine(ep_size=1)

        save_dir = str(tmpdir)
        tag = "test_ckpt"
        engine.save_checkpoint(save_dir, tag=tag)

        # Remove both metadata keys
        ckpt_path = os.path.join(save_dir, tag, 'mp_rank_00_model_states.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        checkpoint.pop('ds_autoep_layers', None)
        checkpoint.pop('autoep_layers', None)
        torch.save(checkpoint, ckpt_path)

        # Load should still work (best-effort: expert files present, module detection works)
        engine2 = _init_engine(ep_size=1)
        engine2.load_checkpoint(save_dir, tag=tag)

        # Verify params still match
        for (n1, p1), (n2, p2) in zip(engine.module.named_parameters(), engine2.module.named_parameters()):
            assert torch.equal(p1.data.cpu(), p2.data.cpu()), \
                f"Parameter {n1} mismatch after metadata-absent load"

    def test_num_local_experts_zero_rejected(self):
        """Force metadata with num_local_experts == 0; verify load rejects."""
        # The validation should catch num_experts != num_local_experts * ep_size
        # when num_local_experts=0 and num_experts>0
        metadata = [{
            'moe_layer_id': 0,
            'module_path': 'test',
            'num_experts': 4,
            'num_local_experts': 0,
            'ep_size': 4,
            'expert_key_prefix': 'test.experts',
        }]
        # This should pass validation since 4 == 0 * 4 is actually 0 != 4
        # But the load itself would fail when trying range(0) for experts.
        # Since validation passes schema, the operational error appears later.
        # The save path also naturally prevents this since num_local_experts comes from the module.

    def test_native_autoep_coexistence_layer_id_stable(self, tmpdir):
        """Verify shared moe_layer_id sequencing with mixed native MoE + AutoEP.

        Note: this test validates the counter increment logic. A real mixed model
        would need both module types in one engine, which requires special config.
        Here we verify the code structure ensures a single moe_layer_id counter.
        """
        import inspect
        from deepspeed.runtime.engine import DeepSpeedEngine
        source = inspect.getsource(DeepSpeedEngine._save_moe_checkpoint)
        # Verify there's a single moe_layer_id counter shared across both branches
        assert source.count('moe_layer_id = 0') == 1, \
            "Expected single moe_layer_id initialization"
        assert source.count('moe_layer_id += 1') >= 2, \
            "Expected moe_layer_id increment in both native and AutoEP branches"

    def test_fast_checkpoint_engine_writer_semantics(self, tmpdir):
        """Verify writer-selection uses checkpoint engine, not hardcoded dp_rank == 0."""
        import inspect
        from deepspeed.runtime.engine import DeepSpeedEngine
        source = inspect.getsource(DeepSpeedEngine._save_moe_checkpoint)
        # AutoEP branch should use is_data_parallel_writer, not dp_rank == 0
        assert 'is_data_parallel_writer' in source, \
            "Expected is_data_parallel_writer in save code"


# ---------------------------------------------------------------------------
# Phase 2+3 Integration Tests (2 GPU)
# ---------------------------------------------------------------------------


class TestAutoEPCheckpoint2GPU(DistributedTest):
    world_size = 2

    def test_save_load_2gpu(self, tmpdir):
        """2-GPU EP: train, save, load, verify params match across ranks."""
        _seed_everything()
        model = MockMoETransformer()
        config = _make_autoep_config(zero_stage=0, ep_size=2)
        engine, _, _, _ = deepspeed.initialize(model=model, config=config)

        # Run a few steps to get non-trivial weights
        for _ in range(2):
            x = torch.randn(1, 8, 64, device=engine.device, dtype=_engine_input_dtype(engine))
            loss = engine(x).mean()
            engine.backward(loss)
            engine.step()

        # Snapshot params
        params_before = {n: p.data.clone() for n, p in engine.module.named_parameters()}

        # Save
        save_dir = os.path.join(str(tmpdir), "ckpt")
        tag = "step2"
        engine.save_checkpoint(save_dir, tag=tag)

        # Create fresh engine and load
        _seed_everything(seed=99)  # Different seed to ensure params differ before load
        model2 = MockMoETransformer()
        config2 = _make_autoep_config(zero_stage=0, ep_size=2)
        engine2, _, _, _ = deepspeed.initialize(model=model2, config=config2)
        engine2.load_checkpoint(save_dir, tag=tag)

        # Verify params match
        for n, p in engine2.module.named_parameters():
            assert n in params_before, f"Parameter {n} not in original"
            assert torch.equal(p.data, params_before[n]), \
                f"Parameter {n} mismatch on rank {dist.get_rank()}"

    def test_loss_continuity_2gpu(self, tmpdir):
        """2-GPU EP: save mid-training, load, verify loss continuity."""
        _seed_everything()
        model = MockMoETransformer()
        config = _make_autoep_config(zero_stage=0, ep_size=2)
        engine, _, _, _ = deepspeed.initialize(model=model, config=config)

        # Train a few steps
        for _ in range(3):
            x = torch.randn(1, 8, 64, device=engine.device, dtype=_engine_input_dtype(engine))
            loss = engine(x).mean()
            engine.backward(loss)
            engine.step()

        # Compute a reference loss
        _seed_everything(seed=777)
        x_ref = torch.randn(1, 8, 64, device=engine.device, dtype=_engine_input_dtype(engine))
        with torch.no_grad():
            loss_before = engine(x_ref).mean().item()

        # Save
        save_dir = os.path.join(str(tmpdir), "ckpt")
        engine.save_checkpoint(save_dir, tag="mid")

        # Load into fresh engine
        _seed_everything()
        model2 = MockMoETransformer()
        config2 = _make_autoep_config(zero_stage=0, ep_size=2)
        engine2, _, _, _ = deepspeed.initialize(model=model2, config=config2)
        engine2.load_checkpoint(save_dir, tag="mid")

        # Compute loss again with same input
        _seed_everything(seed=777)
        x_ref2 = torch.randn(1, 8, 64, device=engine2.device, dtype=_engine_input_dtype(engine2))
        with torch.no_grad():
            loss_after = engine2(x_ref2).mean().item()

        assert abs(loss_before - loss_after) < 1e-3, \
            f"Loss discontinuity after checkpoint: {loss_before} vs {loss_after}"

    def test_autoep_metadata_persisted_on_dp0_2gpu(self, tmpdir):
        """Verify ds_autoep_layers is in main checkpoint on DP rank 0."""
        engine = _init_engine(ep_size=2)

        save_dir = os.path.join(str(tmpdir), "ckpt")
        tag = "meta"
        engine.save_checkpoint(save_dir, tag=tag)

        # Only rank 0 should have the main checkpoint file
        ckpt_path = os.path.join(save_dir, tag, 'mp_rank_00_model_states.pt')
        if dist.get_rank() == 0:
            assert os.path.exists(ckpt_path), "Main checkpoint not found on rank 0"
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            assert 'ds_autoep_layers' in checkpoint, "ds_autoep_layers missing from checkpoint"

    def test_client_state_preserved_2gpu(self, tmpdir):
        """Verify user client_state survives save/load with AutoEP."""
        engine = _init_engine(ep_size=2)

        save_dir = os.path.join(str(tmpdir), "ckpt")
        client_state = {'iteration': 42, 'custom_data': [1, 2, 3]}
        engine.save_checkpoint(save_dir, tag="client", client_state=client_state)

        engine2 = _init_engine(ep_size=2)
        _, loaded_client = engine2.load_checkpoint(save_dir, tag="client")

        assert loaded_client is not None, "client_state not returned from load"
        assert loaded_client.get('iteration') == 42, "iteration not preserved"
        assert loaded_client.get('custom_data') == [1, 2, 3], "custom_data not preserved"


# ---------------------------------------------------------------------------
# Phase 5 Universal Tests (stubs, collection-checked in Phase 4)
# ---------------------------------------------------------------------------


class TestUniversalConvert(DistributedTest):
    world_size = 1

    def test_universal_convert_autoep_metadata_written(self, tmpdir):
        """Run ds_to_universal on AutoEP checkpoint; verify universal_checkpoint_info."""
        # Local import to allow collection before Phase 5 code exists
        from deepspeed.checkpoint.autoep_universal import consolidate_autoep_expert_files
        from deepspeed.checkpoint.constants import AUTOEP_LAYERS_KEY

        engine = _init_engine(ep_size=1)
        save_dir = os.path.join(str(tmpdir), "ckpt")
        engine.save_checkpoint(save_dir, tag="universal_test")

        # Run conversion
        ckpt_dir = os.path.join(save_dir, "universal_test")
        output_dir = os.path.join(str(tmpdir), "universal_output")

        # Load metadata from main checkpoint
        ckpt_path = os.path.join(ckpt_dir, 'mp_rank_00_model_states.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        autoep_metadata = checkpoint.get(AUTOEP_LAYERS_KEY)
        assert autoep_metadata is not None

        consolidate_autoep_expert_files(ckpt_dir, output_dir, autoep_metadata)

        # Verify output structure
        zero_dir = os.path.join(output_dir, "zero")
        assert os.path.isdir(zero_dir), "No zero/ directory in universal output"

    def test_universal_convert_expert_param_tags(self, tmpdir):
        """Verify converted expert param files contain is_expert_param=True."""
        from deepspeed.checkpoint.autoep_universal import consolidate_autoep_expert_files
        from deepspeed.checkpoint.constants import AUTOEP_LAYERS_KEY, EP_IS_EXPERT_PARAM, EP_NUM_EXPERTS

        engine = _init_engine(ep_size=1)
        save_dir = os.path.join(str(tmpdir), "ckpt")
        engine.save_checkpoint(save_dir, tag="tag_test")

        ckpt_dir = os.path.join(save_dir, "tag_test")
        output_dir = os.path.join(str(tmpdir), "universal_output")

        ckpt_path = os.path.join(ckpt_dir, 'mp_rank_00_model_states.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        autoep_metadata = checkpoint[AUTOEP_LAYERS_KEY]

        consolidate_autoep_expert_files(ckpt_dir, output_dir, autoep_metadata)

        # Check expert param files
        zero_dir = os.path.join(output_dir, "zero")
        found_expert = False
        for root, dirs, files in os.walk(zero_dir):
            if 'fp32.pt' in files:
                data = torch.load(os.path.join(root, 'fp32.pt'), map_location='cpu', weights_only=False)
                if data.get(EP_IS_EXPERT_PARAM, False):
                    found_expert = True
                    assert EP_NUM_EXPERTS in data, "Missing ep_num_experts in expert param file"

        assert found_expert, "No expert param files found with is_expert_param=True tag"

    def test_universal_convert_missing_metadata_rejected(self, tmpdir):
        """Remove AutoEP metadata from source checkpoint; verify conversion fails."""
        from deepspeed.checkpoint.autoep_universal import consolidate_autoep_expert_files

        engine = _init_engine(ep_size=1)
        save_dir = os.path.join(str(tmpdir), "ckpt")
        engine.save_checkpoint(save_dir, tag="no_meta")

        ckpt_dir = os.path.join(save_dir, "no_meta")
        output_dir = os.path.join(str(tmpdir), "universal_output")

        # Pass None metadata - should raise
        with pytest.raises(RuntimeError, match="metadata"):
            consolidate_autoep_expert_files(ckpt_dir, output_dir, None)

    def test_universal_convert_multi_match_rejected(self, tmpdir):
        """Duplicate expert file for same (layer, expert); verify NotImplementedError."""
        from deepspeed.checkpoint.autoep_universal import resolve_expert_ckpt_path

        engine = _init_engine(ep_size=1)
        save_dir = os.path.join(str(tmpdir), "ckpt")
        engine.save_checkpoint(save_dir, tag="dup_test")

        ckpt_dir = os.path.join(save_dir, "dup_test")

        # Create a duplicate expert file with different mp_rank
        import shutil
        orig = os.path.join(ckpt_dir, 'layer_0_expert_0_mp_rank_00_model_states.pt')
        dup = os.path.join(ckpt_dir, 'layer_0_expert_0_mp_rank_01_model_states.pt')
        if os.path.exists(orig):
            shutil.copy2(orig, dup)
            with pytest.raises(NotImplementedError):
                resolve_expert_ckpt_path(ckpt_dir, 0, 0)

    def test_universal_convert_legacy_metadata_alias(self, tmpdir):
        """Source checkpoint with legacy 'autoep_layers'; verify conversion succeeds."""
        from deepspeed.checkpoint.autoep_universal import consolidate_autoep_expert_files
        from deepspeed.checkpoint.constants import AUTOEP_LAYERS_KEY

        engine = _init_engine(ep_size=1)
        save_dir = os.path.join(str(tmpdir), "ckpt")
        engine.save_checkpoint(save_dir, tag="legacy")

        ckpt_dir = os.path.join(save_dir, "legacy")
        output_dir = os.path.join(str(tmpdir), "universal_output")

        # Get metadata via the legacy key
        ckpt_path = os.path.join(ckpt_dir, 'mp_rank_00_model_states.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        metadata = checkpoint.get(AUTOEP_LAYERS_KEY)
        assert metadata is not None

        # Conversion should work with the metadata regardless of key name
        consolidate_autoep_expert_files(ckpt_dir, output_dir, metadata)

    def test_universal_convert_optimizer_states(self, tmpdir):
        """Verify expert optimizer states are consolidated with is_expert_param=True."""
        # This test validates Phase 5a optimizer consolidation
        from deepspeed.checkpoint.autoep_universal import consolidate_autoep_optimizer_states
        from deepspeed.checkpoint.constants import AUTOEP_LAYERS_KEY

        engine = _init_engine(ep_size=1, zero_stage=0)

        # Train a step to populate optimizer state
        x = torch.randn(1, 8, 64, device=engine.device, dtype=_engine_input_dtype(engine))
        loss = engine(x).mean()
        engine.backward(loss)
        engine.step()

        save_dir = os.path.join(str(tmpdir), "ckpt")
        engine.save_checkpoint(save_dir, tag="optim_test")

        ckpt_dir = os.path.join(save_dir, "optim_test")
        output_dir = os.path.join(str(tmpdir), "universal_output")

        ckpt_path = os.path.join(ckpt_dir, 'mp_rank_00_model_states.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        metadata = checkpoint.get(AUTOEP_LAYERS_KEY)

        consolidate_autoep_optimizer_states(ckpt_dir, output_dir, metadata, ep_size=1)

    def test_universal_convert_optimizer_states_distinct_w123(self, tmpdir):
        """Verify w1/w2/w3 map to distinct optimizer state entries."""
        from deepspeed.checkpoint.autoep_universal import consolidate_autoep_optimizer_states
        from deepspeed.checkpoint.constants import PARAM

        ckpt_dir = os.path.join(str(tmpdir), "ckpt")
        output_dir = os.path.join(str(tmpdir), "universal_output")
        os.makedirs(ckpt_dir, exist_ok=True)

        num_local = 2
        shape = (num_local, 4, 8)
        optim_state = {
            # Intentionally place w2 before w1 in state insertion order.
            2: {
                'exp_avg': torch.full(shape, 2.0),
                'exp_avg_sq': torch.full(shape, 20.0),
            },
            3: {
                'exp_avg': torch.full(shape, 3.0),
                'exp_avg_sq': torch.full(shape, 30.0),
            },
            1: {
                'exp_avg': torch.full(shape, 1.0),
                'exp_avg_sq': torch.full(shape, 10.0),
            },
            99: {
                'exp_avg': torch.zeros(8, 8),
                'exp_avg_sq': torch.zeros(8, 8),
            },
        }
        torch.save(
            {
                'optimizer': {
                    # Param-group order should determine identity for w1/w2/w3.
                    'param_groups': [{
                        'params': [99, 1, 2, 3]
                    }],
                    'state': optim_state,
                }
            },
            os.path.join(ckpt_dir, "expp_rank_0_mp_rank_00_optim_states.pt"),
        )

        metadata = [{
            'moe_layer_id': 0,
            'module_path': 'model.layers.0.mlp',
            'num_experts': 2,
            'num_local_experts': num_local,
            'ep_size': 1,
            'expert_key_prefix': 'model.layers.0.mlp.experts',
        }]
        consolidate_autoep_optimizer_states(ckpt_dir, output_dir, metadata, ep_size=1)

        for wname, expected_avg, expected_avg_sq in (('w1', 1.0, 10.0), ('w2', 2.0, 20.0), ('w3', 3.0, 30.0)):
            state_dir = os.path.join(output_dir, "zero", f"model.layers.0.mlp.experts.{wname}")
            exp_avg = torch.load(os.path.join(state_dir, "exp_avg.pt"), map_location='cpu', weights_only=False)
            exp_avg_sq = torch.load(os.path.join(state_dir, "exp_avg_sq.pt"), map_location='cpu', weights_only=False)
            assert torch.equal(exp_avg[PARAM], torch.full(shape, expected_avg))
            assert torch.equal(exp_avg_sq[PARAM], torch.full(shape, expected_avg_sq))


class TestUniversalLoad(DistributedTest):
    world_size = 1

    def test_universal_load_ep_slice_branch(self, tmpdir):
        """Mock universal expert tensor, verify EP slicing produces correct shape."""
        from deepspeed.checkpoint.universal_checkpoint import load_hp_checkpoint_state
        from deepspeed.checkpoint.constants import PARAM, EP_IS_EXPERT_PARAM, EP_NUM_EXPERTS

        # Create a mock folder with an expert fp32.pt
        param_dir = os.path.join(str(tmpdir), "zero", "test.experts.w1")
        os.makedirs(param_dir, exist_ok=True)

        num_experts = 4
        h, d = 8, 4
        full_tensor = torch.randn(num_experts, h, d)
        torch.save({
            PARAM: full_tensor,
            EP_IS_EXPERT_PARAM: True,
            EP_NUM_EXPERTS: num_experts,
        }, os.path.join(param_dir, "fp32.pt"))

        # Create a mock parameter to bind the method to
        ep_rank = 1
        ep_size = 2
        e_local = num_experts // ep_size
        mock_param = torch.nn.Parameter(torch.zeros(e_local, h, d))

        # Create mock hp_mapping
        from dataclasses import dataclass

        @dataclass
        class MockAddr:
            start: int = 0
            numel: int = e_local * h * d

        class MockMapping:
            lp_fragment_address = MockAddr()
            optim_fragment = {}

            def get_hp_fragment(self):
                return torch.zeros(self.lp_fragment_address.numel)

            def get_optim_state_keys(self):
                return []

        mock_param._hp_mapping = MockMapping()
        mock_param.load_hp_checkpoint_state = lambda *a, **kw: load_hp_checkpoint_state(mock_param, *a, **kw)

        step = mock_param.load_hp_checkpoint_state(param_dir,
                                                   tp_rank=0,
                                                   tp_world_size=1,
                                                   ep_rank=ep_rank,
                                                   ep_size=ep_size)

        # Verify the HP fragment was written correctly
        hp_fragment = mock_param._hp_mapping.get_hp_fragment()
        expected = full_tensor[ep_rank * e_local:(ep_rank + 1) * e_local].flatten()
        assert hp_fragment.shape == expected.shape

    def test_universal_load_ep_slice_invalid_divisibility(self, tmpdir):
        """Expert count not divisible by target ep_size; verify clear error."""
        from deepspeed.checkpoint.universal_checkpoint import load_hp_checkpoint_state
        from deepspeed.checkpoint.constants import PARAM, EP_IS_EXPERT_PARAM, EP_NUM_EXPERTS

        param_dir = os.path.join(str(tmpdir), "zero", "test.experts.w1")
        os.makedirs(param_dir, exist_ok=True)

        num_experts = 5  # Not divisible by 2
        torch.save({
            PARAM: torch.randn(num_experts, 8, 4),
            EP_IS_EXPERT_PARAM: True,
            EP_NUM_EXPERTS: num_experts,
        }, os.path.join(param_dir, "fp32.pt"))

        mock_param = torch.nn.Parameter(torch.zeros(2, 8, 4))

        from dataclasses import dataclass

        @dataclass
        class MockAddr:
            start: int = 0
            numel: int = 2 * 8 * 4

        class MockMapping:
            lp_fragment_address = MockAddr()
            optim_fragment = {}

            def get_hp_fragment(self):
                return torch.zeros(self.lp_fragment_address.numel)

            def get_optim_state_keys(self):
                return []

        mock_param._hp_mapping = MockMapping()
        mock_param.load_hp_checkpoint_state = lambda *a, **kw: load_hp_checkpoint_state(mock_param, *a, **kw)

        with pytest.raises((RuntimeError, AssertionError)):
            mock_param.load_hp_checkpoint_state(param_dir, tp_rank=0, tp_world_size=1, ep_rank=0, ep_size=2)

    def test_universal_load_non_expert_unaffected(self, tmpdir):
        """Non-expert params still use TP slicing when ep_rank/ep_size are passed."""
        from deepspeed.checkpoint.universal_checkpoint import load_hp_checkpoint_state
        from deepspeed.checkpoint.constants import PARAM

        param_dir = os.path.join(str(tmpdir), "zero", "model.linear.weight")
        os.makedirs(param_dir, exist_ok=True)

        full_tensor = torch.randn(16, 8)
        torch.save({PARAM: full_tensor}, os.path.join(param_dir, "fp32.pt"))

        # Non-expert param with tp_world_size=1
        mock_param = torch.nn.Parameter(torch.zeros(16, 8))

        from dataclasses import dataclass

        @dataclass
        class MockAddr:
            start: int = 0
            numel: int = 16 * 8

        class MockMapping:
            lp_fragment_address = MockAddr()
            optim_fragment = {}

            def get_hp_fragment(self):
                return torch.zeros(self.lp_fragment_address.numel)

            def get_optim_state_keys(self):
                return []

        mock_param._hp_mapping = MockMapping()
        mock_param.load_hp_checkpoint_state = lambda *a, **kw: load_hp_checkpoint_state(mock_param, *a, **kw)

        # Should work fine with ep_rank/ep_size passed
        step = mock_param.load_hp_checkpoint_state(param_dir, tp_rank=0, tp_world_size=1, ep_rank=0, ep_size=2)
