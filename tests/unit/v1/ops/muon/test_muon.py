# Copyright (c) 2025 Peng Du and Zhipeng Wang
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
import deepspeed.comm as dist
import torch
import pytest

from unit.common import DistributedTest
from unit.simple_model import SimpleModel
from deepspeed.accelerator import get_accelerator
if torch.half not in get_accelerator().supported_dtypes():
    pytest.skip(f"fp16 not supported, valid dtype: {get_accelerator().supported_dtypes()}", allow_module_level=True)

# 'optimizer_type, zero_stage, lr, hidden_dim, nlayer, offload_optimizer, save_muon_momentum_buffer_in_memory'

muon_configs = []
for optimizer_name in ['muon', 'adam']:
    for stage in [1, 2, 3]:
        for lr in [0.01, 0.05]:
            for model_dim in [32, 128]:
                for nlayer in [5, 10]:
                    for offload_optimizer in [True, False]:
                        for save_in_mem in ([True, False] if stage == 3 else [False]):
                            muon_configs.append(
                                [optimizer_name, stage, lr, model_dim, nlayer, offload_optimizer, save_in_mem])


@pytest.mark.parametrize(
    'optimizer_type, zero_stage, lr, hidden_dim, nlayer, offload_optimizer, save_muon_momentum_buffer_in_memory',
    muon_configs)
class TestMuonConfigs(DistributedTest):

    def test(self, optimizer_type, zero_stage, lr, hidden_dim, nlayer, offload_optimizer,
             save_muon_momentum_buffer_in_memory):
        optimizer_params = {"lr": lr}
        batch_size = 8
        config_dict = {
            "train_batch_size": batch_size,
            "optimizer": {
                "type": optimizer_type,
                "params": optimizer_params
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "reduce_scatter": False,
                "save_muon_momentum_buffer_in_memory": save_muon_momentum_buffer_in_memory,
            },
        }
        if offload_optimizer:
            config_dict["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }

        # Perform a few training steps to ensure the optimizer works correctly

        model = SimpleModel(hidden_dim=hidden_dim, nlayers=nlayer)
        initial_params = [p.clone().cpu() for p in model.parameters()]
        engine, optimizer, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
            dist_init_required=False,
        )
        assert optimizer_type in optimizer.optimizer.__class__.__name__.lower(
        ), f"Expected optimizer type {optimizer_type}, got {optimizer.optimizer.__class__.__name__}"
        steps = 5
        for _ in range(steps):
            # Random inputs: (batch_size, hidden_dim)
            x = torch.randn(batch_size, hidden_dim, device=engine.device, dtype=torch.half)
            # Random class labels: (batch_size,)
            y = torch.randint(0, hidden_dim, (batch_size, ), device=engine.device)
            # Forward + loss
            loss = engine(x, y)
            # Backward
            engine.backward(loss)
            engine.step()

        # Verify that parameters have been updated
        after_training = [p.clone().cpu() for p in model.parameters()]
        for initial, final in zip(initial_params, after_training):
            assert not torch.equal(initial.cpu(), final.cpu()), "Parameters should have been updated during training"


class TestGramNewtonSchulz(DistributedTest):
    """Test Gram Newton-Schulz integration with Muon optimizer."""

    world_size = 2
    reuse_dist_env = True

    @pytest.mark.parametrize('ns_method', ['gram', 'standard'])
    @pytest.mark.parametrize('zero_stage', [1, 2])
    def test_ns_method_training(self, ns_method, zero_stage):
        """Verify both ns_method values work end-to-end with DeepSpeed."""
        hidden_dim = 64
        batch_size = 8
        config_dict = {
            "train_batch_size": batch_size,
            "optimizer": {
                "type": "muon",
                "params": {
                    "lr": 0.01,
                    "ns_method": ns_method,
                }
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
            },
            "zero_optimization": {
                "stage": zero_stage,
                "reduce_scatter": False,
            },
        }

        model = SimpleModel(hidden_dim=hidden_dim, nlayers=3)
        initial_params = [p.clone().cpu() for p in model.parameters()]
        engine, optimizer, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
            dist_init_required=False,
        )

        for _ in range(3):
            x = torch.randn(batch_size, hidden_dim, device=engine.device, dtype=torch.half)
            y = torch.randint(0, hidden_dim, (batch_size, ), device=engine.device)
            loss = engine(x, y)
            engine.backward(loss)
            engine.step()

        after_training = [p.clone().cpu() for p in model.parameters()]
        for initial, final in zip(initial_params, after_training):
            assert not torch.equal(initial, final), "Parameters should have been updated"

    @pytest.mark.parametrize('ns_method', ['gram', 'standard'])
    def test_ns_method_stage3(self, ns_method):
        """Verify ns_method works with ZeRO Stage 3."""
        hidden_dim = 64
        batch_size = 8
        config_dict = {
            "train_batch_size": batch_size,
            "optimizer": {
                "type": "muon",
                "params": {
                    "lr": 0.01,
                    "ns_method": ns_method,
                }
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
            },
            "zero_optimization": {
                "stage": 3,
                "reduce_scatter": False,
            },
        }

        model = SimpleModel(hidden_dim=hidden_dim, nlayers=3)
        engine, optimizer, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
            dist_init_required=False,
        )

        for _ in range(3):
            x = torch.randn(batch_size, hidden_dim, device=engine.device, dtype=torch.half)
            y = torch.randint(0, hidden_dim, (batch_size, ), device=engine.device)
            loss = engine(x, y)
            engine.backward(loss)
            engine.step()


class TestMuonRejectsReduceScatter(DistributedTest):
    """Optimizer offload does not yet support Muon with reduce-scatter."""

    world_size = 1

    @pytest.mark.parametrize('zero_stage', [1, 2])
    def test_muon_reduce_scatter_with_optimizer_offload_raises(self, zero_stage):
        config_dict = {
            "train_batch_size": 4,
            "optimizer": {
                "type": "muon",
                "params": {
                    "lr": 0.01
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "reduce_scatter": True,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                },
            },
        }
        model = SimpleModel(hidden_dim=32, nlayers=2)
        with pytest.raises(ValueError, match="Muon with reduce scatter does not support optimizer offload"):
            deepspeed.initialize(config=config_dict,
                                 model=model,
                                 model_parameters=model.parameters(),
                                 dist_init_required=False)


class TestMuonZero12NumericalCorrectness(DistributedTest):
    """Numerical-correctness regression for #7807.

    Under ZeRO-1/2, Muon's Newton-Schulz orthogonalization must run on the full DP-averaged
    gradient on every rank that owns part of a parameter. The existing Muon tests only assert
    that parameters changed, which cannot detect a wrong-but-nonzero update. Here a 2D weight
    straddles a gradient-partition boundary and the applied update is compared with an
    independent full-gradient reference using the real muon_update."""

    world_size = 2

    @pytest.mark.parametrize(
        "zero_stage,ns_method,reduce_scatter,gas,overlap_comm,use_multi_rank_bucket_allreduce,"
        "contiguous_gradients,reduce_bucket_size", [
            pytest.param(1, "gram", False, 1, False, True, True, 500000000, id="z1-gram-allreduce"),
            pytest.param(1, "standard", False, 1, False, True, True, 500000000, id="z1-standard-allreduce"),
            pytest.param(2, "gram", False, 1, False, True, True, 500000000, id="z2-gram-allreduce"),
            pytest.param(2, "standard", False, 1, False, True, True, 500000000, id="z2-standard-allreduce"),
            pytest.param(1, "gram", True, 1, False, True, True, 500000000, id="z1-reduce-scatter"),
            pytest.param(2, "gram", True, 1, False, True, True, 500000000, id="z2-reduce-scatter"),
            pytest.param(2, "gram", True, 2, True, True, True, 500000000, id="z2-rs-gas2-overlap"),
            pytest.param(2, "gram", True, 2, False, False, True, 500000000, id="z2-rs-gas2-no-multi-rank"),
            pytest.param(2, "gram", True, 1, False, True, True, 32768, id="z2-rs-extra-large-param"),
            pytest.param(2, "gram", True, 1, False, True, False, 500000000, id="z2-rs-noncontiguous"),
        ])
    def test_update_matches_full_gradient_reference(self, zero_stage, ns_method, reduce_scatter, gas, overlap_comm,
                                                    use_multi_rank_bucket_allreduce, contiguous_gradients,
                                                    reduce_bucket_size):
        import copy
        from deepspeed.utils import safe_get_full_fp32_param
        from deepspeed.runtime.zero.muon.original_muon import muon_update

        hidden_dim, nlayers = 256, 3
        lr, momentum = 0.02, 0.95
        micro = 8
        world = dist.get_world_size()
        rank = dist.get_rank()

        torch.manual_seed(1234)
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=nlayers)
        init_state = copy.deepcopy(model.state_dict())

        config_dict = {
            "train_micro_batch_size_per_gpu": micro,
            "gradient_accumulation_steps": gas,
            # No clipping: keep the applied update exactly -lr * muon_update(grad) for the
            # reference comparison (Muon's orthogonalized update has a large global norm, so the
            # default gradient_clipping=1.0 would otherwise rescale it).
            "gradient_clipping": 0.0,
            "optimizer": {
                "type": "muon",
                "params": {
                    "lr": lr,
                    "momentum": momentum,
                    "ns_method": ns_method
                }
            },
            # Static loss scale so the update is unscaled and matches the reference.
            "fp16": {
                "enabled": True,
                "loss_scale": 1.0
            },
            "zero_optimization": {
                "stage": zero_stage,
                "reduce_scatter": reduce_scatter,
                "overlap_comm": overlap_comm,
                "use_multi_rank_bucket_allreduce": use_multi_rank_bucket_allreduce,
                "contiguous_gradients": contiguous_gradients,
                "reduce_bucket_size": reduce_bucket_size,
            },
        }
        engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                               model=model,
                                               model_parameters=model.parameters(),
                                               dist_init_required=False)
        device = engine.device

        # Precondition on the ACTUAL flattened ZeRO partition (includes alignment padding and the
        # real param ordering): a 2D Muon weight must straddle the rank-0/rank-1 boundary, else
        # #7807 (which only corrupts cross-partition weights) cannot be exercised at all.
        opt = engine.optimizer
        split_muon_params = []
        for gi, params in enumerate(opt.bit16_groups):
            for p in params:
                param_id = opt.get_param_id(p)
                partition_ids = opt.param_to_partition_ids[gi].get(param_id, [])
                if getattr(p, "use_muon", False) and len(partition_ids) > 1:
                    split_muon_params.append(p)
        assert split_muon_params, "no Muon weight straddles a partition boundary; resize the model"

        # Deterministic global batch, identical on every rank; each rank consumes its own slice so
        # the DP-averaged gradient equals the full-batch gradient used by the reference.
        gen = torch.Generator().manual_seed(999)
        gx = torch.randn(gas * world * micro, hidden_dim, generator=gen)
        gy = torch.randint(0, hidden_dim, (gas * world * micro, ), generator=gen)

        muon_named = [(n, p) for n, p in engine.module.named_parameters() if p.ndim >= 2]
        pre = {n: safe_get_full_fp32_param(p).clone() for n, p in muon_named}

        for micro_step in range(gas):
            batch_index = micro_step * world + rank
            start = batch_index * micro
            x = gx[start:start + micro].to(device).half()
            y = gy[start:start + micro].to(device)
            loss = engine(x, y)
            engine.backward(loss)
            engine.step()

        post = {n: safe_get_full_fp32_param(p).clone() for n, p in muon_named}

        # The post-step weight is all-gathered to every rank, so rank 0's assembled weight already
        # reflects every rank's contribution (including the cross-partition slices owned by others).
        if rank != 0:
            return

        # Independent reference: same init, full global batch, real muon_update on the full grad.
        # Run in fp16 to mirror the engine's forward/backward precision (minimizes the legitimate
        # gap). weight_decay=0 and gradient_clipping=0 make the applied update exactly -lr*update.
        ref = SimpleModel(hidden_dim=hidden_dim, nlayers=nlayers).to(device).half()
        ref.load_state_dict({k: v.to(device).half() for k, v in init_state.items()})
        ref.zero_grad(set_to_none=True)
        ref(gx.to(device).half(), gy.to(device)).backward()
        ref_grad = {n: p.grad.detach().float() for n, p in ref.named_parameters() if p.ndim >= 2}

        changed = False
        for n in pre:
            applied_update = ((pre[n] - post[n]) / lr).float().cpu()  # delta = -lr * update (wd=0, no clip)
            if applied_update.abs().max().item() > 0:
                changed = True
            g = ref_grad[n]
            # muon_update mutates grad/momentum in place; pass clones and a fresh zero buffer
            # (matches the engine's lazily-zeroed first-step momentum buffer).
            ref_update = muon_update(g.clone(), torch.zeros_like(g), beta=momentum, ns_method=ns_method).float().cpu()
            rel_err = ((applied_update - ref_update).norm() / (ref_update.norm() + 1e-8)).item()
            # Newton-Schulz amplifies fp16 gradient rounding, so a correct update still differs from
            # the reference by a few percent (measured up to ~0.07 for gram, ~0.22 for standard); the
            # #7807 partition-then-orthogonalize bug diverges by O(1) (measured ~0.6-0.67 on the
            # cross-partition weight). 0.40 separates them robustly for both ns_method values.
            assert rel_err < 0.40, (
                f"{n} (ZeRO-{zero_stage}, ns_method={ns_method}, reduce_scatter={reduce_scatter}, gas={gas}, "
                f"overlap_comm={overlap_comm}, multi_rank={use_multi_rank_bucket_allreduce}): "
                f"Muon update rel error {rel_err:.3f} vs "
                f"full-gradient reference -- orthogonalization likely ran on a partition slice rather than "
                f"the full averaged gradient (#7807)")
        assert changed, "optimizer step did not update any Muon weight (skipped step?)"
