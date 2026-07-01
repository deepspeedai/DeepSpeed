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
    """Muon needs the full all-reduced gradient matrix on each rank for its Newton-Schulz
    orthogonalization. reduce_scatter only delivers each rank its own partition slice, which
    silently corrupts cross-partition parameters in ZeRO-1/2 (#7807). Initialization must fail
    loudly, consistent with the ZeRO-3 guard in stage3.py (added in #7919)."""

    world_size = 1

    @pytest.mark.parametrize('zero_stage', [1, 2])
    def test_muon_reduce_scatter_raises(self, zero_stage):
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
            },
        }
        model = SimpleModel(hidden_dim=32, nlayers=2)
        with pytest.raises(ValueError, match="Muon and reduce scatter cannot be used together"):
            deepspeed.initialize(config=config_dict,
                                 model=model,
                                 model_parameters=model.parameters(),
                                 dist_init_required=False)


class TestMuonZero12NumericalCorrectness(DistributedTest):
    """Numerical-correctness regression for #7807.

    Under ZeRO-1/2, Muon's Newton-Schulz orthogonalization must run on the FULL DP-averaged
    gradient on every rank. The existing Muon tests only assert that parameters changed, which
    cannot detect a wrong-but-nonzero update. Here we run the supported reduce_scatter=False
    path on >=2 ranks, sized so a 2D weight straddles the gradient-partition boundary (exactly
    the case #7807 corrupted), and compare the applied Muon update against an independent
    reference that applies the real muon_update to the full averaged gradient. A
    partition-then-orthogonalize bug diverges by O(1) -- far above fp16/bf16 NS rounding."""

    world_size = 2

    @pytest.mark.parametrize('ns_method', ['gram', 'standard'])
    @pytest.mark.parametrize('zero_stage', [1, 2])
    def test_update_matches_full_gradient_reference(self, zero_stage, ns_method):
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
            "gradient_accumulation_steps": 1,
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
                "reduce_scatter": False
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
        muon_groups = [gi for gi, ps in enumerate(opt.bit16_groups) if ps and all(p.dim() >= 2 for p in ps)]
        assert muon_groups, "could not locate the Muon (2D-weight) param group in the optimizer"
        crosses = False
        for gi in muon_groups:
            boundary = opt.bit16_groups_flat[gi].numel() // world
            offset = 0
            for p in opt.bit16_groups[gi]:
                if offset < boundary < offset + p.numel():
                    crosses = True
                offset += p.numel()
        assert crosses, "no 2D Muon weight straddles the partition boundary; resize the model"

        # Deterministic global batch, identical on every rank; each rank consumes its own slice so
        # the DP-averaged gradient equals the full-batch gradient used by the reference.
        gen = torch.Generator().manual_seed(999)
        gx = torch.randn(world * micro, hidden_dim, generator=gen)
        gy = torch.randint(0, hidden_dim, (world * micro, ), generator=gen)
        x = gx[rank * micro:(rank + 1) * micro].to(device).half()
        y = gy[rank * micro:(rank + 1) * micro].to(device)

        muon_named = [(n, p) for n, p in engine.module.named_parameters() if p.ndim >= 2]
        pre = {n: safe_get_full_fp32_param(p).clone() for n, p in muon_named}

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
                f"{n} (ZeRO-{zero_stage}, ns_method={ns_method}): Muon update rel error {rel_err:.3f} vs "
                f"full-gradient reference -- orthogonalization likely ran on a partition slice rather than "
                f"the full averaged gradient (#7807)")
        assert changed, "optimizer step did not update any Muon weight (skipped step?)"
