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
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
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


class TestMuonOptimizerOffload(DistributedTest):
    """Muon remains correct when optimizer state is kept on the CPU."""

    world_size = 1

    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    def test_muon_with_optimizer_offload(self, zero_stage):
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
                "reduce_scatter": False,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                },
            },
        }
        model = SimpleModel(hidden_dim=32, nlayers=2)
        initial_params = [p.detach().clone().cpu() for p in model.parameters()]
        engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                               model=model,
                                               model_parameters=model.parameters(),
                                               dist_init_required=False)
        x = torch.randn(4, 32, device=engine.device, dtype=torch.half)
        y = torch.randint(0, 32, (4, ), device=engine.device)
        engine.backward(engine(x, y))
        engine.step()
        assert any(not torch.equal(initial,
                                   current.detach().cpu())
                   for initial, current in zip(initial_params, model.parameters()))


class TestMuonOffloadGatherMemory(DistributedTest):
    """Chunking must bound communication memory without changing multi-step updates."""

    world_size = [1, 2, 4]

    @pytest.mark.parametrize("zero_stage", [1, 2])
    def test_bounded_gather_matches_unbucketed_training(self, zero_stage):
        import math
        from deepspeed.utils import safe_get_full_fp32_param

        def train(buffer_bytes):
            torch.manual_seed(42)
            model = SimpleModel(hidden_dim=31, nlayers=4)
            total_numel = sum(param.numel() for param in model.parameters())
            config = {
                "train_micro_batch_size_per_gpu": 4,
                "gradient_clipping": 1.0,
                "optimizer": {
                    "type": "muon",
                    "params": {
                        "lr": 0.01,
                        "momentum": 0.95,
                    },
                },
                "fp16": {
                    "enabled": True,
                    "loss_scale": 16.0,
                },
                "zero_optimization": {
                    "stage": zero_stage,
                    "reduce_scatter": False,
                    "offload_optimizer": {
                        "device": "cpu",
                    },
                },
            }
            engine, _, _, _ = deepspeed.initialize(config=config,
                                                   model=model,
                                                   model_parameters=model.parameters(),
                                                   dist_init_required=False)
            # Force both multi-matrix batches and chunks smaller than one matrix.
            # This pins the regression where an LRU limit did not bound allocations.
            engine.optimizer._muon_allgather_max_buffer_bytes = buffer_bytes
            gather_bytes = []
            all_gather = dist.all_gather_into_tensor

            def record_gather(output_tensor, input_tensor, *args, **kwargs):
                if input_tensor.dtype == torch.float32:
                    gather_bytes.append((input_tensor.numel() + output_tensor.numel()) * input_tensor.element_size())
                return all_gather(output_tensor, input_tensor, *args, **kwargs)

            torch.manual_seed(100 + dist.get_rank())
            x = torch.randn(4, 31, device=engine.device, dtype=torch.half)
            y = torch.randint(0, 31, (4, ), device=engine.device)
            before = [safe_get_full_fp32_param(param).clone() for param in model.parameters()]
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(dist, "all_gather_into_tensor", record_gather)
                for _ in range(3):
                    engine.backward(engine(x, y))
                    engine.step()
            after = [safe_get_full_fp32_param(param).clone() for param in model.parameters()]
            assert any(not torch.equal(initial, final) for initial, final in zip(before, after))

            world_size = dist.get_world_size()
            if world_size > 1:
                assert gather_bytes
                assert max(gather_bytes) <= buffer_bytes
                # Owned slices fit inside one ZeRO partition, not the full group.
                partition_numel = math.ceil(total_numel / world_size)
                assert max(gather_bytes) <= 2 * (world_size + 1) * partition_numel * 4
            else:
                assert not gather_bytes
            engine.destroy()
            return after

        reference = train(1024 * 1024)
        for buffer_bytes in [4096, 16384]:
            actual = train(buffer_bytes)
            for expected, result in zip(reference, actual):
                torch.testing.assert_close(result, expected, rtol=0, atol=1e-6)


class TestMuonAllGatherBufferLifecycle:

    @pytest.mark.parametrize("optimizer_class", [DeepSpeedZeroOptimizer, DeepSpeedZeroOptimizer_Stage3])
    def test_clear_muon_allgather_buffers(self, optimizer_class):
        optimizer = optimizer_class.__new__(optimizer_class)
        optimizer._muon_allgather_buffers = {"test": object()}
        optimizer._muon_allgather_buffer_bytes = 128

        optimizer._clear_muon_allgather_buffers()

        assert not optimizer._muon_allgather_buffers
        assert optimizer._muon_allgather_buffer_bytes == 0


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
        "contiguous_gradients,reduce_bucket_size,offload_optimizer", [
            pytest.param(1, "gram", False, 1, False, True, True, 500000000, False, id="z1-gram-allreduce"),
            pytest.param(1, "standard", False, 1, False, True, True, 500000000, False, id="z1-standard-allreduce"),
            pytest.param(2, "gram", False, 1, False, True, True, 500000000, False, id="z2-gram-allreduce"),
            pytest.param(2, "standard", False, 1, False, True, True, 500000000, False, id="z2-standard-allreduce"),
            pytest.param(1, "gram", True, 1, False, True, True, 500000000, False, id="z1-reduce-scatter"),
            pytest.param(2, "gram", True, 1, False, True, True, 500000000, False, id="z2-reduce-scatter"),
            pytest.param(2, "gram", True, 2, True, True, True, 500000000, False, id="z2-rs-gas2-overlap"),
            pytest.param(2, "gram", True, 2, False, False, True, 500000000, False, id="z2-rs-gas2-no-multi-rank"),
            pytest.param(2, "gram", True, 1, False, True, True, 32768, False, id="z2-rs-extra-large-param"),
            pytest.param(2, "gram", True, 1, False, True, False, 500000000, False, id="z2-rs-noncontiguous"),
            pytest.param(1, "gram", False, 2, False, True, True, 500000000, True, id="z1-offload-gas2"),
            pytest.param(2, "gram", True, 2, False, True, True, 500000000, True, id="z2-offload-rs-gas2"),
        ])
    def test_update_matches_full_gradient_reference(self, zero_stage, ns_method, reduce_scatter, gas, overlap_comm,
                                                    use_multi_rank_bucket_allreduce, contiguous_gradients,
                                                    reduce_bucket_size, offload_optimizer):
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
        if offload_optimizer:
            config_dict["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
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


class TestMuonOffloadLossScaling(DistributedTest):
    """Verify Muon updates under CPU offload are invariant to loss_scale with clipping."""

    world_size = 2

    @pytest.mark.parametrize("zero_stage", [1, 2, 3])
    def test_offload_loss_scale_equivalence(self, zero_stage):
        from deepspeed.utils import safe_get_full_fp32_param

        hidden_dim, nlayers = 128, 2
        lr = 0.02
        clip_grad = 1.0

        def _run_with_scale(loss_scale):
            torch.manual_seed(42)
            model = SimpleModel(hidden_dim=hidden_dim, nlayers=nlayers)
            config_dict = {
                "train_micro_batch_size_per_gpu": 4,
                "gradient_clipping": clip_grad,
                "optimizer": {
                    "type": "muon",
                    "params": {
                        "lr": lr,
                        "momentum": 0.0
                    }
                },
                "fp16": {
                    "enabled": True,
                    "loss_scale": loss_scale
                },
                "zero_optimization": {
                    "stage": zero_stage,
                    "reduce_scatter": False,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    }
                }
            }
            engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                                   model=model,
                                                   model_parameters=model.parameters(),
                                                   dist_init_required=False)
            device = engine.device
            muon_named = [(n, p) for n, p in engine.module.named_parameters()
                          if getattr(p, "use_muon", False) or len(getattr(p, "ds_shape", p.shape)) >= 2]
            assert len(muon_named) > 0, "No Muon parameters identified!"
            pre = {n: safe_get_full_fp32_param(p).clone() for n, p in muon_named}

            torch.manual_seed(999)
            x = torch.randn(4, hidden_dim, device=device).half()
            y = torch.randint(0, hidden_dim, (4, ), device=device)
            loss = engine(x, y)
            engine.backward(loss)
            engine.step()

            post = {n: safe_get_full_fp32_param(p).clone() for n, p in muon_named}
            updates = {n: (pre[n] - post[n]).float() for n in pre}
            return updates

        updates_scale_1 = _run_with_scale(1.0)
        updates_scale_1024 = _run_with_scale(1024.0)

        for n in updates_scale_1:
            norm_1 = updates_scale_1[n].norm().item()
            norm_1024 = updates_scale_1024[n].norm().item()
            assert norm_1 > 0.01, f"Update vanished for {n} under scale 1.0"
            assert norm_1024 > 0.01, f"Update vanished for {n} under scale 1024.0"
            max_diff = (updates_scale_1[n] - updates_scale_1024[n]).abs().max().item()
            assert max_diff < 1e-3, (
                f"Update mismatch between scale 1.0 and 1024.0 under stage {zero_stage} "
                f"with gradient clipping: max_diff={max_diff}, norm_1={norm_1}, norm_1024={norm_1024}")


class TestMuonZero3NVMeMomentumResidency(DistributedTest):
    """Verify ZeRO-3 resident Muon momentum buffers persist across NVMe swapping steps."""

    world_size = 1

    @pytest.mark.parametrize("pipeline", [False, True])
    def test_zero3_nvme_momentum_residency(self, tmpdir, pipeline):
        from deepspeed.ops.aio import AsyncIOBuilder
        from deepspeed.runtime.swap_tensor.partitioned_optimizer_swapper import PartitionedOptimizerSwapper
        from deepspeed.runtime.swap_tensor.pipelined_optimizer_swapper import PipelinedOptimizerSwapper

        if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
            pytest.skip("Skip tests since async-io is not compatible")

        hidden_dim, nlayers = 1024, 2
        lr = 0.01
        momentum = 0.95
        offload_optimizer_cfg = {
            "device": "nvme",
            "nvme_path": str(tmpdir),
        }
        if pipeline:
            offload_optimizer_cfg["pipeline_read"] = True
            offload_optimizer_cfg["pipeline_write"] = True

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "muon",
                "params": {
                    "lr": lr,
                    "momentum": momentum
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 1.0
            },
            "zero_optimization": {
                "stage": 3,
                "reduce_scatter": False,
                "save_muon_momentum_buffer_in_memory": True,
                "offload_optimizer": offload_optimizer_cfg,
                "sub_group_size": 100
            },
            "aio": {
                "block_size": 1048576
            }
        }
        torch.manual_seed(42)
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=nlayers)
        engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                               model=model,
                                               model_parameters=model.parameters(),
                                               dist_init_required=False)

        device = engine.device
        x = torch.randn(1, hidden_dim, device=device).half()
        y = torch.randint(0, hidden_dim, (1, ), device=device)

        opt = engine.optimizer
        assert opt.swap_optimizer, "NVMe swap_optimizer must be enabled"
        assert opt.save_muon_momentum_buffer_in_memory
        assert any(
            opt.optimizer_swapper.is_swappable_tensor(numel=numel) for numel in opt.fp16_partitioned_groups_flat_numel)
        if pipeline:
            assert isinstance(opt.optimizer_swapper, PipelinedOptimizerSwapper)
        else:
            assert isinstance(opt.optimizer_swapper, PartitionedOptimizerSwapper)

        # Step 1
        engine.backward(engine(x, y))
        engine.step()

        assert len(opt.muon_momentum_buffer_partitioned_groups_flat) > 0
        step1_momentums = {}
        for sub_group_id, buf in opt.muon_momentum_buffer_partitioned_groups_flat.items():
            expected_numel = int(opt.fp16_partitioned_groups_flat_numel[sub_group_id])
            assert buf.numel() == expected_numel, (
                f"Resident momentum for subgroup {sub_group_id} had numel={buf.numel()} "
                f"instead of {expected_numel}; storage was evicted by NVMe swapper!")
            assert not getattr(buf, "swappable", True)
            step1_momentums[sub_group_id] = buf.clone()

        # Step 2: verify multi-step execution does not crash and momentum accumulates
        engine.backward(engine(x, y))
        engine.step()

        for sub_group_id, buf in opt.muon_momentum_buffer_partitioned_groups_flat.items():
            expected_numel = int(opt.fp16_partitioned_groups_flat_numel[sub_group_id])
            assert buf.numel() == expected_numel
            diff = (buf - step1_momentums[sub_group_id]).abs().max().item()
            assert diff > 0.0, f"Momentum buffer did not accumulate changes at step 2 for subgroup {sub_group_id}"

        # Step 3
        engine.backward(engine(x, y))
        engine.step()
        for sub_group_id, buf in opt.muon_momentum_buffer_partitioned_groups_flat.items():
            assert buf.numel() == int(opt.fp16_partitioned_groups_flat_numel[sub_group_id])

    @pytest.mark.parametrize("pipeline", [False, True])
    def test_zero3_nvme_aggregate_unswapped_fragments(self, tmpdir, pipeline):
        from deepspeed.ops.aio import AsyncIOBuilder
        from deepspeed.runtime.zero.muon.original_muon import muon_update
        from deepspeed.utils import safe_get_full_fp32_param

        if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
            pytest.skip("Skip tests since async-io is not compatible")

        # 20 layers of 128x128: each parameter is 16,384 elements (< 1 MiB in FP32).
        # Subgroup aggregate is 327,680 elements (> 262,144 elements / 1 MiB in FP32),
        # making the subgroup swappable while all individual gradient fragments are unswapped.
        class SmallMatrixModel(torch.nn.Module):

            def __init__(self, num_layers=20, dim=128):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Linear(dim, dim, bias=False) for _ in range(num_layers)])

            def forward(self, x):
                return sum(layer(x).square().mean() for layer in self.layers) * 1024.0

        config_dict = {
            "train_micro_batch_size_per_gpu": 256,
            "steps_per_print": 1,
            "gradient_clipping": 0.0,
            "optimizer": {
                "type": "muon",
                "params": {
                    "lr": 0.01,
                    "momentum": 0.95
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 1.0
            },
            "zero_optimization": {
                "stage": 3,
                "reduce_scatter": False,
                "save_muon_momentum_buffer_in_memory": True,
                "offload_optimizer": {
                    "device": "nvme",
                    "nvme_path": str(tmpdir),
                    "pipeline_read": pipeline,
                    "pipeline_write": pipeline,
                },
                "sub_group_size": 1000000
            },
            "aio": {
                "block_size": 1048576
            }
        }
        torch.manual_seed(42)
        model = SmallMatrixModel()
        initial_state = {name: param.detach().clone() for name, param in model.named_parameters()}
        engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                               model=model,
                                               model_parameters=model.parameters(),
                                               dist_init_required=False)

        device = engine.device
        x = torch.randn(256, 128, device=device).half()
        opt = engine.optimizer
        assert opt.swap_optimizer

        reference = SmallMatrixModel().to(device).half()
        masters = {name: param.to(device) for name, param in initial_state.items()}
        momentums = {name: torch.zeros_like(param) for name, param in masters.items()}
        for _ in range(2):
            reference.load_state_dict(masters)
            reference.zero_grad(set_to_none=True)
            reference(x).backward()
            with torch.no_grad():
                for name, param in reference.named_parameters():
                    update = muon_update(param.grad.float(), momentums[name], beta=0.95)
                    masters[name].add_(update, alpha=-0.01)

            engine.backward(engine(x))
            engine.step()

            # Writeback must preserve every small fragment for the optimizer's
            # second swap-in, even when no gradient fragment has an NVMe file.
            for name, param in model.named_parameters():
                actual = safe_get_full_fp32_param(param).to(device)
                expected_update = initial_state[name].to(device) - masters[name]
                assert expected_update.norm() > 0
                relative_error = (actual - masters[name]).norm() / expected_update.norm()
                assert relative_error < 0.05, f"{name}: gradient writeback relative error {relative_error.item():.4f}"

    @pytest.mark.parametrize("pipeline", [False, True])
    @pytest.mark.parametrize("save_in_memory", [False, True])
    def test_zero3_nvme_mixed_fragments_numerical_equivalence(self, tmpdir, pipeline, save_in_memory):
        import copy
        from deepspeed.ops.aio import AsyncIOBuilder
        from deepspeed.utils import safe_get_full_fp32_param
        from deepspeed.runtime.swap_tensor.partitioned_optimizer_swapper import PartitionedOptimizerSwapper
        from deepspeed.runtime.swap_tensor.pipelined_optimizer_swapper import PipelinedOptimizerSwapper
        from deepspeed.runtime.zero.muon.original_muon import muon_update

        if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
            pytest.skip("Skip tests since async-io is not compatible")

        class MixedMatrixModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                # 512 x 512 = 262,144 elements (>= 1 MiB in FP32) -> SWAPPED
                self.large = torch.nn.Linear(512, 512, bias=False)
                # 512 x 128 = 65,536 elements (< 1 MiB in FP32) -> UNSWAPPED
                self.proj = torch.nn.Linear(512, 128, bias=False)
                # 128 x 128 = 16,384 elements (< 1 MiB in FP32) -> UNSWAPPED
                self.small = torch.nn.Linear(128, 128, bias=False)

            def forward(self, x):
                # A squared loss over a wide batch keeps the weight gradients well conditioned,
                # so Newton-Schulz does not amplify fp16 noise into the comparison. The constant
                # factor lifts the gradients out of the fp16 subnormal range.
                return self.small(self.proj(self.large(x))).pow(2).mean() * 1024.0

        lr = 0.01
        momentum = 0.95
        num_steps = 2
        micro_batch = 256

        torch.manual_seed(42)
        base_model = MixedMatrixModel()
        init_state = copy.deepcopy(base_model.state_dict())

        # 1. Independent full-gradient reference (pure PyTorch + canonical muon_update across 2 steps)
        device = get_accelerator().current_device_name()
        ref_masters = {n: p.clone().detach().cpu().float() for n, p in init_state.items()}
        init_masters = {n: p.clone() for n, p in ref_masters.items()}
        # Muon momentum is a persistent optimizer state and is kept in fp32, matching ZeRO-1/2/3.
        ref_momentums = {n: torch.zeros_like(p).to(device).float() for n, p in ref_masters.items()}

        gen = torch.Generator().manual_seed(1234)
        inputs = [torch.randn(micro_batch, 512, generator=gen).to(device).half() for _ in range(num_steps)]

        for step in range(num_steps):
            ref_model = MixedMatrixModel().to(device).half()
            ref_model.load_state_dict({k: v.to(device).half() for k, v in ref_masters.items()})

            ref_model.zero_grad(set_to_none=True)
            loss = ref_model(inputs[step])
            loss.backward()
            with torch.no_grad():
                for n, p in ref_model.named_parameters():
                    update = muon_update(p.grad.detach().float(), ref_momentums[n], beta=momentum, ns_method="gram")
                    ref_masters[n].add_(update.cpu().float(), alpha=-lr)

        ref_updates = {n: (init_masters[n] - ref_masters[n]) for n in ref_masters}

        # 2. ZeRO-3 NVMe run with mixed swapped/unswapped fragments in the same subgroup
        nvme_cfg = {
            "device": "nvme",
            "nvme_path": str(tmpdir),
        }
        if pipeline:
            nvme_cfg["pipeline_read"] = True
            nvme_cfg["pipeline_write"] = True

        config_dict = {
            "train_micro_batch_size_per_gpu": micro_batch,
            "steps_per_print": 1,
            "gradient_clipping": 0.0,
            "optimizer": {
                "type": "muon",
                "params": {
                    "lr": lr,
                    "momentum": momentum
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 1.0
            },
            "zero_optimization": {
                "stage": 3,
                "reduce_scatter": False,
                "save_muon_momentum_buffer_in_memory": save_in_memory,
                "offload_optimizer": nvme_cfg,
                "sub_group_size": 1000000
            },
            "aio": {
                "block_size": 1048576
            }
        }
        model = MixedMatrixModel()
        model.load_state_dict({k: v.clone() for k, v in init_state.items()})
        engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                               model=model,
                                               model_parameters=model.parameters(),
                                               dist_init_required=False)

        # Explicitly verify all 3 parameters are identified as Muon parameters under ZeRO-3
        muon_named = [(n, p) for n, p in engine.module.named_parameters()
                      if getattr(p, "use_muon", False) or len(getattr(p, "ds_shape", p.shape)) >= 2]
        assert len(muon_named) == 3, f"Expected 3 Muon parameters, got {len(muon_named)}: {[n for n, p in muon_named]}"
        assert set(n for n, p in muon_named) == {"large.weight", "proj.weight", "small.weight"}
        init_params = {n: safe_get_full_fp32_param(p).clone().cpu() for n, p in muon_named}

        for step in range(num_steps):
            loss = engine(inputs[step])
            engine.backward(loss)
            engine.step()

        opt = engine.optimizer
        assert opt.swap_optimizer
        if pipeline:
            assert isinstance(opt.optimizer_swapper, PipelinedOptimizerSwapper)
        else:
            assert isinstance(opt.optimizer_swapper, PartitionedOptimizerSwapper)

        nvme_final = {n: safe_get_full_fp32_param(p).clone().cpu() for n, p in muon_named}
        applied_updates = {n: (init_params[n] - nvme_final[n]) for n, p in muon_named}

        # Verify non-trivial update and numerical equivalence against independent reference
        for n in ref_updates:
            applied_norm = applied_updates[n].norm().item()
            ref_norm = ref_updates[n].norm().item()
            assert ref_norm > 0.01, f"Reference update vanished for {n}"
            assert applied_norm > 0.01, f"Engine update vanished for {n}"
            rel_err = ((applied_updates[n] - ref_updates[n]).norm() / (ref_norm + 1e-8)).item()
            assert rel_err < 0.05, (f"Numerical divergence for {n} under pipeline={pipeline}: "
                                    f"rel_err={rel_err:.4f}, applied_norm={applied_norm:.4f}, ref_norm={ref_norm:.4f}")


class TestMuonZero3NVMeMultiRankMixedFragments(DistributedTest):
    """Verify ZeRO-3 NVMe Muon multi-rank reconstruction, DP gradient averaging, and mixed fragments."""

    world_size = 2

    @pytest.mark.parametrize("pipeline", [False, True])
    def test_zero3_nvme_multirank_mixed_fragments(self, tmpdir, pipeline):
        import copy
        from deepspeed.ops.aio import AsyncIOBuilder
        from deepspeed.utils import safe_get_full_fp32_param
        from deepspeed.runtime.swap_tensor.partitioned_optimizer_swapper import PartitionedOptimizerSwapper
        from deepspeed.runtime.swap_tensor.pipelined_optimizer_swapper import PipelinedOptimizerSwapper
        from deepspeed.runtime.zero.muon.original_muon import muon_update

        if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
            pytest.skip("Skip tests since async-io is not compatible")

        class MixedMatrixModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                # 512 x 1024 = 524,288 elements (2 MiB in FP32).
                # Partition per rank (world_size=2) = 262,144 elements (>= 1 MiB in FP32) -> SWAPPED
                self.large = torch.nn.Linear(512, 1024, bias=False)
                # 1024 x 128 = 131,072 elements (512 KiB in FP32).
                # Partition per rank = 65,536 elements (< 1 MiB in FP32) -> UNSWAPPED
                self.proj = torch.nn.Linear(1024, 128, bias=False)
                # 128 x 128 = 16,384 elements (64 KiB in FP32).
                # Partition per rank = 8,192 elements (< 1 MiB in FP32) -> UNSWAPPED
                self.small = torch.nn.Linear(128, 129, bias=False)
                # 127 x 129 = 16,383 elements: odd numel, so ZeRO-3 pads the flat partition
                # and the final rank owns a partially out-of-range slice -> UNSWAPPED
                self.odd = torch.nn.Linear(129, 127, bias=False)

            def forward(self, x):
                # A squared loss keeps the per-sample output error distinct, so the weight
                # gradients stay well conditioned and Newton-Schulz does not amplify fp16 noise.
                return self.odd(self.small(self.proj(self.large(x)))).pow(2).mean() * 1024.0

        lr = 0.01
        momentum = 0.95
        num_steps = 2
        micro_batch = 256
        rank = dist.get_rank()
        device = get_accelerator().current_device_name()

        torch.manual_seed(42)
        base_model = MixedMatrixModel()
        init_state = copy.deepcopy(base_model.state_dict())

        # Deterministic global inputs across steps: each rank receives its own slice
        gen = torch.Generator().manual_seed(1234)
        inputs_step = [torch.randn(2 * micro_batch, 512, generator=gen).to(device).half() for _ in range(num_steps)]

        # 1. Independent high-precision oracle (on rank 0):
        # Maintains FP32 master weights & FP32 momentum, computes FP16 forward/backward per rank,
        # explicitly averages DP gradients across ranks, and steps Muon.
        if rank == 0:
            ref_masters = {n: p.clone().detach().cpu().float() for n, p in init_state.items()}
            init_masters = {n: p.clone() for n, p in ref_masters.items()}
            # Muon momentum is a persistent optimizer state and is kept in fp32, matching ZeRO-1/2/3.
            ref_momentums = {n: torch.zeros_like(p).to(device).float() for n, p in ref_masters.items()}

            for step in range(num_steps):
                x0 = inputs_step[step][:micro_batch]
                x1 = inputs_step[step][micro_batch:]

                ref_model = MixedMatrixModel().to(device).half()
                ref_model.load_state_dict({k: v.to(device).half() for k, v in ref_masters.items()})

                ref_model.zero_grad(set_to_none=True)
                loss0 = ref_model(x0)
                loss0.backward()
                grads0 = {n: p.grad.clone() for n, p in ref_model.named_parameters()}

                ref_model.zero_grad(set_to_none=True)
                loss1 = ref_model(x1)
                loss1.backward()
                grads1 = {n: p.grad.clone() for n, p in ref_model.named_parameters()}

                # DeepSpeed averages DP gradients in the fp16 communication dtype, so the oracle
                # must do the same before promoting to fp32 for momentum and Newton-Schulz.
                avg_grads = {n: ((grads0[n] + grads1[n]) / 2.0).float() for n in grads0}
                with torch.no_grad():
                    for n in ref_masters:
                        up = muon_update(avg_grads[n], ref_momentums[n], beta=momentum, ns_method="gram")
                        ref_masters[n].add_(up.cpu().float(), alpha=-lr)

            ref_updates = {n: (init_masters[n] - ref_masters[n]) for n in ref_masters}

        # 2. DeepSpeed ZeRO-3 NVMe run with mixed swapped/unswapped fragments across 2 ranks
        nvme_cfg = {
            "device": "nvme",
            "nvme_path": str(tmpdir),
        }
        if pipeline:
            nvme_cfg["pipeline_read"] = True
            nvme_cfg["pipeline_write"] = True

        config_dict = {
            "train_micro_batch_size_per_gpu": micro_batch,
            "steps_per_print": 1,
            "gradient_clipping": 0.0,
            "optimizer": {
                "type": "muon",
                "params": {
                    "lr": lr,
                    "momentum": momentum
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 1.0
            },
            "zero_optimization": {
                "stage": 3,
                "reduce_scatter": False,
                "save_muon_momentum_buffer_in_memory": True,
                "offload_optimizer": nvme_cfg,
                "sub_group_size": 1000000
            },
            "aio": {
                "block_size": 1048576
            }
        }
        model = MixedMatrixModel()
        model.load_state_dict({k: v.clone() for k, v in init_state.items()})
        engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                               model=model,
                                               model_parameters=model.parameters(),
                                               dist_init_required=False)

        muon_named = [(n, p) for n, p in engine.module.named_parameters()
                      if getattr(p, "use_muon", False) or len(getattr(p, "ds_shape", p.shape)) >= 2]
        assert len(muon_named) == 4
        assert set(n for n, p in muon_named) == {"large.weight", "proj.weight", "small.weight", "odd.weight"}

        # Exercise the padded final partition: odd.weight cannot be split evenly across 2 ranks,
        # so its last partition is padded and reconstruction must drop the out-of-range tail.
        odd_param = dict(muon_named)["odd.weight"]
        world_size = dist.get_world_size()
        assert odd_param.ds_numel % world_size != 0, "odd.weight must not divide evenly across ranks"
        assert odd_param.partition_numel() * world_size > odd_param.ds_numel, "expected a padded final partition"

        init_engine_params = {n: safe_get_full_fp32_param(p).clone().cpu() for n, p in muon_named}

        # Each rank executes on its own micro-batch slice
        rank_slice = slice(rank * micro_batch, (rank + 1) * micro_batch)
        for step in range(num_steps):
            rank_x = inputs_step[step][rank_slice]
            loss = engine(rank_x)
            engine.backward(loss)
            engine.step()

        opt = engine.optimizer
        assert opt.swap_optimizer
        if pipeline:
            assert isinstance(opt.optimizer_swapper, PipelinedOptimizerSwapper)
        else:
            assert isinstance(opt.optimizer_swapper, PartitionedOptimizerSwapper)

        final_engine_params = {n: safe_get_full_fp32_param(p).clone().cpu() for n, p in muon_named}

        if rank == 0:
            applied_updates = {n: (init_engine_params[n] - final_engine_params[n]) for n, p in muon_named}
            for n in ref_updates:
                applied_norm = applied_updates[n].norm().item()
                ref_norm = ref_updates[n].norm().item()
                assert ref_norm > 0.01, f"Reference update vanished for {n}"
                assert applied_norm > 0.01, f"Engine update vanished for {n}"
                rel_err = ((applied_updates[n] - ref_updates[n]).norm() / (ref_norm + 1e-8)).item()
                assert rel_err < 0.05, (
                    f"Update divergence for {n} under pipeline={pipeline}: "
                    f"rel_err={rel_err:.4f}, applied_norm={applied_norm:.4f}, ref_norm={ref_norm:.4f}")
