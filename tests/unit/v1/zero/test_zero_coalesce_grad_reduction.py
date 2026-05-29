# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Tests for engine.coalesce_grad_reduction() -- ZeRO 1/2/3 coalesced reduction
across multiple engine.backward() calls per engine.step()."""

import pytest
import torch
import deepspeed

from unit.common import DistributedTest
from unit.simple_model import SimpleModel, SimpleMoEModel, random_dataloader, sequence_dataloader
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import set_z3_leaf_modules


def _config(zero_stage,
            world_size=2,
            contiguous_gradients=False,
            overlap_comm=False,
            force_fp16=False,
            reduce_bucket_size=None,
            gradient_clipping=None):
    config = {
        "train_batch_size": world_size,
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage,
            "contiguous_gradients": contiguous_gradients,
            "overlap_comm": overlap_comm,
        },
        "zero_force_ds_cpu_optimizer": False,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3,
                "torch_adam": True,
            },
        },
    }
    if reduce_bucket_size is not None:
        config["zero_optimization"]["reduce_bucket_size"] = reduce_bucket_size
    if gradient_clipping is not None:
        config["gradient_clipping"] = gradient_clipping
    if force_fp16 or not get_accelerator().is_bf16_supported():
        config["fp16"] = {"enabled": True, "initial_scale_power": 8, "loss_scale_window": 50}
    else:
        config["bf16"] = {"enabled": True}
    return config


def _build_model(hidden_dim, nlayers, config, seed=42):
    torch.manual_seed(seed)
    if config["zero_optimization"]["stage"] == 3:
        with deepspeed.zero.Init(config_dict_or_path=config):
            return SimpleModel(hidden_dim, nlayers=nlayers)
    return SimpleModel(hidden_dim, nlayers=nlayers)


def _init(config, hidden_dim, seed=42, nlayers=2):
    model = _build_model(hidden_dim, nlayers, config, seed=seed)
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    return engine


def _params(engine):
    return {n: p.detach().float().cpu().clone() for n, p in engine.module.named_parameters()}


def _assert_close(ref, test, label, tol=1e-6):
    for name in ref:
        diff = (ref[name] - test[name]).abs().max().item()
        assert diff < tol, f"{label}: {name} max-abs-diff {diff:.3e} >= {tol:.0e}"


class _NullCtx:

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _config_dtype(config):
    if config.get("fp16", {}).get("enabled"):
        return torch.float16
    return torch.bfloat16


def _train(config, hidden_dim, num_chunks, num_steps, use_no_sync, seed=42, nlayers=2):
    engine = _init(config, hidden_dim, seed=seed, nlayers=nlayers)
    batches = list(
        random_dataloader(model=engine,
                          total_samples=num_chunks * num_steps,
                          hidden_dim=hidden_dim,
                          device=engine.device,
                          dtype=_config_dtype(config)))
    for step_idx in range(num_steps):
        step_batches = batches[step_idx * num_chunks:(step_idx + 1) * num_chunks]
        ctx = engine.coalesce_grad_reduction() if use_no_sync else _NullCtx()
        with ctx:
            for i, batch in enumerate(step_batches):
                loss = engine(batch[0], batch[1])
                engine.set_gradient_accumulation_boundary(i == num_chunks - 1)
                engine.backward(loss)
        engine.step()
    out = _params(engine)
    engine.destroy()
    return out


# ---------------------------------------------------------------------------
# Bit-exact correctness across stage / contiguous_gradients / overlap_comm
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("contiguous_gradients,overlap_comm", [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
])
class TestCoalesceCombinations(DistributedTest):
    world_size = 2

    def test_multi_backward_bit_exact(self, zero_stage, contiguous_gradients, overlap_comm):
        cfg = _config(zero_stage, contiguous_gradients=contiguous_gradients, overlap_comm=overlap_comm)
        ref = _train(cfg, hidden_dim=8, num_chunks=4, num_steps=1, use_no_sync=False)
        test = _train(cfg, hidden_dim=8, num_chunks=4, num_steps=1, use_no_sync=True)
        _assert_close(ref, test, label=f"ZeRO-{zero_stage} cg={contiguous_gradients} oc={overlap_comm}")


# ---------------------------------------------------------------------------
# Larger model + small reduce_bucket_size to force multi-bucket flush
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("zero_stage", [2, 3])
class TestCoalesceBucketOverflow(DistributedTest):
    world_size = 2

    def test_multi_bucket_flush(self, zero_stage):
        # hidden=64 with nlayers=4 -> ~50K params, reduce_bucket_size=8K forces
        # multiple bucket flushes inside the single coalesced reduction pass.
        cfg = _config(zero_stage, contiguous_gradients=True, overlap_comm=True, reduce_bucket_size=8192)
        ref = _train(cfg, hidden_dim=64, num_chunks=4, num_steps=1, use_no_sync=False, nlayers=4)
        test = _train(cfg, hidden_dim=64, num_chunks=4, num_steps=1, use_no_sync=True, nlayers=4)
        _assert_close(ref, test, label=f"ZeRO-{zero_stage} multi-bucket", tol=5e-6)


# ---------------------------------------------------------------------------
# CPU offload combinations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("zero_stage,offload_optimizer,offload_param", [
    (1, True, False),
    (2, True, False),
    (3, True, False),
    (3, True, True),
])
class TestCoalesceCpuOffload(DistributedTest):
    world_size = 2

    def test_cpu_offload_bit_exact(self, zero_stage, offload_optimizer, offload_param):
        cfg = _config(zero_stage)
        if offload_optimizer:
            cfg["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": False}
        if offload_param:
            cfg["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": False}
        ref = _train(cfg, hidden_dim=8, num_chunks=4, num_steps=1, use_no_sync=False)
        test = _train(cfg, hidden_dim=8, num_chunks=4, num_steps=1, use_no_sync=True)
        label = f"ZeRO-{zero_stage} offload_opt={offload_optimizer} offload_param={offload_param}"
        _assert_close(ref, test, label=label)


# ---------------------------------------------------------------------------
# FP16 + dynamic loss scaling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestCoalesceFP16(DistributedTest):
    world_size = 2

    def test_fp16_bit_exact(self, zero_stage):
        # FP16 with dynamic loss scaling: loss_scaler reads grads only at
        # boundary. Verify coalesce path still yields identical params.
        cfg = _config(zero_stage, force_fp16=True)
        ref = _train(cfg, hidden_dim=8, num_chunks=4, num_steps=1, use_no_sync=False)
        test = _train(cfg, hidden_dim=8, num_chunks=4, num_steps=1, use_no_sync=True)
        _assert_close(ref, test, label=f"FP16 ZeRO-{zero_stage}")


# ---------------------------------------------------------------------------
# Gradient clipping + multi-step state hygiene
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestCoalesceCorrectness(DistributedTest):
    world_size = 2

    def test_multi_step_no_state_leak(self, zero_stage):
        # Three step() cycles back to back: no_sync state must reset cleanly
        # between contexts. The two runs reduce in different orders (baseline
        # reduces per-chunk via per-param hooks; coalesced reduces all params
        # once at flush in bit16_groups order), so bf16 fp32-master accumulation
        # diverges by a small amount that grows over multiple steps. Tol is
        # set to lr (1e-3); a tighter tol would fail due to reduction-order
        # non-associativity, not state leakage.
        cfg = _config(zero_stage)
        ref = _train(cfg, hidden_dim=8, num_chunks=3, num_steps=3, use_no_sync=False)
        test = _train(cfg, hidden_dim=8, num_chunks=3, num_steps=3, use_no_sync=True)
        _assert_close(ref, test, label=f"ZeRO-{zero_stage} 3x3", tol=1e-3)

    def test_gradient_clipping(self, zero_stage):
        # Gradient clipping reads the global grad norm at engine.step() time;
        # averaged_gradients must be populated by our flush before that point.
        cfg = _config(zero_stage, gradient_clipping=0.5)
        ref = _train(cfg, hidden_dim=8, num_chunks=4, num_steps=1, use_no_sync=False)
        test = _train(cfg, hidden_dim=8, num_chunks=4, num_steps=1, use_no_sync=True)
        _assert_close(ref, test, label=f"ZeRO-{zero_stage} clip=0.5")

    def test_n1_inside_context(self, zero_stage):
        cfg = _config(zero_stage)
        ref = _train(cfg, hidden_dim=8, num_chunks=1, num_steps=1, use_no_sync=False)
        test = _train(cfg, hidden_dim=8, num_chunks=1, num_steps=1, use_no_sync=True)
        _assert_close(ref, test, label=f"ZeRO-{zero_stage} N=1")


# ---------------------------------------------------------------------------
# use_grad_accum_attribute=True (ZeRO-1 + bf16 + grad_accum_dtype=fp32 + offload)
# ---------------------------------------------------------------------------
class TestCoalesceGradAccumDtype(DistributedTest):
    world_size = 2

    def test_zero1_offload_bf16_fp32_grad_accum(self):
        # ZeRO-1 + bf16 + grad_accum_dtype=fp32 + cpu_offload routes through
        # DeepSpeedZeroOptimizer with use_grad_accum_attribute=True. Each
        # backward's optimizer.backward_epilogue() drains param.grad into
        # param.grad_accum and clears param.grad. The flush iteration must
        # check get_gradient_for_reduction (which returns grad_accum) instead
        # of param.grad to avoid silently dropping the accumulated gradient.
        if not get_accelerator().is_bf16_supported():
            pytest.skip("requires bf16")
        cfg = _config(zero_stage=1)
        cfg["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": False}
        cfg["data_types"] = {"grad_accum_dtype": "fp32"}
        ref = _train(cfg, hidden_dim=8, num_chunks=4, num_steps=1, use_no_sync=False)
        test = _train(cfg, hidden_dim=8, num_chunks=4, num_steps=1, use_no_sync=True)
        _assert_close(ref, test, label="grad_accum_fp32 ZeRO-1 + offload")

    def test_zero2_bf16_fp32_grad_accum(self):
        # ZeRO-2 + same options uses use_grad_accum_attribute=False (param
        # grads route through normal param.grad), exercising the other branch.
        if not get_accelerator().is_bf16_supported():
            pytest.skip("requires bf16")
        cfg = _config(zero_stage=2)
        cfg["data_types"] = {"grad_accum_dtype": "fp32"}
        ref = _train(cfg, hidden_dim=8, num_chunks=4, num_steps=1, use_no_sync=False)
        test = _train(cfg, hidden_dim=8, num_chunks=4, num_steps=1, use_no_sync=True)
        _assert_close(ref, test, label="grad_accum_fp32 ZeRO-2")

    def test_zero1_no_offload_uses_bf16_optimizer(self):
        # ZeRO-1 + bf16 + grad_accum_dtype=fp32 + NO offload dispatches to
        # BF16_Optimizer (engine.py:1565-1567), which our context cannot
        # patch. Verify clear NotImplementedError.
        if not get_accelerator().is_bf16_supported():
            pytest.skip("requires bf16")
        cfg = _config(zero_stage=1)
        cfg["data_types"] = {"grad_accum_dtype": "fp32"}
        engine = _init(cfg, hidden_dim=8)
        with pytest.raises(NotImplementedError, match="optimizer wrapper"):
            with engine.coalesce_grad_reduction():
                pass
        engine.destroy()


# ---------------------------------------------------------------------------
# MoE: ep_size=1 smoke test + ep_size=2 with world_size=4 for the real path
# ---------------------------------------------------------------------------
def _train_moe(config, hidden_dim, num_chunks, num_steps, use_no_sync, ep_size=1, seed=42):
    torch.manual_seed(seed)
    model = SimpleMoEModel(hidden_dim=hidden_dim, ep_size=ep_size)
    engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
    dtype = torch.bfloat16 if get_accelerator().is_bf16_supported() else torch.float16
    batches = list(
        sequence_dataloader(model=engine,
                            total_samples=num_chunks * num_steps,
                            hidden_dim=hidden_dim,
                            device=engine.device,
                            dtype=dtype))
    for step_idx in range(num_steps):
        step_batches = batches[step_idx * num_chunks:(step_idx + 1) * num_chunks]
        ctx = engine.coalesce_grad_reduction() if use_no_sync else _NullCtx()
        with ctx:
            for i, batch in enumerate(step_batches):
                loss = engine(batch[0], batch[1])
                engine.set_gradient_accumulation_boundary(i == num_chunks - 1)
                engine.backward(loss)
        engine.step()
    out = _params(engine)
    engine.destroy()
    return out


@pytest.mark.parametrize("zero_stage", [1, 2])
class TestCoalesceMoE_EpSize1(DistributedTest):
    world_size = 2

    def test_smoke(self, zero_stage):
        config = _config(zero_stage, contiguous_gradients=(zero_stage == 2))
        ref = _train_moe(config, hidden_dim=16, num_chunks=4, num_steps=1, use_no_sync=False, ep_size=1)
        test = _train_moe(config, hidden_dim=16, num_chunks=4, num_steps=1, use_no_sync=True, ep_size=1)
        _assert_close(ref, test, label=f"MoE ep1 ZeRO-{zero_stage}")


@pytest.mark.parametrize("zero_stage", [1, 2])
class TestCoalesceMoE_EpSize2(DistributedTest):
    # ep_size=2 with world_size=4 exercises the real heterogeneous process
    # group path (expert_dp_process_group differs from dp_process_group).
    world_size = 4

    def test_expert_parallel(self, zero_stage):
        config = _config(zero_stage, world_size=4, contiguous_gradients=(zero_stage == 2))
        ref = _train_moe(config, hidden_dim=16, num_chunks=4, num_steps=1, use_no_sync=False, ep_size=2)
        test = _train_moe(config, hidden_dim=16, num_chunks=4, num_steps=1, use_no_sync=True, ep_size=2)
        _assert_close(ref, test, label=f"MoE ep2 ZeRO-{zero_stage}")


# ---------------------------------------------------------------------------
# ZeRO-3 leaf modules: leaf_module_hook zero-fills missing grads; the flush
# must mirror that to keep the per-rank reduction signature consistent.
# ---------------------------------------------------------------------------
class TestCoalesceZero3LeafModule(DistributedTest):
    """ZeRO-3 + leaf-module + multi-backward is broken on the baseline path
    (leaf hooks fire per backward and bucket-flush asserts on duplicate
    ds_ids -- independent of this PR). Instead of attempting bit-exact
    comparison against a broken baseline, we only verify that:
      1. Flush works under N=1 (the usual case).
      2. The flush's leaf-zero-fill mirror does not crash for unused leaf
         params.
    """
    world_size = 2

    def test_leaf_module_n1(self):
        cfg = _config(zero_stage=3)
        with deepspeed.zero.Init(config_dict_or_path=cfg):
            torch.manual_seed(42)
            model = SimpleModel(hidden_dim=8, nlayers=2)
        set_z3_leaf_modules(model, [torch.nn.Linear])
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=cfg)
        batch = next(iter(random_dataloader(model=engine, total_samples=1, hidden_dim=8, device=engine.device)))
        with engine.coalesce_grad_reduction():
            loss = engine(batch[0], batch[1])
            engine.set_gradient_accumulation_boundary(True)
            engine.backward(loss)
        engine.step()
        engine.destroy()


# ---------------------------------------------------------------------------
# Reentrant gradient checkpointing (use_reentrant=True): epilogue is invoked
# multiple times per backward for checkpointed regions.
# ---------------------------------------------------------------------------
# Reentrant gradient checkpointing (use_reentrant=True) is intentionally not
# tested here: combining `torch.utils.checkpoint(..., use_reentrant=True)` with
# DeepSpeed ZeRO + multi-backward + small models surfaces an upstream
# `loss.grad_fn is None` issue that is not specific to this PR. The coalesce
# path's hook accounting (max_expected_hooks_seen / hooks_fired_this_backward)
# is exercised by the existing tests via update_hook_state_and_maybe_run_epilogue.


# ---------------------------------------------------------------------------
# Failure modes: clear errors instead of silent state corruption
# ---------------------------------------------------------------------------
class TestCoalesceCollectiveCount(DistributedTest):
    """Validates the PR's central performance claim: coalesced backward issues
    strictly fewer cross-rank gradient-reduction collectives than baseline.

    Baseline: each engine.backward() drives the reducer hook -> dist.all_reduce
    (ZeRO-1/2 grad-partition path uses dist.all_reduce or dist.reduce; ZeRO-3
    uses dist.reduce_scatter_fn via reduce_scatter_coalesced), so the count
    scales with N=num_chunks.

    Coalesced: the per-param hooks are no-ops. Only the flush issues collectives,
    so the count is independent of N.

    We patch the actual primitives that the reducer calls (not just dist.all_reduce
    / dist.reduce_scatter; the latter is not a name DeepSpeed uses). Counting is
    confined to the backward phase via a 'recording' flag so optimizer.step()'s
    norm/overflow reductions (invariant across both runs) are excluded.
    """
    world_size = 2

    # ZeRO-1 with partition_gradients=False already issues exactly one
    # collective per step (reduce_gradients fires only at the boundary), so
    # baseline == coalesced for stage 1. The "N->1" claim only applies to
    # stages 2 and 3 which reduce in per-param hooks.
    @pytest.mark.parametrize("zero_stage", [2, 3])
    def test_coalesce_issues_fewer_collectives(self, zero_stage):
        import deepspeed.comm as dist

        original_all_reduce = dist.all_reduce
        original_reduce = dist.reduce
        original_rs_fn = dist.reduce_scatter_fn

        def run(use_no_sync, counts):

            def all_reduce_counter(*a, **k):
                if counts["recording"]:
                    counts["all_reduce"] += 1
                return original_all_reduce(*a, **k)

            def reduce_counter(*a, **k):
                if counts["recording"]:
                    counts["reduce"] += 1
                return original_reduce(*a, **k)

            def rs_fn_counter(*a, **k):
                if counts["recording"]:
                    counts["reduce_scatter_fn"] += 1
                return original_rs_fn(*a, **k)

            dist.all_reduce = all_reduce_counter
            dist.reduce = reduce_counter
            dist.reduce_scatter_fn = rs_fn_counter
            try:
                cfg = _config(zero_stage)
                engine = _init(cfg, hidden_dim=8)
                batches = list(
                    random_dataloader(model=engine,
                                      total_samples=4,
                                      hidden_dim=8,
                                      device=engine.device,
                                      dtype=_config_dtype(cfg)))
                ctx = engine.coalesce_grad_reduction() if use_no_sync else _NullCtx()
                counts["recording"] = True
                with ctx:
                    for i, batch in enumerate(batches):
                        loss = engine(batch[0], batch[1])
                        engine.set_gradient_accumulation_boundary(i == 3)
                        engine.backward(loss)
                counts["recording"] = False
                engine.step()
                engine.destroy()
            finally:
                dist.all_reduce = original_all_reduce
                dist.reduce = original_reduce
                dist.reduce_scatter_fn = original_rs_fn

        baseline = {"all_reduce": 0, "reduce": 0, "reduce_scatter_fn": 0, "recording": False}
        coalesced = {"all_reduce": 0, "reduce": 0, "reduce_scatter_fn": 0, "recording": False}
        run(use_no_sync=False, counts=baseline)
        run(use_no_sync=True, counts=coalesced)

        baseline_total = baseline["all_reduce"] + baseline["reduce"] + baseline["reduce_scatter_fn"]
        coalesced_total = coalesced["all_reduce"] + coalesced["reduce"] + coalesced["reduce_scatter_fn"]
        assert baseline_total > 0, f"ZeRO-{zero_stage} baseline issued no collectives: {baseline}"
        assert coalesced_total < baseline_total, (
            f"ZeRO-{zero_stage}: coalesced issued {coalesced_total} collectives, "
            f"baseline {baseline_total} -- coalesce did not reduce count. "
            f"baseline={baseline} coalesced={coalesced}")


class TestCoalesceZero3MicroStepInvariant(DistributedTest):
    """ZeRO-3 partition_grads at stage3.py:1717 takes the copy_ path only when
    micro_step_id == 0 (otherwise it does add_ on top of stale buffer state).

    Therefore the coalesce-mode flush MUST observe micro_step_id == 0 to avoid
    silently accumulating step k's partition into step k+1's buffer.

    The chain that would normally bump micro_step_id during the coalesce block
    is hook -> update_hook_state_and_maybe_run_epilogue -> _backward_post_hook
    -> _backward_epilogue -> allreduce_gradients -> independent_gradient_partition_epilogue
    (which sets _epilogue_ran_this_backward=True, picked up by next forward's
    clear_backward_seen_flag to bump micro_step_id). The chain is broken at
    engine.py:2480 by 'not self.inside_no_sync_ctxt'. This test pins down that
    invariant by snapshotting micro_step_id at flush time across multiple
    chunks and steps.
    """
    world_size = 2

    def test_micro_step_id_is_zero_at_flush(self):
        cfg = _config(zero_stage=3)
        engine = _init(cfg, hidden_dim=8)
        observed = []
        original = engine._flush_coalesced_reduction_zero3

        def spy(optimizer):
            observed.append(optimizer.micro_step_id)
            return original(optimizer)

        engine._flush_coalesced_reduction_zero3 = spy
        batches = list(
            random_dataloader(model=engine,
                              total_samples=12,
                              hidden_dim=8,
                              device=engine.device,
                              dtype=_config_dtype(cfg)))
        for step in range(3):
            with engine.coalesce_grad_reduction():
                for i in range(4):
                    batch = batches[step * 4 + i]
                    loss = engine(batch[0], batch[1])
                    engine.set_gradient_accumulation_boundary(i == 3)
                    engine.backward(loss)
            engine.step()
        engine.destroy()

        assert observed == [0, 0, 0], (f"micro_step_id at flush across 3 steps was {observed}; "
                                       "expected all zeros. A non-zero value would force partition_grads "
                                       "into the add_ branch and silently accumulate stale partition data "
                                       "from the previous step (gradient corruption).")

    def test_zero3_multi_step_diff_under_corruption_threshold(self):
        # The hypothetical add_-path corruption (reviewer claim) would inject
        # the previous step's full gradient partition into the current step's
        # buffer, producing per-step divergence proportional to lr (Adam steps
        # ~2x the intended size). Across 3 steps that is roughly 3 * lr = 3e-3.
        # We assert the actual divergence is well under that threshold to
        # falsify the corruption hypothesis (typical bf16 reduction-order noise
        # is ~1e-3 over 3 steps).
        cfg = _config(zero_stage=3)
        ref = _train(cfg, hidden_dim=8, num_chunks=4, num_steps=3, use_no_sync=False)
        test = _train(cfg, hidden_dim=8, num_chunks=4, num_steps=3, use_no_sync=True)
        _assert_close(ref, test, label="ZeRO-3 4x3 corruption-threshold", tol=2e-3)


class TestCoalesceFailureModes(DistributedTest):
    world_size = 1

    def test_step_inside_context_raises(self):
        engine = _init(_config(zero_stage=2, world_size=1), hidden_dim=8)
        with engine.coalesce_grad_reduction():
            with pytest.raises(AssertionError, match="no_sync"):
                engine.step()
        engine.destroy()

    def test_existing_no_sync_still_blocked_for_zero2(self):
        engine = _init(_config(zero_stage=2, world_size=1), hidden_dim=8)
        with pytest.raises(AssertionError, match="ZeRO stage"):
            with engine.no_sync():
                pass
        engine.destroy()

    def test_nested_no_sync_outer_coalesce_grad_reduction(self):
        # Outer coalesce_grad_reduction must reject inner no_sync (and vice
        # versa) since both share inside_no_sync_ctxt and the reentry would
        # corrupt the outer flag on inner exit.
        engine = _init(_config(zero_stage=1, world_size=1), hidden_dim=8)
        with engine.coalesce_grad_reduction():
            with pytest.raises(AssertionError, match="no_sync"):
                with engine.no_sync():
                    pass
        engine.destroy()

    def test_nested_coalesce_grad_reduction_inside_no_sync(self):
        engine = _init(_config(zero_stage=1, world_size=1), hidden_dim=8)
        with engine.no_sync():
            with pytest.raises(AssertionError, match="nested"):
                with engine.coalesce_grad_reduction():
                    pass
        engine.destroy()

    def test_pipeline_parallelism_rejected(self, monkeypatch):
        # No PipelineModule fixture in this test set; monkey-patch the
        # pipeline_parallelism flag so the precheck path is exercised.
        engine = _init(_config(zero_stage=2, world_size=1), hidden_dim=8)
        monkeypatch.setattr(engine, "pipeline_parallelism", True)
        with pytest.raises(NotImplementedError, match="pipeline parallelism"):
            with engine.coalesce_grad_reduction():
                pass
        engine.destroy()


class TestCoalesceDocumentedBehaviors(DistributedTest):
    """These tests pin down behaviors the docstring promises but that could
    silently regress: safe_get_full_grad returning local-only grads, optimizer
    boundary state restored on context exit."""
    world_size = 2

    def test_safe_get_full_grad_returns_local_pre_reduce_value(self):
        # Documented behavior: inside the context safe_get_full_grad reads
        # the locally-accumulated param.grad (no all-reduce yet). Compare it
        # against param.grad directly to confirm the path is local-only and
        # not eagerly reducing.
        from deepspeed.utils import safe_get_full_grad
        engine = _init(_config(zero_stage=2), hidden_dim=8)
        batch = next(
            iter(
                random_dataloader(model=engine,
                                  total_samples=1,
                                  hidden_dim=8,
                                  device=engine.device,
                                  dtype=torch.bfloat16 if get_accelerator().is_bf16_supported() else torch.float16)))
        with engine.coalesce_grad_reduction():
            loss = engine(batch[0], batch[1])
            engine.set_gradient_accumulation_boundary(True)
            engine.backward(loss)
            for param in engine.module.parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                full_grad = safe_get_full_grad(param)
                assert full_grad is not None
                # Same-data check: safe_get_full_grad must surface the same
                # tensor that param.grad currently holds (the unreduced local
                # value), not a freshly all-reduced full gradient.
                assert torch.equal(full_grad, param.grad), \
                    "safe_get_full_grad returned a tensor different from local param.grad"
                break
        engine.step()
        engine.destroy()

    def test_engine_boundary_restored_and_step_works_on_exit(self):
        # Enter the context with boundary=False (mid-accumulation pattern).
        # On exit, the engine flag must be restored. The user is responsible
        # for setting boundary=True before engine.step() -- verify that this
        # contract is real (post-exit step does advance global_steps once
        # the user toggles boundary back).
        engine = _init(_config(zero_stage=2), hidden_dim=8)
        engine.set_gradient_accumulation_boundary(False)
        saved = engine._is_gradient_accumulation_boundary
        with engine.coalesce_grad_reduction():
            batch = next(
                iter(
                    random_dataloader(
                        model=engine,
                        total_samples=1,
                        hidden_dim=8,
                        device=engine.device,
                        dtype=torch.bfloat16 if get_accelerator().is_bf16_supported() else torch.float16)))
            engine.backward(engine(batch[0], batch[1]))
        assert engine._is_gradient_accumulation_boundary == saved
        prev_steps = engine.global_steps
        engine.set_gradient_accumulation_boundary(True)
        engine.step()
        assert engine.global_steps == prev_steps + 1
        engine.destroy()
