# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Contract tests for Zero3Repartition: one ZeRO-3 engine changes its data-parallel ranks in place.

One engine (B) is built over four ranks, trains, shrinks to two ranks, trains, grows back to four, trains, and
shrinks again with the first shrink's (cached) plan. It keeps the global batch, so its accumulation steps double on
two ranks. Every micro-batch is the whole batch on every rank, so every layout computes the same gradient exactly: two
or four equal gradients average without rounding, and halving the loss for two accumulation steps is exact. The
oracle is a plain engine (C) that stays on four ranks and trains on the same batches: after each phase, B's fp32 master
and Adam moments (the full tensors, gathered over its current ranks) must equal C's bit for bit, and the move itself
must keep B's state bit for bit. The existing path (A: one engine per layout, Zero3StateMove between them) is checked
against C the same way.

A torch AdamW model is no oracle over several steps: plain ZeRO-3 differs from it by 5e-7 after one step, on two and
four ranks alike, and Adam amplifies that to 1e-3 after five steps.

Attention is eager, for deterministic backward passes.
"""

import copy

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.native import TrainerMPU
from deepspeed.runtime.rollout.zero3_state import Zero3Repartition, Zero3StateMove
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_optimizer_state
from unit.common import DistributedTest

transformers = pytest.importorskip("transformers")

CONFIG = {
    "train_micro_batch_size_per_gpu": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-3,
            "weight_decay": 0.01,
            "torch_adam": True
        }
    },
    "zero_optimization": {
        "stage": 3,
        "stage3_param_persistence_threshold": 0
    }
}
MOMENTS = ("exp_avg", "exp_avg_sq")


def tiny_qwen3():
    cfg = transformers.Qwen3Config(vocab_size=101,
                                   hidden_size=64,
                                   intermediate_size=96,
                                   num_hidden_layers=2,
                                   num_attention_heads=4,
                                   num_key_value_heads=2,
                                   head_dim=16,
                                   max_position_embeddings=64,
                                   tie_word_embeddings=False,
                                   attn_implementation="eager")
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
    return model


def full_state(engine):
    """Every trainable parameter's full fp32 master and moments (collective over the engine's ranks)."""
    state = {}
    for name, p in engine.module.named_parameters():
        state[f"fp32:{name}"] = safe_get_full_fp32_param(p).clone()
        for k in MOMENTS:
            state[f"{k}:{name}"] = safe_get_full_optimizer_state(p, k).clone()
    return state


class TestZero3Repartition(DistributedTest):
    world_size = 4

    @pytest.mark.parametrize("dtype", ["fp32", "bf16"])
    @pytest.mark.parametrize("planner", ["rect", "affine"])
    def test_training_continues_across_dp_sizes(self, planner, dtype):
        device = torch.device(get_accelerator().current_device_name())
        rank = dist.get_rank()
        config = copy.deepcopy(CONFIG)
        if dtype == "bf16":
            if not get_accelerator().is_bf16_supported():
                pytest.skip("bf16 is not supported")
            config["bf16"] = {"enabled": True}
        config2 = copy.deepcopy(config)  # A's two-rank engine keeps the global batch of four as B does
        config2["gradient_accumulation_steps"] = 2
        # Each engine its own mpu, all made by every rank in the same order; the engines over every rank first, so
        # that ZeRO-3's parameter numbers agree over each engine's ranks.
        mpus = [TrainerMPU(ranks) for ranks in ([0, 1, 2, 3], [0, 1], [0, 1, 2, 3], [0, 1], [0, 1, 2, 3])]
        b4, b2, a4_mpu, a2_mpu, c_mpu = mpus

        def build(cfg, mpu):
            model = tiny_qwen3()
            return deepspeed.initialize(model=model, model_parameters=model.parameters(), config=cfg, mpu=mpu)[0]

        engine, a4, plain = build(config, b4), build(config, a4_mpu), build(config, c_mpu)
        a2 = build(config2, a2_mpu) if rank in (0, 1) else None
        numel = sum(p.ds_numel for p in plain.module.parameters())
        problems = []  # this rank's failed checks, raised on every rank together (see agree)
        steps = 0

        def check(ok, what):
            if not ok:
                problems.append(what)

        def agree(where):
            # Some checks run on part of the ranks; a rank that raised alone would leave the others in the next
            # collective, so every rank raises together here.
            everyone = [None] * dist.get_world_size()
            dist.all_gather_object(everyone, problems)
            failed = {r: p for r, p in enumerate(everyone) if p}
            assert not failed, f"{where}: {failed}"

        def guarded(what, fn, *args):
            # An exception on the ranks of a two-rank phase would leave ranks 2 and 3 in the next collective: it is
            # recorded instead, and every rank raises at the next agree.
            try:
                fn(*args)
            except Exception as err:
                problems.append(f"{what} raised {type(err).__name__}: {err}")

        def train(pairs):
            # One batch per step for every engine; every micro-batch is the whole batch on every rank of a layout,
            # ``gas`` micro-batches per step. C trains on all four ranks every step.
            nonlocal steps
            batch = torch.randint(1, 101, (4, 6), generator=torch.Generator().manual_seed(100 + steps)).to(device)

            def micro_steps(e, gas):
                for _ in range(gas):
                    loss = e(input_ids=batch).logits.float().mean()
                    e.backward(loss)
                    e.step()

            steps += 1
            for e, ranks in pairs + [(plain, [0, 1, 2, 3])]:
                if rank in ranks:
                    gas = 4 // len(ranks)
                    check(e.gradient_accumulation_steps() == gas, f"accumulation {e.gradient_accumulation_steps()}")
                    guarded(f"step {steps}", micro_steps, e, gas)
                agree(f"step {steps}")  # before the next engine, whose collectives may span the ranks that failed

        def check_equal(after, before, what):
            check(after.keys() == before.keys(), f"{what}: entries differ")
            bad = {
                k: (after[k] - before[k]).abs().max().item()
                for k in before if k in after and not torch.equal(after[k], before[k])
            }
            check(not bad, f"{what}: {len(bad)} of {len(before)} differ, e.g. {dict(list(bad.items())[:3])}")

        def check_against_plain(where, b, a, ranks):
            expected = full_state(plain)  # every rank: C is over all four
            if rank in ranks:
                check_equal(full_state(b), expected, f"B vs C {where}")
                if a is not None:
                    check_equal(full_state(a), expected, f"A vs C {where}")
            agree(where)

        # Three steps, so that the micro-step counter (3) is not a multiple of the next layout's accumulation steps.
        for _ in range(3):
            train([(engine, [0, 1, 2, 3]), (a4, [0, 1, 2, 3])])
        before = full_state(engine)
        check_against_plain("on four ranks", engine, a4, [0, 1, 2, 3])

        # Shrink: ranks 2 and 3 leave and give back their state.
        allocated = get_accelerator().memory_allocated()
        to_two, to_four = Zero3Repartition(engine, b2, planner=planner), Zero3Repartition(engine, b4, planner=planner)
        guarded("the shrink", to_two.execute)
        agree("the shrink")
        if rank in (2, 3):
            # A quarter of the model's lp (= fp32 master for fp32) or fp32 master, two moments and gradients.
            freed = allocated - get_accelerator().memory_allocated()
            check(freed >= 3 * 4 * numel // 4, f"freed only {freed} bytes")
        Zero3StateMove(a4, a2, planner=planner).execute(release_source=True)
        if rank in (0, 1):
            check_equal(full_state(engine), before, "B's move to two ranks")
            check(engine.gradient_accumulation_steps() == 2, "accumulation after the shrink")
            check(engine.dp_world_size == 2, "dp_world_size after the shrink")
        agree("after the shrink")
        for _ in range(2):
            train([(engine, [0, 1]), (a2, [0, 1])])
        before = full_state(engine) if rank in (0, 1) else None
        check_against_plain("on two ranks", engine, a2, [0, 1])

        # Grow: ranks 2 and 3 come back, filled from ranks 0 and 1, and take over the step counters.
        guarded("the grow", to_four.execute)
        agree("the grow")
        Zero3StateMove(a2, a4, planner=planner).execute(release_source=True)
        after = full_state(engine)
        if rank in (0, 1):
            check_equal(after, before, "B's move back to four ranks")
        check(engine.gradient_accumulation_steps() == 1, "accumulation after the grow")
        check(engine.global_steps == 5, f"global_steps {engine.global_steps}")
        counters = [None] * 4
        dist.all_gather_object(counters, (engine.global_steps, engine.global_samples, engine.micro_steps))
        check(len(set(counters)) == 1, f"counters differ: {counters}")
        agree("after the grow")
        train([(engine, [0, 1, 2, 3]), (a4, [0, 1, 2, 3])])
        check_against_plain("after a step back on four ranks", engine, a4, [0, 1, 2, 3])

        # Shrink again with the same object: the source has the first shrink's ranks and partition sizes (DeepSpeed
        # built that layout, the grow installed this one), so the first plan serves again; B must still match C.
        first_plan = to_two.plan
        guarded("the second shrink", to_two.execute)
        agree("the second shrink")
        check(to_two.plan_cached and to_two.plan is first_plan, "the second shrink made a new plan")
        train([(engine, [0, 1])])
        check_against_plain("on two ranks again", engine, None, [0, 1])

    def test_ranks_that_never_stepped_rejoin(self):
        # The engine shrinks before its first step, so ranks 2 and 3 join the grow without a parameter fetch trace
        # while ranks 0 and 1 have a complete one. ZeRO-3 compares a trace that is not complete with the other ranks'
        # in a collective; without a common restart the first step after the grow fails (a list-length disagreement)
        # or hangs. Then it shrinks again with the same object, from the same source layout as its first shrink: the
        # plan made then, when the optimizer had no moments yet, moved only the fp32 master and must not serve again.
        # B must match C, which trains on four ranks throughout.
        device = torch.device(get_accelerator().current_device_name())
        rank = dist.get_rank()
        b4, b2, c_mpu = (TrainerMPU(ranks) for ranks in ([0, 1, 2, 3], [0, 1], [0, 1, 2, 3]))
        engines = []
        for mpu in (b4, c_mpu):
            model = tiny_qwen3()
            engines.append(
                deepspeed.initialize(model=model, model_parameters=model.parameters(), config=CONFIG, mpu=mpu)[0])
        engine, plain = engines
        to_two, to_four = Zero3Repartition(engine, b2), Zero3Repartition(engine, b4)
        to_two.execute()
        layouts = [[0, 1], [0, 1], [0, 1, 2, 3], [0, 1]]
        for step, ranks in enumerate(layouts):
            batch = torch.randint(1, 101, (4, 6), generator=torch.Generator().manual_seed(200 + step)).to(device)
            if step == 2:
                to_four.execute()
            if step == 3:
                to_two.execute()
            for e, members in ((engine, ranks), (plain, [0, 1, 2, 3])):
                if rank in members:
                    for _ in range(4 // len(members)):
                        e.backward(e(input_ids=batch).logits.float().mean())
                        e.step()
                dist.barrier()
        expected = full_state(plain)
        if rank in layouts[-1]:
            got = full_state(engine)
            bad = [k for k in expected if not torch.equal(got[k], expected[k])]
            assert not bad, f"{len(bad)} of {len(expected)} differ from the engine that stayed on four ranks: {bad[:3]}"
