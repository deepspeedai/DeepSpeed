# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Contract tests for NativeRollout (two ranks, one accelerator each).

The oracle is HuggingFace greedy decoding with the trainer's current weights: after every sync and every
reconfiguration (TP degree, replicas, membership, source of the weights), the rollout's greedy responses must be the
reference model's, also when a reconfiguration reuses a cached plan and a group made earlier, and when the trainer
runs on one rank and the rollout on the other.
"""

import copy

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.native import NativeRollout, NativeRolloutConfig, TrainerMPU
from deepspeed.runtime.rollout.zero3_state import Zero3StateMove
from unit.common import DistributedTest

transformers = pytest.importorskip("transformers")

NEW_TOKENS = 6


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = None


def tiny_qwen3():
    cfg = transformers.Qwen3Config(vocab_size=101,
                                   hidden_size=64,
                                   intermediate_size=96,
                                   num_hidden_layers=2,
                                   num_attention_heads=4,
                                   num_key_value_heads=2,
                                   head_dim=16,
                                   max_position_embeddings=64,
                                   tie_word_embeddings=False)
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
    return model


def prompts(rank, device):
    # Different row counts and widths per rank: a replica pads its ranks' prompts to a common width.
    g = torch.Generator().manual_seed(rank)
    rows, width = (2, 5) if rank == 0 else (3, 7)
    ids = torch.randint(1, 101, (rows, width), generator=g)
    mask = torch.ones(rows, width, dtype=torch.long)
    mask[0, :2] = 0
    ids[0, :2] = 0
    return ids.to(device), mask.to(device)


def hf_greedy(model, ids, mask, n):
    out = []
    with torch.no_grad():
        for _ in range(n):
            positions = (mask.cumsum(1) - 1).clamp(min=0)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=positions).logits[:, -1]
            token = logits.argmax(-1, keepdim=True)
            out.append(token)
            ids = torch.cat([ids, token], 1)
            mask = torch.cat([mask, torch.ones_like(token)], 1)
    return torch.cat(out, 1)


# Threshold 0: every weight partitioned, however small. With an optimizer, the partitions are views into one flat
# buffer (not each at the start of its own storage).
CONFIG = {
    "train_micro_batch_size_per_gpu": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-3,
            "torch_adam": True
        }
    },
    "zero_optimization": {
        "stage": 3,
        "stage3_param_persistence_threshold": 0
    }
}


class TestNativeRollout(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("planner", ["rect", "affine"])
    def test_generations_follow_the_trainer_across_syncs_and_reconfigurations(self, planner):
        device = torch.device(get_accelerator().current_device_name())
        model = tiny_qwen3()
        reference = copy.deepcopy(model).to(device)
        engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=CONFIG)
        cfg = NativeRolloutConfig(tp_size=2, max_batch_size=8, max_len=32, eos_token_ids=[], planner=planner)
        rollout = NativeRollout(engine, _Tokenizer(), cfg)
        rank = dist.get_rank()
        ids, mask = prompts(rank, device)
        sampling = SamplingConfig(max_new_tokens=NEW_TOKENS, temperature=0.0)

        def update():
            # The same elementwise change on the trainer's partitions and on the reference.
            with torch.no_grad():
                for p in engine.module.parameters():
                    p.ds_tensor.mul_(1.25)
                for p in reference.parameters():
                    p.mul_(1.25)

        def assert_follows_reference():
            batch = rollout.generate(RolloutRequest(ids, mask), sampling)
            start = int(batch.response_start_idx[0])
            assert torch.equal(batch.input_ids[:, start:], hf_greedy(reference, ids, mask, NEW_TOKENS))
            assert batch.attention_mask[:, start:].all()

        tp2_group = rollout.layout.group
        rollout.sync_weights(0)
        rng = get_accelerator().get_rng_state()
        assert_follows_reference()
        # Sampling draws from its own seeds; the training RNG stream is untouched.
        assert torch.equal(get_accelerator().get_rng_state(), rng)
        update()
        rollout.sync_weights(1)
        assert_follows_reference()

        # Two TP1 replicas, from the TP2 arenas.
        rollout.reconfigure([0, 1], 1)
        assert_follows_reference()
        # Rank 1 alone (its TP1 replica kept: same decoder, nothing moved); then an update synced into its arena only.
        kept = rollout.layout.decoder
        rollout.reconfigure([1], 1)
        if rank == 1:
            assert rollout.layout.decoder is kept and rollout.last_reconfigure["reused"]
            assert_follows_reference()
        update()
        rollout.sync_weights(2)
        if rank == 1:
            assert_follows_reference()
        else:
            with pytest.raises(RuntimeError, match="not a rollout rank"):
                rollout.generate(RolloutRequest(ids, mask), sampling)
        # Back to TP2 on both ranks, from the trainer's partitions, in the group made at the start.
        rollout.reconfigure([0, 1], 2, source="trainer")
        assert rollout.layout.group is tp2_group
        assert_follows_reference()
        # The first transition again, after an update: its plan comes from the cache, the sync plan of TP1 replicas too.
        update()
        rollout.sync_weights(3)
        rollout.reconfigure([0, 1], 1)
        assert rollout.last_reconfigure["plan_cached"]
        assert_follows_reference()
        update()
        rollout.sync_weights(4)
        assert_follows_reference()


class TestNativeRolloutApart(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("planner", ["rect", "affine"])
    def test_rollout_on_other_ranks_than_the_trainer(self, planner):
        # The trainer on rank 0 alone (ZeRO-3 over one rank), the rollout on rank 1; then the rollout grows onto the
        # trainer's rank and shrinks back.
        device = torch.device(get_accelerator().current_device_name())
        rank = dist.get_rank()
        model = tiny_qwen3()
        reference = copy.deepcopy(model).to(device)
        mpu = TrainerMPU([0])
        engine = None
        if rank == 0:
            engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=CONFIG, mpu=mpu)
        cfg = NativeRolloutConfig(ranks=[1],
                                  tp_size=1,
                                  max_batch_size=8,
                                  max_len=32,
                                  eos_token_ids=[],
                                  planner=planner)
        rollout = NativeRollout(engine, _Tokenizer(), cfg)
        assert rollout.trainer_ranks == [0] and len(rollout.names) == len(list(reference.parameters()))
        ids, mask = prompts(rank, device)
        sampling = SamplingConfig(max_new_tokens=NEW_TOKENS, temperature=0.0)

        def update():
            with torch.no_grad():
                if engine is not None:
                    for p in engine.module.parameters():
                        p.ds_tensor.mul_(1.25)
                for p in reference.parameters():
                    p.mul_(1.25)

        def assert_follows_reference():
            batch = rollout.generate(RolloutRequest(ids, mask), sampling)
            start = int(batch.response_start_idx[0])
            assert torch.equal(batch.input_ids[:, start:], hf_greedy(reference, ids, mask, NEW_TOKENS))

        def train_step():
            # A real step on the trainer rank alone (ZeRO-3's forward and backward must not wait for rank 1); the
            # reference then takes the trainer's new weights.
            if engine is not None:
                loss = engine(input_ids=ids, attention_mask=mask).logits.float().mean()
                engine.backward(loss)
                engine.step()
            for name, ref in reference.named_parameters():
                if engine is not None:
                    param = dict(engine.module.named_parameters())[name]
                    with deepspeed.zero.GatheredParameters([param]):
                        ref.data.copy_(param.data)
                dist.broadcast(ref.data, src=0)

        rollout.sync_weights(0)
        if rank == 1:
            assert_follows_reference()
        update()
        rollout.sync_weights(1)
        if rank == 1:
            assert_follows_reference()
        for step in range(2):
            train_step()
            rollout.sync_weights(10 + step)
            if rank == 1:
                assert_follows_reference()
        # Onto both ranks as TP2, from the rollout's arena; an update synced into both.
        rollout.reconfigure([0, 1], 2)
        assert_follows_reference()
        update()
        rollout.sync_weights(2)
        assert_follows_reference()
        # Back to rank 1, from the trainer.
        rollout.reconfigure([1], 1, source="trainer")
        update()
        rollout.sync_weights(3)
        if rank == 1:
            assert_follows_reference()


class TestTrainerSwitch(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("planner", ["rect", "affine"])
    def test_training_continues_across_trainer_layouts(self, planner):
        # The trainer moves from rank 0 alone to both ranks and back (Zero3StateMove), the rollout on rank 1 following
        # it (bind); each move frees the engine left behind and restores the one taken up. The oracle is one plain
        # model trained by torch AdamW alongside: every rank of an engine is fed the whole batch, so ZeRO's averaged
        # gradients are the reference's.
        device = torch.device(get_accelerator().current_device_name())
        rank = dist.get_rank()
        config = copy.deepcopy(CONFIG)
        config["optimizer"]["params"]["weight_decay"] = 0.01
        # Each engine its own mpu: an engine without one would take the previous engine's. The engine over every
        # rank first: ZeRO-3 numbers parameters per process, and an engine's numbers must agree over its ranks.
        mpu, mpu_both = TrainerMPU([0]), TrainerMPU([0, 1])
        model = tiny_qwen3()
        both, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config, mpu=mpu_both)
        alone = None
        if rank == 0:
            model = tiny_qwen3()
            alone, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config, mpu=mpu)
        reference = tiny_qwen3().to(device)
        reference_opt = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=0.01)
        cfg = NativeRolloutConfig(ranks=[1],
                                  tp_size=1,
                                  max_batch_size=8,
                                  max_len=32,
                                  eos_token_ids=[],
                                  planner=planner)
        rollout = NativeRollout(alone, _Tokenizer(), cfg)
        ids, mask = prompts(rank, device)
        batch = torch.randint(1, 101, (4, 6), generator=torch.Generator().manual_seed(7)).to(device)
        sampling = SamplingConfig(max_new_tokens=NEW_TOKENS, temperature=0.0)

        def train(engine):
            if engine is not None:
                loss = engine(input_ids=batch).logits.float().mean()
                engine.backward(loss)
                engine.step()
            reference_opt.zero_grad()
            reference(input_ids=batch).logits.float().mean().backward()
            reference_opt.step()

        def assert_matches_reference(engine):
            for name, param in engine.module.named_parameters():
                with deepspeed.zero.GatheredParameters([param]):
                    torch.testing.assert_close(param.data,
                                               dict(reference.named_parameters())[name].data,
                                               rtol=1e-4,
                                               atol=1e-6)

        def assert_rollout_follows_reference(step):
            rollout.sync_weights(step)
            if rank == 1:
                batch_out = rollout.generate(RolloutRequest(ids, mask), sampling)
                start = int(batch_out.response_start_idx[0])
                assert torch.equal(batch_out.input_ids[:, start:], hf_greedy(reference, ids, mask, NEW_TOKENS))

        for _ in range(2):
            train(alone)
        to_both = Zero3StateMove(alone, both, planner=planner)
        allocated = get_accelerator().memory_allocated()
        to_both.execute(release_source=True)
        if rank == 0:
            # rank 0 held the whole model's fp32 master weights and two Adam moments alone
            numel = sum(p.numel() for p in reference.parameters())
            assert allocated - get_accelerator().memory_allocated() >= 3 * 4 * numel
        assert both.global_steps == 2
        rollout.bind(both)
        assert rollout.trainer_ranks == [0, 1]
        assert_rollout_follows_reference(2)
        # A step on both ranks continues the optimizer (moments and step count, hence bias correction).
        train(both)
        assert_matches_reference(both)
        Zero3StateMove(both, alone, planner=planner).execute(release_source=True)
        rollout.bind(alone)
        train(alone)
        if rank == 0:
            assert_matches_reference(alone)
        assert_rollout_follows_reference(4)
        # The first plan again, after more training on rank 0.
        to_both.execute(release_source=True)
        assert_matches_reference(both)
