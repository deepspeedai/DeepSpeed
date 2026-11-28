# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Contract tests for relayout planning: moving weights between sharded layouts on different rank sets.

Ranks are simulated in one process: every rank's holdings and needs are described, the planner runs on all of them
(as each rank does after gathering), and the planned blocks are copied between the simulated ranks' tensors. The
oracle is the decoder's own loader: a destination arena filled by the plan must equal, bit for bit, an arena of the
same layout loaded from the full weights. Covers ZeRO-3 to tensor parallel on other ranks, a change of the
tensor-parallel degree, a lost rank whose shards come from the trainer instead, and replicas on two nodes, where the
inter-node bytes are checked against the least possible, the relay's stages pipelined over buckets, and inter-node
fetches spread over several remote holders. Each of these runs with both planners: the rectangle planner and the
affine IR's ``plan_transfer`` (``affine_plan``), which the last tests also compare with each other.
"""

import itertools
import random

import pytest
import torch

from deepspeed.checkpoint.affine import AffinePiece
from deepspeed.checkpoint.affine_transfer import InvalidMapError
from deepspeed.checkpoint.zero_affine import zero_storage_affine_map
from deepspeed.runtime.rollout.affine_plan import plan_transfers_affine
from deepspeed.runtime.rollout.native import DecoderSpec, TPDecoder
from deepspeed.runtime.rollout.relayout import (RELAYED, RelayoutPlan, describe, flat_window_pieces, local_block,
                                                piece_blocks, plan_transfers)

SPEC = DecoderSpec(vocab_size=101,
                   hidden_size=64,
                   intermediate_size=96,
                   num_layers=2,
                   num_heads=4,
                   num_kv_heads=4,
                   head_dim=16,
                   rms_norm_eps=1e-6,
                   rope_theta=10000.0,
                   tie_word_embeddings=False,
                   qk_norm=True)


def full_weights():
    s, d = SPEC, SPEC.head_dim
    shapes = {
        "model.embed_tokens.weight": (s.vocab_size, s.hidden_size),
        "lm_head.weight": (s.vocab_size, s.hidden_size),
        "model.norm.weight": (s.hidden_size, )
    }
    for i in range(s.num_layers):
        p = f"model.layers.{i}."
        shapes.update({
            p + "input_layernorm.weight": (s.hidden_size, ),
            p + "post_attention_layernorm.weight": (s.hidden_size, ),
            p + "self_attn.q_proj.weight": (s.num_heads * d, s.hidden_size),
            p + "self_attn.k_proj.weight": (s.num_kv_heads * d, s.hidden_size),
            p + "self_attn.v_proj.weight": (s.num_kv_heads * d, s.hidden_size),
            p + "self_attn.o_proj.weight": (s.hidden_size, s.num_heads * d),
            p + "self_attn.q_norm.weight": (d, ),
            p + "self_attn.k_norm.weight": (d, ),
            p + "mlp.gate_proj.weight": (s.intermediate_size, s.hidden_size),
            p + "mlp.up_proj.weight": (s.intermediate_size, s.hidden_size),
            p + "mlp.down_proj.weight": (s.hidden_size, s.intermediate_size),
        })
    g = torch.Generator().manual_seed(0)
    return {n: torch.randn(shape, generator=g) for n, shape in shapes.items()}


def decoder(tp_rank, tp_size):
    return TPDecoder(SPEC, dtype=torch.float32, device="cpu", max_len=4, tp_rank=tp_rank, tp_size=tp_size)


class Rank:

    def __init__(self):
        self.holdings, self.needs, self.sources, self.dests = {}, {}, {}, {}

    def hold_arena(self, dec, weights):
        for w, pieces in dec.layout_pieces({n: t.shape for n, t in weights.items()}).items():
            self.holdings.setdefault(w, []).extend(pieces)
        self.sources["arena"] = dec.arena.buffer

    def hold_zero3(self, weights, rank, world):
        # The partitions are views into one flat buffer, as ZeRO-3 keeps them once an optimizer is attached.
        windows = {w: zero3_window(t.numel(), world, rank) for w, t in weights.items()}
        flat = torch.cat([weights[w].reshape(-1)[lo:hi] for w, (lo, hi) in windows.items()])
        start = 0
        for w, (lo, hi) in windows.items():
            self.holdings.setdefault(w, []).extend(flat_window_pieces(weights[w].numel(), lo, hi, "zero3"))
            self.sources[f"zero3:{w}"] = flat[start:start + hi - lo]
            start += hi - lo

    def need_arena(self, dec, weights):
        self.needs = dec.layout_pieces({n: t.shape for n, t in weights.items()})
        self.dests["arena"] = dec.arena.buffer


def zero3_window(numel, world_size, rank):
    """The flat range of a ZeRO-3 parameter that ``rank`` holds (ZeRO-3 pads to a multiple of the world size)."""
    part = -(-numel // world_size)
    lo = min(rank * part, numel)
    return lo, min(lo + part, numel)


def lookup(table, name, w):
    return table[f"{name}:{w}"] if f"{name}:{w}" in table else table[name]


def read(ranks, transfer):
    w, src, _, shape, s_name, s_off, s_str, *_ = transfer
    if s_name.startswith(RELAYED):
        source = lookup(ranks[src].dests, s_name[len(RELAYED):], w)
    else:
        source = lookup(ranks[src].sources, s_name, w)
    return local_block(source, shape, s_str, s_off).clone()


def write(ranks, transfer, block):
    w, _, dst, shape, _, _, _, d_name, d_off, d_str, _ = transfer
    local_block(lookup(ranks[dst].dests, d_name, w), shape, d_str, d_off).copy_(block)


def run(ranks, weights, preference=None, buckets=None, planner="rect", stats=None, **plan_options):
    """Plan on every rank's description and move the blocks between the simulated ranks: stage by stage, or with
    ``buckets`` (numbers of weights) step by step as :meth:`RelayoutPlan.execute` pipelines them. Every step reads
    all its blocks before writing any, as an all-to-all does. ``planner="affine"`` plans through the affine IR
    (``stats`` then collects its statistics)."""
    names = sorted(weights)
    shapes = {n: t.shape for n, t in weights.items()}
    if planner == "affine":
        transfers = plan_transfers_affine([(r.holdings, r.needs) for r in ranks],
                                          names,
                                          shapes,
                                          preference,
                                          stats=stats,
                                          **plan_options)
    else:
        everyone = [describe(r.holdings, r.needs, shapes, names) for r in ranks]
        transfers = plan_transfers(everyone, names, preference, **plan_options)
    if buckets is None:
        steps = [[t for t in transfers if t[-1] == s] for s in sorted({t[-1] for t in transfers})]
    else:
        groups = [names[i:i + buckets] for i in range(0, len(names), buckets)]
        per_rank = []
        for r in range(len(ranks)):
            plan = object.__new__(RelayoutPlan)  # the pipelining alone, without a process group
            plan.rank, plan.world_size, plan.transfers, plan._steps = r, len(ranks), transfers, {}
            plan.stages = 1 + max(t[-1] for t in transfers)
            per_rank.append(plan._pipeline(groups))
        # a transfer is in the step lists of both its ranks: execute it from the destination's
        steps = [[
            t for local, _, recvs in (per_rank[r][k] for r in range(len(ranks))) for t in local + sum(recvs, [])
        ] for k in range(len(per_rank[0]))]
        assert len(steps) == len(groups) + max(t[-1] for t in transfers)
    for step in steps:
        blocks = [read(ranks, t) for t in step]
        for t, block in zip(step, blocks):
            write(ranks, t, block)
    return transfers


def loaded(tp_rank, tp_size, weights):
    dec = decoder(tp_rank, tp_size)
    dec.load_weights(weights.items())
    return dec.arena.buffer


def assert_each_element_written_once(ranks, transfers):
    written = {}
    for _, _, dst, shape, *_ in transfers:
        written[dst] = written.get(dst, 0) + shape[0] * shape[1]
    for r, rank in enumerate(ranks):
        needed = sum(p.shape[0] * (p.shape[1] if len(p.shape) > 1 else 1) for pieces in rank.needs.values()
                     for _, p in pieces)
        assert written.get(r, 0) == needed


@pytest.mark.parametrize("planner", ["rect", "affine"])
def test_zero3_to_tensor_parallel_on_other_ranks(planner):
    weights = full_weights()
    ranks = [Rank() for _ in range(4)]
    for r, rank in enumerate(ranks):
        rank.hold_zero3(weights, r, 4)
    new = {2: decoder(0, 2), 3: decoder(1, 2)}  # the rollout runs on ranks 2 and 3 only
    for r, dec in new.items():
        ranks[r].need_arena(dec, weights)
    transfers = run(ranks, weights, planner=planner)
    assert_each_element_written_once(ranks, transfers)
    for r, dec in new.items():
        assert torch.equal(dec.arena.buffer, loaded(r - 2, 2, weights))


@pytest.mark.parametrize("old_tp,new_tp", [(2, 4), (4, 2), (2, 1)])
@pytest.mark.parametrize("planner", ["rect", "affine"])
def test_tensor_parallel_degree_change_from_the_rollout_alone(old_tp, new_tp, planner):
    weights = full_weights()
    world = max(old_tp, new_tp)
    ranks = [Rank() for _ in range(world)]
    for r in range(old_tp):
        old = decoder(r, old_tp)
        old.load_weights(weights.items())
        ranks[r].hold_arena(old, weights)
    new = {r: decoder(r, new_tp) for r in range(new_tp)}
    for r, dec in new.items():
        ranks[r].need_arena(dec, weights)
    transfers = run(ranks, weights, planner=planner)
    assert_each_element_written_once(ranks, transfers)
    for r, dec in new.items():
        assert torch.equal(dec.arena.buffer, loaded(r, new_tp, weights))


@pytest.mark.parametrize("planner", ["rect", "affine"])
def test_lost_rank_is_filled_from_the_trainer_and_survivors_are_preferred(planner):
    weights = full_weights()
    ranks = [Rank() for _ in range(4)]
    for r in range(4):
        ranks[r].hold_zero3(weights, r, 4)
    for r in range(3):  # rank 3's rollout shard is gone
        old = decoder(r, 4)
        old.load_weights(weights.items())
        ranks[r].hold_arena(old, weights)
    new = {r: decoder(r, 2) for r in range(2)}
    for r, dec in new.items():
        ranks[r].need_arena(dec, weights)
    transfers = run(ranks, weights, planner=planner, preference={"arena": 0, "zero3": 1})
    for r, dec in new.items():
        assert torch.equal(dec.arena.buffer, loaded(r, 2, weights))
    from_trainer = {w for w, src, dst, _, s_name, *_ in transfers if s_name == "zero3" and src != dst}
    # Over the network, only what rank 3 alone held comes from the trainer: never a replicated norm the survivors
    # still have (a destination's own trainer partition is a local copy and may be used for anything).
    assert from_trainer and not any("norm" in w for w in from_trainer)


@pytest.mark.parametrize("topology", ["flat", "node", "relay"])
@pytest.mark.parametrize("planner", ["rect", "affine"])
def test_replicas_on_two_nodes_from_the_trainer(topology, planner):
    # Two nodes of four ranks; ZeRO-3 over all eight; the rollout is four TP2 replicas, two per node. Each node holds
    # half of every weight and needs all of it (once per replica): at least the other half has to cross, once.
    weights = full_weights()
    nodes = [0, 0, 0, 0, 1, 1, 1, 1]
    ranks = [Rank() for _ in nodes]
    for r, rank in enumerate(ranks):
        rank.hold_zero3(weights, r, len(nodes))
    new = {r: decoder(r % 2, 2) for r in range(len(nodes))}
    for r, dec in new.items():
        ranks[r].need_arena(dec, weights)
    transfers = run(ranks,
                    weights,
                    planner=planner,
                    nodes=None if topology == "flat" else nodes,
                    relay=topology == "relay",
                    min_relay=1)
    for r, dec in new.items():
        assert torch.equal(dec.arena.buffer, loaded(r % 2, 2, weights))
    inter = sum(shape[0] * shape[1] for _, src, dst, shape, *_ in transfers if nodes[src] != nodes[dst])
    least = sum(t.numel() for t in weights.values())
    if topology == "relay":
        assert inter == least
    else:
        assert inter > least  # every replica fetches its own copy


@pytest.mark.parametrize("tp,buckets,spread", [(2, 3, False), (1, 5, False), (1, 5, True), (2, 1, True)])
@pytest.mark.parametrize("planner", ["rect", "affine"])
def test_relay_pipelined_over_buckets(tp, buckets, spread, planner):
    # The second stage of a bucket rides in the next bucket's all-to-all; it must read what the first stage wrote.
    weights = full_weights()
    nodes = [0, 0, 0, 0, 1, 1, 1, 1]
    ranks = [Rank() for _ in nodes]
    for r, rank in enumerate(ranks):
        rank.hold_zero3(weights, r, len(nodes))
    new = {r: decoder(r % tp, tp) for r in range(len(nodes))}
    for r, dec in new.items():
        ranks[r].need_arena(dec, weights)
    transfers = run(ranks,
                    weights,
                    planner=planner,
                    buckets=buckets,
                    nodes=nodes,
                    relay=True,
                    spread=spread,
                    min_split=1,
                    min_relay=1)
    assert max(t[-1] for t in transfers) == 1
    for r, dec in new.items():
        assert torch.equal(dec.arena.buffer, loaded(r % tp, tp, weights))
    inter = sum(shape[0] * shape[1] for _, src, dst, shape, *_ in transfers if nodes[src] != nodes[dst])
    assert inter == sum(t.numel() for t in weights.values())


@pytest.mark.parametrize("planner", ["rect", "affine"])
def test_spread_has_replicas_fetch_bands_of_a_block(planner):
    # Four TP1 replicas on node 0 move to node 1 with a relay: every rank of node 1 needs the whole embedding. Without
    # spreading one of them fetches it and relays it; with it, each fetches a band of rows and relays its band.
    weights = full_weights()
    nodes = [0, 0, 0, 0, 1, 1, 1, 1]
    ranks = [Rank() for _ in nodes]
    for r in range(4):
        old = decoder(0, 1)
        old.load_weights(weights.items())
        ranks[r].hold_arena(old, weights)
    new = {r: decoder(0, 1) for r in range(4, 8)}
    for r, dec in new.items():
        ranks[r].need_arena(dec, weights)
    fetchers = {}
    for spread in (False, True):
        for dec in new.values():
            dec.arena.buffer.zero_()
        transfers = run(ranks,
                        weights,
                        planner=planner,
                        nodes=nodes,
                        relay=True,
                        spread=spread,
                        min_split=1,
                        min_relay=1)
        for r, dec in new.items():
            assert torch.equal(dec.arena.buffer, loaded(0, 1, weights))
        assert_each_element_written_once(ranks, transfers)
        fetchers[spread] = {
            dst
            for w, src, dst, *_ in transfers if w == "model.embed_tokens.weight" and nodes[src] != nodes[dst]
        }
        inter = sum(shape[0] * shape[1] for _, src, dst, shape, *_ in transfers if nodes[src] != nodes[dst])
        assert inter == sum(t.numel() for t in weights.values())
    assert len(fetchers[False]) == 1 and fetchers[True] == {4, 5, 6, 7}


@pytest.mark.parametrize("old_tp", [4, 1])
@pytest.mark.parametrize("planner", ["rect", "affine"])
def test_relay_skips_small_blocks(old_tp, planner):
    # The rollout on node 0 moves to node 1 from its arenas alone, relaying only blocks of at least min_relay elements
    # (the projections and embeddings here, not the norms). A TP4 replica's shards each cross once whatever the plan:
    # only the norms could be relayed, so the plan keeps one stage. Four TP1 replicas still relay every matrix.
    weights = full_weights()
    nodes = [0, 0, 0, 0, 1, 1, 1, 1]
    ranks = [Rank() for _ in nodes]
    for r in range(4):
        old = decoder(r % old_tp, old_tp)
        old.load_weights(weights.items())
        ranks[r].hold_arena(old, weights)
    new = {r: decoder((r - 4) % old_tp, old_tp) for r in range(4, 8)}
    for r, dec in new.items():
        ranks[r].need_arena(dec, weights)
    transfers = run(ranks, weights, planner=planner, nodes=nodes, relay=True, min_relay=1000)
    for r, dec in new.items():
        assert torch.equal(dec.arena.buffer, loaded((r - 4) % old_tp, old_tp, weights))
    assert_each_element_written_once(ranks, transfers)
    relayed = {w for w, *_, s_name, _, _, _, _, _, _ in transfers if s_name.startswith(RELAYED)}
    assert not any("norm" in w for w in relayed)
    inter = sum(shape[0] * shape[1] for _, src, dst, shape, *_ in transfers if nodes[src] != nodes[dst])
    norms = sum(t.numel() for w, t in weights.items() if "norm" in w)
    if old_tp == 4:
        assert max(t[-1] for t in transfers) == 0 and not relayed
        assert inter == sum(t.numel() for t in weights.values()) + 3 * norms
    else:
        assert relayed and max(t[-1] for t in transfers) == 1
        assert inter == sum(t.numel() for t in weights.values()) + 3 * norms


def test_spread_ignores_preference_across_nodes_only():
    # A TP4 replica on node 0 moves to node 1, from its arenas and the trainer partitions (preferring the arenas).
    # Across nodes, spreading takes the least loaded holders instead, so the trainer partitions of all four node-0
    # ranks serve rank 4 rather than rank 0's arena alone.
    weights = full_weights()
    nodes = [0, 0, 0, 0, 1, 1, 1, 1]
    ranks = [Rank() for _ in nodes]
    for r, rank in enumerate(ranks):
        rank.hold_zero3(weights, r, len(nodes))
    for r in range(4):
        old = decoder(r, 4)
        old.load_weights(weights.items())
        ranks[r].hold_arena(old, weights)
    new = {r: decoder(r - 4, 4) for r in range(4, 8)}
    for r, dec in new.items():
        ranks[r].need_arena(dec, weights)
    senders = {}
    for spread in (False, True):
        transfers = run(ranks, weights, preference={"arena": 0, "zero3": 1}, nodes=nodes, spread=spread)
        for r, dec in new.items():
            assert torch.equal(dec.arena.buffer, loaded(r - 4, 4, weights))
        # the projections, whose TP4 shard 0 only rank 0's arena holds (the replicated norms are in every arena)
        senders[spread] = {src for w, src, dst, *_ in transfers if "proj" in w and dst == 4 and nodes[src] == 0}
    assert senders[False] == {0} and senders[True] == {0, 1, 2, 3}


@pytest.mark.parametrize("planner", ["rect", "affine"])
def test_missing_holder_is_an_error(planner):
    weights = full_weights()
    ranks = [Rank(), Rank()]
    old = decoder(0, 2)  # only half of the old layout survives, and no trainer
    old.load_weights(weights.items())
    ranks[0].hold_arena(old, weights)
    ranks[1].need_arena(decoder(0, 1), weights)
    with pytest.raises(ValueError, match="no holder"):
        run(ranks, weights, planner=planner)


def pairs(piece):
    out = []
    for idx in itertools.product(*[range(n) for n in piece.shape]):
        src = piece.source_offset + sum(i * s for i, s in zip(idx, piece.source_strides))
        dst = piece.dest_offset + sum(i * s for i, s in zip(idx, piece.dest_strides))
        out.append((src, dst))
    return sorted(out)


@pytest.mark.parametrize("seed", range(100))
def test_blocks_cover_exactly_the_piece(seed):
    rng = random.Random(seed)
    rows, cols = rng.randint(1, 9), rng.randint(1, 12)
    if rng.random() < 0.5:  # a flat range (a ZeRO-3 window)
        lo = rng.randint(0, rows * cols - 1)
        hi = rng.randint(lo + 1, rows * cols)
        piece = AffinePiece((hi - lo, ), lo, (1, ), rng.randint(0, 20), (1, ), [0])
    else:  # a row-major block (a tensor-parallel row or column slice) landing in a wider local tensor
        r0, c0 = rng.randint(0, rows - 1), rng.randint(0, cols - 1)
        h, w = rng.randint(1, rows - r0), rng.randint(1, cols - c0)
        piece = AffinePiece((h, w), r0 * cols + c0, (cols, 1), rng.randint(0, 20), (w + rng.randint(0, 3), 1), [0])
    got = []
    for (r0, r1, c0, c1), _, off, (s0, s1) in piece_blocks(piece, "t", (rows, cols)):
        for r, c in itertools.product(range(r0, r1), range(c0, c1)):
            got.append((r * cols + c, off + (r - r0) * s0 + (c - c0) * s1))
    assert sorted(got) == pairs(piece)


# ---------------------------------------------------------------------------------------------- the affine planner


def node_layout(case):
    """Two nodes of four ranks. ``trainer``: ZeRO-3 over node 0 (or both nodes); the rollout on node 1 (or both) in
    replicas of ``tp``. Returns the ranks, nodes, new decoders and the weights."""
    trainer, rollout, tp = case
    weights = full_weights()
    nodes = [0, 0, 0, 0, 1, 1, 1, 1]
    ranks = [Rank() for _ in nodes]
    holders = range(4) if trainer == "node0" else range(8)
    for i, r in enumerate(holders):
        ranks[r].hold_zero3(weights, i, len(holders))
    targets = range(4, 8) if rollout == "node1" else range(8)
    new = {r: decoder(i % tp, tp) for i, r in enumerate(targets)}
    for r, dec in new.items():
        ranks[r].need_arena(dec, weights)
    return ranks, nodes, new, weights


def traffic(transfers, nodes):
    """Elements sent between ranks of one node and across nodes, and received per rank."""
    intra = inter = 0
    received = {}
    for _, src, dst, shape, *_ in transfers:
        n = shape[0] * shape[1]
        if src != dst:
            if nodes[src] == nodes[dst]:
                intra += n
            else:
                inter += n
        received[dst] = received.get(dst, 0) + n
    return intra, inter, received


CASES = [("node0", "node1", 4), ("node0", "node1", 2), ("node0", "node1", 1), ("both", "both", 2), ("both", "both", 1)]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("variant", ["flat", "relay", "relay+spread"])
def test_affine_plans_move_the_bytes_the_rect_plans_move(case, variant):
    # The same declarations through both planners: every arena exact, no weight left to the fallback, and as many
    # bytes across nodes. With a relay, each element crosses into a node that lacks it once, whichever planner.
    options = dict(nodes=[0, 0, 0, 0, 1, 1, 1, 1],
                   relay=variant != "flat",
                   spread="spread" in variant,
                   min_split=1,
                   min_relay=1)
    if variant == "flat":
        options = {}
    results = {}
    for planner in ("rect", "affine"):
        ranks, nodes, new, weights = node_layout(case)
        stats = {}
        transfers = run(ranks, weights, planner=planner, stats=stats, **options)
        for r, dec in new.items():
            tp = case[2]
            assert torch.equal(dec.arena.buffer, loaded(list(new).index(r) % tp, tp, weights))
        if planner == "affine":
            assert stats["fallback"] == {} and stats["plans"] > 0
        results[planner] = traffic(transfers, nodes)
    assert results["affine"][1] == results["rect"][1]
    if variant != "flat":
        # into a node, what it does not hold, once: the whole model in all (half into each node when both hold half)
        assert results["affine"][1] == sum(t.numel() for t in weights.values())


def test_affine_relay_writes_into_the_destinations_themselves():
    # The relay map's shards are bands of the destinations: the first stage writes only into needed pieces, and the
    # second stage's copy of each band onto its own holder is dropped, not executed.
    ranks, nodes, new, weights = node_layout(("node0", "node1", 1))
    stats = {}
    transfers = run(ranks,
                    weights,
                    planner="affine",
                    stats=stats,
                    nodes=nodes,
                    relay=True,
                    spread=True,
                    min_split=1,
                    min_relay=1)
    assert stats["in_place"] > 0
    assert_each_element_written_once(ranks, transfers)
    for w, src, dst, shape, s_name, s_off, s_str, d_name, d_off, d_str, stage in transfers:
        assert d_name == "arena"
        assert not (src == dst and s_name == RELAYED + d_name and s_off == d_off)
        if stage == 1:
            assert s_name == RELAYED + "arena" and nodes[src] == nodes[dst] == 1


def test_affine_falls_back_on_holders_in_two_layouts_and_on_partial_targets():
    weights = full_weights()
    # The trainer partitions and a surviving rollout shard hold one weight with different boundaries.
    ranks = [Rank() for _ in range(2)]
    for r in range(2):
        ranks[r].hold_zero3(weights, r, 2)
    old = decoder(0, 1)
    old.load_weights(weights.items())
    ranks[0].hold_arena(old, weights)
    ranks[1].need_arena(decoder(0, 1), weights)
    stats = {}
    run(ranks, weights, planner="affine", stats=stats, preference={"arena": 0, "zero3": 1})
    assert "different boundaries" in stats["fallback"]["model.layers.0.self_attn.q_proj.weight"]
    # Only half of a TP2 replica is needed (its other rank is kept): a partial target.
    ranks = [Rank() for _ in range(2)]
    for r in range(2):
        ranks[r].hold_zero3(weights, r, 2)
    dec = decoder(1, 2)
    ranks[1].need_arena(dec, weights)
    stats = {}
    run(ranks, weights, planner="affine", stats=stats)
    assert torch.equal(dec.arena.buffer, loaded(1, 2, weights))
    assert "partial target" in stats["fallback"]["model.layers.0.self_attn.q_proj.weight"]
    with pytest.raises(Exception, match="partial target"):
        run(ranks, weights, planner="affine", fallback=False)


def test_affine_refuses_a_contradictory_declaration():
    # Two pieces of one weight declared at the same arena addresses: a map error, never a fallback.
    weights = full_weights()
    ranks = [Rank(), Rank()]
    ranks[0].hold_zero3(weights, 0, 1)
    dec = decoder(0, 1)
    ranks[1].need_arena(dec, weights)
    w = "model.layers.0.self_attn.k_proj.weight"
    (name, piece), = ranks[1].needs[w]
    rows = piece.shape[0] // 2
    half = AffinePiece((rows, piece.shape[1]), piece.source_offset, piece.source_strides, piece.dest_offset,
                       piece.dest_strides, [0])
    other = AffinePiece((piece.shape[0] - rows, piece.shape[1]), piece.source_offset + rows * piece.source_strides[0],
                        piece.source_strides, piece.dest_offset, piece.dest_strides, [0])
    ranks[1].needs[w] = [(name, half), (name, other)]
    with pytest.raises(InvalidMapError, match="same addresses"):
        run(ranks, weights, planner="affine")


@pytest.mark.parametrize("numel,world", [(1000, 4), (4096, 8), (7, 4), (64, 3)])
def test_storage_map_windows_are_the_zero3_windows(numel, world):
    # The holdings NativeRollout and Zero3StateMove now take from the ZeRO storage maps are the windows they computed.
    part = -(-numel // world)
    storage = zero_storage_affine_map(numel, list(range(world)), part)
    for r in range(world):
        pieces = [p for p in storage.pieces_by_rank[r] if p.numel]
        lo, hi = zero3_window(numel, world, r)
        assert [(p.source_offset, p.numel, p.dest_offset) for p in pieces] == ([(lo, hi - lo, 0)] if hi > lo else [])
