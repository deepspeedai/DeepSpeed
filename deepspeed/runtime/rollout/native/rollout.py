# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""A rollout engine on the native decoder, with in-place weight sync and in-memory resharding.

The rollout ranks form replicas of ``tp_size`` consecutive ranks (data parallel over replicas, tensor parallel inside
one). Each rank keeps its slice of the weights in a :class:`WeightArena`, and everything the decode graphs read
keeps its address, so:

* :meth:`NativeRollout.sync_weights` writes the trainer's ZeRO-3 partitions straight into the arenas with one
  :class:`RelayoutPlan` (built at the first sync), and the decode graphs captured before stay valid;
* :meth:`NativeRollout.reconfigure` moves the rollout to other ranks or another TP degree: new groups and arenas,
  the weights moved by a plan from the old arenas (or the ZeRO-3 partitions) to the new ones, and the decode graphs
  captured again at the next generation.

Every rank of the world calls ``sync_weights`` and ``reconfigure`` (they are collectives over the world);
``generate`` is called on the rollout ranks, and a replica decodes the prompts of all its ranks together. The trainer
may run on a subset of the ranks (ZeRO-3 over those alone, e.g. one node, with the rollout on another): the other
ranks pass no engine.
"""

import time
from dataclasses import dataclass
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.zero_affine import zero_param_storage_affine_map
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.native.decoder import DecoderSpec, TPDecoder
from deepspeed.runtime.rollout.native.generator import NativeGenerator
from deepspeed.runtime.rollout.relayout import RelayoutPlan


@dataclass
class NativeRolloutConfig:
    """Configuration of :class:`NativeRollout`."""
    engine: str = "native"
    # Tensor-parallel degree of each replica.
    tp_size: int = 1
    # Rollout ranks (default: all), split into replicas of tp_size consecutive entries.
    ranks: Optional[List[int]] = None
    # Rows a replica decodes at once: the prompts of all its ranks, times n_samples_per_prompt.
    max_batch_size: int = 64
    # Prompt plus response tokens.
    max_len: int = 2048
    use_graph: bool = True
    # Fused elementwise kernels (torch.compile), see TPDecoder.
    fuse: bool = False
    # Drop finished rows during a generation with EOS (see NativeGenerator): the long tail of a batch then decodes at a
    # smaller batch size.
    compact: bool = False
    # End-of-sequence tokens (default: the tokenizer's).
    eos_token_ids: Optional[List[int]] = None
    seed: int = 0
    # Weights per all-to-all when moving weights.
    bucket: int = 64
    # How weight moves see the machine: "flat", "node" (same-node holders first) or "relay" (also relay blocks that
    # crossed nodes inside their node), see RelayoutPlan.
    topology: str = "flat"
    # Run a relay's stages as a pipeline over the buckets (the copies inside a node overlap the next inter-node fetch).
    pipeline: bool = True
    # Spread inter-node fetches: least loaded remote holders first; with a relay, the replicas on a node fetch each
    # large block in bands of rows, one per rank, and relay the bands to each other (every NIC carries a share).
    spread: bool = False
    # Who plans weight moves: "rect" (RelayoutPlan's own planner) or "affine" (the affine IR's plan_transfer, with the
    # same execution), see RelayoutPlan.
    planner: str = "rect"
    # Relayout plans kept for repeated syncs and reconfigurations (keyed by the layouts, the source and the options
    # above); planning on many ranks takes a large part of a reconfiguration.
    plan_cache: int = 16


class TrainerMPU:
    """An ``mpu`` for ``deepspeed.initialize`` when the trainer runs on a subset of the ranks: data parallel (ZeRO)
    over ``ranks``, no model parallelism. Every rank of the world creates it (it makes groups); only the trainer
    ranks pass it to ``deepspeed.initialize``. With several engines in one process (e.g. trainers over different
    ranks, see :class:`~deepspeed.runtime.rollout.zero3_state.Zero3StateMove`), give each its own, also one over
    every rank: DeepSpeed keeps the last engine's mpu in a module global, which an engine without one would take.
    Create the engines over every rank first: ZeRO-3 numbers parameters per process, and an engine's numbers must
    agree over its ranks.
    """

    def __init__(self, ranks: Sequence[int]):
        self.ranks = sorted(ranks)
        self._dp = dist.new_group(self.ranks)
        self._mp = None
        for r in range(dist.get_world_size()):  # new_group needs every rank, so each rank's own group is made by all
            group = dist.new_group([r])
            if r == dist.get_rank():
                self._mp = group

    def get_data_parallel_group(self):
        return self._dp

    def get_data_parallel_world_size(self) -> int:
        return len(self.ranks)

    def get_data_parallel_rank(self) -> int:
        return self.ranks.index(dist.get_rank())

    def get_data_parallel_group_ranks(self) -> List[int]:
        return list(self.ranks)

    def get_model_parallel_group(self):
        return self._mp

    def get_model_parallel_world_size(self) -> int:
        return 1

    def get_model_parallel_rank(self) -> int:
        return 0


class _Layout:
    """Where the rollout runs: its replicas, this rank's group and decoder (None off the rollout). A replica that is
    also one of ``previous``'s (same ranks, same TP degree) keeps its decoder: its arena, KV cache and captured
    decode graphs (``reused``)."""

    def __init__(self, ranks: List[int], tp_size: int, rollout: "NativeRollout", previous: "_Layout" = None):
        if not ranks or len(ranks) % tp_size:
            raise ValueError(f"{len(ranks)} rollout ranks do not split into replicas of {tp_size}")
        self.ranks, self.tp_size = list(ranks), tp_size
        self.replicas = [self.ranks[i:i + tp_size] for i in range(0, len(self.ranks), tp_size)]
        self.group, self.replica = None, None  # a single-rank replica has no group
        t = time.perf_counter()
        for i, members in enumerate(self.replicas):
            group = rollout._group(members)
            if rollout.rank in members:
                self.group, self.replica = group, i
        self.times = {"groups": time.perf_counter() - t}
        t = time.perf_counter()
        self.decoder, self.generator, self.reused = None, None, False
        if (self.replica is not None and previous is not None and previous.replica is not None
                and previous.replicas[previous.replica] == self.replicas[self.replica]):
            self.decoder, self.generator, self.reused = previous.decoder, previous.generator, True
        elif self.replica is not None:
            cfg = rollout.cfg
            self.decoder = TPDecoder(rollout.spec,
                                     group=self.group,
                                     dtype=rollout.dtype,
                                     device=rollout.device,
                                     max_batch_size=cfg.max_batch_size,
                                     max_len=cfg.max_len,
                                     fuse=cfg.fuse)
            self.generator = NativeGenerator(self.decoder, use_graph=cfg.use_graph, compact=cfg.compact)
        self.times["decoder"] = time.perf_counter() - t

    def needs(self, shapes) -> Dict:
        return self.decoder.layout_pieces(shapes) if self.decoder is not None else {}

    def arena(self) -> Dict[str, torch.Tensor]:
        return {"arena": self.decoder.arena.buffer} if self.decoder is not None else {}


class NativeRollout(RolloutEngine):
    """Rollout on the native decoder, synced from a ZeRO-3 engine.

    Args:
        engine: DeepSpeed ZeRO-3 engine wrapping a HuggingFace causal LM of the Llama / Qwen3 family, or None on a
            rank outside the trainer. The trainer ranks are those with an engine; its ZeRO-3 partitions must be over
            exactly them, in rank order.
        tokenizer: its tokenizer (pad and EOS tokens).
        cfg: :class:`NativeRolloutConfig`.
    """

    name = "native"

    def __init__(self, engine, tokenizer, cfg: NativeRolloutConfig):
        self.cfg = cfg
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.device = torch.device(get_accelerator().current_device_name())
        self.names = None
        self._plans = OrderedDict()
        self.bind(engine)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        eos = cfg.eos_token_ids if cfg.eos_token_ids is not None else [tokenizer.eos_token_id]
        self.eos_token_ids = [t for t in eos if t is not None]
        ranks = cfg.ranks if cfg.ranks is not None else list(range(self.world))
        self._groups = {}
        self.layout = _Layout(ranks, cfg.tp_size, self)
        self._calls = 0

    def bind(self, engine) -> None:
        """Sync from ``engine`` from now on (every rank calls it; None on ranks outside the new trainer): another
        ZeRO-3 engine over other ranks holding the same model, e.g. after moving the training state to it. The
        plans of both trainers stay cached."""
        self.entries = []
        mine = None
        if engine is not None:
            module = engine.module
            # named_parameters() gives each distinct weight once, in the same order on every rank.
            for name, param in module.named_parameters():
                if "lora_" in name or ".base_layer." in name or not hasattr(param, "ds_tensor"):
                    raise NotImplementedError(f"{name}: NativeRollout syncs ZeRO-3 parameters only (no LoRA)")
                self.entries.append((name, param))
            group = self.entries[0][1].ds_process_group
            mine = (DecoderSpec.from_hf_config(module.config), [n for n, _ in self.entries], {
                n: tuple(p.ds_shape)
                for n, p in self.entries
            }, self.entries[0][1].ds_tensor.dtype, dist.get_rank(group),
                    {dist.get_world_size(group=p.ds_process_group)
                     for _, p in self.entries})
        # The ranks without an engine learn the model from the trainer's.
        everyone = [None] * self.world
        dist.all_gather_object(everyone, mine)
        self.trainer_ranks = [r for r, m in enumerate(everyone) if m is not None]
        if not self.trainer_ranks:
            raise ValueError("no rank passed a ZeRO-3 engine")
        for i, r in enumerate(self.trainer_ranks):
            if everyone[r][4] != i or everyone[r][5] != {len(self.trainer_ranks)}:
                raise NotImplementedError(f"rank {r}'s ZeRO-3 partitions are not over the trainer ranks "
                                          f"{self.trainer_ranks} in rank order")
        spec, names, shapes, dtype = everyone[self.trainer_ranks[0]][:4]
        if self.names is not None and (names, shapes, dtype) != (self.names, self.shapes, self.dtype):
            raise ValueError("the new engine holds a different model")
        self.spec, self.names, self.shapes, self.dtype = spec, names, shapes, dtype
        self._sync_plan = None
        self._zero3_holdings = None

    def _group(self, members: Sequence[int]):
        """The process group of a replica, created on first use and kept (None for a single rank). Every rank calls it
        for every replica of a layout, in the same order, as ``new_group`` requires."""
        key = tuple(members)
        if len(key) == 1:
            return None
        if key not in self._groups:
            self._groups[key] = dist.new_group(list(key))
        return self._groups[key]

    def prepare(self, layouts: Sequence[Tuple[Sequence[int], int]]) -> None:
        """Create the groups of the given ``(ranks, tp_size)`` layouts now (every rank calls it with the same list), so
        that reconfiguring to them later does not."""
        for ranks, tp_size in layouts:
            ranks = list(ranks)
            for i in range(0, len(ranks), tp_size):
                self._group(ranks[i:i + tp_size])

    def _plan(self, key: tuple, build) -> Tuple[RelayoutPlan, bool]:
        """The cached plan for ``key`` or a new one from ``build()``, and whether it was cached. Every rank asks for
        the same keys in the same order, so the caches stay alike and a miss builds the plan on every rank."""
        key = key + (tuple(self.trainer_ranks), self.cfg.topology, self.cfg.spread, self.cfg.planner)
        plan = self._plans.pop(key, None)
        cached = plan is not None
        if plan is None:
            plan = build()
        self._plans[key] = plan
        while len(self._plans) > self.cfg.plan_cache:
            self._plans.popitem(last=False)
        return plan, cached

    @staticmethod
    def _key(layout: "_Layout") -> tuple:
        return (tuple(layout.ranks), layout.tp_size)

    # ------------------------------------------------------------------ weights

    def _zero3(self):
        """This rank's ZeRO-3 holdings, from the parameters' affine storage maps, and the tensors behind them (none
        outside the trainer). The maps are made once per engine; the tensors are the partitions themselves, read
        directly: ``zero_param_storage_shard`` refuses a parameter ZeRO-3 keeps gathered (a persistent one), which
        guards a writer, while a sync only reads the partition."""
        if self._zero3_holdings is None:
            holdings = {}
            for n, p in self.entries:
                pieces = zero_param_storage_affine_map(p).pieces_by_rank.get(self.rank, [])
                holdings[n] = [("zero3", piece) for piece in pieces if piece.numel]
            self._zero3_holdings = holdings
        sources = {}
        for n, p in self.entries:
            held = sum(piece.numel for _, piece in self._zero3_holdings[n])
            sources[f"zero3:{n}"] = p.ds_tensor.reshape(-1)[:held]
        return self._zero3_holdings, sources

    def _move(self, plan: RelayoutPlan, sources, dests) -> None:
        buckets = [self.names[i:i + self.cfg.bucket] for i in range(0, len(self.names), self.cfg.bucket)]
        if self.cfg.pipeline:
            plan.execute(sources, dests, buckets=buckets, dtype=self.dtype, device=self.device)
            return
        for bucket in buckets:
            plan.execute(sources, dests, weights=bucket, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def sync_weights(self, step: int) -> None:
        """Write the ZeRO-3 partitions into the arenas in place (every rank calls it)."""
        holdings, sources = self._zero3()
        if self._sync_plan is None:
            self._sync_plan, _ = self._plan(("sync", ) + self._key(self.layout),
                                            lambda: RelayoutPlan(holdings,
                                                                 self.layout.needs(self.shapes),
                                                                 self.shapes,
                                                                 self.names,
                                                                 topology=self.cfg.topology,
                                                                 spread=self.cfg.spread,
                                                                 planner=self.cfg.planner))
        self._move(self._sync_plan, sources, self.layout.arena())

    @torch.no_grad()
    def reconfigure(self, ranks: Sequence[int], tp_size: int, source: str = "rollout") -> RelayoutPlan:
        """Move the rollout to ``ranks`` in replicas of ``tp_size`` (every rank calls it); returns the plan used.

        ``source="rollout"`` moves the weights from the current arenas, which must hold the latest sync;
        ``"trainer"`` takes them from the ZeRO-3 partitions instead (e.g. when a rollout rank was lost); ``"both"``
        lets the plan pick either (the arenas when equally close, which with a node topology means a new replica can
        fill most of its arena from the trainer partitions on its own node).
        ``self.last_reconfigure`` has the phases' times: ``layout`` (groups, arenas), ``plan`` (0 when the plan was
        cached, see ``plan_cached``) and ``move``. :meth:`prepare` creates the groups of expected layouts beforehand.
        A replica present in both layouts (same ranks and TP degree) keeps its decoder, arena and decode graphs
        (``reused``); from ``"rollout"`` or ``"both"`` it receives nothing.
        """
        if source not in ("rollout", "trainer", "both"):
            raise ValueError(f"unknown source {source!r}; choose from 'rollout', 'trainer', 'both'")
        acc = get_accelerator()
        # NCCL allocates outside the caching allocator: give back what the training step left cached first.
        acc.empty_cache()
        times = {}
        t = time.perf_counter()
        old, new = self.layout, _Layout(list(ranks), tp_size, self, previous=self.layout)
        acc.synchronize()
        times["layout"] = time.perf_counter() - t
        times.update(new.times)
        times["reused"] = float(new.reused)
        t = time.perf_counter()
        sources, preference = {}, None
        if source in ("rollout", "both"):
            sources.update(old.arena())
        if source in ("trainer", "both"):
            z_holdings, z_sources = self._zero3()
            sources.update(z_sources)
            preference = {"arena": 0, "zero3": 1}

        def build(old=old, new=new):
            holdings = dict(old.needs(self.shapes)) if source in ("rollout", "both") else {}
            if source in ("trainer", "both"):
                for w, pieces in z_holdings.items():
                    holdings[w] = holdings.get(w, []) + pieces
            return RelayoutPlan(
                holdings,
                # a kept replica already holds the rollout's weights
                {} if new.reused and source != "trainer" else new.needs(self.shapes),
                self.shapes,
                self.names,
                preference=preference,
                topology=self.cfg.topology,
                spread=self.cfg.spread,
                planner=self.cfg.planner)

        plan, times["plan_cached"] = self._plan(("reconfigure", ) + self._key(old) + self._key(new) + (source, ),
                                                build)
        times["plan"] = time.perf_counter() - t
        t = time.perf_counter()
        self._move(plan, sources, new.arena())
        acc.synchronize()
        times["move"] = time.perf_counter() - t
        self.layout = new
        self._sync_plan = None
        self.last_reconfigure = times
        del old
        acc.empty_cache()
        return plan

    # ------------------------------------------------------------------ generation

    def _gather_rows(self, ids: torch.Tensor, mask: torch.Tensor):
        """The replica's rows (its ranks' prompts in group-rank order, left-padded to a common width) and where this
        rank's rows start."""
        group, tp = self.layout.group, self.layout.tp_size
        if tp == 1:
            return ids, mask, 0
        meta = torch.tensor(ids.shape, dtype=torch.long, device=self.device)
        metas = [torch.empty_like(meta) for _ in range(tp)]
        dist.all_gather(metas, meta, group=group)
        rows = [int(m[0]) for m in metas]
        width = max(int(m[1]) for m in metas)
        padded_ids = torch.full((max(rows), width), self.pad_token_id, dtype=ids.dtype, device=self.device)
        padded_mask = torch.zeros(max(rows), width, dtype=mask.dtype, device=self.device)
        padded_ids[:ids.shape[0], width - ids.shape[1]:] = ids
        padded_mask[:ids.shape[0], width - ids.shape[1]:] = mask
        all_ids = [torch.empty_like(padded_ids) for _ in range(tp)]
        all_mask = [torch.empty_like(padded_mask) for _ in range(tp)]
        dist.all_gather(all_ids, padded_ids, group=group)
        dist.all_gather(all_mask, padded_mask, group=group)
        start = sum(rows[:dist.get_rank(group)])
        ids = torch.cat([x[:n] for x, n in zip(all_ids, rows)])
        mask = torch.cat([x[:n] for x, n in zip(all_mask, rows)])
        return ids, mask, start

    @torch.no_grad()
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        if sampling.top_p < 1.0 or sampling.top_k > 0:
            raise NotImplementedError("NativeRollout samples from the full distribution (top_p=1, top_k<=0)")
        gen = self.layout.generator
        if gen is None:
            raise RuntimeError(f"rank {self.rank} is not a rollout rank")
        n = sampling.n_samples_per_prompt
        ids = request.prompt_ids.to(self.device).repeat_interleave(n, 0)
        mask = request.prompt_attention_mask.to(self.device).long().repeat_interleave(n, 0)
        mine = ids.shape[0]
        ids, mask, start = self._gather_rows(ids, mask)
        N = sampling.max_new_tokens
        # The ranks of a replica must draw the same samples; replicas and calls draw different ones. The training
        # RNG stream is left as it was.
        acc = get_accelerator()
        rng = acc.get_rng_state()
        seed = self.cfg.seed + 1000003 * self.layout.replica + 7919 * self._calls
        self._calls += 1
        tokens, logprobs, lengths = [], [], []
        try:
            for lo in range(0, ids.shape[0], self.cfg.max_batch_size):
                hi = lo + self.cfg.max_batch_size
                t, lp, ln = gen.generate(ids[lo:hi],
                                         mask[lo:hi],
                                         N,
                                         temperature=sampling.temperature,
                                         eos_token_ids=self.eos_token_ids,
                                         seed=seed + lo)
                # A chunk whose rows all ended early comes back shorter.
                tokens.append(torch.nn.functional.pad(t, (0, N - t.shape[1]), value=self.pad_token_id))
                logprobs.append(torch.nn.functional.pad(lp, (0, N - lp.shape[1])))
                lengths.append(ln)
        finally:
            acc.set_rng_state(rng)
        rows = slice(start, start + mine)
        tokens, logprobs, lengths = torch.cat(tokens)[rows], torch.cat(logprobs)[rows], torch.cat(lengths)[rows]
        width = int(lengths.max()) if mine else 0
        valid = torch.arange(width, device=self.device).view(1, -1) < lengths.view(-1, 1)
        response = tokens[:, :width].masked_fill(~valid, self.pad_token_id)
        prompt_ids, prompt_mask = ids[rows], mask[rows]
        return RolloutBatch(input_ids=torch.cat([prompt_ids, response], dim=1),
                            attention_mask=torch.cat([prompt_mask, valid.long()], dim=1),
                            response_start_idx=torch.full((mine, ),
                                                          prompt_ids.shape[1],
                                                          dtype=torch.long,
                                                          device=self.device),
                            response_logprobs=logprobs[:, :width].masked_fill(~valid, 0.0))

    def shutdown(self) -> None:
        self.layout = None
        self._sync_plan = None
        self._plans.clear()
        get_accelerator().empty_cache()
