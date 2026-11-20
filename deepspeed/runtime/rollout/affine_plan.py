# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Planning relayouts through the affine IR (:func:`~deepspeed.checkpoint.affine_transfer.plan_transfer`).

:func:`~.relayout.plan_transfers` plans every move itself, over rectangles of each weight's 2-D view. This module
plans the same moves through the affine IR and returns transfers in the same form, so
:class:`~.relayout.RelayoutPlan` executes either (``planner="affine"``). Per weight, the declared pieces become two
:class:`~deepspeed.checkpoint.affine.ParamAffineMap` s, of what is held and of what is needed, and ``plan_transfer``
composes them, with its map checks on every plan.

Locations. A map has one shard per rank; here a location is a (rank, local tensor) pair, and its shard is the run of
that tensor which the location's pieces of the weight occupy, described as flat storage, as ZeRO's is: each piece a
box of the weight stored densely. A ZeRO-3 window (one flat range of the flattened weight, as the storage maps of
:mod:`deepspeed.checkpoint.zero_affine` describe it) is cut into boxes by
:func:`~deepspeed.checkpoint.affine.nest_affine_maps`, as AutoTP's storage maps are.

Holders. ``plan_transfer`` accepts several holders of a block when they hold it with the same boundaries (replicas).
The rollout's surviving shards and the trainer's partitions of one weight do not, and a weight whose needed blocks do
not cover it (a partial target) has no full target map; such weights are planned by :func:`~.relayout.plan_transfers`
instead, and ``stats["fallback"]`` says which and why. Among replicas, the ``replica_selector`` keeps the order of
:func:`~.relayout.plan_transfers`: the destination itself, then its node, then ``preference``, then the least loaded
holder.

Relay (RFC #9026). With ``relay`` and nodes, a node that does not hold all of a weight receives it through a relay map
with one holder per element: a first plan moves every element into one rank of that node that needs it (what the node
holds from its own ranks, the rest across nodes, once), and a second plan spreads it inside the node. A rank's relay shard is the band of its own destination that it receives, a
view of the destination rather than a separate buffer, so the relay costs no memory, every relay shard is a whole
shard (``plan_transfer`` needs no partial targets for it), and the second plan's copy of a band onto its own holder is
the same memory and is dropped. With ``spread`` a block of at least ``min_split`` elements needed by several ranks of
the node is cut into bands along axis 0, one per rank; otherwise one rank, rotating over the weights, receives it
whole. A weight whose blocks on that node are all smaller than ``min_relay`` elements is fetched directly.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from deepspeed.checkpoint.affine import AffinePiece, ParamAffineMap, nest_affine_maps, replicated_map
from deepspeed.checkpoint.affine_transfer import (InvalidMapError, ReplicaSelectionRequest, UnsupportedLayoutError,
                                                  plan_transfer)
from deepspeed.runtime.rollout.relayout import RELAYED, describe, plan_transfers

# (origin, extent) of a box of the logical weight
Box = Tuple[Tuple[int, ...], Tuple[int, ...]]
# (rank, local tensor name, offset of the shard's first element in the local tensor)
Location = Tuple[int, str, int]


def _product(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _row_major(shape: Sequence[int]) -> Tuple[int, ...]:
    strides, step = [], 1
    for d in reversed(tuple(shape)):
        strides.append(step)
        step *= d
    return tuple(reversed(strides))


def _dense(strides: Sequence[int], shape: Sequence[int]) -> bool:
    return all(size <= 1 or got == want for size, got, want in zip(shape, strides, _row_major(shape)))


def _box(piece: AffinePiece, shape: Sequence[int]) -> Box:
    """The box of the logical weight a piece covers; refuses a piece that is not one."""
    if len(piece.shape) != len(shape) or not all(
            size <= 1 or got == want for size, got, want in zip(piece.shape, piece.source_strides, _row_major(shape))):
        raise UnsupportedLayoutError(f"{piece!r} is not a box of a {tuple(shape)} weight")
    origin, rest = [], piece.source_offset
    for stride in _row_major(shape):
        q, rest = divmod(rest, stride)
        origin.append(q)
    if any(o + e > d for o, e, d in zip(origin, piece.shape, shape)):
        raise InvalidMapError(f"{piece!r} reaches outside a {tuple(shape)} weight")
    return tuple(origin), tuple(piece.shape)


def _overlap(a: Box, b: Box) -> bool:
    return all(max(ao, bo) < min(ao + ae, bo + be) for ao, ae, bo, be in zip(a[0], a[1], b[0], b[1]))


def _piece(box: Box, shape: Sequence[int], dest_offset: int, locations) -> AffinePiece:
    origin, extent = box
    strides = _row_major(shape)
    return AffinePiece(extent,
                       sum(o * s for o, s in zip(origin, strides)),
                       strides,
                       dest_offset,
                       _row_major(extent),
                       locations=locations)


class _Side:
    """One side (held or needed) of one weight as a map over locations."""

    def __init__(self, shape: Tuple[int, ...], locations: List[Location], boxes: Dict[int, List[Tuple[Box, int]]]):
        self.shape, self.locations = shape, locations
        self.boxes = boxes  # location index -> [(box, offset in the location's shard)]
        holders: Dict[Box, set] = {}
        for loc, held in boxes.items():
            for box, _ in held:
                holders.setdefault(box, set()).add(loc)
        self.holders = holders
        distinct = sorted(holders)
        for i, a in enumerate(distinct):
            for b in distinct[i + 1:]:
                if _overlap(a, b):
                    raise UnsupportedLayoutError(f"blocks {a} and {b} of one weight are held with different "
                                                 "boundaries (holders in different layouts)")
        self.full = sum(_product(extent) for _, extent in distinct) == _product(shape)

    def held_on(self, node: int, nodes: Sequence[int]) -> int:
        """Elements of the weight held on ``node``."""
        boxes = {box for loc, held in self.boxes.items() if nodes[self.locations[loc][0]] == node for box, _ in held}
        return sum(_product(extent) for _, extent in boxes)

    def affine_map(self, keep=None) -> ParamAffineMap:
        """The map over the locations in ``keep`` (default all), keyed by location index."""
        keep = set(self.boxes) if keep is None else set(keep)
        shards, pieces = {}, {}
        for loc in sorted(keep):
            held = self.boxes[loc]
            shards[loc] = (sum(_product(box[1]) for box, _ in held), )
            pieces[loc] = [_piece(box, self.shape, offset, self.holders[box] & keep) for box, offset in held]
        return ParamAffineMap(self.shape, shards, pieces)


def _side(everyone, w: str, shape: Tuple[int, ...], index: int) -> Optional[_Side]:
    """The held (``index`` 0) or needed (1) side of ``w`` from every rank's declarations, or None if empty."""
    per_location: Dict[Tuple[int, str], List[AffinePiece]] = {}
    for r, declared in enumerate(everyone):
        for local, piece in declared[index].get(w, []):
            if piece.numel:
                per_location.setdefault((r, local), []).append(piece)
    if not per_location:
        return None
    keys = sorted(per_location)
    locations: List[Location] = []
    boxes: Dict[int, List[Tuple[Box, int]]] = {}
    flat: Dict[int, AffinePiece] = {}
    for i, (r, local) in enumerate(keys):
        pieces = per_location[(r, local)]
        if len(shape) > 1 and all(len(p.shape) == 1 for p in pieces):
            # a window of the flattened weight (ZeRO storage): cut into boxes below, with the others like it
            if len(pieces) != 1 or pieces[0].source_strides != (1, ) or pieces[0].dest_strides != (1, ):
                raise UnsupportedLayoutError(f"rank {r} holds {w!r} in {local!r} as {len(pieces)} strided runs")
            flat[i] = pieces[0]
            locations.append((r, local, pieces[0].dest_offset))
            continue
        runs = []
        for p in pieces:
            if not _dense(p.dest_strides, p.shape):
                raise UnsupportedLayoutError(f"{p!r} of {w!r} is not stored densely in {local!r} on rank {r}")
            runs.append((p.dest_offset, p.dest_offset + p.numel, _box(p, shape)))
        runs.sort()
        for (_, end, _), (start, _, _) in zip(runs, runs[1:]):
            if start < end:
                raise InvalidMapError(f"two pieces of {w!r} occupy the same addresses of {local!r} on rank {r}")
            if start > end:
                raise UnsupportedLayoutError(f"the pieces of {w!r} in {local!r} on rank {r} are not one run")
        base = runs[0][0]
        locations.append((r, local, base))
        boxes[i] = [(box, start - base) for start, _, box in runs]
    if flat:
        numel = _product(shape)
        inner = ParamAffineMap((numel, ), {
            i: (p.numel, )
            for i, p in flat.items()
        }, {
            i: [AffinePiece((p.numel, ), p.source_offset, (1, ), 0, (1, ), locations=[i])]
            for i, p in flat.items()
        })
        try:
            nested = nest_affine_maps(replicated_map(shape, 1), {0: inner})
        except ValueError as error:
            raise UnsupportedLayoutError(f"the flat windows of {w!r} do not store all of it: {error}") from error
        for i, pieces in nested.pieces_by_rank.items():
            boxes[i] = [(_box(p, shape), p.dest_offset) for p in pieces if p.numel]
    return _Side(shape, locations, boxes)


class _Selector:
    """``replica_selector`` keeping :func:`~.relayout.plan_transfers`' order; charges the bytes it assigns."""

    def __init__(self, sources: _Side, targets: Dict[int, Location], nodes, preference, spread, load):
        self.sources, self.targets = sources, targets
        self.nodes, self.preference, self.spread, self.load = nodes, preference, spread, load
        self.charged = [0] * len(load)

    def __call__(self, request: ReplicaSelectionRequest) -> int:
        dst = self.targets[request.target_rank][0]

        def order(candidate):
            r, local, _ = self.sources.locations[candidate.source_rank]
            remote = self.nodes[r] != self.nodes[dst]
            preferred = 0 if self.spread and remote else self.preference.get(local, 0)
            return (r != dst, remote, preferred, self.load[r], r, candidate.source_rank)

        chosen = min(request.candidates, key=order).source_rank
        r = self.sources.locations[chosen][0]
        if r != dst:
            n = _product(request.shape)
            self.load[r] += n
            self.charged[r] += n
        return chosen


def _two_d(shape, source_strides, target_strides):
    """A segment's shape and strides in the 2-D form :class:`~.relayout.RelayoutPlan` executes."""
    if len(shape) == 2:
        return tuple(shape), tuple(source_strides), tuple(target_strides)
    if len(shape) == 1:
        n = shape[0]
        return (1, n), (n * source_strides[0], source_strides[0]), (n * target_strides[0], target_strides[0])
    raise UnsupportedLayoutError(f"a {len(shape)}-axis segment")


def _transfers(w,
               target_map,
               targets: Dict[int, Location],
               source_map,
               sources: Dict[int, Location],
               stage,
               selector: _Selector,
               stats,
               relayed: str = "") -> List[tuple]:
    """``plan_transfer(target_map, source_map)`` as transfers; ``relayed`` prefixes the source tensors' names (the
    sources of a relay's second plan are destinations)."""
    plan = plan_transfer(target_map, source_map, replica_selector=selector)
    stats["plans"] += 1
    stats["segments"] += plan.statistics.segments
    stats["segments_before_merge"] += plan.statistics.segments_before_merge
    out, sent = [], [0] * len(selector.load)
    for seg in plan.segments:
        sr, s_local, s_base = sources[seg.source_rank]
        dr, d_local, d_base = targets[seg.target_rank]
        shape, s_str, d_str = _two_d(seg.shape, seg.source_strides, seg.target_strides)
        s_off, d_off = s_base + seg.source_offset, d_base + seg.target_offset
        if relayed and sr == dr and s_local == d_local and s_off == d_off and s_str == d_str:
            stats["in_place"] += 1  # a band onto its own holder: the same memory
            continue
        out.append((w, sr, dr, shape, relayed + s_local, s_off, s_str, d_local, d_off, d_str, stage))
        if sr != dr:
            sent[sr] += shape[0] * shape[1]
    for r, n in enumerate(sent):  # what the selector did not choose (single holders) is charged here
        selector.load[r] += n - selector.charged[r]
        selector.charged[r] = 0
    return out


def _relay_map(targets: _Side, keep: Sequence[int], shape, turn: int, spread: bool, min_split: int):
    """A map with one holder per element over the target locations ``keep`` (one node): each needed block goes to
    one of the ranks that need it, cut into bands along axis 0 over them with ``spread``. Returns the map and its
    locations, each a band of its holder's destination."""
    by_box: Dict[Box, List[int]] = {}
    for loc in keep:
        for box, _ in targets.boxes[loc]:
            by_box.setdefault(box, []).append(loc)
    relay_locations: Dict[int, Location] = {}
    shards, pieces = {}, {}
    for box in sorted(by_box):
        holders = sorted(by_box[box], key=lambda loc: targets.locations[loc][0])
        k = turn % len(holders)
        holders = holders[k:] + holders[:k]
        origin, extent = box
        rows = extent[0] if extent else 1
        n = min(len(holders), rows) if spread and _product(extent) >= min_split else 1
        cuts = [rows * i // n for i in range(n + 1)]
        row = _product(extent[1:])
        for (a, b), loc in zip(zip(cuts, cuts[1:]), holders):
            r, local, base = targets.locations[loc]
            offset = next(o for bx, o in targets.boxes[loc] if bx == box)
            band = ((origin[0] + a, ) + origin[1:], (b - a, ) + extent[1:])
            i = len(relay_locations)
            relay_locations[i] = (r, local, base + offset + a * row)
            shards[i] = (_product(band[1]), )
            pieces[i] = [_piece(band, shape, 0, [i])]
    return ParamAffineMap(shape, shards, pieces), relay_locations


def plan_transfers_affine(everyone,
                          weights: Sequence[str],
                          shapes: Dict[str, Sequence[int]],
                          preference: Optional[Dict[str, int]] = None,
                          nodes: Optional[Sequence[int]] = None,
                          relay: bool = False,
                          spread: bool = False,
                          min_split: int = 1 << 16,
                          min_relay: int = 1 << 16,
                          fallback: bool = True,
                          stats: Optional[dict] = None) -> List[tuple]:
    """Transfers covering every needed piece, planned through ``plan_transfer``.

    ``everyone[r]`` is rank ``r``'s ``(holdings, needs)``, each ``{weight: [(local tensor, AffinePiece)]}`` as
    :class:`~.relayout.RelayoutPlan` takes them (a piece's ``locations`` are not read: holders are whoever declares a
    block). The other arguments and the transfers are those of :func:`~.relayout.plan_transfers`. A weight the affine
    path does not express is planned by :func:`~.relayout.plan_transfers` when ``fallback`` (else the error is
    raised); ``stats``, when given, collects the plans, segments, in-place relay segments and fallbacks.
    """
    preference = preference or {}
    world_size = len(everyone)
    nodes = list(nodes) if nodes is not None else [0] * world_size
    stats = stats if stats is not None else {}
    for key in ("plans", "segments", "segments_before_merge", "in_place"):
        stats.setdefault(key, 0)
    stats.setdefault("fallback", {})
    load = [0] * world_size
    transfers = []
    for turn, w in enumerate(weights):
        shape = tuple(shapes[w])
        try:
            if len(shape) > 2:
                raise UnsupportedLayoutError(f"a {len(shape)}-axis weight")
            targets = _side(everyone, w, shape, 1)
            if targets is None:
                continue
            sources = _side(everyone, w, shape, 0)
            if sources is None:
                raise ValueError(f"{w!r} is needed but no rank holds it")
            if not targets.full:
                raise UnsupportedLayoutError("the needed blocks do not cover the weight (a partial target)")
            if not sources.full:
                raise UnsupportedLayoutError("the held blocks do not cover the weight")
            planned = _plan_weight(w, shape, sources, targets, nodes, preference, relay, spread, min_split, min_relay,
                                   turn, load, stats)
        except UnsupportedLayoutError as error:
            if not fallback:
                raise
            stats["fallback"][w] = str(error)
            planned = plan_transfers([describe(h, n, shapes, [w]) for h, n in everyone], [w],
                                     preference,
                                     nodes=nodes,
                                     relay=relay,
                                     spread=spread,
                                     min_split=min_split,
                                     min_relay=min_relay)
        transfers += planned
    return transfers


def _plan_weight(w, shape, sources: _Side, targets: _Side, nodes, preference, relay, spread, min_split, min_relay,
                 turn, load, stats) -> List[tuple]:
    by_node: Dict[int, List[int]] = {}
    for loc in sorted(targets.boxes):
        by_node.setdefault(nodes[targets.locations[loc][0]], []).append(loc)
    numel = _product(shape)
    relayed, direct = [], []
    for node, keep in sorted(by_node.items()):
        largest = max(_product(box[1]) for loc in keep for box, _ in targets.boxes[loc])
        covers = sum(_product(e) for o, e in {box for loc in keep for box, _ in targets.boxes[loc]})
        # Relay when part of the weight must cross into this node. The first plan takes what the node already holds
        # from its own ranks (the selector prefers them), so only the rest crosses, once.
        if relay and sources.held_on(node, nodes) < numel and largest >= min_relay and covers == numel:
            relayed.append(keep)
        else:
            direct += keep
    source_map = sources.affine_map()
    source_locations = dict(enumerate(sources.locations))
    target_locations = dict(enumerate(targets.locations))
    out = []
    if direct:
        covered = sum(_product(e) for o, e in {box for loc in direct for box, _ in targets.boxes[loc]})
        if covered != numel:
            raise UnsupportedLayoutError("the blocks needed on nodes fetching directly do not cover the weight")
        selector = _Selector(sources, target_locations, nodes, preference, spread, load)
        out += _transfers(w, targets.affine_map(direct), target_locations, source_map, source_locations, 0, selector,
                          stats)
    for keep in relayed:
        relay_map, relay_locations = _relay_map(targets, keep, shape, turn, spread, min_split)
        selector = _Selector(sources, relay_locations, nodes, preference, spread, load)
        out += _transfers(w, relay_map, relay_locations, source_map, source_locations, 0, selector, stats)
        relay_sources = _Side(shape, [relay_locations[i] for i in sorted(relay_locations)],
                              {i: [(_box(p, shape), 0) for p in relay_map.pieces_by_rank[i]]
                               for i in relay_locations})
        selector = _Selector(relay_sources, target_locations, nodes, preference, spread, load)
        out += _transfers(w,
                          targets.affine_map(keep),
                          target_locations,
                          relay_map,
                          relay_locations,
                          1,
                          selector,
                          stats,
                          relayed=RELAYED)
    return out
