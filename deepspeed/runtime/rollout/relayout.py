# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Moving weights between two arbitrary sharded layouts in GPU memory.

A rollout whose membership or tensor-parallel degree changes, or a trainer that moves to other ranks, needs a move
between two layouts that share nothing by construction: the destination ranks may differ from the source ranks, a
weight may have several sources (the trainer's ZeRO-3 partitions, the surviving shards of the previous rollout
layout, replicated norms), and some ranks only send or only receive.

Both sides are described the same way: per rank, ``{weight: [(local tensor name, AffinePiece)]}``, where a piece
maps a block of the full (logical) weight to a block of a local tensor (a ZeRO-3 partition, an inference arena). Local
offsets count from the local tensor's first element, which need not be its storage's (ZeRO-3 partitions are views
into one flat buffer once an optimizer is attached).
Every piece is turned into rectangles of the weight's 2-D view (rows x last dim; a 1-D weight is one row), so ZeRO-3
flat windows and tensor-parallel row or column slices are the same kind of object, and intersecting a source with a
destination is intersecting two rectangles.

Planning runs identically on every rank from the gathered descriptions. Each destination rectangle is covered by the
holders in order of preference: a holder on the destination rank itself first (a local copy), then (with a node
topology) a holder on the destination's node, then by the layout order the caller gives (e.g. surviving rollout
shards before the trainer) and the bytes already assigned to the holder's rank, so replicated blocks are spread over
their holders. Execution is one bucketed ``all_to_all_single`` per stage over a group containing every sender and
receiver; ranks with nothing to move take part with empty buffers.

Across nodes, several ranks of one node often need the same block (one per rollout replica on the node), and each
fetching it over the network multiplies the inter-node bytes. With ``topology="relay"`` a block that crossed nodes
becomes a holder on its receiving rank for a second stage, so the other ranks of that node copy it over the node's
own links; which rank of a node fetches first rotates over the weights, spreading the network traffic over the
node's ranks (and their NICs). Executed over buckets of weights, the stages form a pipeline: one all-to-all carries
the first stage of a bucket together with the second stage of the bucket before, so the copies inside a node overlap
the next fetch across nodes.

One flow between two ranks of different nodes gets a fraction of the network a node has, and a relay that lets one
rank fetch a whole large weight (an embedding) loads that rank's link alone. With ``spread=True`` blocks fetched
across nodes come from the least loaded remote holders regardless of ``preference``, and replicas on a node fetch
each large block in bands of rows, one band per rank, relaying the bands to each other.
"""

import socket
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine import AffinePiece

# (row0, row1, col0, col1) in the weight's 2-D view
Rect = Tuple[int, int, int, int]
# (rect, local tensor name, local offset of (row0, col0), local strides (row, col))
Block = Tuple[Rect, str, int, Tuple[int, int]]


def view_2d(shape: Sequence[int]) -> Tuple[int, int]:
    """``(rows, cols)`` of a weight's 2-D view: all leading dims flattened into rows."""
    cols = shape[-1] if len(shape) else 1
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    return rows, cols


def piece_blocks(piece: AffinePiece, local: str, shape: Sequence[int]) -> Optional[List[Block]]:
    """Rectangles of the 2-D view of a weight of ``shape`` that ``piece`` covers, with their local addresses.

    Supports flat ranges (1-D pieces with unit source stride: ZeRO-3 windows, 1-D weights) and 2-D blocks whose source
    strides are those of the row-major full tensor (tensor-parallel row or column slices). Returns None otherwise.
    """
    rows, cols = view_2d(shape)
    s_off, d_off = piece.source_offset, piece.dest_offset
    if len(piece.shape) == 1:
        if piece.source_strides[0] != 1:
            return None
        n, d_step = piece.shape[0], piece.dest_strides[0]
        lo, hi = s_off, s_off + n

        def local_at(flat):
            return d_off + (flat - lo) * d_step

        blocks = []
        r_first, r_last = lo // cols, (hi - 1) // cols
        if r_first == r_last:
            return [((r_first, r_first + 1, lo % cols, (hi - 1) % cols + 1), local, local_at(lo), (cols * d_step,
                                                                                                   d_step))]
        if lo % cols:
            blocks.append(((r_first, r_first + 1, lo % cols, cols), local, local_at(lo), (cols * d_step, d_step)))
            r_first += 1
        tail = hi % cols
        full_end = r_last if tail else r_last + 1
        if full_end > r_first:
            blocks.append(((r_first, full_end, 0, cols), local, local_at(r_first * cols), (cols * d_step, d_step)))
        if tail:
            blocks.append(((r_last, r_last + 1, 0, tail), local, local_at(r_last * cols), (cols * d_step, d_step)))
        return blocks
    if len(piece.shape) != 2 or tuple(piece.source_strides) != (cols, 1) or rows * cols == 0:
        return None
    h, w = piece.shape
    r0, c0 = divmod(s_off, cols)
    if c0 + w > cols or r0 + h > rows:
        return None
    return [((r0, r0 + h, c0, c0 + w), local, d_off, tuple(piece.dest_strides))]


def _intersect(a: Rect, b: Rect) -> Optional[Rect]:
    r0, r1, c0, c1 = max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
    return (r0, r1, c0, c1) if r0 < r1 and c0 < c1 else None


def _subtract(a: Rect, cut: Rect) -> List[Rect]:
    """``a`` minus ``cut`` (which lies inside ``a``) as up to four rectangles."""
    out = []
    if a[0] < cut[0]:
        out.append((a[0], cut[0], a[2], a[3]))
    if cut[1] < a[1]:
        out.append((cut[1], a[1], a[2], a[3]))
    if a[2] < cut[2]:
        out.append((cut[0], cut[1], a[2], cut[2]))
    if cut[3] < a[3]:
        out.append((cut[0], cut[1], cut[3], a[3]))
    return out


def _area(r: Rect) -> int:
    return (r[1] - r[0]) * (r[3] - r[2])


def _shift(block: Block, sub: Rect) -> Tuple[int, Tuple[int, int]]:
    """Local offset and strides of ``sub`` inside ``block``."""
    rect, _, off, strides = block
    return off + (sub[0] - rect[0]) * strides[0] + (sub[2] - rect[2]) * strides[1], strides


Pieces = Dict[str, List[Tuple[str, AffinePiece]]]


def describe(holdings: Pieces, needs: Pieces, shapes: Dict[str, Sequence[int]],
             weights: Sequence[str]) -> Dict[str, Tuple[List[Block], List[Block]]]:
    """This rank's ``{weight: (held blocks, needed blocks)}``, the part of the plan's input each rank contributes."""
    local = {}
    for w in weights:
        have, want = [], []
        for side, out in ((holdings, have), (needs, want)):
            for name, piece in side.get(w, []):
                blocks = piece_blocks(piece, name, shapes[w])
                if blocks is None:
                    raise NotImplementedError(f"a piece of {w!r} is neither a row-major block nor a flat range")
                out += blocks
        local[w] = (have, want)
    return local


RELAYED = "@"  # prefix of a source tensor name that is a destination tensor filled in an earlier stage


def _dst_order(nodes: Sequence[int], turn: int) -> List[int]:
    """Destination ranks node by node, each node's ranks rotated by ``turn``."""
    order = []
    for node in sorted(set(nodes)):
        members = [r for r, n in enumerate(nodes) if n == node]
        k = turn % len(members)
        order += members[k:] + members[:k]
    return order


def _split_rows(sub: Rect, k: int) -> List[Rect]:
    """``sub`` cut into ``k`` (at most its row count) bands of rows of nearly equal height."""
    rows = sub[1] - sub[0]
    k = max(1, min(k, rows))
    cuts = [sub[0] + rows * i // k for i in range(k + 1)]
    return [(cuts[i], cuts[i + 1], sub[2], sub[3]) for i in range(k)]


def _replica_groups(everyone, w: str, nodes: Sequence[int]) -> List[List[int]]:
    """Ranks of one node that need exactly the same rectangles of ``w`` (replicas), groups of two or more."""
    groups = {}
    for r, desc in enumerate(everyone):
        rects = tuple(b[0] for b in desc[w][1])
        if rects:
            groups.setdefault((nodes[r], rects), []).append(r)
    return [g for g in groups.values() if len(g) > 1]


def plan_transfers(everyone: List[Dict[str, Tuple[List[Block], List[Block]]]],
                   weights: Sequence[str],
                   preference: Optional[Dict[str, int]] = None,
                   nodes: Optional[Sequence[int]] = None,
                   relay: bool = False,
                   spread: bool = False,
                   min_split: int = 1 << 16,
                   min_relay: int = 1 << 16) -> List[tuple]:
    """Transfers covering every needed block, from every rank's :func:`describe` (index = rank).

    ``nodes`` gives each rank's node (default: one node); ``relay`` lets blocks that crossed nodes serve the other
    ranks of their node in a later stage, if it has at least ``min_relay`` elements (a small block, such as a norm,
    is fetched by each rank that needs it: relaying it saves no measurable time, but its stage costs an all-to-all
    when nothing else needs relaying). ``spread`` takes inter-node blocks from the least loaded remote holders
    (their smaller blocks first, so a whole shard on one rank does not crowd out the others) and, with ``relay``,
    has the replicas on a node (ranks needing the same blocks) fetch a block of at least ``min_split`` elements in
    bands of rows, one band each, before relaying the bands to each other: every rank's network link carries a share
    of each large weight instead of one rank fetching it whole. Each transfer is ``(weight, src rank, dst rank, shape,
    src tensor, src offset, src strides, dst tensor, dst offset, dst strides, stage)``; a source tensor named
    ``RELAYED + name`` is the source rank's destination tensor ``name``, filled in an earlier stage. Deterministic,
    so every rank computes the same plan.
    """
    preference = preference or {}
    world_size = len(everyone)
    nodes = list(nodes) if nodes is not None else [0] * world_size
    transfers = []
    load = [0] * world_size  # elements each rank already sends to others
    for turn, w in enumerate(weights):
        holders = [(r, b, 0) for r in range(world_size) for b in everyone[r][w][0]]

        def cover(dst, dblock, uncovered):
            """Cover the rectangles ``uncovered`` of ``dst``'s needed block ``dblock`` from the holders."""

            def order(item):
                r, b, stage = item
                remote = nodes[r] != nodes[dst]
                if spread and remote:  # least loaded rank first, and its smaller blocks before a whole shard
                    return (r != dst, remote, stage, 0, load[r], _area(b[0]), r)
                return (r != dst, remote, stage, preference.get(b[1].lstrip(RELAYED), 0), load[r], 0, r)

            for src, hblock, stage in sorted(holders, key=order):
                if not uncovered:
                    break
                rest = []
                for u in uncovered:
                    sub = _intersect(u, hblock[0])
                    if sub is None:
                        rest.append(u)
                        continue
                    s_off, s_str = _shift(hblock, sub)
                    d_off, d_str = _shift(dblock, sub)
                    shape = (sub[1] - sub[0], sub[3] - sub[2])
                    transfers.append((w, src, dst, shape, hblock[1], s_off, s_str, dblock[1], d_off, d_str, stage))
                    if src != dst:
                        load[src] += _area(sub)
                    if relay and nodes[src] != nodes[dst] and _area(sub) >= min_relay:
                        holders.append((dst, (sub, RELAYED + dblock[1], d_off, d_str), stage + 1))
                    rest += _subtract(u, sub)
                uncovered = rest
            if uncovered:
                missing = sum(_area(u) for u in uncovered)
                raise ValueError(f"{missing} elements of {w!r} needed on rank {dst} have no holder")

        covered = {}  # (dst, needed block index) -> bands its replica group already fetched for it
        if relay and spread:
            for group in _replica_groups(everyone, w, nodes):
                k = turn % len(group)
                group = group[k:] + group[:k]
                for i, dblock in enumerate(everyone[group[0]][w][1]):
                    if _area(dblock[0]) < min_split:
                        continue
                    for dst, band in zip(group, _split_rows(dblock[0], len(group))):
                        cover(dst, everyone[dst][w][1][i], [band])
                        covered[(dst, i)] = band
        for dst in (_dst_order(nodes, turn) if relay else range(world_size)):
            for i, dblock in enumerate(everyone[dst][w][1]):
                band = covered.get((dst, i))
                cover(dst, dblock, _subtract(dblock[0], band) if band else [dblock[0]])
    return transfers


class RelayoutPlan:
    """Plan moving every destination piece's elements from the source pieces that hold them.

    Args:
        holdings: this rank's sources, ``{weight: [(local tensor, piece)]}``.
        needs: this rank's destinations, ``{weight: [(local tensor, piece)]}``.
        shapes: ``{weight: full shape}`` for every weight in ``weights``.
        weights: weights to plan, the same list on every rank.
        group: a group containing every rank with holdings or needs (default: the world); ranks in plans and
            transfers are ranks in this group.
        preference: ``{local tensor name: rank}``; holders whose tensor has a lower value are preferred after
            local copies (e.g. ``{"arena": 0, "zero3": 1}`` to move rollout shards before trainer partitions).
        topology: ``"flat"`` (every rank alike), ``"node"`` (holders on the destination's node first; nodes told
            apart by host name) or ``"relay"`` (also relay blocks that crossed nodes inside their node).
        spread: spread inter-node fetches over remote holders (see :func:`plan_transfers`).
        planner: ``"rect"`` (:func:`plan_transfers`) or ``"affine"`` (through the affine IR's ``plan_transfer``, see
            :mod:`~.affine_plan`; ``self.planner_stats`` then has its plans, segments and per-weight fallbacks).

    ``profile``, when set to a list, collects each all-to-all's host-side times (pack, exchange, unpack; with device
    synchronization, so only for diagnosis) and bytes during :meth:`execute`.
    """

    def __init__(self,
                 holdings: Pieces,
                 needs: Pieces,
                 shapes: Dict[str, Sequence[int]],
                 weights: Sequence[str],
                 group=None,
                 preference: Optional[Dict[str, int]] = None,
                 topology: str = "flat",
                 spread: bool = False,
                 planner: str = "rect"):
        if topology not in ("flat", "node", "relay"):
            raise ValueError(f"unknown topology {topology!r}; choose from 'flat', 'node', 'relay'")
        if planner not in ("rect", "affine"):
            raise ValueError(f"unknown planner {planner!r}; choose from 'rect', 'affine'")
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if planner == "affine":
            mine = (_pack(holdings, weights), _pack(needs, weights))
        else:
            mine = describe(holdings, needs, shapes, weights)
        everyone = [None] * self.world_size
        dist.all_gather_object(everyone, (socket.gethostname(), mine), group=group)
        hosts = [host for host, _ in everyone]
        self.nodes = [sorted(set(hosts), key=hosts.index).index(h) for h in hosts]
        options = dict(nodes=self.nodes if topology != "flat" else None, relay=topology == "relay", spread=spread)
        self.planner_stats = None
        if planner == "affine":
            from deepspeed.runtime.rollout.affine_plan import plan_transfers_affine
            self.planner_stats = {}
            self.transfers = plan_transfers_affine([(_unpack(h), _unpack(n)) for _, (h, n) in everyone],
                                                   weights,
                                                   shapes,
                                                   preference,
                                                   stats=self.planner_stats,
                                                   **options)
        else:
            self.transfers = plan_transfers([d for _, d in everyone], weights, preference, **options)
        self.stages = 1 + max((t[-1] for t in self.transfers), default=0)
        self._steps = {}
        self.profile = None

    def bytes_per_rank(self, element_size: int) -> Tuple[int, int]:
        """(sent, received) bytes of this rank for one execution, excluding local copies."""
        sent = received = 0
        for _, src, dst, shape, *_ in self.transfers:
            n = shape[0] * shape[1] * element_size
            if src == self.rank and dst != self.rank:
                sent += n
            if dst == self.rank and src != self.rank:
                received += n
        return sent, received

    def traffic(self, element_size: int) -> Dict[str, int]:
        """Bytes of one execution over the whole group: between ranks of one node (``intra``), across nodes
        (``inter``), and the most any one node receives across nodes (``inter_max_node``)."""
        out = {"intra": 0, "inter": 0}
        per_node = [0] * (max(self.nodes) + 1)
        for _, src, dst, shape, *_ in self.transfers:
            n = shape[0] * shape[1] * element_size
            if src == dst:
                continue
            if self.nodes[src] == self.nodes[dst]:
                out["intra"] += n
            else:
                out["inter"] += n
                per_node[self.nodes[dst]] += n
        out["inter_max_node"] = max(per_node)
        return out

    @torch.no_grad()
    def execute(self,
                sources: Dict[str, torch.Tensor],
                dests: Dict[str, torch.Tensor],
                weights: Optional[Sequence[str]] = None,
                dtype: Optional[torch.dtype] = None,
                device=None,
                buckets: Optional[Sequence[Sequence[str]]] = None) -> None:
        """Move the planned blocks (of ``weights``, default all) from ``sources`` into ``dests``.

        ``sources`` / ``dests`` map this rank's local tensor names to tensors; a name may stand for one tensor per
        weight as ``"<name>:<weight>"`` (e.g. a ZeRO-3 partition), looked up before ``"<name>"``. ``dtype`` and
        ``device`` give the wire buffers' type when this rank has neither sources nor destinations. ``buckets``
        (lists of weights, instead of ``weights``) runs the stages as a pipeline over the buckets: all-to-all ``k``
        moves stage ``s`` of bucket ``k - s``.
        """
        any_tensor = next(iter(list(dests.values()) + list(sources.values())), None)
        dtype = dtype or any_tensor.dtype
        device = device or any_tensor.device
        if buckets is None:
            buckets = [list(weights) if weights is not None else None]
        for todo in self._pipeline(buckets):
            self._exchange(todo, sources, dests, dtype, device)

    def _pipeline(self, buckets) -> List[tuple]:
        """This rank's work per all-to-all for ``buckets`` (cached): ``(local copies, sends per peer, receives per
        peer)``, stage ``s`` of bucket ``b`` in all-to-all ``b + s``."""
        key = tuple(tuple(b) if b is not None else None for b in buckets)
        if key not in self._steps:
            index = {}
            for b, ws in enumerate(buckets):
                for w in (ws if ws is not None else {t[0] for t in self.transfers}):
                    index[w] = b
            steps = [([], [[] for _ in range(self.world_size)], [[] for _ in range(self.world_size)])
                     for _ in range(len(buckets) + self.stages - 1)]
            for t in self.transfers:
                b = index.get(t[0])
                if b is None or self.rank not in (t[1], t[2]):
                    continue
                local, sends, recvs = steps[b + t[-1]]
                if t[1] == t[2]:
                    local.append(t)
                elif t[1] == self.rank:
                    sends[t[2]].append(t)
                else:
                    recvs[t[1]].append(t)
            # every rank takes part in every all-to-all, so a step with nothing for this rank is kept
            self._steps[key] = steps
        return self._steps[key]

    def _exchange(self, step, sources, dests, dtype, device) -> None:
        """One all-to-all: this rank's part of one pipeline step."""
        local, sends, recvs = step
        profile = self.profile
        if profile is not None:
            get_accelerator().synchronize()
            t0 = time.perf_counter()

        def lookup(table, name, w):
            return table[f"{name}:{w}"] if f"{name}:{w}" in table else table[name]

        def source(name, w):
            if name.startswith(RELAYED):
                return lookup(dests, name[len(RELAYED):], w)
            return lookup(sources, name, w)

        for w, _, _, shape, s_name, s_off, s_str, d_name, d_off, d_str, _ in local:
            local_block(lookup(dests, d_name, w), shape, d_str,
                        d_off).copy_(local_block(source(s_name, w), shape, s_str, s_off))
        send_parts = [[local_block(source(t[4], t[0]), t[3], t[6], t[5]).reshape(-1) for t in peer] for peer in sends]
        send_sizes = [sum(p.numel() for p in parts) for parts in send_parts]
        recv_sizes = [sum(t[3][0] * t[3][1] for t in peer) for peer in recvs]
        if sum(send_sizes):
            send = torch.cat([p.to(dtype) for parts in send_parts for p in parts])
        else:
            send = torch.empty(0, dtype=dtype, device=device)
        recv = torch.empty(sum(recv_sizes), dtype=dtype, device=device)
        if profile is not None:
            get_accelerator().synchronize()
            t1 = time.perf_counter()
        dist.all_to_all_single(recv,
                               send,
                               output_split_sizes=recv_sizes,
                               input_split_sizes=send_sizes,
                               group=self.group)
        if profile is not None:
            get_accelerator().synchronize()
            t2 = time.perf_counter()
        offset = 0
        for peer in recvs:
            for w, _, _, shape, _, _, _, d_name, d_off, d_str, _ in peer:
                n = shape[0] * shape[1]
                local_block(lookup(dests, d_name, w), shape, d_str, d_off).copy_(recv[offset:offset + n].view(shape))
                offset += n
        if profile is not None:
            get_accelerator().synchronize()
            t3 = time.perf_counter()
            profile.append({
                "pack_s": t1 - t0,
                "exchange_s": t2 - t1,
                "unpack_s": t3 - t2,
                "sent": send.numel() * send.element_size(),
                "received": recv.numel() * recv.element_size(),
                "transfers": len(local) + sum(map(len, sends)) + sum(map(len, recvs)),
            })


def _pack(pieces: Pieces, weights: Sequence[str]) -> Dict[str, list]:
    """Pieces as plain tuples for the all-gather (holders are whoever declares a block, so locations are dropped)."""
    return {
        w: [(name, (p.shape, p.source_offset, p.source_strides, p.dest_offset, p.dest_strides))
            for name, p in pieces.get(w, [])]
        for w in weights if pieces.get(w)
    }


def _unpack(packed: Dict[str, list]) -> Pieces:
    return {
        w: [(name, AffinePiece(*fields, locations=[0])) for name, fields in pieces]
        for w, pieces in packed.items()
    }


def local_block(tensor: torch.Tensor, shape, strides, offset: int) -> torch.Tensor:
    """The block of a local tensor at ``offset`` (from its first element) with ``strides``."""
    # as_strided counts the offset from the start of the storage, not from the view's first element.
    return tensor.as_strided(shape, strides, tensor.storage_offset() + offset)


def flat_window_pieces(numel: int, lo: int, hi: int, local: str) -> List[Tuple[str, AffinePiece]]:
    """A ZeRO-3-style holding: elements ``[lo, hi)`` of the flattened weight, contiguous at offset 0 of ``local``."""
    if hi <= lo:
        return []
    return [(local, AffinePiece((hi - lo, ), lo, (1, ), 0, (1, ), locations=[0]))]
