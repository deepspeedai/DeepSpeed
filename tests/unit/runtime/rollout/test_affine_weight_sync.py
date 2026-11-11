# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Refreshing a tensor-parallel copy of a ZeRO model through affine plans."""

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine import contiguous_split_map, replicated_map
from deepspeed.checkpoint.affine_transfer_executor import AffineTransferExecutionError
from deepspeed.checkpoint.zero_affine import zero_param_storage_shard
from deepspeed.runtime.rollout.affine_weight_sync import (AffineWeightSync, WeightArena, replicated_layout_map,
                                                          zero_param_shaped_storage_map)
from unit.common import DistributedTest
from unit.simple_model import random_dataloader

HIDDEN, INNER = 13, 18


class MLPBlock(torch.nn.Module):
    """A column-parallel and a row-parallel linear and a norm, as a transformer MLP is split."""

    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(HIDDEN, INNER)
        self.down = torch.nn.Linear(INNER, HIDDEN)
        self.norm = torch.nn.LayerNorm(HIDDEN)

    def forward(self, x, y):
        return torch.nn.functional.cross_entropy(self.norm(self.down(torch.relu(self.up(x)))), y)


def _even(size, parts):
    base, extra = divmod(size, parts)
    return [base + (part < extra) for part in range(parts)]


def _layouts(tp_size):
    # INNER does not divide by 4, so TP=4 shards are uneven.
    return {
        'up.weight': contiguous_split_map((INNER, HIDDEN), _even(INNER, tp_size), 0),
        'up.bias': contiguous_split_map((INNER, ), _even(INNER, tp_size), 0),
        'down.weight': contiguous_split_map((HIDDEN, INNER), _even(INNER, tp_size), 1),
        'down.bias': replicated_map((HIDDEN, ), tp_size),
        'norm.weight': replicated_map((HIDDEN, ), tp_size),
        'norm.bias': replicated_map((HIDDEN, ), tp_size),
    }


def _full_param(param):
    if hasattr(param, 'ds_id'):
        with deepspeed.zero.GatheredParameters([param]):
            return param.data.detach().clone()
    return param.data.detach().clone()


def _total(value):
    tensor = torch.tensor([value], dtype=torch.int64, device=get_accelerator().current_device_name())
    dist.all_reduce(tensor)
    return int(tensor.item())


def _all(value):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _tp_rank(replicas, rank):
    for ranks in replicas:
        if rank in ranks:
            return ranks.index(rank)
    return None


class TestAffineWeightSync(DistributedTest):
    world_size = 4

    # Two TP=2 replicas, one TP=4 replica, and two TP=1 replicas on half of the ranks.
    @pytest.mark.parametrize('replicas', [[[0, 1], [2, 3]], [[0, 1, 2, 3]], [[3], [1]]])
    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    def test_sync_writes_every_replica_in_place(self, zero_stage, replicas):
        dtype = torch.bfloat16 if get_accelerator().is_bf16_supported() else torch.float32
        config = {
            'train_micro_batch_size_per_gpu': 1,
            'optimizer': {
                'type': 'Adam',
                'params': {
                    'lr': 1e-2,
                    'torch_adam': True
                }
            },
            'zero_optimization': {
                'stage': zero_stage
            },
        }
        if dtype == torch.bfloat16:
            config['bf16'] = {'enabled': True}
        if zero_stage == 3:
            with deepspeed.zero.Init(config_dict_or_path=config):
                model = MLPBlock()
        else:
            model = MLPBlock()
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        params = dict(engine.module.named_parameters())
        layouts = _layouts(len(replicas[0]))
        rank = dist.get_rank()
        tp_rank = _tp_rank(replicas, rank)

        source_maps = {name: zero_param_shaped_storage_map(param) for name, param in params.items()}
        target_maps = {name: replicated_layout_map(layouts[name], replicas) for name in params}
        sync = AffineWeightSync(source_maps, target_maps, dtype)
        arena = None
        if tp_rank is not None:
            shapes = {name: layouts[name].shard_shapes[tp_rank] for name in params}
            arena = WeightArena(shapes, dtype, engine.device)

        synced = []
        for step in range(2):
            if zero_stage == 3:
                engine.empty_partition_cache()
            sources = {name: zero_param_storage_shard(param) for name, param in params.items()}
            targets = dict(arena.items()) if arena is not None else {}
            stats = sync.sync(sources, targets)
            if zero_stage < 3:
                # Every rank holds a replica of every parameter, so each target is filled from its own process.
                assert _total(stats.sent_bytes) == 0
            for name, param in params.items():
                full = _full_param(param)
                if arena is not None:
                    assert torch.equal(arena[name], layouts[name].extract(full, tp_rank)), name
            if arena is not None:
                synced.append((arena.buffer.data_ptr(), arena.buffer.clone()))
            for x, y in random_dataloader(engine, 1, HIDDEN, engine.device, dtype):
                engine.backward(engine(x, y))
                engine.step()

        if arena is not None:
            (first_address, first_values), (second_address, second_values) = synced
            assert second_address == first_address
            # Otherwise the second sync proves nothing about writing in place.
            assert not torch.equal(second_values, first_values)
        engine.destroy()

    def test_reads_of_replicated_parameters_spread_over_their_holders(self):
        # Ranks 0-2 each hold all six parameters whole, and rank 3 needs them all. Each parameter is a single region,
        # so the even split comes from the load carried across parameters.
        shapes = {f'w{index}': (4, 6) for index in range(6)}
        rank = dist.get_rank()
        device = get_accelerator().current_device_name()
        source_maps, target_maps = {}, {}
        for name, shape in shapes.items():
            source_maps[name] = replicated_layout_map(replicated_map(shape, 1), [[0], [1], [2]])
            target_maps[name] = replicated_layout_map(replicated_map(shape, 1), [[3]])
        sync = AffineWeightSync(source_maps, target_maps, torch.float32)
        full = {
            name: torch.arange(24, dtype=torch.float32, device=device).view(shape) + 100 * index
            for index, (name, shape) in enumerate(shapes.items())
        }
        arena = WeightArena(shapes, torch.float32, device) if rank == 3 else None

        targets = dict(arena.items()) if arena is not None else {}
        # Every rank passes the same sources; rank 3 stores none, so its entries are ignored.
        stats = sync.sync(full, targets)

        sent = _all(stats.sent_bytes)
        assert sent[0] == sent[1] == sent[2] == 2 * 24 * 4 and sent[3] == 0
        if arena is not None:
            for name in shapes:
                assert torch.equal(arena[name], full[name])

    def test_local_copies_are_free_and_single_holders_count_as_load(self):
        # Rank 0 alone holds 'a', which rank 3 needs, so rank 0 must send it. Ranks 0 and 1 hold 'b', which ranks 1
        # and 3 need: rank 1 copies its own, which uses no link, so rank 3 reads 'b' from rank 1, which sends less.
        shape = (4, 6)
        rank = dist.get_rank()
        device = get_accelerator().current_device_name()
        whole = replicated_map(shape, 1)
        source_maps = {'a': replicated_layout_map(whole, [[0]]), 'b': replicated_layout_map(whole, [[0], [1]])}
        target_maps = {'a': replicated_layout_map(whole, [[3]]), 'b': replicated_layout_map(whole, [[1], [3]])}
        sync = AffineWeightSync(source_maps, target_maps, torch.float32)
        full = {name: torch.full(shape, float(index + 1), device=device) for index, name in enumerate(source_maps)}
        needed = [name for name in target_maps if rank in target_maps[name].shard_shapes]
        arena = WeightArena({name: shape for name in needed}, torch.float32, device) if needed else None
        targets = dict(arena.items()) if arena is not None else {}

        stats = sync.sync(full, targets)

        assert _all(stats.sent_bytes) == [24 * 4, 24 * 4, 0, 0]
        for name in needed:
            assert torch.equal(arena[name], full[name]), name

    def test_maps_stay_keyed_by_global_rank_with_a_custom_group(self):
        # The group leaves rank 0 out, so its ranks differ from global ranks: rank 1 (group rank 0) holds 'w' and
        # rank 3 (group rank 2) needs it.
        shape = (4, 6)
        rank = dist.get_rank()
        device = get_accelerator().current_device_name()
        group = dist.new_group([1, 2, 3])
        if rank == 0:
            return
        whole = replicated_map(shape, 1)
        sync = AffineWeightSync({'w': replicated_layout_map(whole, [[1]])}, {'w': replicated_layout_map(whole, [[3]])},
                                torch.float32,
                                group=group)
        full = torch.arange(24, dtype=torch.float32, device=device).view(shape)
        arena = WeightArena({'w': shape}, torch.float32, device) if rank == 3 else None

        stats = sync.sync({'w': full} if rank == 1 else {}, dict(arena.items()) if arena is not None else {})

        assert stats.sent_bytes == (24 * 4 if rank == 1 else 0)
        if arena is not None:
            assert torch.equal(arena['w'], full)

    def test_a_rank_missing_a_shard_makes_every_rank_refuse_before_anything_moves(self):
        shape = (4, 6)
        rank = dist.get_rank()
        device = get_accelerator().current_device_name()
        sync = AffineWeightSync({'w': replicated_layout_map(replicated_map(shape, 1), [[0]])},
                                {'w': replicated_layout_map(contiguous_split_map(shape, [2, 2], 0), [[2, 3]])},
                                torch.float32)
        arena = WeightArena({'w': (2, 6)}, torch.float32, device) if rank in (2, 3) else None
        # Rank 3 forgets its target shard. Had it raised alone, the other ranks would wait for it in the preflight.
        targets = dict(arena.items()) if arena is not None and rank != 3 else {}
        with pytest.raises(AffineTransferExecutionError):
            sync.sync({'w': torch.ones(shape, device=device)}, targets)
        if rank == 2:
            assert torch.all(arena['w'] == 0)

    def test_a_graph_captured_before_a_sync_reads_the_synced_weights(self):
        if get_accelerator().device_name() != 'cuda':
            pytest.skip('requires CUDA graphs')
        shape = (6, 8)
        layout = contiguous_split_map(shape, [3, 3], 0)
        rank = dist.get_rank()
        device = get_accelerator().current_device_name()
        sync = AffineWeightSync({'w': replicated_layout_map(replicated_map(shape, 1), [[0]])},
                                {'w': replicated_layout_map(layout, [[2, 3]])}, torch.float32)
        tp_rank = _tp_rank([[2, 3]], rank)
        arena = None
        if tp_rank is not None:
            arena = WeightArena({'w': layout.shard_shapes[tp_rank]}, torch.float32, device)
        x = torch.randn(4, shape[1], device=device)
        if arena is not None:
            # Warm up outside the graph, as capture requires.
            y = x @ arena['w'].t()
            graph = get_accelerator().create_graph()
            with get_accelerator().capture_to_graph(graph):
                y = x @ arena['w'].t()

        for version in range(2):
            full = torch.randn(shape, device=device) + version
            targets = dict(arena.items()) if arena is not None else {}
            sync.sync({'w': full}, targets)
            # Every rank needs the source values to check its shard.
            dist.broadcast(full, 0)
            if arena is not None:
                get_accelerator().replay_graph(graph)
                assert torch.equal(y, x @ layout.extract(full, tp_rank).t())


def test_arena_views_are_aligned_disjoint_and_zeroed():
    shapes = {'a': (3, 5), 'b': (7, ), 'c': (2, 2, 3)}
    # Several arenas alive at once, at different addresses: an allocation may happen to be aligned.
    arenas = [WeightArena(shapes, torch.float16) for _ in range(8)]
    for arena in arenas:
        assert torch.all(arena.buffer == 0)
        for index, (name, view) in enumerate(arena.items()):
            assert view.shape == shapes[name]
            assert view.data_ptr() % 256 == 0, name
            view.fill_(index + 1)
        # A view that overlapped another would have overwritten part of it.
        for index, (name, view) in enumerate(arena.items()):
            assert torch.all(view == index + 1), name


def test_replicated_layout_map_gives_each_replica_its_layout_shard():
    full = torch.arange(60, dtype=torch.float32).view(6, 10)
    layout = contiguous_split_map((6, 10), [4, 3, 3], 1)
    replicas = [[5, 0, 3], [1, 4, 2]]
    target = replicated_layout_map(layout, replicas)
    for ranks in replicas:
        for tp_rank, rank in enumerate(ranks):
            shard = target.extract(full, rank).view(layout.shard_shapes[tp_rank])
            assert torch.equal(shard, layout.extract(full, tp_rank))


@pytest.mark.parametrize('replicas', [[[0, 1], [2]], [[0, 1], [1, 2]], []])
def test_replicated_layout_map_refuses_replicas_that_do_not_fit_the_layout(replicas):
    with pytest.raises(ValueError):
        replicated_layout_map(contiguous_split_map((4, 4), [2, 2], 0), replicas)
