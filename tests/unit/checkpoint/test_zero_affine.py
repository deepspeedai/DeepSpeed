# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine import (contiguous_split_map, nest_affine_maps, replicated_map, segmented_map,
                                         sub_param_map)
from deepspeed.checkpoint.affine_transfer import PlanBudget, plan_transfer
from deepspeed.checkpoint.zero_affine import (autotp_storage_affine_maps, zero_optimizer_storage_affine_maps,
                                              zero_optimizer_storage_shard, zero_param_storage_affine_map,
                                              zero_param_storage_shard, zero_storage_affine_map)
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_optimizer_state
from unit.common import DistributedTest
from unit.simple_model import random_dataloader


def test_replicas_list_every_holder():
    full = torch.arange(6.)
    affine_map = zero_storage_affine_map(6, [4, 1, 6])

    assert set(affine_map.shard_shapes) == {1, 4, 6}
    for rank in (1, 4, 6):
        assert torch.equal(affine_map.extract(full, rank), full)

    asked = []

    def selector(request):
        asked.append(sorted(candidate.source_rank for candidate in request.candidates))
        return request.candidates[-1].source_rank

    plan_transfer(contiguous_split_map((6, ), [3, 3], 0), affine_map, PlanBudget(), replica_selector=selector)
    # Each target block may be read from any of the three replicas.
    assert asked == [[1, 4, 6], [1, 4, 6]]


def test_zero3_partitions_drop_padding():
    # 10 elements in partitions of 4: holders keep 4, 4 and 2 elements; the fourth is all padding.
    full = torch.arange(10.)
    affine_map = zero_storage_affine_map(10, [2, 0, 3, 1], partition_numel=4)

    assert affine_map.shard_shapes == {2: (4, ), 0: (4, ), 3: (2, ), 1: (0, )}
    assert torch.equal(affine_map.extract(full, 2), full[0:4])
    assert torch.equal(affine_map.extract(full, 0), full[4:8])
    assert torch.equal(affine_map.extract(full, 3), full[8:10])
    assert affine_map.extract(full, 1).numel() == 0


@pytest.mark.parametrize('numel, holders, partition', [(9, [0, 1], 4), (4, [0, 0], None), (4, [], None)])
def test_rejects_storage_that_cannot_hold_the_parameter(numel, holders, partition):
    with pytest.raises(ValueError):
        zero_storage_affine_map(numel, holders, partition)


@pytest.mark.parametrize('accessor', [zero_param_storage_affine_map, zero_param_storage_shard])
def test_rejects_quantized_zero3_parameter_storage(accessor):
    param = torch.nn.Parameter(torch.zeros(8), requires_grad=False)
    param.ds_id = 0
    param.ds_numel = 8
    param.ds_tensor = torch.zeros(4, dtype=torch.int8)
    param.ds_tensor.ds_numel = 4
    param.ds_tensor.ds_quant_scale = torch.ones(1)

    with pytest.raises(NotImplementedError, match='Quantized ZeRO-3 parameter storage'):
        accessor(param)


@pytest.mark.parametrize('accessor', [zero_param_storage_affine_map, zero_param_storage_shard])
def test_rejects_nvme_offloaded_zero3_parameter_storage(accessor):
    # A partition swapped out to NVMe leaves a one-element placeholder in memory.
    param = torch.nn.Parameter(torch.zeros(8), requires_grad=False)
    param.ds_id = 0
    param.ds_numel = 8
    param.ds_tensor = torch.zeros(1)
    param.ds_tensor.ds_numel = 4
    param.ds_tensor.final_location = OffloadDeviceEnum.nvme

    with pytest.raises(NotImplementedError, match='NVMe-offloaded ZeRO-3 parameter storage'):
        accessor(param)


def _stored_by_rank(layout_shards, holders_by_layout_rank, partition_by_layout_rank=None):
    """What each process stores, by slicing each flattened layout shard with ZeRO's rule."""
    stored = {}
    for layout_rank, shard in layout_shards.items():
        flat = shard.reshape(-1)
        for index, rank in enumerate(holders_by_layout_rank[layout_rank]):
            if partition_by_layout_rank is None:
                stored[rank] = flat.clone()
                continue
            partition = partition_by_layout_rank[layout_rank]
            stored[rank] = flat[index * partition:(index + 1) * partition].clone()
    return stored


def _nested(layout, holders_by_layout_rank, partition_by_layout_rank=None):
    inner = {}
    for layout_rank, holders in holders_by_layout_rank.items():
        numel = torch.Size(layout.shard_shapes[layout_rank]).numel()
        partition = None if partition_by_layout_rank is None else partition_by_layout_rank[layout_rank]
        inner[layout_rank] = zero_storage_affine_map(numel, holders, partition)
    return nest_affine_maps(layout, inner)


def _check_storage(affine_map, full, stored):
    # validate_coverage also proves every piece names exactly the processes that store its elements.
    affine_map.validate_coverage()
    assert affine_map.logical_shape == tuple(full.shape)
    assert affine_map.shard_shapes == {rank: tuple(shard.shape) for rank, shard in stored.items()}
    for rank, shard in stored.items():
        torch.testing.assert_close(affine_map.extract(full, rank), shard, rtol=0, atol=0)
    torch.testing.assert_close(affine_map.rebuild(stored), full, rtol=0, atol=0)


# Uneven column split of a 4 x 8 weight over two tensor-parallel ranks, the layout of a row-parallel
# linear layer. Its pieces are strided in the flattened weight.
_FULL = torch.arange(32, dtype=torch.float32).reshape(4, 8)
_COLUMNS = {0: _FULL[:, :3], 1: _FULL[:, 3:]}
_HOLDERS = {0: [0, 2], 1: [1, 3]}


@pytest.mark.parametrize('partition', [None, {0: 6, 1: 10}, {0: 7, 1: 11}], ids=['replicas', 'even', 'mid_row'])
def test_nest_describes_zero_storage_of_a_column_split(partition):
    layout = contiguous_split_map(_FULL.shape, [3, 5], partition_dim=1)

    affine_map = _nested(layout, _HOLDERS, partition)

    _check_storage(affine_map, _FULL, _stored_by_rank(_COLUMNS, _HOLDERS, partition))


def test_nest_describes_a_contiguous_storage_run_as_a_few_boxes():
    # A process stores one contiguous run of its layout shard: whole rows between two partial rows.
    # One piece per row would make planning a large row-parallel weight slow, so the run must stay
    # at most three boxes however many rows it spans.
    full = torch.arange(64 * 48, dtype=torch.float32).reshape(64, 48)
    layout = contiguous_split_map(full.shape, [20, 28], partition_dim=1)
    holders = {0: [0, 1, 2], 1: [3, 4]}
    partition = {0: 450, 1: 900}

    affine_map = _nested(layout, holders, partition)

    _check_storage(affine_map, full, _stored_by_rank({0: full[:, :20], 1: full[:, 20:]}, holders, partition))
    for rank in range(5):
        assert len(affine_map.pieces_by_rank[rank]) <= 3


def test_nest_merges_holders_of_replicated_layout_shards_split_at_different_places():
    # Two layout ranks hold the whole tensor but partition it over groups of different sizes, so an
    # element's holders are the union of one process from each group, cut at both groups' boundaries.
    layout = replicated_map(_FULL.shape, 2)
    holders = {0: [0, 1], 1: [2, 3, 4]}
    partition = {0: 16, 1: 11}

    affine_map = _nested(layout, holders, partition)

    _check_storage(affine_map, _FULL, _stored_by_rank({0: _FULL, 1: _FULL}, holders, partition))
    for offset, ranks in affine_map.holders().items():
        assert ranks == {offset // 16, 2 + offset // 11}


def test_nest_keeps_fused_sub_parameters_apart():
    full = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    # Fused QKV: rows [0, 2), [2, 4) and [4, 6) are each split between two ranks.
    layout = sub_param_map(full.shape, [2, 2, 2], [[1, 1], [1, 1], [1, 1]], partition_dim=0)
    shards = {rank: full[[rank, 2 + rank, 4 + rank]] for rank in range(2)}
    holders = {0: [0, 1], 1: [2, 3]}
    partition = {0: 7, 1: 6}

    _check_storage(_nested(layout, holders, partition), full, _stored_by_rank(shards, holders, partition))


def test_nest_keeps_boxes_dense_in_every_replica():
    # The last column is a strided column of rank 0's whole-weight shard but all of rank 1's shard,
    # so a box spanning both rows would be one run of addresses on rank 1 only.
    full = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    layout = segmented_map(full.shape, [(2, False), (1, True)], 1, 2, [[2, 0]])
    holders = {0: [0], 1: [1]}

    _check_storage(_nested(layout, holders), full, _stored_by_rank({0: full, 1: full[:, 2:]}, holders))


def test_nest_and_plan_a_scalar_parameter():
    full = torch.tensor(5.0)
    holders = {0: [0, 1], 1: [2]}
    source = _nested(replicated_map((), 2), holders)
    stored = _stored_by_rank({0: full, 1: full}, holders)
    _check_storage(source, full, stored)

    plan = plan_transfer(replicated_map((), 1), source)

    target = torch.full((), float('nan'))
    for segment in plan.segments:
        source_view = torch.as_strided(stored[segment.source_rank], segment.shape, segment.source_strides,
                                       segment.source_offset)
        torch.as_strided(target, segment.shape, segment.target_strides, segment.target_offset).copy_(source_view)
    assert target.item() == 5.0


def test_nested_storage_plans_into_another_layout():
    source = _nested(contiguous_split_map(_FULL.shape, [3, 5], partition_dim=1), _HOLDERS, {0: 7, 1: 11})
    target_holders = {0: [4], 1: [5, 6], 2: [7]}
    target = _nested(contiguous_split_map(_FULL.shape, [1, 2, 1], partition_dim=0), target_holders, {0: 8, 1: 9, 2: 8})
    source_storage = _stored_by_rank(_COLUMNS, _HOLDERS, {0: 7, 1: 11})
    expected = _stored_by_rank({0: _FULL[:1], 1: _FULL[1:3], 2: _FULL[3:]}, target_holders, {0: 8, 1: 9, 2: 8})

    plan = plan_transfer(target, source)

    target_storage = {rank: torch.full(shape, float('nan')) for rank, shape in target.shard_shapes.items()}
    writes = {rank: torch.zeros(shape, dtype=torch.int64) for rank, shape in target.shard_shapes.items()}
    for segment in plan.segments:
        source_view = torch.as_strided(source_storage[segment.source_rank], segment.shape, segment.source_strides,
                                       segment.source_offset)
        torch.as_strided(target_storage[segment.target_rank], segment.shape, segment.target_strides,
                         segment.target_offset).copy_(source_view)
        torch.as_strided(writes[segment.target_rank], segment.shape, segment.target_strides,
                         segment.target_offset).add_(1)
    for rank, shard in expected.items():
        torch.testing.assert_close(target_storage[rank], shard, rtol=0, atol=0)
        assert torch.all(writes[rank] == 1)


def test_nest_rejects_storage_that_does_not_match_the_layout():
    layout = contiguous_split_map(_FULL.shape, [3, 5], partition_dim=1)
    with pytest.raises(ValueError, match='layout ranks'):
        nest_affine_maps(layout, {0: zero_storage_affine_map(12, [0])})
    with pytest.raises(ValueError, match='holds 12 elements'):
        nest_affine_maps(layout, {0: zero_storage_affine_map(32, [0]), 1: zero_storage_affine_map(20, [1])})
    with pytest.raises(ValueError, match='more than one layout shard'):
        nest_affine_maps(layout, {0: zero_storage_affine_map(12, [0]), 1: zero_storage_affine_map(20, [0])})


class StraddlingModel(torch.nn.Module):
    # 351 trainable elements split over two ranks put the boundary inside ``first.bias``, so one
    # parameter is held in part by each rank. ``frozen`` has no optimizer state.
    def __init__(self, hidden_dim):
        super().__init__()
        self.frozen = torch.nn.Linear(hidden_dim, hidden_dim)
        self.frozen.requires_grad_(False)
        self.first = torch.nn.Linear(hidden_dim, hidden_dim)
        self.second = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, y):
        return torch.nn.functional.cross_entropy(self.second(self.first(self.frozen(x))), y)


def _full_param(param):
    if hasattr(param, 'ds_id'):
        with deepspeed.zero.GatheredParameters([param]):
            return param.data.detach().clone().view(-1)
    return param.data.detach().clone().view(-1)


def _full_optimizer_value(param, key):
    if key is None:
        return safe_get_full_fp32_param(param).view(-1)
    return safe_get_full_optimizer_state(param, key).view(-1)


class TestZeroOptimizerStorageMaps(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('zero_stage, bf16_optimizer', [(1, False), (2, False), (3, False), (1, True)])
    def test_maps_describe_live_storage(self, zero_stage, bf16_optimizer):
        if not get_accelerator().is_bf16_supported():
            pytest.skip('requires bf16')
        hidden_dim = 13
        config = {
            'train_micro_batch_size_per_gpu': 1,
            'optimizer': {
                'type': 'Adam',
                'params': {
                    'lr': 1e-2,
                    'torch_adam': True
                }
            },
            'bf16': {
                'enabled': True
            },
            'zero_optimization': {
                'stage': zero_stage
            },
        }
        if bf16_optimizer:
            # fp32 gradient accumulation under ZeRO-1 selects the bf16 optimizer.
            config['data_types'] = {'grad_accum_dtype': 'fp32'}
        if zero_stage == 3:
            with deepspeed.zero.Init(config_dict_or_path=config):
                model = StraddlingModel(hidden_dim)
        else:
            model = StraddlingModel(hidden_dim)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        params = list(engine.module.parameters())
        rank = dist.get_rank()

        if zero_stage != 3:
            # first.bias straddles the partition boundary, so every rank holds part of it.
            with pytest.raises(RuntimeError, match='first optimizer step'):
                zero_optimizer_storage_shard(engine.module.first.bias, 'exp_avg')

        for x, y in random_dataloader(engine, 3, hidden_dim, engine.device, torch.bfloat16):
            engine.backward(engine(x, y))
            engine.step()

        if zero_stage == 3:
            # ZeRO-3 keeps small parameters gathered between steps; a write to the shard alone would be lost.
            with pytest.raises(RuntimeError, match='empty_partition_cache'):
                zero_param_storage_shard(engine.module.second.weight)
            engine.empty_partition_cache()

        maps = zero_optimizer_storage_affine_maps(params)
        assert maps[0] is None and maps[1] is None
        assert zero_optimizer_storage_shard(engine.module.frozen.weight) is None
        for param, affine_map in zip(params[2:], maps[2:]):
            affine_map.validate_coverage()

            # The low-precision parameter is the fp32 master copy rounded, which checks the master
            # copy's map against storage that does not come from the optimizer.
            full_param = _full_param(param)
            assert torch.equal(_full_optimizer_value(param, None).to(full_param.dtype), full_param)
            param_map = zero_param_storage_affine_map(param)
            assert torch.equal(zero_param_storage_shard(param), param_map.extract(full_param, rank))

            for key in (None, 'exp_avg', 'exp_avg_sq'):
                full = _full_optimizer_value(param, key)
                assert torch.equal(zero_optimizer_storage_shard(param, key), affine_map.extract(full, rank))

            # The shard is ZeRO's storage: writing each rank's id into it must show up in the gathered
            # state exactly where the map places that rank's range.
            zero_optimizer_storage_shard(param, 'exp_avg').fill_(rank + 1)
            written = _full_optimizer_value(param, 'exp_avg')
            expected = torch.zeros_like(written)
            for holder, pieces in affine_map.pieces_by_rank.items():
                for piece in pieces:
                    expected[piece.source_offset:piece.source_offset + piece.shape[0]] = holder + 1
            assert torch.equal(written, expected)

        # A write to the parameter's shard must reach the next forward.
        zero_param_storage_shard(engine.module.second.weight).zero_()
        outputs = []
        hook = engine.module.second.register_forward_hook(lambda module, args, output: outputs.append(output))
        for x, y in random_dataloader(engine, 1, hidden_dim, engine.device, torch.bfloat16):
            engine(x, y)
        hook.remove()
        assert torch.all(outputs[0] == 0)

        engine.destroy()


class ThreeLinears(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(3)])

    def forward(self, x, y):
        for linear in self.linears:
            x = linear(x)
        return torch.nn.functional.cross_entropy(x, y)


def _all_shards(shard):
    gathered = [None] * dist.get_world_size()
    # .clone() drops the view's offset into the flat buffer; on CPU .cpu() alone would pickle it.
    dist.all_gather_object(gathered, shard.detach().cpu().clone())
    return dict(enumerate(gathered))


def _one_holder_each(layout):
    # Every layout rank is a process of its own, as a rollout engine without ZeRO would be.
    inner = {
        rank: zero_storage_affine_map(torch.Size(shape).numel(), [rank])
        for rank, shape in layout.shard_shapes.items()
    }
    return nest_affine_maps(layout, inner)


class TestAutoTPStorageMaps(DistributedTest):
    world_size = 4
    reuse_dist_env = False

    @pytest.mark.parametrize('zero_stage, zero_init', [(1, False), (2, False), (3, False), (3, True)])
    def test_maps_describe_live_storage_of_tp2_dp2(self, zero_stage, zero_init):
        if not get_accelerator().is_bf16_supported():
            pytest.skip('requires bf16')
        hidden_dim = 12
        config = {
            'train_micro_batch_size_per_gpu': 1,
            'optimizer': {
                'type': 'Adam',
                'params': {
                    'lr': 1e-2,
                    'torch_adam': True
                }
            },
            'bf16': {
                'enabled': True
            },
            'zero_optimization': {
                'stage': zero_stage
            },
            'tensor_parallel': {
                'autotp_size': 2,
                'partition_config': {
                    'use_default_specs':
                    False,
                    'layer_specs': [
                        {
                            'patterns': [r'.*linears\.0\.weight$'],
                            'partition_type': 'column'
                        },
                        {
                            'patterns': [r'.*linears\.1\.weight$'],
                            'partition_type': 'row'
                        },
                        {
                            'patterns': [r'.*linears\.2\.weight$'],
                            'partition_type': 'skip'
                        },
                    ],
                },
            },
        }
        torch.manual_seed(0)
        if zero_init:
            # zero.Init partitions every parameter over the whole world before AutoTP sees the model.
            with deepspeed.zero.Init(config_dict_or_path=config):
                model = ThreeLinears(hidden_dim)
            # The weights before AutoTP and ZeRO touch them: the oracle for what each process stores.
            reference = {
                name: _full_param(param).cpu().view(param.ds_shape)
                for name, param in model.named_parameters()
            }
        else:
            model = ThreeLinears(hidden_dim)
            reference = {name: param.detach().to(torch.bfloat16) for name, param in model.named_parameters()}
        shapes = {name: tuple(full.shape) for name, full in reference.items()}
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        params = dict(engine.module.named_parameters())
        rank = dist.get_rank()

        maps = autotp_storage_affine_maps(engine.module)

        assert set(maps) == set(reference)
        for name, affine_map in maps.items():
            affine_map.validate_coverage()
            assert torch.equal(zero_param_storage_shard(params[name]).cpu(), affine_map.extract(reference[name], rank))

            # Plan from the live storage into four tensor-parallel ranks split along dim 0.
            target = _one_holder_each(contiguous_split_map(shapes[name], [hidden_dim // 4] * 4, 0))
            plan = plan_transfer(target, affine_map)
            source_storage = _all_shards(zero_param_storage_shard(params[name]))
            written = {r: torch.full(s, float('nan'), dtype=torch.bfloat16) for r, s in target.shard_shapes.items()}
            for segment in plan.segments:
                torch.as_strided(written[segment.target_rank], segment.shape, segment.target_strides,
                                 segment.target_offset).copy_(
                                     torch.as_strided(source_storage[segment.source_rank], segment.shape,
                                                      segment.source_strides, segment.source_offset))
            for target_rank, shard in written.items():
                assert torch.equal(shard, target.extract(reference[name], target_rank))

        for x, y in random_dataloader(engine, 2, hidden_dim, engine.device, torch.bfloat16):
            engine.backward(engine(x, y))
            engine.step()
        engine.empty_partition_cache()

        optimizer_maps = autotp_storage_affine_maps(engine.module, optimizer_states=True)
        for name, affine_map in optimizer_maps.items():
            affine_map.validate_coverage()
            # The trained low-precision parameter, rebuilt through its map, is the fp32 master copy rounded.
            full_param = maps[name].rebuild(_all_shards(zero_param_storage_shard(params[name])))
            master = zero_optimizer_storage_shard(params[name]).to(torch.bfloat16).cpu()
            assert torch.equal(master, affine_map.extract(full_param, rank))

        engine.destroy()
