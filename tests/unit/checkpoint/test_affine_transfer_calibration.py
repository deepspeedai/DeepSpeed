# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import json
import os

import pytest

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine_transfer_calibration import (CALIBRATION_SCHEMA, INTER_NODE, INTRA_NODE, LOCAL,
                                                              calibrate_links, fit_links, node_path_of)
from unit.common import DistributedTest


def test_rejects_empty_samples():
    with pytest.raises(ValueError):
        calibrate_links(repeats=0)
    with pytest.raises(ValueError):
        calibrate_links(sizes=[0])


def _synthetic_calibration(costs, sizes, world=2):
    """A calibration result whose times follow known alpha and bandwidth per path exactly."""

    def seconds(path, size, slowdown=1.0):
        alpha, bandwidth = costs[path]
        exact = alpha + size / bandwidth
        # One outlier per entry; the median-based fit must ignore it.
        return [exact, exact, exact * slowdown * 10]

    path_of = node_path_of(1)
    return {
        'schema':
        CALIBRATION_SCHEMA,
        'local': [{
            'bytes': size,
            'process': p,
            'path': LOCAL,
            'seconds': seconds(LOCAL, size)
        } for size in sizes for p in range(world)],
        'peer': [{
            'bytes': size,
            'from_process': s,
            'to_process': t,
            'path': path_of(s, t),
            'seconds': seconds(path_of(s, t), size)
        } for size in sizes for s in range(world) for t in range(world) if s != t],
        'concurrent': [{
            'bytes': size,
            'shift': 1,
            'paths': [path_of(p, (p + 1) % world) for p in range(world)],
            'seconds_by_process': [seconds(INTER_NODE, size) for _ in range(world)],
        } for size in sizes],
        **{
            kind: [{
                'bytes': size,
                'root': 0,
                'peers': list(range(1, world)),
                'paths': [INTER_NODE] * (world - 1),
                'root_seconds': seconds(INTER_NODE, size),
                'seconds_by_peer': [seconds(INTER_NODE, size) for _ in range(1, world)],
            } for size in sizes]
            for kind in ('fan_out', 'fan_in')
        },
    }


def test_fit_recovers_known_link_costs():
    costs = {LOCAL: (2e-6, 400e9), INTER_NODE: (30e-6, 20e9)}
    sizes = [4096, 1 << 16, 1 << 20, 1 << 24]
    fits = fit_links(_synthetic_calibration(costs, sizes))

    for group, path in (('local', LOCAL), ('peer', INTER_NODE), ('concurrent', INTER_NODE), ('fan_out', INTER_NODE),
                        ('fan_in', INTER_NODE)):
        fit = fits[group][path]
        alpha, bandwidth = costs[path]
        assert fit['alpha_seconds'] == pytest.approx(alpha, rel=1e-6)
        assert fit['bytes_per_second'] == pytest.approx(bandwidth, rel=1e-6)
        assert fit['max_relative_error'] == pytest.approx(0, abs=1e-6)
        assert fit['sizes'] == sizes
    assert set(fits['peer']) == {INTER_NODE}


def test_fit_reports_error_where_the_line_does_not_hold():
    calibration = _synthetic_calibration({LOCAL: (1e-6, 100e9), INTER_NODE: (1e-5, 10e9)}, [4096, 1 << 20, 1 << 24])
    # A protocol switch at the largest size halves bandwidth there.
    for sample in calibration['peer']:
        if sample['bytes'] == 1 << 24:
            sample['seconds'] = [value * 2 for value in sample['seconds']]
    assert fit_links(calibration)['peer'][INTER_NODE]['max_relative_error'] > 0.1


def test_fit_rejects_unknown_schema_and_single_size():
    with pytest.raises(ValueError):
        fit_links({'schema': CALIBRATION_SCHEMA + 1})
    with pytest.raises(ValueError):
        fit_links(_synthetic_calibration({LOCAL: (1e-6, 1e9), INTER_NODE: (1e-6, 1e9)}, [4096]))


@pytest.mark.skipif(get_accelerator().device_count() < 4, reason='requires four visible accelerator devices')
class TestCalibrateLinks(DistributedTest):
    world_size = 4

    def test_every_path_is_sampled_and_labeled(self):
        get_accelerator().set_device(int(os.environ.get('LOCAL_RANK', dist.get_rank())))
        sizes = [4096, 1 << 20]
        # Pretend two processes per node so both peer path classes occur on one machine.
        result = calibrate_links(sizes=sizes, repeats=3, warmup=1, path_of=node_path_of(2))

        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, json.dumps(result, sort_keys=True))
        assert len(set(gathered)) == 1, 'every process must return the same result'

        world = 4
        assert result['schema'] == CALIBRATION_SCHEMA
        assert result['world_size'] == world
        assert sorted((s['bytes'], s['process']) for s in result['local']) == sorted(
            (size, process) for size in sizes for process in range(world))
        assert sorted((s['bytes'], s['from_process'], s['to_process']) for s in result['peer']) == sorted(
            (size, source, target) for size in sizes for source in range(world) for target in range(world)
            if source != target)
        assert sorted((s['bytes'], s['shift']) for s in result['concurrent']) == sorted(
            (size, shift) for size in sizes for shift in range(1, world))

        for sample in result['local']:
            assert sample['path'] == LOCAL
        for sample in result['peer']:
            expected = INTRA_NODE if sample['from_process'] // 2 == sample['to_process'] // 2 else INTER_NODE
            assert sample['path'] == expected
        for sample in result['concurrent']:
            assert sample['paths'] == [node_path_of(2)(p, (p + sample['shift']) % world) for p in range(world)]
            assert len(sample['seconds_by_process']) == world

        samples = [s['seconds'] for s in result['local'] + result['peer']]
        samples += [seconds for s in result['concurrent'] for seconds in s['seconds_by_process']]
        for seconds in samples:
            assert len(seconds) == 3
            assert all(value > 0 for value in seconds)

        assert result['fan_width'] == world - 1
        for kind in ('fan_out', 'fan_in'):
            assert sorted((s['bytes'], s['root']) for s in result[kind]) == sorted(
                (size, root) for size in sizes for root in range(world))
            for sample in result[kind]:
                assert sorted(sample['peers'] + [sample['root']]) == list(range(world))
                for seconds in [sample['root_seconds']] + sample['seconds_by_peer']:
                    assert len(seconds) == 3
                    assert all(value > 0 for value in seconds)

        fits = fit_links(result)
        assert set(fits['local']) == {LOCAL}
        assert set(fits['peer']) == {INTRA_NODE, INTER_NODE}
        assert set(fits['concurrent']) <= {INTRA_NODE, INTER_NODE}
        assert set(fits['fan_out']) == set(fits['fan_in']) == {INTRA_NODE, INTER_NODE}

    def test_fans_label_each_message_by_its_direction(self):
        get_accelerator().set_device(int(os.environ.get('LOCAL_RANK', dist.get_rank())))
        world = 4
        with pytest.raises(ValueError):
            calibrate_links(sizes=[4096], repeats=1, warmup=0, fan_width=world)

        # A direction-sensitive label, so a fan-in message labeled as root -> peer is caught.
        result = calibrate_links(sizes=[4096],
                                 repeats=1,
                                 warmup=0,
                                 path_of=lambda source, target: f'{source}->{target}',
                                 fan_width=2)
        assert result['fan_width'] == 2
        for sample in result['fan_out']:
            root = sample['root']
            assert sample['peers'] == [(root + 1) % world, (root + 2) % world]
            assert sample['paths'] == [f'{root}->{peer}' for peer in sample['peers']]
        for sample in result['fan_in']:
            root = sample['root']
            assert sample['paths'] == [f'{peer}->{root}' for peer in sample['peers']]

        skipped = calibrate_links(sizes=[4096], repeats=1, warmup=0, fan_width=0)
        assert skipped['fan_out'] == skipped['fan_in'] == []
