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
                'width': world - 1,
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

    width = 1  # the synthetic fans have one peer, so a root moves one message per fan
    for fit, path in ((fits['local'][LOCAL], LOCAL), (fits['peer'][INTER_NODE], INTER_NODE),
                      (fits['concurrent'][INTER_NODE], INTER_NODE), (fits['fan_out'][width][INTER_NODE], INTER_NODE),
                      (fits['fan_in'][width][INTER_NODE],
                       INTER_NODE), (fits['send'][width][0], INTER_NODE), (fits['receive'][width][0], INTER_NODE)):
        alpha, bandwidth = costs[path]
        assert fit['alpha_seconds'] == pytest.approx(alpha, rel=1e-6)
        assert fit['bytes_per_second'] == pytest.approx(bandwidth, rel=1e-6)
        assert fit['max_relative_error'] == pytest.approx(0, abs=1e-6)
        assert fit['sizes'] == sizes
    assert set(fits['peer']) == {INTER_NODE}


def test_root_fits_give_each_process_its_capacity_at_each_width():
    """A root's time covers every message of its fan, so its capacity is fitted against all of them.

    Fitting the root's time against one message's bytes would understate the capacity by the
    width, and pooling widths would blur the capacity's dependence on how many peers share it.
    """
    sizes = [4096, 1 << 20, 1 << 24]
    alpha = 40e-6
    receive_capacity = {(1, 0): 30e9, (1, 1): 12e9, (3, 0): 20e9, (3, 1): 9e9}
    calibration = _synthetic_calibration({LOCAL: (1e-6, 100e9), INTER_NODE: (1e-5, 10e9)}, sizes, world=4)
    calibration['fan_in'] = [{
        'bytes': size,
        'width': width,
        'root': root,
        'peers': [(root + step) % 4 for step in range(1, width + 1)],
        'paths': [INTER_NODE] * width,
        'root_seconds': [alpha + width * size / receive_capacity[(width, root)]] * 3,
        'seconds_by_peer': [[alpha + size / 10e9] * 3] * width,
    } for size in sizes for width in (1, 3) for root in (0, 1)]

    fits = fit_links(calibration)
    assert set(fits['receive']) == set(fits['fan_in']) == {1, 3}
    for (width, root), capacity in receive_capacity.items():
        fit = fits['receive'][width][root]
        assert fit['bytes_per_second'] == pytest.approx(capacity, rel=1e-6)
        assert fit['alpha_seconds'] == pytest.approx(alpha, rel=1e-6)
        assert fit['sizes'] == [width * size for size in sizes]


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

        assert result['fan_widths'] == [world - 1]
        for kind in ('fan_out', 'fan_in'):
            assert sorted((s['bytes'], s['root']) for s in result[kind]) == sorted(
                (size, root) for size in sizes for root in range(world))
            for sample in result[kind]:
                assert sample['width'] == world - 1
                assert sorted(sample['peers'] + [sample['root']]) == list(range(world))
                for seconds in [sample['root_seconds']] + sample['seconds_by_peer']:
                    assert len(seconds) == 3
                    assert all(value > 0 for value in seconds)

        fits = fit_links(result)
        assert set(fits['local']) == {LOCAL}
        assert set(fits['peer']) == {INTRA_NODE, INTER_NODE}
        assert set(fits['concurrent']) <= {INTRA_NODE, INTER_NODE}
        assert set(fits['fan_out'][world - 1]) == set(fits['fan_in'][world - 1]) == {INTRA_NODE, INTER_NODE}
        assert set(fits['send'][world - 1]) == set(fits['receive'][world - 1]) == set(range(world))

    def test_fans_label_each_message_by_its_direction(self):
        get_accelerator().set_device(int(os.environ.get('LOCAL_RANK', dist.get_rank())))
        world = 4
        for invalid in (world, [1, world], [-1]):
            with pytest.raises(ValueError):
                calibrate_links(sizes=[4096], repeats=1, warmup=0, fan_width=invalid)

        # A direction-sensitive label, so a fan-in message labeled as root -> peer is caught.
        result = calibrate_links(sizes=[4096],
                                 repeats=1,
                                 warmup=0,
                                 path_of=lambda source, target: f'{source}->{target}',
                                 fan_width=2)
        assert result['fan_widths'] == [2]
        for sample in result['fan_out']:
            root = sample['root']
            assert sample['peers'] == [(root + 1) % world, (root + 2) % world]
            assert sample['paths'] == [f'{root}->{peer}' for peer in sample['peers']]
        for sample in result['fan_in']:
            root = sample['root']
            assert sample['paths'] == [f'{peer}->{root}' for peer in sample['peers']]

        widths = calibrate_links(sizes=[4096], repeats=1, warmup=0, fan_width=[2, 1, 2])
        assert widths['fan_widths'] == [1, 2]
        for kind in ('fan_out', 'fan_in'):
            assert sorted((s['width'], s['root']) for s in widths[kind]) == sorted(
                (width, root) for width in (1, 2) for root in range(world))
            for sample in widths[kind]:
                assert sample['peers'] == [(sample['root'] + step) % world for step in range(1, sample['width'] + 1)]

        for skip in (0, []):
            skipped = calibrate_links(sizes=[4096], repeats=1, warmup=0, fan_width=skip)
            assert skipped['fan_out'] == skipped['fan_in'] == []
