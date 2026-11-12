# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import json
import os
from unittest.mock import patch

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
            'both_ways': both_ways,
            'path': path_of(s, t),
            'seconds': seconds(path_of(s, t), size)
        } for size in sizes for s in range(world) for t in range(world) if s != t for both_ways in (False, True)],
        'relay': [{
            'bytes': size,
            'process': p,
            'from_process': (p - shift) % world,
            'to_process': (p + shift) % world,
            'receive_path': path_of((p - shift) % world, p),
            'send_path': path_of(p, (p + shift) % world),
            'seconds': seconds(path_of(p, (p + shift) % world), size)
        } for size in sizes for shift in range(1, world) for p in range(world)],
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

    fan = (0, (1, ))  # the synthetic fans have one peer, so a root moves one message per fan
    checks = [(fits['local'][LOCAL], LOCAL), (fits['peer'][INTER_NODE], INTER_NODE),
              (fits['peer_both_ways'][INTER_NODE], INTER_NODE), (fits['relay'][(INTER_NODE, INTER_NODE)], INTER_NODE)]
    checks += [(fits[kind][fan][INTER_NODE], INTER_NODE) for kind in ('fan_out', 'fan_in')]
    checks += [(fits[direction][fan], INTER_NODE) for direction in ('send', 'receive')]
    for fit, path in checks:
        alpha, bandwidth = costs[path]
        assert fit['alpha_seconds'] == pytest.approx(alpha, rel=1e-6)
        assert fit['bytes_per_second'] == pytest.approx(bandwidth, rel=1e-6)
        assert fit['max_relative_error'] == pytest.approx(0, abs=1e-6)
        assert fit['sizes'] == sizes
    assert set(fits['peer']) == {INTER_NODE}


def test_root_fits_give_each_sender_set_its_rate():
    """A root's time covers every message of its fan, so its rate is fitted against all of them.

    Fitting the root's time against one message's bytes would understate the rate by the width.
    Pooling fans of one width would average sender sets that a PCIe tree tells apart, such as
    root 0 receiving from (1, 2) or from (2, 3) below.
    """
    sizes = [4096, 1 << 20, 1 << 24]
    alpha = 40e-6
    receive_rate = {(0, (1, )): 30e9, (1, (2, )): 12e9, (0, (1, 2)): 12e9, (0, (2, 3)): 54e9, (1, (0, 2, 3)): 9e9}
    calibration = _synthetic_calibration({LOCAL: (1e-6, 100e9), INTER_NODE: (1e-5, 10e9)}, sizes, world=4)
    # Peers are listed in reverse, as the key must not depend on the order they are listed in.
    calibration['fan_in'] = [{
        'bytes': size,
        'width': len(peers),
        'root': root,
        'peers': list(reversed(peers)),
        'paths': [INTER_NODE] * len(peers),
        'root_seconds': [alpha + len(peers) * size / rate] * 3,
        'seconds_by_peer': [[alpha + size / 10e9] * 3] * len(peers),
    } for size in sizes for (root, peers), rate in receive_rate.items()]

    fits = fit_links(calibration)
    assert set(fits['receive']) == set(fits['fan_in']) == set(receive_rate)
    for (root, peers), rate in receive_rate.items():
        fit = fits['receive'][(root, peers)]
        assert fit['bytes_per_second'] == pytest.approx(rate, rel=1e-6)
        assert fit['alpha_seconds'] == pytest.approx(alpha, rel=1e-6)
        assert fit['sizes'] == [len(peers) * size for size in sizes]


def test_fit_reports_error_where_the_line_does_not_hold():
    calibration = _synthetic_calibration({LOCAL: (1e-6, 100e9), INTER_NODE: (1e-5, 10e9)}, [4096, 1 << 20, 1 << 24])
    # A protocol switch at the largest size halves bandwidth there.
    for sample in calibration['peer']:
        if sample['bytes'] == 1 << 24:
            sample['seconds'] = [value * 2 for value in sample['seconds']]
    assert fit_links(calibration)['peer'][INTER_NODE]['max_relative_error'] > 0.1


def test_peer_pair_fits_keep_each_direction():
    """Pooling by path would average the fast and the slow direction of one link."""
    sizes = [4096, 1 << 20, 1 << 24]
    alpha = 1e-5
    bandwidth = {(0, 1): 20e9, (1, 0): 5e9}
    calibration = _synthetic_calibration({LOCAL: (1e-6, 100e9), INTER_NODE: (alpha, 10e9)}, sizes)
    for sample in calibration['peer']:
        pair = (sample['from_process'], sample['to_process'])
        sample['seconds'] = [alpha + sample['bytes'] / bandwidth[pair]] * 3

    fits = fit_links(calibration)
    for pair, expected in bandwidth.items():
        assert fits['peer_pairs'][pair]['bytes_per_second'] == pytest.approx(expected, rel=1e-6)


def test_two_way_fits_are_kept_apart_from_one_way_fits():
    """Pooling two-way samples into ``peer`` would hide what a link loses per direction when both
    directions are busy, which a policy pricing a colocated resharding needs.
    """
    sizes = [4096, 1 << 20, 1 << 24]
    alpha, one_way, both_ways = 1e-5, 20e9, 12e9
    calibration = _synthetic_calibration({LOCAL: (1e-6, 100e9), INTER_NODE: (alpha, one_way)}, sizes)
    for sample in calibration['peer']:
        if sample['both_ways']:
            sample['seconds'] = [alpha + sample['bytes'] / both_ways] * 3

    fits = fit_links(calibration)
    assert fits['peer'][INTER_NODE]['bytes_per_second'] == pytest.approx(one_way, rel=1e-6)
    assert fits['peer_both_ways'][INTER_NODE]['bytes_per_second'] == pytest.approx(both_ways, rel=1e-6)
    for pair in ((0, 1), (1, 0)):
        assert fits['peer_pairs'][pair]['bytes_per_second'] == pytest.approx(one_way, rel=1e-6)
        assert fits['peer_both_ways_pairs'][pair]['bytes_per_second'] == pytest.approx(both_ways, rel=1e-6)


def test_relay_fits_are_keyed_by_receive_and_send_path():
    """A relay's time covers a receive and a send, so fitting it under either path alone would
    mix processes that received over a slow link with ones that only sent over it.
    """
    sizes = [4096, 1 << 20, 1 << 24]
    alpha = 1e-5
    rate = {(INTRA_NODE, INTRA_NODE): 40e9, (INTRA_NODE, INTER_NODE): 15e9, (INTER_NODE, INTRA_NODE): 8e9}
    calibration = _synthetic_calibration({LOCAL: (1e-6, 100e9), INTER_NODE: (alpha, 10e9)}, sizes)
    calibration['relay'] = [{
        'bytes': size,
        'process': 0,
        'from_process': 1,
        'to_process': 2,
        'receive_path': receive_path,
        'send_path': send_path,
        'seconds': [alpha + size / bandwidth] * 3,
    } for size in sizes for (receive_path, send_path), bandwidth in rate.items()]

    fits = fit_links(calibration)
    assert set(fits['relay']) == set(rate)
    for paths, bandwidth in rate.items():
        assert fits['relay'][paths]['bytes_per_second'] == pytest.approx(bandwidth, rel=1e-6)


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
        every_ordered_pair = sorted((size, source, target) for size in sizes for source in range(world)
                                    for target in range(world) if source != target)
        for both_ways in (False, True):
            assert sorted((s['bytes'], s['from_process'], s['to_process']) for s in result['peer']
                          if s['both_ways'] == both_ways) == every_ordered_pair
        assert sorted(
            (s['bytes'], s['process'], s['from_process'], s['to_process']) for s in result['relay']) == sorted(
                (size, process, (process - shift) % world, (process + shift) % world) for size in sizes
                for shift in range(1, world) for process in range(world))

        def path_between(source, target):
            return INTRA_NODE if source // 2 == target // 2 else INTER_NODE

        for sample in result['local']:
            assert sample['path'] == LOCAL
        for sample in result['peer']:
            assert sample['path'] == path_between(sample['from_process'], sample['to_process'])
        for sample in result['relay']:
            assert sample['receive_path'] == path_between(sample['from_process'], sample['process'])
            assert sample['send_path'] == path_between(sample['process'], sample['to_process'])

        samples = [s['seconds'] for s in result['local'] + result['peer'] + result['relay']]
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
        assert set(fits['peer_both_ways']) == {INTRA_NODE, INTER_NODE}
        assert set(fits['peer_both_ways_pairs']) == set(fits['peer_pairs'])
        assert set(fits['relay']) == {(receive, send)
                                      for receive in (INTRA_NODE, INTER_NODE)
                                      for send in (INTRA_NODE, INTER_NODE)} - {(INTRA_NODE, INTRA_NODE)}
        every_fan = {(root, tuple(sorted(set(range(world)) - {root}))) for root in range(world)}
        assert set(fits['fan_out']) == set(fits['fan_in']) == set(fits['send']) == set(fits['receive']) == every_fan
        for kind in ('fan_out', 'fan_in'):
            assert all(set(by_path) == {INTRA_NODE, INTER_NODE} for by_path in fits[kind].values())

    def test_records_the_backend_it_measured(self):
        get_accelerator().set_device(int(os.environ.get('LOCAL_RANK', dist.get_rank())))
        measured = get_accelerator().communication_backend_name()
        # The accelerator's preferred backend is not necessarily the one the group runs on.
        with patch.object(get_accelerator(), 'communication_backend_name', return_value='not-the-group-backend'):
            result = calibrate_links(sizes=[4096], repeats=1, warmup=0, fan_width=0)
        assert result['communication_backend'] == measured

    @pytest.mark.parametrize('processes_per_node', [1, 2])
    def test_default_fans_stay_within_a_node(self, processes_per_node):
        get_accelerator().set_device(int(os.environ.get('LOCAL_RANK', dist.get_rank())))
        node = dist.get_rank() // processes_per_node
        with patch('socket.gethostname', return_value=f'node-{node}'):
            result = calibrate_links(sizes=[4096], repeats=1, warmup=0)

        # With one process per node, fans still run, across nodes.
        assert result['fan_widths'] == [1]
        for sample in result['fan_out'] + result['fan_in']:
            same_node = sample['peers'][0] // processes_per_node == sample['root'] // processes_per_node
            assert same_node or processes_per_node == 1

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
        for sample in result['peer']:
            assert sample['path'] == f'{sample["from_process"]}->{sample["to_process"]}'
        for sample in result['relay']:
            assert sample['receive_path'] == f'{sample["from_process"]}->{sample["process"]}'
            assert sample['send_path'] == f'{sample["process"]}->{sample["to_process"]}'
        # A path_of that tells devices apart must not have their local copies pooled into one fit.
        for sample in result['local']:
            assert sample['path'] == f'{sample["process"]}->{sample["process"]}'

        widths = calibrate_links(sizes=[4096], repeats=1, warmup=0, fan_width=[2, 1, 2])
        assert widths['fan_widths'] == [1, 2]
        for kind in ('fan_out', 'fan_in'):
            assert sorted((s['width'], s['root']) for s in widths[kind]) == sorted(
                (width, root) for width in (1, 2) for root in range(world))
            for sample in widths[kind]:
                assert sample['peers'] == [(sample['root'] + step) % world for step in range(1, sample['width'] + 1)]

        chosen = calibrate_links(sizes=[4096],
                                 repeats=1,
                                 warmup=0,
                                 fan_width=1,
                                 fan_peers=lambda root, width: [(root + 2) % world])
        for sample in chosen['fan_out'] + chosen['fan_in']:
            assert sample['peers'] == [(sample['root'] + 2) % world]
        with pytest.raises(ValueError):
            calibrate_links(sizes=[4096], repeats=1, warmup=0, fan_width=1, fan_peers=lambda root, width: [root])

        for skip in (0, []):
            skipped = calibrate_links(sizes=[4096], repeats=1, warmup=0, fan_width=skip)
            assert skipped['fan_out'] == skipped['fan_in'] == []
