# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.launcher import runner as dsrun


def test_parser_mutual_exclusive():
    '''Ensure dsrun.parse_resource_filter() raises a ValueError when include_str and
    exclude_str are both provided.
    '''
    with pytest.raises(ValueError):
        dsrun.parse_resource_filter({}, include_str='A', exclude_str='B')


def test_parser_local():
    ''' Test cases with only one node. '''
    # First try no include/exclude
    hosts = {'worker-0': [0, 1, 2, 3]}
    ret = dsrun.parse_resource_filter(hosts)
    assert (ret == hosts)

    # exclude slots
    ret = dsrun.parse_resource_filter(hosts, exclude_str='worker-0:1')
    assert (ret == {'worker-0': [0, 2, 3]})

    ret = dsrun.parse_resource_filter(hosts, exclude_str='worker-0:1,2')
    assert (ret == {'worker-0': [0, 3]})

    # only use one slot
    ret = dsrun.parse_resource_filter(hosts, include_str='worker-0:1')
    assert (ret == {'worker-0': [1]})

    # including slots multiple times shouldn't break things
    ret = dsrun.parse_resource_filter(hosts, include_str='worker-0:1,1')
    assert (ret == {'worker-0': [1]})
    ret = dsrun.parse_resource_filter(hosts, include_str='worker-0:1@worker-0:0,1')
    assert (ret == {'worker-0': [0, 1]})

    # including just 'worker-0' without : should still use all GPUs
    ret = dsrun.parse_resource_filter(hosts, include_str='worker-0')
    assert (ret == hosts)

    # excluding just 'worker-0' without : should eliminate everything
    ret = dsrun.parse_resource_filter(hosts, exclude_str='worker-0')
    assert (ret == {})

    # exclude all slots manually
    ret = dsrun.parse_resource_filter(hosts, exclude_str='worker-0:0,1,2,3')
    assert (ret == {})


def test_parser_multinode():
    # First try no include/exclude
    hosts = {'worker-0': [0, 1, 2, 3], 'worker-1': [0, 1, 2, 3]}
    ret = dsrun.parse_resource_filter(hosts)
    assert (ret == hosts)

    # include a node
    ret = dsrun.parse_resource_filter(hosts, include_str='worker-1:0,3')
    assert (ret == {'worker-1': [0, 3]})

    # exclude a node
    ret = dsrun.parse_resource_filter(hosts, exclude_str='worker-1')
    assert (ret == {'worker-0': [0, 1, 2, 3]})

    # exclude part of each node
    ret = dsrun.parse_resource_filter(hosts, exclude_str='worker-0:0,1@worker-1:3')
    assert (ret == {'worker-0': [2, 3], 'worker-1': [0, 1, 2]})


def test_parse_inclusion_exclusion():
    '''parse_inclusion_exclusion() maps a slot-count pool onto parse_resource_filter().

    The pool holds slot counts rather than slot lists, so this is the only place the
    real slots are known; the tests above all hand parse_resource_filter() a correct
    dict directly and so never cover this.
    '''
    pool = {'worker-0': 4, 'worker-1': 4}

    # no filtering
    ret = dsrun.parse_inclusion_exclusion(pool, '', '')
    assert (ret == {'worker-0': [0, 1, 2, 3], 'worker-1': [0, 1, 2, 3]})

    # a bare hostname means every slot on that node, as parse_resource_filter documents
    ret = dsrun.parse_inclusion_exclusion(pool, 'worker-0', '')
    assert (ret == {'worker-0': [0, 1, 2, 3]})

    # the example from parse_resource_filter's own docstring
    ret = dsrun.parse_inclusion_exclusion(pool, 'worker-0@worker-1:0,2', '')
    assert (ret == {'worker-0': [0, 1, 2, 3], 'worker-1': [0, 2]})

    ret = dsrun.parse_inclusion_exclusion(pool, 'worker-1:0,2', '')
    assert (ret == {'worker-1': [0, 2]})

    ret = dsrun.parse_inclusion_exclusion(pool, '', 'worker-0:1')
    assert (ret == {'worker-0': [0, 2, 3], 'worker-1': [0, 1, 2, 3]})


def test_parse_inclusion_exclusion_errors():
    '''An out-of-range slot has to be rejected for --include, not just --exclude.'''
    pool = {'worker-0': 4}

    for spec in ('worker-0:4', 'worker-0:99', 'worker-0:0,4'):
        with pytest.raises(ValueError):
            dsrun.parse_inclusion_exclusion(pool, spec, '')
        with pytest.raises(ValueError):
            dsrun.parse_inclusion_exclusion(pool, '', spec)

    with pytest.raises(ValueError):
        dsrun.parse_inclusion_exclusion(pool, 'jeff', '')


class _FakeAccelerator:
    '''Enough of an accelerator for main()'s VISIBLE_DEVICES branch.'''

    def __init__(self, device_count):
        self._device_count = device_count

    def visible_devices_envs(self):
        return ['CUDA_VISIBLE_DEVICES']

    def device_count(self):
        return self._device_count


def _resolve_visible_devices(monkeypatch, visible_devices, device_count):
    '''Run main() as far as parse_inclusion_exclusion() and return what it was handed.

    Stopping there rather than launching keeps this CPU-only; the resource
    resolution is the part under test.
    '''
    captured = {}
    real_parse = dsrun.parse_inclusion_exclusion

    class _Stop(Exception):
        pass

    def _spy(resource_pool, inclusion, exclusion):
        captured['resource_pool'] = dict(resource_pool)
        captured['include'] = inclusion
        captured['exclude'] = exclusion
        raise _Stop

    monkeypatch.setattr(dsrun, 'get_accelerator', lambda: _FakeAccelerator(device_count))
    monkeypatch.setattr(dsrun, 'fetch_hostfile', lambda hostfile: {})
    monkeypatch.setattr(dsrun, 'parse_inclusion_exclusion', _spy)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible_devices)

    with pytest.raises(_Stop):
        dsrun.main(['dummy.py'])

    # put the real one back so the caller can resolve what main() handed over
    monkeypatch.setattr(dsrun, 'parse_inclusion_exclusion', real_parse)
    return captured


def test_visible_devices_resolve_to_the_requested_slots(monkeypatch):
    '''CUDA_VISIBLE_DEVICES=0,2 has to launch on slots 0 and 2.

    main() turns the mask into --include and then unsets it, so the device
    count it asks for afterwards is the physical one. Guards against the
    #5818 shape, where the count is still the masked one and slot 2 does not
    exist as far as the resource pool is concerned.
    '''
    captured = _resolve_visible_devices(monkeypatch, '0,2', device_count=4)

    assert captured['include'] == 'localhost:0,2'
    assert captured['resource_pool'] == {'localhost': 4}

    ret = dsrun.parse_inclusion_exclusion(captured['resource_pool'], captured['include'], captured['exclude'])
    assert (ret == {'localhost': [0, 2]})


def test_visible_devices_are_rejected_when_the_device_count_is_masked(monkeypatch):
    '''A masked count now fails loudly instead of launching on the wrong slots.

    If an accelerator ever caches the count while the mask is still set, the
    pool holds 2 slots while --include names slot 2. That request is now
    rejected by name rather than silently passed through, which is the
    trade this filter fix makes.
    '''
    captured = _resolve_visible_devices(monkeypatch, '0,2', device_count=2)

    assert captured['include'] == 'localhost:0,2'
    assert captured['resource_pool'] == {'localhost': 2}

    with pytest.raises(ValueError, match="No slot '2' specified on host 'localhost'"):
        dsrun.parse_inclusion_exclusion(captured['resource_pool'], captured['include'], captured['exclude'])


def test_parser_errors():
    '''Ensure we catch errors. '''
    hosts = {'worker-0': [0, 1, 2, 3], 'worker-1': [0, 1, 2, 3]}

    # host does not exist
    with pytest.raises(ValueError):
        dsrun.parse_resource_filter(hosts, include_str='jeff')
    with pytest.raises(ValueError):
        dsrun.parse_resource_filter(hosts, exclude_str='jeff')

    # slot does not exist
    with pytest.raises(ValueError):
        dsrun.parse_resource_filter(hosts, include_str='worker-1:4')
    with pytest.raises(ValueError):
        dsrun.parse_resource_filter(hosts, exclude_str='worker-1:4')

    # formatting
    with pytest.raises(ValueError):
        dsrun.parse_resource_filter(hosts, exclude_str='worker-1@worker-0:1@5')


def test_num_plus_parser():
    ''' Ensure we catch errors relating to num_nodes/num_gpus + -i/-e being mutually exclusive'''

    # inclusion
    with pytest.raises(ValueError):
        dsrun.main(args="--num_nodes 1 -i localhost foo.py".split())
    with pytest.raises(ValueError):
        dsrun.main(args="--num_nodes 1 --num_gpus 1 -i localhost foo.py".split())
    with pytest.raises(ValueError):
        dsrun.main(args="--num_gpus 1 -i localhost foo.py".split())

    # exclusion
    with pytest.raises(ValueError):
        dsrun.main(args="--num_nodes 1 -e localhost foo.py".split())
    with pytest.raises(ValueError):
        dsrun.main(args="--num_nodes 1 --num_gpus 1 -e localhost foo.py".split())
    with pytest.raises(ValueError):
        dsrun.main(args="--num_gpus 1 -e localhost foo.py".split())


def test_hostfile_good():
    # good hostfile w. empty lines and comment
    hostfile = """
    worker-1 slots=2
    worker-2 slots=2

    localhost slots=1
    123.23.12.10 slots=2

    #worker-1 slots=3
    # this is a comment

    """
    r = dsrun._parse_hostfile(hostfile.splitlines())
    assert "worker-1" in r
    assert "worker-2" in r
    assert "localhost" in r
    assert "123.23.12.10" in r
    assert r["worker-1"] == 2
    assert r["worker-2"] == 2
    assert r["localhost"] == 1
    assert r["123.23.12.10"] == 2
    assert len(r) == 4


def test_hostfiles_bad():
    # duplicate host
    hostfile = """
    worker-1 slots=2
    worker-2 slots=1
    worker-1 slots=1
    """
    with pytest.raises(ValueError):
        dsrun._parse_hostfile(hostfile.splitlines())

    # incorrect whitespace
    hostfile = """
    this is bad slots=1
    """
    with pytest.raises(ValueError):
        dsrun._parse_hostfile(hostfile.splitlines())

    # no whitespace
    hostfile = """
    missingslots
    """
    with pytest.raises(ValueError):
        dsrun._parse_hostfile(hostfile.splitlines())

    # empty
    hostfile = """
    """
    with pytest.raises(ValueError):
        dsrun._parse_hostfile(hostfile.splitlines())

    # mix of good/bad
    hostfile = """
    worker-1 slots=2
    this is bad slots=1
    worker-2 slots=4
    missingslots

    """
    with pytest.raises(ValueError):
        dsrun._parse_hostfile(hostfile.splitlines())
