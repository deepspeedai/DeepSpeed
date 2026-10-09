# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""CPU-only tests for the ZenFlow affinity split.

Fakes stand in for the allowed-CPU list (psutil), sysfs CPU topology, and rank / local-size
information. They do not replace helpers inside zenflow_utils, so another implementation that
reads those same sources still passes.
"""

import builtins
import io
import os
import re

import pytest
import torch
from types import SimpleNamespace

import deepspeed.runtime.zenflow.zenflow_utils as zu

_TOPOLOGY_PATH = re.compile(r"/cpu(\d+)/topology/([^/]+)$")
_SIBLING_LISTS = {"thread_siblings_list", "core_cpus_list"}
_SIBLING_MASKS = {"thread_siblings", "core_cpus"}


def _range_list(cpus):
    ordered = sorted(set(cpus))
    ranges = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def _cpu_mask(cpus):
    """Sysfs bitmap: 32-bit hex groups, most significant group first."""
    width = ((max(cpus) // 32) + 1) * 32
    words = []
    for word in range(width // 32):
        base = word * 32
        value = 0
        for cpu in cpus:
            if base <= cpu < base + 32:
                value |= 1 << (cpu - base)
        words.append(f"{value:08x}")
    return ",".join(reversed(words))


def _slurm_allocation():
    """Mask ``64,97-127`` paired into 16 physical cores on two sockets."""
    allowed = [64] + list(range(97, 128))
    topology = {}
    for index, cpu in enumerate(allowed):
        group = index % 16
        package = 0 if group < 8 else 1
        topology[cpu] = (package, group % 8)
    return allowed, topology


class _AffinityWorld:
    """One rank's view of psutil, sysfs, and the distributed launch."""

    def __init__(self, monkeypatch, masks, topology, local_size):
        self.monkeypatch = monkeypatch
        self.masks = [list(mask) for mask in masks]
        self.topology = topology
        self.local_size = local_size
        self.world_size = len(self.masks)
        self.rank = 0
        self.real_open = builtins.open
        hosts = [f"node{index // local_size}" for index in range(self.world_size)]
        self.hosts = hosts
        monkeypatch.setattr(zu.psutil, "Process", self._process)
        monkeypatch.setattr(zu.dist, "get_rank", lambda group=None: self.rank)
        monkeypatch.setattr(zu.dist, "get_world_size", lambda group=None: self.world_size)
        monkeypatch.setattr(zu.dist, "all_gather", self._all_gather)
        monkeypatch.setattr(zu.dist, "all_gather_object", self._all_gather_object)
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(self.masks[self.rank]), raising=False)
        monkeypatch.setattr(builtins, "open", self._open)
        monkeypatch.setattr(io, "open", self._open)

    def _process(self, pid=None):
        world = self

        class _Process:

            def cpu_affinity(self, cpus=None):
                if cpus is not None:
                    return None
                return list(world.masks[world.rank])

        return _Process()

    def _all_gather(self, tensor_list, tensor, group=None, async_op=False, **kwargs):
        rank = self.rank
        current = list(self.masks[rank])
        values = [int(value) for value in tensor.detach().cpu().reshape(-1).tolist()]
        if values == sorted(current):
            payloads = [sorted(mask) for mask in self.masks]
        elif values == current:
            payloads = [list(mask) for mask in self.masks]
        elif len(values) == 1 and values[0] == len(current):
            payloads = [[len(mask)] for mask in self.masks]
        else:
            # Opaque per-rank blob (for example a hostname token). Ranks on one node share it.
            for index, out in enumerate(tensor_list):
                out.copy_(tensor if self.hosts[index] == self.hosts[rank] else tensor + 1)
            return None
        for index, out in enumerate(tensor_list):
            source = torch.tensor(payloads[index], dtype=out.dtype, device=out.device).reshape(out.shape)
            out.copy_(source)
        return None

    def _all_gather_object(self, object_list, obj, group=None, **kwargs):
        if isinstance(obj, (bytes, str)):
            values = list(self.hosts)
        elif isinstance(obj, (list, tuple)):
            values = [list(mask) for mask in self.masks]
        else:
            values = [obj for _ in object_list]
        for index in range(len(object_list)):
            object_list[index] = values[index]
        return None

    def _open(self, file, mode="r", *args, **kwargs):
        name = os.fspath(file) if isinstance(file, (str, bytes, os.PathLike)) else None
        if isinstance(name, bytes):
            name = name.decode()
        match = _TOPOLOGY_PATH.search(name) if isinstance(name, str) else None
        if match is None:
            return self.real_open(file, mode, *args, **kwargs)
        cpu = int(match.group(1))
        leaf = match.group(2)
        if cpu not in self.topology:
            raise FileNotFoundError(name)
        package, core = self.topology[cpu]
        if leaf == "physical_package_id":
            text = f"{package}\n"
        elif leaf == "core_id":
            text = f"{core}\n"
        elif leaf in _SIBLING_LISTS:
            text = _range_list(self._siblings(cpu)) + "\n"
        elif leaf in _SIBLING_MASKS:
            text = _cpu_mask(self._siblings(cpu)) + "\n"
        else:
            text = "0\n"
        return io.StringIO(text)

    def _siblings(self, cpu):
        key = self.topology[cpu]
        return [other for other, other_key in self.topology.items() if other_key == key]

    def _export(self, local_rank, rank):
        values = {
            "LOCAL_RANK": local_rank,
            "LOCAL_WORLD_SIZE": self.local_size,
            "LOCAL_SIZE": self.local_size,
            "OMPI_COMM_WORLD_LOCAL_RANK": local_rank,
            "OMPI_COMM_WORLD_LOCAL_SIZE": self.local_size,
            "OMPI_COMM_WORLD_RANK": rank,
            "OMPI_COMM_WORLD_SIZE": self.world_size,
            "MPI_LOCALRANKID": local_rank,
            "MPI_LOCALNRANKS": self.local_size,
            "SLURM_LOCALID": local_rank,
            "SLURM_NTASKS_PER_NODE": self.local_size,
            "SLURM_PROCID": rank,
            "RANK": rank,
            "WORLD_SIZE": self.world_size,
        }
        for key, value in values.items():
            self.monkeypatch.setenv(key, str(value))
        self.rank = rank

    def compute(self, local_rank, rank, reserved=0.5):
        self._export(local_rank, rank)
        optimizer = SimpleNamespace(pt_reserved_cores_perc=reserved)
        return zu._compute_zf_pt_affinity(optimizer)


def _split_local_ranks(monkeypatch, masks, topology, local_size, global_ranks, reserved=0.5):
    world = _AffinityWorld(monkeypatch, masks, topology, local_size)
    return [world.compute(local_rank, rank, reserved) for local_rank, rank in enumerate(global_ranks)]


def _assert_even_physical_split(results, allowed, topology, local_size):
    all_keys = {topology[cpu] for cpu in allowed}
    per_rank = len(all_keys) // local_size
    assert per_rank >= 1
    covered = set()
    allowed_set = set(allowed)
    for zf_affinity, pt_affinity in results:
        assert zf_affinity, "optimizer affinity is empty, so OMP_NUM_THREADS would be 0"
        assert pt_affinity, "training affinity is empty"
        chosen = set(zf_affinity) | set(pt_affinity)
        assert chosen <= allowed_set
        keys = {topology[cpu] for cpu in chosen}
        # One logical CPU per physical core, and an even share of this node's cores.
        assert len(keys) == len(chosen) == per_rank
        if per_rank > 1:
            assert set(zf_affinity).isdisjoint(pt_affinity)
        else:
            assert set(zf_affinity) == set(pt_affinity) == chosen
        assert keys.isdisjoint(covered)
        covered |= keys
    assert covered == all_keys
    return covered


def test_low_id_single_node_keeps_one_thread_per_core(monkeypatch):
    """Physical cores 0-3 with hyperthreads 4-7, two ranks on one node."""
    allowed = list(range(8))
    topology = {cpu: (0, cpu % 4) for cpu in allowed}
    results = _split_local_ranks(monkeypatch, [allowed, allowed], topology, local_size=2, global_ranks=[0, 1])
    covered = _assert_even_physical_split(results, allowed, topology, local_size=2)
    assert covered == {(0, core) for core in range(4)}


def test_slurm_high_id_mask_splits_across_local_ranks(monkeypatch):
    """A cpuset of 64,97-127 has no low CPU ids. Each local rank still gets a share."""
    allowed, topology = _slurm_allocation()
    assert 0 not in allowed
    assert len({topology[cpu] for cpu in allowed}) == 16
    local_size = 4
    results = _split_local_ranks(monkeypatch, [allowed] * local_size,
                                 topology,
                                 local_size=local_size,
                                 global_ranks=list(range(local_size)))
    covered = _assert_even_physical_split(results, allowed, topology, local_size)
    assert {package for package, _core in covered} == {0, 1}
    # The same core id on two sockets is two cores. Collapsing on core id alone drops a socket.
    assert len({core for _package, core in covered}) < len(covered)


def test_repeated_core_ids_stay_distinct_across_sockets(monkeypatch):
    topology = {}
    for core in range(2):
        topology[core] = (0, core)
        topology[core + 4] = (0, core)
        topology[core + 2] = (1, core)
        topology[core + 6] = (1, core)
    allowed = list(range(8))
    results = _split_local_ranks(monkeypatch, [allowed], topology, local_size=1, global_ranks=[0])
    covered = _assert_even_physical_split(results, allowed, topology, local_size=1)
    assert covered == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_multinode_identical_masks_use_local_rank(monkeypatch):
    """Eight ranks, four per node, four physical cores. Global rank 4 is local rank 0."""
    allowed = [0, 1, 2, 3]
    topology = {cpu: (0, cpu) for cpu in allowed}
    local_size = 4
    world_size = 8
    results = _split_local_ranks(monkeypatch, [allowed] * world_size,
                                 topology,
                                 local_size=local_size,
                                 global_ranks=[4, 5, 6, 7])
    covered = _assert_even_physical_split(results, allowed, topology, local_size)
    assert covered == {(0, cpu) for cpu in allowed}
    for zf_affinity, pt_affinity in results:
        assert len(set(zf_affinity) | set(pt_affinity)) == 1


def test_too_few_physical_cores_raises(monkeypatch):
    allowed = [97, 113]
    topology = {97: (0, 0), 113: (0, 1)}
    with pytest.raises(RuntimeError) as caught:
        _split_local_ranks(monkeypatch, [allowed] * 4, topology, local_size=4, global_ranks=[0, 1, 2, 3])
    message = str(caught.value)
    assert "97" in message
    assert "113" in message
    assert "4" in message


def test_distinct_rank_masks_keep_their_own_cpus(monkeypatch):
    """``--bind_cores_to_rank`` already gave each rank a different mask. Do not shard it again."""
    masks = [[20, 21, 120, 121], [10, 11, 110, 111]]
    topology = {
        20: (0, 0),
        120: (0, 0),
        21: (0, 1),
        121: (0, 1),
        10: (0, 2),
        110: (0, 2),
        11: (0, 3),
        111: (0, 3),
    }
    results = _split_local_ranks(monkeypatch, masks, topology, local_size=2, global_ranks=[0, 1])
    for (zf_affinity, pt_affinity), mask in zip(results, masks):
        assert zf_affinity
        assert pt_affinity
        assert set(zf_affinity).isdisjoint(pt_affinity)
        assert set(zf_affinity) | set(pt_affinity) == set(mask)
