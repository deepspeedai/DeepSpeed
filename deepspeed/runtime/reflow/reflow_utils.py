# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Reflow utilities: NUMA-aware CPU-core/affinity planning for the optimizer worker threads,
plus small contiguous-range helpers for the bucketwise pipeline. These are pure
helpers that hold no optimizer state, so they live in a separate module from the
optimizer itself.
"""

import os

import psutil

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger


def resolve_available_cores():
    """Return (total_cpus, available_cores) honoring an existing process affinity."""
    try:
        current_process = psutil.Process()
        current_affinity = current_process.cpu_affinity()
        if current_affinity:
            # Process is already pinned (taskset/numactl) - only use those cores.
            return len(current_affinity), sorted(current_affinity)
        total_cpus = psutil.cpu_count(logical=False)
        return total_cpus, list(range(total_cpus))
    except (AttributeError, psutil.AccessDenied):
        total_cpus = psutil.cpu_count(logical=False) if hasattr(psutil, 'cpu_count') else 1
        total_cpus = total_cpus if total_cpus and total_cpus > 0 else 1
        return total_cpus, list(range(total_cpus))


def numa_cores_by_id():
    """Return {numa_node_id: [cores]} for non-empty nodes. The key is the real system NUMA node
    id (so a GPU's NUMA node maps directly into this dict); empty/HBM nodes are dropped."""
    try:
        from deepspeed.utils.numa import get_numa_cores
        raw = get_numa_cores() or []
    except Exception:
        return {}
    return {nid: sorted(int(c) for c in cores) for nid, cores in enumerate(raw) if cores}


def gpu_input_count():
    """Number of local GPUs/processes the job was launched with (the 'GPU input'), matching the
    num_local_procs DeepSpeed uses for --bind_cores_to_rank core binding. The DeepSpeed launcher
    exports it as LOCAL_SIZE; torchrun exports LOCAL_WORLD_SIZE; otherwise fall back to the world
    size or the visible device count. This drives the per-rank core split."""
    for var in ("LOCAL_SIZE", "LOCAL_WORLD_SIZE"):
        value = os.environ.get(var)
        if value is not None:
            try:
                count = int(value)
                if count > 0:
                    return count
            except ValueError:
                pass
    try:
        if dist.is_initialized():
            return max(1, dist.get_world_size())
    except Exception:
        pass
    try:
        return max(1, get_accelerator().device_count())
    except Exception:
        return 1


def gpu_numa_node_for_device(device_index):
    """Return the NUMA node id that a CUDA device's PCIe link is physically attached to, or -1
    if unknown. Reads the device's PCI address numa_node from sysfs, with an NVML memory-affinity
    fallback, so each rank binds to its GPU's real node instead of a positional guess."""
    bdf = None
    try:
        import torch
        p = torch.cuda.get_device_properties(device_index)  #ignore-cuda
        bdf = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        with open(f"/sys/bus/pci/devices/{bdf}/numa_node") as fh:
            node = int(fh.read().strip())
        if node >= 0:
            return node
    except Exception:
        pass
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByPciBusId(bdf) if bdf \
            else pynvml.nvmlDeviceGetHandleByIndex(device_index)
        affinity = pynvml.nvmlDeviceGetMemoryAffinity(handle, 8, 0)
        bits = 0
        for word, value in enumerate(affinity):
            bits |= (int(value) << (64 * word))
        for node in range(256):
            if bits & (1 << node):
                return node
    except Exception:
        pass
    return -1


def gpu_pci_id(device_index):
    """Return a unique integer key for a CUDA device's real PCI address (domain/bus/device), used
    to order GPUs deterministically across ranks; -1 if unavailable."""
    try:
        import torch
        p = torch.cuda.get_device_properties(device_index)  #ignore-cuda
        return (int(p.pci_domain_id) << 24) | (int(p.pci_bus_id) << 16) | (int(p.pci_device_id) << 8)
    except Exception:
        return -1


def resolve_gpu_numa_assignment(local_rank, local_world_size):
    """Return (my_numa_node, num_gpus_sharing_node, my_position_on_node) from the real GPU
    PCIe-NUMA topology, considering only the GPUs this job actually uses.

    Preferred path: when distributed is initialized, every participating rank exchanges
    (host, gpu-pci-id, gpu-numa-node), so a node's cores are divided only among the job's GPUs on
    the same host - correct for any number of physical GPUs and any CUDA_VISIBLE_DEVICES layout.
    Fallbacks: locally-visible devices, then the whole node, then (None, ...) when the GPU node
    cannot be read. Returns the device count from the job's ranks, never the physical GPU count.
    """
    # Distributed gate: globally consistent on every rank (same dist state, same world size), so all
    # ranks take the same branch here.
    try:
        distributed = dist.is_initialized() and dist.get_world_size() > 1
    except Exception:
        distributed = False

    if distributed:
        # Every rank must reach the collective, even when its topology is unreadable.
        # Use a sentinel for missing topology and choose the fallback after the exchange.
        import socket
        import zlib
        import torch
        try:
            host_id = zlib.crc32(socket.gethostname().encode()) & 0x7fffffff
        except Exception:
            host_id = -1
        try:
            my_dev = get_accelerator().current_device()
        except Exception:
            my_dev = local_rank if local_rank is not None and local_rank >= 0 else 0
        my_node = gpu_numa_node_for_device(my_dev)
        my_pci = gpu_pci_id(my_dev)
        row = [int(host_id), int(my_pci), int(my_node)]
        local_t = torch.tensor(row, dtype=torch.long, device=get_accelerator().current_device_name())
        gathered = [torch.zeros_like(local_t) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local_t)
        rows = [g.detach().cpu().tolist() for g in gathered]
        if my_node < 0 or my_pci < 0:
            return None, None, None
        # Only this host's GPUs on my NUMA node, ordered by PCI id for a stable split.
        same = sorted(pci for h, pci, node in rows if h == host_id and node == my_node and pci >= 0)
        if my_pci in same:
            return my_node, len(same), same.index(my_pci)
        return my_node, None, None
    # Single-process (non-distributed) fallback: no collective here, so an early return is safe.
    # Derive co-location from the locally-visible job GPUs.
    try:
        if not get_accelerator().is_available():
            return None, None, None
        try:
            my_dev = get_accelerator().current_device()
        except Exception:
            my_dev = local_rank if local_rank is not None and local_rank >= 0 else 0
        my_node = gpu_numa_node_for_device(my_dev)
        if my_node < 0:
            return None, None, None
        visible = get_accelerator().device_count()
        count = local_world_size if (local_world_size and local_world_size > 0) else visible
        count = min(count, visible)
        if count > 0 and local_rank is not None and local_rank >= 0 and local_rank < count:
            rank_to_node = {r: gpu_numa_node_for_device(r) for r in range(count)}
            if all(node_id >= 0 for node_id in rank_to_node.values()):
                co_located = sorted(r for r, node_id in rank_to_node.items() if node_id == my_node)
                if local_rank in co_located:
                    return my_node, len(co_located), co_located.index(local_rank)
        return my_node, None, None
    except Exception:
        return None, None, None


def heuristic_rank_cores(numa_nodes, local_rank, local_world_size):
    """Positional NUMA assignment used only when the real GPU->NUMA topology is unavailable."""
    if not numa_nodes:
        total = psutil.cpu_count(logical=False) or 1
        return list(range(total))
    n_nodes = len(numa_nodes)
    if local_rank < 0:
        return sorted(c for node in numa_nodes for c in node)
    if local_world_size <= n_nodes:
        nodes_per_rank = max(1, n_nodes // max(1, local_world_size))
        start = (local_rank * nodes_per_rank) % n_nodes
        chosen = [numa_nodes[(start + k) % n_nodes] for k in range(nodes_per_rank)]
        return sorted(c for node in chosen for c in node)
    node = numa_nodes[local_rank % n_nodes]
    ranks_per_node = max(1, local_world_size // n_nodes)
    pos = local_rank // n_nodes
    if ranks_per_node <= len(node):
        # Common case: enough cores to give each co-located rank a contiguous chunk.
        chunk = max(1, len(node) // ranks_per_node)
        start = pos * chunk
        return node[start:start + chunk] if start < len(node) else node
    # Extreme oversubscription (more co-located ranks than cores): split the node's cores into
    # disjoint slices via a base size + remainder so co-located ranks never share a core. Ranks
    # past the core count fall back to a degenerate (<1 core/rank) split, so warn once on rank 0.
    if local_rank == 0:
        logger.warning(f"Reflow NUMA affinity hint: {ranks_per_node} co-located ranks but only "
                       f"{len(node)} cores on this NUMA node; cores cannot be split disjointly.")
    base = len(node) // ranks_per_node
    remainder = len(node) % ranks_per_node
    start = pos * base + min(pos, remainder)
    size = base + (1 if pos < remainder else 0)
    return node[start:start + size]


def plan_cpu_core_layout(main_thread_cores):
    """Automatically find NUMA-aware cores for this rank.

    Priority:
      1. If the process is already pinned (numactl/taskset), respect exactly those cores.
      2. Otherwise bind to the NUMA node this rank's GPU is physically attached to (read from the
         PCIe topology via sysfs/NVML), splitting that node's cores among the GPUs that share it,
         so the CPU optimizer runs on the cores local to the GPU's memory controller.
      3. If the GPU->NUMA topology cannot be read, fall back to a positional NUMA guess.
      4. If no NUMA topology is available at all, fall back to all physical cores.

    Returns (total_cpus, rank_cores, node_worker_groups, main_cores, worker_cores). The worker
    cores are grouped per NUMA node so a single worker is never split across two nodes.
    """
    node_cores_by_id = numa_cores_by_id()
    numa_nodes = [node_cores_by_id[nid] for nid in sorted(node_cores_by_id)]
    try:
        affinity = sorted(psutil.Process().cpu_affinity())
    except (AttributeError, psutil.AccessDenied):
        affinity = None
    # psutil returns ALL cores when the process is not actually pinned; only treat affinity as an
    # explicit pin when it is a strict subset of the machine's cores.
    all_cores = set(c for cores in node_cores_by_id.values() for c in cores)
    total_visible = len(all_cores) if all_cores else (psutil.cpu_count() or 0)
    is_pinned = bool(affinity) and total_visible > 0 and len(affinity) < total_visible

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # The GPU input count (DeepSpeed num_local_procs) drives the per-rank core split, matching
    # the DeepSpeed --bind_cores_to_rank convention.
    local_world_size = gpu_input_count()
    # Exchange topology before branching on local affinity so pinned and unpinned ranks
    # participate in the same collective. Pinned ranks can ignore the result.
    gpu_node, gpu_n_on_node, gpu_pos = resolve_gpu_numa_assignment(local_rank, local_world_size)

    rank_cores = None
    if is_pinned:
        # Already pinned (e.g. launcher numactl --cpunodebind) - honor it.
        rank_cores = affinity
    elif node_cores_by_id:
        # Bind to the NUMA node this rank's GPU is physically attached to (read from the PCIe
        # topology), and split that node's cores evenly among the GPUs that share it. Fall back
        # to a positional guess only when the GPU->NUMA topology cannot be read.
        my_node, n_on_node, my_pos = gpu_node, gpu_n_on_node, gpu_pos
        if my_node is not None and my_node in node_cores_by_id:
            node_cores = node_cores_by_id[my_node]
            if n_on_node and n_on_node > 1 and my_pos is not None:
                chunk = max(1, len(node_cores) // n_on_node)
                start = my_pos * chunk
                rank_cores = node_cores[start:start + chunk] if start < len(node_cores) else node_cores
            else:
                rank_cores = node_cores
        else:
            rank_cores = heuristic_rank_cores(numa_nodes, local_rank, local_world_size)

    if not rank_cores:
        total = psutil.cpu_count(logical=False) or 1
        rank_cores = list(range(total))
    rank_cores = sorted(rank_cores)
    total_cpus = len(rank_cores)

    # Group this rank's cores by NUMA node so worker masks stay within a single node.
    rank_set = set(rank_cores)
    node_groups = []
    for node in numa_nodes:
        cores_on_node = [c for c in node if c in rank_set]
        if cores_on_node:
            node_groups.append(cores_on_node)
    accounted = set(c for node_group in node_groups for c in node_group)
    leftover = [c for c in rank_cores if c not in accounted]
    if leftover:
        node_groups.append(sorted(leftover))
    if not node_groups:
        node_groups = [rank_cores]
    # Reserve cores for the main (forward/backward) thread from the first node; the remaining
    # cores become optimizer-worker cores, kept grouped per node.
    main_core_count = min(max(1, int(main_thread_cores)), max(1, total_cpus - 1))
    main_cores = rank_cores[:main_core_count]
    main_set = set(main_cores)
    node_worker_groups = []
    for node_group in node_groups:
        worker_cores_on_node = [c for c in node_group if c not in main_set]
        if worker_cores_on_node:
            node_worker_groups.append(worker_cores_on_node)
    if not node_worker_groups:
        node_worker_groups = [rank_cores]
    worker_cores = [c for node_group in node_worker_groups for c in node_group]
    return total_cpus, rank_cores, node_worker_groups, main_cores, worker_cores


def merge_contiguous_ranges(ranges):
    # Merge contiguous (offset, size) tuples to reduce copy call overhead.
    if not ranges:
        return []
    ranges_sorted = sorted(ranges, key=lambda x: x[0])
    merged = []
    cur_off, cur_sz = ranges_sorted[0]
    for off, sz in ranges_sorted[1:]:
        if off == cur_off + cur_sz:
            cur_sz += sz
        else:
            merged.append((cur_off, cur_sz))
            cur_off, cur_sz = off, sz
    merged.append((cur_off, cur_sz))
    return merged


def clip_range_to_valid_partition(valid_partition_element_count, dest_offset, num_elements):
    """Clamp a single (dest_offset, num_elements) range to the valid partition.

    Returns (offset, size) clamped so it never extends past valid_partition_element_count,
    or None if the range starts at or beyond the valid region (i.e. it is entirely padding).
    """
    if dest_offset >= valid_partition_element_count:
        return None
    clipped_num_elements = min(int(num_elements), valid_partition_element_count - int(dest_offset))
    if clipped_num_elements <= 0:
        return None
    return (int(dest_offset), int(clipped_num_elements))


def clip_ranges_to_valid_partition(ranges, valid_partition_element_count):
    """Clamp a list of (offset, size) ranges to the valid partition, dropping empties."""
    clipped = []
    for off, sz in ranges:
        clipped_range = clip_range_to_valid_partition(valid_partition_element_count, off, sz)
        if clipped_range is not None:
            clipped.append(clipped_range)
    return clipped


def split_ranges_by_min_size(merged_ranges, min_range_elems):
    """Partition merged ranges into (submit, pending) by a minimum element threshold.

    Ranges whose size is >= min_range_elems are returned in ``submit``; smaller ranges are
    returned in ``pending`` and their total element count is returned as ``pending_elems``.
    """
    submit = []
    pending = []
    pending_elems = 0
    for off, sz in merged_ranges:
        if sz >= min_range_elems:
            submit.append((off, sz))
        else:
            pending.append((off, sz))
            pending_elems += sz
    return submit, pending, pending_elems
