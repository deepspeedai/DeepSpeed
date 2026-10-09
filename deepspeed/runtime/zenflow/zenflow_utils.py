# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import math
import socket
import hashlib
import torch
import psutil
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator

# How long the training side blocks on a single semaphore wait for the optimizer process before
# waking up to check that the process is still alive. A normal step completes far sooner; this
# only bounds how long we hang if the optimizer process dies mid-step.
ZENFLOW_OPTIMIZER_WAIT_POLL_SECONDS = 60


def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    transposed_tensors = [t.transpose(0, 1).contiguous() if t.dim() == 2 else t for t in tensors]
    return torch._C._nn.flatten_dense_tensors(transposed_tensors)


def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Args:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    transposed_tensors = [t.transpose(0, 1) if t.dim() == 2 else t for t in tensors]
    unflat = torch._C._nn.unflatten_dense_tensors(flat, transposed_tensors)
    return [t.transpose(0, 1) if t.dim() == 2 else t for t in unflat]


def disable_accelerator():
    accelerator = get_accelerator()
    accelerator.is_available = lambda: False
    accelerator.device_count = lambda: 0
    accelerator.current_device = lambda: -1
    # Optionally mark it as initialized if needed
    if hasattr(accelerator, "_initialized"):
        accelerator._initialized = True


def _split_affinity(cores, pt_reserved_cores_perc):
    """Split a rank's core list into (zf_affinity, pt_affinity): reserve the first
    ceil(pt_reserved_cores_perc * n) cores for the training thread and give the rest to the
    optimizer. If the reserve rounds to zero or to every core, both share the full set (there
    is nothing to gain from isolating an empty side)."""
    pt_num_cores = math.ceil(pt_reserved_cores_perc * len(cores))
    if 0 < pt_num_cores < len(cores):
        return cores[pt_num_cores:], cores[:pt_num_cores]
    return cores, cores


def _env_int(*names):
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def _format_cpu_list(cpus):
    if not cpus:
        return "(none)"
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


def _physical_core_key(cpu):
    """One logical CPU per physical core. core_id repeats across sockets, so the package id is part of the key."""
    topology = f"/sys/devices/system/cpu/cpu{cpu}/topology"
    try:
        with open(os.path.join(topology, "physical_package_id")) as handle:
            package = int(handle.read())
        with open(os.path.join(topology, "core_id")) as handle:
            core = int(handle.read())
    except (OSError, ValueError):
        return (cpu, 0)
    return (package, core)


def _one_logical_per_physical_core(allowed):
    chosen = {}
    for cpu in allowed:
        key = _physical_core_key(cpu)
        current = chosen.get(key)
        if current is None or cpu < current:
            chosen[key] = cpu
    return [chosen[key] for key in sorted(chosen)]


def _affinity_device():
    # Collectives in a GPU job run on the accelerator. A CPU tensor here breaks NCCL.
    return get_accelerator().current_device_name()


def _host_token():
    digest = hashlib.sha256(socket.gethostname().encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=True)


def _local_rank_and_size_from_hosts():
    world = dist.get_world_size()
    rank = dist.get_rank()
    device = _affinity_device()
    token = torch.tensor([_host_token()], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(token) for _ in range(world)]
    dist.all_gather(gathered, token)
    mine = int(gathered[rank].item())
    group = [index for index, item in enumerate(gathered) if int(item.item()) == mine]
    return group.index(rank), len(group)


def _local_rank_and_size():
    # Keep each rank name paired with its own size. Mixing a rank from one launcher with a size
    # from another reports the wrong node shape when both are present and disagree.
    pairs = (
        ("LOCAL_RANK", "LOCAL_WORLD_SIZE"),
        ("LOCAL_RANK", "LOCAL_SIZE"),
        ("OMPI_COMM_WORLD_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_SIZE"),
        ("MPI_LOCALRANKID", "MPI_LOCALNRANKS"),
        ("SLURM_LOCALID", "SLURM_NTASKS_PER_NODE"),
    )
    for rank_name, size_name in pairs:
        local_rank = _env_int(rank_name)
        local_size = _env_int(size_name)
        if local_rank is not None and local_size is not None:
            return local_rank, local_size
    return _local_rank_and_size_from_hosts()


def _masks_match(current_affinity):
    world = dist.get_world_size()
    if world <= 1:
        return True
    # Lengths first: an all_gather of the ids themselves deadlocks on NCCL when ranks were
    # handed different-sized masks. Different lengths are already enough to say they differ.
    device = _affinity_device()
    length = torch.tensor([len(current_affinity)], dtype=torch.int64, device=device)
    lengths = [torch.empty_like(length) for _ in range(world)]
    dist.all_gather(lengths, length)
    if not all(torch.equal(length, item) for item in lengths):
        return False
    if not current_affinity:
        return True
    ids = torch.tensor(sorted(current_affinity), dtype=torch.int64, device=device)
    gathered = [torch.empty_like(ids) for _ in range(world)]
    dist.all_gather(gathered, ids)
    return all(torch.equal(ids, item) for item in gathered)


def _shard_physical_cores(allowed):
    physical = _one_logical_per_physical_core(allowed)
    local_rank, local_size = _local_rank_and_size()
    allowed_text = _format_cpu_list(allowed)
    if local_size < 1 or not 0 <= local_rank < local_size:
        raise RuntimeError(f"ZenFlow local rank {local_rank} is outside local size {local_size}. "
                           f"Allowed CPUs: {allowed_text}.")
    if len(physical) < local_size:
        raise RuntimeError(f"ZenFlow has {len(physical)} usable physical cores for {local_size} local ranks. "
                           f"Allowed CPUs: {allowed_text}.")
    per_rank = len(physical) // local_size
    start = local_rank * per_rank
    return physical[start:start + per_rank]


def _compute_zf_pt_affinity(zf_optimizer):
    """Split this rank's cores into a ZenFlow-optimizer set and a training (PyTorch) set.

    When every rank reports the same affinity the launcher did not bind workers. Shard the
    allowed CPUs by local rank, keeping one logical CPU per physical core, then carve off
    pt_reserved_cores_perc for training. Ranks that already have different masks keep them.
    """
    current_affinity = list(psutil.Process().cpu_affinity() or [])
    # Compare masks before raising so every rank joins the collective, including a rank whose
    # allowed set is empty.
    masks_match = _masks_match(current_affinity)
    if not current_affinity:
        raise RuntimeError("ZenFlow found no allowed CPUs. Allowed CPUs: (none).")
    if masks_match:
        current_affinity = _shard_physical_cores(current_affinity)
    return _split_affinity(current_affinity, zf_optimizer.pt_reserved_cores_perc)


def zenflow_optimizer_process(groups, ctrl, ready, zf_affinity, adamw_mode):
    """ZenFlow overlapped optimizer process (ZeRO stage 1/2/3). Builds the native ZenFlowAdam
    pinned pool and runs the worker loop driven by the shared-memory control block (no pickling
    pipe). The Adam state is allocated here, in this process pinned to the optimizer cores, so
    it is NUMA-local to the pool -- which is what makes a separate process worthwhile over an
    in-process thread for large, memory-bandwidth-bound updates."""
    disable_accelerator()
    current_process = psutil.Process()
    current_process.cpu_affinity(zf_affinity)
    os.environ['OMP_NUM_THREADS'] = str(len(zf_affinity))

    from deepspeed.ops.op_builder import CPUAdamBuilder
    op = CPUAdamBuilder().load()
    op.create_adam(0, 1e-3, 0.9, 0.999, 1e-8, 0.0, adamw_mode, False)
    handle = op.zenflow_adam_create(0, list(zf_affinity))
    for param, overlap_grad0, overlap_grad1, stale in groups:
        exp_avg0 = torch.zeros_like(param)
        exp_avg1 = torch.zeros_like(param)
        exp_avg_sq0 = torch.zeros_like(param)
        exp_avg_sq1 = torch.zeros_like(param)
        op.zenflow_adam_register_group(handle, param, overlap_grad0, overlap_grad1, exp_avg0, exp_avg1, exp_avg_sq0,
                                       exp_avg_sq1, stale)
    ready.set()
    op.zenflow_adam_run_worker(handle, ctrl.data_ptr())
    op.zenflow_adam_destroy(handle)
    op.destroy_adam(0)


def start_optimizer_process(zf_optimizer):
    """Start ZenFlow's overlapped optimizer (ZeRO stage 1/2/3) in a dedicated process,
    coordinated through a shared-memory semaphore control block. A separate process keeps the
    Adam state NUMA-local to the optimizer cores and free of contention with the training
    thread, while the native control block avoids per-step Python/IPC overhead."""
    from multiprocessing import get_context
    from deepspeed.ops.op_builder import CPUAdamBuilder

    op = CPUAdamBuilder().load()
    zf_optimizer.zf_op = op

    # Stage 3 steps each flattened sub-group partition; stage 1/2 steps one flat partition per
    # param group. Both carry overlap_grad double buffers and a stale snapshot.
    if zf_optimizer.zf_stage3:
        params = list(zf_optimizer.fp32_partitioned_groups_flat)
    else:
        params = [group["params"][0] for group in zf_optimizer.optimizer.param_groups]

    # Share the tensors the optimizer process reads/writes; the Adam state stays process-local.
    groups = []
    for param in params:
        param.data.share_memory_()
        if not hasattr(param, "stale_param"):
            param.stale_param = torch.zeros_like(param.data, dtype=param.dtype, device=param.device)
        param.stale_param.data.share_memory_()
        param.overlap_grad[0].data.share_memory_()
        param.overlap_grad[1].data.share_memory_()
        groups.append((param.data, param.overlap_grad[0].data, param.overlap_grad[1].data, param.stale_param.data))

    ctrl = torch.zeros(op.zenflow_adam_ctrl_size(), dtype=torch.uint8).share_memory_()
    op.zenflow_adam_ctrl_init(ctrl.data_ptr(), len(groups))
    zf_optimizer.zf_ctrl = ctrl

    zf_affinity, pt_affinity = _compute_zf_pt_affinity(zf_optimizer)

    ctx = get_context("spawn")
    ready = ctx.Event()
    proc = ctx.Process(target=zenflow_optimizer_process, args=(groups, ctrl, ready, zf_affinity, True))
    proc.daemon = True
    proc.start()
    # Wait for the optimizer process to finish building its pool and registering tensors.
    # If it crashed during init (e.g. it never signals), fail loudly instead of blocking the
    # training process forever on the first step's wait.
    if not ready.wait(timeout=600):
        proc.terminate()
        raise RuntimeError("ZenFlow optimizer process failed to become ready (it likely crashed "
                           "during initialization; check the optimizer process traceback above)")
    zf_optimizer.process = proc

    psutil.Process().cpu_affinity(pt_affinity)
    os.environ['OMP_NUM_THREADS'] = str(len(pt_affinity))

    zf_optimizer.process_optimizer_established = True
