# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import math
import torch
import psutil
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator


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


def zenflow_optimizer_process(pipe, param_groups, shared_overlap_grad_map, shared_stale_param_map, zf_affinity):
    disable_accelerator()

    current_process = psutil.Process()
    current_process.cpu_affinity(zf_affinity)
    os.environ['OMP_NUM_THREADS'] = str(len(zf_affinity))

    from deepspeed.ops.adam import ZenFlowCPUAdam
    optimizer = ZenFlowCPUAdam(param_groups, overlap_step=True)

    pipe.send({"type": "ready"})

    # TODO: replace this with rpc

    while True:
        cmd = pipe.recv()
        if cmd["type"] == "step":
            now_state = cmd["now_state"]
            micro_step = cmd["micro_step"]
            group_infos = cmd["group_infos"]

            for group_no, group_info in enumerate(group_infos):
                original_param_groups = optimizer.param_groups
                optimizer.param_groups = [original_param_groups[group_no]]
                group = optimizer.param_groups[0]

                for param_idx, param in enumerate(group["params"]):
                    key = (group_no, param_idx)
                    if key in shared_overlap_grad_map:
                        param.overlap_grad = shared_overlap_grad_map[key]
                    if key in shared_stale_param_map:
                        param.stale_param = shared_stale_param_map[key]

                optimizer.step(step_id=micro_step + 1, now_state=now_state, group_info=group_info)

                optimizer.param_groups = original_param_groups

            pipe.send({"type": "done"})
        elif cmd["type"] == "exit":
            break


def all_tensors_equal(tensor_list):
    first_tensor = tensor_list[0]
    for tensor in tensor_list[1:]:
        if not torch.equal(first_tensor, tensor):
            return False
    return True


def start_optimizer_process(zf_optimizer):
    from multiprocessing import Pipe, get_context, Manager

    ctx = get_context("spawn")
    zf_optimizer.parent_conn, zf_optimizer.child_conn = Pipe()

    manager = Manager()
    zf_optimizer.shared_overlap_grad_map = manager.dict()
    zf_optimizer.shared_stale_param_map = manager.dict()

    if zf_optimizer.zf_stage3:
        params_iter = [((group_no, 0), param)
                       for group_no, param in enumerate(zf_optimizer.fp32_partitioned_groups_flat)]
    else:
        params_iter = [((group_no, param_idx), param)
                       for group_no, group in enumerate(zf_optimizer.optimizer.param_groups)
                       for param_idx, param in enumerate(group["params"])]

    for key, param in params_iter:
        param.data.share_memory_()

        if not hasattr(param, "stale_param"):
            param.stale_param = torch.zeros_like(param.data, dtype=param.dtype, device=param.device)
            param.stale_param.data.share_memory_()
            zf_optimizer.shared_stale_param_map[key] = param.stale_param

        if getattr(param, "overlap_grad", None) is not None:
            param.overlap_grad[0].data.share_memory_()
            param.overlap_grad[1].data.share_memory_()
            zf_optimizer.shared_overlap_grad_map[key] = param.overlap_grad

    param_groups_data = ([{
        "params": [param]
    } for param in zf_optimizer.fp32_partitioned_groups_flat]
                         if zf_optimizer.zf_stage3 else zf_optimizer.optimizer.param_groups)

    curr_rank = dist.get_rank()
    total_rank = dist.get_world_size()

    current_process = psutil.Process()
    current_affinity = current_process.cpu_affinity()
    all_affinities = [
        torch.zeros(len(current_affinity),
                    dtype=type(current_affinity[0]),
                    device=get_accelerator().current_device_name()) for _ in range(total_rank)
    ]
    dist.all_gather(
        all_affinities,
        torch.tensor(current_affinity, dtype=type(current_affinity[0]),
                     device=get_accelerator().current_device_name()))
    # When affinity across all ranks are the same, the workers are not binded.  Do a soft bind here
    if all_tensors_equal(all_affinities):
        num_phy_cores = psutil.cpu_count(logical=False)
        available_phy_cores = [i for i in current_affinity if i < num_phy_cores]
        num_available_phy_cores = len(available_phy_cores)
        my_rank = curr_rank
        my_size = total_rank
        cores_per_rank = num_available_phy_cores // my_size
        current_affinity = available_phy_cores[my_rank * cores_per_rank:(my_rank + 1) * cores_per_rank]
    pt_num_cores = math.ceil(zf_optimizer.pt_reserved_cores_perc * len(current_affinity))
    if pt_num_cores > 0 and pt_num_cores < len(current_affinity):
        zf_affinity = current_affinity[pt_num_cores:]
        pt_affinity = current_affinity[:pt_num_cores]
    else:
        zf_affinity = current_affinity
        pt_affinity = current_affinity

    zf_optimizer.process = ctx.Process(
        target=zenflow_optimizer_process,
        args=(zf_optimizer.child_conn, param_groups_data, zf_optimizer.shared_overlap_grad_map,
              zf_optimizer.shared_stale_param_map, zf_affinity),
    )
    zf_optimizer.process.daemon = True
    zf_optimizer.process.start()

    current_process.cpu_affinity(pt_affinity)
    os.environ['OMP_NUM_THREADS'] = str(len(pt_affinity))

    msg = zf_optimizer.parent_conn.recv()
    assert msg["type"] == "ready", "Optimizer process did not initialize correctly."

    zf_optimizer.process_optimizer_established = True
