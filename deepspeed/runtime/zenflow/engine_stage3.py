# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.zero.partition_parameters import *

import torch
import math
from deepspeed import comm as dist
from deepspeed.utils import logger
from deepspeed.ops.adam import ZenFlowSelectiveAdamW_stage3
from deepspeed.runtime.utils import see_memory_usage
from typing import List
from deepspeed.accelerator import get_accelerator
from typing import TYPE_CHECKING
from deepspeed.runtime.zenflow.zenflow_utils import start_optimizer_process

if TYPE_CHECKING:
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3

OPTIMIZER_SWAP_IN_STATE_TIMER = 'optimizer_swap_in_state'
INIT_OPTIMIZER_TIMER = 'init_optimizer_state'
OPTIMIZER_SWAP_OUT_STATE_TIMER = 'optimizer_swap_out_state'
OPTIMIZER_STEP_TIMER = 'optimizer_step'


def configure_zenflow(optimizer_z3, zenflow_config):

    optimizer_z3.select_strategy = zenflow_config.select_strategy
    if optimizer_z3.select_strategy == 'auto':
        optimizer_z3.select_strategy = "epoch"
        if isinstance(zenflow_config.select_interval, int):
            raise Warning(
                "If use auto select strategy, select_interval will be set to 1 and select_strategy will be set to epoch, thus select_interval would be overwritten."
            )
        optimizer_z3.select_interval = 1
    else:
        if isinstance(zenflow_config.select_interval, str):
            raise ValueError("If don't use auto select strategy, select_interval must be a number.")
        optimizer_z3.select_interval = int(zenflow_config.select_interval)

    if isinstance(zenflow_config.update_interval, str):
        optimizer_z3.auto_update = True
        optimizer_z3.update_interval = 0
    else:
        optimizer_z3.auto_update = False
        optimizer_z3.update_interval = int(zenflow_config.update_interval)

    if optimizer_z3.select_strategy == 'epoch':
        if zenflow_config.steps_per_epoch is not None:
            optimizer_z3.select_interval = optimizer_z3.select_interval * zenflow_config.steps_per_epoch
        else:
            optimizer_z3.select_interval = 0

    if not optimizer_z3.auto_update and optimizer_z3.select_interval != 0 and optimizer_z3.select_interval < optimizer_z3.update_interval:
        raise ValueError("Select interval must be greater or equal to update interval")

    optimizer_z3.topk_ratio = zenflow_config.topk_ratio

    optimizer_z3.param_id_grad_sum_buffer_offset = {}

    optimizer_z3.zf_stage3 = True

    if optimizer_z3.auto_update:
        optimizer_z3.param_id_sum_buffer_offset = {}
        optimizer_z3.auto_ratio = zenflow_config.auto_ratio
        optimizer_z3.zenflow_need_update = [False, False]
        optimizer_z3.zenflow_state = 0
        optimizer_z3.num_need_update = 0


def _initialize_zenflow_stage3_prologue(optimizer_z3: "DeepSpeedZeroOptimizer_Stage3",
                                        module,
                                        zenflow_config: dict = None):

    optimizer_z3.zenflow = True if zenflow_config is not None else False

    if not optimizer_z3.zenflow:
        return

    optimizer_z3.pt_reserved_cores_perc = zenflow_config.pt_reserved_cores_perc

    for p in module.parameters():
        p.data = p.data.t().contiguous() if len(p.shape) != 1 else p.data


def _initialize_zenflow_stage3_epilogue(optimizer_z3: "DeepSpeedZeroOptimizer_Stage3",
                                        zenflow_config: dict = None,
                                        overlap_comm: bool = False):

    if not optimizer_z3.zenflow:
        return

    optimizer_z3.micro_step = -1
    optimizer_z3.full_warm_up_rounds = zenflow_config.full_warm_up_rounds
    optimizer_z3.offload_selective_optimizer = zenflow_config.offload
    optimizer_z3.zenflow_overlap_step = zenflow_config.overlap_step

    if optimizer_z3.offload_selective_optimizer:
        assert overlap_comm, "offload selective optimizer should be used with overlap_comm"

    if optimizer_z3.zenflow_overlap_step:
        optimizer_z3.process_optimizer_established = False
        optimizer_z3.first_update_round_after_warmup = True
        optimizer_z3.initialize_optimizer_states = lambda: initialize_optimizer_states(optimizer_z3)
        optimizer_z3.step = lambda closure=None: step(optimizer_z3, closure)
        optimizer_z3.zenflow_cpu_optimizer_overlap_step = lambda now_state, scaled_global_grad_norm: zenflow_cpu_optimizer_overlap_step(
            optimizer_z3, now_state, scaled_global_grad_norm)
        optimizer_z3.wait_last_update_and_copy = lambda timer_names: wait_last_update_and_copy(
            optimizer_z3, timer_names)
        optimizer_z3.partition_grads = lambda params_to_release, grad_partitions: partition_grads(
            optimizer_z3, params_to_release, grad_partitions)
        optimizer_z3.get_overlap_step_state = lambda: get_overlap_step_state(optimizer_z3)
        optimizer_z3.start_optimizer_process = lambda: start_optimizer_process(optimizer_z3)
        optimizer_z3.unscale_and_clip_grads = lambda sub_group_id, total_norm, now_state: unscale_and_clip_grads(
            optimizer_z3, sub_group_id, total_norm, now_state)

    configure_zenflow(optimizer_z3, zenflow_config)
    optimizer_z3.selective_optimizer = ZenFlowSelectiveAdamW_stage3([{
        k: v
        for k, v in group.items() if k != "params"
    } | {
        "params": group["params"]
    } for group in optimizer_z3.optimizer.param_groups],
                                                                    offload=optimizer_z3.offload_selective_optimizer)
    optimizer_z3.num_total_param = sum(
        sum(1 for param in group["params"] if len(param.ds_shape) != 1)
        for group in optimizer_z3.optimizer.param_groups)


def zenflow_cpu_optimizer_step(optimizer_z3: "DeepSpeedZeroOptimizer_Stage3"):
    return optimizer_z3.optimizer.step(step_id=optimizer_z3.micro_step + 1)


def _sync_selective_optimizer_lr(optimizer_z3: "DeepSpeedZeroOptimizer_Stage3"):
    for group_selected, group in zip(optimizer_z3.selective_optimizer.param_groups,
                                     optimizer_z3.optimizer.param_groups):
        group_selected["lr"] = group["lr"]


def selective_optimizer_step(optimizer_z3: "DeepSpeedZeroOptimizer_Stage3"):
    optimizer_z3.selective_optimizer.step()


def is_zenflow_select_boundary(optimizer_z3: "DeepSpeedZeroOptimizer_Stage3") -> bool:
    return optimizer_z3.zenflow and (optimizer_z3.micro_step - optimizer_z3.full_warm_up_rounds) >= 0 and (
        (optimizer_z3.micro_step - optimizer_z3.full_warm_up_rounds) == 0 or
        (optimizer_z3.select_interval != 0 and optimizer_z3.micro_step % optimizer_z3.select_interval == 0))


def update_selected_channels(optimizer_z3: "DeepSpeedZeroOptimizer_Stage3", params_to_update, grad_partitions):
    src_rk = dist.get_rank(optimizer_z3.dp_process_group)
    total_rk = dist.get_world_size(optimizer_z3.dp_process_group)

    total_chunk_size = 0
    param_local_offset = [0 for _ in range(total_rk)]

    for param, grad_partition in zip(params_to_update, grad_partitions):
        param_max_chunk_size = 0
        param_rk_offset = 0
        for rk in range(total_rk):
            contains_real_data = param.partition_numel() * rk < param.ds_numel
            if not contains_real_data:
                param.grad = None
                continue

            num_column, num_row = param.ds_shape if len(param.ds_shape) != 1 else (param.ds_shape[0], 1)
            if num_row == 1:
                continue

            partition_size = param.partition_numel()
            start = partition_size * rk
            end = min(start + partition_size, param.ds_numel)

            start_idx = math.ceil(start / num_row)
            end_idx = end // num_row
            num_cols = end_idx - start_idx

            if param.ds_id not in optimizer_z3.param_id_grad_sum_buffer_offset:
                optimizer_z3.param_id_grad_sum_buffer_offset[param.ds_id] = []

            optimizer_z3.param_id_grad_sum_buffer_offset[param.ds_id].append(
                (param_local_offset[rk], num_cols, param_rk_offset))

            param_max_chunk_size = max(param_max_chunk_size, num_cols)
            param_rk_offset += num_cols
            param_local_offset[rk] += num_cols

        total_chunk_size += param_max_chunk_size

    optimizer_z3.grad_sum_buffer = torch.zeros(total_chunk_size, dtype=optimizer_z3.dtype, device='cuda')

    for param, grad_partition in zip(params_to_update, grad_partitions):
        contains_real_data = param.partition_numel() * src_rk < param.ds_numel
        if not contains_real_data:
            # this grad partition is empty - don't need to do anything
            param.grad = None
            continue

        #ds_shape is the transposed shape, it should not be same as param.shape
        num_column, num_row = param.ds_shape if len(param.ds_shape) != 1 else (param.ds_shape[0], 1)

        if num_row == 1:
            continue

        partition_size = param.partition_numel()
        start = partition_size * src_rk
        end = min(start + partition_size, param.ds_numel)

        start_idx = math.ceil(start / num_row)
        end_idx = end // num_row

        num_elements = (end_idx - start_idx) * num_row

        param.complete_column_offset = start_idx * num_row - start
        param.complete_numel = (end_idx - start_idx) * num_row

        sum_per_column = grad_partition.narrow(0, param.complete_column_offset, num_elements)
        sum_per_column = sum_per_column.view(end_idx - start_idx, num_row)
        sum_array = sum_per_column.abs().sum(dim=1)

        offset, length, _ = optimizer_z3.param_id_grad_sum_buffer_offset[param.ds_id][src_rk]
        optimizer_z3.grad_sum_buffer.narrow(0, offset, length).copy_(sum_array)

    gathered_chunks = [torch.zeros_like(optimizer_z3.grad_sum_buffer) for _ in range(total_rk)]
    dist.all_gather(gathered_chunks, optimizer_z3.grad_sum_buffer, group=optimizer_z3.dp_process_group)

    for param in params_to_update:

        num_column, num_row = param.ds_shape if len(param.ds_shape) != 1 else (param.ds_shape[0], 1)

        if num_row == 1:
            continue

        param_column_sum = []
        for rk in range(total_rk):
            offset, length, _ = optimizer_z3.param_id_grad_sum_buffer_offset[param.ds_id][rk]
            param_column_sum.append(gathered_chunks[rk].narrow(0, offset, length))
        global_param_column_sum = torch.cat(param_column_sum, dim=0)

        num_select = max(1, int(global_param_column_sum.numel() * optimizer_z3.topk_ratio))
        _, global_topk_indices = torch.topk(global_param_column_sum, num_select, largest=True)

        _, length, rk_offset = optimizer_z3.param_id_grad_sum_buffer_offset[param.ds_id][src_rk]
        local_indices = [(idx.item() - rk_offset) for idx in global_topk_indices
                         if rk_offset <= idx < rk_offset + length]
        param.selected_indices = torch.tensor(local_indices, device='cuda')
        optimizer_z3.param_id_grad_sum_buffer_offset[param.ds_id] = []

    optimizer_z3.grad_sum_buffer = None


def _process_selected_fp32_groups_grad(optimizer_z3, params_to_update, grad_partitions):

    if optimizer_z3.auto_update:
        optimizer_z3.sum_buffer = torch.zeros(optimizer_z3.num_total_param, dtype=optimizer_z3.dtype, device='cuda')
        optimizer_z3.critic_sum_buffer = torch.zeros(optimizer_z3.num_total_param,
                                                     dtype=optimizer_z3.dtype,
                                                     device='cuda')
        curr_buffer_idx = 0

    for param, grad_partition in zip(params_to_update, grad_partitions):

        rk = dist.get_rank(optimizer_z3.dp_process_group)

        contains_real_data = param.partition_numel() * rk < param.ds_numel
        if not contains_real_data:
            # this grad partition is empty - don't need to do anything
            param.grad = None
            continue

        #ds_shape is the transposed shape, it should not be same as param.shape
        num_column, num_row = param.ds_shape if len(param.ds_shape) != 1 else (param.ds_shape[0], 1)

        if num_row == 1:
            param.selected_grad = grad_partition.clone().detach()
        else:
            grad_2d = grad_partition.narrow(0, param.complete_column_offset,
                                            param.complete_numel).view(param.complete_numel // num_row, num_row)
            param.selected_grad = grad_2d[param.selected_indices, :].clone().detach()

            if optimizer_z3.auto_update:
                optimizer_z3.sum_buffer[curr_buffer_idx] = grad_partition.abs().sum()
                optimizer_z3.critic_sum_buffer[curr_buffer_idx] = param.selected_grad.abs().sum()
                curr_buffer_idx += 1

        if optimizer_z3.offload_selective_optimizer and not hasattr(param, 'exp_avg_cpu_data'):
            buffer = torch.zeros(param.selected_grad.numel(), dtype=param.dtype, device=optimizer_z3.device)
            param.exp_avg_cpu_data = get_accelerator().pin_memory(
                buffer) if optimizer_z3.offload_optimizer_pin_memory else buffer
            param.exp_avg_sq_cpu_data = get_accelerator().pin_memory(
                buffer.clone()) if optimizer_z3.offload_optimizer_pin_memory else buffer.clone()

    if optimizer_z3.auto_update:
        total_rk = dist.get_world_size(optimizer_z3.dp_process_group)
        sum_gather_list = [torch.zeros_like(optimizer_z3.sum_buffer) for _ in range(total_rk)]
        critic_gather_list = [torch.zeros_like(optimizer_z3.critic_sum_buffer) for _ in range(total_rk)]
        curr_buffer_idx = 0

        dist.all_gather(sum_gather_list, optimizer_z3.sum_buffer, group=optimizer_z3.dp_process_group)
        dist.all_gather(critic_gather_list, optimizer_z3.critic_sum_buffer, group=optimizer_z3.dp_process_group)

        for param in params_to_update:
            if len(param.ds_shape) == 1:
                continue

            if not hasattr(param, 'non_critic_sum'):
                param.non_critic_sum = 0
            if not hasattr(param, 'avg_critic_sum'):
                param.avg_critic_sum = 0

            grad_total_sum = sum(sum_gather_list[rk][curr_buffer_idx] for rk in range(total_rk))
            grad_critic_sum = sum(critic_gather_list[rk][curr_buffer_idx] for rk in range(total_rk))

            param.avg_critic_sum = (param.avg_critic_sum * (optimizer_z3.update_interval - 1) +
                                    grad_critic_sum) / optimizer_z3.update_interval / (optimizer_z3.topk_ratio * 10)
            param.non_critic_sum += (grad_total_sum - grad_critic_sum) / ((1 - optimizer_z3.topk_ratio) * 10)
            if param.non_critic_sum >= param.avg_critic_sum:
                optimizer_z3.num_need_update += 1
            if optimizer_z3.num_need_update >= int(optimizer_z3.auto_ratio * optimizer_z3.num_total_param):
                optimizer_z3.zenflow_need_update[optimizer_z3.zenflow_state] = True

            curr_buffer_idx += 1

    if not optimizer_z3.is_gradient_accumulation_boundary:
        optimizer_z3.selective_optimizer.group_step(params_to_update)
    else:
        optimizer_z3.selective_optimizer.temp_copy_param(params_to_update)

    if optimizer_z3.auto_update:
        optimizer_z3.sum_buffer = None
        optimizer_z3.critic_sum_buffer = None


def sync_fp32_param_from_gpu(optimizer_z3: "DeepSpeedZeroOptimizer_Stage3"):

    if optimizer_z3.micro_step == 0:
        return

    for fp16_partitions, fp32_partition in zip(optimizer_z3.fp16_partitioned_groups_flat,
                                               optimizer_z3.fp32_partitioned_groups_flat):
        fp32_partition.data.copy_(fp16_partitions.data)


def zenflow_backward_prologue(optimizer_z3: "DeepSpeedZeroOptimizer_Stage3"):
    optimizer_z3.micro_step += 1
    if optimizer_z3.auto_update:
        optimizer_z3.zenflow_need_update[optimizer_z3.zenflow_state] = False
        optimizer_z3.num_need_update = 0
        if optimizer_z3.zenflow_need_update[optimizer_z3.zenflow_state ^ 1]:
            optimizer_z3.update_interval = 0
            for group in optimizer_z3.fp16_groups:
                for p in group:
                    p.non_critic_sum = 0
        optimizer_z3.update_interval += 1
    if optimizer_z3.is_zenflow_select_boundary():
        sync_fp32_param_from_gpu(optimizer_z3)
        optimizer_z3.selective_optimizer.clear_selected_mv()


def zenflow_backward_epilogue(optimizer_z3: "DeepSpeedZeroOptimizer_Stage3"):
    optimizer_z3._partition_all_parameters()


def log_selective_optimizer_timers(optimizer_z3: "DeepSpeedZeroOptimizer_Stage3"):
    pass


def initialize_optimizer_states(optimizer_z3: "DeepSpeedZeroOptimizer_Stage3"):
    num_subgroups = len(optimizer_z3.fp16_groups)

    largest_numel = max([sum([p.ds_numel for p in psg]) for psg in optimizer_z3.fp16_partitioned_groups])
    gradient_dtype = optimizer_z3.fp32_partitioned_groups_flat[0].dtype
    gradient_buffer = torch.zeros(int(largest_numel), dtype=gradient_dtype, device=optimizer_z3.device)

    timer_names = set()

    # State initialization for the Adagrad optimizer occurs at construction as opposed to other optimizers
    # which do lazy initialization of the state at the first call to step.
    is_adagrad = isinstance(optimizer_z3.optimizer, torch.optim.Adagrad)

    if optimizer_z3.swap_optimizer:
        optimizer_z3.optimizer_swapper.init_timers()

    timer_names.add(INIT_OPTIMIZER_TIMER)
    optimizer_z3.timers(INIT_OPTIMIZER_TIMER).start()

    for i, group in enumerate(optimizer_z3.fp16_groups):
        swappable_optimizer_subgroup = optimizer_z3._swappable_optimizer_subgroup(i)
        swappable_param_subgroup = optimizer_z3.fp16_partitioned_groups_flat[i] is None

        num_elements = int(optimizer_z3.fp16_partitioned_groups_flat_numel[i])

        see_memory_usage(
            f'[Begin] Initialize optimizer states {i} / {num_subgroups} subgroups, num_elems: {num_elements}, swappable opt/param:{swappable_optimizer_subgroup}/{swappable_param_subgroup}',
            force=False)

        if swappable_optimizer_subgroup:
            optimizer_z3._optimizer_states_and_gradient_swap_in(i, timer_names)

        if optimizer_z3.offload_optimizer and not swappable_optimizer_subgroup:
            subgroup_gradient_buffer = torch.zeros(num_elements, dtype=gradient_dtype, device=optimizer_z3.device)
            if optimizer_z3.offload_optimizer_pin_memory:
                subgroup_gradient_buffer = get_accelerator().pin_memory(subgroup_gradient_buffer)

            optimizer_z3.fp32_partitioned_groups_flat[i].grad = None
            optimizer_z3.fp32_partitioned_groups_flat[i].overlap_grad = [
                subgroup_gradient_buffer.to(optimizer_z3.subgroup_to_device[i]),
                subgroup_gradient_buffer.clone().to(optimizer_z3.subgroup_to_device[i])
            ]
        else:
            optimizer_z3.fp32_partitioned_groups_flat[i].grad = gradient_buffer.narrow(0, 0, num_elements)

        if swappable_param_subgroup:
            optimizer_z3._partitioned_params_swap_out(i)

        if swappable_optimizer_subgroup:
            optimizer_z3._optimizer_states_and_gradient_swap_out(i, timer_names)

        see_memory_usage(
            f'[End] Initialize optimizer states {i} / {num_subgroups} subgroups, num_elems: {num_elements}, swappable opt/param:{swappable_optimizer_subgroup}/{swappable_param_subgroup}',
            force=False)

    # Initialize the optimizer states with the flattened fp32 partition.
    if is_adagrad:
        optimizer_z3.optimizer = torch.optim.Adagrad(optimizer_z3.fp32_partitioned_groups_flat,
                                                     **optimizer_z3.optimizer.defaults)

    optimizer_z3.timers(INIT_OPTIMIZER_TIMER).stop()
    optimizer_z3.timers.log(timer_names)

    if optimizer_z3.swap_optimizer:
        optimizer_z3.optimizer_swapper.log_timers()

    if not optimizer_z3.offload_optimizer:
        for group in optimizer_z3.fp32_partitioned_groups_flat:
            group.grad = None

    # Reset steps
    return


def get_overlap_step_state(optimizer_z3: "DeepSpeedZeroOptimizer_Stage3") -> int:
    if optimizer_z3.micro_step < optimizer_z3.full_warm_up_rounds:
        return optimizer_z3.micro_step & 1
    else:
        if not optimizer_z3.auto_update:
            return (optimizer_z3.micro_step // optimizer_z3.update_interval) & 1
        else:
            return optimizer_z3.zenflow_state


@instrument_w_nvtx
def partition_grads(optimizer_z3, params_to_release: List[Parameter], grad_partitions: List[Tensor]) -> None:
    offload_fp32_gradients = {}
    offload_fp32_offsets = {}
    buffers = []
    for param, grad_partition in zip(params_to_release, grad_partitions):

        contains_real_data = param.partition_numel() * dist.get_rank(optimizer_z3.dp_process_group) < param.ds_numel
        if not contains_real_data:
            # this grad partition is empty - don't need to do anything
            param.grad = None
            continue

        # move or accumulate gradient partition to target buffer
        param_id_to_grad_partition = getattr(optimizer_z3,
                                             f"_{optimizer_z3.__class__.__name__}__param_id_to_grad_partition")
        grad_buffer = param_id_to_grad_partition[param.ds_id].narrow(0, 0, grad_partition.numel())
        buffers.append(grad_buffer)
        if optimizer_z3.micro_step_id == 0:  # don't accumulate
            grad_buffer.copy_(grad_partition, non_blocking=True)
            # ensure grad buffer is a CUDA buffer to speed up the next few
            # operations and so it can be used asynchronously
            grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
        elif get_accelerator().on_accelerator(grad_buffer):
            grad_buffer.add_(grad_partition.to(optimizer_z3.gradient_accumulation_dtype).view(grad_buffer.shape))
        else:
            # if dst is CPU, copy first to src device, do the addition
            # there, then move back to dst. adding directly to cpu is very slow
            cuda_grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
            cuda_grad_buffer.add_(
                grad_partition.to(optimizer_z3.gradient_accumulation_dtype).view(cuda_grad_buffer.shape))
            grad_buffer.copy_(cuda_grad_buffer, non_blocking=True)
            # ensure grad buffer is a CUDA buffer to speed up the next few
            # operations and so it can be used asynchronously
            grad_buffer = cuda_grad_buffer

        # offload the gradient partition if applicable
        if optimizer_z3.offload_optimizer:
            i, dest_offset, _ = optimizer_z3.grad_position[optimizer_z3.get_param_id(param)]
            now_state = optimizer_z3.get_overlap_step_state()

            if optimizer_z3.is_gradient_accumulation_boundary:
                optimizer_z3.norm_for_param_grads[optimizer_z3.get_param_id(
                    param)] = optimizer_z3._constant_buffered_norm2(grad_buffer)

                if optimizer_z3._swappable_optimizer_subgroup(i):
                    if not i in offload_fp32_gradients.keys():
                        offload_fp32_gradients[i] = []
                        offload_fp32_offsets[i] = []

                    offload_fp32_gradients[i].append(grad_buffer.float())
                    offload_fp32_offsets[i].append(dest_offset)
                else:
                    fp32_grad_tensor = optimizer_z3.fp32_partitioned_groups_flat[i].overlap_grad[now_state].narrow(
                        0, dest_offset, grad_buffer.numel())
                    fp32_grad_tensor.copy_(grad_buffer.float())

        # free the gradient
        if not get_accelerator().is_synchronized_device():
            if param.grad is not None:
                param.grad.record_stream(get_accelerator().current_stream())
        param.grad = None

    if optimizer_z3.offload_optimizer and optimizer_z3.swap_optimizer:
        for i in offload_fp32_gradients.keys():
            optimizer_z3.optimizer_swapper.swap_out_gradients(parameter=optimizer_z3.fp32_partitioned_groups_flat[i],
                                                              gradient_offsets=offload_fp32_offsets[i],
                                                              gradient_tensors=offload_fp32_gradients[i])
    return buffers


@instrument_w_nvtx
def unscale_and_clip_grads(self, sub_group_id, total_norm, now_state):
    # compute combined scale factor for this group
    combined_scale = self.loss_scale
    if self.clip_grad > 0.:
        # norm is in fact norm*scale
        clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
        clip = torch.clamp(clip, min=1.0)
        combined_scale = clip * self.loss_scale

    self.fp32_partitioned_groups_flat[sub_group_id].overlap_grad[now_state].mul_(1. / combined_scale)


def zenflow_cpu_optimizer_overlap_step(optimizer_z3, now_state, scaled_global_grad_norm):

    if not optimizer_z3.process_optimizer_established:
        optimizer_z3.start_optimizer_process()

    group_infos = []
    for group_no, group in enumerate(optimizer_z3.fp16_groups):
        optimizer_z3.unscale_and_clip_grads(group_no, scaled_global_grad_norm, now_state)
        param_group_id = optimizer_z3.sub_group_to_group_id[group_no]

        group_info = {
            "lr": optimizer_z3.optimizer.param_groups[param_group_id]["lr"],
            "betas": optimizer_z3.optimizer.param_groups[param_group_id]["betas"],
            "eps": optimizer_z3.optimizer.param_groups[param_group_id]["eps"],
            "weight_decay": optimizer_z3.optimizer.param_groups[param_group_id]["weight_decay"],
            "bias_correction": optimizer_z3.optimizer.param_groups[param_group_id]["bias_correction"],
        }

        group_infos.append(group_info)

    optimizer_z3.parent_conn.send({
        "type": "step",
        "now_state": now_state,
        "micro_step": optimizer_z3.micro_step,
        "group_infos": group_infos
    })


def wait_last_update_and_copy(optimizer_z3, timer_names):

    if not hasattr(optimizer_z3, 'parent_conn'):
        return

    if optimizer_z3.micro_step + 1 > optimizer_z3.full_warm_up_rounds and optimizer_z3.first_update_round_after_warmup:
        optimizer_z3.first_update_round_after_warmup = False
        return

    msg = optimizer_z3.parent_conn.recv()
    assert msg["type"] == "done", "Optimizer process did not finish stepping correctly."

    for sub_group_id, group in enumerate(optimizer_z3.fp16_groups):
        if optimizer_z3.fp16_partitioned_groups_flat[sub_group_id] is not None:
            optimizer_z3.fp16_partitioned_groups_flat[sub_group_id].data.copy_(
                optimizer_z3.fp32_partitioned_groups_flat[sub_group_id].stale_param.data)

            #unflatten fp16 parameter subgroup
            optimizer_z3._unflatten_partitioned_parameters(sub_group_id)
        else:
            optimizer_z3._partitioned_params_swap_out(sub_group_id)

    optimizer_z3._post_step(timer_names)

    # warn user about caching allocator flushes
    memory_stats = get_accelerator().memory_stats()
    alloc_retries = memory_stats.get("num_alloc_retries")
    if alloc_retries is None:
        alloc_retries = 0
    if alloc_retries > optimizer_z3.n_caching_allocator_flushes:
        if dist.get_rank() == 0:
            logger.warning(
                "%d pytorch allocator cache flushes since last step. this happens "
                "when there is high memory pressure and is detrimental to "
                "performance. if this is happening frequently consider adjusting "
                "settings to reduce memory consumption. If you are unable to "
                "make the cache flushes go away consider adding "
                "get_accelerator().empty_cache() calls in your training loop to ensure "
                "that all ranks flush their caches at the same time",
                alloc_retries - optimizer_z3.n_caching_allocator_flushes)
        optimizer_z3.n_caching_allocator_flushes = alloc_retries


@instrument_w_nvtx
def step(optimizer_z3, closure=None):
    """
        Not supporting closure.
    """
    optimizer_z3._pre_step()
    optimizer_z3._partition_all_parameters()

    #checks for overflow, adjust the loss scale accordingly
    if optimizer_z3._overflow_check_and_loss_scale_update():
        if optimizer_z3.swap_optimizer:
            optimizer_z3.optimizer_swapper.log_timers()
        return

    norm_groups = optimizer_z3._get_norm_groups()
    scaled_global_grad_norm = torch.linalg.vector_norm(torch.stack(norm_groups))

    # Stash unscaled gradient norm
    optimizer_z3._global_grad_norm = scaled_global_grad_norm / optimizer_z3.loss_scale

    if optimizer_z3.micro_step < optimizer_z3.full_warm_up_rounds:
        optimizer_z3.zenflow_cpu_optimizer_overlap_step(optimizer_z3.get_overlap_step_state(), scaled_global_grad_norm)

    timer_names = set()

    timer_names.add(OPTIMIZER_STEP_TIMER)

    optimizer_z3.wait_last_update_and_copy(timer_names)

    if optimizer_z3.micro_step >= optimizer_z3.full_warm_up_rounds:
        optimizer_z3.zenflow_cpu_optimizer_overlap_step(optimizer_z3.get_overlap_step_state(), scaled_global_grad_norm)

    return
