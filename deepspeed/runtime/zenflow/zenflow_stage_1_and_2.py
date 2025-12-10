# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed import comm as dist

from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.zenflow.zenflow_utils import start_optimizer_process
from deepspeed.runtime.utils import (see_memory_usage)
from deepspeed.ops.adam import ZenFlowSelectiveAdamW

from deepspeed.moe.utils import is_moe_param

from deepspeed.accelerator import get_accelerator

from deepspeed.runtime.utils import all_gather_dp_groups

# Toggle this to true to enable correctness test
# with gradient partitioning and without
pg_correctness_test = False

OPTIMIZER_ALLGATHER_TIMER = 'optimizer_allgather'
OPTIMIZER_GRADIENTS_TIMER = 'optimizer_gradients'
OPTIMIZER_STEP_TIMER = 'optimizer_step'
OPTIMIZER_TRANSMIT_TIMER = 'optimizer_transmit_time'
OPTIMIZER_CALC_TIMER = 'optimizer_calc_time'
OPTIMIZER_RECV_PARAMS_TIMER = 'optimizer_receive_params_time'
OPTIMIZER_UPDATE_MODEL_TIMER = 'optimizer_update_model_time'
OPTIMIZER_TIMERS = [
    OPTIMIZER_ALLGATHER_TIMER, OPTIMIZER_GRADIENTS_TIMER, OPTIMIZER_STEP_TIMER, OPTIMIZER_TRANSMIT_TIMER,
    OPTIMIZER_CALC_TIMER, OPTIMIZER_RECV_PARAMS_TIMER, OPTIMIZER_UPDATE_MODEL_TIMER
]
INITIAL_MICRO_STEP_ID = -1

SELECTIVE_OPTIMIZER_UPDATE_TIMER = 'selective_optimizer_update'
SELECTIVE_OPTIMIZER_PROCESS_TIMER = 'selective_optimizer_process'
SELECTIVE_OPTIMIZER_STEP_TIMER = 'selective_optimizer_step'
SELECTIVE_OPTIMIZER_SYNC_TIMER = 'selective_optimizer_sync'
SELECTIVE_OPTIMIZER_TIMERS = [
    SELECTIVE_OPTIMIZER_UPDATE_TIMER, SELECTIVE_OPTIMIZER_PROCESS_TIMER, SELECTIVE_OPTIMIZER_STEP_TIMER,
    SELECTIVE_OPTIMIZER_SYNC_TIMER
]


class ZenFlowZeroOptimizer(DeepSpeedZeroOptimizer):

    def __init__(
        self,
        init_optimizer,
        param_names,
        timers,
        optimizer_params,
        **kwargs,
    ):

        super().__init__(init_optimizer, param_names, timers, optimizer_params, **kwargs)

        zenflow_config = kwargs.get("zenflow_config", None)

        self.micro_step = -1
        self.full_warm_up_rounds = zenflow_config.full_warm_up_rounds
        self.offload_selective_optimizer = zenflow_config.offload
        self.pt_reserved_cores_perc = zenflow_config.pt_reserved_cores_perc
        self.start_optimizer_process = lambda: start_optimizer_process(self)
        self.zf_stage3 = False

        if self.offload_selective_optimizer:
            assert kwargs.get("overlap_comm", False), "offload selective optimizer should be used with overlap_comm"

        self._configure_zenflow(zenflow_config)


        self.selective_optimizer = ZenFlowSelectiveAdamW([{"params": group} for group in self.bit16_groups], \
                                                        offload=zenflow_config.offload,
                                                        bucket_size=self.allgather_bucket_size,
                                                        **optimizer_params)
        self.num_total_param = sum(sum(1 for param in group if len(param.shape) != 1) for group in self.bit16_groups)

    @classmethod
    def create(cls, zenflow_config):
        if zenflow_config.overlap_step:
            return ZenFlowZeroOptimizerParallel
        else:
            return ZenFlowZeroOptimizerSequential

    def _configure_zenflow(self, zenflow_config):
        """
        Configure ZenFlow optimizer
        """
        if not self.cpu_offload:
            raise ValueError("Zenflow must be used with cpu offload")

        self.select_strategy = zenflow_config.select_strategy
        if self.select_strategy == 'auto':
            self.select_strategy = "epoch"
            if isinstance(zenflow_config.select_interval, int):
                raise Warning(
                    "If use auto select strategy, select_interval will be set to 1 and select_strategy will be set to epoch, thus select_interval would be overwritten."
                )
            self.select_interval = 1
        else:
            if isinstance(zenflow_config.select_interval, str):
                raise ValueError("If don't use auto select strategy, select_interval must be a number.")
            self.select_interval = int(zenflow_config.select_interval)

        if isinstance(zenflow_config.update_interval, str):
            self.auto_update = True
            self.update_interval = 0
        else:
            self.auto_update = False
            self.update_interval = int(zenflow_config.update_interval)

        if self.select_strategy == 'epoch':
            if zenflow_config.steps_per_epoch is not None:
                self.select_interval = self.select_interval * zenflow_config.steps_per_epoch
            else:
                self.select_interval = 0

        if not self.auto_update and self.select_interval != 0 and self.select_interval < self.update_interval:
            raise ValueError("Select interval must be greater or equal to update interval")

        self.topk_ratio = zenflow_config.topk_ratio

        self.param_id_index_buffer_offset = {}
        self.param_id_grad_buffer_offset = {}

        if self.auto_update:
            self.param_id_sum_buffer_offset = {}
            self.auto_ratio = zenflow_config.auto_ratio
            self.zenflow_need_update = [False, False]
            self.zenflow_state = 0
            self.num_need_update = 0

    def is_zenflow_select_boundary(self):
        return self.zenflow and (self.micro_step - self.full_warm_up_rounds) >= 0 and (
            (self.micro_step - self.full_warm_up_rounds) == 0 or
            (self.select_interval != 0 and self.micro_step % self.select_interval == 0))

    def sync_fp32_param_from_gpu(self):
        if self.micro_step == 0:
            return

        for i, group in enumerate(self.bit16_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])

            bit16_partitions = self.parallel_partitioned_bit16_groups[i]
            fp32_partition = self.single_partition_of_fp32_groups[i]

            with torch.no_grad():
                fp32_partition.copy_(bit16_partitions[partition_id].to(dtype=fp32_partition.dtype,
                                                                       device=fp32_partition.device))

    def update_selected_channels(self, tensor, total_size, communication_data_type):
        curr_size = 0
        curr_index_buffer_size = 0
        rank_and_offsets = []
        prev_id, prev_process_group = -1, None

        process_group = self.dp_process_group
        rank = dist.get_rank(process_group)

        self.index_buffer = torch.empty(total_size, dtype=torch.int32, device=get_accelerator().current_device_name())

        bucket = self.ipg_buckets[communication_data_type]
        for i, param_idx_in_group, param_id in bucket.params:
            param = self.bit16_groups[i][param_idx_in_group]

            if len(param.shape) == 1:
                continue

            if not hasattr(param, 'selected_indices'):
                param.selected_indices = None

            partition_ids = self.param_to_partition_ids[i][param_id]

            # Get all partition ids + their offsets
            partition_ids_w_offsets = []
            for partition_id in partition_ids:
                offset = self.grad_start_offset[i][partition_id][param_id]
                partition_ids_w_offsets.append((partition_id, offset))
            partition_ids_w_offsets.sort(key=lambda t: t[1])

            # Calculate rank and offsets for grad slices
            for idx in range(len(partition_ids_w_offsets)):
                partition_id, offset = partition_ids_w_offsets[idx]

                if idx == len(partition_ids_w_offsets) - 1:
                    numel = param.numel() - offset
                else:
                    numel = partition_ids_w_offsets[idx + 1][1] - offset

                num_row, num_col = param.shape if len(param.shape) == 2 else (1, param.shape[0])
                start_column = 0 if not offset else int((offset - 1) / num_row) + 1
                end_column = int((offset + numel) / num_row)
                num_select = int(self.topk_ratio * (end_column - start_column))

                if partition_id == rank:

                    start_idx = int(curr_size + start_column * num_row - offset)
                    num_elements = (end_column - start_column) * num_row
                    sum_per_column = tensor.narrow(0, start_idx, num_elements)
                    sum_per_column = sum_per_column.view(end_column - start_column, num_row)
                    sum_array = sum_per_column.abs().sum(dim=1)

                    _, top_indices = torch.topk(sum_array, num_select)
                    top_indices += start_column
                    self.index_buffer.narrow(0, curr_index_buffer_size, num_select).copy_(top_indices)

                if partition_id == prev_id and process_group == prev_process_group:
                    prev_pid, prev_size, prev_numel = rank_and_offsets[-1]
                    rank_and_offsets[-1] = (prev_pid, prev_size, prev_numel + num_select)
                else:
                    rank_and_offsets.append((partition_id, curr_index_buffer_size, num_select))

                if param_id not in self.param_id_index_buffer_offset:
                    self.param_id_index_buffer_offset[param_id] = []
                self.param_id_index_buffer_offset[param_id].append((curr_index_buffer_size, num_select))

                curr_size += numel
                curr_index_buffer_size += num_select

        for src_rank, offset, num_select in rank_and_offsets:
            index_slice = self.index_buffer.narrow(0, offset, num_select)
            dist.broadcast(index_slice, src=src_rank, group=process_group)

        for i, param_idx_in_group, param_id in bucket.params:
            param = self.bit16_groups[i][param_idx_in_group]

            if len(param.shape) == 1:
                continue

            param.selected_indices = None
            param.partition_selected_indices = []

            for offset, num_select in self.param_id_index_buffer_offset[param_id]:
                selected = self.index_buffer.narrow(0, offset, num_select).clone().sort()[0]
                if param.selected_indices is None:
                    param.selected_indices = selected
                else:
                    param.selected_indices = torch.cat([param.selected_indices, selected])
                param.partition_selected_indices.append(selected)

            self.param_id_index_buffer_offset[param_id] = []

            num_row, num_col = param.shape if len(param.shape) == 2 else (1, param.shape[0])
            param.selected_indices.sort()
            param.selected_shape = (param.selected_indices.shape[0],
                                    num_row) if num_row != 1 else (param.selected_indices.shape[0], )

        self.index_buffer = None

    def _process_selected_fp32_groups_grad(self, tensor, total_size, communication_data_type):
        """
        Process gradients for selected columns in FP32 groups

        Args:
            param: The parameter to process
            param_id: ID of the parameter
        """

        curr_size = 0
        curr_grad_buffer_size = 0
        curr_sum_buffer_size = 0
        rank_and_offsets = []
        prev_id, prev_process_group = -1, None

        process_group = self.dp_process_group
        rank = dist.get_rank(process_group)

        self.grad_buffer = torch.empty(total_size, dtype=self.dtype, device=get_accelerator().current_device_name())

        bucket = self.ipg_buckets[communication_data_type]
        if self.auto_update:
            self.sum_buffer = torch.empty(len(bucket.params) + dist.get_world_size(group=process_group),
                                          dtype=torch.bfloat16,
                                          device=get_accelerator().current_device_name())

        group_to_paramlist = {}

        for i, param_idx_in_group, param_id in bucket.params:
            param = self.bit16_groups[i][param_idx_in_group]

            if not hasattr(param, 'selected_indices'):
                param.selected_indices = None

            partition_ids = self.param_to_partition_ids[i][param_id]

            # Get all partition ids + their offsets
            partition_ids_w_offsets = []
            for partition_id in partition_ids:
                offset = self.grad_start_offset[i][partition_id][param_id]
                partition_ids_w_offsets.append((partition_id, offset))
            partition_ids_w_offsets.sort(key=lambda t: t[1])

            # Calculate rank and offsets for grad slices
            for idx in range(len(partition_ids_w_offsets)):
                partition_id, offset = partition_ids_w_offsets[idx]

                if idx == len(partition_ids_w_offsets) - 1:
                    numel = param.numel() - offset
                else:
                    numel = partition_ids_w_offsets[idx + 1][1] - offset

                num_row, num_col = param.shape if len(param.shape) == 2 else (1, param.shape[0])
                start_column = 0 if not offset else int((offset - 1) / num_row) + 1
                end_column = int((offset + numel) / num_row)
                num_select = int(self.topk_ratio * (end_column - start_column)) if len(param.shape) == 2 else numel
                grad_size = num_select * num_row

                if partition_id == rank:
                    selected_grad = param.grad[
                        param.partition_selected_indices[idx], :] if num_row != 1 else param.grad[offset:offset +
                                                                                                  numel]
                    self.grad_buffer.narrow(0, curr_grad_buffer_size, grad_size).copy_(selected_grad.view(-1))

                    if self.auto_update:
                        self.sum_buffer[curr_sum_buffer_size] = tensor.narrow(0, int(curr_size),
                                                                              int(numel)).abs().sum()

                if partition_id == prev_id and process_group == prev_process_group:
                    if self.auto_update:
                        prev_pid, prev_size, prev_numel, prev_sum_size, prev_sum_num = rank_and_offsets[-1]
                        rank_and_offsets[-1] = (prev_pid, prev_size, prev_numel + grad_size, prev_sum_size,
                                                prev_sum_num + 1)
                    else:
                        prev_pid, prev_size, prev_numel = rank_and_offsets[-1]
                        rank_and_offsets[-1] = (prev_pid, prev_size, prev_numel + grad_size)
                else:
                    if self.auto_update:
                        rank_and_offsets.append(
                            (partition_id, curr_grad_buffer_size, grad_size, curr_sum_buffer_size, 1))
                    else:
                        rank_and_offsets.append((partition_id, curr_grad_buffer_size, grad_size))

                if param_id not in self.param_id_grad_buffer_offset:
                    self.param_id_grad_buffer_offset[param_id] = []
                if self.auto_update and param_id not in self.param_id_sum_buffer_offset:
                    self.param_id_sum_buffer_offset[param_id] = []
                self.param_id_grad_buffer_offset[param_id].append((curr_grad_buffer_size, grad_size))
                if self.auto_update:
                    self.param_id_sum_buffer_offset[param_id].append(curr_sum_buffer_size)

                curr_size += numel
                curr_grad_buffer_size += grad_size
                curr_sum_buffer_size += 1

        for item in rank_and_offsets:
            if self.auto_update:
                src_rank, offset, grad_size, sum_offset, sum_num = item
            else:
                src_rank, offset, grad_size = item

            grad_slice = self.grad_buffer.narrow(0, offset, grad_size)
            dist.broadcast(grad_slice, src=src_rank, group=process_group)

            if self.auto_update:
                sum_slice = self.sum_buffer.narrow(0, sum_offset, sum_num)
                dist.broadcast(sum_slice, src=src_rank, group=process_group)

        for i, param_idx_in_group, param_id in bucket.params:
            param = self.bit16_groups[i][param_idx_in_group]

            selected_grad = None
            for offset, grad_size in self.param_id_grad_buffer_offset[param_id]:
                selected_grad_buffer = self.grad_buffer.narrow(0, offset, grad_size).clone().detach()
                if selected_grad is None:
                    selected_grad = selected_grad_buffer
                else:
                    selected_grad = torch.cat([selected_grad, selected_grad_buffer])
            param.selected_grad = selected_grad.view(param.selected_shape).t() if len(
                param.shape) != 1 else selected_grad

            if self.offload_selective_optimizer and not hasattr(param, 'exp_avg_cpu_data'):
                buffer = torch.zeros(param.selected_grad.numel(), dtype=param.dtype, device=self.device)
                param.exp_avg_cpu_data = get_accelerator().pin_memory(
                    buffer) if self.cpu_offload_pin_memory else buffer
                param.exp_avg_sq_cpu_data = get_accelerator().pin_memory(
                    buffer.clone()) if self.cpu_offload_pin_memory else buffer.clone()

            param_list = group_to_paramlist.setdefault(i, [])
            param_list.append(param)

            self.param_id_grad_buffer_offset[param_id] = []

            if self.auto_update:
                grad_total_sum = 0
                num_row, num_col = param.shape if len(param.shape) == 2 else (1, param.shape[0])
                if num_row == 1:
                    continue

                for offset in self.param_id_sum_buffer_offset[param_id]:
                    grad_total_sum += self.sum_buffer.narrow(0, offset, 1)

                grad_critic_sum = param.selected_grad.abs().sum()

                if not hasattr(param, 'non_critic_sum'):
                    param.non_critic_sum = 0
                if not hasattr(param, 'avg_critic_sum'):
                    param.avg_critic_sum = 0

                param.avg_critic_sum = (param.avg_critic_sum * (self.update_interval - 1) +
                                        grad_critic_sum) / self.update_interval / (self.topk_ratio * 10)
                param.non_critic_sum += (grad_total_sum - grad_critic_sum) / ((1 - self.topk_ratio) * 10)
                if param.non_critic_sum >= param.avg_critic_sum:
                    self.num_need_update += 1

                if self.num_need_update >= int(self.auto_ratio * self.num_total_param):
                    self.zenflow_need_update[self.zenflow_state] = True

                self.param_id_sum_buffer_offset[param_id] = []

        if not self.is_gradient_accumulation_boundary:
            self.selective_optimizer.group_step(group_to_paramlist)
        else:
            self.selective_optimizer.temp_copy_param(group_to_paramlist)

        self.grad_buffer = None
        if self.auto_update:
            self.sum_buffer = None

    def average_tensor(self, tensor: torch.Tensor, communication_data_type: torch.dtype):
        if self.overlap_comm:
            stream = self.reduction_stream
            if not get_accelerator().resolves_data_dependency():
                stream.wait_stream(get_accelerator().current_stream())
                get_accelerator().current_stream().wait_stream(stream)
        else:
            stream = get_accelerator().current_stream()

        with get_accelerator().stream(stream):
            if not self.reduce_scatter:
                self.gradient_reduction_w_predivide(tensor)
                return

            # Accumulate destination ranks and bucket offsets for each gradient slice.
            # Note: potential future optimization, record access pattern of parameters
            # in backward pass and partition gradients w.r.t. access pattern so that our
            # bucket is guaranteed to be contiguous w.r.t. ranks
            rank_and_offsets = []
            real_dp_process_group = []
            curr_size = 0
            prev_id, prev_process_group = -1, None

            curr_column_size = 0
            curr_selected_reduce_size = 0

            process_group = self.dp_process_group
            bucket = self.ipg_buckets[communication_data_type]
            for i, param_idx_in_group, param_id in bucket.params:
                param = self.bit16_groups[i][param_idx_in_group]

                process_group = self.dp_process_group

                if bucket.has_moe_params:
                    process_group = self.expert_dp_process_group[param.group_name] if is_moe_param(
                        param) else self.dp_process_group

                partition_ids = self.param_to_partition_ids[i][param_id]
                assert all([p_id < dist.get_world_size(group=process_group) for p_id in partition_ids
                            ]), f"world size {dist.get_world_size(group=process_group)} and p_ids: {partition_ids}"
                partition_size = self.partition_size[i]
                # Get all partition ids + their offsets
                partition_ids_w_offsets = []
                for partition_id in partition_ids:
                    offset = self.grad_start_offset[i][partition_id][param_id]
                    partition_ids_w_offsets.append((partition_id, offset))
                partition_ids_w_offsets.sort(key=lambda t: t[1])

                num_row, num_col = param.shape if len(param.shape) == 2 else (1, param.shape[0])
                curr_column_size += int(num_col * self.topk_ratio) if num_row != 1 else 0

                # Calculate rank and offsets for grad slices
                for idx in range(len(partition_ids_w_offsets)):
                    partition_id, offset = partition_ids_w_offsets[idx]

                    # Calculate numel for grad slice depending on partition location
                    if idx == len(partition_ids_w_offsets) - 1:
                        # Last partition_id uses its own offset
                        numel = param.numel() - offset
                    else:
                        # Set numel to next partition's offset
                        numel = partition_ids_w_offsets[idx + 1][1] - offset

                    # Merge bucket ranges if they belong to the same rank
                    if partition_id == prev_id and process_group == prev_process_group:
                        prev_pid, prev_size, prev_numel = rank_and_offsets[-1]
                        rank_and_offsets[-1] = (prev_pid, prev_size, prev_numel + numel)
                    else:
                        rank_and_offsets.append((partition_id, curr_size, numel))
                        real_dp_process_group.append(process_group)
                    curr_size += numel
                    curr_selected_reduce_size += int(numel * self.topk_ratio) if num_row != 1 else numel

                    prev_id, prev_process_group = partition_id, process_group

            tensor.div_(dist.get_world_size(group=self.dp_process_group) / float(self.sequence_parallel_size))

            buckets = {}
            for i, (dst, bucket_offset, numel) in enumerate(rank_and_offsets):
                grad_slice = tensor.narrow(0, int(bucket_offset), int(numel))
                bucket_key = real_dp_process_group[i] if self.use_multi_rank_bucket_allreduce else (
                    dst, real_dp_process_group[i])
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                if self.use_multi_rank_bucket_allreduce:
                    buckets[bucket_key].append((dst, grad_slice))
                else:
                    buckets[bucket_key].append(grad_slice)

            for bucket_key in buckets:
                if self.use_multi_rank_bucket_allreduce:
                    self.allreduce_and_scatter(buckets[bucket_key],
                                               communication_data_type,
                                               numel_per_bucket=self.reduce_bucket_size,
                                               divide=False,
                                               process_group=bucket_key)
                else:
                    dst, process_group = bucket_key
                    self.allreduce_no_retain(buckets[bucket_key],
                                             communication_data_type,
                                             numel_per_bucket=self.reduce_bucket_size,
                                             rank=dst,
                                             divide=False,
                                             process_group=process_group)

            if self.is_zenflow_select_boundary():
                self.timers(SELECTIVE_OPTIMIZER_UPDATE_TIMER).start()
                self.update_selected_channels(tensor, curr_column_size, communication_data_type)
                self.timers(SELECTIVE_OPTIMIZER_UPDATE_TIMER).stop()
            elif self.zenflow:
                self.timers(SELECTIVE_OPTIMIZER_UPDATE_TIMER).start()
                self.timers(SELECTIVE_OPTIMIZER_UPDATE_TIMER).stop()

            if self.zenflow and self.micro_step >= self.full_warm_up_rounds:
                self.timers(SELECTIVE_OPTIMIZER_PROCESS_TIMER).start()
                self._process_selected_fp32_groups_grad(tensor, curr_selected_reduce_size, communication_data_type)
                self.timers(SELECTIVE_OPTIMIZER_PROCESS_TIMER).stop()

    def backward(self, loss, retain_graph=False):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        self.backward_prologue()
        self.micro_step += 1

        if self.auto_update:
            self.zenflow_need_update[self.zenflow_state] = False
            self.num_need_update = 0
            if self.zenflow_need_update[self.zenflow_state ^ 1]:
                self.update_interval = 0
                for group in self.bit16_groups:
                    for p in group:
                        p.non_critic_sum = 0
            self.update_interval += 1

        if self.is_zenflow_select_boundary():
            self.timers(SELECTIVE_OPTIMIZER_SYNC_TIMER).start()
            self.sync_fp32_param_from_gpu()
            self.selective_optimizer.clear_selected_mv()
            self.timers(SELECTIVE_OPTIMIZER_SYNC_TIMER).stop()

        self.enter_backward()
        if self.custom_loss_scaler:
            scaled_loss = self.external_loss_scale * loss
            scaled_loss.backward(retain_graph=retain_graph)
        else:
            self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

        self.backward_epilogue()
        self.exit_backward()

    def log_selective_optimizer_timers(self):
        self.timers.log(SELECTIVE_OPTIMIZER_TIMERS)

    def _sync_selective_optimizer_lr(self):
        for group_selected, group in zip(self.selective_optimizer.param_groups, self.optimizer.param_groups):
            group_selected["lr"] = group["lr"]

    def _selective_optimizer_step(self, group_no):
        original_param_groups = self.selective_optimizer.param_groups
        self.selective_optimizer.param_groups = [original_param_groups[group_no]]
        self.selective_optimizer.step()
        self.selective_optimizer.param_groups = original_param_groups

    def selective_optimizer_step(self, closure=None):
        for i, group in enumerate(self.bit16_groups):
            self.timers(SELECTIVE_OPTIMIZER_STEP_TIMER).start()
            self._selective_optimizer_step(i)
            self.timers(SELECTIVE_OPTIMIZER_STEP_TIMER).stop()

        self.timers.log(SELECTIVE_OPTIMIZER_TIMERS)


class ZenFlowZeroOptimizerSequential(ZenFlowZeroOptimizer):

    def __init__(self, *args, **kwargs):
        super(ZenFlowZeroOptimizerSequential, self).__init__(*args, **kwargs)

    def zenflow_cpu_optimizer_step(self, group_no):
        self.optimizer.step(step_id=self.micro_step + 1)


class ZenFlowZeroOptimizerParallel(ZenFlowZeroOptimizer):

    def __init__(self, *args, **kwargs):
        super(ZenFlowZeroOptimizerParallel, self).__init__(*args, **kwargs)
        self.process_optimizer_established = False
        self.first_update_round_after_warmup = True

    def initialize_optimizer_states(self):

        for i, group in enumerate(self.bit16_groups):
            single_grad_partition = torch.zeros(int(self.partition_size[i]),
                                                dtype=self.single_partition_of_fp32_groups[i].dtype,
                                                device=self.device)
            self.single_partition_of_fp32_groups[i].grad = None
            buffer = get_accelerator().pin_memory(
                single_grad_partition) if self.cpu_offload_pin_memory else single_grad_partition
            self.single_partition_of_fp32_groups[i].overlap_grad = [buffer, buffer.clone()]

        # Initialize the optimizer states with the flattened fp32 partition.
        # State initialization for the Adagrad optimizer occurs at construction as opposed to other optimizers
        # which do lazy initialization of the state at the first call to step.
        if isinstance(self.optimizer, torch.optim.Adagrad):
            self.optimizer = torch.optim.Adagrad(self.single_partition_of_fp32_groups, **self.optimizer.defaults)

        if not self.cpu_offload:
            for group in self.single_partition_of_fp32_groups:
                group.grad = None  #class init

        return

    def _get_offload_gradient_dict(self):
        for param_group_index, _ in enumerate(self.optimizer.param_groups):
            self.offload_gradient_dict[param_group_index] = []
            for lp_param in self.params_in_partition[param_group_index]:
                param_id = self.get_param_id(lp_param)
                [_, _, dest_offset, num_elements] = self.grad_position[param_id]
                dest_tensor = self.single_partition_of_fp32_groups[param_group_index].overlap_grad[0].view(-1).narrow(
                    0, dest_offset, num_elements)
                self.offload_gradient_dict[param_group_index].append(dest_tensor)

    def get_overlap_step_state(self):
        if self.micro_step < self.full_warm_up_rounds:
            return self.micro_step & 1
        else:
            if not self.auto_update:
                return (self.micro_step // self.update_interval) & 1
            else:
                return self.zenflow_state

    def async_inplace_copy_grad_to_fp32_buffer_from_gpu(self, param):
        param_id = self.get_param_id(param)
        now_state = self.get_overlap_step_state()

        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        dest_tensor = self.single_partition_of_fp32_groups[i].overlap_grad[now_state].view(-1).narrow(
            0, dest_offset, num_elements)

        grad_accum = self.get_param_gradient_attribute(param)
        if grad_accum is None:
            src_tensor = grad_accum.view(-1).narrow(0, source_offset, num_elements)
        else:
            src_tensor = grad_accum.view(-1).narrow(0, source_offset, num_elements)
        if src_tensor.dtype != self.master_weights_and_grads_dtype:
            src_tensor = src_tensor.to(self.master_weights_and_grads_dtype)

        dest_tensor.copy_(src_tensor, non_blocking=True)
        param.grad = None  #offload only

    def wait_last_update_and_copy(self):

        if not hasattr(self, 'parent_conn'):
            return

        if self.micro_step + 1 > self.full_warm_up_rounds and self.first_update_round_after_warmup:
            self.first_update_round_after_warmup = False
            return

        self.timers(OPTIMIZER_RECV_PARAMS_TIMER).start()
        msg = self.parent_conn.recv()
        assert msg["type"] == "done", "Optimizer process did not finish stepping correctly."
        self.timers(OPTIMIZER_RECV_PARAMS_TIMER).stop()

        for i, group in enumerate(self.bit16_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            bit16_partitions = self.parallel_partitioned_bit16_groups[i]
            fp32_partition = self.optimizer.param_groups[i]['params'][0].stale_param.data
            self.timers(OPTIMIZER_TRANSMIT_TIMER).start()
            bit16_partitions[partition_id].data.copy_(fp32_partition.to(get_accelerator().current_device_name()).data)
            self.timers(OPTIMIZER_TRANSMIT_TIMER).stop()

        see_memory_usage('After optimizer before all-gather')
        if self.cpu_offload:
            self.reset_cpu_buffers()

        self.timers(OPTIMIZER_ALLGATHER_TIMER).start()
        # Gather the updated weights from everyone.
        # Then all partitions of the model parameters are updated and ready for next round forward.
        all_gather_dp_groups(groups_flat=self.bit16_groups_flat,
                             partitioned_param_groups=self.parallel_partitioned_bit16_groups,
                             dp_process_group=self.real_dp_process_group,
                             start_alignment_factor=self.nccl_start_alignment_factor,
                             allgather_bucket_size=self.allgather_bucket_size)
        self.timers(OPTIMIZER_ALLGATHER_TIMER).stop()

        self.timers(OPTIMIZER_UPDATE_MODEL_TIMER).start()
        # TODO: we probably don't need this? just to be safe
        for i in range(len(self.bit16_groups)):
            self._update_model_bit16_weights(i)
        self.timers(OPTIMIZER_UPDATE_MODEL_TIMER).stop()

        self.timers.log(OPTIMIZER_TIMERS)
        see_memory_usage('After zero_optimizer step')

    def zenflow_cpu_optimizer_step(self, now_state, scaled_global_grad_norm):

        if not self.process_optimizer_established:
            self.start_optimizer_process()

        group_infos = []
        for group_no, group in enumerate(self.bit16_groups):
            single_grad_partition = self.single_partition_of_fp32_groups[group_no].overlap_grad[now_state]
            self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

            group_info = {
                "lr": self.optimizer.param_groups[group_no]["lr"],
                "betas": self.optimizer.param_groups[group_no]["betas"],
                "eps": self.optimizer.param_groups[group_no]["eps"],
                "weight_decay": self.optimizer.param_groups[group_no]["weight_decay"],
                "bias_correction": self.optimizer.param_groups[group_no]["bias_correction"],
            }

            group_infos.append(group_info)

        self.parent_conn.send({
            "type": "step",
            "now_state": now_state,
            "micro_step": self.micro_step,
            "group_infos": group_infos
        })

    def step(self, closure=None):
        """
        Not supporting closure.
        """
        self.micro_step_id = INITIAL_MICRO_STEP_ID

        see_memory_usage(f"In step before checking overflow")

        # First compute norm for all group so we know if there is overflow
        if self.dtype == torch.float16:
            self.check_overflow()

        self._update_scale(self.overflow)
        if self.overflow:
            see_memory_usage('After overflow before clearing gradients')
            self.zero_grad(set_to_none=True)
            if self.cpu_offload:
                self.reset_cpu_buffers()
            else:
                self.averaged_gradients = {}

            see_memory_usage('After overflow after clearing gradients')

            for timer in OPTIMIZER_TIMERS:
                self.timers(timer).start()
                self.timers(timer).stop()
            return

        prev_scale = self.loss_scale
        # Step 1:- Calculate gradient norm using bit-16 grads
        see_memory_usage('Before norm calculation')
        scaled_global_grad_norm = self.scaled_global_norm()
        self._global_grad_norm = scaled_global_grad_norm / prev_scale
        see_memory_usage('After norm before optimizer')

        if self.micro_step < self.full_warm_up_rounds:
            self.zenflow_cpu_optimizer_step(self.get_overlap_step_state(), scaled_global_grad_norm)

        self.wait_last_update_and_copy()

        if self.micro_step >= self.full_warm_up_rounds:
            self.zenflow_cpu_optimizer_step(self.get_overlap_step_state(), scaled_global_grad_norm)

        return
