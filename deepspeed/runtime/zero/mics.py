# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import List

import deepspeed
import torch
from deepspeed import comm as dist
from deepspeed.runtime.zero.mics_utils import (MiCS_CommGroups, create_mics_comm_groups, scale_tensors)
from deepspeed.runtime.zero.parameter_offload import (DeepSpeedZeRoOffload, is_zero_param)
from deepspeed.runtime.zero.partition_parameters import Init, AllGatherCoalescedHandle, ZeroParamStatus
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.utils import instrument_w_nvtx, log_dist
from deepspeed.accelerator import get_accelerator
from torch import Tensor
from torch.nn import Parameter


def has_hierarchical_all_gather_groups(comm_groups: MiCS_CommGroups):
    result = False
    if comm_groups.param_intra_node_group is not None and comm_groups.param_inter_node_shard_group is not None:
        result = True
    return result


class MiCS_AllGatherCoalescedHandle(AllGatherCoalescedHandle):
    """ This handle assumes that no need to
    copy data out from a contiguous tensor
    """

    def __init__(self, allgather_handle, params: List[Parameter], partitions: List[Tensor], world_size: int) -> None:
        super().__init__(allgather_handle, params, partitions, world_size)

    def wait(self) -> None:
        """
        """
        # let the current stream to op
        instrument_w_nvtx(self.allgather_handle.wait)()
        if self.complete:
            return

        for _, param in enumerate(self.params):
            assert param.ds_status == ZeroParamStatus.INFLIGHT, f"expected param {param.ds_summary()} to be inflight"
            param.ds_status = ZeroParamStatus.AVAILABLE

        self.complete = True


class MiCS_Init(Init):

    def __init__(self,
                 module=None,
                 data_parallel_group=None,
                 mem_efficient_linear=True,
                 remote_device=None,
                 pin_memory=False,
                 config_dict_or_path=None,
                 config=None,
                 enabled=True,
                 dtype=None,
                 mpu=None):
        """A context manager to partition the model parameters during the model
        construction with MiCS partition strategy. Model states are partitioned
        to the number of devices specified via ``mics_shard_size`` field in the
        deepspeed config json file. The context manager also introduces
        hierarchical communication method to reduce the cost of inter-node
        communications, which can be enabled with
        ``mics_hierarchical_params_gather`` field in deepspeed config.

        Args:
            module (``torch.nn.Module``, optional): If provided, partition the model as
                if it was constructed in the context.
            data_parallel_group (``deepspeed.comm`` process group, optional):
                The group of processes to partition among. Defaults to all processes.
            mem_efficient_linear (bool, optional): Replace
                torch.nn.functional.linear with an implementation that allows
                DeepSpeed to partition parameters. Defaults to ``True``.
            remote_device (string, optional): The initial device to store model
                weights e.g., ``cpu``, ``nvme``. Passing ``"cpu"`` will create the model in CPU
                memory. The model may still be moved to GPU based on the
                offload settings for training. Defaults to param offload device if a config is
                defined, otherwise GPU.
            pin_memory (bool, optional): Potentially increase performance by
                using pinned memory for model weights. ``remote_device`` must be
                ``"cpu"``. Defaults to pin_memory value in config, otherwise ``False``.
            config_dict_or_path (dict or ``json file``, optional): If provided, provides configuration
                for swapping fp16 params to NVMe.
            config (dict or ``json file``, optional): Deprecated, use config_dict_or_path instead.
            enabled (bool, optional): If ``False``, this context has no
                effect. Defaults to ``True``.
            dtype (``dtype``, optional): Can be used to change the data type of the parameters.
                Supported options are ``torch.half`` and ``torch.float``. Defaults to ``None``
            mpu (``object``, optional): A model parallelism unit object that implements get_{model,data}_parallel_{rank,group,world_size}.

        This context follows the same logic as ``deepspeed.zero.Init()``, but
        with the modification for partition size of each parameter.

        Examples
        --------

        #. Allocate a model and partition it among all processes:

            .. code-block:: python
                # the config_dict_or_path is required to let the context manager know
                # how partition the parameters.
                # The configuration has to include the field ``mics_shard_size``
                with deepspeed.zero.MiCS_Init(config_dict_or_path=ds_config):
                    model = MyLargeModel()


        #. Allocate a model in pinned CPU memory and partition it among a subgroup of processes:

            .. code-block:: python

                with deepspeed.zero.MiCS_Init(data_parallel_group=mpu.get_data_parallel_group(),
                                              remote_device="cpu",
                                              pin_memory=True
                                              config_dict_or_path=ds_config):
                    model = MyLargeModel()


        #. Partition an already-allocated model in CPU memory:

            .. code-block:: python

                model = deepspeed.zero.MiCS_Init(module=model,
                                                 config_dict_or_path=ds_config)
        """

        assert config_dict_or_path is not None, "Must provide configuration for MiCS Initialization"
        _ds_config = deepspeed.runtime.config.DeepSpeedConfig(config_dict_or_path, mpu)
        if not dist.is_initialized():
            dist.init_distributed()
            assert dist.is_initialized(), "Parameters cannot be scattered without initializing deepspeed.comm"
        self.mics_comm_groups = create_mics_comm_groups(
            _ds_config.mics_shard_size,
            data_parallel_group,
            hierarchical_allgather=_ds_config.mics_hierarchial_params_gather,
            mpu=mpu)

        super().__init__(module, data_parallel_group, mem_efficient_linear, remote_device, pin_memory,
                         config_dict_or_path, config, enabled, dtype, mpu)

    def _convert_to_deepspeed_param(self, param):
        super()._convert_to_deepspeed_param(param)
        # attach communication groups to every param
        param.comm = self.mics_comm_groups

        # record existing all_gather_coalesced implementation
        # so that we can fallback later
        old_all_gather_coalesced = param.all_gather_coalesced

        def _param_all_gather_coalesced(params, param_buffers=None, **kwargs):
            """"""
            mics_comm_groups: MiCS_CommGroups = params[0].comm
            hierarchical_all_gather = has_hierarchical_all_gather_groups(mics_comm_groups)
            if dist.has_coalescing_manager() and hierarchical_all_gather:
                return self._hierarchical_all_gather_params(params, param_buffers)
            elif dist.has_coalescing_manager():
                return self._flat_all_gather_with_coalescing_manager(params, param_buffers)
            else:
                return old_all_gather_coalesced(params, **kwargs)

        # change the all_gather_coalesced method
        param.all_gather_coalesced = _param_all_gather_coalesced

    def _pre_all_gather(self, params, params_buffers=None):
        # fetches from nvme if the partition is not available and in nvme
        self._ensure_availability_of_partitioned_params(params)

        for param in params:
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(param.ds_summary())
            param.ds_status = ZeroParamStatus.INFLIGHT

        # ensure that each rank has params in same order. the allgather
        # is done by flattening the parameter list into a single tensor that
        # can be allgathered in a single call - this means that if each rank
        # gives a list of the same parameters in a different order we will
        # silently get incorrect parameter values, and have very difficult
        # to debug correctness issues.
        params = sorted(params, key=lambda p: p.ds_id)
        return params, params_buffers

    def _flat_all_gather_with_coalescing_manager(self, params, params_buffers=None):
        """"""
        # must have to change the status of the param
        # and ensure they are on the device
        params, params_buffers = self._pre_all_gather(params, params_buffers)

        mics_comm_groups: MiCS_CommGroups = params[0].comm
        param_shard_size = mics_comm_groups.param_shard_size

        output_tensors = []
        input_tensors = []
        for i, p in enumerate(params):
            t_size = p.ds_tensor.ds_numel * param_shard_size
            if params_buffers is not None and params_buffers[i] is not None:
                assert params_buffers[i].numel(
                ) == t_size, f'params_to_gather_buffers[{i}] size {params_buffers[i].numel()} does not match with t_size {t_size}'
                flat_out = params_buffers[i]
            else:
                flat_out = torch.empty(t_size, dtype=p.dtype, device=self.local_device, requires_grad=False).view(-1)
            output_tensors.append(flat_out)
            _flat_input = p.ds_tensor.data.view(-1)
            input_tensors.append(_flat_input)

        all_gather_handle = dist.all_gather_coalesced(output_tensors,
                                                      input_tensors,
                                                      group=mics_comm_groups.param_shard_group,
                                                      async_op=True)

        for idx, param in enumerate(params):
            param.data = output_tensors[idx].narrow(0, 0, param.ds_numel).view(param.ds_shape).data

        return MiCS_AllGatherCoalescedHandle(allgather_handle=all_gather_handle,
                                             params=params,
                                             partitions=[],
                                             world_size=param_shard_size)

    def _hierarchical_all_gather_params(self, params, params_buffers=None):
        """"""
        params, params_buffers = self._pre_all_gather(params, params_buffers)

        mics_comm_groups: MiCS_CommGroups = params[0].comm
        local_rank = dist.get_rank(group=mics_comm_groups.param_intra_node_group)
        inter_node_comm_group = mics_comm_groups.param_inter_node_shard_group
        intra_node_comm_group = mics_comm_groups.param_intra_node_group
        param_shard_size = mics_comm_groups.param_shard_size

        inter_node_size = dist.get_world_size(group=inter_node_comm_group)
        intra_node_size = dist.get_world_size(group=intra_node_comm_group)
        param_tensors = []
        for i, p in enumerate(params):
            param_size = p.ds_tensor.ds_numel * param_shard_size
            if params_buffers is not None and params_buffers[i] is not None:
                assert params_buffers[i].numel(
                ) == param_size, f'param_buffers[{i}] size {params_buffers[i].numel()} does not match with param_size {param_size}'
                param_tensor = params_buffers[i]
            else:
                param_tensor = torch.empty(param_size, dtype=p.dtype, device=self.local_device,
                                           requires_grad=False).view(-1)
            param_tensors.append(param_tensor)

        # inter node all-gather
        inter_outputs = []
        inter_inputs = []
        for i, p in enumerate(params):
            inter_size = p.ds_tensor.ds_numel * inter_node_size
            _out = param_tensors[i].narrow(0, local_rank * inter_size, inter_size)
            inter_outputs.append(_out)
            inter_inputs.append(p.ds_tensor.data.view(-1).to(self.local_device))
        # sync enqueue
        dist.all_gather_coalesced(inter_outputs, inter_inputs, group=inter_node_comm_group, async_op=False)

        # intra node all-gather
        intra_outputs = []
        intra_inputs = []
        for i, p in enumerate(params):
            # partition param into multiple chunks for allgather
            # because inter-node all-gather outputs are in a continues memory
            # while in param memory, those inter-node data are placed in different
            # location.
            # each chunk is an intra-node output
            param_chunk = param_tensors[i].view(
                (inter_node_size, intra_node_size, p.ds_tensor.ds_numel)).narrow(1, local_rank, 1)
            param_chunk.copy_(inter_outputs[i].detach().clone().view(param_chunk.size()))
            output_chunks = torch.chunk(param_tensors[i], inter_node_size)
            for j, _out in enumerate(output_chunks):
                intra_chunk_size = intra_node_size * p.ds_tensor.ds_numel
                local_offset = local_rank * p.ds_tensor.ds_numel
                _in = param_tensors[i].narrow(0, j * intra_chunk_size + local_offset, p.ds_tensor.ds_numel)
                intra_outputs.append(_out)
                intra_inputs.append(_in)

        all_gather_handle = dist.all_gather_coalesced(intra_outputs,
                                                      intra_inputs,
                                                      group=intra_node_comm_group,
                                                      async_op=True)
        for i, param in enumerate(params):
            param.data = param_tensors[i].narrow(0, 0, param.ds_numel).view(param.ds_shape).data

        return MiCS_AllGatherCoalescedHandle(
            allgather_handle=all_gather_handle,
            params=params,
            partitions=[],
            world_size=param_shard_size,
        )

    def get_partition_dp_group(self, param):
        return param.comm.param_shard_group

    def get_partition_rank(self):
        return self.mics_comm_groups.param_shard_rank

    @property
    def num_partitions(self):
        return self.mics_comm_groups.param_shard_size


class MiCS_Offload(DeepSpeedZeRoOffload):
    """ Wrapper to change the behavior for parameter sharding
    """

    def _convert_to_zero_parameters(self, ds_config, module, mpu):
        """ overload the parent class function for convert the parameters

        """
        log_dist(f'Convert to zero parameters from MiCS Offload manager', ranks=[0])
        non_zero_params = [p for p in module.parameters() if not is_zero_param(p)]
        if non_zero_params:
            zero_params = [p for p in module.parameters() if is_zero_param(p)]
            if zero_params:
                zero_params[0].convert_to_zero_parameters(param_list=non_zero_params)
            else:
                group = None
                if mpu:
                    group = mpu.get_data_parallel_group()

                MiCS_Init(module=module,
                          data_parallel_group=group,
                          dtype=self.dtype,
                          config_dict_or_path=ds_config,
                          remote_device=self.offload_device,
                          pin_memory=self.offload_param_pin_memory,
                          mpu=mpu)


class MiCS_Optimizer(DeepSpeedZeroOptimizer_Stage3):
    """
    MiCS Optimizer
    """

    def __init__(self,
                 module,
                 init_optimizer,
                 timers,
                 ds_config,
                 static_loss_scale=1,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 contiguous_gradients=True,
                 reduce_bucket_size=500000000,
                 prefetch_bucket_size=50000000,
                 max_reuse_distance=1000000000,
                 max_live_parameters=1000000000,
                 param_persistence_threshold=100000,
                 model_persistence_threshold=sys.maxsize,
                 dp_process_group=None,
                 reduce_scatter=True,
                 overlap_comm=False,
                 offload_optimizer_config=None,
                 offload_param_config=None,
                 sub_group_size=1000000000000,
                 mpu=None,
                 clip_grad=0,
                 gradient_accumulation_dtype=torch.float16,
                 communication_data_type=torch.float16,
                 postscale_gradients=True,
                 gradient_predivide_factor=1,
                 gradient_accumulation_steps=1,
                 elastic_checkpoint=False,
                 aio_config=None):

        log_dist("Init MiCS optimizer", ranks=[0])
        super().__init__(module, init_optimizer, timers, ds_config, static_loss_scale, dynamic_loss_scale,
                         dynamic_loss_args, verbose, contiguous_gradients, reduce_bucket_size, prefetch_bucket_size,
                         max_reuse_distance, max_live_parameters, param_persistence_threshold,
                         model_persistence_threshold, dp_process_group, reduce_scatter, overlap_comm,
                         offload_optimizer_config, offload_param_config, sub_group_size, mpu, clip_grad,
                         gradient_accumulation_dtype, communication_data_type, postscale_gradients,
                         gradient_predivide_factor, gradient_accumulation_steps, elastic_checkpoint, aio_config)
        first_param = next(module.parameters())
        # overload the dp_process_group and partition_count
        assert hasattr(first_param, "comm"), " ".join([
            "Sharded parameters don't have the MiCS_CommGroups attached.",
            "Might due to the use of deepspeed.zero.Init context for initializing the weights.",
            "To use MiCS sharding, please use deepspeed.zero.MiCS_Init instead for initializing parameter."
        ])
        self.dp_process_group = first_param.comm.param_shard_group
        self.partition_count = first_param.comm.param_shard_size

    def initialize_ds_offload(
        self,
        *args,
        **kwargs,
    ):
        return MiCS_Offload(*args, **kwargs)

    def partition_grads(self, params_to_release: List[Parameter], grad_partitions: List[Tensor]) -> None:
        grad_buffers = super().partition_grads(params_to_release, grad_partitions)
        # perform all-reduce among replication groups
        # the function will perform accumulation boundary check
        self.allreduce_mics_shard_grads(params_to_release, grad_buffers)

    @instrument_w_nvtx
    def allreduce_mics_shard_grads(self, params, partitioned_grads_buffers: List[Tensor]):
        """
        """
        # TODO: improve the condition check
        if not self.is_gradient_accumulation_boundary or \
            len(partitioned_grads_buffers) == 0:
            return

        mics_comm_groups: MiCS_CommGroups = params[0].comm
        param_repli_group = mics_comm_groups.param_repli_group
        param_repli_size = mics_comm_groups.param_repli_size

        if param_repli_size is None or param_repli_size <= 1:
            return
        if not get_accelerator().on_accelerator(partitioned_grads_buffers[0]):
            raise RuntimeError("Local sharding has no support for CPU offloading")

        if dist.has_all_reduce_coalesced():
            scale_tensors(partitioned_grads_buffers, param_repli_size)
            dist.all_reduce_coalesced(tensors=partitioned_grads_buffers, group=param_repli_group)
        else:
            # manually coalescing all-reduce
            aggregated_buffer: Tensor = torch.cat(partitioned_grads_buffers)
            aggregated_buffer.div_(param_repli_size)
            dist.all_reduce(aggregated_buffer, group=param_repli_group)
            offset = 0
            for grad_buff in partitioned_grads_buffers:
                grad_buff.view(-1).copy_(aggregated_buffer.narrow(0, offset, grad_buff.numel()))
                offset += grad_buff.numel()

    def load_state_dict(self,
                        state_dict_list,
                        load_optimizer_states=True,
                        load_from_fp32_weights=False,
                        checkpoint_folder=None,
                        load_serial=None):
        r""" Loading the ZeRO-3/MiCS partitioned checkpoints
        Because the self.dp_process_group is replaced with the communicator for
        partition group we can call the load_state_dict logic from ZeRO-3.
        """
        super().load_state_dict(state_dict_list, load_optimizer_states, load_from_fp32_weights, checkpoint_folder)
