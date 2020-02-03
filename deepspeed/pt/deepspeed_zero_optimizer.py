'''
Copyright 2019 The Microsoft DeepSpeed Team

Copyright NVIDIA/apex
This file is adapted from FP16_Optimizer in NVIDIA/apex
'''

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed.distributed_c10d import _get_global_rank
import torch.distributed as dist
import math
from torch._six import inf

from deepspeed.pt.loss_scaler import DynamicLossScaler
from deepspeed.pt.deepspeed_utils import get_grad_norm, CheckOverflow


# create a flat tensor aligned at the alignment boundary
def flatten_dense_tensors_aligned(tensor_list, alignment, pg):
    num_elements = 0
    for tensor in tensor_list:
        num_elements = num_elements + tensor.numel()

    remaining = num_elements % alignment

    if remaining:
        elements_to_add = alignment - remaining
        pad_tensor = torch.zeros(elements_to_add,
                                 device=tensor_list[0].device,
                                 dtype=tensor_list[0].dtype)
        padded_tensor_list = tensor_list + [pad_tensor]

        num_elements = num_elements + elements_to_add
    else:
        padded_tensor_list = tensor_list

    if dist.get_rank(group=pg) == 0:
        print("Number of Elements is ", num_elements)

    return _flatten_dense_tensors(padded_tensor_list)


def _initialize_parameter_parallel_groups(parameter_parallel_size=None):
    data_parallel_size = int(dist.get_world_size())
    if parameter_parallel_size is None:
        parameter_parallel_size = int(data_parallel_size)
    print(data_parallel_size, parameter_parallel_size)
    assert data_parallel_size % parameter_parallel_size == 0, \
        'world size should be divisible by parameter parallel size'
    rank = dist.get_rank()
    my_group = None
    for i in range(dist.get_world_size() // parameter_parallel_size):
        ranks = range(i * parameter_parallel_size, (i + 1) * parameter_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            my_group = group
    return my_group


class FP16_DeepSpeedZeroOptimizer(object):
    """
    DeepSpeedZeroOptimizer designed to reduce the memory footprint
    required for training large deep learning models.

    For more details please see ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
    https://arxiv.org/abs/1910.02054

    For usage examples, refer to TODO: DeepSpeed V2 Tutorial

    """
    def __init__(self,
                 init_optimizer,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 dp_process_group=None,
                 partition_size=None,
                 mpu=None,
                 all_gather_partitions=True,
                 allgather_size=500000000,
                 clip_grad=0.0):

        if dp_process_group is not None and partition_size is not None:
            raise ValueError("Cannot specify both dp_process_group "
                             "and partition size")

        if dp_process_group is None:
            dp_process_group = _initialize_parameter_parallel_groups(partition_size)

        if not torch.cuda.is_available:
            raise SystemError("Cannot use fp16 without CUDA.")
        self.optimizer = init_optimizer

        self.verbose = verbose
        self.dp_process_group = dp_process_group

        # TODO: automatically turn off if #params > some_limit
        self.all_gather_partitions = all_gather_partitions
        self.allgather_size = allgather_size

        # param flattened by groups
        self.fp16_groups = []
        self.fp16_groups_flat = []

        #param partitioned by data parallel degree
        #this will contain a list of equal sized tensors
        #each of which will be updated by a different process
        self.parallel_partitioned_fp16_groups = []

        #a single 32-bit partition of the parallel partitioned parameters
        #that this process will update
        self.single_partition_of_fp32_groups = []

        #param partition info

        #These are the parameters in each group that will not be updated by this process directly
        self.params_not_in_partition = []

        #These are the parameters that will be updated by this process directly
        self.params_in_partition = []

        #Offset from the first paramter in the the self.params_in_partition
        #the parameter boundaries may not align with partition boundaries
        #so we need to keep track of the offset
        self.first_offset = []

        #number of elements per partition in each group
        self.partition_size = []

        partition_id = dist.get_rank(group=self.dp_process_group)

        # loop to deal with groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            # push this group to list before modify
            self.fp16_groups.append(param_group['params'])

            self.fp16_groups_flat.append(
                flatten_dense_tensors_aligned(
                    self.fp16_groups[i],
                    dist.get_world_size(group=self.dp_process_group),
                    self.dp_process_group))

            # set model fp16 weight to slices of flattened buffer
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i],
                                                      self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data

            #divide the flat weights into near equal paritition equal to the data parallel degree
            #each process will compute on a different part of the partition
            data_parallel_partitions = self.get_data_parallel_partitions(
                self.fp16_groups_flat[i])
            self.parallel_partitioned_fp16_groups.append(data_parallel_partitions)

            # a partition of the fp32 master weights that will be updated by this process
            self.single_partition_of_fp32_groups.append(
                self.parallel_partitioned_fp16_groups[i]
                [partition_id].clone().float().detach())

            # modify optimizer of have flat master weight
            self.single_partition_of_fp32_groups[
                i].requires_grad = True  # keep this in case internal optimizer uses it
            param_group['params'] = [self.single_partition_of_fp32_groups[i]]

            partition_size = len(self.fp16_groups_flat[i]) / dist.get_world_size(
                group=self.dp_process_group)
            params_in_partition, params_not_in_partition, first_offset = self.get_partition_info(self.fp16_groups[i], partition_size, partition_id)

            self.partition_size.append(partition_size)
            self.params_in_partition.append(params_in_partition)
            self.params_not_in_partition.append(params_not_in_partition)
            self.first_offset.append(first_offset)

        # we may have a way of fusing dynamic scale. Do not support for now
        if dynamic_loss_scale:
            if dynamic_loss_args is None:
                self.loss_scaler = DynamicLossScaler()
            else:
                self.loss_scaler = DynamicLossScaler(**dynamic_loss_args)

            self.dynamic_loss_scale = True

        else:
            self.dynamic_loss_scale = False
            self.cur_iter = 0

        self.mpu = mpu
        self.clip_grad = clip_grad

        self.overflow = False
        self.overflow_checker = CheckOverflow(self.fp16_groups, mpu=self.mpu)

    #views the tensor as multiple partitions and returns
    #those partitions
    def get_data_parallel_partitions(self, tensor):
        partitions = []

        dp = dist.get_world_size(group=self.dp_process_group)
        total_num_elements = tensor.numel()

        base_size = total_num_elements // dp
        remaining = total_num_elements % dp

        start = 0
        for id in range(dp):
            partition_size = base_size
            if id < remaining:
                partition_size = partition_size + 1
            partitions.append(tensor.narrow(0, start, partition_size))
            start = start + partition_size
        return partitions

    def get_partition_info(self, tensor_list, partition_size, partition_id):
        params_in_partition = []
        params_not_in_partition = []

        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        current_index = 0
        first_offset = 0

        for tensor in tensor_list:

            tensor_size = tensor.numel()

            if (current_index >= start_index and current_index < end_index):
                params_in_partition.append(tensor)

            elif start_index > current_index and start_index < (current_index +
                                                                tensor_size):
                params_in_partition.append(tensor)

                assert (first_offset==0), "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index

            else:
                params_not_in_partition.append(tensor)

            current_index = current_index + tensor_size

        return params_in_partition, params_not_in_partition, first_offset

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        # FP32 grad should never exist.
        # For speed, set model fp16 grad to None by default
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    #creates a flat fused tensor from the tensor list starting at the first_offset
    #in the first tensor of the list. If there are not enough elements in the tensor
    #list then the flat tensor will be padded with zeros
    def get_flat_partition(self, tensor_list, first_offset, partition_size, dtype=None):
        flat_tensor_list = []
        current_size = 0

        if dtype is None:
            dtype = tensor_list[0].dtype

        for i, tensor in enumerate(tensor_list):
            if tensor.grad is None:
                tensor.grad = torch.zeros(tensor.size(),
                                          dtype=tensor.dtype,
                                          device=tensor.device)
            tensor = tensor.grad
            num_elements = tensor.numel()
            tensor_offset = 0

            #we need to offset to get to the right element
            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset

            #we dont need all elements of the tensor
            if num_elements > (partition_size - current_size):
                num_elements = partition_size - current_size

            #we need a narrow view of the tensor based on the tensor offset and number of elements that
            #we need from this tensor
            if tensor_offset > 0 or num_elements < tensor.numel():
                flat_tensor_list.append(tensor.contiguous().view(-1).narrow(
                    0,
                    int(tensor_offset),
                    int(num_elements)).to(dtype))
            else:
                flat_tensor_list.append(tensor.to(dtype))

            current_size = current_size + num_elements

        #this means its the last partition and does not align with the dp boundary. We need to pad before flattening
        if current_size < partition_size:
            flat_tensor_list.append(
                torch.zeros(int(partition_size - current_size),
                            dtype=dtype,
                            device=tensor_list[0].device))

        return _flatten_dense_tensors(flat_tensor_list)

    def free_grad_in_param_list(self, param_list):
        for p in param_list:
            p.grad = None

    def see_memory_usage(self):
        print("Memory Allocated ",
              torch.cuda.memory_allocated() / (1024 * 1024 * 1024),
              "GigaBytes")
        print("Max Memory Allocated ",
              torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),
              "GigaBytes")
        print("Cache Allocated ",
              torch.cuda.memory_cached() / (1024 * 1024 * 1024),
              "GigaBytes")
        print("Max cache Allocated ",
              torch.cuda.max_memory_cached() / (1024 * 1024 * 1024),
              "GigaBytes")

    def print_first_n(self, caption, tensor, n=10):
        if tensor is not None:
            print(caption,
                  tensor.data.contiguous().view(-1).narrow(0,
                                                           0,
                                                           n).cpu().numpy())
        else:
            print(caption, None)

    def step(self, closure=None):
        """
        Not supporting closure.
        """
        # First compute norm for all group so we know if there is overflow

        self.overflow = self.overflow_checker.check()

        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            self.zero_grad()
            if self.verbose:
                print("[deepspeed] OVERFLOW! Skipping step. Attempted loss "
                      "scale: {}, reducing to {}".format(prev_scale,
                                                         self.loss_scale))
            return self.overflow

        norm_groups = []
        single_partition_grad_groups = []

        partition_id = dist.get_rank(group=self.dp_process_group)
        for i, group in enumerate(self.fp16_groups):

            norm_groups.append(get_grad_norm(group, mpu=self.mpu))

            #free gradients for all the parameters that are not updated by this process
            self.free_grad_in_param_list(self.params_not_in_partition[i])

            #create a flat gradients for parameters updated by this process
            single_grad_partition = self.get_flat_partition(
                self.params_in_partition[i],
                self.first_offset[i],
                self.partition_size[i],
                dtype=self.single_partition_of_fp32_groups[i].dtype)

            self.single_partition_of_fp32_groups[i].grad = single_grad_partition

            #release all the gradient since we have already created a necessary copy in dp_grad_partition
            self.free_grad_in_param_list(self.params_in_partition[i])

            single_partition_grad_groups.append(single_grad_partition)

        self.unscale_and_clip_grads(single_partition_grad_groups, norm_groups)

        self.optimizer.step()

        #get rid of the fp32 gradients. Not needed anymore
        for group in self.single_partition_of_fp32_groups:
            group.grad = None

        for i in range(len(norm_groups)):
            for fp16_partitions, fp32_partition in zip(self.parallel_partitioned_fp16_groups, self.single_partition_of_fp32_groups):
                fp16_partitions[partition_id].data.copy_(fp32_partition.data)

        dp_world_size = dist.get_world_size(group=self.dp_process_group)
        #gather the updated weights from everyone
        for _, partitioned_params in enumerate(self.parallel_partitioned_fp16_groups):
            if self.all_gather_partitions:
                # controllable memory-time tradeoff
                num_shards = max(
                    1,
                    partitioned_params[partition_id].numel() * dp_world_size //
                    self.allgather_size)
                shard_size = partitioned_params[partition_id].numel() // num_shards
                num_elements = shard_size
                for shard_id in range(num_shards + 1):
                    if shard_id == num_shards:
                        if shard_size * num_shards >= partitioned_params[
                                partition_id].numel():
                            break
                        else:
                            num_elements = partitioned_params[partition_id].numel(
                            ) - shard_id * shard_size
                    shard_list = []
                    for dp_id in range(dp_world_size):
                        curr_shard = partitioned_params[dp_id].narrow(
                            0,
                            shard_id * shard_size,
                            num_elements)
                        shard_list.append(curr_shard)
                    dist.all_gather(shard_list,
                                    shard_list[partition_id],
                                    group=self.dp_process_group)
            else:
                #this should require less memory but should be faster
                for src, partitioned_param in enumerate(partitioned_params):
                    global_src = _get_global_rank(self.dp_process_group, src)
                    dist.broadcast(partitioned_param,
                                   global_src,
                                   group=self.dp_process_group)

        # TODO: we probably don't need this? just to be safe
        for i in range(len(norm_groups)):
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i],
                                                      self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data

        return self.overflow

    def unscale_and_clip_grads(self, grad_groups_flat, norm_groups):
        total_norm = 0.0
        for norm in norm_groups:
            total_norm += norm**2.0
        total_norm = math.sqrt(total_norm)

        # compute combined scale factor for this group
        combined_scale = self.loss_scale
        if self.clip_grad > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.loss_scale

        for grad in grad_groups_flat:
            grad.data.mul_(1. / combined_scale)

    def backward(self, loss, retain_graph=False):
        self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

    def _update_scale(self, has_overflow=False):
        self.loss_scaler.update_scale(has_overflow)

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    # Promote loss scale so it can be retrieved or set via "fp16_optimizer_instance.loss_scale"
    def _get_loss_scale(self):
        return self.loss_scaler.loss_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)
    cur_scale = property(_get_loss_scale, _set_loss_scale)

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['loss_scaler'] = self.loss_scaler
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['overflow'] = self.overflow
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict[
            'single_partition_of_fp32_groups'] = self.single_partition_of_fp32_groups
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        # I think it should actually be ok to reload the optimizer before the model.
        self.loss_scaler = state_dict['loss_scaler']
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.overflow = state_dict['overflow']
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        for current, saved in zip(self.single_partition_of_fp32_groups, state_dict['single_partition_of_fp32_groups']):
            current.data.copy_(saved.data)
