# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed import comm as dist


def print_rank_0(message):
    if dist.get_rank() == 0:
        print(message)


class ContiguousMemoryAllocator(object):

    def __init__(self, size, dtype, device):
        self.buffer = torch.zeros(size, dtype=dtype, device=device)

        #address to contiguous size available
        self.contiguous_sizes = {}

        self.contiguous_sizes[0] = size

        #tensor id to its address
        self.tensor_addresses = {}

        #tensor address to its size
        self.tensor_sizes = {}

        #tensor address to ids
        self.tensor_ids = {}

        #id to tensors
        self.tensor_map = {}

        #id to params. Maps each tensor buffer to list of parameters that uses it
        self.id_to_params = {}

        self.total_size = size
        self.total_free = size
        self.largest_contiguous = size
        self.max_allocated = 0

        self.count = 0

    #create a tensor of size from the pre-allocated buffer
    #if not enough free space will fail
    #if not enough contiguous space, will defragment and allocate
    def allocate_tensor(self, size):
        free_before = self.total_free

        assert size <= self.total_free, "Not enough memory in buffer. Allocation failed"
        if self.largest_contiguous < size:
            print_rank_0("Needs defragmentation to allocate. Before Defragmentation:")
            self.print_allocation(resolution=100)
            self._defragment_memory()
            #set the param data to the new tensor buffer locations
            self._reset_param_data()
            print_rank_0("After defragmentation:")
            self.print_allocation(resolution=100)

        self.total_free = self.total_free - size

        allocated = self.total_size - self.total_free
        if allocated > self.max_allocated:
            self.max_allocated = allocated

        tensor_address = self._get_new_tensor_address(size)

        ret_tensor = self._get_new_tensor(tensor_address, size)
        print_rank_0(
            f"Free before allocation {free_before}. Allocating {size}. Free after allocation {self.total_free}. Max allocated {self.max_allocated}"
        )
        assert self.total_free + size == free_before, "Allocation bookkeeping error"

        return ret_tensor

    #assigns the tensor data to the param data and keeps track of the assignment
    #any change the underlying buffer from defragmentation will cause a
    #reassignment of the param data
    def assign_to_param(self, tensor, param, numel, shape):
        tensor_id = id(tensor)

        assert tensor_id in self.tensor_map.keys(), "No such tensor allocated by the allocator."
        assert tensor.numel() >= numel, "Assert tensor buffer does is not large enough"
        assert not tensor_id in self.id_to_params.keys(), "This tensor has already been assigned to a param"

        self.id_to_params[tensor_id] = [param]

        replicated_tensor = tensor.narrow(0, 0, numel).view(shape)
        param.data = replicated_tensor.data
        param.contiguous_tensor_id = tensor_id

    #deletes the tensor and frees up the underlying buffer
    def release_tensor(self, tensor):
        free_before = self.total_free
        tensor_id = id(tensor)
        tensor_size = tensor.numel()
        self._release_tensor(tensor_id)
        self._unassign_params(tensor_id)
        self.total_free += tensor_size
        print_rank_0(
            f"Free before release {free_before}. Released {tensor.numel()}. Total free after {self.total_free}.")
        assert self.total_free - tensor_size == free_before, "Release bookkeeping error"

    def release_tensor_with_id(self, tensor_id):
        free_before = self.total_free
        assert tensor_id in self.tensor_map.keys(), "Invalid tensor id"
        tensor = self.tensor_map[tensor_id]
        tensor_size = tensor.numel()
        self._release_tensor(tensor_id)
        self._unassign_params(tensor_id)
        self.total_free += tensor_size
        print_rank_0(
            f"Free before release {free_before}. Released {tensor.numel()}. Total free after {self.total_free}.")
        assert self.total_free - tensor_size == free_before, "Release bookkeeping error"

    #shows the current memory allocation at specified resolution
    def print_allocation(self, resolution=200):
        total_size = self.buffer.numel() * 1.0
        empty = []
        for addr, size in self.contiguous_sizes.items():
            start = int(addr * resolution / total_size)
            end = int((addr + size) * resolution / total_size)
            empty.extend(range(start, end))
        s = ''
        for i in range(resolution):
            s += '.' if i in empty else '|'
        print_rank_0(s)

    def max_allocated(self):
        return self.max_allocated

    #to be called after defragmentation that moves the tensor buffers
    #this call reassigns the data of all the parameters using the tensor buffers
    def _reset_param_data(self):
        for id, tensor in self.tensor_map.items():
            for param in self.id_to_params[id]:
                param.data = tensor.narrow(0, 0, param.numel()).view(param.data.shape).data

    def _unassign_params(self, tensor_id):
        if tensor_id in self.id_to_params.keys():
            del self.id_to_params[tensor_id]

    def _release_tensor(self, tensor_id):
        assert tensor_id in self.tensor_addresses, f"Tensor id {tensor_id} not found"

        address = self.tensor_addresses[tensor_id]
        contiguous_size = self.tensor_map[tensor_id].numel()

        del self.tensor_addresses[tensor_id]
        del self.tensor_ids[address]
        del self.tensor_map[tensor_id]
        del self.tensor_sizes[address]

        self._consolidate_address(address, contiguous_size)
        self.largest_contiguous = self._largest_contiguous()

    def _consolidate_address(self, address, contiguous_size):

        #consolidate next buffer
        end_address = address + contiguous_size
        if end_address in self.contiguous_sizes:
            contiguous_size += self.contiguous_sizes[end_address]
            del self.contiguous_sizes[end_address]

        #consolidate previous buffer
        for addr, size in self.contiguous_sizes.items():
            if addr + size == address:
                del self.contiguous_sizes[addr]
                contiguous_size += size
                address = addr
                break

        self.contiguous_sizes[address] = contiguous_size

    def _defragment_memory(self):
        empty_addresses = sorted(self.contiguous_sizes.keys())
        tensor_addresses = sorted(self.tensor_addresses.values())

        tensor_index = 0

        while tensor_index < len(tensor_addresses):

            empty_addr = empty_addresses[0]
            empty_size = self.contiguous_sizes[empty_addr]

            tensor_addr = tensor_addresses[tensor_index]
            tensor_size = self.tensor_sizes[tensor_addr]
            tensor_id = self.tensor_ids[tensor_addr]
            tensor = self.tensor_map[self.tensor_ids[tensor_addr]]

            assert tensor_size == tensor.numel(), \
                f"Size mismatch. {tensor_size} is allocated at addr {tensor_addr} but tensor size is {tensor.numel()} "

            assert empty_addr != tensor_addr, \
                f"Cannot have same empty address {empty_addr} and tensor address {tensor_addr}"

            if empty_addr < tensor_addr:

                if empty_size >= tensor_size:
                    dest_buffer = self.buffer.narrow(0, empty_addr, tensor_size)
                    src_buffer = self.buffer.narrow(0, tensor_addr, tensor_size)
                    dest_buffer.data.copy_(src_buffer.data)
                else:

                    #print_rank_0(f'empty addr : {empty_addr}, empty size {empty_size} tensor addr {tensor_addr} tensor size {tensor_size}')
                    src_addr = tensor_addr
                    dest_addr = empty_addr
                    while src_addr < (tensor_addr + tensor_size):
                        copy_size = min(empty_size, tensor_addr + tensor_size - src_addr)

                        dest_buffer = self.buffer.narrow(0, dest_addr, copy_size)
                        src_buffer = self.buffer.narrow(0, src_addr, copy_size)

                        dest_buffer.data.copy_(src_buffer.data)

                        src_addr += copy_size
                        dest_addr += copy_size

                self._replace_old_address_with_new(tensor_id, empty_addr)

                tensor_index += 1

            else:
                tensor_index += 1

            empty_addresses = sorted(self.contiguous_sizes.keys())

    def _replace_old_address_with_new(self, tensor_id, new_address):

        tensor = self.tensor_map[tensor_id]
        tensor_size = tensor.numel()
        tensor.data = self.buffer.narrow(0, new_address, tensor_size).data

        self._release_tensor(tensor_id)
        self._mark_as_occupied(new_address, tensor_size)

        self.tensor_ids[new_address] = tensor_id
        self.tensor_map[tensor_id] = tensor
        self.tensor_addresses[tensor_id] = new_address
        self.tensor_sizes[new_address] = tensor_size

    def _get_new_tensor_address(self, size):
        tensor_address = None
        for address, contiguous_size in self.contiguous_sizes.items():
            if contiguous_size >= size and \
                    (tensor_address is None or \
                    contiguous_size < self.contiguous_sizes[tensor_address]):
                tensor_address = address
        assert tensor_address is not None, "address cannot be None"
        return tensor_address

    def _get_new_tensor(self, address, size):
        available_contiguous_size = self.contiguous_sizes[address]

        assert size <= available_contiguous_size, \
            f"Tensor numel {size} is large than available contiguous size {available_contiguous_size}"
        self.count += 1
        new_tensor = self.buffer.narrow(0, address, size)
        tensor_id = id(new_tensor)
        self.tensor_addresses[tensor_id] = address
        self.tensor_sizes[address] = size

        self.tensor_ids[address] = tensor_id
        self.tensor_map[tensor_id] = new_tensor

        self._mark_as_occupied(address, size)

        return new_tensor

    def _largest_contiguous(self):
        if len(self.contiguous_sizes) > 0:
            return max([size for _, size in self.contiguous_sizes.items()])
        else:
            return 0

    def _mark_as_occupied(self, address, size):
        available_contiguous_size = self.contiguous_sizes[address]
        del self.contiguous_sizes[address]

        if available_contiguous_size != size:
            self.contiguous_sizes[address + size] = available_contiguous_size - size

        self.largest_contiguous = self._largest_contiguous()
