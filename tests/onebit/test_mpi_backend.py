from mpi4py import MPI
import time
import torch
import torch.distributed as dist
import numpy as np
import deepspeed

from deepspeed.runtime.comm.mpi import MpiBackend

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

deepspeed.init_distributed(dist_backend='nccl')

# Change cuda_aware to True to test out CUDA-Aware MPI communication
backend = MpiBackend(cuda_aware=False)

device = torch.device('cuda', rank % torch.cuda.device_count())


# A simulated compression function using torch.distributed
def torch_sim(a):
    a_sign = a.sign().add_(1).bool().float().add_(-0.5).mul_(2.0)
    scale = a.norm() / np.sqrt(a.numel())
    a_compressed = scale * a_sign
    a_sign = None
    worker_error = a - a_compressed
    dist.all_reduce(a_compressed)
    a_compressed.mul_(1 / dist.get_world_size())
    a_server_sign = a_compressed.sign().add_(1).bool().float().add_(-0.5).mul_(2.0)
    a_list = torch.chunk(a_compressed, chunks=dist.get_world_size())
    server_scale = [chunk_a.norm() / np.sqrt(chunk_a.numel()) for chunk_a in a_list]
    a_sign_list = torch.chunk(a_server_sign, dist.get_world_size())
    a_server_compressed = torch.cat(
        [server_scale[i] * a_sign_list[i] for i in range(dist.get_world_size())])
    rank = dist.get_rank()
    server_error = a_list[rank] - server_scale[rank] * a_sign_list[rank]
    torch.cuda.synchronize()
    torch.distributed.barrier()
    return a_server_compressed, worker_error, server_error


tensor_size = 100 * 2**20
server_size = int(tensor_size / size)
if tensor_size % (8 * size) != 0:
    right_tensor_size = tensor_size + (8 * size - (tensor_size % (8 * size)))
else:
    right_tensor_size = tensor_size
right_server_size = right_tensor_size // size

# Adding bias to the initialization of the gradient we are communicating
# In order to get rid of the case where some elements in the gradient are too small
a = (torch.rand(tensor_size, device=device) - 0.5) + 0.01 * rank

worker_error = torch.zeros(right_tensor_size, device=device)
server_error = torch.zeros(right_server_size, device=device)

a_torch, worker_error_torch, server_error_torch = torch_sim(a)
torch.cuda.empty_cache()
local_rank = rank % torch.cuda.device_count()

a_after = backend.compressed_allreduce(a, worker_error, server_error, local_rank)

threshold = 1e-6
magnitude_threshold = 1e-6
diff_mask = (a_after - a_torch) > threshold
diff_server_mask = torch.chunk(diff_mask, size)[rank]
mpi_server = torch.chunk(a_after, size)[rank] + server_error
torch_server = torch.chunk(a_torch, size)[rank] + server_error_torch

test_correctness = True

# If the number in the compensated_server_m is too small (e.g 1e-8), then calling sign() might be problematic
# The test would skip those numbers that are too small in compensated_server_m
if test_correctness:
    if torch.sum(diff_server_mask) == 0:
        print('Successfully passed the test for MPI Backend at Rank {}'.format(rank))
    else:
        check_mag_mask = mpi_server[diff_server_mask] > magnitude_threshold
        if torch.sum(check_mag_mask) == 0:
            print('Successfully passed the test for MPI Backend at Rank {}'.format(rank))
        else:
            print('Fails at {} of positions'.format(torch.sum(check_mag_mask)))
