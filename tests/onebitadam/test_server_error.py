from mpi4py import MPI
import time
import torch
import torch.distributed as dist
import numpy as np
import deepspeed
from deepspeed.runtime.fp16.onebit_adam import OnebitAdam

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

torch.distributed.init_process_group(backend='nccl',
                                     init_method='tcp://worker-0:2245',
                                     world_size=size,
                                     rank=rank)

dummy_model = [torch.nn.Parameter(torch.ones(10))]
dummy_optim = OnebitAdam(dummy_model, cuda_aware=False)

device = torch.device('cuda', rank % torch.cuda.device_count())


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


# Input Tensor size
tensor_size = 100 * 2**20

server_size = int(tensor_size / size)
if tensor_size % (8 * size) != 0:
    right_tensor_size = tensor_size + (8 * size - (tensor_size % (8 * size)))
else:
    right_tensor_size = tensor_size

right_server_size = right_tensor_size // size

# The -0.5 is required for avoiding sign flips/errors
a = torch.rand(tensor_size, device=device) - 0.5

worker_error = torch.zeros(right_tensor_size, device=device)
server_error = torch.zeros(right_server_size, device=device)
a_torch, worker_error_torch, server_error_torch = torch_sim(a)
torch.cuda.empty_cache()
local_rank = rank % torch.cuda.device_count()

# Test the 1-bit Adam optimizer
a_after = dummy_optim.Compressed_Allreduce(a,
                                           worker_error,
                                           server_error,
                                           rank,
                                           size,
                                           comm,
                                           local_rank)

# If the error is below the threshold, it is acceptable for training
threshold = 1e-6

diff_pos = ((a_after - a_torch) > threshold)

if rank == 0:
    before_diff = torch.chunk(a_after - a_torch,
                              size)[rank] + server_error - server_error_torch
    if torch.norm(before_diff) / torch.norm(torch.chunk(a_after,
                                                        size)[rank]) < threshold:
        print('Successfully passed the test')
    else:
        print('The difference for the tensor before allgather is {}'.format(
            torch.norm(before_diff)))
