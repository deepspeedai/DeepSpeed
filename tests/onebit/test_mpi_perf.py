# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from mpi4py import MPI
import torch
import deepspeed

from deepspeed.runtime.comm.mpi import MpiBackend

# Configure wall clock timer
from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.accelerator import get_accelerator

from statistics import mean

timers = SynchronizedWallClockTimer()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

deepspeed.init_distributed(dist_backend=get_accelerator().communication_backend_name())
# Change cuda_aware to True to test out CUDA-Aware MPI communication
backend = MpiBackend(cuda_aware=False)

local_rank = rank % get_accelerator().device_count()
device = torch.device(get_accelerator().device_name(), local_rank)

tensor_size = 300 * 2**20
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

warmup = 10
iters = 10

# Warmup
for i in range(warmup):
    backend.compressed_allreduce(a, worker_error, server_error, local_rank)

time_list = []

for i in range(iters):
    timers('compressed_allreduce').start()
    backend.compressed_allreduce(a, worker_error, server_error, local_rank)
    timers('compressed_allreduce').stop()
    time_list.append(timers('compressed_allreduce').elapsed())

timer_names = ['compressed_allreduce']
timers.log(names=timer_names, normalizer=1, memory_breakdown=None)

places = 2
convert = 1e3
float_size = 4

if rank == 0:
    for i in range(iters):
        lat = time_list[i]
        print("latency = ", lat * convert)

minlat = round(min(time_list) * convert)
maxlat = round(max(time_list) * convert)
meanlat = round(mean(time_list) * convert, places)
print("min, max, and mean = {} ms, {} ms, {} ms".format(minlat, maxlat, meanlat))
