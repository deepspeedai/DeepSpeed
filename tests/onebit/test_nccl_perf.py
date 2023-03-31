# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed.comm as dist
import numpy as np
import argparse
import deepspeed
import os

from deepspeed.runtime.comm.nccl import NcclBackend
from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.accelerator import get_accelerator
from statistics import mean

timers = SynchronizedWallClockTimer()

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
args = parser.parse_args()

deepspeed.init_distributed(dist_backend=get_accelerator().communication_backend_name())
args.local_rank = int(os.environ['LOCAL_RANK'])

get_accelerator().set_device(args.local_rank)
device = torch.device(get_accelerator().device_name(), args.local_rank)

size = dist.get_world_size()
rank = dist.get_rank()

backend = NcclBackend()
local_rank = args.local_rank

# Setting tensor_size (BERT-Large)
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

a_sign = a.sign().add_(1).bool().float().add_(-0.5).mul_(2.0)
scale = a.norm() / np.sqrt(a.numel())
a_compressed = scale * a_sign

print("Shape of the compressed buffer:", a_compressed.shape) if rank == 0 else None

for i in range(iters):
    timers('compressed_allreduce').start()
    backend.compressed_allreduce(a, worker_error, server_error, local_rank)
    #deepspeed.comm.all_reduce(a_compressed)
    timers('compressed_allreduce').stop()
    time_list.append(timers('compressed_allreduce').elapsed())

#timer_names = ['compressed_allreduce']
#timers.log(names=timer_names, normalizer=1, memory_breakdown=None)

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
print("min, max, and mean = {} ms, {} ms, {} ms".format(minlat, maxlat, meanlat)) if rank == 0 else None
#print("tensor shape", a.shape)
duration = meanlat / 1e3
tput = ((tensor_size * 4) / duration)
print("algo throughput: %f Bytes/s, %f GB/s" % (tput, tput / 1e9)) if rank == 0 else None
size = tensor_size * 4
n = dist.get_world_size()
busbw = (size / duration) * (2 * (n - 1) / n)
print("busbw: %f GB/s" % (busbw / 1e9)) if rank == 0 else None
