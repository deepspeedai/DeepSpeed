'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch
import cupy
import time
import numpy as np
from mpi4py import MPI

from deepspeed.runtime.compression.cupy import CupyBackend


class MpiBackend(object):
    def __init__(self, cuda_aware):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.cuda_aware = cuda_aware
        self.compression_backend = CupyBackend()

    def my_igather(self, rank, size, comm, sendbuf, recbuf, root):
        req = []
        if rank == root:
            for idx in range(size):
                if idx != rank:
                    req.append(comm.Irecv(recbuf[idx], source=idx))
                else:
                    recbuf[rank] = sendbuf
        else:
            req.append(comm.Isend(sendbuf, dest=root))
        return req

    def gather_cuda(self,
                    rank,
                    world_size,
                    comm,
                    cupy_sign_list_packed,
                    cupy_recvbuf_sign,
                    cupy_worker_scale,
                    cupy_recvbuf_scale):
        # We do in-place operations on cupy buffers so we do not return any buffers
        requests = []
        for idx in range(world_size):
            req_sign = self.my_igather(rank,
                                       world_size,
                                       comm,
                                       cupy_sign_list_packed[idx],
                                       cupy_recvbuf_sign,
                                       root=idx)
            requests += req_sign

        for idx in range(world_size):
            req_scale = self.my_igather(rank,
                                        world_size,
                                        comm,
                                        cupy_worker_scale,
                                        cupy_recvbuf_scale,
                                        root=idx)
            requests += req_scale

        MPI.Request.Waitall(requests)

    def gather_host(self,
                    rank,
                    world_size,
                    comm,
                    cupy_sign_list_packed,
                    cupy_recvbuf_sign,
                    cupy_worker_scale,
                    cupy_recvbuf_scale):

        # In-place operations are not possible for newly created cupy arrays
        # so we need to return the new buffers
        numpy_recvbuf_sign = np.zeros([world_size,
                                       cupy_sign_list_packed[rank].size],
                                      dtype=cupy_sign_list_packed[0].dtype)
        numpy_recvbuf_scale = np.zeros([world_size, 1], dtype=cupy_worker_scale.dtype)

        # 1. convert from cupy to numpy
        numpy_sign_list_packed = cupy_sign_list_packed

        for idx in range(world_size):
            numpy_sign_list_packed[idx] = cupy.asnumpy(cupy_sign_list_packed[idx])

        numpy_worker_scale = cupy.asnumpy(cupy_worker_scale)
        numpy_recvbuf_scale = cupy.asnumpy(cupy_recvbuf_scale)

        cupy.cuda.get_current_stream().synchronize()

        # 2. use numpy buffers for communication
        requests = []

        for idx in range(world_size):
            req_sign = self.my_igather(rank,
                                       world_size,
                                       comm,
                                       numpy_sign_list_packed[idx],
                                       numpy_recvbuf_sign,
                                       root=idx)
            requests += req_sign

        for idx in range(world_size):
            req_scale = self.my_igather(rank,
                                        world_size,
                                        comm,
                                        numpy_worker_scale,
                                        numpy_recvbuf_scale,
                                        root=idx)
            requests += req_scale

        MPI.Request.Waitall(requests)

        # 3. Convert back from numpy to cupy
        cupy_recvbuf_sign = cupy.asarray(numpy_recvbuf_sign)
        for idx in range(world_size):
            cupy_sign_list_packed[idx] = cupy.asarray(numpy_sign_list_packed[idx])

        cupy_worker_scale = cupy.asarray(numpy_worker_scale)
        cupy_recvbuf_scale = cupy.asarray(numpy_recvbuf_scale)
        cupy.cuda.get_current_stream().synchronize()

        return cupy_sign_list_packed, cupy_recvbuf_sign, cupy_worker_scale, cupy_recvbuf_scale

    def allgather_cuda(self,
                       comm,
                       cupy_server_sign_packed,
                       cupy_recvbuf_sign_server,
                       cupy_server_scale,
                       cupy_recvbuf_scale_server):
        comm.Allgather(cupy_server_sign_packed, cupy_recvbuf_sign_server)
        comm.Allgather(cupy_server_scale, cupy_recvbuf_scale_server)

    def allgather_host(self,
                       comm,
                       cupy_server_sign_packed,
                       cupy_recvbuf_sign_server,
                       cupy_server_scale,
                       cupy_recvbuf_scale_server):

        # 1. Convert cupy to numpy
        numpy_recvbuf_sign_server = np.zeros(
            [comm.Get_size(),
             cupy_server_sign_packed.size],
            dtype=cupy_server_sign_packed.dtype)
        numpy_recvbuf_scale_server = np.zeros([comm.Get_size(),
                                               1],
                                              dtype=cupy_server_scale.dtype)

        numpy_server_sign_packed = cupy.asnumpy(cupy_server_sign_packed)
        numpy_recvbuf_sign_server = cupy.asnumpy(cupy_recvbuf_sign_server)
        numpy_server_scale = cupy.asnumpy(cupy_server_scale)
        numpy_recvbuf_scale_server = cupy.asnumpy(cupy_recvbuf_scale_server)
        cupy.cuda.get_current_stream().synchronize()

        # 2. Communicate numpy buffers
        comm.Allgather(numpy_server_sign_packed, numpy_recvbuf_sign_server)
        comm.Allgather(numpy_server_scale, numpy_recvbuf_scale_server)
        comm.Barrier()

        # 3. Convert numpy back to cupy
        cupy_server_sign_packed = cupy.asarray(numpy_server_sign_packed)
        cupy_recvbuf_sign_server = cupy.asarray(numpy_recvbuf_sign_server)
        cupy_server_scale = cupy.asarray(numpy_server_scale)
        cupy_recvbuf_scale_server = cupy.asarray(numpy_recvbuf_scale_server)
        cupy.cuda.get_current_stream().synchronize()

        return cupy_server_sign_packed, cupy_recvbuf_sign_server, cupy_server_scale, cupy_recvbuf_scale_server

    def compressed_allreduce(self,
                             buffer_m: torch.tensor,
                             worker_error,
                             server_error,
                             local_rank):

        all_start_time = time.time()
        original_shape = buffer_m.size()
        if len(original_shape) > 1:
            buffer_m = torch.flatten(buffer_m)
        original_size = buffer_m.numel()
        worker_error_size = worker_error.numel()
        cupy.cuda.Device(local_rank).use()

        if original_size != worker_error_size:
            empty_tensor = torch.zeros(worker_error_size - original_size,
                                       device=buffer_m.device)
            buffer_m = torch.cat([buffer_m, empty_tensor])

        buffer_m.add_(worker_error)
        worker_scale = torch.norm(buffer_m) / np.sqrt(torch.numel(buffer_m))
        worker_error.set_(buffer_m - worker_scale *
                          buffer_m.sign().add_(1).bool().float().add_(-0.5).mul_(2.0))

        cupy_sign_list_packed = self.compression_backend.compress_by_chunk(
            self.compression_backend.torch2cupy(buffer_m.sign_().add_(1).bool()),
            self.size)
        cupy_worker_scale = self.compression_backend.torch2cupy(worker_scale)

        cupy_recvbuf_sign = cupy.zeros(
            [self.size,
             cupy_sign_list_packed[self.rank].size],
            dtype=cupy_sign_list_packed[0].dtype)
        cupy_recvbuf_scale = cupy.zeros([self.size, 1], dtype=cupy_worker_scale.dtype)

        # Communication Phase 1
        gather_start = time.time()
        if self.cuda_aware:
            self.gather_cuda(self.rank,
                             self.size,
                             self.comm,
                             cupy_sign_list_packed,
                             cupy_recvbuf_sign,
                             cupy_worker_scale,
                             cupy_recvbuf_scale)
        else:
            _, cupy_recvbuf_sign, _, cupy_recvbuf_scale = self.gather_host(self.rank,
               self.size,
               self.comm,
               cupy_sign_list_packed,
               cupy_recvbuf_sign,
               cupy_worker_scale,
               cupy_recvbuf_scale)
        gather_end = time.time()

        # cupy_sign_list_packed, cupy_worker_scale, worker_scale = None, None, None
        cupy_sign_list_packed = None

        compensated_server_m = self.compression_backend.cupy2torch(
            (cupy.unpackbits(cupy_recvbuf_sign.flatten())).reshape(
                self.size,
                -1)).float().add_(-0.5).mul_(2.0).mul_(
                    self.compression_backend.cupy2torch(cupy_recvbuf_scale).mul_(
                        1 / self.size)).sum(0)
        compensated_server_m.add_(server_error)
        server_scale = torch.norm(compensated_server_m) / np.sqrt(
            compensated_server_m.numel())
        server_error.set_(
            compensated_server_m - server_scale *
            compensated_server_m.sign().add_(1).bool().float().add_(-0.5).mul_(2.0))

        cupy_server_scale = self.compression_backend.torch2cupy(server_scale)

        cupy_server_sign_packed = self.compression_backend.compress_by_chunk(
            self.compression_backend.torch2cupy(
                compensated_server_m.sign_().add_(1).bool()),
            1)
        compensated_server_m = None

        cupy_recvbuf_sign_server = cupy.zeros(
            [self.size,
             cupy_server_sign_packed[0].size],
            dtype=cupy_recvbuf_sign.dtype)
        cupy_recvbuf_scale_server = cupy.zeros([self.size,
                                                1],
                                               dtype=cupy_recvbuf_scale.dtype)
        # cupy_recvbuf_sign, cupy_recvbuf_scale = None, None
        cupy_recvbuf_sign = None

        # Communication Phase 2
        if self.cuda_aware:
            self.allgather_cuda(self.comm,
                                cupy_server_sign_packed[0],
                                cupy_recvbuf_sign_server,
                                cupy_server_scale,
                                cupy_recvbuf_scale_server)
        else:
            _, cupy_recvbuf_sign_server, _, cupy_recvbuf_scale_server = self.allgather_host(self.comm,
                  cupy_server_sign_packed[0],
                  cupy_recvbuf_sign_server,
                  cupy_server_scale,
                  cupy_recvbuf_scale_server)

        # cupy_server_sign_packed, cupy_server_scale, server_scale = None, None, None
        cupy_server_sign_packed = None

        buffer_m.data.copy_(
            self.compression_backend.cupy2torch(
                (cupy.unpackbits(cupy_recvbuf_sign_server.flatten())).reshape(
                    self.size,
                    -1)).float().add_(-0.5).mul_(2.0).mul_(
                        self.compression_backend.cupy2torch(
                            cupy_recvbuf_scale_server)).flatten().data)
        if original_size != worker_error_size:
            buffer_m = buffer_m[0:original_size]
        if len(original_shape) > 1:
            buffer_m = buffer_m.reshape(original_shape)

        # cupy_recvbuf_sign_server, cupy_recvbuf_scale_server = None, None

        return buffer_m
