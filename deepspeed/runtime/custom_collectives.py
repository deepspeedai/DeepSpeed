'''
Copyright 2019 The Microsoft DeepSpeed Team
'''

from mpi4py import MPI
import numpy as np
import cupy


def my_igather(rank, size, comm, sendbuf, recbuf, root):
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


def gather_cuda(rank,
                world_size,
                comm,
                cupy_sign_list_packed,
                cupy_recvbuf_sign,
                cupy_worker_scale,
                cupy_recvbuf_scale):
    # We do in-place operations on cupy buffers so we do not return any buffers
    requests = []
    for idx in range(world_size):
        req_sign = my_igather(rank,
                              world_size,
                              comm,
                              cupy_sign_list_packed[idx],
                              cupy_recvbuf_sign,
                              root=idx)
        requests += req_sign

    for idx in range(world_size):
        req_scale = my_igather(rank,
                               world_size,
                               comm,
                               cupy_worker_scale,
                               cupy_recvbuf_scale,
                               root=idx)
        requests += req_scale

    MPI.Request.Waitall(requests)


def gather_host(rank,
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
        req_sign = my_igather(rank,
                              world_size,
                              comm,
                              numpy_sign_list_packed[idx],
                              numpy_recvbuf_sign,
                              root=idx)
        requests += req_sign

    for idx in range(world_size):
        req_scale = my_igather(rank,
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


def allgather_cuda(comm,
                   cupy_server_sign_packed,
                   cupy_recvbuf_sign_server,
                   cupy_server_scale,
                   cupy_recvbuf_scale_server):
    comm.Allgather(cupy_server_sign_packed, cupy_recvbuf_sign_server)
    comm.Allgather(cupy_server_scale, cupy_recvbuf_scale_server)


def allgather_host(comm,
                   cupy_server_sign_packed,
                   cupy_recvbuf_sign_server,
                   cupy_server_scale,
                   cupy_recvbuf_scale_server):

    # 1. Convert cupy to numpy
    numpy_recvbuf_sign_server = np.zeros([comm.Get_size(),
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
