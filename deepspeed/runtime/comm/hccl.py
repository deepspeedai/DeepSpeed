# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import numpy as np
import torch
import torch_npu
import deepspeed.comm as dist


class HcclBackend(object):

    def __init__(self, mpu=None):
        if mpu is None:
            self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
        else:
            self.mpu = mpu
            self.world_group = self.mpu.get_data_parallel_group()
        self.size = dist.get_world_size(group=self.world_group)
        self.rank = dist.get_rank(group=self.world_group)

    def my_igather(self, rank, size, group, sendbuf, recvbuf, root):
        req = []
        if rank == root:
            for idx in range(size):
                if idx != rank:
                    req.append(dist.irecv(recvbuf[idx], src=idx, group=group))
                else:
                    recvbuf[rank] = sendbuf
        else:
            req.append(dist.isend(sendbuf, group=group, dst=root))
        return req

    def my_gather(self, rank, size, group, sendbuf, recvbuf, root):
        if rank == root:
            for idx in range(size):
                if idx != rank:
                    dist.recv(recvbuf[idx], src=idx, group=group)
                else:
                    recvbuf[rank] = sendbuf
        else:
            dist.send(sendbuf, group=group, dst=root)

    def compressed_allreduce(self, buffer_m: torch.tensor, worker_error, server_error, local_rank):
        original_shape = buffer_m.size()
        if len(original_shape) > 1:
            buffer_m = torch.flatten(buffer_m)

        # align size of original_buffer and error
        original_size = buffer_m.numel()
        worker_error_size = worker_error.numel()
        if original_size != worker_error_size:
            empty_tensor = torch.zeros(worker_error_size - original_size, device=buffer_m.device)
            buffer_m = torch.cat([buffer_m, empty_tensor])

        buffer_m.add_(worker_error)
        worker_scale = torch.linalg.norm(buffer_m) / np.sqrt(torch.numel(buffer_m))

        worker_error.set_(buffer_m - worker_scale * buffer_m.sign().add_(1).bool().float().add_(-0.5).mul_(2.0))

        sign_list_packed_tmp = torch_npu.npu_sign_bits_pack(buffer_m, self.size).type(torch.int8)

        recvbuf_sign = torch.zeros([self.size, len(sign_list_packed_tmp[self.rank])],
                                   dtype=sign_list_packed_tmp[0].dtype,
                                   device=sign_list_packed_tmp.device)

        sign_list_packed = [sign_list_packed_tmp[idx] for idx in range(self.size)]

        recvbuf_scale = [
            torch.zeros(1, dtype=worker_scale.dtype, device=torch.device(local_rank)) for _ in range(self.size)
        ]

        # communication phase 1
        # all to all for sign
        dist.all_to_all_single(recvbuf_sign, torch.stack(sign_list_packed), group=self.world_group)
        # all gather for scale
        dist.all_gather(recvbuf_scale, worker_scale, group=self.world_group)

        flattened_recvbuf_sign = recvbuf_sign.type(torch.uint8).flatten()
        compensated_server_m = torch_npu.npu_sign_bits_unpack(flattened_recvbuf_sign, self.size, torch.float32) \
            .mul_(torch.stack(recvbuf_scale).mul_(1 / self.size)).sum(0)

        compensated_server_m.add_(server_error)

        server_scale = torch.norm(compensated_server_m) / np.sqrt(compensated_server_m.numel())

        server_error.set_(compensated_server_m -
                          server_scale * compensated_server_m.sign().add_(1).bool().float().add_(-0.5).mul_(2.0))

        server_sign_packed = torch_npu.npu_sign_bits_pack(compensated_server_m, 1).type(torch.int8)

        # recvbuf_sign_server
        recvbuf_sign_server_tmp = torch.zeros([self.size, len(server_sign_packed[0])],
                                              dtype=recvbuf_sign.dtype,
                                              device=server_sign_packed.device)

        recvbuf_sign_server = [recvbuf_sign_server_tmp[idx] for idx in range(self.size)]

        # recvbuf_scale_server
        recvbuf_scale_server_tmp = torch.zeros([self.size, 1],
                                               dtype=worker_scale.dtype,
                                               device=server_sign_packed.device)

        recvbuf_scale_server = [recvbuf_scale_server_tmp[idx] for idx in range(self.size)]

        # communication Phase 2
        dist.all_gather(recvbuf_sign_server, server_sign_packed[0], group=self.world_group)
        dist.all_gather(recvbuf_scale_server, server_scale, group=self.world_group)

        recvbuf_sign_server = torch.stack(recvbuf_sign_server)

        flattened_recvbuf_sign_server = recvbuf_sign_server.type(torch.uint8).flatten()

        buffer_m.data.copy_(
            torch_npu.npu_sign_bits_unpack(flattened_recvbuf_sign_server, self.size,
                                           torch.float32).mul_(recvbuf_scale_server_tmp).flatten().data)

        if original_size != worker_error_size:
            buffer_m = buffer_m[0:original_size]
        if len(original_shape) > 1:
            buffer_m = buffer_m.reshape(original_shape)

        return buffer_m
