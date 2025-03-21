// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "z1.h"
#include "deepcompile.h"

#define USE_C10D_NCCL

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>

namespace dc {

class Z1CustomOpExecutor : public CustomOpExecutor {
public:
    Z1CustomOpExecutor(c10::intrusive_ptr<c10d::ProcessGroup> process_group,
                       std::shared_ptr<DSParamRegistry> param_registry,
                       std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets,
                       std::vector<long> ds_ids,
                       ncclComm_t nccl_comm,
                       at::cuda::CUDAStream rs_stream,
                       at::cuda::CUDAStream copy_stream,
                       bool pre_div_reduce)
        : CustomOpExecutor(process_group,
                           param_registry,
                           reduce_buckets,
                           ds_ids,
                           nccl_comm,
                           rs_stream,
                           copy_stream,
                           pre_div_reduce)
    {
    }
    ~Z1CustomOpExecutor() {}

    void endBackward() override
    {
        if (param_updated_) {
            for (auto& it : has_acc_grad_) { it.second = false; }
        }
    }

    void flushReduceBucket(at::ScalarType scalar_type) override
    {
        int rank = process_group_->getRank();

        if (!hasKey(reduce_tasks_, scalar_type)) { return; }

        int64_t tmp_recv_numel = 0;
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            auto copy_done_event = rs_copy_done_events_.at(t.getDSId());
            copy_done_event->block(rs_stream_);
        }

        ncclGroupStart();
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            ncclRedOp_t op = pre_div_reduce_ ? ncclSum : ncclAvg;
            if (pre_div_reduce_) {
                at::cuda::CUDAStreamGuard guard(rs_stream_);
                t.getSendBuf().div_(process_group_->getSize());
            }

            // inplace
            ncclResult_t result = ncclAllReduce(t.getSendBuf().data_ptr(),
                                                t.getSendBuf().data_ptr(),
                                                t.getSendBuf().numel(),
                                                get_nccl_data_type(scalar_type),
                                                op,
                                                nccl_comm_,
                                                rs_stream_);
            if (result != ncclSuccess) { throw std::runtime_error("NCCL AllReduce failed"); }
        }
        ncclGroupEnd();

        {
            at::cuda::CUDAStreamGuard guard(rs_stream_);
            for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
                bool acc_grad = has_acc_grad_.at(t.getDSId());
                auto param = param_registry_->getParam(t.getDSId());
                auto grad_buf = param.getGradBuffer().flatten();

                if (grad_buf.numel() == 0) { continue; }

                int64_t offset = param.getOffset();
                auto recv_buf = t.getSendBuf().flatten().index(
                    {torch::indexing::Slice(offset, offset + grad_buf.numel())});
                if (acc_grad) {
                    grad_buf.add_(recv_buf);
                } else {
                    grad_buf.copy_(recv_buf);
                }
                has_acc_grad_[t.getDSId()] = true;
            }
        }

        reduce_buckets_->swap(scalar_type, rs_stream_, copy_stream_);

        // Not very sure if this is necessary
        // Want to prevent grad tensor from being released before the copy is done
        auto comp_stream = at::cuda::getCurrentCUDAStream();
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            auto copy_done_event = rs_copy_done_events_.at(t.getDSId());
            copy_done_event->block(comp_stream);
        }
        reduce_tasks_[scalar_type].clear();
    }
};

static at::cuda::CUDAStream rs_stream = at::cuda::getStreamFromPool(true);
static at::cuda::CUDAStream copy_stream = at::cuda::getStreamFromPool(true);

void register_graph_z1(long graph_id, const std::vector<long>& ds_ids)
{
    executors[graph_id] = std::make_shared<Z1CustomOpExecutor>(process_group,
                                                               param_registry,
                                                               reduce_buckets,
                                                               ds_ids,
                                                               nccl_comm,
                                                               rs_stream,
                                                               copy_stream,
                                                               pre_div_reduce);
}

void register_z1_param(long ds_id,
                       const std::vector<int64_t>& ds_shape,
                       at::Tensor ds_tensor,
                       at::Tensor grad_buffer,
                       int64_t offset)
{
    param_registry->registerParam(ds_id, ds_shape, ds_tensor, grad_buffer, false, offset, false);
}

}  // namespace dc
