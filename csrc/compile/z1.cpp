// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "z1.h"
#include "deepcompile.h"

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

    at::Tensor reduceGrad(at::Tensor grad_tensor, long ds_id) override
    {
        if (!hasKey(grad_tensors_, ds_id)) {
            grad_tensors_[ds_id] = grad_tensor;
        } else {
            grad_tensors_[ds_id].add_(grad_tensor);
        }

        if (param_updated_) {
            CustomOpExecutor::reduceGrad(grad_tensors_[ds_id], ds_id);
            grad_tensors_.erase(ds_id);
        }

        return at::Tensor();
    }

    void flushReduceBucket(at::ScalarType scalar_type) override
    {
        if (!hasKey(reduce_tasks_, scalar_type)) { return; }

        blockCopyEvents(scalar_type);
        applyPreDivision(scalar_type);

        // NCCL AllReduce operation
        ncclGroupStart();
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            ncclResult_t result = ncclAllReduce(t.getSendBuf().data_ptr(),
                                                t.getSendBuf().data_ptr(),
                                                t.getSendBuf().numel(),
                                                get_nccl_data_type(scalar_type),
                                                getReductionOp(),
                                                nccl_comm_,
                                                rs_stream_);
            if (result != ncclSuccess) { throw std::runtime_error("NCCL AllReduce failed"); }
        }
        ncclGroupEnd();

        // Copy results to gradient buffers
        {
            at::cuda::CUDAStreamGuard guard(rs_stream_);
            for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
                auto param = param_registry_->getParam(t.getDSId());
                auto grad_buf = param.getGradBuffer().flatten();

                if (grad_buf.numel() == 0) { continue; }

                int64_t offset = param.getOffset();
                auto recv_buf = t.getSendBuf().flatten().index(
                    {torch::indexing::Slice(offset, offset + grad_buf.numel())});
                grad_buf.copy_(recv_buf);
            }
        }

        performCleanup(scalar_type);
    }

protected:
    std::unordered_map<long, at::Tensor> grad_tensors_;
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

void register_param(long ds_id,
                    const std::vector<int64_t>& ds_shape,
                    at::Tensor ds_tensor,
                    at::Tensor grad_buffer,
                    int64_t offset)
{
    param_registry->registerParam(ds_id, ds_shape, ds_tensor, grad_buffer, false, offset, false);
}

}  // namespace dc
