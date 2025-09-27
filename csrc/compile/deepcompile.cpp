// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepcompile.h"

#define USE_C10D_NCCL

namespace dc {

std::shared_ptr<DSParamRegistry> param_registry;
std::unordered_map<long, std::shared_ptr<CustomOpExecutor>> executors;
std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets = nullptr;

c10::intrusive_ptr<c10d::ProcessGroup> process_group = nullptr;
c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem = nullptr;
ncclComm_t nccl_comm;
bool use_symm_mem;
bool profile = false;
bool pre_div_reduce = true;

int64_t free_activation_threshold;

bool sync_before_reduce;     // for debugging
bool sync_after_reduce;      // for debugging
bool sync_before_allgather;  // for debugging
bool sync_after_allgather;   // for debugging

std::vector<int64_t> sizes_to_int_vector(at::IntArrayRef sizes)
{
    std::vector<int64_t> result;
    for (int i = 0; i < sizes.size(); i++) { result.push_back(sizes[i]); }
    return result;
}

void enable_profiling(bool enable) { profile = enable; }

bool is_profiling() { return profile; }

c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> getSymmMemWorkspace(int64_t size)
{
    c10::Device device = c10::Device(c10::kCUDA, c10::cuda::current_device());
    std::vector<int64_t> sizes = {size};
    std::vector<int64_t> strides = {1};
    at::Tensor sym_mem_ws = c10d::symmetric_memory::empty_strided_p2p(
        {size}, {1}, c10::ScalarType::Byte, device, process_group->getGroupName(), std::nullopt);
    return c10d::symmetric_memory::rendezvous(sym_mem_ws);
}

void lazy_init_symm_memory()
{
    if (use_symm_mem && !symm_mem) {
        int64_t max_param_size = 0;
        for (const auto& it : param_registry->getParams()) {
            int64_t size = it.second.getDSTensor().numel() * it.second.getDSTensor().element_size();
            if (size > max_param_size) { max_param_size = size; }
        }
        symm_mem = getSymmMemWorkspace(max_param_size);
    }
}

ncclDataType_t get_nccl_data_type(at::ScalarType scalar_type)
{
    switch (scalar_type) {
        case at::kFloat: return ncclFloat;
        case at::kHalf: return ncclHalf;
        case at::kDouble: return ncclDouble;
        case at::kBFloat16: return ncclBfloat16;
        case at::kLong: return ncclInt64;
        case at::kInt: return ncclInt;
        case at::kChar: return ncclInt8;
        default: throw std::runtime_error("Unsupported scalar type");
    }
}

void reset()
{
    executors.clear();
    // We keep the buckets for memory estimation
    // reduce_buckets->clear();
}

void cleanup()
{
    reset();

    ncclCommDestroy(nccl_comm);
    process_group = nullptr;
    symm_mem = nullptr;
}

at::Tensor reduce_grad(at::Tensor grad_tensor, long graph_id, long ds_id)
{
    if (sync_before_reduce) { c10::cuda::device_synchronize(); }

    assert(hasKey(executors, graph_id));
    if (!profile) { executors[graph_id]->reduceGrad(grad_tensor, ds_id); }

    if (sync_after_reduce) { c10::cuda::device_synchronize(); }

    return at::Tensor();
}

at::Tensor reduce_grad_meta(at::Tensor grad_tensor, long graph_id, long ds_id)
{
    return at::Tensor();
}

void free_tensors(std::vector<at::Tensor> tensors)
{
    if (!profile) {
        for (auto& tensor : tensors) {
            if (tensor.is_cuda() && tensor.numel() > free_activation_threshold) {
                tensor.record_stream(at::cuda::getCurrentCUDAStream());
                tensor.set_data(torch::empty({0}, tensor.options()));
            }
        }
    }
}

void free_tensors_meta(std::vector<at::Tensor> tensors) {}

template <typename T>
static T get_config(pybind11::object& config, const char* name)
{
    return pybind11::getattr(config, name).cast<T>();
}

void init(c10::intrusive_ptr<c10d::ProcessGroup> pg,
          pybind11::object& config,
          int64_t initial_reduce_bucket_size)
{
    process_group = pg;

    ncclUniqueId ncclID;
    ncclGetUniqueId(&ncclID);

    // ProcessGroup doesn't have an API to get the CUDA stream for comm calls.
    // So we create a NCCL communicator and call NCCL APIs directly.
    auto vec = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(&ncclID),
                                    reinterpret_cast<uint8_t*>(&ncclID) + NCCL_UNIQUE_ID_BYTES);
    auto device = torch::Device(torch::kCUDA);
    at::Tensor tensor = torch::from_blob(vec.data(), {static_cast<long>(vec.size())}, torch::kUInt8)
                            .to(torch::Device(torch::kCUDA));
    std::vector<at::Tensor> bcast_input = {tensor};

    process_group->broadcast(bcast_input, c10d::BroadcastOptions())->wait();

    // create a new nccl communicator
    std::memcpy(&ncclID, tensor.to(torch::Device(torch::kCPU)).data_ptr(), NCCL_UNIQUE_ID_BYTES);
    ncclCommInitRank(&nccl_comm, process_group->getSize(), ncclID, process_group->getRank());

    param_registry = std::make_shared<DSParamRegistry>();
    reduce_buckets = std::make_shared<DoubleBufferedReduceBucket>(
        initial_reduce_bucket_size, get_config<bool>(config, "double_buffer"));
    use_symm_mem = get_config<bool>(config, "symmetric_memory");
    free_activation_threshold = get_config<int64_t>(config, "free_activation_threshold");

    sync_before_reduce = get_config<bool>(config, "sync_before_reduce");
    sync_after_reduce = get_config<bool>(config, "sync_after_reduce");
    sync_before_allgather = get_config<bool>(config, "sync_before_allgather");
    sync_after_allgather = get_config<bool>(config, "sync_after_allgather");
}

void start_forward()
{
    lazy_init_symm_memory();
    for (auto& it : executors) { it.second->startForward(); }
}

void end_forward()
{
    for (auto& it : executors) { it.second->endForward(); }
}

void start_backward(bool update)
{
    for (auto& it : executors) { it.second->startBackward(update); }
}

void end_backward(long graph_id)
{
    auto executor = getExecutor<CustomOpExecutor>(graph_id, executors);
    executor->endBackward();
}

}  // namespace dc
