// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#define NOMINMAX  // Windows idiosyncrasy
                  // https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c

#define USE_C10D_NCCL

#include <stdio.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#if __has_include(<torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>)
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#else
#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>
#endif

namespace dc {

template <typename K, typename V>
static bool hasKey(const std::unordered_map<K, V>& map, const K& key)
{
    return map.find(key) != map.end();
}

template <typename T>
inline std::string to_string(const T& v)
{
    std::stringstream ss;
    ss << v;
    return ss.str();
}

template <typename L>
size_t productDim(const L& dim)
{
    size_t prod = 1;
    for (auto d : dim) { prod *= d; }
    return prod;
}

template <typename T>
std::string join_as_str(const T& v, const char* delim = ",", const size_t maxlen = 0)
{
    std::stringstream ss;

    if (!v.empty()) {
        auto it = v.begin();
        ss << to_string(*it);
        it++;
        for (; it != v.end(); ++it) {
            if (delim) ss << delim;
            ss << to_string(*it);
        }
    }

    std::string s = ss.str();
    if (maxlen > 0 && s.length() > maxlen) { s = s.substr(0, maxlen) + " ..."; }

    return "[" + s + "]";
}

template <typename T>
std::string tensorPtrToString(T* ptr, size_t size, size_t str_len = 100)
{
    std::vector<T> vals;
    for (size_t i = 0; i < size; i++) {
        vals.push_back(*ptr);
        ptr++;
    }
    return join_as_str(vals, ",", str_len);
}

std::string tensorPtrToString(void* ptr,
                              size_t size,
                              c10::ScalarType datatype,
                              size_t max_elem = 20,
                              size_t max_str_len = 100);

std::string tensorToString(const at::Tensor& t, size_t max_elem = 20, size_t max_str_len = 100);

std::string tensorDimToString(const at::Tensor& t);

at::Tensor test_call(at::Tensor param);

extern c10::intrusive_ptr<c10d::ProcessGroup> process_group;
extern c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem;
extern ncclComm_t nccl_comm;
extern bool use_symm_mem;
extern bool profile;
extern bool pre_div_reduce;

extern bool sync_before_reduce;     // for debugging
extern bool sync_after_reduce;      // for debugging
extern bool sync_before_allgather;  // for debugging
extern bool sync_after_allgather;   // for debugging

std::vector<int64_t> sizes_to_int_vector(at::IntArrayRef sizes);
void enable_profiling(bool enable);
bool is_profiling();

c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> getSymmMemWorkspace(int64_t size);
void lazy_init_symm_memory();
ncclDataType_t get_nccl_data_type(at::ScalarType scalar_type);
void cleanup();

class ReduceTask {
public:
    ReduceTask(long ds_id, at::Tensor grad, at::Tensor send_buf)
        : ds_id_(ds_id), grad_(std::move(grad)), send_buf_(std::move(send_buf))
    {
    }

    long getDSId() const { return ds_id_; }
    at::Tensor getSendBuf() const { return send_buf_; }

private:
    long ds_id_;
    at::Tensor grad_;
    at::Tensor send_buf_;
};

class ReduceBucket {
public:
    ReduceBucket(int64_t size, at::ScalarType scalar_type) : size_(size), scalar_type_(scalar_type)
    {
        buffer_ = torch::empty({size}, at::TensorOptions().dtype(scalar_type).device(at::kCUDA));
        offset_ = 0;
    }

    int64_t getSize() const { return size_; }
    int64_t getOffset() const { return offset_; }
    at::Tensor getBuffer() const { return buffer_; }
    at::ScalarType getScalarType() const { return scalar_type_; }

    void reserve(int64_t size)
    {
        if (size > size_) {
            buffer_ =
                torch::empty({size}, at::TensorOptions().dtype(scalar_type_).device(at::kCUDA));
            size_ = size;
        }
    }

    at::Tensor allocate(int64_t numel)
    {
        if (offset_ + numel > size_) {
            throw std::runtime_error("Buffer size exceeds the reduce bucket size");
        }

        at::Tensor result = buffer_.index({torch::indexing::Slice(offset_, offset_ + numel)});
        offset_ += numel;
        return result;
    }

    bool shouldFlush(int64_t numel) { return offset_ > 0 && offset_ + numel > size_; }

    void reset() { offset_ = 0; }

private:
    int64_t size_;
    int64_t offset_;
    at::Tensor buffer_;
    at::ScalarType scalar_type_;
};

class DoubleBufferedReduceBucket {
public:
    DoubleBufferedReduceBucket(int64_t initial_bucket_size, bool enable_double_buffer)
        : initial_bucket_size_(initial_bucket_size), enable_double_buffer_(enable_double_buffer)
    {
    }

    void swap(at::ScalarType scalar_type,
              at::cuda::CUDAStream rs_stream,
              at::cuda::CUDAStream copy_stream)
    {
        assert(hasKey(current_buffer_, scalar_type));
        assert(hasKey(current_buffer_events_, scalar_type));

        current_buffer_.at(scalar_type)->reset();
        current_buffer_events_.at(scalar_type)->record(rs_stream);

        if (enable_double_buffer_) {
            assert(hasKey(shadow_buffer_, scalar_type));
            assert(hasKey(shadow_buffer_events_, scalar_type));

            auto tmp = current_buffer_.at(scalar_type);
            current_buffer_[scalar_type] = shadow_buffer_.at(scalar_type);
            shadow_buffer_[scalar_type] = tmp;

            auto tmp_event = current_buffer_events_.at(scalar_type);
            current_buffer_events_[scalar_type] = shadow_buffer_events_.at(scalar_type);
            shadow_buffer_events_[scalar_type] = tmp_event;
        }
    }

    std::shared_ptr<ReduceBucket> getBuffer(at::ScalarType scalar_type)
    {
        if (!hasKey(current_buffer_, scalar_type)) {
            current_buffer_[scalar_type] =
                std::make_shared<ReduceBucket>(initial_bucket_size_, scalar_type);
            current_buffer_events_[scalar_type] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);

            if (enable_double_buffer_) {
                shadow_buffer_[scalar_type] =
                    std::make_shared<ReduceBucket>(initial_bucket_size_, scalar_type);
                shadow_buffer_events_[scalar_type] =
                    std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            }
        }

        return current_buffer_.at(scalar_type);
    }

    std::shared_ptr<at::cuda::CUDAEvent> getEvent(at::ScalarType scalar_type)
    {
        assert(hasKey(current_buffer_events_, scalar_type));
        return current_buffer_events_.at(scalar_type);
    }

    void clear()
    {
        current_buffer_.clear();
        shadow_buffer_.clear();
        current_buffer_events_.clear();
        shadow_buffer_events_.clear();
    }

private:
    int64_t initial_bucket_size_;
    bool enable_double_buffer_;
    std::unordered_map<at::ScalarType, std::shared_ptr<ReduceBucket>> current_buffer_;
    std::unordered_map<at::ScalarType, std::shared_ptr<ReduceBucket>> shadow_buffer_;
    std::unordered_map<at::ScalarType, std::shared_ptr<at::cuda::CUDAEvent>> current_buffer_events_;
    std::unordered_map<at::ScalarType, std::shared_ptr<at::cuda::CUDAEvent>> shadow_buffer_events_;
};

class DSParam {
public:
    DSParam(long id,
            std::vector<int64_t> ds_shape,
            at::Tensor ds_tensor,
            at::Tensor grad_buffer,
            bool partitioned,
            int64_t offset,  // for Z1
            bool persistent  // for Z3
            )
        : id_(id),
          shape_(std::move(ds_shape)),
          ds_tensor_(ds_tensor),
          ds_dtype_(ds_tensor.scalar_type()),
          grad_buffer_(grad_buffer),
          partitioned_(partitioned),
          offset_(offset),
          persistent_(persistent),
          offload_stream_(at::cuda::getStreamFromPool()),
          reload_stream_(at::cuda::getStreamFromPool())
    {
    }

    long getId() const { return id_; }
    std::vector<int64_t> getShape() const { return shape_; }
    at::ScalarType getDtype() const { return ds_dtype_; }
    at::Tensor getDSTensor() const
    {
        // If the reload event exists and is complete, return the reloaded tensor (if defined)
        if (reload_done_event_) {
            if (!reload_done_event_->query()) {
                reload_done_event_->block(at::cuda::getCurrentCUDAStream());
            }
            if (ds_reload_tensor_.defined()) { return ds_reload_tensor_; }
        }
        // Otherwise, if an offload event exists, wait for it to complete
        if (offload_done_event_) {
            if (!offload_done_event_->query()) {
                offload_done_event_->block(at::cuda::getCurrentCUDAStream());
            }
        }
        return ds_tensor_;
    }
    at::Tensor getGradBuffer() const { return grad_buffer_; }
    bool isPartitioned() const { return partitioned_; }
    int64_t getOffset() const { return offset_; }
    void setPersistent(bool persistent) { persistent_ = persistent; }
    bool isPersistent() const { return persistent_; }

    void offload()
    {
        // If a reloaded tensor exists, offload its data back to ds_tensor_
        if (ds_reload_tensor_.defined()) {
            auto comp_stream = at::cuda::getCurrentCUDAStream();
            comp_done_event_ = std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            // Record completion and wait on the offload stream
            comp_done_event_->record(comp_stream);
            comp_done_event_->block(offload_stream_);
            offload_done_event_ = std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);

            {
                at::cuda::CUDAStreamGuard guard(offload_stream_);
                ds_tensor_.copy_(ds_reload_tensor_, /*non_blocking=*/true);
                ds_reload_tensor_.reset();  // Clear the reloaded tensor
                offload_done_event_->record(offload_stream_);
            }
            // Reset the reload event to indicate that no valid reload is present.
            if (reload_done_event_) { reload_done_event_.reset(); }
        }
    }

    void reload()
    {
        // Reload only if the current ds_tensor_ is on CPU
        if (ds_tensor_.device().is_cpu()) {
            auto comp_stream = at::cuda::getCurrentCUDAStream();
            comp_done_event_ = std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            // Record and wait on the reload stream
            comp_done_event_->record(comp_stream);
            comp_done_event_->block(reload_stream_);
            reload_done_event_ = std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);

            {
                at::cuda::CUDAStreamGuard guard(reload_stream_);
                ds_reload_tensor_ =
                    at::empty_like(ds_tensor_, ds_tensor_.options().device(torch::kCUDA));
                ds_reload_tensor_.copy_(ds_tensor_, /*non_blocking=*/true);
                reload_done_event_->record(reload_stream_);
            }
            // Reset offload_done_event if it exists to clear any stale offload state.
            if (offload_done_event_) { offload_done_event_.reset(); }
        }
    }

private:
    long id_;
    std::vector<int64_t> shape_;
    at::ScalarType ds_dtype_;
    at::Tensor ds_tensor_;
    at::Tensor ds_reload_tensor_;
    at::Tensor grad_buffer_;
    bool partitioned_;
    int64_t offset_;   // for Z1
    bool persistent_;  // for Z3
    mutable bool is_reloaded = false;

    at::cuda::CUDAStream offload_stream_;
    at::cuda::CUDAStream reload_stream_;
    std::shared_ptr<at::cuda::CUDAEvent> comp_done_event_;
    std::shared_ptr<at::cuda::CUDAEvent> offload_done_event_;
    std::shared_ptr<at::cuda::CUDAEvent> reload_done_event_;
};

class DSParamRegistry {
public:
    DSParamRegistry() {}
    ~DSParamRegistry() {}

    void registerParam(long ds_id,
                       const std::vector<int64_t>& ds_shape,
                       at::Tensor ds_tensor,
                       at::Tensor grad_buffer,
                       bool partitioned,
                       int64_t offset,  // for Z1
                       bool persistent  // for Z3
    )
    {
        grad_buffer.zero_();
        params_.emplace(
            ds_id,
            DSParam(ds_id, ds_shape, ds_tensor, grad_buffer, partitioned, offset, persistent));
        valid_[ds_id] = false;
    }

    void registerGatheredParam(long ds_id, at::Tensor ds_tensor)
    {
        gathered_params_.emplace(ds_id, ds_tensor);
    }

    void unregisterGatheredParam(long ds_id)
    {
        assert(hasKey(gathered_params_, ds_id));
        gathered_params_.erase(ds_id);
        valid_[ds_id] = false;
    }

    const std::unordered_map<long, DSParam>& getParams() const { return params_; }

    const DSParam& getParam(long ds_id) const { return params_.at(ds_id); }
    const size_t getNumParams() const { return params_.size(); }
    const at::Tensor& getGatheredParam(long ds_id) const
    {
        assert(hasKey(gathered_params_, ds_id));
        return gathered_params_.at(ds_id);
    }
    bool hasGatheredParam(long ds_id) const { return hasKey(gathered_params_, ds_id); }
    void setPersistent(long ds_id, bool persistent) { params_.at(ds_id).setPersistent(persistent); }
    void offload(long ds_id) { params_.at(ds_id).offload(); }
    void reload(long ds_id) { params_.at(ds_id).reload(); }

    void setValid(long ds_id, bool valid) { valid_[ds_id] = valid; }
    bool isValid(long ds_id) const
    {
        assert(hasKey(valid_, ds_id));
        return valid_.at(ds_id);
    }

private:
    std::unordered_map<long, DSParam> params_;
    std::unordered_map<long, at::Tensor> gathered_params_;
    std::unordered_map<long, bool> valid_;
};

class CustomOpExecutor {
public:
    CustomOpExecutor(c10::intrusive_ptr<c10d::ProcessGroup> process_group,
                     std::shared_ptr<DSParamRegistry> param_registry,
                     std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets,
                     std::vector<long> ds_ids,
                     ncclComm_t nccl_comm,
                     at::cuda::CUDAStream rs_stream,
                     at::cuda::CUDAStream copy_stream,
                     bool pre_div_reduce)
        : process_group_(process_group),
          param_registry_(std::move(param_registry)),
          reduce_buckets_(std::move(reduce_buckets)),
          ds_ids_(std::move(ds_ids)),
          nccl_comm_(nccl_comm),
          rs_stream_(rs_stream),
          copy_stream_(copy_stream),
          pre_div_reduce_(pre_div_reduce)
    {
        for (long ds_id : ds_ids_) {
            has_acc_grad_[ds_id] = false;

            rs_comp_done_events_[ds_id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            rs_copy_done_events_[ds_id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
        }
        reduce_counter_ = ds_ids_.size();
    }
    ~CustomOpExecutor() {}

    virtual void startForward() {}

    virtual void endForward() {}

    virtual void startBackward(bool update) { param_updated_ = update; }

    virtual void endBackward()
    {
        flushAllReduceBuckets();

        // This synchronization ensures all of reduce calls are done before optimizer's step.
        at::cuda::stream_synchronize(rs_stream_);
    }

    virtual at::Tensor reduceGrad(at::Tensor grad_tensor, long ds_id)
    {
        int world_size = process_group_->getSize();
        const DSParam& param = param_registry_->getParam(ds_id);
        const auto scalar_type = grad_tensor.scalar_type();
        std::shared_ptr<ReduceBucket> reduce_bucket = reduce_buckets_->getBuffer(scalar_type);

        auto comp_stream = at::cuda::getCurrentCUDAStream();

        if (reduce_bucket->shouldFlush(grad_tensor.numel())) {
            int rank = process_group_->getRank();

            flushReduceBucket(scalar_type);

            // reduce_bucket is swapped in flushReduceBucket if double buffering is enabled
            reduce_bucket = reduce_buckets_->getBuffer(scalar_type);
        }

        if (grad_tensor.numel() > reduce_bucket->getSize()) {
            // extend buckets
            at::cuda::stream_synchronize(rs_stream_);
            reduce_bucket->reserve(grad_tensor.numel());
        }

        at::Tensor reduce_in_buffer = reduce_bucket->allocate(grad_tensor.numel());

        // This ensures the order of reduce_scatter -> copy
        // Without this block, copy may start while reduce_scatter is still running
        reduce_buckets_->getEvent(scalar_type)->block(comp_stream);
        auto copy_src = grad_tensor.contiguous().view({-1}).detach();
        // keep references to copy src
        reduce_tasks_[scalar_type].emplace_back(ds_id, copy_src, reduce_in_buffer);

        // computation must be done before copy
        rs_comp_done_events_[ds_id]->record(comp_stream);
        rs_comp_done_events_[ds_id]->block(copy_stream_);
        {
            at::cuda::CUDAStreamGuard guard(copy_stream_);
            reduce_in_buffer.copy_(copy_src, true);
            rs_copy_done_events_[ds_id]->record(copy_stream_);
        }

        return at::Tensor();
    }

    bool hasParam(long ds_id) const { return hasKey(has_acc_grad_, ds_id); }

protected:
    c10::intrusive_ptr<c10d::ProcessGroup> process_group_;
    std::shared_ptr<DSParamRegistry> param_registry_;
    std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets_;
    std::vector<long> ds_ids_;
    ncclComm_t nccl_comm_;
    at::cuda::CUDAStream rs_stream_;
    at::cuda::CUDAStream copy_stream_;

    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> rs_comp_done_events_;
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> rs_copy_done_events_;

    size_t reduce_counter_ = 0;
    bool param_updated_ = false;
    std::unordered_map<at::ScalarType, std::vector<ReduceTask>> reduce_tasks_;
    std::unordered_map<long, bool> has_acc_grad_;
    bool pre_div_reduce_;

    virtual void flushReduceBucket(at::ScalarType scalar_type) = 0;

    void flushAllReduceBuckets()
    {
        for (const auto& it : reduce_tasks_) { flushReduceBucket(it.first); }
    }

    // Common helper methods for flushReduceBucket implementations
    void blockCopyEvents(at::ScalarType scalar_type)
    {
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            auto copy_done_event = rs_copy_done_events_.at(t.getDSId());
            copy_done_event->block(rs_stream_);
        }
    }

    void applyPreDivision(at::ScalarType scalar_type)
    {
        if (pre_div_reduce_) {
            at::cuda::CUDAStreamGuard guard(rs_stream_);
            for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
                t.getSendBuf().div_(process_group_->getSize());
            }
        }
    }

    ncclRedOp_t getReductionOp() const { return pre_div_reduce_ ? ncclSum : ncclAvg; }

    void performCleanup(at::ScalarType scalar_type)
    {
        reduce_buckets_->swap(scalar_type, rs_stream_, copy_stream_);

        // Prevent grad tensor from being released before the copy is done
        auto comp_stream = at::cuda::getCurrentCUDAStream();
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            auto copy_done_event = rs_copy_done_events_.at(t.getDSId());
            copy_done_event->block(comp_stream);
        }
        reduce_tasks_[scalar_type].clear();
    }
};

template <typename T, typename U>
std::shared_ptr<T> getExecutor(long graph_id,
                               const std::unordered_map<long, std::shared_ptr<U>>& executors)
{
    assert(hasKey(executors, graph_id));
    if (auto executor = std::dynamic_pointer_cast<T>(executors.at(graph_id))) { return executor; }
    throw std::runtime_error("Invalid executor type");
}

extern std::shared_ptr<DSParamRegistry> param_registry;
extern std::unordered_map<long, std::shared_ptr<CustomOpExecutor>> executors;
extern std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets;

at::Tensor reduce_grad(at::Tensor grad_tensor, long graph_id, long ds_id);
at::Tensor reduce_grad_meta(at::Tensor grad_tensor, long graph_id, long ds_id);
void free_tensors(std::vector<at::Tensor> tensors);
void free_tensors_meta(std::vector<at::Tensor> tensors);

void init(c10::intrusive_ptr<c10d::ProcessGroup> pg,
          pybind11::object& config,
          int64_t initial_reduce_bucket_size);
void reset();
void cleanup();

void start_forward();
void end_forward();
void start_backward(bool update);

}  // namespace dc
