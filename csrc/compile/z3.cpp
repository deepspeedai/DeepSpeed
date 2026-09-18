// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "z3.h"
#include "deepcompile.h"

#include <ATen/native/cuda/Resize.h>

namespace dc {

const size_t TIMEOUT_SYMMETRIC_MEMORY_BARRIER = 60000;

struct GatherArenaPlanEntry {
    long occurrence;
    int64_t offset_bytes;
    int64_t nbytes;
    at::ScalarType dtype;
};

struct GatherArenaLease {
    long occurrence;
    int64_t offset_bytes;
    int64_t nbytes;
};

struct GatherArenaConfig {
    int64_t capacity_bytes = 0;
    int64_t alignment = 256;
    std::string signature;
    std::unordered_map<long, std::vector<GatherArenaPlanEntry>> plan;
};

class Z3CustomOpExecutor : public CustomOpExecutor {
public:
    Z3CustomOpExecutor(c10::intrusive_ptr<c10d::ProcessGroup> process_group,
                       std::shared_ptr<DSParamRegistry> param_registry,
                       std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets,
                       std::vector<long> ds_ids,
                       ncclComm_t nccl_comm,
                       at::cuda::CUDAStream ag_stream,
                       at::cuda::CUDAStream rs_stream,
                       at::cuda::CUDAStream copy_stream,
                       at::cuda::CUDAStream offload_stream,
                       at::cuda::CUDAStream reload_stream,
                       bool pre_div_reduce)
        : CustomOpExecutor(process_group,
                           param_registry,
                           reduce_buckets,
                           ds_ids,
                           nccl_comm,
                           rs_stream,
                           copy_stream,
                           pre_div_reduce),
          ag_stream_(ag_stream),
          offload_stream_(offload_stream),
          reload_stream_(reload_stream)
    {
        for (long ds_id : ds_ids_) {
            ag_comm_done_events_[ds_id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            ag_comp_done_events_[ds_id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);

            param_use_count_[ds_id] = 0;
        }
        // Graph registration is performed by the forward compiler, which may
        // create this executor after the engine's forward prologue has run.
        active_gather_arena_config_ = &forward_gather_arena_config_;
    }
    ~Z3CustomOpExecutor() {}

    void startForward() override { startGatherArenaCycle(false); }

    void endForward() override { endGatherArenaCycle(); }

    void startBackward(bool update) override
    {
        CustomOpExecutor::startBackward(update);
        startGatherArenaCycle(true);
    }

    void endBackward() override
    {
        CustomOpExecutor::endBackward();

        if (param_updated_) {
            for (auto& it : has_acc_grad_) {
                it.second = false;
                param_registry_->setValid(it.first, false);
            }
        }

        for (auto& it : reload_buffers_) {
            it.second.record_stream(at::cuda::getCurrentCUDAStream());
        }
        reload_buffers_.clear();
    }

    void endBackwardPhase() override { endGatherArenaCycle(); }

    void configureGatherArena(bool bwd,
                              int64_t capacity_bytes,
                              int64_t alignment,
                              const std::vector<long>& ds_ids,
                              const std::vector<long>& occurrences,
                              const std::vector<int64_t>& offsets,
                              const std::vector<int64_t>& nbytes,
                              const std::vector<at::ScalarType>& dtypes,
                              const std::string& signature)
    {
        const size_t size = ds_ids.size();
        if (occurrences.size() != size || offsets.size() != size || nbytes.size() != size ||
            dtypes.size() != size) {
            throw std::runtime_error("ZeRO-3 gather arena plan vectors have different sizes");
        }
        if (capacity_bytes < 0 || alignment <= 0 || (alignment & (alignment - 1)) != 0) {
            throw std::runtime_error("ZeRO-3 gather arena has invalid capacity or alignment");
        }
        if (!gather_arena_active_leases_.empty()) {
            throw std::runtime_error("Cannot reconfigure ZeRO-3 gather arena with active leases");
        }

        GatherArenaConfig& config = bwd ? backward_gather_arena_config_
                                        : forward_gather_arena_config_;
        if (active_gather_arena_config_ == &config && gather_arena_backing_.defined()) {
            throw std::runtime_error(
                "Cannot reconfigure active ZeRO-3 gather arena after backing allocation");
        }

        config = GatherArenaConfig();
        config.capacity_bytes = capacity_bytes;
        config.alignment = alignment;
        config.signature = signature;
        if (active_gather_arena_config_ == &config) { gather_arena_occurrence_cursors_.clear(); }

        std::unordered_map<long, long> next_occurrence;
        for (size_t i = 0; i < size; ++i) {
            const long ds_id = ds_ids[i];
            if (occurrences[i] != next_occurrence[ds_id]++) {
                throw std::runtime_error("ZeRO-3 gather arena occurrences are not contiguous");
            }
            if (nbytes[i] < 0) {
                throw std::runtime_error("ZeRO-3 gather arena occurrence has negative bytes");
            }
            if (offsets[i] >= 0 &&
                (offsets[i] % alignment != 0 || offsets[i] + nbytes[i] > capacity_bytes)) {
                throw std::runtime_error(
                    "ZeRO-3 gather arena slice is misaligned or over capacity");
            }
            config.plan[ds_id].push_back({occurrences[i], offsets[i], nbytes[i], dtypes[i]});
        }
    }

    void launchAllGather(at::Tensor output_buf,
                         long ds_id,
                         c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem)
    {
        const DSParam& param = param_registry_->getParam(ds_id);
        at::Tensor ds_tensor = param.getDSTensor();

        if (ds_tensor.scalar_type() != output_buf.scalar_type()) {
            at::cuda::CUDAStreamGuard guard(ag_stream_);
            ds_tensor = ds_tensor.to(output_buf.scalar_type(), true, true);
        }

        if (symm_mem == nullptr) {
            // Fast path: assume uniform shard sizes (ZeRO-3 partitions are padded to uniform size)
            const int world_size = process_group_->getSize();
            const int64_t shard_elems = ds_tensor.numel();

            // Perform all-gather directly into the pre-allocated padded output buffer
            // NCCL requires contiguous storage; use .contiguous() explicitly
            ncclResult_t result = ncclAllGather(ds_tensor.contiguous().data_ptr(),
                                                output_buf.data_ptr(),
                                                shard_elems,
                                                get_nccl_data_type(ds_tensor.scalar_type()),
                                                nccl_comm_,
                                                ag_stream_);

            if (result != ncclSuccess) { throw std::runtime_error("NCCL AllGather failed"); }
        } else {
            at::cuda::CUDAStreamGuard guard(ag_stream_);
            int world_size = process_group_->getSize();
            int rank = process_group_->getRank();

            at::Tensor local_buf =
                symm_mem->get_buffer(rank, ds_tensor.sizes(), ds_tensor.scalar_type(), 0);
            local_buf.copy_(ds_tensor, true);

            symm_mem->barrier(0, TIMEOUT_SYMMETRIC_MEMORY_BARRIER);
            auto chunks = output_buf.flatten().chunk(world_size);
            for (int step = 0; step < world_size; step++) {
                int remote_rank = (rank - step + world_size) % world_size;
                auto src_buf = symm_mem->get_buffer(
                    remote_rank, ds_tensor.sizes(), ds_tensor.scalar_type(), 0);
                chunks[remote_rank].copy_(src_buf.flatten(), true);
            }
            symm_mem->barrier(0, TIMEOUT_SYMMETRIC_MEMORY_BARRIER);
        }

        param_registry_->registerGatheredParam(ds_id, output_buf);
        param_registry_->setValid(ds_id, true);
    }

    at::Tensor allgatherParam(long ds_id,
                              std::optional<at::ScalarType> dtype,
                              c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem)
    {
        const DSParam& param = param_registry_->getParam(ds_id);
        const at::Tensor& ds_tensor = param.getDSTensor();
        const int world_size = process_group_->getSize();
        const int64_t true_numel = static_cast<int64_t>(productDim(param.getShape()));
        const int64_t padded_per_rank = (true_numel + world_size - 1) / world_size;
        const int64_t padded_numel = static_cast<int64_t>(world_size) * padded_per_rank;
        at::ScalarType target_dtype = dtype ? dtype.value() : ds_tensor.scalar_type();

        if (param_registry_->isValid(ds_id)) {
            // Return a view sliced to the true size with the original shape
            //
            // Persistent params are gathered in their original dtype which may
            // be different from the requested.
            auto base = param_registry_->getGatheredParam(ds_id);
            return base.flatten()
                .to(target_dtype)
                .index({torch::indexing::Slice(0, true_numel)})
                .view(param.getShape());
        }

        at::Tensor output_buf;
        if (param_registry_->hasGatheredParam(ds_id)) {
            auto existing = param_registry_->getGatheredParam(ds_id);
            if (existing.defined() && existing.numel() == padded_numel) { output_buf = existing; }
        }
        if (!output_buf.defined()) {
            output_buf = acquireGatherArenaSlice(ds_id, padded_numel, target_dtype);
            if (!output_buf.defined()) {
                at::cuda::CUDAStreamGuard guard(ag_stream_);
                output_buf = torch::empty({padded_numel}, ds_tensor.options().dtype(target_dtype));
            }
        }

        assert(hasKey(ag_comp_done_events_, ds_id));
        ag_comp_done_events_[ds_id]->record();
        ag_comp_done_events_[ds_id]->block(ag_stream_);

        launchAllGather(output_buf, ds_id, symm_mem);

        ag_comm_done_events_[ds_id]->record(ag_stream_);
        // Return a view of the gathered padded buffer matching the true param shape
        return output_buf.flatten()
            .index({torch::indexing::Slice(0, true_numel)})
            .view(param.getShape());
    }

    std::vector<at::Tensor> prefetchParamsFused(
        const std::vector<long>& ds_ids,
        const std::optional<std::vector<at::ScalarType>> dtypes,
        c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem)
    {
        std::vector<std::tuple<long, std::optional<at::ScalarType>>> invalid_params;
        for (int i = 0; i < ds_ids.size(); i++) {
            if (!param_registry_->isValid(ds_ids[i])) {
                auto dtype = dtypes ? dtypes.value()[i] : std::optional<at::ScalarType>();
                invalid_params.push_back(std::make_tuple(ds_ids[i], dtype));
            }
        }

        std::unordered_map<long, at::Tensor> output_bufs;
        for (const auto& [ds_id, dtype] : invalid_params) {
            const DSParam& param = param_registry_->getParam(ds_id);
            const at::Tensor& ds_tensor = param.getDSTensor();
            const int world_size = process_group_->getSize();
            const int64_t shard_elems = ds_tensor.numel();
            const int64_t padded_numel = static_cast<int64_t>(world_size) * shard_elems;

            if (param_registry_->hasGatheredParam(ds_id)) {
                auto existing = param_registry_->getGatheredParam(ds_id);
                if (existing.defined() && existing.numel() == padded_numel) {
                    output_bufs[ds_id] = existing;
                    continue;
                }
            }
            auto target_dtype = dtype ? dtype.value() : ds_tensor.scalar_type();
            output_bufs[ds_id] = acquireGatherArenaSlice(ds_id, padded_numel, target_dtype);
            if (!output_bufs[ds_id].defined()) {
                at::cuda::CUDAStreamGuard guard(ag_stream_);
                output_bufs[ds_id] =
                    torch::empty({padded_numel}, ds_tensor.options().dtype(target_dtype));
            }
        }

        for (const auto& [ds_id, _] : invalid_params) {
            ag_comp_done_events_[ds_id]->record();
            ag_comp_done_events_[ds_id]->block(ag_stream_);
        }

        ncclGroupStart();
        for (const auto& [ds_id, _] : invalid_params) {
            assert(hasKey(output_bufs, ds_id));
            launchAllGather(output_bufs.at(ds_id), ds_id, symm_mem);
        }
        ncclGroupEnd();

        for (const auto& [ds_id, _] : invalid_params) {
            ag_comm_done_events_[ds_id]->record(ag_stream_);
        }

        std::vector<at::Tensor> outputs;
        outputs.reserve(ds_ids.size());
        for (int i = 0; i < ds_ids.size(); i++) {
            const long ds_id = ds_ids[i];
            const DSParam& param = param_registry_->getParam(ds_id);
            const auto target_dtype = dtypes ? dtypes.value()[i]
                                             : param.getDSTensor().scalar_type();
            const int64_t true_numel = static_cast<int64_t>(productDim(param.getShape()));
            auto gathered = param_registry_->getGatheredParam(ds_id);
            outputs.push_back(gathered.flatten()
                                  .to(target_dtype)
                                  .index({torch::indexing::Slice(0, true_numel)})
                                  .view(param.getShape()));
        }
        return outputs;
    }

    void releaseParam(long ds_id, long n_users)
    {
        const DSParam& param = param_registry_->getParam(ds_id);

        assert(hasKey(param_use_count_, ds_id));
        if (param_use_count_[ds_id] == 0) { param_use_count_[ds_id] = n_users; }
        param_use_count_[ds_id]--;

        if (param_use_count_[ds_id] == 0 && !param.isPersistent()) {
            at::Tensor gathered_param = param_registry_->getGatheredParam(ds_id);

            if (gathered_param.defined()) {  // gathered param is undefined while profiling
                auto storage = gathered_param.storage();
                auto arena_lease = gather_arena_active_leases_.find(ds_id);
                if (arena_lease != gather_arena_active_leases_.end()) {
                    if (!gather_arena_backing_.defined() ||
                        storage.unsafeGetStorageImpl() !=
                            gather_arena_backing_.storage().unsafeGetStorageImpl()) {
                        throw std::runtime_error(
                            "ZeRO-3 gather arena lease storage identity changed before release");
                    }
                    const auto& lease = arena_lease->second;
                    const auto expected_ptr =
                        static_cast<const char*>(gather_arena_backing_.data_ptr()) +
                        lease.offset_bytes;
                    if (gathered_param.data_ptr() != expected_ptr ||
                        gathered_param.numel() * gathered_param.element_size() != lease.nbytes) {
                        throw std::runtime_error(
                            "ZeRO-3 gather arena lease view changed before release");
                    }
                    gathered_param.record_stream(at::cuda::getCurrentCUDAStream());
                    gather_arena_active_leases_.erase(arena_lease);
                } else if (storage.nbytes() > 0) {
                    // Required so the caching allocator defers reuse for consumer-stream kernels
                    // queued behind wait_allgather.
                    gathered_param.record_stream(at::cuda::getCurrentCUDAStream());
                    at::native::resize_bytes_cuda(storage.unsafeGetStorageImpl(), 0);
                }

                const auto options = gathered_param.options();
                at::Tensor empty_buffer = torch::empty({0}, options);
                gathered_param.set_data(empty_buffer);
            }

            param_registry_->unregisterGatheredParam(ds_id);
        }
    }

    at::Tensor waitAllgather(at::Tensor v, long ds_id)
    {
        assert(hasKey(ag_comm_done_events_, ds_id));
        ag_comm_done_events_[ds_id]->block(at::cuda::getCurrentCUDAStream());
        return v;
    }

    void flushReduceBucket(at::ScalarType scalar_type) override
    {
        if (!hasKey(reduce_tasks_, scalar_type)) { return; }

        blockCopyEvents(scalar_type);

        // Calculate temporary buffer size for accumulated gradients or
        // communication/storage dtype mismatches.
        int64_t tmp_recv_numel = 0;
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            auto recv_buf = param_registry_->getParam(t.getDSId()).getGradBuffer();
            int64_t recv_numel = recv_buf.numel();
            bool use_tmp_recv = recv_numel > 0 && (has_acc_grad_.at(t.getDSId()) ||
                                                   recv_buf.scalar_type() != scalar_type);
            if (use_tmp_recv) { tmp_recv_numel += recv_numel; }
        }

        // Allocate temporary buffer if needed
        at::Tensor tmp_recv_buf = at::Tensor();
        if (tmp_recv_numel > 0) {
            at::cuda::CUDAStreamGuard guard(rs_stream_);
            tmp_recv_buf = torch::empty({tmp_recv_numel},
                                        at::TensorOptions().dtype(scalar_type).device(at::kCUDA));
        }

        applyPreDivision(scalar_type);

        // NCCL ReduceScatter operation
        ncclGroupStart();
        int64_t offset = 0;
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            auto recv_buf = param_registry_->getParam(t.getDSId()).getGradBuffer();
            bool acc_grad = has_acc_grad_.at(t.getDSId());
            int64_t recv_numel = recv_buf.numel();
            bool use_tmp_recv =
                recv_numel > 0 && (acc_grad || recv_buf.scalar_type() != scalar_type);

            if (use_tmp_recv) {
                recv_buf =
                    tmp_recv_buf.index({torch::indexing::Slice(offset, offset + recv_numel)});
            }

            ncclResult_t result = ncclReduceScatter(t.getSendBuf().data_ptr(),
                                                    recv_buf.data_ptr(),
                                                    recv_numel,
                                                    get_nccl_data_type(scalar_type),
                                                    getReductionOp(),
                                                    nccl_comm_,
                                                    rs_stream_);
            if (result != ncclSuccess) { throw std::runtime_error("NCCL ReduceScatter failed"); }

            if (use_tmp_recv) { offset += recv_numel; }
        }
        ncclGroupEnd();

        // Move temporary receive results into the ZeRO grad buffer.
        {
            at::cuda::CUDAStreamGuard guard(rs_stream_);
            int64_t offset = 0;
            for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
                auto recv_buf = param_registry_->getParam(t.getDSId()).getGradBuffer();
                bool acc_grad = has_acc_grad_.at(t.getDSId());
                int64_t recv_numel = recv_buf.numel();
                bool use_tmp_recv =
                    recv_numel > 0 && (acc_grad || recv_buf.scalar_type() != scalar_type);

                if (use_tmp_recv) {
                    auto reduced_slice =
                        tmp_recv_buf.index({torch::indexing::Slice(offset, offset + recv_numel)});
                    if (reduced_slice.scalar_type() != recv_buf.scalar_type()) {
                        reduced_slice = reduced_slice.to(recv_buf.scalar_type());
                    }
                    if (acc_grad) {
                        recv_buf.add_(reduced_slice);
                    } else {
                        recv_buf.copy_(reduced_slice, true);
                    }
                    offset += recv_numel;
                }
                has_acc_grad_[t.getDSId()] = true;
            }
        }

        performCleanup(scalar_type);

        // Record stream for temporary buffer to prevent early deallocation
        if (tmp_recv_numel > 0) { tmp_recv_buf.record_stream(rs_stream_); }
    }

    at::Tensor offloadTensor(at::Tensor tensor, long id)
    {
        if (!hasKey(offload_events_, id)) {
            offload_events_[id] = std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            offload_comp_done_events_[id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);

            // Pin by default so async D2H/H2D can use full PCIe bandwidth; controlled by
            // compile.offload_activation_pin_memory.
            const auto options = at::TensorOptions()
                                     .pinned_memory(offload_activation_pin_memory)
                                     .device(torch::kCPU);
            offload_buffers_[id] = at::empty_like(tensor, options);
        }

        offload_comp_done_events_[id]->record();
        offload_comp_done_events_[id]->block(offload_stream_);
        {
            at::cuda::CUDAStreamGuard guard(offload_stream_);
            offload_buffers_.at(id).copy_(tensor, true);
        }

        tensor.record_stream(offload_stream_);

        offload_events_[id]->record(offload_stream_);
        assert(hasKey(offload_buffers_, id));
        return offload_buffers_.at(id);
    }

    at::Tensor reloadTensor(at::Tensor tensor, long id)
    {
        if (!hasKey(reload_events_, id)) {
            reload_events_[id] = std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
        }

        assert(hasKey(offload_buffers_, id));
        offload_events_[id]->block(reload_stream_);

        at::Tensor ten;
        {
            at::cuda::CUDAStreamGuard guard(reload_stream_);

            assert(hasKey(offload_buffers_, id));
            at::Tensor buf = offload_buffers_.at(id);
            const auto options = at::TensorOptions().device(torch::kCUDA);
            ten = at::empty_like(buf, options);
            ten.copy_(buf, true);

            reload_buffers_[id] = ten;
        }

        reload_events_[id]->record(reload_stream_);
        return ten;
    }

    at::Tensor waitOffload(at::Tensor tensor, long id)
    {
        assert(hasKey(offload_events_, id));
        offload_events_[id]->block(at::cuda::getCurrentCUDAStream());

        assert(hasKey(offload_buffers_, id));
        return offload_buffers_.at(id);
    }

    at::Tensor waitReload(at::Tensor tensor, long id)
    {
        assert(hasKey(reload_events_, id));
        reload_events_[id]->block(at::cuda::getCurrentCUDAStream());

        assert(hasKey(reload_buffers_, id));
        auto ten = reload_buffers_.at(id);

        // We can't release here because the tensor is still being used
        // We will need "freeReloadedTensor" after the last user of the tensor to call
        // ".record_stream". As it is a bit complicated, we clear the buffer and do at the end of
        // the backward pass for now. reload_buffers_.erase(id);
        return ten;
    }

    void offloadParameter(at::Tensor tensor, long ds_id) { param_registry_->offload(ds_id); }
    void reloadParameter(at::Tensor tensor, long ds_id) { param_registry_->reload(ds_id); }

    bool hasReloadBuffer(long id) { return hasKey(reload_buffers_, id); }

    bool hasParam(long ds_id) const { return hasKey(has_acc_grad_, ds_id); }

private:
    void startGatherArenaCycle(bool bwd)
    {
        if (!gather_arena_active_leases_.empty()) {
            throw std::runtime_error(
                "ZeRO-3 gather arena reached lifecycle boundary with active leases");
        }
        GatherArenaConfig* next_config = bwd ? &backward_gather_arena_config_
                                             : &forward_gather_arena_config_;
        if (active_gather_arena_config_ != next_config && gather_arena_backing_.defined()) {
            throw std::runtime_error("ZeRO-3 gather arena changed phase before backing teardown");
        }
        active_gather_arena_config_ = next_config;
        gather_arena_occurrence_cursors_.clear();
    }

    void endGatherArenaCycle()
    {
        if (!gather_arena_active_leases_.empty()) {
            throw std::runtime_error(
                "ZeRO-3 gather arena reached lifecycle boundary with active leases");
        }
        gather_arena_occurrence_cursors_.clear();
        if (gather_arena_backing_.defined()) {
            // All planned views have been released. Record the executor's
            // consumer stream before dropping the last executor-owned handle.
            gather_arena_backing_.record_stream(at::cuda::getCurrentCUDAStream());
            gather_arena_backing_ = at::Tensor();
        }
        active_gather_arena_config_ = nullptr;
    }

    at::Tensor acquireGatherArenaSlice(long ds_id,
                                       int64_t padded_numel,
                                       at::ScalarType target_dtype)
    {
        const GatherArenaConfig* config = active_gather_arena_config_;
        if (profile || config == nullptr || config->capacity_bytes == 0) { return at::Tensor(); }
        auto plan = config->plan.find(ds_id);
        if (plan == config->plan.end() || plan->second.empty()) { return at::Tensor(); }
        if (hasKey(gather_arena_active_leases_, ds_id)) {
            throw std::runtime_error(
                "ZeRO-3 gather arena attempted overlapping leases for one parameter");
        }

        size_t& cursor = gather_arena_occurrence_cursors_[ds_id];
        if (cursor >= plan->second.size()) { return at::Tensor(); }
        const GatherArenaPlanEntry& entry = plan->second.at(cursor);
        cursor++;

        const DSParam& param = param_registry_->getParam(ds_id);
        if (param.isPersistent()) { return at::Tensor(); }
        if (entry.offset_bytes < 0) { return at::Tensor(); }

        const int64_t element_size = c10::elementSize(target_dtype);
        const int64_t actual_nbytes = padded_numel * element_size;
        if (entry.dtype != target_dtype || entry.nbytes != actual_nbytes ||
            entry.offset_bytes % element_size != 0 ||
            entry.offset_bytes + actual_nbytes > config->capacity_bytes) {
            return at::Tensor();
        }
        for (const auto& active : gather_arena_active_leases_) {
            const auto& lease = active.second;
            const bool disjoint = entry.offset_bytes + actual_nbytes <= lease.offset_bytes ||
                                  lease.offset_bytes + lease.nbytes <= entry.offset_bytes;
            if (!disjoint) { return at::Tensor(); }
        }

        const at::Tensor& ds_tensor = param.getDSTensor();
        if (gather_arena_backing_.defined() &&
            gather_arena_backing_.device() != ds_tensor.device()) {
            return at::Tensor();
        }
        if (!gather_arena_backing_.defined()) {
            at::cuda::CUDAStreamGuard guard(ag_stream_);
            gather_arena_backing_ =
                torch::empty({config->capacity_bytes}, ds_tensor.options().dtype(at::kByte));
        }

        at::Tensor output = torch::empty({0}, ds_tensor.options().dtype(target_dtype));
        output.set_(gather_arena_backing_.storage(),
                    entry.offset_bytes / element_size,
                    {padded_numel},
                    {1});
        if (output.storage().unsafeGetStorageImpl() !=
            gather_arena_backing_.storage().unsafeGetStorageImpl()) {
            throw std::runtime_error("ZeRO-3 gather arena failed to create a backing-storage view");
        }
        gather_arena_active_leases_[ds_id] = {entry.occurrence, entry.offset_bytes, actual_nbytes};
        return output;
    }

    at::cuda::CUDAStream ag_stream_;
    at::cuda::CUDAStream offload_stream_;
    at::cuda::CUDAStream reload_stream_;

    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> ag_comp_done_events_;
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> ag_comm_done_events_;

    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> offload_events_;
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> offload_comp_done_events_;
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> reload_events_;
    std::unordered_map<long, at::Tensor> offload_buffers_;
    std::unordered_map<long, at::Tensor> reload_buffers_;

    std::unordered_map<long, long> param_use_count_;

    GatherArenaConfig forward_gather_arena_config_;
    GatherArenaConfig backward_gather_arena_config_;
    GatherArenaConfig* active_gather_arena_config_ = nullptr;
    at::Tensor gather_arena_backing_;
    std::unordered_map<long, size_t> gather_arena_occurrence_cursors_;
    std::unordered_map<long, GatherArenaLease> gather_arena_active_leases_;
};

namespace {

at::cuda::CUDAStream get_ag_stream()
{
    static at::cuda::CUDAStream ag_stream = at::cuda::getStreamFromPool(true);
    return ag_stream;
}

at::cuda::CUDAStream get_rs_stream()
{
    static at::cuda::CUDAStream rs_stream = at::cuda::getStreamFromPool(true);
    return rs_stream;
}

at::cuda::CUDAStream get_copy_stream()
{
    static at::cuda::CUDAStream copy_stream = at::cuda::getStreamFromPool(true);
    return copy_stream;
}

at::cuda::CUDAStream get_offload_stream()
{
    static at::cuda::CUDAStream offload_stream = at::cuda::getStreamFromPool(true);
    return offload_stream;
}

at::cuda::CUDAStream get_reload_stream()
{
    static at::cuda::CUDAStream reload_stream = at::cuda::getStreamFromPool(true);
    return reload_stream;
}

}  // namespace

void register_graph_z3(long graph_id, const std::vector<long>& ds_ids)
{
    executors[graph_id] = std::make_shared<Z3CustomOpExecutor>(process_group,
                                                               param_registry,
                                                               reduce_buckets,
                                                               ds_ids,
                                                               nccl_comm,
                                                               get_ag_stream(),
                                                               get_rs_stream(),
                                                               get_copy_stream(),
                                                               get_offload_stream(),
                                                               get_reload_stream(),
                                                               pre_div_reduce);
}

void configure_z3_gather_arena(long graph_id,
                               bool bwd,
                               int64_t capacity_bytes,
                               int64_t alignment,
                               const std::vector<long>& ds_ids,
                               const std::vector<long>& occurrences,
                               const std::vector<int64_t>& offsets,
                               const std::vector<int64_t>& nbytes,
                               const std::vector<at::ScalarType>& dtypes,
                               const std::string& signature)
{
    auto executor = getExecutor<Z3CustomOpExecutor>(graph_id, executors);
    executor->configureGatherArena(
        bwd, capacity_bytes, alignment, ds_ids, occurrences, offsets, nbytes, dtypes, signature);
}

void register_z3_param(long ds_id,
                       const std::vector<int64_t>& ds_shape,
                       at::Tensor ds_tensor,
                       at::Tensor grad_buffer,
                       bool persistent,
                       std::optional<at::ScalarType> expected_grad_dtype)
{
    param_registry->registerParam(
        ds_id, ds_shape, ds_tensor, grad_buffer, true, 0, persistent, expected_grad_dtype);
    if (persistent) { param_registry->registerGatheredParam(ds_id, ds_tensor); }

    // Validate that padded shard sizes are uniform across ranks at registration time
    // DeepSpeed pads parameters to ensure even division, so we check the padded size
    // which should be uniform across all ranks for correct allgather behavior
    const int64_t local_count = ds_tensor.numel();
    const int world_size = process_group->getSize();

    // Calculate padded size (aligned to world_size)
    // Use ds_shape to compute the full (unpartitioned) parameter size
    int64_t total_numel = 1;
    for (const auto dim : ds_shape) { total_numel *= dim; }
    const int64_t padded_per_rank = (total_numel + world_size - 1) / world_size;

    // For verification: all ranks should have the same padded size
    auto count_options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA);
    at::Tensor local_padded_tensor = torch::tensor({padded_per_rank}, count_options);
    std::vector<at::Tensor> all_padded_counts(world_size);
    for (int i = 0; i < world_size; ++i) {
        all_padded_counts[i] = torch::empty_like(local_padded_tensor);
    }

    // Build lvalue buffers for output and input as required by ProcessGroup::allgather
    // The first argument must be a single-element vector containing a vector of WORLD_SIZE tensors
    std::vector<std::vector<at::Tensor>> output_tensors(1);
    output_tensors[0] = all_padded_counts;
    std::vector<at::Tensor> input_tensors = {local_padded_tensor};
    process_group->allgather(output_tensors, input_tensors)->wait();

    // Verify all ranks agree on the padded size
    for (int i = 0; i < world_size; ++i) {
        int64_t padded_count = all_padded_counts[i].to(torch::kCPU).item<int64_t>();
        if (padded_count != padded_per_rank) {
            throw std::runtime_error(
                "ZeRO-3 registration error: inconsistent padded shard sizes across ranks. "
                "This is an internal error - please report this issue.");
        }
    }
}

at::Tensor allgather_param(at::Tensor param_tensor,
                           long graph_id,
                           long ds_id,
                           std::optional<at::ScalarType> dtype)
{
    auto executor = getExecutor<Z3CustomOpExecutor>(graph_id, executors);

    if (sync_before_allgather) { c10::cuda::device_synchronize(); }
    auto ret = executor->allgatherParam(ds_id, dtype, symm_mem);
    if (sync_after_allgather) { c10::cuda::device_synchronize(); }
    return ret;
}

void set_persistent(long ds_id)
{
    param_registry->setPersistent(ds_id, true);

    // Allocate buffer here
    // Memory fragmentation will be more severe if we allocate in forward/backward
    for (auto& it : executors) {
        if (it.second->hasParam(ds_id)) {
            auto executor = getExecutor<Z3CustomOpExecutor>(it.first, executors);
            auto dtype = param_registry->getParam(ds_id).getDtype();
            executor->allgatherParam(ds_id, dtype, symm_mem);
        }
    }
}

std::vector<at::Tensor> prefetch_params_fused(
    long graph_id,
    const std::vector<at::Tensor>& params,
    const std::vector<long>& ds_ids,
    const std::optional<std::vector<at::ScalarType>>& dtypes)
{
    auto executor = getExecutor<Z3CustomOpExecutor>(graph_id, executors);
    return executor->prefetchParamsFused(ds_ids, dtypes, symm_mem);
}

std::vector<at::Tensor> prefetch_params_fused_meta(
    long graph_id,
    const std::vector<at::Tensor>& params,
    const std::vector<long>& ds_ids,
    const std::optional<std::vector<at::ScalarType>>& dtypes)
{
    std::vector<at::Tensor> outputs;
    outputs.reserve(ds_ids.size());
    for (int i = 0; i < ds_ids.size(); i++) {
        auto dtype = dtypes ? std::optional<at::ScalarType>(dtypes.value()[i])
                            : std::optional<at::ScalarType>();
        outputs.push_back(allgather_param_meta(params[i], graph_id, ds_ids[i], dtype));
    }
    return outputs;
}

// for profiling
void invalidate_gathered_param(long ds_id)
{
    const DSParam& param = param_registry->getParam(ds_id);
    if (param.isPersistent()) { return; }

    param_registry->unregisterGatheredParam(ds_id);
    param_registry->registerGatheredParam(ds_id, at::Tensor());
}

void clear_all_gathered_params()
{
    for (const auto& it : param_registry->getParams()) {
        long ds_id = it.first;
        const DSParam& param = param_registry->getParam(ds_id);
        if (param.isPersistent()) { continue; }
        if (param_registry->hasGatheredParam(ds_id)) {
            param_registry->unregisterGatheredParam(ds_id);
        }
    }
}

at::Tensor allgather_param_meta(at::Tensor param_tensor,
                                long graph_id,
                                long ds_id,
                                std::optional<at::ScalarType> dtype)
{
    const DSParam& param = param_registry->getParam(ds_id);
    auto options = param.getDSTensor().options().device(c10::kMeta);
    at::Tensor output_buf = torch::empty(param.getShape(), options.dtype(dtype));
    return output_buf;
}

at::Tensor release_param(at::Tensor dummy, long graph_id, long ds_id, long n_users)
{
    auto executor = getExecutor<Z3CustomOpExecutor>(graph_id, executors);
    executor->releaseParam(ds_id, n_users);
    return dummy;
}

at::Tensor release_param_meta(at::Tensor dummy, long graph_id, long ds_id, long n_users)
{
    return dummy;
}

at::Tensor wait_allgather(at::Tensor v, long graph_id, long ds_id)
{
    auto executor = getExecutor<Z3CustomOpExecutor>(graph_id, executors);
    executor->waitAllgather(v, ds_id);
    return v;
}

at::Tensor wait_allgather_meta(at::Tensor v, long graph_id, long ds_id) { return v; }

at::Tensor offload_tensor(at::Tensor tensor, long graph_id, long id)
{
    auto executor = getExecutor<Z3CustomOpExecutor>(graph_id, executors);
    return executor->offloadTensor(tensor, id);
}

at::Tensor reload_tensor(at::Tensor tensor, long graph_id, long id)
{
    auto executor = getExecutor<Z3CustomOpExecutor>(graph_id, executors);
    return executor->reloadTensor(tensor, id);
}

at::Tensor wait_offload(at::Tensor tensor, long graph_id, long id)
{
    auto executor = getExecutor<Z3CustomOpExecutor>(graph_id, executors);
    return executor->waitOffload(tensor, id);
}

at::Tensor wait_reload(at::Tensor tensor, long graph_id, long id)
{
    auto executor = getExecutor<Z3CustomOpExecutor>(graph_id, executors);
    if (profile && !executor->hasReloadBuffer(id)) { return tensor; }
    return executor->waitReload(tensor, id);
}

at::Tensor test_call(at::Tensor a)
{
    std::cout << "test_call" << std::endl;
    return a;
}

void reload_parameter(at::Tensor tensor, long graph_id, long ds_id)
{
    auto executor = getExecutor<Z3CustomOpExecutor>(graph_id, executors);
    executor->reloadParameter(tensor, ds_id);
}

void offload_parameter(at::Tensor tensor, long graph_id, long ds_id)
{
    auto executor = getExecutor<Z3CustomOpExecutor>(graph_id, executors);
    executor->offloadParameter(tensor, ds_id);
}
void reload_parameter_meta(at::Tensor param_tensor, long graph_id, long ds_id) {}
void offload_parameter_meta(at::Tensor tensor, long graph_id, long ds_id) {}

}  // namespace dc
