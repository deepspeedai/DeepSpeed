// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "z3.h"
#include "deepcompile.h"

#include <ATen/native/cuda/Resize.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <vector>

namespace dc {

const size_t TIMEOUT_SYMMETRIC_MEMORY_BARRIER = 60000;

namespace {

class GatherBufferPool {
public:
    enum class ReleaseDisposition { Retained, ResizeByCaller, Retired };

    at::Tensor acquire(int64_t numel,
                       at::ScalarType dtype,
                       const c10::Device& device,
                       at::cuda::CUDAStream stream)
    {
        flushCompletedPressureRecovery();
        observeAllocatorPressure(device.index());
        if (!enabled_ || pressure_recovery_in_progress_) { return at::Tensor(); }

        Entry* best = nullptr;
        for (auto& entry : entries_) {
            if (entry.checked_out || entry.buffer.scalar_type() != dtype ||
                entry.buffer.device() != device || entry.buffer.numel() < numel) {
                continue;
            }
            if (best == nullptr || entry.buffer.numel() < best->buffer.numel()) { best = &entry; }
        }
        if (best == nullptr) { return at::Tensor(); }

        best->ready_event->block(stream);
        at::Tensor result = best->buffer.narrow(0, 0, numel);
        best->checked_out = true;
        best->ready_event.reset();
        best->last_use = ++clock_;
        return result;
    }

    void excludeFromAdmission(const at::Tensor& buffer)
    {
        if (!buffer.defined() || buffer.storage().nbytes() == 0) { return; }
        non_retainable_storages_.insert(buffer.storage().unsafeGetStorageImpl());
    }

    void cancelAdmissionExclusion(const at::Tensor& buffer)
    {
        if (!buffer.defined()) { return; }
        non_retainable_storages_.erase(buffer.storage().unsafeGetStorageImpl());
    }

    void discard(const at::Tensor& buffer)
    {
        if (!buffer.defined()) { return; }

        auto storage = buffer.storage();
        auto storage_impl = storage.unsafeGetStorageImpl();
        non_retainable_storages_.erase(storage_impl);
        if (storage.nbytes() == 0) {
            removeEntryOwnership(storage_impl, "discarded", false);
            return;
        }

        buffer.record_stream(at::cuda::getCurrentCUDAStream());
        removeEntryOwnership(storage_impl, "discarded", false);
    }

    ReleaseDisposition release(const at::Tensor& buffer)
    {
        auto storage = buffer.storage();
        if (storage.nbytes() == 0) {
            auto storage_impl = storage.unsafeGetStorageImpl();
            non_retainable_storages_.erase(storage_impl);
            if (removeEntryOwnership(storage_impl, "released_zero_storage", false)) {
                return ReleaseDisposition::Retired;
            }
            return ReleaseDisposition::ResizeByCaller;
        }

        auto consumer_stream = at::cuda::getCurrentCUDAStream();
        buffer.record_stream(consumer_stream);
        observeAllocatorPressure(buffer.device().index());

        auto storage_impl = storage.unsafeGetStorageImpl();
        const bool excluded = non_retainable_storages_.erase(storage_impl) > 0;

        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (storageImpl(*it) != storage_impl) { continue; }
            if (excluded) { return retireEntry(it, storage_impl, "excluded", false); }
            if (pressure_recovery_in_progress_) {
                return retireEntry(it, storage_impl, "recovery_release", true);
            }
            if (!enabled_ || charged_bytes_ > budget_bytes_) {
                return retireEntry(it, storage_impl, "evicted_on_return", true);
            }
            auto ready_event = std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            ready_event->record(consumer_stream);
            it->ready_event = ready_event;
            it->checked_out = false;
            it->last_use = ++clock_;
            return ReleaseDisposition::Retained;
        }

        if (excluded || !enabled_ || pressure_recovery_in_progress_) {
            return ReleaseDisposition::ResizeByCaller;
        }

        const size_t capacity_bytes = storage.nbytes();
        if (pressure_recovery_complete_ &&
            !hasRecoveryAdmissionSlot(capacity_bytes, buffer.scalar_type(), buffer.device())) {
            return ReleaseDisposition::ResizeByCaller;
        }
        if (capacity_bytes > budget_bytes_ || !makeRoom(capacity_bytes)) {
            return ReleaseDisposition::ResizeByCaller;
        }

        const int64_t capacity_numel = static_cast<int64_t>(capacity_bytes / buffer.element_size());
        at::Tensor candidate = at::as_strided(buffer.detach(), {capacity_numel}, {1}, 0);
        auto ready_event = std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
        ready_event->record(consumer_stream);
        entries_.push_back({candidate, ready_event, false, ++clock_, capacity_bytes});
        charged_bytes_ += capacity_bytes;
        if (charged_bytes_ > high_water_bytes_) {
            high_water_bytes_ = charged_bytes_;
            logState("high_water");
        }
        return ReleaseDisposition::Retained;
    }

    void setBudgetForTest(int64_t budget_bytes)
    {
        test_override_ = true;
        enabled_ = budget_bytes > 0;
        budget_bytes_ = budget_bytes > 0 ? static_cast<size_t>(budget_bytes) : 0;
        baseline_retries_.reset();
        idle_pressure_score_ = 0;
        pressure_recovery_in_progress_ = false;
        pressure_recovery_complete_ = false;
        pressure_recovery_flush_pending_ = false;
        pressure_recovery_budget_bytes_ = 0;
        pressure_recovery_targets_.clear();
        makeRoom(0);
        logState("test_budget");
    }

    void reset()
    {
        entries_.clear();
        non_retainable_storages_.clear();
        baseline_retries_.reset();
        idle_pressure_score_ = 0;
        pressure_recovery_in_progress_ = false;
        pressure_recovery_complete_ = false;
        pressure_recovery_flush_pending_ = false;
        pressure_recovery_budget_bytes_ = 0;
        pressure_recovery_targets_.clear();
        budget_bytes_ = 0;
        charged_bytes_ = 0;
        high_water_bytes_ = 0;
        clock_ = 0;
        enabled_ = false;
        test_override_ = false;
        adaptive_budget_initialized_ = false;
    }

    void observeAllocatorPressureForTest(int64_t retries, int64_t free_bytes, int64_t total_bytes)
    {
        TORCH_CHECK(!test_override_,
                    "allocator-pressure simulation requires the production adaptive pool");
        TORCH_CHECK(retries >= 0 && free_bytes >= 0 && total_bytes >= 0,
                    "allocator-pressure simulation values must be nonnegative");
        updateBudgetForAllocatorPressure(
            retries, static_cast<size_t>(free_bytes), static_cast<size_t>(total_bytes));
    }

    std::vector<int64_t> stateForTest() const
    {
        size_t checked_out = 0;
        for (const auto& entry : entries_) { checked_out += entry.checked_out ? 1 : 0; }
        return {static_cast<int64_t>(budget_bytes_),
                static_cast<int64_t>(charged_bytes_),
                static_cast<int64_t>(high_water_bytes_),
                static_cast<int64_t>(entries_.size()),
                static_cast<int64_t>(checked_out),
                baseline_retries_.value_or(-1),
                enabled_ ? 1 : 0,
                adaptive_budget_initialized_ ? 1 : 0,
                idle_pressure_score_,
                pressure_recovery_complete_ ? 1 : 0,
                static_cast<int64_t>(pressure_recovery_budget_bytes_),
                static_cast<int64_t>(recoveryPendingEntries()),
                pressure_recovery_in_progress_ ? 1 : 0};
    }

private:
    struct Entry {
        at::Tensor buffer;
        std::shared_ptr<at::cuda::CUDAEvent> ready_event;
        bool checked_out;
        uint64_t last_use;
        size_t capacity_bytes;
    };

    struct RecoveryTarget {
        size_t capacity_bytes;
        at::ScalarType dtype;
        c10::Device device;
        uint64_t last_use;
    };

    static bool matchesRecoveryTarget(const Entry& entry, const RecoveryTarget& target)
    {
        return entry.capacity_bytes == target.capacity_bytes &&
               entry.buffer.scalar_type() == target.dtype && entry.buffer.device() == target.device;
    }

    bool hasRecoveryAdmissionSlot(size_t capacity_bytes,
                                  at::ScalarType dtype,
                                  const c10::Device& device) const
    {
        const size_t target_count =
            std::count_if(pressure_recovery_targets_.begin(),
                          pressure_recovery_targets_.end(),
                          [&](const RecoveryTarget& target) {
                              return capacity_bytes == target.capacity_bytes &&
                                     dtype == target.dtype && device == target.device;
                          });
        const size_t resident_count =
            std::count_if(entries_.begin(), entries_.end(), [&](const Entry& entry) {
                return entry.capacity_bytes == capacity_bytes &&
                       entry.buffer.scalar_type() == dtype && entry.buffer.device() == device;
            });
        return resident_count < target_count;
    }

    size_t recoveryPendingEntries() const
    {
        std::vector<bool> matched(pressure_recovery_targets_.size(), false);
        size_t resident_targets = 0;
        for (const auto& entry : entries_) {
            for (size_t i = 0; i < pressure_recovery_targets_.size(); ++i) {
                if (!matched[i] && matchesRecoveryTarget(entry, pressure_recovery_targets_[i])) {
                    matched[i] = true;
                    ++resident_targets;
                    break;
                }
            }
        }
        return pressure_recovery_targets_.size() - resident_targets;
    }

    static c10::StorageImpl* storageImpl(const Entry& entry)
    {
        return entry.buffer.storage().unsafeGetStorageImpl();
    }

    static bool recoveryTargetPriority(const RecoveryTarget& lhs, const RecoveryTarget& rhs)
    {
        if (lhs.capacity_bytes != rhs.capacity_bytes) {
            return lhs.capacity_bytes > rhs.capacity_bytes;
        }
        return lhs.last_use > rhs.last_use;
    }

    void boundRecoveryTargets(size_t hard_cap)
    {
        std::stable_sort(pressure_recovery_targets_.begin(),
                         pressure_recovery_targets_.end(),
                         recoveryTargetPriority);
        std::vector<RecoveryTarget> bounded_targets;
        bounded_targets.reserve(pressure_recovery_targets_.size());
        size_t bounded_bytes = 0;
        for (const auto& target : pressure_recovery_targets_) {
            if (target.capacity_bytes > hard_cap - bounded_bytes) { continue; }
            bounded_targets.push_back(target);
            bounded_bytes += target.capacity_bytes;
        }
        pressure_recovery_targets_ = std::move(bounded_targets);
        pressure_recovery_budget_bytes_ = bounded_bytes;
    }

    void dropRecoveryTarget(const Entry& entry)
    {
        auto target = std::find_if(pressure_recovery_targets_.begin(),
                                   pressure_recovery_targets_.end(),
                                   [&](const RecoveryTarget& candidate) {
                                       return candidate.last_use == entry.last_use &&
                                              matchesRecoveryTarget(entry, candidate);
                                   });
        if (target == pressure_recovery_targets_.end()) { return; }
        pressure_recovery_budget_bytes_ -= target->capacity_bytes;
        pressure_recovery_targets_.erase(target);
        budget_bytes_ = std::min(budget_bytes_, pressure_recovery_budget_bytes_);
        enabled_ = budget_bytes_ > 0;
    }

    ReleaseDisposition retireEntry(std::vector<Entry>::iterator entry,
                                   c10::StorageImpl* storage_impl,
                                   const char* event,
                                   bool keep_recovery_target)
    {
        // The final consumer stream was recorded before this method. Resize
        // must succeed before pool accounting forgets the live storage.
        at::native::resize_bytes_cuda(storage_impl, 0);
        if (pressure_recovery_in_progress_ && !keep_recovery_target) { dropRecoveryTarget(*entry); }
        charged_bytes_ -= entry->capacity_bytes;
        entries_.erase(entry);
        if (pressure_recovery_in_progress_ && entries_.empty()) {
            completePressureRecovery();
        } else {
            logState(event);
        }
        return ReleaseDisposition::Retired;
    }

    bool removeEntryOwnership(c10::StorageImpl* storage_impl,
                              const char* event,
                              bool keep_recovery_target)
    {
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (storageImpl(*it) != storage_impl) { continue; }
            if (pressure_recovery_in_progress_ && !keep_recovery_target) {
                dropRecoveryTarget(*it);
            }
            charged_bytes_ -= it->capacity_bytes;
            entries_.erase(it);
            if (pressure_recovery_in_progress_ && entries_.empty()) {
                completePressureRecovery();
            } else {
                logState(event);
            }
            return true;
        }
        return false;
    }

    void observeAllocatorPressure(c10::DeviceIndex device)
    {
        if (test_override_) { return; }

        const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (baseline_retries_ && stats.num_alloc_retries > baseline_retries_.value()) {
            C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        }
        updateBudgetForAllocatorPressure(stats.num_alloc_retries, free_bytes, total_bytes);
    }

    void updateBudgetForAllocatorPressure(int64_t retries, size_t free_bytes, size_t total_bytes)
    {
        if (!baseline_retries_) {
            baseline_retries_ = retries;
            idle_pressure_score_ = 0;
            return;
        }
        if (retries < baseline_retries_.value()) {
            baseline_retries_ = retries;
            idle_pressure_score_ = 0;
            return;
        }
        if (retries == baseline_retries_.value()) { return; }
        const int64_t retry_delta = retries - baseline_retries_.value();
        baseline_retries_ = retries;

        constexpr size_t granularity = 2 * 1024 * 1024;
        // Bound prefetch-expanded overlap while preserving headroom for large
        // non-gather allocations in the compiled graph. Allocator retries are
        // also the signal that idle cached gathers may need to yield memory.
        // Keep the byte-exact working set on isolated retries. On the first
        // sustained-pressure wave, stop new retention, release idle entries,
        // and retire checked-out entries as their final consumers return. Keep
        // a byte-exact, typed recovery plan, preserving the complete hot set
        // when it fits the hard cap. The lifecycle latch prevents later waves
        // from draining it again and recreating per-step churn.
        size_t hard_cap = total_bytes / 32;
        hard_cap -= hard_cap % granularity;
        size_t free_target = free_bytes / 4;
        free_target -= free_target % granularity;
        const size_t observed_budget = std::min(hard_cap, std::max(free_target, charged_bytes_));
        budget_bytes_ = adaptive_budget_initialized_ ? std::min(budget_bytes_, observed_budget)
                                                     : observed_budget;
        if (pressure_recovery_in_progress_ || pressure_recovery_complete_) {
            boundRecoveryTargets(hard_cap);
            budget_bytes_ = std::max(budget_bytes_, pressure_recovery_budget_bytes_);
        }
        adaptive_budget_initialized_ = true;
        enabled_ = budget_bytes_ > 0;
        idle_pressure_score_ =
            retry_delta > std::numeric_limits<int64_t>::max() - idle_pressure_score_
                ? std::numeric_limits<int64_t>::max()
                : idle_pressure_score_ + retry_delta;
        if (pressure_recovery_in_progress_) {
            idle_pressure_score_ = 0;
        } else if (!pressure_recovery_complete_ &&
                   idle_pressure_score_ >= kIdlePressureEvictionThreshold &&
                   charged_bytes_ >= budget_bytes_ && beginPressureRecovery(hard_cap)) {
            return;
        } else {
            const size_t charged_before_hard_cap = charged_bytes_;
            makeRoom(0);
            if (charged_bytes_ < charged_before_hard_cap) { idle_pressure_score_ = 0; }
        }
        logState(enabled_ ? "pressure" : "disabled");
    }

    bool beginPressureRecovery(size_t hard_cap)
    {
        if (entries_.empty()) { return false; }

        std::vector<RecoveryTarget> recovery_targets;
        recovery_targets.reserve(entries_.size());
        for (const auto& entry : entries_) {
            recovery_targets.push_back(RecoveryTarget{entry.capacity_bytes,
                                                      entry.buffer.scalar_type(),
                                                      entry.buffer.device(),
                                                      entry.last_use});
        }
        pressure_recovery_targets_ = std::move(recovery_targets);
        boundRecoveryTargets(hard_cap);

        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->checked_out) {
                ++it;
                continue;
            }
            // Relinquish the pool's active ownership before flushing free
            // blocks. Later gathers can still reuse allocator-cached storage.
            auto victim_storage = it->buffer.storage();
            at::native::resize_bytes_cuda(victim_storage.unsafeGetStorageImpl(), 0);
            if (non_retainable_storages_.erase(victim_storage.unsafeGetStorageImpl()) > 0) {
                dropRecoveryTarget(*it);
            }
            charged_bytes_ -= it->capacity_bytes;
            it = entries_.erase(it);
        }

        idle_pressure_score_ = 0;
        budget_bytes_ = pressure_recovery_budget_bytes_;
        enabled_ = budget_bytes_ > 0;
        pressure_recovery_in_progress_ = !entries_.empty();
        pressure_recovery_complete_ = false;
        if (entries_.empty()) {
            completePressureRecovery();
        } else {
            logState("recovering_pressure");
        }
        return true;
    }

    void completePressureRecovery()
    {
        pressure_recovery_in_progress_ = false;
        pressure_recovery_complete_ = true;
        pressure_recovery_flush_pending_ = true;
        idle_pressure_score_ = 0;
        budget_bytes_ = pressure_recovery_budget_bytes_;
        enabled_ = budget_bytes_ > 0;
        logState("recovered_pressure");
    }

    void flushCompletedPressureRecovery()
    {
        if (!pressure_recovery_flush_pending_) { return; }
        c10::cuda::CUDACachingAllocator::emptyCache();
        pressure_recovery_flush_pending_ = false;
        logState("flushed_recovered_pressure");
    }

    bool makeRoom(size_t capacity_bytes)
    {
        while (charged_bytes_ + capacity_bytes > budget_bytes_) {
            auto victim = entries_.end();
            for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                if (it->checked_out) { continue; }
                if (victim == entries_.end() || it->last_use < victim->last_use) { victim = it; }
            }
            if (victim == entries_.end()) { return false; }

            auto victim_storage = victim->buffer.storage();
            at::native::resize_bytes_cuda(victim_storage.unsafeGetStorageImpl(), 0);
            charged_bytes_ -= victim->capacity_bytes;
            entries_.erase(victim);
        }
        return true;
    }

    void logState(const char* event) const
    {
        if (std::getenv("DEEPSPEED_ALLOCATOR_TELEMETRY") == nullptr) { return; }
        size_t checked_out = 0;
        for (const auto& entry : entries_) { checked_out += entry.checked_out ? 1 : 0; }
        std::cout << "DEEPSPEED_Z3_GATHER_BUFFER_POOL event=" << event
                  << " budget_bytes=" << budget_bytes_ << " charged_bytes=" << charged_bytes_
                  << " high_water_bytes=" << high_water_bytes_ << " entries=" << entries_.size()
                  << " checked_out=" << checked_out
                  << " idle_pressure_score=" << idle_pressure_score_
                  << " pressure_recovery_in_progress=" << (pressure_recovery_in_progress_ ? 1 : 0)
                  << " pressure_recovery_complete=" << (pressure_recovery_complete_ ? 1 : 0)
                  << " pressure_recovery_budget_bytes=" << pressure_recovery_budget_bytes_
                  << " pressure_recovery_pending_entries=" << recoveryPendingEntries() << std::endl;
    }

    static constexpr int64_t kIdlePressureEvictionThreshold = 3;
    std::vector<Entry> entries_;
    std::unordered_set<c10::StorageImpl*> non_retainable_storages_;
    std::optional<int64_t> baseline_retries_;
    int64_t idle_pressure_score_ = 0;
    bool pressure_recovery_in_progress_ = false;
    bool pressure_recovery_complete_ = false;
    bool pressure_recovery_flush_pending_ = false;
    size_t pressure_recovery_budget_bytes_ = 0;
    std::vector<RecoveryTarget> pressure_recovery_targets_;
    size_t budget_bytes_ = 0;
    size_t charged_bytes_ = 0;
    size_t high_water_bytes_ = 0;
    uint64_t clock_ = 0;
    bool enabled_ = false;
    bool test_override_ = false;
    bool adaptive_budget_initialized_ = false;
};

class AdmissionExclusionRollback {
public:
    explicit AdmissionExclusionRollback(std::shared_ptr<GatherBufferPool> pool)
        : pool_(std::move(pool))
    {
    }

    AdmissionExclusionRollback(const AdmissionExclusionRollback&) = delete;
    AdmissionExclusionRollback& operator=(const AdmissionExclusionRollback&) = delete;

    ~AdmissionExclusionRollback() noexcept
    {
        if (committed_) { return; }
        for (const auto& buffer : buffers_) {
            try {
                pool_->cancelAdmissionExclusion(buffer);
            } catch (...) {
                // Destructors must not replace the original prefetch exception.
            }
        }
    }

    void exclude(const at::Tensor& buffer)
    {
        // Retain the Tensor before installing its raw StorageImpl identity so
        // allocation/event exceptions cannot leave an ABA-prone dangling key.
        buffers_.push_back(buffer);
        pool_->excludeFromAdmission(buffers_.back());
    }

    size_t size() const { return buffers_.size(); }

    void commit()
    {
        committed_ = true;
        buffers_.clear();
    }

private:
    std::shared_ptr<GatherBufferPool> pool_;
    std::vector<at::Tensor> buffers_;
    bool committed_ = false;
};

std::weak_ptr<GatherBufferPool> weak_gather_buffer_pool;
std::optional<int64_t> gather_buffer_pool_test_budget;
int64_t prefetch_fail_after_exclusions_for_test = 0;

std::shared_ptr<GatherBufferPool> get_gather_buffer_pool()
{
    auto pool = weak_gather_buffer_pool.lock();
    if (!pool) {
        pool = std::make_shared<GatherBufferPool>();
        weak_gather_buffer_pool = pool;
        if (gather_buffer_pool_test_budget) {
            pool->setBudgetForTest(gather_buffer_pool_test_budget.value());
        }
    }
    return pool;
}

}  // namespace

void set_z3_gather_buffer_pool_budget_for_test(int64_t budget_bytes)
{
    gather_buffer_pool_test_budget = budget_bytes;
    if (auto pool = weak_gather_buffer_pool.lock()) { pool->setBudgetForTest(budget_bytes); }
}

void update_z3_gather_buffer_pool_allocator_pressure_for_test(int64_t retries,
                                                              int64_t free_bytes,
                                                              int64_t total_bytes)
{
    get_gather_buffer_pool()->observeAllocatorPressureForTest(retries, free_bytes, total_bytes);
}

std::vector<int64_t> get_z3_gather_buffer_pool_state_for_test()
{
    return get_gather_buffer_pool()->stateForTest();
}

void set_z3_param_valid_for_test(long ds_id, bool valid) { param_registry->setValid(ds_id, valid); }

void set_z3_prefetch_fail_after_exclusions_for_test(int64_t count)
{
    TORCH_CHECK(count >= 0, "prefetch exclusion failure count must be nonnegative");
    prefetch_fail_after_exclusions_for_test = count;
}

void reset_z3_gather_buffer_pool()
{
    if (auto pool = weak_gather_buffer_pool.lock()) { pool->reset(); }
    weak_gather_buffer_pool.reset();
    gather_buffer_pool_test_budget.reset();
    prefetch_fail_after_exclusions_for_test = 0;
}

class Z3CustomOpExecutor : public CustomOpExecutor {
public:
    Z3CustomOpExecutor(c10::intrusive_ptr<c10d::ProcessGroup> process_group,
                       std::shared_ptr<DSParamRegistry> param_registry,
                       std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets,
                       std::shared_ptr<GatherBufferPool> gather_buffer_pool,
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
          gather_buffer_pool_(gather_buffer_pool),
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
    }
    ~Z3CustomOpExecutor() {}

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
            output_buf = gather_buffer_pool_->acquire(
                padded_numel, target_dtype, ds_tensor.device(), ag_stream_);
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

    void prefetchParamsFused(const std::vector<long>& ds_ids,
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
        AdmissionExclusionRollback admission_exclusions(gather_buffer_pool_);
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
                }
            }
            if (!hasKey(output_bufs, ds_id)) {
                auto target_dtype = dtype ? dtype.value() : ds_tensor.scalar_type();
                at::cuda::CUDAStreamGuard guard(ag_stream_);
                output_bufs[ds_id] =
                    torch::empty({padded_numel}, ds_tensor.options().dtype(target_dtype));
            }
            // Prefetch lifetimes are already controlled by the memory-aware scheduler.
            // Bind the exclusion to the selected storage so a stale ds_id generation
            // cannot admit a different demand-gather buffer into the independent pool.
            admission_exclusions.exclude(output_bufs.at(ds_id));
            if (prefetch_fail_after_exclusions_for_test > 0 &&
                admission_exclusions.size() >=
                    static_cast<size_t>(prefetch_fail_after_exclusions_for_test)) {
                throw std::runtime_error("injected prefetch preparation failure");
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
        admission_exclusions.commit();
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
                if (storage.nbytes() > 0) {
                    // Demand gathers may enter the byte-bounded pool. Prefetched gathers
                    // keep the scheduler's ownership boundary and use the original
                    // resize-to-zero path after their final consumer.
                    const auto release_disposition = gather_buffer_pool_->release(gathered_param);
                    if (release_disposition ==
                        GatherBufferPool::ReleaseDisposition::ResizeByCaller) {
                        at::native::resize_bytes_cuda(storage.unsafeGetStorageImpl(), 0);
                    }
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

            const auto options = at::TensorOptions().pinned_memory(true).device(torch::kCPU);
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
    std::shared_ptr<GatherBufferPool> gather_buffer_pool_;
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
                                                               get_gather_buffer_pool(),
                                                               ds_ids,
                                                               nccl_comm,
                                                               get_ag_stream(),
                                                               get_rs_stream(),
                                                               get_copy_stream(),
                                                               get_offload_stream(),
                                                               get_reload_stream(),
                                                               pre_div_reduce);
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
    auto gather_buffer_pool = get_gather_buffer_pool();
    bool gathered = false;
    for (auto& it : executors) {
        if (it.second->hasParam(ds_id)) {
            auto executor = getExecutor<Z3CustomOpExecutor>(it.first, executors);
            auto dtype = param_registry->getParam(ds_id).getDtype();
            executor->allgatherParam(ds_id, dtype, symm_mem);
            gathered = true;
        }
    }
    // Selective unsharding owns persistent storage for the remainder of the
    // compiled lifecycle. If the initial gather reused a demand-pool entry,
    // transfer that storage out of pool accounting without freeing it so
    // pressure recovery cannot wait forever for a release that persistent
    // parameters intentionally never issue.
    if (gathered) { gather_buffer_pool->discard(param_registry->getGatheredParam(ds_id)); }
}

void prefetch_params_fused(long graph_id,
                           const std::vector<at::Tensor>& params,
                           const std::vector<long>& ds_ids,
                           const std::optional<std::vector<at::ScalarType>>& dtypes)
{
    auto executor = getExecutor<Z3CustomOpExecutor>(graph_id, executors);
    executor->prefetchParamsFused(ds_ids, dtypes, symm_mem);
}

void prefetch_params_fused_meta(long graph_id,
                                const std::vector<at::Tensor>& params,
                                const std::vector<long>& ds_ids,
                                const std::optional<std::vector<at::ScalarType>>& dtypes)
{
}

// for profiling
void invalidate_gathered_param(long ds_id)
{
    const DSParam& param = param_registry->getParam(ds_id);
    if (param.isPersistent()) { return; }

    auto gathered_param = param_registry->getGatheredParam(ds_id);
    get_gather_buffer_pool()->discard(gathered_param);
    param_registry->unregisterGatheredParam(ds_id);
    param_registry->registerGatheredParam(ds_id, at::Tensor());
}

void clear_all_gathered_params()
{
    auto gather_buffer_pool = get_gather_buffer_pool();
    for (const auto& it : param_registry->getParams()) {
        long ds_id = it.first;
        const DSParam& param = param_registry->getParam(ds_id);
        if (param.isPersistent()) { continue; }
        if (param_registry->hasGatheredParam(ds_id)) {
            auto gathered_param = param_registry->getGatheredParam(ds_id);
            gather_buffer_pool->discard(gathered_param);
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
