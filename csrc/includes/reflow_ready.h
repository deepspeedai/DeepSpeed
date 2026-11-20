// SPDX-License-Identifier: Apache-2.0
// DeepSpeed Team

#pragma once

#include <c10/core/StreamGuard.h>
#include <pybind11/functional.h>
#include <torch/extension.h>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX512__) || defined(__AVX256__)
#include <immintrin.h>
#endif
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include "reflow_cpu_affinity.h"
#if defined(__linux__)
#include <sys/mman.h>
#endif

// A single optimizer worker lends its OpenMP team to consecutive independent Ready ranges.
// Each writer can enter the next chunk once it has fenced its own stores in this one.
class ReflowReadyWork {
public:
    using Work = std::shared_ptr<ReflowReadyWork>;
    using Kernel = std::function<void(size_t, size_t)>;

    ReflowReadyWork(at::Tensor destination, c10::Stream stream, int64_t chunk_elems)
        : destination_(destination.detach()), stream_(stream), chunk_elems_(chunk_elems)
    {
        TORCH_CHECK(chunk_elems > 0 && destination_.dim() == 1 && destination_.is_contiguous() &&
                    destination_.device() == stream.device());
    }

    void add(size_t offset,
             int threads,
             at::Tensor output,
             std::vector<at::Tensor> inputs,
             Kernel kernel)
    {
        std::lock_guard<std::mutex> lock(setup_mutex_);
        TORCH_CHECK(!sealed_, "Ready work has already started");
        const size_t size = output.numel();
        TORCH_CHECK(offset <= static_cast<size_t>(destination_.numel()) &&
                    size <= static_cast<size_t>(destination_.numel()) - offset &&
                    destination_.scalar_type() == output.scalar_type());
        TORCH_CHECK(output.dim() == 1);
        auto prepare_tensor = [size](const at::Tensor& tensor) {
            TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous() &&
                        static_cast<size_t>(tensor.numel()) == size);
#if defined(__linux__)
            // Advise once per range, before splitting it among writers.
            constexpr uintptr_t page_size = 4096;
            const auto addr = reinterpret_cast<uintptr_t>(tensor.data_ptr());
            const auto page = addr & ~(page_size - 1);
            madvise(reinterpret_cast<void*>(page), addr - page + tensor.nbytes(), MADV_HUGEPAGE);
#endif
        };
        prepare_tensor(output);
        for (const auto& input : inputs) prepare_tensor(input);
        threads_ = threads;
        for (size_t start = 0; start < size; start += chunk_elems_)
            chunks_.emplace_back(ranges_.size(), start, std::min(size - start, chunk_elems_));
        ranges_.push_back({offset, std::move(output), std::move(inputs), std::move(kernel)});
    }

    void cancel() { cancelled_.store(true, std::memory_order_release); }

    void run(const std::function<Work()>& take_next)
    {
        seal();
        std::exception_ptr failure;
        std::mutex failure_mutex;
        auto fail = [&] {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) failure = std::current_exception();
            cancel();
        };
        const ReflowCPUAffinity affinity;
#pragma omp parallel num_threads(threads_)
        {
            affinity.apply();
#ifdef _OPENMP
            const size_t writer = omp_get_thread_num(), team = omp_get_num_threads();
#else
            const size_t writer = 0, team = 1;
#endif
            try {
                c10::OptionalStreamGuard copy_stream;
                ReflowReadyWork* work = this;
                while (work && !cancelled_.load(std::memory_order_acquire)) {
                    auto prepare_next = [&] {
                        if (cancelled_.load(std::memory_order_acquire)) return;
                        try {
                            auto next = take_next();
                            if (next) {
                                TORCH_CHECK(next->stream_ == stream_,
                                            "Ready jobs must use the same copy stream");
                                next->seal();
                                work->next_ = std::move(next);
                            }
                        } catch (...) {
                            // A failed admission must not consume another successor.
                            fail();
                        }
                    };
                    for (auto& chunk : work->chunks_) {
                        if (cancelled_.load(std::memory_order_acquire)) break;
                        const auto& range = work->ranges_[chunk.range];
                        if (!chunk.write_partition(range.kernel, writer, team) ||
                            cancelled_.load(std::memory_order_acquire))
                            continue;

                        if (!copy_stream.current_stream()) copy_stream.reset_stream(work->stream_);
                        work->destination_.narrow(0, range.offset + chunk.start, chunk.size)
                            .copy_(range.output.narrow(0, chunk.start, chunk.size), true);
                        // Prepare one independent successor during the remaining CPU work.
                        if (&chunk == &work->chunks_.front() && work->chunks_.size() > 1)
                            std::call_once(work->prefetch_, prepare_next);
                    }
                    std::call_once(work->admit_, [&] {
                        // A leading writer can reach the boundary before the first DMA.
                        // Close or join the early lookup before inspecting its result.
                        std::call_once(work->prefetch_, [] {});
                        if (!work->next_) prepare_next();
                    });
                    work = work->next_.get();
                }
            } catch (...) {
                fail();
            }
        }
        if (failure) std::rethrow_exception(failure);
    }

private:
    void seal()
    {
        std::lock_guard<std::mutex> lock(setup_mutex_);
        TORCH_CHECK(!sealed_ && !chunks_.empty(), "Ready work must contain fresh ranges");
        sealed_ = true;
    }
    std::mutex setup_mutex_;
    bool sealed_ = false;
    struct Range {
        size_t offset;
        at::Tensor output;
        std::vector<at::Tensor> inputs;  // Own the storage used by the deferred native kernel.
        Kernel kernel;
    };
    struct Chunk {
        size_t range, start, size;
        std::atomic<size_t> finished{0};
        Chunk(size_t range, size_t start, size_t size) : range(range), start(start), size(size) {}

        // Only the last writer may submit this chunk to DMA.
        bool write_partition(const Kernel& kernel, size_t writer, size_t team)
        {
            // Whole SIMD blocks keep the scalar tail identical to the unsplit kernel.
            constexpr size_t block = 128;
            const size_t blocks = std::max<size_t>(1, size / block);
            const size_t writers = std::min(team, blocks);
            if (writer >= writers) return false;
            const size_t begin = (blocks * writer / writers) * block;
            const size_t end = writer + 1 == writers ? size
                                                     : (blocks * (writer + 1) / writers) * block;
            kernel(start + begin, end - begin);
            // Each kernel fences its own stores. The last arrival acquires all those releases.
            return finished.fetch_add(1, std::memory_order_acq_rel) + 1 == writers;
        }
    };
    at::Tensor destination_;
    c10::Stream stream_;
    size_t chunk_elems_;
    int threads_ = 1;
    std::vector<Range> ranges_;
    std::deque<Chunk> chunks_;
    std::atomic<bool> cancelled_{false};
    std::once_flag prefetch_, admit_;
    Work next_;
};

inline void bind_reflow_ready_work(pybind11::module_& module)
{
    pybind11::class_<ReflowReadyWork, ReflowReadyWork::Work>(
        module, "ReadyWork", pybind11::module_local())
        .def(pybind11::init<at::Tensor, c10::Stream, int64_t>())
        .def("cancel", &ReflowReadyWork::cancel)
        .def("run", &ReflowReadyWork::run, pybind11::call_guard<pybind11::gil_scoped_release>());
}

template <typename F>
inline void reflow_ready_loop(size_t size,
                              size_t stride,
                              bool single_writer,
                              bool streaming,
                              F update)
{
    if (!size) return;
    if (single_writer) {
        for (size_t i = 0; i < size; i += stride) update(i);
#if defined(__AVX512__) || defined(__AVX256__)
        if (streaming) _mm_sfence();
#endif
        return;
    }
    ReflowCPUAffinity affinity;
#pragma omp parallel
    {
        affinity.apply();
        bool wrote = false;
#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < size; i += stride) {
            update(i);
            wrote = true;
        }
#if defined(__AVX512__) || defined(__AVX256__)
        if (wrote && streaming) _mm_sfence();
#endif
    }
}
