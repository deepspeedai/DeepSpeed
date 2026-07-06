// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>
#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "cpu_adam.h"
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <cerrno>
#include <ctime>
#endif

using namespace std::string_literals;
static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

// C++ interface

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Adam_Optimizer::Step_1(ds_params_precision_t* _params,
                            ds_params_precision_t* grads,
                            ds_state_precision_t* _exp_avg,
                            ds_state_precision_t* _exp_avg_sq,
                            size_t _param_size,
                            bool parallel)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<1>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq, _param_size, parallel);
#endif
    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - _betta1;
        float betta2_minus1 = 1 - _betta2;

        float step_size = -1 * _alpha / _bias_correction1;
        float w_decay = -1 * _alpha * _weight_decay;

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#pragma omp parallel for if (parallel)
            for (size_t k = t; k < offset; k++) {
                float grad = (float)grads[k];
                float param = (float)_params[k];
                float momentum = _exp_avg[k];
                float variance = _exp_avg_sq[k];
                if (_weight_decay > 0 && !_adamw_mode) { grad = param * _weight_decay + grad; }
                momentum = momentum * _betta1;
                momentum = grad * betta1_minus1 + momentum;

                variance = variance * _betta2;
                grad = grad * grad;
                variance = grad * betta2_minus1 + variance;

                grad = sqrt(variance);
                grad = grad * _bias_correction2 + _eps;
                grad = momentum / grad;
                if (_weight_decay > 0 && _adamw_mode) { param += w_decay * param; }
                param = grad * step_size + param;
                _params[k] = param;
                _exp_avg[k] = momentum;
                _exp_avg_sq[k] = variance;
            }
        }
    }
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Adam_Optimizer::Step_4(ds_params_precision_t* _params,
                            ds_params_precision_t* grads,
                            ds_state_precision_t* _exp_avg,
                            ds_state_precision_t* _exp_avg_sq,
                            size_t _param_size,
                            bool parallel)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<4>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq, _param_size, parallel);
#endif
    if (_param_size > rounded_size)
        Step_1((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               parallel);
}

int create_adam_optimizer(int optimizer_id,
                          float alpha,
                          float betta1,
                          float betta2,
                          float eps,
                          float weight_decay,
                          bool adamw_mode,
                          bool should_log)
{
    auto opt =
        std::make_shared<Adam_Optimizer>(alpha, betta1, betta2, eps, weight_decay, adamw_mode);

    s_optimizers[optimizer_id] = opt;

    if (should_log) {
        std::string avx_type = "";
#if defined(__AVX512__)
        avx_type = "AVX512";
#else
#if defined(__AVX256__)
        avx_type = "AVX2";
#else
        avx_type = "scalar";
#endif
#endif

        printf("Adam Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f, adam_w=%d\n",
               alpha,
               betta1,
               betta2,
               weight_decay,
               (int)adamw_mode);
    }

    return 0;
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Adam_Optimizer::Step_8(ds_params_precision_t* _params,
                            ds_params_precision_t* grads,
                            ds_state_precision_t* _exp_avg,
                            ds_state_precision_t* _exp_avg_sq,
                            size_t _param_size,
                            bool parallel)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<8>(&rounded_size, _params, grads, _exp_avg, _exp_avg_sq, _param_size, parallel);
#endif
    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               parallel);
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void step_invoker(std::shared_ptr<Adam_Optimizer> opt,
                  void* _params,
                  void* grads,
                  void* _exp_avg,
                  void* _exp_avg_sq,
                  size_t _param_size,
                  bool parallel)
{
    opt->Step_8((ds_params_precision_t*)(_params),
                (ds_params_precision_t*)(grads),
                (ds_state_precision_t*)(_exp_avg),
                (ds_state_precision_t*)(_exp_avg_sq),
                _param_size,
                parallel);
}

std::map<
    std::tuple<c10::ScalarType, c10::ScalarType>,
    std::function<void(std::shared_ptr<Adam_Optimizer>, void*, void*, void*, void*, size_t, bool)>>
    invokers;

// Fill map with template functions for each type
template <class ds_params_precision_t, class ds_state_precision_t>
void create_invoker()
{
    invokers[std::tuple(c10::CppTypeToScalarType<ds_params_precision_t>(),
                        c10::CppTypeToScalarType<ds_state_precision_t>())] =
        step_invoker<ds_params_precision_t, ds_state_precision_t>;
}
struct InvokerInitializer {
    InvokerInitializer()
    {
        create_invoker<c10::Half, float>();
        create_invoker<c10::Half, c10::Half>();
        create_invoker<c10::BFloat16, float>();
        create_invoker<c10::BFloat16, c10::BFloat16>();
        create_invoker<float, float>();
    }
} _invoker_initializer;

void invoke(std::shared_ptr<Adam_Optimizer> opt,
            torch::Tensor& params,
            torch::Tensor& grads,
            torch::Tensor& exp_avg,
            torch::Tensor& exp_avg_sq,
            size_t param_size,
            bool parallel = true)
{
    c10::ScalarType params_type = at::typeMetaToScalarType(params.options().dtype());
    c10::ScalarType state_type = at::typeMetaToScalarType(exp_avg.options().dtype());

    auto it = invokers.find(std::tuple(params_type, state_type));
    if (it == invokers.end()) {
        throw std::runtime_error("Adam optimizer with param type "s + c10::toString(params_type) +
                                 " and state type "s + c10::toString(state_type) +
                                 " is not supported on current hardware"s);
    }

    it->second(opt,
               params.data_ptr(),
               grads.data_ptr(),
               exp_avg.data_ptr(),
               exp_avg_sq.data_ptr(),
               param_size,
               parallel);
}

int ds_adam_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float weight_decay,
                 bool bias_correction,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg,
                 torch::Tensor& exp_avg_sq)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);

    invoke(opt, params_c, grads_c, exp_avg_c, exp_avg_sq_c, params_c.numel());

    return 0;
}

void adamw_rollback_inplace(float* params,
                            const float* grads,
                            float* momentum,
                            float* variance,
                            size_t param_size,
                            float learning_rate,
                            float beta1,
                            float beta2,
                            float eps,
                            float weight_decay,
                            int& step_count)
{
    const float lr = learning_rate;
    const float lambda = weight_decay;
    const float beta1_pow = std::pow(beta1, step_count);
    const float beta2_pow = std::pow(beta2, step_count);
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const float lr_lambda = lr * lambda;
    const float one_minus_lr_lambda = 1.0f - lr_lambda;

#pragma omp parallel for
    for (size_t i = 0; i < param_size; ++i) {
        const float bias_correction1 = 1.0f - beta1_pow;
        const float bias_correction2 = 1.0f - beta2_pow;

        const float m_hat = momentum[i] / bias_correction1;
        const float v_hat = variance[i] / bias_correction2;

        const float denominator = std::sqrt(v_hat) + eps;

        // Rollback parameter update
        const float update = lr * m_hat / denominator;
        float new_param = (params[i] + update) / one_minus_lr_lambda;

        // Handle numerical instability
        if (!std::isfinite(new_param)) { new_param = 0.0f; }

        params[i] = new_param;

        const float grad = grads[i];
        momentum[i] = (momentum[i] - one_minus_beta1 * grad) / beta1;
        variance[i] = (variance[i] - one_minus_beta2 * grad * grad) / beta2;
    }

    --step_count;
}

int ds_adam_rollback(int optimizer_id,
                     size_t step,
                     float lr,
                     float beta1,
                     float beta2,
                     float epsilon,
                     float weight_decay,
                     bool bias_correction,
                     torch::Tensor& params,
                     torch::Tensor& grads,
                     torch::Tensor& exp_avg,
                     torch::Tensor& exp_avg_sq)
{
    try {
        // Validate tensor types - rollback currently only supports float32
        if (params.scalar_type() != torch::kFloat32 || grads.scalar_type() != torch::kFloat32 ||
            exp_avg.scalar_type() != torch::kFloat32 ||
            exp_avg_sq.scalar_type() != torch::kFloat32) {
            printf("Error: Adam rollback currently only supports float32 tensors\n");
            return -1;
        }

        float* params_ptr = params.data_ptr<float>();
        const float* grads_ptr = grads.data_ptr<float>();
        float* momentum_ptr = exp_avg.data_ptr<float>();
        float* variance_ptr = exp_avg_sq.data_ptr<float>();
        const size_t param_size = params.numel();
        int step_count = static_cast<int>(step);

        adamw_rollback_inplace(params_ptr,
                               grads_ptr,
                               momentum_ptr,
                               variance_ptr,
                               param_size,
                               lr,
                               beta1,
                               beta2,
                               epsilon,
                               weight_decay,
                               step_count);

        return 0;
    } catch (const std::exception& e) {
        printf("Error in Adam rollback for optimizer #%d: %s\n", optimizer_id, e.what());
        return -1;
    }
}

int destroy_adam_optimizer(int optimizer_id)
{
    s_optimizers.erase(optimizer_id);

    return 0;
}

// ---------------------------------------------------------------------------
// ZenFlowAdam: the native CPU Adam that backs ZenFlow's overlapped optimizer step.
//
// The optimizer runs in a dedicated process (see zenflow_utils.start_optimizer_process):
// run_worker() blocks on a shared-memory control block and, for each requested step, fans
// the heavy per-element math out to a pool of worker threads pinned to ZenFlow's dedicated
// cores, each running its element slice through the serial (parallel=false) kernel. The
// Adam state lives in that process, NUMA-local to the pool.
// ---------------------------------------------------------------------------

// A persistent pool of threads pinned to a fixed core set. parallel_for() splits
// [0, total) into one contiguous chunk per thread and blocks until all finish.
class PinnedThreadPool {
public:
    explicit PinnedThreadPool(const std::vector<int>& affinity)
    {
        n_ = std::max<size_t>(1, affinity.size());
        for (size_t i = 0; i < n_; ++i) {
            int core = affinity.empty() ? -1 : affinity[i];
            threads_.emplace_back([this, i, core] { worker(i, core); });
        }
    }

    ~PinnedThreadPool()
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            ++gen_;
        }
        cv_start_.notify_all();
        for (auto& t : threads_) t.join();
    }

    size_t size() const { return n_; }

    // Split [0, total) into one chunk per thread. Chunk boundaries are rounded up to a
    // multiple of `align` so each slice's AVX/scalar split lines up with the whole-tensor
    // kernel's split -- otherwise an element could be computed by AVX (FMA) in one layout
    // and the scalar tail (mul+add) in another, which differ in the last bit.
    void parallel_for(size_t total, size_t align, std::function<void(size_t, size_t)> fn)
    {
        {
            std::unique_lock<std::mutex> lk(m_);
            fn_ = std::move(fn);
            total_ = total;
            align_ = std::max<size_t>(1, align);
            done_count_ = 0;
            ++gen_;
        }
        cv_start_.notify_all();
        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk, [this] { return done_count_ == n_; });
    }

private:
    void worker(size_t tid, int core)
    {
#if defined(__linux__)
        if (core >= 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(core, &set);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
        }
#endif
        long seen = 0;
        while (true) {
            std::function<void(size_t, size_t)> fn;
            size_t total = 0;
            size_t align = 1;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_start_.wait(lk, [this, seen] { return gen_ != seen; });
                seen = gen_;
                if (stop_) return;
                fn = fn_;
                total = total_;
                align = align_;
            }
            size_t chunk = (total + n_ - 1) / n_;
            chunk = ((chunk + align - 1) / align) * align;  // round up to SIMD-block alignment
            size_t begin = std::min(tid * chunk, total);
            size_t end = std::min(begin + chunk, total);
            if (end > begin) fn(begin, end);
            {
                std::lock_guard<std::mutex> lk(m_);
                ++done_count_;
                if (done_count_ == n_) cv_done_.notify_one();
            }
        }
    }

    size_t n_;
    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable cv_start_, cv_done_;
    std::function<void(size_t, size_t)> fn_;
    size_t total_ = 0;
    size_t align_ = 1;
    size_t done_count_ = 0;
    long gen_ = 0;
    bool stop_ = false;
};

// SIMD block the Adam AVX kernel rounds to (Step_8 => span 8). Slicing on multiples of
// this keeps each slice's AVX/scalar boundary identical to the whole-tensor kernel.
#if defined(__AVX512__) or defined(__AVX256__)
static constexpr size_t kZenAdamAlign = SIMD_WIDTH * 8;
#else
static constexpr size_t kZenAdamAlign = 1;
#endif

struct ZenHP {
    float lr, beta1, beta2, eps, weight_decay;
    bool bias_correction;
};

struct ZenGroup {
    torch::Tensor param;
    torch::Tensor grad[2];
    torch::Tensor exp_avg[2];
    torch::Tensor exp_avg_sq[2];
    torch::Tensor stale;  // may be undefined -> stale snapshot skipped
};

#if defined(__linux__)
// Control block placed in a shared-memory buffer (a shared torch tensor's storage) so the
// main process and the optimizer process coordinate through two process-shared semaphores
// instead of a pickling pipe. The main process writes a command + per-group hyperparameters
// and posts cmd_ready; the worker runs the step and posts done. `done` is a counting
// semaphore, so a skipped wait (the engine's post-warmup transition) is drained later.
static constexpr int ZEN_MAX_GROUPS = 1024;
// Hyperparameters packed per group in hp[]: lr, beta1, beta2, eps, weight_decay.
static constexpr int ZEN_HP_PER_GROUP = 5;
enum { ZEN_CMD_STEP = 0, ZEN_CMD_EXIT = 1 };

struct ZenControl {
    sem_t cmd_ready;
    sem_t done;
    int cmd;
    int now_state;
    int64_t step;
    int num_groups;
    float hp[ZEN_MAX_GROUPS * ZEN_HP_PER_GROUP];
    uint8_t bias_correction[ZEN_MAX_GROUPS];
};
#endif

class ZenFlowAdam {
public:
    ZenFlowAdam(int optimizer_id, std::vector<int> zf_affinity) : opt_id_(optimizer_id)
    {
        pool_ = std::make_unique<PinnedThreadPool>(zf_affinity);
    }

    ~ZenFlowAdam() = default;

    void register_group(torch::Tensor param,
                        torch::Tensor grad0,
                        torch::Tensor grad1,
                        torch::Tensor exp_avg0,
                        torch::Tensor exp_avg1,
                        torch::Tensor exp_avg_sq0,
                        torch::Tensor exp_avg_sq1,
                        torch::Tensor stale)
    {
        TORCH_CHECK(param.is_contiguous(), "ZenFlowAdam: param must be contiguous");
        ZenGroup g;
        g.param = param;
        g.grad[0] = grad0;
        g.grad[1] = grad1;
        g.exp_avg[0] = exp_avg0;
        g.exp_avg[1] = exp_avg1;
        g.exp_avg_sq[0] = exp_avg_sq0;
        g.exp_avg_sq[1] = exp_avg_sq1;
        g.stale = stale;
        groups_.push_back(std::move(g));
    }

#if defined(__linux__)
    // Process-mode driver: run in the optimizer process, block on the shared-memory control
    // block, and run each requested step on the pinned pool. Returns on the exit command.
    void run_worker(void* control_ptr)
    {
        ZenControl* ctrl = reinterpret_cast<ZenControl*>(control_ptr);
        while (true) {
            while (sem_wait(&ctrl->cmd_ready) != 0) {}  // retry on EINTR
            if (ctrl->cmd == ZEN_CMD_EXIT) break;
            const int ng = ctrl->num_groups;
            std::vector<ZenHP> hps(ng);
            for (int g = 0; g < ng; ++g) {
                hps[g] = {ctrl->hp[g * ZEN_HP_PER_GROUP + 0],
                          ctrl->hp[g * ZEN_HP_PER_GROUP + 1],
                          ctrl->hp[g * ZEN_HP_PER_GROUP + 2],
                          ctrl->hp[g * ZEN_HP_PER_GROUP + 3],
                          ctrl->hp[g * ZEN_HP_PER_GROUP + 4],
                          (bool)ctrl->bias_correction[g]};
            }
            run_step(ctrl->now_state, ctrl->step, hps);
            sem_post(&ctrl->done);
        }
    }
#endif

private:
    void run_step(int now_state, int64_t step, const std::vector<ZenHP>& hps)
    {
        auto opt = std::static_pointer_cast<Adam_Optimizer>(s_optimizers[opt_id_]);
        for (size_t g = 0; g < groups_.size(); ++g) {
            const ZenHP& hp = hps[g];
            // Groups share one Adam_Optimizer; advance its bias-correction state for
            // this group before the pool reads it (pool is idle here -> no race).
            opt->IncrementStep(step, hp.beta1, hp.beta2);
            opt->update_state(hp.lr, hp.eps, hp.weight_decay, hp.bias_correction);

            ZenGroup& grp = groups_[g];
            torch::Tensor& P = grp.param;
            torch::Tensor& G = grp.grad[now_state];
            torch::Tensor& M = grp.exp_avg[now_state];
            torch::Tensor& V = grp.exp_avg_sq[now_state];

            auto it = invokers.find(std::tuple(P.scalar_type(), M.scalar_type()));
            TORCH_CHECK(it != invokers.end(),
                        "ZenFlowAdam: unsupported param/state dtype combination");
            auto fn = it->second;

            char* pp = static_cast<char*>(P.data_ptr());
            char* gp = static_cast<char*>(G.data_ptr());
            char* mp = static_cast<char*>(M.data_ptr());
            char* vp = static_cast<char*>(V.data_ptr());
            char* sp = grp.stale.defined() ? static_cast<char*>(grp.stale.data_ptr()) : nullptr;
            const size_t pe = P.element_size();
            const size_t se = M.element_size();
            const size_t numel = P.numel();

            pool_->parallel_for(numel, kZenAdamAlign, [=](size_t b, size_t e) {
                const size_t len = e - b;
                // parallel=false: each pinned thread runs its slice serially.
                fn(opt, pp + b * pe, gp + b * pe, mp + b * se, vp + b * se, len, false);
                if (sp) std::memcpy(sp + b * pe, pp + b * pe, len * pe);
            });
        }
    }

    int opt_id_;
    std::vector<ZenGroup> groups_;
    std::unique_ptr<PinnedThreadPool> pool_;
};

// Handle-indexed registry, mirroring s_optimizers, so the Python side refers to a
// ZenFlowAdam by an int handle and the class itself stays encapsulated here.
static std::unordered_map<int, std::unique_ptr<ZenFlowAdam>> s_zenflow_adams;
static int s_next_zenflow_id = 0;

int zenflow_adam_create(int optimizer_id, std::vector<int> zf_affinity)
{
    int handle = s_next_zenflow_id++;
    s_zenflow_adams[handle] = std::make_unique<ZenFlowAdam>(optimizer_id, std::move(zf_affinity));
    return handle;
}

void zenflow_adam_register_group(int handle,
                                 torch::Tensor param,
                                 torch::Tensor grad0,
                                 torch::Tensor grad1,
                                 torch::Tensor exp_avg0,
                                 torch::Tensor exp_avg1,
                                 torch::Tensor exp_avg_sq0,
                                 torch::Tensor exp_avg_sq1,
                                 torch::Tensor stale)
{
    s_zenflow_adams.at(handle)->register_group(
        param, grad0, grad1, exp_avg0, exp_avg1, exp_avg_sq0, exp_avg_sq1, stale);
}

void zenflow_adam_destroy(int handle)
{
    // Erasing the unique_ptr runs ~ZenFlowAdam, which tears down the pinned pool.
    s_zenflow_adams.erase(handle);
}

#if defined(__linux__)
// Size (bytes) the shared control tensor must hold.
int64_t zenflow_adam_ctrl_size() { return (int64_t)sizeof(ZenControl); }

// Called once by the main process before spawning the optimizer process.
void zenflow_adam_ctrl_init(uintptr_t control_ptr, int num_groups)
{
    TORCH_CHECK(num_groups <= ZEN_MAX_GROUPS, "ZenFlowAdam: too many param groups");
    auto* ctrl = reinterpret_cast<ZenControl*>(control_ptr);
    ctrl->num_groups = num_groups;
    ctrl->cmd = ZEN_CMD_STEP;
    sem_init(&ctrl->cmd_ready, /*pshared=*/1, 0);
    sem_init(&ctrl->done, /*pshared=*/1, 0);
}

// Called in the optimizer process; blocks running steps until the exit command.
void zenflow_adam_run_worker(int handle, uintptr_t control_ptr)
{
    s_zenflow_adams.at(handle)->run_worker(reinterpret_cast<void*>(control_ptr));
}

void zenflow_adam_submit(uintptr_t control_ptr,
                         int now_state,
                         int64_t step,
                         std::vector<float> lr,
                         std::vector<float> beta1,
                         std::vector<float> beta2,
                         std::vector<float> eps,
                         std::vector<float> weight_decay,
                         std::vector<uint8_t> bias_correction)
{
    auto* ctrl = reinterpret_cast<ZenControl*>(control_ptr);
    const int ng = (int)lr.size();
    for (int g = 0; g < ng; ++g) {
        ctrl->hp[g * ZEN_HP_PER_GROUP + 0] = lr[g];
        ctrl->hp[g * ZEN_HP_PER_GROUP + 1] = beta1[g];
        ctrl->hp[g * ZEN_HP_PER_GROUP + 2] = beta2[g];
        ctrl->hp[g * ZEN_HP_PER_GROUP + 3] = eps[g];
        ctrl->hp[g * ZEN_HP_PER_GROUP + 4] = weight_decay[g];
        ctrl->bias_correction[g] = bias_correction[g];
    }
    ctrl->now_state = now_state;
    ctrl->step = step;
    ctrl->cmd = ZEN_CMD_STEP;
    sem_post(&ctrl->cmd_ready);  // release: hyperparameters above are visible to the worker
}

// Wait up to timeout_s for the optimizer process to post one completion. Returns true if a
// completion was consumed, false on timeout -- so the training side can re-check that the
// optimizer process is still alive and fail loudly instead of blocking forever if the process
// died mid-step (e.g. an OOM or TORCH_CHECK in run_step after it signalled ready).
bool zenflow_adam_wait(uintptr_t control_ptr, double timeout_s)
{
    auto* ctrl = reinterpret_cast<ZenControl*>(control_ptr);
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += (time_t)timeout_s;
    deadline.tv_nsec += (long)((timeout_s - (double)(time_t)timeout_s) * 1e9);
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000000000L;
    }
    while (sem_timedwait(&ctrl->done, &deadline) != 0) {
        if (errno == EINTR) continue;  // retry on signal
        return false;                  // timed out (or error): caller re-checks process liveness
    }
    return true;
}

void zenflow_adam_ctrl_exit(uintptr_t control_ptr)
{
    auto* ctrl = reinterpret_cast<ZenControl*>(control_ptr);
    ctrl->cmd = ZEN_CMD_EXIT;
    sem_post(&ctrl->cmd_ready);
}
#endif
