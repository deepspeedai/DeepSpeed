// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Resolve the process-wide pin-tensor manager exported by the pin_memory op.
The manager is compiled only into pin_memory; async_io/gds must load that op
first (see AsyncIOBuilder.load) so this symbol can be found here -- via
RTLD_GLOBAL/dlsym on POSIX, or by scanning loaded modules on Windows.
*/

#include "deepspeed_pin_tensor.h"

#include <stdexcept>

#if defined(_WIN32)
#define NOMINMAX
#define PSAPI_VERSION 2
#include <psapi.h>
#include <windows.h>
#include <vector>
#else
#include <dlfcn.h>
#endif

using holder_fn_t = void* (*)();

namespace {
const char* const kMissingSymbolMessage =
    "DeepSpeed pin_memory op must be loaded before async_io/gds (missing exported pin_memory "
    "symbol). Load PinMemoryBuilder first.";
}  // namespace

#if defined(_WIN32)
namespace {
FARPROC find_exported_symbol(const char* name)
{
    DWORD needed = 0;
    EnumProcessModules(GetCurrentProcess(), nullptr, 0, &needed);
    std::vector<HMODULE> modules(needed / sizeof(HMODULE));
    if (!EnumProcessModules(GetCurrentProcess(), modules.data(), needed, &needed)) {
        return nullptr;
    }
    for (auto mod : modules) {
        if (auto* addr = GetProcAddress(mod, name)) { return addr; }
    }
    return nullptr;
}

template <typename FnPtr>
FnPtr resolve(const char* name)
{
    auto* addr = find_exported_symbol(name);
    if (addr == nullptr) { throw std::runtime_error(kMissingSymbolMessage); }
    return reinterpret_cast<FnPtr>(addr);
}
}  // namespace
#endif

std::shared_ptr<deepspeed_pin_tensor_t> deepspeed_pin_tensor_t::shared()
{
#if defined(_WIN32)
    auto* fn = resolve<holder_fn_t>("deepspeed_pin_tensor_mgr_holder");
#else
    auto* fn =
        reinterpret_cast<holder_fn_t>(dlsym(RTLD_DEFAULT, "deepspeed_pin_tensor_mgr_holder"));
    if (fn == nullptr) { throw std::runtime_error(kMissingSymbolMessage); }
#endif
    auto* holder = static_cast<std::shared_ptr<deepspeed_pin_tensor_t>*>(fn());
    return *holder;
}

#if defined(_WIN32)
torch::Tensor deepspeed_pin_tensor_t::alloc(const int64_t num_elem, const at::ScalarType& elem_type)
{
    using fn_t = torch::Tensor (*)(const int64_t, const at::ScalarType);
    return resolve<fn_t>("deepspeed_pin_tensor_alloc_by_scalartype")(num_elem, elem_type);
}

torch::Tensor deepspeed_pin_tensor_t::alloc(const int64_t num_elem,
                                            const torch::TensorOptions& options)
{
    using fn_t = torch::Tensor (*)(const int64_t, const torch::TensorOptions&);
    return resolve<fn_t>("deepspeed_pin_tensor_alloc_by_options")(num_elem, options);
}

bool deepspeed_pin_tensor_t::free(torch::Tensor& locked_tensor)
{
    using fn_t = bool (*)(torch::Tensor&);
    return resolve<fn_t>("deepspeed_pin_tensor_free_tensor")(locked_tensor);
}

bool deepspeed_pin_tensor_t::is_managed(const torch::Tensor& buffer)
{
    using fn_t = bool (*)(const torch::Tensor&);
    return resolve<fn_t>("deepspeed_pin_tensor_check_is_managed")(buffer);
}
#endif
