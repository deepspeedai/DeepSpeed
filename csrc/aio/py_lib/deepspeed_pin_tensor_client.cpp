// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Resolve the process-wide pin-tensor manager exported by the pin_memory op.
The manager is compiled only into pin_memory; async_io/gds must load that op
first (see AsyncIOBuilder.load) so this symbol is visible via RTLD_GLOBAL.
*/

#include "deepspeed_pin_tensor.h"

#include <dlfcn.h>
#include <stdexcept>

std::shared_ptr<deepspeed_pin_tensor_t> deepspeed_pin_tensor_t::shared()
{
    using holder_fn_t = void* (*)();
    auto* fn =
        reinterpret_cast<holder_fn_t>(dlsym(RTLD_DEFAULT, "deepspeed_pin_tensor_mgr_holder"));
    if (fn == nullptr) {
        throw std::runtime_error(
            "DeepSpeed pin_memory op must be loaded before async_io/gds (missing "
            "deepspeed_pin_tensor_mgr_holder). Load PinMemoryBuilder first.");
    }
    auto* holder = static_cast<std::shared_ptr<deepspeed_pin_tensor_t>*>(fn());
    return *holder;
}
