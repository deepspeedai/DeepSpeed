// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>

#include "mixed_gemm.h"
#include "moe_gemm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // mixed_gemm.h
    m.def("mixed_gemm", &mixed_gemm, "Mixed-precision GEMM");

    // moe_gemm.h
    m.def("moe_gemm", &moe_gemm, "MultiGEMM for MoE (16-bit weights)");
    m.def("mixed_moe_gemm", &mixed_moe_gemm, "MultiGEMM for MoE (4-bit/8-bit weights)");
}
