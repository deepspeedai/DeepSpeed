/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holdvr nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include "custom_mma_multistage.h"
#include "custom_mma_pipelined.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"

template <typename Mma, int kMaxK>
struct MakeCustomMma;

template <typename Shape,
          typename IteratorA,
          typename SmemIteratorA,
          cutlass::arch::CacheOperation::Kind CacheOpA,
          typename IteratorB,
          typename SmemIteratorB,
          cutlass::arch::CacheOperation::Kind CacheOpB,
          typename ElementC,
          typename LayoutC,
          typename Policy,
          int Stages,
          cutlass::gemm::SharedMemoryClearOption SharedMemoryClear,
          int kMaxK>
struct MakeCustomMma<cutlass::gemm::threadblock::MmaMultistage<Shape,
                                                               IteratorA,
                                                               SmemIteratorA,
                                                               CacheOpA,
                                                               IteratorB,
                                                               SmemIteratorB,
                                                               CacheOpB,
                                                               ElementC,
                                                               LayoutC,
                                                               Policy,
                                                               Stages,
                                                               SharedMemoryClear>,
                     kMaxK> {
    // Reduce the number of stages if we don't need that many
    static int constexpr kStages =
        kMaxK == cutlass::platform::numeric_limits<int>::max()
            ? Stages
            : cutlass::const_min(Stages, (kMaxK + int(Shape::kK) - 1) / int(Shape::kK));
    using Mma = cutlass::gemm::threadblock::CustomMmaMultistage<Shape,
                                                                IteratorA,
                                                                SmemIteratorA,
                                                                CacheOpA,
                                                                IteratorB,
                                                                SmemIteratorB,
                                                                CacheOpB,
                                                                ElementC,
                                                                LayoutC,
                                                                Policy,
                                                                kStages,
                                                                SharedMemoryClear,
                                                                kMaxK>;
};

template <typename Shape,
          typename IteratorA,
          typename SmemIteratorA,
          typename IteratorB,
          typename SmemIteratorB,
          typename ElementC,
          typename LayoutC,
          typename Policy,
          int kMaxK>
struct MakeCustomMma<cutlass::gemm::threadblock::MmaPipelined<Shape,
                                                              IteratorA,
                                                              SmemIteratorA,
                                                              IteratorB,
                                                              SmemIteratorB,
                                                              ElementC,
                                                              LayoutC,
                                                              Policy>,
                     kMaxK> {
    using Mma = cutlass::gemm::threadblock::CustomMmaPipelined<Shape,
                                                               IteratorA,
                                                               SmemIteratorA,
                                                               IteratorB,
                                                               SmemIteratorB,
                                                               ElementC,
                                                               LayoutC,
                                                               Policy>;
};
