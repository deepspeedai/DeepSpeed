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

#include "predicated_tile_access_iterator_residual_last.h"
#include "predicated_tile_iterator_residual_last.h"

namespace cutlass {
namespace transform {
namespace threadblock {

template <typename BaseIterator>
struct MakeIteratorResidualLast;

template <typename Shape,
          typename Element,
          typename Layout,
          int AdvanceRank,
          typename ThreadMap,
          int AccessSize,
          bool Gather>
struct MakeIteratorResidualLast<
    PredicatedTileIterator<Shape, Element, Layout, AdvanceRank, ThreadMap, AccessSize, Gather>> {
    using Iterator = PredicatedTileIteratorResidualLast<Shape,
                                                        Element,
                                                        Layout,
                                                        AdvanceRank,
                                                        ThreadMap,
                                                        AccessSize,
                                                        Gather>;
};

template <typename Shape,
          typename Element,
          typename Layout,
          int AdvanceRank,
          typename ThreadMap,
          typename AccessType,
          bool Gather>
struct MakeIteratorResidualLast<PredicatedTileAccessIterator<Shape,
                                                             Element,
                                                             Layout,
                                                             AdvanceRank,
                                                             ThreadMap,
                                                             AccessType,
                                                             Gather>> {
    using Iterator = PredicatedTileAccessIteratorResidualLast<Shape,
                                                              Element,
                                                              Layout,
                                                              AdvanceRank,
                                                              ThreadMap,
                                                              AccessType,
                                                              Gather>;
};
}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass
