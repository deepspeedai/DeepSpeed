/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once
#include <cutlass/cutlass.h>
#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

template <typename scalar_t,              // scalar type
          typename ThreadblockTileShape,  // size of tile to load
          int Threads,                    // number of participating threads
          int ElementsPerAccess>          // thread access width in elements
class TileSmemLoader {
public:
    using Shape = ThreadblockTileShape;
    using SmemTile = cutlass::AlignedBuffer<scalar_t, ThreadblockTileShape::kCount>;

    using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
        cutlass::layout::PitchLinearShape<ThreadblockTileShape::kColumn,  // contiguous
                                          ThreadblockTileShape::kRow>,    // strided
        Threads,                                                          // Threads
        ElementsPerAccess>;                                               // ElementsPerAccess

    using GmemTileIterator = cutlass::transform::threadblock::PredicatedTileIterator<
        ThreadblockTileShape,       // Shape
        scalar_t,                   // Element
        cutlass::layout::RowMajor,  // Layout
        0,                          // AdvanceRank
        ThreadMap>;                 // ThreadMap

    using SmemTileIterator =
        cutlass::transform::threadblock::RegularTileIterator<ThreadblockTileShape,       // Shape
                                                             scalar_t,                   // Element
                                                             cutlass::layout::RowMajor,  // Layout
                                                             0,           // AdvanceRank
                                                             ThreadMap>;  // ThreadMap

    using Fragment = typename GmemTileIterator::Fragment;

    /// load a tile from global memory into shared memory
    CUTLASS_DEVICE
    static void load(GmemTileIterator tile_load_iter, SmemTileIterator tile_store_iter)
    {
        Fragment tb_frag;
        tb_frag.clear();
        tile_load_iter.load(tb_frag);
        tile_store_iter.store(tb_frag);

        __syncthreads();
    }
};
