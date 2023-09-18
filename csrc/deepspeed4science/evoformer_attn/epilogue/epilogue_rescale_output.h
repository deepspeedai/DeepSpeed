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

/*! \file
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory
  to match canonical tensor layouts in global memory. Epilogues support
  conversion and reduction operations.

  This is a copy of cutlass/epilogue/threadblock/epilogue.h that can
  handle "row_id" as a first argument, as uses it to get the corresponding
  `m_prime` / `s_prime` to rescale the output.
*/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/vector.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/numeric_types.h"

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "epilogue_pipelined.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
// output <- alpha * accumulator + beta * source
//   with:
//     alpha = 1 / s_prime (to normalize when isLast=True, 1 otherwise)
//     beta = alpha / m_prime (renormalize the output when the max changes)
//     source is the current output
template <typename ElementOutput_,  ///< Data type used to store tensors
          typename ElementSource_,  //< Data type for source (usually matches
                                    //`ElementOutput`)
          int Count,                ///< Number of elements computed per operation.
                                    ///< Usually it is 128/sizeof_bits<ElementOutput_>,
          ///< but we use 64 or 32 sometimes when there are not enough data
          ///< to store
          typename ElementAccumulator_,  ///< Accumulator data type
          typename ElementCompute_,      ///< Data type used to compute linear combination
          bool isFirst,
          bool isLast,
          typename FragmentAlphaBeta_,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
class MemoryEfficientAttentionNormalize {
public:
    using ElementOutput = ElementOutput_;
    using ElementSource = ElementSource_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;

    static int const kCount = Count;

    using FragmentOutput = Array<ElementOutput, kCount>;
    using FragmentSource = Array<ElementSource, kCount>;
    using FragmentAccumulator = Array<ElementAccumulator, kCount>;
    using ComputeFragment = Array<ElementCompute, kCount>;
    using FragmentAlphaBeta = FragmentAlphaBeta_;

    static FloatRoundStyle const kRound = Round;

private:
    //
    // Data members
    //

    FragmentAlphaBeta const& s_prime_;
    FragmentAlphaBeta const& m_prime_;

public:
    /// Constructs the function object, possibly loading from pointers in host
    /// memory
    CUTLASS_HOST_DEVICE
    MemoryEfficientAttentionNormalize(FragmentAlphaBeta const& s_prime,
                                      FragmentAlphaBeta const& m_prime)
        : s_prime_(s_prime), m_prime_(m_prime)
    {
    }

    /// Returns true if source is needed
    CUTLASS_HOST_DEVICE
    bool is_source_needed() const { return !isFirst; }

    /// Functionally required for serial reduction in the epilogue
    CUTLASS_HOST_DEVICE
    void set_k_partition(int k_partition, int k_partition_count) {}

    /// Computes linear scaling: D = alpha * accumulator + beta * source
    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(int row,
                              FragmentAccumulator const& accumulator,
                              FragmentSource const& source) const
    {
        assert(!isFirst);

        // Convert source to internal compute numeric type
        NumericArrayConverter<ElementCompute, ElementSource, kCount, Round> source_converter;
        NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
            accumulator_converter;

        // Convert to destination numeric type
        NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

        ComputeFragment converted_source = source_converter(source);
        ComputeFragment converted_accumulator = accumulator_converter(accumulator);

        // Perform binary operations
        ComputeFragment intermediate;

        multiplies<ComputeFragment> mul_add_source;
        multiply_add<ComputeFragment> mul_add_accumulator;

        ElementCompute alpha = isLast ? (1 / s_prime_[row]) : 1;
        ElementCompute beta = alpha * m_prime_[row];

        intermediate = mul_add_source(beta, converted_source);  // X =  beta * C

        intermediate = mul_add_accumulator(
            alpha, converted_accumulator, intermediate);  // D = alpha * Accum + X

        return destination_converter(intermediate);
    }

    /// Computes linear scaling: D = alpha * accumulator
    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(int row, FragmentAccumulator const& accumulator) const
    {
        assert(isFirst);

        // Convert source to internal compute numeric type
        NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
            accumulator_converter;

        // Convert to destination numeric type
        NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

        ComputeFragment converted_accumulator = accumulator_converter(accumulator);

        ComputeFragment intermediate;
        multiplies<ComputeFragment> mul_accumulator;

        ElementCompute alpha = isLast ? (1 / s_prime_[row]) : 1;

        intermediate = mul_accumulator(alpha, converted_accumulator);  // X =  alpha * C + uniform

        return destination_converter(intermediate);
    }
};

}  // namespace thread

namespace threadblock {
template <typename EO,
          typename ES,
          int Count,
          typename EA,
          typename EC,
          bool F,
          bool L,
          typename FAB,
          FloatRoundStyle R>
struct ApplyEpilogueOp<
    thread::MemoryEfficientAttentionNormalize<EO, ES, Count, EA, EC, F, L, FAB, R>> {
    using Op = thread::MemoryEfficientAttentionNormalize<EO, ES, Count, EA, EC, F, L, FAB, R>;
    static CUTLASS_DEVICE typename Op::FragmentOutput apply(
        Op const& output_op,
        int row_id,
        typename Op::FragmentAccumulator const& accum,
        typename Op::FragmentSource const& source)
    {
        return output_op(row_id, accum, source);
    }
    static CUTLASS_DEVICE typename Op::FragmentOutput
    apply(Op const& output_op, int row_id, typename Op::FragmentAccumulator const& accum)
    {
        return output_op(row_id, accum);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
