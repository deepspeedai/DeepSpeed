// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

// This does nothing.
template <typename ThreadMap, typename Shape, typename scalar_t>
struct BroadcastNoLoad {
    using Fragment =
        cutlass::Array<scalar_t, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
    static const bool kEnable = false;
    CUTLASS_DEVICE static void load(Fragment& frag,
                                    scalar_t* ptr,
                                    int thread_id,
                                    const cutlass::MatrixCoord& extent,
                                    int stride)
    {
    }
    CUTLASS_DEVICE static scalar_t*
    advance(scalar_t* ptr, int B_id, int N_id, int H_id, int strideB, int strideN, int strideH)
    {
        return ptr;
    }
};

// This is to load the bias matrix from the global memory with on-the-fly
// broadcast. The shape in global memory is [B, N, 1, 1, L]. Each time we load
// the last dimension as a L row vector, and we further broadcast the L vector
// to a tile of size [L, L] by repeating the L vector L times
template <typename ThreadMap, typename Shape, typename scalar_t>
struct BroadcastA : public BroadcastNoLoad<ThreadMap, Shape, scalar_t> {
    using Base = BroadcastNoLoad<ThreadMap, Shape, scalar_t>;
    static const bool kEnable = true;
    using layout = cutlass::layout::AffineRank2RowMajor;

    using GmemTileIterator = cutlass::transform::threadblock::
        PredicatedTileIterator<Shape, scalar_t, layout, 0, ThreadMap>;
    using Fragment = typename GmemTileIterator::Fragment;

    CUTLASS_DEVICE static void load(Fragment& frag,
                                    scalar_t* ptr,
                                    int thread_id,
                                    const cutlass::MatrixCoord& extent,
                                    int stride)
    {
        GmemTileIterator iter({layout(0, 1)}, ptr, extent, thread_id);
        iter.load(frag);
    }

    CUTLASS_DEVICE static scalar_t*
    advance(scalar_t* ptr, int B_id, int N_id, int H_id, int strideB, int strideN, int strideH)
    {
        return ptr + B_id * strideB + N_id * strideN;
    }
};

// This is to load the bias matrix from the global memory with on-the-fly
// broadcast. The shape in global memory is [B, 1, H, L, L]. Each time we load
// a [L, L] matrix. Different N use the same bias matrix when B and H are the
// same.
template <typename ThreadMap, typename Shape, typename scalar_t>
struct BroadcastB : public BroadcastNoLoad<ThreadMap, Shape, scalar_t> {
    using Base = BroadcastNoLoad<ThreadMap, Shape, scalar_t>;
    static const bool kEnable = true;
    using layout = cutlass::layout::RowMajor;

    using GmemTileIterator = cutlass::transform::threadblock::
        PredicatedTileIterator<Shape, scalar_t, layout, 0, ThreadMap>;
    using Fragment = typename GmemTileIterator::Fragment;

    CUTLASS_DEVICE static void load(Fragment& frag,
                                    scalar_t* ptr,
                                    int thread_id,
                                    const cutlass::MatrixCoord& extent,
                                    int stride)
    {
        GmemTileIterator iter({layout(stride)}, ptr, extent, thread_id);
        iter.load(frag);
    }

    CUTLASS_DEVICE static scalar_t*
    advance(scalar_t* ptr, int B_id, int N_id, int H_id, int strideB, int strideN, int strideH)
    {
        return ptr + B_id * strideB + H_id * strideH;
    }
};

template <typename Shape,
          typename scalar_t,
          int kThreads,
          template <typename, typename, typename>
          class Broadcast1_,
          template <typename, typename, typename>
          class Broadcast2_>
struct AttentionBiasEpilogue {
    using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<
        cutlass::layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
        kThreads,
        1>;

    using Broadcast1 = Broadcast1_<ThreadMap, Shape, scalar_t>;
    using Broadcast2 = Broadcast2_<ThreadMap, Shape, scalar_t>;

    Broadcast1 broadcast1;
    Broadcast2 broadcast2;

    using Ref = cutlass::TensorRef<float, cutlass::layout::RowMajor>;
    using SmemTileIterator = cutlass::transform::threadblock::
        RegularTileIterator<Shape, float, cutlass::layout::RowMajor, 0, ThreadMap>;

    CUTLASS_DEVICE void operator()(const Ref& ref,
                                   scalar_t* ptr1,
                                   scalar_t* ptr2,
                                   int thread_id,
                                   const cutlass::MatrixCoord& extent,
                                   int stride)
    {
        static_assert(Broadcast1::Fragment::kElements == Broadcast2::Fragment::kElements,
                      "The two broadcast fragments must have the same number of "
                      "elements");
        typename SmemTileIterator::Fragment frag;
        frag.clear();
        float* frag_ptr = reinterpret_cast<float*>(&frag);
        if (Broadcast1::kEnable) {
            typename Broadcast1::Fragment frag1;
            frag1.clear();
            broadcast1.load(frag1, ptr1, thread_id, extent, stride);
            scalar_t* frag1_ptr = reinterpret_cast<scalar_t*>(&frag1);
            for (int i = 0; i < Broadcast1::Fragment::kElements; ++i) {
                frag_ptr[i] += static_cast<float>(frag1_ptr[i]);
            }
        }
        if (Broadcast2::kEnable) {
            typename Broadcast2::Fragment frag2;
            frag2.clear();
            broadcast2.load(frag2, ptr2, thread_id, extent, stride);
            scalar_t* frag2_ptr = reinterpret_cast<scalar_t*>(&frag2);
            for (int i = 0; i < Broadcast2::Fragment::kElements; ++i) {
                frag_ptr[i] += static_cast<float>(frag2_ptr[i]);
            }
        }
        SmemTileIterator iter(ref, thread_id);
        iter.store(frag);
        __syncthreads();
    }
};
