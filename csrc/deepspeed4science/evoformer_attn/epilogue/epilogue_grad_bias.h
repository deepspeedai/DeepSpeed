// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h>
#include "../iterators/predicated_tile_iterator_atomic.h"
#include "cutlass/epilogue/threadblock/epilogue.h"

namespace cutlass {
namespace epilogue {
namespace threadblock {
template <int Rank,
          typename Shape_,
          typename WarpMmaTensorOp_,
          int PartitionsK,
          typename OutputOp_,
          int ElementsPerAccess>
struct EpilogueTensorOpAffineRankN : public DefaultEpilogueTensorOpAffineRankN<Rank,
                                                                               Shape_,
                                                                               WarpMmaTensorOp_,
                                                                               PartitionsK,
                                                                               OutputOp_,
                                                                               ElementsPerAccess> {
    using Base = DefaultEpilogueTensorOpAffineRankN<Rank,
                                                    Shape_,
                                                    WarpMmaTensorOp_,
                                                    PartitionsK,
                                                    OutputOp_,
                                                    ElementsPerAccess>;
    using OutputTileIterator =
        cutlass::epilogue::threadblock::PredicatedTileIteratorAffineRankNAtomic<
            typename Base::OutputTileThreadMap,
            typename Base::ElementOutput,
            Rank>;

    using Epilogue =
        cutlass::epilogue::threadblock::Epilogue<typename Base::Shape,
                                                 typename Base::WarpMmaTensorOp,
                                                 Base::kPartitionsK,
                                                 OutputTileIterator,
                                                 typename Base::AccumulatorFragmentIterator,
                                                 typename Base::WarpTileIterator,
                                                 typename Base::SharedLoadIterator,
                                                 typename Base::OutputOp,
                                                 typename Base::Padding,
                                                 Base::kFragmentsPerIteration>;
};

template <int Rank,
          typename Shape_,
          typename WarpMmaTensorOp_,
          int PartitionsK,
          typename OutputOp_,
          int ElementsPerAccess>
struct EpilogueVoltaTensorOpAffineRankN
    : public DefaultEpilogueVoltaTensorOpAffineRankN<Rank,
                                                     Shape_,
                                                     WarpMmaTensorOp_,
                                                     PartitionsK,
                                                     OutputOp_,
                                                     ElementsPerAccess> {
    using Base = DefaultEpilogueVoltaTensorOpAffineRankN<Rank,
                                                         Shape_,
                                                         WarpMmaTensorOp_,
                                                         PartitionsK,
                                                         OutputOp_,
                                                         ElementsPerAccess>;
    using OutputTileIterator =
        cutlass::epilogue::threadblock::PredicatedTileIteratorAffineRankNAtomic<
            typename Base::OutputTileThreadMap,
            typename Base::ElementOutput,
            Rank>;

    using Epilogue =
        cutlass::epilogue::threadblock::Epilogue<typename Base::Shape,
                                                 typename Base::WarpMmaTensorOp,
                                                 Base::kPartitionsK,
                                                 OutputTileIterator,
                                                 typename Base::AccumulatorFragmentIterator,
                                                 typename Base::WarpTileIterator,
                                                 typename Base::SharedLoadIterator,
                                                 typename Base::OutputOp,
                                                 typename Base::Padding>;
};

template <typename Shape_,
          typename WarpMmaTensorOp_,
          int PartitionsK,
          typename OutputOp_,
          int ElementsPerAccess,
          bool ScatterD = false,
          typename PermuteDLayout = layout::NoPermute>
struct EpilogueTensorOp : public DefaultEpilogueTensorOp<Shape_,
                                                         WarpMmaTensorOp_,
                                                         PartitionsK,
                                                         OutputOp_,
                                                         ElementsPerAccess,
                                                         ScatterD,
                                                         PermuteDLayout> {
    using Base = DefaultEpilogueTensorOp<Shape_,
                                         WarpMmaTensorOp_,
                                         PartitionsK,
                                         OutputOp_,
                                         ElementsPerAccess,
                                         ScatterD,
                                         PermuteDLayout>;
    using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorAtomic<
        typename Base::OutputTileThreadMap,
        typename Base::ElementOutput,
        ScatterD,
        PermuteDLayout>;
    using Epilogue =
        cutlass::epilogue::threadblock::Epilogue<typename Base::Shape,
                                                 typename Base::WarpMmaTensorOp,
                                                 Base::kPartitionsK,
                                                 OutputTileIterator,
                                                 typename Base::AccumulatorFragmentIterator,
                                                 typename Base::WarpTileIterator,
                                                 typename Base::SharedLoadIterator,
                                                 typename Base::OutputOp,
                                                 typename Base::Padding,
                                                 Base::kFragmentsPerIteration>;
};

template <typename Shape_,
          typename WarpMmaTensorOp_,
          int PartitionsK,
          typename OutputOp_,
          int ElementsPerAccess,
          bool ScatterD = false,
          typename PermuteDLayout = layout::NoPermute>
struct EpilogueVoltaTensorOp : public DefaultEpilogueVoltaTensorOp<Shape_,
                                                                   WarpMmaTensorOp_,
                                                                   PartitionsK,
                                                                   OutputOp_,
                                                                   ElementsPerAccess,
                                                                   ScatterD,
                                                                   PermuteDLayout> {
    using Base = DefaultEpilogueVoltaTensorOp<Shape_,
                                              WarpMmaTensorOp_,
                                              PartitionsK,
                                              OutputOp_,
                                              ElementsPerAccess,
                                              ScatterD,
                                              PermuteDLayout>;
    using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorAtomic<
        typename Base::OutputTileThreadMap,
        typename Base::ElementOutput,
        ScatterD,
        PermuteDLayout>;
    using Epilogue =
        cutlass::epilogue::threadblock::Epilogue<typename Base::Shape,
                                                 typename Base::WarpMmaTensorOp,
                                                 Base::kPartitionsK,
                                                 OutputTileIterator,
                                                 typename Base::AccumulatorFragmentIterator,
                                                 typename Base::WarpTileIterator,
                                                 typename Base::SharedLoadIterator,
                                                 typename Base::OutputOp,
                                                 typename Base::Padding>;
};
}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

template <typename Arch_,
          typename Shape_,
          typename WarpMmaTensorOp_,
          int PartitionsK,
          typename OutputOp_,
          int ElementsPerAccess,
          bool ScatterD = false,
          typename PermuteDLayout = cutlass::layout::NoPermute>
struct BiasGradEpilogue {
    using Epilogue =
        typename cutlass::epilogue::threadblock::EpilogueTensorOp<Shape_,
                                                                  WarpMmaTensorOp_,
                                                                  PartitionsK,
                                                                  OutputOp_,
                                                                  ElementsPerAccess,
                                                                  ScatterD,
                                                                  PermuteDLayout>::Epilogue;
};

template <typename Shape_,
          typename WarpMmaTensorOp_,
          int PartitionsK,
          typename OutputOp_,
          int ElementsPerAccess,
          bool ScatterD,
          typename PermuteDLayout>
struct BiasGradEpilogue<cutlass::arch::Sm70,
                        Shape_,
                        WarpMmaTensorOp_,
                        PartitionsK,
                        OutputOp_,
                        ElementsPerAccess,
                        ScatterD,
                        PermuteDLayout> {
    using Epilogue =
        typename cutlass::epilogue::threadblock::EpilogueVoltaTensorOp<Shape_,
                                                                       WarpMmaTensorOp_,
                                                                       PartitionsK,
                                                                       OutputOp_,
                                                                       ElementsPerAccess,
                                                                       ScatterD,
                                                                       PermuteDLayout>::Epilogue;
};

template <typename Arch_,
          int Rank,
          typename Shape_,
          typename WarpMmaTensorOp_,
          int PartitionsK,
          typename OutputOp_,
          int ElementsPerAccess>
struct BiasGradEpilogueAffineRankN {
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueTensorOpAffineRankN<
        Rank,
        Shape_,
        WarpMmaTensorOp_,
        PartitionsK,
        OutputOp_,
        ElementsPerAccess>::Epilogue;
};

template <int Rank,
          typename Shape_,
          typename WarpMmaTensorOp_,
          int PartitionsK,
          typename OutputOp_,
          int ElementsPerAccess>
struct BiasGradEpilogueAffineRankN<cutlass::arch::Sm70,
                                   Rank,
                                   Shape_,
                                   WarpMmaTensorOp_,
                                   PartitionsK,
                                   OutputOp_,
                                   ElementsPerAccess> {
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueVoltaTensorOpAffineRankN<
        Rank,
        Shape_,
        WarpMmaTensorOp_,
        PartitionsK,
        OutputOp_,
        ElementsPerAccess>::Epilogue;
};
