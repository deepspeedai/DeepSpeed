// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/threadblock/predicated_tile_iterator.h>
#include <cutlass/tensor_coord.h>
namespace cutlass {
namespace epilogue {
namespace threadblock {

template <class AccessType, class Enable = void>
struct atomic_store {};

template <class AccessType>
struct atomic_store<AccessType,
                    typename platform::enable_if<
                        platform::is_same<typename AccessType::Element, half_t>::value>::type> {
    using Element = typename AccessType::Element;
    static const int kCount = AccessType::kElements;

    CUTLASS_DEVICE
    atomic_store(AccessType const& D, void* ptr, bool pred_guard)
    {
        static_assert(!(kCount % 2), "kCount must be even");
        half2* p = reinterpret_cast<half2*>(ptr);
        uint const* data = reinterpret_cast<uint const*>(&D);
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %0, 0;\n"
            :
            : "r"((int)pred_guard));
        for (int i = 0; i < kCount / 2; i++) {
            asm volatile("  @p red.relaxed.global.add.noftz.f16x2  [%0], %1;\n"
                         :
                         : "l"(p + i), "r"(data[i]));
        }
        asm volatile("}\n" ::);
    }
};

template <class AccessType>
struct atomic_store<AccessType,
                    typename platform::enable_if<
                        platform::is_same<typename AccessType::Element, float>::value>::type> {
    using Element = typename AccessType::Element;
    static const int kCount = AccessType::kElements;

    CUTLASS_DEVICE
    atomic_store(AccessType const& D, void* ptr, bool pred_guard)
    {
        Element* p = reinterpret_cast<Element*>(ptr);
        uint const* data = reinterpret_cast<uint const*>(&D);
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %0, 0;\n"
            :
            : "r"((int)pred_guard));
        for (int i = 0; i < kCount; i++) {
            asm volatile("  @p red.relaxed.global.add.f32  [%0], %1;\n"
                         :
                         : "l"(p + i), "r"(data[i]));
        }
        asm volatile("}\n" ::);
    }
};

template <typename ThreadMap_,  ///< Thread map (conept: OutputTileThreadMap)
          typename Element_,    ///< Element data type
          int Rank>
class PredicatedTileIteratorAffineRankNAtomic {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout = layout::AffineRankN<Rank>;
    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
    static int const kThreads = ThreadMap::kThreads;
    static int const kIterations = ThreadMap::Count::kTile;

    static_assert(ThreadMap::Iterations::kRow > 0, "ThreadMap::Iterations::kRow must be > 0");
    static_assert(ThreadMap::Iterations::kGroup > 0, "ThreadMap::Iterations::kGroup must be > 0");
    static_assert(ThreadMap::Iterations::kCluster > 0,
                  "ThreadMap::Iterations::kCluster must be > 0");
    static_assert(ThreadMap::Iterations::kColumn > 0, "ThreadMap::Iterations::kColumn must be > 0");
    static_assert(!(Layout::kRank % 2),
                  "Layout rank must be even. This assumes the first half of the "
                  "modes correspond to the 'row' "
                  "and the second half of the modes correspond to the 'column'");

    static bool const kBigEndian = false;

    /// Fragment object
    using Fragment = Array<Element,
                           ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow *
                               ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster *
                               ThreadMap::kElementsPerAccess>;

    /// Memory access size
    using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

    //
    // Parameters struct
    //

    /// Parameters structure
    struct Params {
        //
        // Data members
        //

        Layout layout;

        /// Stride in units of bytes along M modes
        Coord<Layout::kRank / 2, typename Layout::LongIndex> stride_m;

        /// Stride in units of bytes along N modes
        Coord<Layout::kRank / 2, typename Layout::LongIndex> stride_n;

        /// Fast divmod objects divided by tensor extents
        FastDivmod divmod_m[(Layout::kRank == 2) ? 1 : (Layout::kRank / 2 - 1)];

        /// Fast divmod objects divided by tensor extents
        FastDivmod divmod_n[(Layout::kRank == 2) ? 1 : (Layout::kRank / 2 - 1)];

        int64_t rank2_inc_col;
        int64_t rank2_inc_row;

        //
        // Methods
        //
        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        Params(TensorCoord const& extent, Layout const& layout_) : layout(layout_)
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < Layout::kRank / 2; ++i) {
                stride_m[i] = OffsetBytes<Element>(layout_.stride()[i]);
                stride_n[i] = OffsetBytes<Element>(layout_.stride()[i + Layout::kRank / 2]);
            }

            if (kBigEndian) {
                // "Big Endian" scheme
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < Layout::kRank / 2 - 1; ++i) {
                    divmod_m[i] = FastDivmod(extent[i + 1]);
                    divmod_n[i] = FastDivmod(extent[i + Layout::kRank / 2 + 1]);
                }
            } else {
                // "Little Endian" scheme
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < Layout::kRank / 2 - 1; ++i) {
                    divmod_m[i] = FastDivmod(extent[i]);
                    divmod_n[i] = FastDivmod(extent[i + Layout::kRank / 2]);
                }
            }
        }

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout_) : layout(layout_)
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < Layout::kRank / 2; ++i) {
                stride_m[i] = OffsetBytes<Element>(layout_.stride()[i]);
                stride_n[i] = OffsetBytes<Element>(layout_.stride()[i + Layout::kRank / 2]);
            }

            rank2_inc_col = ThreadMap::Delta::kColumn * stride_n[0];
            rank2_inc_row = ThreadMap::Delta::kRow * stride_m[0];
        }
    };

    /// Mask object
    struct Mask {
        static int const kCount = ThreadMap::Iterations::kColumn;

        /// Predicate state
        bool predicates[kCount];

        //
        // Mask
        //
        CUTLASS_HOST_DEVICE
        Mask() { enable(); }

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear()
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) { predicates[i] = false; }
        }

        ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
        CUTLASS_DEVICE void enable()
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) { predicates[i] = true; }
        }
    };

private:
    //
    // Data members
    //

    /// Parameters structure containing reference and precomputed state.
    Params params_;

    /// Byte-level pointer
    uint8_t* byte_pointer_;

    /// Array of boolean values to contain steady-state predicates
    Mask mask_;

    /// Extent of the matrix tile in rows
    Index extent_row_;

    /// Extent of the matrix tile in columns
    Index extent_col_;

    /// A thread's starting row position (assuming steady-state predicates have
    /// been computed)
    Index thread_start_row_;

    /// A thread's starting column position (assuming steady-state predicates have
    /// been computed)
    Index thread_start_column_;

    /// Internal state counter
    int state_[3];

    /// Offsets in columns, cached for performance
    int64_t offset_modes_n_[ThreadMap::Iterations::kColumn];

    //
    // Static asserts about internal strides
    //

    static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
    static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");

private:
    //
    // Methods
    //

public:
    //
    // Methods
    //

    /// Constructor
    CUTLASS_DEVICE
    PredicatedTileIteratorAffineRankNAtomic(
        Params const& params,
        Element* pointer,
        MatrixCoord extent,
        int thread_idx,
        MatrixCoord threadblock_offset = MatrixCoord(),
        int const* indices = nullptr  ///< gather/scatter indices, note no support for
                                      ///< gather/scatter at this specialization
        )
        : params_(params)
    {
        MatrixCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

        extent_row_ = extent.row();
        extent_col_ = extent.column();

        thread_start_row_ = thread_offset.row();
        thread_start_column_ = thread_offset.column();

        if (Layout::kRank > 2) {
            // Initialize predicates
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
                //
                // Compute coordinate and decompose into N modes
                //

                int coord_n = thread_start_column_ + c * ThreadMap::Delta::kColumn;

                mask_.predicates[c] = coord_n < extent.column();

                Coord<Layout::kRank / 2, Index> modes_n;

                int64_t offset_modes_n = 0;

                if (kBigEndian) {
                    modes_n = CoordinateDecomposition<Layout::kRank / 2>(coord_n, params_.divmod_n);

                    offset_modes_n = dot(modes_n, params_.stride_n);
                } else {
                    modes_n = CoordinateDecompositionLittleEndian<Layout::kRank / 2>(
                        coord_n, params_.divmod_n);

                    offset_modes_n = dot(modes_n, params_.stride_n);
                }

                offset_modes_n_[c] = offset_modes_n;
            }

            if (!pointer) { mask_.clear(); }
        }

        // Initialize pointer
        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer);

        // Initialize internal state counter
        state_[0] = state_[1] = state_[2] = 0;
    }

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset)
    {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset)
    {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
                int row_begin = thread_start_row_ + group * ThreadMap::Delta::kGroup +
                                cluster * ThreadMap::Delta::kCluster;
                int64_t offset_modes_m = row_begin * params_.stride_m[0];

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int frag_row_idx =
                        (row + ThreadMap::Iterations::kRow *
                                   (group + ThreadMap::Iterations::kGroup * cluster));

                    //
                    // Compute coordinate and decompose into M modes
                    //

                    int coord_m = row * ThreadMap::Delta::kRow + row_begin;

                    Coord<Layout::kRank / 2, Index> modes_m;

                    if (Layout::kRank > 2) {
                        if (kBigEndian) {
                            modes_m = CoordinateDecomposition<Layout::kRank / 2>(coord_m,
                                                                                 params_.divmod_m);
                        } else {
                            modes_m = CoordinateDecompositionLittleEndian<Layout::kRank / 2>(
                                coord_m, params_.divmod_m);
                        }

                        offset_modes_m = dot(modes_m, params_.stride_m);
                    }

                    //
                    // Compute the offset due to modes M
                    //

                    bool row_guard = (coord_m < extent_row_);
                    int64_t offset_modes_n = thread_start_column_ * params_.stride_n[0];

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
                        //
                        // Compute coordinate and decompose into N modes
                        //

                        if (Layout::kRank > 2) { offset_modes_n = offset_modes_n_[column]; }

                        //
                        // Compute the pointer and access
                        //
                        bool guard;
                        if (Layout::kRank > 2) {
                            guard = row_guard && mask_.predicates[column];
                        } else {
                            guard = (coord_m < extent_row_) &&
                                    ((thread_start_column_ + ThreadMap::Delta::kColumn * column) <
                                     extent_col_);
                        }

                        atomic_store<AccessType>(
                            frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                            (void*)(byte_pointer + offset_modes_m + offset_modes_n + byte_offset),
                            guard);

                        if (Layout::kRank == 2) { offset_modes_n += params_.rank2_inc_col; }
                    }

                    if (Layout::kRank == 2) { offset_modes_m += params_.rank2_inc_row; }
                }
            }
        }
    }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void load(Fragment& frag) {}

    /// Advances to the next position to load or store
    CUTLASS_HOST_DEVICE
    PredicatedTileIteratorAffineRankNAtomic& operator++()
    {
        ++state_[0];
        thread_start_row_ += ThreadMap::Shape::kRow;

        if (state_[0] == ThreadMap::Count::kRow) {
            state_[0] = 0;
            ++state_[1];

            thread_start_row_ +=
                (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

            if (state_[1] == ThreadMap::Count::kGroup) {
                state_[1] = 0;
                ++state_[2];

                thread_start_row_ += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
                                     ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

                if (state_[2] == ThreadMap::Count::kCluster) { state_[2] = 0; }
            }
        }

        return *this;
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_DEVICE void clear_mask() { mask_.clear(); }

    ///< Efficiently enables all accesses guarded by mask
    CUTLASS_DEVICE void enable_mask() { mask_.enable(); }

    ///< Sets the mask
    CUTLASS_DEVICE void get_mask(Mask& mask) { mask = mask_; }

    ///< Sets the mask
    CUTLASS_DEVICE void set_mask(Mask const& mask) { mask_ = mask; }
};

template <typename ThreadMap_,    ///< Thread map (conept: OutputTileThreadMap)
          typename Element_,      ///< Element data type
          bool ScatterD = false,  ///< Scatter D operand or not
          typename PermuteDLayout = layout::NoPermute,  ///< Permute D operand or not
          bool UseCUDAStore = false>
class PredicatedTileIteratorAtomic {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout = layout::RowMajor;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = MatrixCoord;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
    static int const kThreads = ThreadMap::kThreads;
    static int const kIterations = ThreadMap::Count::kTile;

    static bool constexpr PermuteD = !layout::is_trivial_permute<PermuteDLayout>;

    static_assert(ThreadMap::Iterations::kRow > 0, "ThreadMap::Iterations::kRow must be > 0");
    static_assert(ThreadMap::Iterations::kGroup > 0, "ThreadMap::Iterations::kGroup must be > 0");
    static_assert(ThreadMap::Iterations::kCluster > 0,
                  "ThreadMap::Iterations::kCluster must be > 0");
    static_assert(ThreadMap::Iterations::kColumn > 0, "ThreadMap::Iterations::kColumn must be > 0");

    /// Fragment object
    using Fragment = Array<Element,
                           ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow *
                               ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster *
                               ThreadMap::kElementsPerAccess>;

    /// Memory access size
    using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

    //
    // Parameters struct
    //

    /// Uses a non-template class
    struct Params : PredicatedTileIteratorParams {
        using Base = PredicatedTileIteratorParams;

        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout)
            : PredicatedTileIteratorParams(
                  layout.stride(0) * int(sizeof(AccessType)) / kElementsPerAccess,
                  make_OutputTileThreadMapDesc<ThreadMap>())
        {
        }

        CUTLASS_HOST_DEVICE
        Params(Base const& base) : Base(base) {}
    };

    /// Mask object
    struct Mask {
        static int const kCount = ThreadMap::Iterations::kColumn;

        /// Predicate state
        bool predicates[kCount];

        //
        // Mask
        //
        CUTLASS_HOST_DEVICE
        Mask() { enable(); }

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear()
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) { predicates[i] = false; }
        }

        ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
        CUTLASS_DEVICE void enable()
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) { predicates[i] = true; }
        }
    };

private:
    //
    // Data members
    //

    /// Parameters structure containing reference and precomputed state.
    PredicatedTileIteratorParams params_;

    /// Byte-level pointer. This pointer is usually for both load() and store(),
    /// unless PermuteD is performed. When having PermuteD, byte_pointer_ is only
    /// for load().
    uint8_t* byte_pointer_;

    /// Byte-level pointer for store(). Due to PermuteD Op, store_byte_pointer_
    /// may be with different address computation compared to byte_pointer_.
    uint8_t* store_byte_pointer_;

    /// Array of boolean values to contain steady-state predicates
    Mask mask_;

    /// Extent of the matrix tile in rows
    Index extent_row_;

    /// Extent of the matrix tile in rows
    Index extent_column_;

    /// A thread's starting row position (assuming steady-state predicates have
    /// been computed)
    Index thread_start_row_;

    /// A thread's starting column
    Index thread_start_column_;

    /// Internal state counter
    int state_[3];

    /// Scatter indices
    int const* indices_;

    /// PermuteDLayout
    PermuteDLayout permute_layout_;

    //
    // Static asserts about internal strides
    //

    static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
    static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
    static_assert(sizeof(PredicatedTileIteratorParams::stride) == 8, "Expected 64b strides");

private:
    //
    // Methods
    //

public:
    //
    // Methods
    //

    /// Constructor
    CUTLASS_DEVICE
    PredicatedTileIteratorAtomic(PredicatedTileIteratorParams const& params,
                                 Element* pointer,
                                 TensorCoord extent,
                                 int thread_idx,
                                 TensorCoord threadblock_offset = TensorCoord(),
                                 int const* indices = nullptr)
        : params_(params),
          indices_(indices),
          permute_layout_(PitchLinearCoord(extent.column(), extent.row()),
                          params_.stride * kElementsPerAccess / sizeof(AccessType))
    {
        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

        extent_row_ = extent.row();
        extent_column_ = extent.column();

        thread_start_row_ = thread_offset.row();
        thread_start_column_ = thread_offset.column();

        // Initialize predicates
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
            mask_.predicates[c] =
                ((thread_offset.column() + ThreadMap::Delta::kColumn * c) < extent.column());
        }

        // Null pointer performs no accesses
        if (!pointer) { mask_.clear(); }

        if (ScatterD && !indices) { mask_.clear(); }

        // Initialize byte_pointer_
        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                        LongIndex(thread_offset.row()) * LongIndex(params_.stride) +
                        LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;

        if (ScatterD) {
            byte_pointer_ =
                reinterpret_cast<uint8_t*>(pointer) +
                LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;
        }

        // store_byte_pointer_ is set to be the same with byte_pointer_ unless
        // PermuteD is used.
        store_byte_pointer_ = PermuteD ? reinterpret_cast<uint8_t*>(pointer) : byte_pointer_;

        // Initialize internal state counter
        state_[0] = state_[1] = state_[2] = 0;
    }

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset)
    {
        store_byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) const
    {
        uint8_t* byte_pointer = store_byte_pointer_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int frag_row_idx =
                        (row + ThreadMap::Iterations::kRow *
                                   (group + ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow +
                                     group * ThreadMap::Delta::kGroup +
                                     cluster * ThreadMap::Delta::kCluster;

                    bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

                    AccessType* memory_pointer =
                        reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

                    if (ScatterD && row_guard) {
                        assert(indices_);

                        memory_pointer = reinterpret_cast<AccessType*>(
                            byte_pointer + byte_offset +
                            LongIndex(indices_[row_offset + thread_start_row_]) *
                                LongIndex(params_.stride));
                    }

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
                        bool guard = row_guard && mask_.predicates[column];

                        if (PermuteD) {
                            int col_offset = column * ThreadMap::Delta::kColumn;

                            int col = col_offset + thread_start_column_;
                            int row = row_offset + thread_start_row_;

                            // Locate memory_pointer
                            memory_pointer = reinterpret_cast<AccessType*>(
                                byte_pointer + byte_offset +
                                permute_layout_(PitchLinearCoord(col, row)) * sizeof(AccessType) /
                                    kElementsPerAccess);
                        }
                        atomic_store<AccessType>(
                            frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                            (void*)&memory_pointer[0],
                            guard);

                        if (!PermuteD) {
                            memory_pointer += (ThreadMap::Delta::kColumn / kElementsPerAccess);
                        }
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        if (!ScatterD && !PermuteD) { byte_pointer += params_.increment_row; }
                    }
                }

                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    byte_pointer += params_.increment_group;
                }
            }

            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                byte_pointer += params_.increment_cluster;
            }
        }
    }

    /// Stores a fragment to memory
    CUTLASS_DEVICE
    void store(Fragment const& frag) const { store_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void load(Fragment& frag) {}

    CUTLASS_DEVICE
    MatrixCoord thread_start() const
    {
        return MatrixCoord(thread_start_row_, thread_start_column_);
    }

    /// Need to get the thread start row from the tile iterator
    CUTLASS_DEVICE
    int32_t thread_start_row() const { return thread_start_row_; }

    /// Need to get the thread start row from the tile iterator
    CUTLASS_DEVICE
    int32_t thread_start_column() const { return thread_start_column_; }

    /// Extent of the matrix in rows
    CUTLASS_DEVICE
    Index extent_row() const { return extent_row_; }

    /// Extent of the matrix in columns
    CUTLASS_DEVICE
    Index extent_column() const { return extent_column_; }

    /// Advances to the next position to load or store
    CUTLASS_HOST_DEVICE
    PredicatedTileIteratorAtomic& operator++()
    {
        ++state_[0];

        if (!ScatterD && !PermuteD) { store_byte_pointer_ += params_.advance_row; }

        if (!ScatterD) { byte_pointer_ += params_.advance_row; }

        thread_start_row_ += ThreadMap::Shape::kRow;

        if (state_[0] == ThreadMap::Count::kRow) {
            state_[0] = 0;
            ++state_[1];
            byte_pointer_ += params_.advance_group;
            store_byte_pointer_ += params_.advance_group;

            thread_start_row_ +=
                (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

            if (state_[1] == ThreadMap::Count::kGroup) {
                state_[1] = 0;
                ++state_[2];
                byte_pointer_ += params_.advance_cluster;
                store_byte_pointer_ += params_.advance_cluster;

                thread_start_row_ += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
                                     ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

                if (state_[2] == ThreadMap::Count::kCluster) {
                    state_[2] = 0;
                    byte_pointer_ += params_.advance_tile;
                    store_byte_pointer_ += params_.advance_tile;

                    thread_start_row_ += ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow *
                                         ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile;
                }
            }
        }

        return *this;
    }

    /// Advances a number of positions to load or store
    CUTLASS_HOST_DEVICE
    PredicatedTileIteratorAtomic& operator+=(int increment)
    {
        // Row
        state_[0] += increment;
        int increment_row = state_[0] / ThreadMap::Count::kRow;
        state_[0] = state_[0] % ThreadMap::Count::kRow;

        byte_pointer_ += (params_.advance_row * increment);
        store_byte_pointer_ += (params_.advance_row * increment);
        thread_start_row_ += (ThreadMap::Shape::kRow * increment);

        // Group
        state_[1] += increment_row;
        int increment_group = state_[1] / ThreadMap::Count::kGroup;
        state_[1] = state_[1] % ThreadMap::Count::kGroup;

        byte_pointer_ += (params_.advance_group * increment_row);
        store_byte_pointer_ += (params_.advance_group * increment_row);
        thread_start_row_ += (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow *
                             ThreadMap::Count::kRow * increment_row;

        // Cluster
        state_[2] += increment_group;
        int increment_cluster = state_[2] / ThreadMap::Count::kCluster;
        state_[2] = state_[2] % ThreadMap::Count::kCluster;

        byte_pointer_ += (params_.advance_cluster * increment_group);
        store_byte_pointer_ += (params_.advance_cluster * increment_group);
        thread_start_row_ += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
                             ThreadMap::Count::kRow * ThreadMap::Shape::kRow * increment_group;

        // Tile
        byte_pointer_ += (params_.advance_tile * increment_cluster);
        store_byte_pointer_ += (params_.advance_tile * increment_cluster);
        thread_start_row_ += ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow *
                             ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile *
                             increment_cluster;

        return *this;
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_DEVICE void clear_mask() { mask_.clear(); }

    ///< Efficiently enables all accesses guarded by mask
    CUTLASS_DEVICE void enable_mask() { mask_.enable(); }

    ///< Sets the mask
    CUTLASS_DEVICE void get_mask(Mask& mask) const { mask = mask_; }

    ///< Sets the mask
    CUTLASS_DEVICE void set_mask(Mask const& mask) { mask_ = mask; }
};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass
