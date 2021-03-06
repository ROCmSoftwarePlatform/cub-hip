#include "hip/hip_runtime.h"

/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * cub::DeviceReduce provides device-wide, parallel operations for computing a reduction across a sequence of data items residing within global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "../../block_sweep/block_reduce_sweep.cuh"
#include "../../iterator/constant_input_iterator.cuh"
#include "../../thread/thread_operators.cuh"
#include "../../grid/grid_even_share.cuh"
#include "../../grid/grid_queue.cuh"
#include "../../iterator/arg_index_input_iterator.cuh"
#include "../../util_debug.cuh"
#include "../../util_device.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * Reduce region kernel entry point (multi-block).  Computes privatized reductions, one per thread block.
 */
template <
    typename                BlockReduceSweepPolicy,     ///< Parameterized BlockReduceSweepPolicy tuning policy type
    typename                InputIterator,              ///< Random-access input iterator type for reading input items \iterator
    typename                OutputIterator,             ///< Output iterator type for recording the reduced aggregate \iterator
    typename                Offset,                     ///< Signed integer type for global offsets
    typename                ReductionOp>                ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt>
__launch_bounds__ (int(BlockReduceSweepPolicy::BLOCK_THREADS))
__global__ void DeviceReduceSweepKernel(
    hipLaunchParm 			lp,							///< HIP grid_launch_parm parameter
    InputIterator           d_in,                       ///< [in] Pointer to the input sequence of data items
    OutputIterator          d_out,                      ///< [out] Pointer to the output aggregate
    Offset                  num_items,                  ///< [in] Total number of input data items
    GridEvenShare<Offset>   even_share,                 ///< [in] Even-share descriptor for mapping an equal number of tiles onto each thread block
    GridQueue<Offset>       queue,                      ///< [in] Drain queue descriptor for dynamically mapping tile data onto thread blocks
    ReductionOp             reduction_op)               ///< [in] Binary reduction functor (e.g., an instance of cub::Sum, cub::Min, cub::Max, etc.)
{
    // Data type
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Thread block type for reducing input tiles
    typedef BlockReduceSweep<BlockReduceSweepPolicy, InputIterator, Offset, ReductionOp> BlockReduceSweepT;

    // Block-wide aggregate
    T block_aggregate;

    // Shared memory storage
    __shared__ typename BlockReduceSweepT::TempStorage temp_storage;

    // Consume input tiles
    BlockReduceSweepT(temp_storage, d_in, reduction_op).ConsumeRange(
        num_items,
        even_share,
        queue,
        block_aggregate,
        Int2Type<BlockReduceSweepPolicy::GRID_MAPPING>());

    // Output result
    if (hipThreadIdx_x == 0)
    {
        d_out[hipBlockIdx_x] = block_aggregate;
    }
}


/**
 * Reduce a single tile kernel entry point (single-block).  Can be used to aggregate privatized threadblock reductions from a previous multi-block reduction pass.
 */
template <
    typename                BlockReduceSweepPolicy,     ///< Parameterized BlockReduceSweepPolicy tuning policy type
    typename                InputIterator,              ///< Random-access input iterator type for reading input items \iterator
    typename                OutputIterator,             ///< Output iterator type for recording the reduced aggregate \iterator
    typename                Offset,                     ///< Signed integer type for global offsets
    typename                ReductionOp>                ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt>
__launch_bounds__ (int(BlockReduceSweepPolicy::BLOCK_THREADS), 1)
__global__ void SingleReduceSweepKernel(
    hipLaunchParm 			lp,							///< HIP grid_launch_parm parameter
    InputIterator           d_in,                       ///< [in] Pointer to the input sequence of data items
    OutputIterator          d_out,                      ///< [out] Pointer to the output aggregate
    Offset                  num_items,                  ///< [in] Total number of input data items
    ReductionOp             reduction_op)               ///< [in] Binary reduction functor (e.g., an instance of cub::Sum, cub::Min, cub::Max, etc.)
{
    // Data type
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Thread block type for reducing input tiles
    typedef BlockReduceSweep<BlockReduceSweepPolicy, InputIterator, Offset, ReductionOp> BlockReduceSweepT;

    // Block-wide aggregate
    T block_aggregate;

    // Shared memory storage
    __shared__ typename BlockReduceSweepT::TempStorage temp_storage;

    // Consume input tiles
    BlockReduceSweepT(temp_storage, d_in, reduction_op).ConsumeRange(
        Offset(0),
        Offset(num_items),
        block_aggregate);

    // Output result
    if (hipThreadIdx_x == 0)
    {
        d_out[hipBlockIdx_x] = block_aggregate;
    }
}




/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceReduce
 */
template <
    typename InputIterator,     ///< Random-access input iterator type for reading input items \iterator
    typename OutputIterator,    ///< Output iterator type for recording the reduced aggregate \iterator
    typename Offset,            ///< Signed integer type for global offsets
    typename ReductionOp>       ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt>
struct DeviceReduceDispatch
{
    // Data type of input iterator
    typedef typename std::iterator_traits<InputIterator>::value_type T;


    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

    /// SM35
    struct Policy350
    {
        // RangeReducePolicy1B (GTX Titan: 228.7 GB/s @ 192M 1B items)
        typedef BlockReduceSweepPolicy<
                128,                                ///< Threads per thread block
                24,                                 ///< Items per thread per tile of input
                4,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_LDG,                           ///< Cache load modifier
                GRID_MAPPING_DYNAMIC>               ///< How to map tiles of input onto thread blocks
            RangeReducePolicy1B;

        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 20,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        // RangeReducePolicy4B (GTX Titan: 255.1 GB/s @ 48M 4B items)
        typedef BlockReduceSweepPolicy<
                256,                                ///< Threads per thread block
                ITEMS_PER_THREAD,                   ///< Items per thread per tile of input
                2,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_LDG,                           ///< Cache load modifier
                GRID_MAPPING_DYNAMIC>               ///< How to map tiles of input onto thread blocks
            RangeReducePolicy4B;

        // RangeReducePolicy
        typedef typename If<(sizeof(T) >= 4),
            RangeReducePolicy4B,
            RangeReducePolicy1B>::Type RangeReducePolicy;

        // SingleTilePolicy
        typedef BlockReduceSweepPolicy<
                256,                                ///< Threads per thread block
                8,                                  ///< Items per thread per tile of input
                1,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_WARP_REDUCTIONS,       ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            SingleTilePolicy;
    };

    /// SM30
    struct Policy300
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 2,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        // RangeReducePolicy (GTX670: 154.0 @ 48M 4B items)
        typedef BlockReduceSweepPolicy<
                256,                                ///< Threads per thread block
                ITEMS_PER_THREAD,                   ///< Items per thread per tile of input
                1,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_WARP_REDUCTIONS,       ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            RangeReducePolicy;

        // SingleTilePolicy
        typedef BlockReduceSweepPolicy<
                256,                                ///< Threads per thread block
                24,                                 ///< Items per thread per tile of input
                4,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_WARP_REDUCTIONS,       ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            SingleTilePolicy;
    };

    /// SM20
    struct Policy200
    {
        // RangeReducePolicy1B (GTX 580: 158.1 GB/s @ 192M 1B items)
        typedef BlockReduceSweepPolicy<
                192,                                ///< Threads per thread block
                24,                                 ///< Items per thread per tile of input
                4,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                (sizeof(T) == 1) ?                  ///< How to map tiles of input onto thread blocks
                    GRID_MAPPING_EVEN_SHARE :
                    GRID_MAPPING_DYNAMIC>
            RangeReducePolicy1B;

        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 8,
            NOMINAL_4B_VEC_ITEMS        = 4,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
            VEC_ITEMS                   = CUB_MIN(NOMINAL_4B_VEC_ITEMS, CUB_MAX(1, (NOMINAL_4B_VEC_ITEMS * 4 / sizeof(T)))),
        };

        // RangeReducePolicy4B (GTX 580: 178.9 GB/s @ 48M 4B items)
        typedef BlockReduceSweepPolicy<
                128,                                ///< Threads per thread block
                ITEMS_PER_THREAD,                   ///< Items per thread per tile of input
                VEC_ITEMS,                          ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_DYNAMIC>               ///< How to map tiles of input onto thread blocks
            RangeReducePolicy4B;

        // RangeReducePolicy
        typedef typename If<(sizeof(T) < 4),
            RangeReducePolicy1B,
            RangeReducePolicy4B>::Type RangeReducePolicy;

        // SingleTilePolicy
        typedef BlockReduceSweepPolicy<
                192,                                ///< Threads per thread block
                7,                                  ///< Items per thread per tile of input
                1,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            SingleTilePolicy;
    };

    /// SM13
    struct Policy130
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 8,
            NOMINAL_4B_VEC_ITEMS        = 2,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
            VEC_ITEMS                   = CUB_MIN(NOMINAL_4B_VEC_ITEMS, CUB_MAX(1, (NOMINAL_4B_VEC_ITEMS * 4 / sizeof(T)))),
        };

        // RangeReducePolicy
        typedef BlockReduceSweepPolicy<
                128,                                ///< Threads per thread block
                ITEMS_PER_THREAD,                   ///< Items per thread per tile of input
                VEC_ITEMS,                          ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            RangeReducePolicy;

        // SingleTilePolicy
        typedef BlockReduceSweepPolicy<
                32,                                 ///< Threads per thread block
                4,                                  ///< Items per thread per tile of input
                VEC_ITEMS,                          ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            SingleTilePolicy;
    };

    /// SM10
    struct Policy100
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 8,
            NOMINAL_4B_VEC_ITEMS        = 2,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
            VEC_ITEMS                   = CUB_MIN(NOMINAL_4B_VEC_ITEMS, CUB_MAX(1, (NOMINAL_4B_VEC_ITEMS * 4 / sizeof(T)))),
        };

        // RangeReducePolicy
        typedef BlockReduceSweepPolicy<
                128,                                ///< Threads per thread block
                ITEMS_PER_THREAD,                   ///< Items per thread per tile of input
                VEC_ITEMS,                          ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            RangeReducePolicy;

        // SingleTilePolicy
        typedef BlockReduceSweepPolicy<
                32,                                 ///< Threads per thread block
                4,                                  ///< Items per thread per tile of input
                4,                                  ///< Number of items per vectorized load
                BLOCK_REDUCE_RAKING,                ///< Cooperative block-wide reduction algorithm to use
                LOAD_DEFAULT,                       ///< Cache load modifier
                GRID_MAPPING_EVEN_SHARE>            ///< How to map tiles of input onto thread blocks
            SingleTilePolicy;
    };


    /******************************************************************************
     * Tuning policies of current PTX compiler pass
     ******************************************************************************/

#if (CUB_PTX_ARCH >= 350)
    typedef Policy350 PtxPolicy;

#elif (CUB_PTX_ARCH >= 300)
    typedef Policy300 PtxPolicy;

#elif (CUB_PTX_ARCH >= 200)
    typedef Policy200 PtxPolicy;

#elif (CUB_PTX_ARCH >= 130)
    typedef Policy130 PtxPolicy;

#else
    typedef Policy100 PtxPolicy;

#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxRangeReducePolicy   : PtxPolicy::RangeReducePolicy {};
    struct PtxSingleTilePolicy     : PtxPolicy::SingleTilePolicy {};


    /******************************************************************************
     * Utilities
     ******************************************************************************/

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitConfigs(
        int             ptx_version,
        KernelConfig    &device_reduce_sweep_config,
        KernelConfig    &single_reduce_sweep_config)
    {
    #if (CUB_PTX_ARCH > 0)

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        device_reduce_sweep_config.template Init<PtxRangeReducePolicy>();
        single_reduce_sweep_config.template Init<PtxSingleTilePolicy>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 350)
        {
            device_reduce_sweep_config.template     Init<typename Policy350::RangeReducePolicy>();
            single_reduce_sweep_config.template     Init<typename Policy350::SingleTilePolicy>();
        }
        else if (ptx_version >= 300)
        {
            device_reduce_sweep_config.template     Init<typename Policy300::RangeReducePolicy>();
            single_reduce_sweep_config.template     Init<typename Policy300::SingleTilePolicy>();
        }
        else if (ptx_version >= 200)
        {
            device_reduce_sweep_config.template     Init<typename Policy200::RangeReducePolicy>();
            single_reduce_sweep_config.template     Init<typename Policy200::SingleTilePolicy>();
        }
        else if (ptx_version >= 130)
        {
            device_reduce_sweep_config.template     Init<typename Policy130::RangeReducePolicy>();
            single_reduce_sweep_config.template     Init<typename Policy130::SingleTilePolicy>();
        }
        else
        {
            device_reduce_sweep_config.template     Init<typename Policy100::RangeReducePolicy>();
            single_reduce_sweep_config.template     Init<typename Policy100::SingleTilePolicy>();
        }

    #endif
    }


    /**
     * Kernel kernel dispatch configuration
     */
    struct KernelConfig
    {
        int                     block_threads;
        int                     items_per_thread;
        int                     vector_load_length;
        BlockReduceAlgorithm    block_algorithm;
        CacheLoadModifier       load_modifier;
        GridMappingStrategy     grid_mapping;

        template <typename BlockPolicy>
        CUB_RUNTIME_FUNCTION __forceinline__
        void Init()
        {
            block_threads               = BlockPolicy::BLOCK_THREADS;
            items_per_thread            = BlockPolicy::ITEMS_PER_THREAD;
            vector_load_length          = BlockPolicy::VECTOR_LOAD_LENGTH;
            block_algorithm             = BlockPolicy::BLOCK_ALGORITHM;
            load_modifier               = BlockPolicy::LOAD_MODIFIER;
            grid_mapping                = BlockPolicy::GRID_MAPPING;
        }

        CUB_RUNTIME_FUNCTION __forceinline__
        void Print()
        {
            printf("%d threads, %d per thread, %d veclen, %d algo, %d loadmod, %d mapping",
                block_threads,
                items_per_thread,
                vector_load_length,
                block_algorithm,
                load_modifier,
                grid_mapping);
        }
    };

    /******************************************************************************
     * Dispatch entrypoints
     ******************************************************************************/

    /**
     * Internal dispatch routine for computing a device-wide reduction using the
     * specified kernel functions.
     *
     * If the input is larger than a single tile, this method uses two-passes of
     * kernel invocations.
     */
    template <
        typename                    DeviceReduceSweepKernelPtr,         ///< Function type of cub::DeviceReduceSweepKernel
        typename                    SingleReducePartialsKernelPtr,      ///< Function type of cub::SingleReduceSweepKernel for consuming partial reductions (T*)
        typename                    SingleReduceSweepKernelPtr,         ///< Function type of cub::SingleReduceSweepKernel for consuming input (InputIterator)
        typename                    FillAndResetDrainKernelPtr>         ///< Function type of cub::FillAndResetDrainKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    static hipError_t Dispatch(
        void                            *d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                          &temp_storage_bytes,            ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator                   d_in,                           ///< [in] Pointer to the input sequence of data items
        OutputIterator                  d_out,                          ///< [out] Pointer to the output aggregate
        Offset                          num_items,                      ///< [in] Total number of input items (i.e., length of \p d_in)
        ReductionOp                     reduction_op,                   ///< [in] Binary reduction functor (e.g., an instance of cub::Sum, cub::Min, cub::Max, etc.)
        hipStream_t                    stream,                         ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                            debug_synchronous,              ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        FillAndResetDrainKernelPtr      prepare_drain_kernel,           ///< [in] Kernel function pointer to parameterization of cub::FillAndResetDrainKernel
        DeviceReduceSweepKernelPtr      device_reduce_sweep_kernel,     ///< [in] Kernel function pointer to parameterization of cub::DeviceReduceSweepKernel
        SingleReducePartialsKernelPtr   single_reduce_partials_kernel,  ///< [in] Kernel function pointer to parameterization of cub::SingleReduceSweepKernel for consuming partial reductions (T*)
        SingleReduceSweepKernelPtr      single_reduce_sweep_kernel,     ///< [in] Kernel function pointer to parameterization of cub::SingleReduceSweepKernel for consuming input (InputIterator)
        KernelConfig                    device_reduce_sweep_config,     ///< [in] Dispatch parameters that match the policy that \p range_reduce_kernel_ptr was compiled for
        KernelConfig                    single_reduce_sweep_config)     ///< [in] Dispatch parameters that match the policy that \p single_reduce_sweep_kernel was compiled for
    {
#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported );

#else
        hipError_t error = hipSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = hipGetDevice(&device_ordinal))) break;

            // Get device SM version
            int sm_version;
            if (CubDebug(error = SmVersion(sm_version, device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = hipDeviceGetAttribute (&sm_count, hipDeviceAttributeMultiprocessorCount, device_ordinal))) break;

            // Tile size of device_reduce_sweep_kernel
            int tile_size = device_reduce_sweep_config.block_threads * device_reduce_sweep_config.items_per_thread;

            if ((device_reduce_sweep_kernel == NULL) || (num_items <= tile_size))
            {
                // Dispatch a single-block reduction kernel

                // Return if the caller is simply requesting the size of the storage allocation
                if (d_temp_storage == NULL)
                {
                    temp_storage_bytes = 1;
                    return hipSuccess;
                }

                // Log single_reduce_sweep_kernel configuration
                if (debug_synchronous) CubLog("Invoking hipLaunchKernel(HIP_KERNEL_NAME(ReduceSingle), dim3(1), dim3(%d), 0, %lld, ), %d items per thread\n",
                    single_reduce_sweep_config.block_threads, (long long) stream, single_reduce_sweep_config.items_per_thread);

                // Invoke single_reduce_sweep_kernel
                hipLaunchKernel(HIP_KERNEL_NAME(single_reduce_sweep_kernel), dim3(1), dim3(single_reduce_sweep_config.block_threads), 0, 0, 
                    d_in,
                    d_out,
                    num_items,
                    reduction_op);

                // Check for failure to launch
                if (CubDebug(error = hipPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            }
            else
            {
                // Dispatch two kernels: (1) a multi-block kernel to compute
                // privatized per-block reductions, and (2) a single-block
                // to reduce those partial reductions

                // Get SM occupancy for device_reduce_sweep_kernel
                int range_reduce_sm_occupancy;
                if (CubDebug(error = MaxSmOccupancy(
                    range_reduce_sm_occupancy,
                    sm_version,
                    device_reduce_sweep_kernel,
                    device_reduce_sweep_config.block_threads))) break;

                // Get device occupancy for device_reduce_sweep_kernel
                int range_reduce_occupancy = range_reduce_sm_occupancy * sm_count;

                // Even-share work distribution
                int subscription_factor = range_reduce_sm_occupancy;     // Amount of CTAs to oversubscribe the device beyond actively-resident (heuristic)
                GridEvenShare<Offset> even_share(
                    num_items,
                    range_reduce_occupancy * subscription_factor,
                    tile_size);

                // Get grid size for device_reduce_sweep_kernel
                int range_reduce_grid_size;
                switch (device_reduce_sweep_config.grid_mapping)
                {
                case GRID_MAPPING_EVEN_SHARE:

                    // Work is distributed evenly
                    range_reduce_grid_size = even_share.grid_size;
                    break;

                case GRID_MAPPING_DYNAMIC:

                    // Work is distributed dynamically
                    int num_tiles = (num_items + tile_size - 1) / tile_size;
                    range_reduce_grid_size = (num_tiles < range_reduce_occupancy) ?
                        num_tiles :                     // Not enough to fill the device with threadblocks
                        range_reduce_occupancy;         // Fill the device with threadblocks
                    break;
                };

                // Temporary storage allocation requirements
                void* allocations[2];
                size_t allocation_sizes[2] =
                {
                    range_reduce_grid_size * sizeof(T),     // bytes needed for privatized block reductions
                    GridQueue<int>::AllocationSize()        // bytes needed for grid queue descriptor
                };

                // Alias the temporary allocations from the single storage blob (or set the necessary size of the blob)
                if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
                if (d_temp_storage == NULL)
                {
                    // Return if the caller is simply requesting the size of the storage allocation
                    return hipSuccess;
                }

                // Alias the allocation for the privatized per-block reductions
                T *d_block_reductions = (T*) allocations[0];

                // Alias the allocation for the grid queue descriptor
                GridQueue<Offset> queue(allocations[1]);

                // Prepare the dynamic queue descriptor if necessary
                if (device_reduce_sweep_config.grid_mapping == GRID_MAPPING_DYNAMIC)
                {
                    // Prepare queue using a kernel so we know it gets prepared once per operation
                    if (debug_synchronous) CubLog("Invoking hipLaunchKernel(HIP_KERNEL_NAME(prepare_drain_kernel), dim3(1), dim3(1), 0, %lld, )\n", (long long) stream);

                    // Invoke prepare_drain_kernel
                    hipLaunchKernel(HIP_KERNEL_NAME(prepare_drain_kernel), dim3(1), dim3(1), 0, stream, queue, num_items);

                    // Check for failure to launch
                    if (CubDebug(error = hipPeekAtLastError())) break;

                    // Sync the stream if specified to flush runtime errors
                    if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
                }

                // Log device_reduce_sweep_kernel configuration
                if (debug_synchronous) CubLog("Invoking hipLaunchKernel(HIP_KERNEL_NAME(device_reduce_sweep_kernel), dim3(%d), dim3(%d), 0, %lld, ), %d items per thread, %d SM occupancy\n",
                    range_reduce_grid_size, device_reduce_sweep_config.block_threads, (long long) stream, device_reduce_sweep_config.items_per_thread, range_reduce_sm_occupancy);

                // Invoke device_reduce_sweep_kernel
                hipLaunchKernel(HIP_KERNEL_NAME(device_reduce_sweep_kernel), dim3(range_reduce_grid_size), dim3(device_reduce_sweep_config.block_threads), 0, stream, 
                    d_in,
                    d_block_reductions,
                    num_items,
                    even_share,
                    queue,
                    reduction_op);

                // Check for failure to launch
                if (CubDebug(error = hipPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

                // Log single_reduce_sweep_kernel configuration
                if (debug_synchronous) CubLog("Invoking hipLaunchKernel(HIP_KERNEL_NAME(single_reduce_sweep_kernel), dim3(%d), dim3(%d), 0, %lld, ), %d items per thread\n",
                    1, single_reduce_sweep_config.block_threads, (long long) stream, single_reduce_sweep_config.items_per_thread);

                // Invoke single_reduce_sweep_kernel
                hipLaunchKernel(HIP_KERNEL_NAME(single_reduce_partials_kernel), dim3(1), dim3(single_reduce_sweep_config.block_threads), 0, stream, 
                    d_block_reductions,
                    d_out,
                    range_reduce_grid_size,
                    reduction_op);

                // Check for failure to launch
                if (CubDebug(error = hipPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
            }
        }
        while (0);

        return error;

#endif // CUB_RUNTIME_ENABLED
    }


    /**
     * Internal dispatch routine for computing a device-wide reduction
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static hipError_t Dispatch(
        void                        *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIterator               d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIterator              d_out,                              ///< [out] Pointer to the output aggregate
        Offset                      num_items,                          ///< [in] Total number of input items (i.e., length of \p d_in)
        ReductionOp                 reduction_op,                       ///< [in] Binary reduction functor (e.g., an instance of cub::Sum, cub::Min, cub::Max, etc.)
        hipStream_t                stream,                             ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous)                  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        hipError_t error = hipSuccess;
        do
        {
            // Get PTX version
            int ptx_version;
    #if (CUB_PTX_ARCH == 0)
            if (CubDebug(error = PtxVersion(ptx_version))) break;
    #else
            ptx_version = CUB_PTX_ARCH;
    #endif

            // Get kernel kernel dispatch configurations
            KernelConfig device_reduce_sweep_config;
            KernelConfig single_reduce_sweep_config;
            InitConfigs(ptx_version, device_reduce_sweep_config, single_reduce_sweep_config);

            // Dispatch
            if (CubDebug(error = Dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                num_items,
                reduction_op,
                stream,
                debug_synchronous,
                FillAndResetDrainKernel<Offset>,
                DeviceReduceSweepKernel<PtxRangeReducePolicy, InputIterator, T*, Offset, ReductionOp>,
                SingleReduceSweepKernel<PtxSingleTilePolicy, T*, OutputIterator, Offset, ReductionOp>,
                SingleReduceSweepKernel<PtxSingleTilePolicy, InputIterator, OutputIterator, Offset, ReductionOp>,
                device_reduce_sweep_config,
                single_reduce_sweep_config))) break;
        }
        while (0);

        return error;
    }
};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


