/******************************************************************************
 * Copyright (c) 2011-2017, NVIDIA CORPORATION.  All rights reserved.
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

#include <test/test_util.h>

#include <hip/hip_runtime.h>

namespace histogram_gmem_atomics
{
    // Decode float4 pixel into bins
    template <int NUM_BINS, int ACTIVE_CHANNELS>
    __device__ __forceinline__ void DecodePixel(float4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
    {
        float* samples = reinterpret_cast<float*>(&pixel);

        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
            bins[CHANNEL] = (unsigned int) (samples[CHANNEL] * float(NUM_BINS));
    }

    // Decode uchar4 pixel into bins
    template <int NUM_BINS, int ACTIVE_CHANNELS>
    __device__ __forceinline__ void DecodePixel(uchar4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
    {
        unsigned char* samples = reinterpret_cast<unsigned char*>(&pixel);

        #pragma unroll
        for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
            bins[CHANNEL] = (unsigned int) (samples[CHANNEL]);
    }

    // Decode uchar1 pixel into bins
    template <int NUM_BINS, int ACTIVE_CHANNELS>
    __device__ __forceinline__ void DecodePixel(uchar1 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
    {
        bins[0] = (unsigned int) pixel.x;
    }

    // First-pass histogram kernel (binning into privatized counters)
    template <
        int         NUM_PARTS,
        int         ACTIVE_CHANNELS,
        int         NUM_BINS,
        typename    PixelType>
    __global__ void histogram_gmem_atomics(
        const PixelType *in,
        int width,
        int height,
        unsigned int *out)
    {
        // global position and size
        int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
        int nx = hipBlockDim_x * hipGridDim_x;
        int ny = hipBlockDim_y * hipGridDim_y;

        // threads in workgroup
        int t = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x; // thread index in workgroup, linear in 0..nt-1
        int nt = hipBlockDim_x * hipBlockDim_y; // total threads in workgroup

        // group index in 0..ngroups-1
        int g = hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x;

        // initialize smem
        unsigned int *gmem = out + g * NUM_PARTS;
        for (int i = t; i < ACTIVE_CHANNELS * NUM_BINS; i += nt)
            gmem[i] = 0;
        __syncthreads();

        // process pixels (updates our group's partial histogram in gmem)
        for (int col = x; col < width; col += nx)
        {
            for (int row = y; row < height; row += ny)
            {
                PixelType pixel = in[row * width + col];

                unsigned int bins[ACTIVE_CHANNELS];
                DecodePixel<NUM_BINS>(pixel, bins);

                #pragma unroll
                for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
                    atomicAdd(&gmem[(NUM_BINS * CHANNEL) + bins[CHANNEL]], 1);
            }
        }
    }

    // Second pass histogram kernel (accumulation)
    template <
        int         NUM_PARTS,
        int         ACTIVE_CHANNELS,
        int         NUM_BINS>
    __global__ void histogram_gmem_accum(
        const unsigned int *in,
        int n,
        unsigned int *out)
    {
        int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        if (i > ACTIVE_CHANNELS * NUM_BINS)
            return; // out of range

        unsigned int total = 0;
        for (int j = 0; j < n; j++)
            total += in[i + NUM_PARTS * j];

        out[i] = total;
    }


}   // namespace histogram_gmem_atomics


template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
double run_gmem_atomics(
    PixelType *d_image,
    int width,
    int height,
    unsigned int *d_hist,
    bool warmup)
{
    enum
    {
        NUM_PARTS = 1024
    };

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);

    dim3 block(32, 4);
    dim3 grid(16, 16);
    int total_blocks = grid.x * grid.y;

    // allocate partial histogram
    unsigned int *d_part_hist;
    hipMalloc(&d_part_hist, total_blocks * NUM_PARTS * sizeof(unsigned int));

    dim3 block2(128);
    dim3 grid2((3 * NUM_BINS + block.x - 1) / block.x);

    GpuTimer gpu_timer;
    gpu_timer.Start();

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            histogram_gmem_atomics::histogram_gmem_atomics<
                NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS>),
        dim3(grid),
        dim3(block),
        0,
        0,
        static_cast<const PixelType*>(d_image),
        width,
        height,
        d_part_hist);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            histogram_gmem_atomics::histogram_gmem_accum<
                NUM_PARTS, ACTIVE_CHANNELS, NUM_BINS>),
        dim3(grid2),
        dim3(block2),
        0,
        0,
        static_cast<const unsigned int*>(d_part_hist),
        total_blocks,
        d_hist);

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    hipFree(d_part_hist);

    return elapsed_millis;
}

