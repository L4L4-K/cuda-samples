/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// JP: この file では memory ownership と host/device transfer、kernel launch と thread indexing、stream/event による非同期実行と同期 を確認します。英語の識別子/API/出力文字列は保持します。

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#include "../inc/piestimator.h"

namespace cg = cooperative_groups;
#include <curand_kernel.h>

using std::string;
using std::vector;

// RNG init kernel
// JP: `curandState`: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
__global__ void initRNG(curandState *const rngStates, const unsigned int seed)
{
    // Determine thread ID
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}

__device__ unsigned int reduce_sum(unsigned int in, cg::thread_block cta)
{
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    extern __shared__ unsigned int sdata[];

    // Perform first level of reduction:
    // - Write to shared memory
    // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    unsigned int ltid = threadIdx.x;

    sdata[ltid] = in;
    // JP: sync: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    cg::sync(cta);

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (ltid < s) {
            sdata[ltid] += sdata[ltid + s];
        }

        // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
        cg::sync(cta);
    }

    return sdata[0];
}

// JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
__device__ inline void getPoint(float &x, float &y, curandState &state)
{
    x = curand_uniform(&state);
    y = curand_uniform(&state);
}
__device__ inline void getPoint(double &x, double &y, curandState &state)
{
    x = curand_uniform_double(&state);
    y = curand_uniform_double(&state);
}

// Estimator kernel
template <typename Real>
__global__ void computeValue(unsigned int *const results, curandState *const rngStates, const unsigned int numSims)
{
    // Handle to thread block group
    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    cg::thread_block cta = cg::this_thread_block();
    // Determine thread ID
    unsigned int bid  = blockIdx.x;
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // Initialise the RNG
    // JP: この anchor では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    curandState localState = rngStates[tid];

    // Count the number of points which lie inside the unit quarter-circle
    unsigned int pointsInside = 0;

    for (unsigned int i = tid; i < numSims; i += step) {
        Real x;
        Real y;
        getPoint(x, y, localState);
        Real l2norm2 = x * x + y * y;

        if (l2norm2 < static_cast<Real>(1)) {
            pointsInside++;
        }
    }

    // Reduce within the block
    pointsInside = reduce_sum(pointsInside, cta);

    // Store the result
    // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    if (threadIdx.x == 0) {
        results[bid] = pointsInside;
    }
}

template <typename Real>
PiEstimator<Real>::PiEstimator(unsigned int numSims,
                               unsigned int device,
                               unsigned int threadBlockSize,
                               unsigned int seed)
    : m_numSims(numSims)
    , m_device(device)
    , m_threadBlockSize(threadBlockSize)
    , m_seed(seed)
{
}

template <typename Real> Real PiEstimator<Real>::operator()()
{
    cudaError_t               cudaResult = cudaSuccess;
    struct cudaDeviceProp     deviceProperties;
    struct cudaFuncAttributes funcAttributes;

    // Get device properties
    cudaResult = cudaGetDeviceProperties(&deviceProperties, m_device);

    if (cudaResult != cudaSuccess) {
        string msg("Could not get device properties: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Check precision is valid
    if (typeid(Real) == typeid(double)
        && (deviceProperties.major < 1 || (deviceProperties.major == 1 && deviceProperties.minor < 3))) {
        throw std::runtime_error("Device does not have double precision support");
    }

    // Attach to GPU
    cudaResult = cudaSetDevice(m_device);

    if (cudaResult != cudaSuccess) {
        string msg("Could not set CUDA device: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Determine how to divide the work between cores
    dim3 block;
    dim3 grid;
    block.x = m_threadBlockSize;
    grid.x  = (m_numSims + m_threadBlockSize - 1) / m_threadBlockSize;

    // Aim to launch around ten or more times as many blocks as there
    // are multiprocessors on the target device.
    unsigned int blocksPerSM = 10;
    unsigned int numSMs      = deviceProperties.multiProcessorCount;

    while (grid.x > 2 * blocksPerSM * numSMs) {
        grid.x >>= 1;
    }

    // Get initRNG function properties and check the maximum block size
    cudaResult = cudaFuncGetAttributes(&funcAttributes, initRNG);

    if (cudaResult != cudaSuccess) {
        string msg("Could not get function attributes: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    if (block.x > (unsigned int)funcAttributes.maxThreadsPerBlock) {
        throw std::runtime_error("Block X dimension is too large for initRNG kernel");
    }

    // Get computeValue function properties and check the maximum block size
    cudaResult = cudaFuncGetAttributes(&funcAttributes, computeValue<Real>);

    if (cudaResult != cudaSuccess) {
        string msg("Could not get function attributes: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    if (block.x > (unsigned int)funcAttributes.maxThreadsPerBlock) {
        throw std::runtime_error("Block X dimension is too large for computeValue kernel");
    }

    // Check the dimensions are valid
    if (block.x > (unsigned int)deviceProperties.maxThreadsDim[0]) {
        throw std::runtime_error("Block X dimension is too large for device");
    }

    if (grid.x > (unsigned int)deviceProperties.maxGridSize[0]) {
        throw std::runtime_error("Grid X dimension is too large for device");
    }

    // Allocate memory for RNG states
    curandState *d_rngStates = 0;
    // JP: `cudaResult`, `cudaMalloc`, `curandState`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    cudaResult               = cudaMalloc((void **)&d_rngStates, grid.x * block.x * sizeof(curandState));

    if (cudaResult != cudaSuccess) {
        string msg("Could not allocate memory on device for RNG states: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Allocate memory for result
    // Each thread block will produce one result
    unsigned int *d_results = 0;
    // JP: この anchor では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    cudaResult              = cudaMalloc((void **)&d_results, grid.x * sizeof(unsigned int));

    if (cudaResult != cudaSuccess) {
        string msg("Could not allocate memory on device for partial results: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Initialise RNG
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    initRNG<<<grid, block>>>(d_rngStates, m_seed);

    // Count the points inside unit quarter-circle
    computeValue<Real><<<grid, block, block.x * sizeof(unsigned int)>>>(d_results, d_rngStates, m_numSims);

    // Copy partial results back
    vector<unsigned int> results(grid.x);
    // JP: `cudaResult`, `cudaMemcpy`, `cudaMemcpyDeviceToHost`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    cudaResult = cudaMemcpy(&results[0], d_results, grid.x * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    if (cudaResult != cudaSuccess) {
        string msg("Could not copy partial results to host: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Complete sum-reduction on host
    Real value = static_cast<Real>(std::accumulate(results.begin(), results.end(), 0));

    // Determine the proportion of points inside the quarter-circle,
    // i.e. the area of the unit quarter-circle
    value /= m_numSims;

    // Value is currently an estimate of the area of a unit quarter-circle, so we
    // can scale to a full circle by multiplying by four. Now since the area of a
    // circle is pi * r^2, and r is one, the value will be an estimate for the
    // value of pi.
    value *= 4;

    // Cleanup
    if (d_rngStates) {
        // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        cudaFree(d_rngStates);
        d_rngStates = 0;
    }

    if (d_results) {
        cudaFree(d_results);
        d_results = 0;
    }

    return value;
}

// Explicit template instantiation
template class PiEstimator<float>;
template class PiEstimator<double>;
