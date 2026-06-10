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

#include <cstdint>
#include <helper_cuda.h> // helper functions for CUDA error check
#include <iostream>

const int manualBlockSize = 32;

////////////////////////////////////////////////////////////////////////////////
// Test kernel
//
// This kernel squares each array element. Each thread addresses
// himself with threadIdx and blockIdx, so that it can handle any
// execution configuration, including anything the launch configurator
// API suggests.
////////////////////////////////////////////////////////////////////////////////
__global__ void square(uint32_t *array, int arrayCount)
{
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    extern __shared__ int dynamicSmem[];
    // JP: `threadIdx`, `blockIdx`, `blockDim`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int                   idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < arrayCount) {
        array[idx] *= array[idx];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Potential occupancy calculator
//
// The potential occupancy is calculated according to the kernel and
// execution configuration the user desires. Occupancy is defined in
// terms of active blocks per multiprocessor, and the user can convert
// it to other metrics.
//
// This wrapper routine computes the occupancy of kernel, and reports
// it in terms of active warps / maximum warps per SM.
////////////////////////////////////////////////////////////////////////////////
static double reportPotentialOccupancy(void *kernel, int blockSize, size_t dynamicSMem)
{
    int            device;
    cudaDeviceProp prop;

    int numBlocks;
    int activeWarps;
    int maxWarps;

    double occupancy;

    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));

    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, blockSize, dynamicSMem));

    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps    = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    occupancy = (double)activeWarps / maxWarps;

    return occupancy;
}

////////////////////////////////////////////////////////////////////////////////
// Occupancy-based launch configurator
//
// The launch configurator, cudaOccupancyMaxPotentialBlockSize and
// cudaOccupancyMaxPotentialBlockSizeVariableSMem, suggests a block
// size that achieves the best theoretical occupancy. It also returns
// the minimum number of blocks needed to achieve the occupancy on the
// whole device.
//
// This launch configurator is purely occupancy-based. It doesn't
// translate directly to performance, but the suggestion should
// nevertheless be a good starting point for further optimizations.
//
// This function configures the launch based on the "automatic"
// argument, records the runtime, and reports occupancy and runtime.
////////////////////////////////////////////////////////////////////////////////
static int launchConfig(uint32_t *array, int arrayCount, bool automatic)
{
    int    blockSize;
    int    minGridSize;
    int    gridSize;
    size_t dynamicSMemUsage = 0;

    // JP: `cudaEvent_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    cudaEvent_t start;
    cudaEvent_t end;

    float elapsedTime;

    double potentialOccupancy;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));

    if (automatic) {
        checkCudaErrors(
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)square, dynamicSMemUsage, arrayCount));

        std::cout << "Suggested block size: " << blockSize << std::endl
                  << "Minimum grid size for maximum occupancy: " << minGridSize << std::endl;
    }
    else {
        // This block size is too small. Given limited number of
        // active blocks per multiprocessor, the number of active
        // threads will be limited, and thus unable to achieve maximum
        // occupancy.
        //
        blockSize = manualBlockSize;
    }

    // Round up
    //
    gridSize = (arrayCount + blockSize - 1) / blockSize;

    // Launch and profile
    //
    checkCudaErrors(cudaEventRecord(start));
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    square<<<gridSize, blockSize, dynamicSMemUsage>>>(array, arrayCount);
    checkCudaErrors(cudaEventRecord(end));

    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaDeviceSynchronize());

    // Calculate occupancy
    //
    potentialOccupancy = reportPotentialOccupancy((void *)square, blockSize, dynamicSMemUsage);

    std::cout << "Potential occupancy: " << potentialOccupancy * 100 << "%" << std::endl;

    // Report elapsed time
    //
    // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, end));
    std::cout << "Elapsed time: " << elapsedTime << "ms" << std::endl;

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// The test
//
// The test generates an array and squares it with a CUDA kernel, then
// verifies the result.
////////////////////////////////////////////////////////////////////////////////
static int test(bool automaticLaunchConfig, const int count = 1000000)
{
    uint32_t *array;
    uint32_t *dArray;
    int       size = count * sizeof(uint32_t);

    array = new uint32_t[count];

    for (uint32_t i = 0; i < count; i += 1) {
        array[i] = i;
    }

    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc(&dArray, size));
    // JP: `cudaMemcpy`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpy(dArray, array, size, cudaMemcpyHostToDevice));

    for (uint32_t i = 0; i < count; i += 1) {
        array[i] = 0;
    }

    launchConfig(dArray, count, automaticLaunchConfig);

    checkCudaErrors(cudaMemcpy(array, dArray, size, cudaMemcpyDeviceToHost));
    // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFree(dArray));

    // Verify the return data
    // Both GPU and CPU use uint32_t * uint32_t, which has well-defined overflow behavior (modulo 2^32)
    //
    for (uint32_t i = 0; i < count; i += 1) {
        if (array[i] != i * i) {
            std::cout << "element " << i << " expected " << i * i << " actual " << array[i] << std::endl;
            return 1;
        }
    }
    delete[] array;

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Sample Main
//
// The sample runs the test with manually configured launch and
// automatically configured launch, and reports the occupancy and
// performance.
////////////////////////////////////////////////////////////////////////////////
int main()
{
    int status;

    std::cout << "starting Simple Occupancy" << std::endl << std::endl;

    std::cout << "[ Manual configuration with " << manualBlockSize << " threads per block ]" << std::endl;

    status = test(false);
    if (status) {
        std::cerr << "Test failed\n" << std::endl;
        return -1;
    }

    std::cout << std::endl;

    std::cout << "[ Automatic, occupancy-based configuration ]" << std::endl;
    status = test(true);
    if (status) {
        std::cerr << "Test failed\n" << std::endl;
        return -1;
    }

    std::cout << std::endl;
    std::cout << "Test PASSED\n" << std::endl;

    return 0;
}
