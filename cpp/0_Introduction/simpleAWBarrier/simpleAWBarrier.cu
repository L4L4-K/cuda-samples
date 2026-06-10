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

// Includes, system
#include <stdio.h>

// Includes CUDA
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h> // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h> // helper functions for CUDA error check

namespace cg = cooperative_groups;

#if __CUDA_ARCH__ >= 700
template <bool writeSquareRoot>
__device__ void reduceBlockData(cuda::barrier<cuda::thread_scope_block> &barrier,
                                cg::thread_block_tile<32>               &tile32,
                                double                                  &threadSum,
                                double                                  *result)
{
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    extern __shared__ double tmp[];

#pragma unroll
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
        threadSum += tile32.shfl_down(threadSum, offset);
    }
    if (tile32.thread_rank() == 0) {
        tmp[tile32.meta_group_rank()] = threadSum;
    }

    auto token = barrier.arrive();

    barrier.wait(std::move(token));

    // The warp 0 will perform last round of reduction
    if (tile32.meta_group_rank() == 0) {
        double beta = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;

#pragma unroll
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            beta += tile32.shfl_down(beta, offset);
        }

        if (tile32.thread_rank() == 0) {
            if (writeSquareRoot)
                *result = sqrt(beta);
            else
                *result = beta;
        }
    }
}
#endif

__global__ void normVecByDotProductAWBarrier(float *vecA, float *vecB, double *partialResults, int size)
{
#if __CUDA_ARCH__ >= 700
#pragma diag_suppress static_var_with_dynamic_init
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block cta  = cg::this_thread_block();
    cg::grid_group   grid = cg::this_grid();
    ;
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    // JP: この anchor では shared memory の block-local scratchpad です。producer/consumer の順序と必要な barrier を確認します。
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }

    // JP: sync: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    cg::sync(cta);

    double threadSum = 0.0;
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        threadSum += (double)(vecA[i] * vecB[i]);
    }

    // Each thread block performs reduction of partial dotProducts and writes to
    // global mem.
    // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    reduceBlockData<false>(barrier, tile32, threadSum, &partialResults[blockIdx.x]);

    // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
    cg::sync(grid);

    // One block performs the final summation of partial dot products
    // of all the thread blocks and writes the sqrt of final dot product.
    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    if (blockIdx.x == 0) {
        threadSum = 0.0;
        for (int i = cta.thread_rank(); i < gridDim.x; i += cta.size()) {
            threadSum += partialResults[i];
        }
        reduceBlockData<true>(barrier, tile32, threadSum, &partialResults[0]);
    }

    // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
    cg::sync(grid);

    const double finalValue = partialResults[0];

    // Perform normalization of vecA & vecB.
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        vecA[i] = (float)vecA[i] / finalValue;
        vecB[i] = (float)vecB[i] / finalValue;
    }
#endif
}

int runNormVecByDotProductAWBarrier(int argc, char **argv, int deviceId);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("%s starting...\n", argv[0]);

    // This will pick the best possible CUDA capable device
    int dev = findCudaDevice(argc, (const char **)argv);

    int major = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));

    // Arrive-Wait Barrier require a GPU of Volta (SM7X) architecture or higher.
    if (major < 7) {
        printf("simpleAWBarrier requires SM 7.0 or higher.  Exiting...\n");
        exit(EXIT_WAIVED);
    }

    int supportsCooperativeLaunch = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&supportsCooperativeLaunch, cudaDevAttrCooperativeLaunch, dev));

    if (!supportsCooperativeLaunch) {
        printf("\nSelected GPU (%d) does not support Cooperative Kernel Launch, "
               "Waiving the run\n",
               dev);
        exit(EXIT_WAIVED);
    }

    int testResult = runNormVecByDotProductAWBarrier(argc, argv, dev);

    printf("%s completed, returned %s\n", argv[0], testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

int runNormVecByDotProductAWBarrier(int argc, char **argv, int deviceId)
{
    float  *vecA, *d_vecA;
    float  *vecB, *d_vecB;
    double *d_partialResults;
    int     size = 10000000;

    // JP: `cudaMallocHost`: page-locked host memory は DMA/async copy を安定させます。通常の free ではなく対応する CUDA API で解放します。
    checkCudaErrors(cudaMallocHost(&vecA, sizeof(float) * size));
    checkCudaErrors(cudaMallocHost(&vecB, sizeof(float) * size));

    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc(&d_vecA, sizeof(float) * size));
    checkCudaErrors(cudaMalloc(&d_vecB, sizeof(float) * size));

    float baseVal = 2.0;
    for (int i = 0; i < size; i++) {
        vecA[i] = vecB[i] = baseVal;
    }

    // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // JP: `cudaMemcpyAsync`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpyAsync(d_vecA, vecA, sizeof(float) * size, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_vecB, vecB, sizeof(float) * size, cudaMemcpyHostToDevice, stream));

    // Kernel configuration, where a one-dimensional
    // grid and one-dimensional blocks are configured.
    int minGridSize = 0, blockSize = 0;
    checkCudaErrors(
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)normVecByDotProductAWBarrier, 0, size));

    int smemSize = ((blockSize / 32) + 1) * sizeof(double);

    int numBlocksPerSm = 0;
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, normVecByDotProductAWBarrier, blockSize, smemSize));

    int multiProcessorCount = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, deviceId));

    minGridSize = multiProcessorCount * numBlocksPerSm;
    // JP: この anchor では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaMalloc(&d_partialResults, minGridSize * sizeof(double)));

    printf("Launching normVecByDotProductAWBarrier kernel with numBlocks = %d "
           "blockSize = %d\n",
           minGridSize,
           blockSize);

    dim3 dimGrid(minGridSize, 1, 1), dimBlock(blockSize, 1, 1);

    void *kernelArgs[] = {(void *)&d_vecA, (void *)&d_vecB, (void *)&d_partialResults, (void *)&size};

    checkCudaErrors(cudaLaunchCooperativeKernel(
        (void *)normVecByDotProductAWBarrier, dimGrid, dimBlock, kernelArgs, smemSize, stream));

    // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpyAsync(vecA, d_vecA, sizeof(float) * size, cudaMemcpyDeviceToHost, stream));
    // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
    checkCudaErrors(cudaStreamSynchronize(stream));

    float        expectedResult = (baseVal / sqrt(size * baseVal * baseVal));
    unsigned int matches        = 0;
    for (int i = 0; i < size; i++) {
        if ((vecA[i] - expectedResult) > 0.00001) {
            printf("mismatch at i = %d\n", i);
            break;
        }
        else {
            matches++;
        }
    }

    printf("Result = %s\n", matches == size ? "PASSED" : "FAILED");
    // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFree(d_vecA));
    checkCudaErrors(cudaFree(d_vecB));
    checkCudaErrors(cudaFree(d_partialResults));

    checkCudaErrors(cudaFreeHost(vecA));
    checkCudaErrors(cudaFreeHost(vecB));
    return matches == size;
}
