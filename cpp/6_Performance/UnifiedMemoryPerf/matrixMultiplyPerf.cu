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

#include <helper_cuda.h>
#include <helper_timer.h>

#include "commonDefs.hpp"
#include "commonKernels.hpp"

#define VERIFY_GPU_CORRECTNESS 0

size_t maxSampleSizeInMb = 64;
int    numKernelRuns     = 20;
int    verboseResults    = 0;

const char *memAllocTypeStr[MEMALLOC_TYPE_COUNT] = {"Managed_Memory_With_Hints",
                                                    "Managed_Memory_With_Hints_FullyAsync",
                                                    "Managed_Memory_NoHints",
                                                    "Zero_Copy",
                                                    "Memcpy_HostMalloc_DeviceCudaMalloc",
                                                    "MemcpyAsync_HostMalloc_DeviceCudaMalloc",
                                                    "Memcpy_HostCudaHostAlloc_DeviceCudaMalloc",
                                                    "MemcpyAsync_HostCudaHostAlloc_DeviceCudaMalloc"};

const char *memAllocTypeShortStr[MEMALLOC_TYPE_COUNT] = {
    "UMhint",  // Managed Memory With Hints
    "UMhntAs", // Managed Memory With_Hints Async
    "UMeasy",  // Managed_Memory with No Hints
    "0Copy",   // Zero Copy
    "MemCopy", // USE HOST PAGEABLE AND DEVICE_MEMORY
    "CpAsync", // USE HOST PAGEABLE AND DEVICE_MEMORY ASYNC
    // JP: pinned_memory: page-locked host memory は DMA/async copy を安定させます。通常の free ではなく対応する CUDA API で解放します。
    "CpHpglk", // USE HOST PAGELOCKED AND DEVICE MEMORY
    "CpPglAs"  // USE HOST PAGELOCKED AND DEVICE MEMORY ASYNC
};

static float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

void fillMatrixWithRandomValues(float *matrix, unsigned int matrixDim)
{
    unsigned int i, j;
    for (i = 0; i < matrixDim; ++i) {
        for (j = 0; j < matrixDim; ++j) {
            matrix[j + i * matrixDim] = RandFloat(0.0f, 10.0f);
        }
    }
}

#if VERIFY_GPU_CORRECTNESS
void verifyMatrixMultiplyCorrectness(float *C, float *A, float *B, unsigned int matrixDim)
{
    unsigned int i, j, k, numErrors = 0;
    for (i = 0; i < matrixDim; ++i) {
        for (j = 0; j < matrixDim; ++j) {
            float result = 0.0f;
            for (k = 0; k < matrixDim; ++k) {
                result += A[k + i * matrixDim] * B[j + k * matrixDim];
            }
            if (fabs(C[j + i * matrixDim] - result) > 0.001 * matrixDim) {
                printf("At [%u, %u]: Expected %f, Found %f\n", i, j, result, C[j + i * matrixDim]);
                ++numErrors;
            }
        }
    }
    if (numErrors != 0) {
        printf("%d value mismatches occured\n", numErrors);
        fflush(stdout);
        exit(EXIT_FAILURE); // exit since value mismatches occured
    }
}
#endif

void copyMatrix(float *dstMatrix, float *srcMatrix, unsigned int matrixDim)
{
    size_t size = matrixDim * matrixDim * sizeof(float);
    memcpy(dstMatrix, srcMatrix, size);
}

void verifyMatrixData(float *expectedData, float *observedData, unsigned int matrixDim)
{
    unsigned int i, j, numErrors = 0;
    for (i = 0; i < matrixDim; ++i) {
        for (j = 0; j < matrixDim; ++j) {
            if (expectedData[j + i * matrixDim] != observedData[j + i * matrixDim]) {
                ++numErrors;
                if (verboseResults) {
                    printf("At [%u, %u]: Expected %f, Found %f\n",
                           i,
                           j,
                           expectedData[j + i * matrixDim],
                           observedData[j + i * matrixDim]);
                }
            }
        }
    }
    if (numErrors != 0) {
        printf("%d value mismatches occured\n", numErrors);
        fflush(stdout);
        exit(EXIT_FAILURE); // exit since value mismatches occured
    }
}

#define BLOCK_SIZE 32
__global__ void matrixMultiplyKernel(float *C, float *A, float *B, unsigned int matrixDim)
{
    // Block index
    // JP: `blockIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    unsigned int wA = matrixDim;
    unsigned int wB = matrixDim;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = matrixDim * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c               = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void runMatrixMultiplyKernel(unsigned int matrixDim,
                             int          allocType,
                             unsigned int numLoops,
                             double      *gpuLaunchCallsTimes,
                             double      *gpuTransferToCallsTimes,
                             double      *gpuTransferFromCallsTimes,
                             double      *gpuLaunchAndTransferCallsTimes,
                             double      *gpuLaunchTransferSyncTimes,
                             double      *cpuAccessTimes,
                             double      *overallTimes,
                             int          device_id)
{
    float              *dptrA = NULL, *hptrA = NULL;
    float              *dptrB = NULL, *hptrB = NULL;
    float              *dptrC = NULL, *hptrC = NULL;
    float              *randValuesX = NULL, *randValuesY = NULL;
    float              *randValuesVerifyXmulY = NULL, *randValuesVerifyYmulX = NULL;
    bool                copyRequired = false, hintsRequired = false;
    bool                someTransferOpRequired;
    bool                isAsync = false;
    // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    cudaStream_t        streamToRunOn;
    unsigned int       *latch;
    size_t              size = matrixDim * matrixDim * sizeof(float);
    dim3                threads(32, 32);
    dim3                grid(matrixDim / threads.x, matrixDim / threads.y);
    StopWatchInterface *gpuLaunchCallsTimer = 0, *gpuTransferCallsTimer = 0;
    StopWatchInterface *gpuSyncTimer = 0, *cpuAccessTimer = 0;
    sdkCreateTimer(&gpuLaunchCallsTimer);
    sdkCreateTimer(&gpuTransferCallsTimer);
    sdkCreateTimer(&gpuSyncTimer);
    sdkCreateTimer(&cpuAccessTimer);
    unsigned int i;

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device_id));
    // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaStreamCreate(&streamToRunOn));

    randValuesX = (float *)malloc(size);
    if (!randValuesX) {
        exit(EXIT_FAILURE); // exit since memory allocation error
    }
    randValuesY = (float *)malloc(size);
    if (!randValuesY) {
        exit(EXIT_FAILURE); // exit since memory allocation error
    }
    randValuesVerifyXmulY = (float *)malloc(size);
    if (!randValuesVerifyXmulY) {
        exit(EXIT_FAILURE); // exit since memory allocation error
    }
    randValuesVerifyYmulX = (float *)malloc(size);
    if (!randValuesVerifyYmulX) {
        exit(EXIT_FAILURE); // exit since memory allocation error
    }
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc(&dptrA, size));
    checkCudaErrors(cudaMalloc(&dptrB, size));
    checkCudaErrors(cudaMalloc(&dptrC, size));

    fillMatrixWithRandomValues(randValuesX, matrixDim);
    fillMatrixWithRandomValues(randValuesY, matrixDim);

    // JP: `cudaMemcpyAsync`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpyAsync(dptrA, randValuesX, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(dptrB, randValuesY, size, cudaMemcpyHostToDevice));
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    matrixMultiplyKernel<<<grid, threads>>>(dptrC, dptrA, dptrB, matrixDim);
    checkCudaErrors(cudaMemcpyAsync(randValuesVerifyXmulY, dptrC, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaStreamSynchronize(NULL));
    matrixMultiplyKernel<<<grid, threads>>>(dptrC, dptrB, dptrA, matrixDim);
    // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpyAsync(randValuesVerifyYmulX, dptrC, size, cudaMemcpyDeviceToHost));
    // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
    checkCudaErrors(cudaStreamSynchronize(NULL));
#if VERIFY_GPU_CORRECTNESS
    verifyMatrixMultiplyCorrectness(randValuesVerifyXmulY, randValuesX, randValuesY, matrixDim);
    verifyMatrixMultiplyCorrectness(randValuesVerifyYmulX, randValuesY, randValuesX, matrixDim);
#endif
    // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFree(dptrA));
    checkCudaErrors(cudaFree(dptrB));
    checkCudaErrors(cudaFree(dptrC));

    // JP: この anchor では pinned host memory の登録/確保/解放です。async transfer や overlap の条件と lifetime を確認します。
    checkCudaErrors(cudaMallocHost(&latch, sizeof(unsigned int)));

    switch (allocType) {
    case USE_HOST_PAGEABLE_AND_DEVICE_MEMORY:
    case USE_HOST_PAGEABLE_AND_DEVICE_MEMORY_ASYNC:
        hptrA = (float *)malloc(size);
        if (!hptrA) {
            exit(EXIT_FAILURE); // exit since memory allocation error
        }
        hptrB = (float *)malloc(size);
        if (!hptrB) {
            exit(EXIT_FAILURE); // exit since memory allocation error
        }
        hptrC = (float *)malloc(size);
        if (!hptrC) {
            exit(EXIT_FAILURE); // exit since memory allocation error
        }
        // JP: この連続する anchor 群では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
        checkCudaErrors(cudaMalloc(&dptrA, size));
        checkCudaErrors(cudaMalloc(&dptrB, size));
        checkCudaErrors(cudaMalloc(&dptrC, size));
        copyRequired = true;
        break;

    case USE_HOST_PAGELOCKED_AND_DEVICE_MEMORY:
    case USE_HOST_PAGELOCKED_AND_DEVICE_MEMORY_ASYNC:
        // JP: この連続する anchor 群では pinned host memory の登録/確保/解放です。async transfer や overlap の条件と lifetime を確認します。
        checkCudaErrors(cudaMallocHost(&hptrA, size));
        checkCudaErrors(cudaMallocHost(&hptrB, size));
        checkCudaErrors(cudaMallocHost(&hptrC, size));
        // JP: この連続する anchor 群では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
        checkCudaErrors(cudaMalloc(&dptrA, size));
        checkCudaErrors(cudaMalloc(&dptrB, size));
        checkCudaErrors(cudaMalloc(&dptrC, size));
        copyRequired = true;
        break;

    case USE_ZERO_COPY:
        // JP: この連続する anchor 群では pinned host memory の登録/確保/解放です。async transfer や overlap の条件と lifetime を確認します。
        checkCudaErrors(cudaMallocHost(&hptrA, size));
        checkCudaErrors(cudaMallocHost(&hptrB, size));
        checkCudaErrors(cudaMallocHost(&hptrC, size));
        checkCudaErrors(cudaHostGetDevicePointer(&dptrA, hptrA, 0));
        checkCudaErrors(cudaHostGetDevicePointer(&dptrB, hptrB, 0));
        checkCudaErrors(cudaHostGetDevicePointer(&dptrC, hptrC, 0));
        break;

    case USE_MANAGED_MEMORY:
        // JP: `cudaMallocManaged`: Unified Memory は CPU/GPU で同じ pointer を使います。prefetch や同期で移動タイミングを意識します。
        checkCudaErrors(cudaMallocManaged(&dptrA, size));
        checkCudaErrors(cudaMallocManaged(&dptrB, size));
        checkCudaErrors(cudaMallocManaged(&dptrC, size));
        hptrA = dptrA;
        hptrB = dptrB;
        hptrC = dptrC;
        break;

    case USE_MANAGED_MEMORY_WITH_HINTS:
    case USE_MANAGED_MEMORY_WITH_HINTS_ASYNC:
        if (deviceProp.concurrentManagedAccess) {
            // JP: この連続する anchor 群では Unified Memory allocation/prefetch/advice です。migration、host/device visibility、同期位置 を確認します。
            checkCudaErrors(cudaMallocManaged(&dptrA, size));
            checkCudaErrors(cudaMallocManaged(&dptrB, size));
            checkCudaErrors(cudaMallocManaged(&dptrC, size));
            cudaMemLocation hostLoc;
            hostLoc.type = cudaMemLocationTypeHost;
            checkCudaErrors(cudaMemPrefetchAsync(dptrA, size, hostLoc, 0));
            checkCudaErrors(cudaMemPrefetchAsync(dptrB, size, hostLoc, 0));
            checkCudaErrors(cudaMemPrefetchAsync(dptrC, size, hostLoc, 0));
        }
        else {
            checkCudaErrors(cudaMallocManaged(&dptrA, size, cudaMemAttachHost));
            checkCudaErrors(cudaMallocManaged(&dptrB, size, cudaMemAttachHost));
            checkCudaErrors(cudaMallocManaged(&dptrC, size, cudaMemAttachHost));
        }
        hptrA         = dptrA;
        hptrB         = dptrB;
        hptrC         = dptrC;
        hintsRequired = true;
        break;

    default:
        exit(EXIT_FAILURE); // exit with error
    }

    if (allocType == USE_HOST_PAGEABLE_AND_DEVICE_MEMORY_ASYNC
        || allocType == USE_HOST_PAGELOCKED_AND_DEVICE_MEMORY_ASYNC
        || allocType == USE_MANAGED_MEMORY_WITH_HINTS_ASYNC) {
        isAsync = true;
    }

    someTransferOpRequired = copyRequired || hintsRequired;

    // fill buffers with 0 to avoid any first access page-fault overheads.
    memset(hptrA, 0, size);
    memset(hptrB, 0, size);
    memset(hptrC, 0, size);

    for (i = 0; i < numLoops; i++) {
        cpuAccessTimes[i]            = 0.0;
        gpuLaunchCallsTimes[i]       = 0.0;
        gpuTransferToCallsTimes[i]   = 0.0;
        gpuTransferFromCallsTimes[i] = 0.0;

        sdkStartTimer(&cpuAccessTimer);
        {
            copyMatrix(hptrA, (i & 0x1 == 0) ? randValuesX : randValuesY, matrixDim);
            copyMatrix(hptrB, (i & 0x1 == 0) ? randValuesY : randValuesX, matrixDim);
        }
        sdkStopTimer(&cpuAccessTimer);
        cpuAccessTimes[i] += sdkGetAverageTimerValue(&cpuAccessTimer);
        sdkResetTimer(&cpuAccessTimer);

        if (isAsync && hintsRequired) {
            *latch = 0;
            // Prevent any work on stream from starting until all work is pushed
            // JP: この anchor では kernel launch の grid/block/shared-memory/stream 指定です。後続の sync/error check と完了確認を対応させます。
            spinWhileLessThanOne<<<1, 1, 0, streamToRunOn>>>(latch);
        }

        if (someTransferOpRequired) {
            sdkStartTimer(&gpuTransferCallsTimer);
            if (copyRequired) {
                if (isAsync) {
                    // JP: この連続する anchor 群では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
                    checkCudaErrors(cudaMemcpyAsync(dptrA, hptrA, size, cudaMemcpyHostToDevice, streamToRunOn));
                    checkCudaErrors(cudaMemcpyAsync(dptrB, hptrB, size, cudaMemcpyHostToDevice, streamToRunOn));
                }
                else {
                    checkCudaErrors(cudaMemcpy(dptrA, hptrA, size, cudaMemcpyHostToDevice));
                    checkCudaErrors(cudaMemcpy(dptrB, hptrB, size, cudaMemcpyHostToDevice));
                }
            }
            if (hintsRequired) {
                if (deviceProp.concurrentManagedAccess) {
                    cudaMemLocation deviceLoc;
                    deviceLoc.type = cudaMemLocationTypeDevice;
                    deviceLoc.id   = device_id;
                    // JP: この連続する anchor 群では Unified Memory allocation/prefetch/advice です。migration、host/device visibility、同期位置 を確認します。
                    checkCudaErrors(cudaMemPrefetchAsync(dptrA, size, deviceLoc, 0, streamToRunOn));
                    checkCudaErrors(cudaMemPrefetchAsync(dptrB, size, deviceLoc, 0, streamToRunOn));
                    checkCudaErrors(cudaMemPrefetchAsync(dptrC, size, deviceLoc, 0, streamToRunOn));
                }
                else {
                    // JP: この連続する anchor 群では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
                    checkCudaErrors(cudaStreamAttachMemAsync(streamToRunOn, dptrA, 0, cudaMemAttachGlobal));
                    checkCudaErrors(cudaStreamAttachMemAsync(streamToRunOn, dptrB, 0, cudaMemAttachGlobal));
                    checkCudaErrors(cudaStreamAttachMemAsync(streamToRunOn, dptrC, 0, cudaMemAttachGlobal));
                }
                if (!isAsync) {
                    checkCudaErrors(cudaStreamSynchronize(streamToRunOn));
                }
            }

            sdkStopTimer(&gpuTransferCallsTimer);
            gpuTransferToCallsTimes[i] += sdkGetAverageTimerValue(&gpuTransferCallsTimer);
            sdkResetTimer(&gpuTransferCallsTimer);
        }

        sdkStartTimer(&gpuLaunchCallsTimer);
        {
            // JP: この anchor では kernel launch の grid/block/shared-memory/stream 指定です。後続の sync/error check と完了確認を対応させます。
            matrixMultiplyKernel<<<grid, threads, 0, streamToRunOn>>>(dptrC, dptrA, dptrB, matrixDim);
            if (!isAsync) {
                // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
                checkCudaErrors(cudaStreamSynchronize(streamToRunOn));
            }
        }
        sdkStopTimer(&gpuLaunchCallsTimer);

        gpuLaunchCallsTimes[i] += sdkGetAverageTimerValue(&gpuLaunchCallsTimer);
        sdkResetTimer(&gpuLaunchCallsTimer);

        if (someTransferOpRequired) {
            sdkStartTimer(&gpuTransferCallsTimer);
            if (hintsRequired) {
                if (deviceProp.concurrentManagedAccess) {
                    cudaMemLocation hostLoc;
                    hostLoc.type = cudaMemLocationTypeHost;
                    // JP: この連続する anchor 群では Unified Memory allocation/prefetch/advice です。migration、host/device visibility、同期位置 を確認します。
                    checkCudaErrors(cudaMemPrefetchAsync(dptrA, size, hostLoc, 0));
                    checkCudaErrors(cudaMemPrefetchAsync(dptrB, size, hostLoc, 0));
                    checkCudaErrors(cudaMemPrefetchAsync(dptrC, size, hostLoc, 0));
                }
                else {
                    // JP: この連続する anchor 群では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
                    checkCudaErrors(cudaStreamAttachMemAsync(streamToRunOn, dptrA, 0, cudaMemAttachHost));
                    checkCudaErrors(cudaStreamAttachMemAsync(streamToRunOn, dptrB, 0, cudaMemAttachHost));
                    checkCudaErrors(cudaStreamAttachMemAsync(streamToRunOn, dptrC, 0, cudaMemAttachHost));
                }
                if (!isAsync) {
                    checkCudaErrors(cudaStreamSynchronize(streamToRunOn));
                }
            }
            if (copyRequired) {
                if (isAsync) {
                    // JP: この連続する anchor 群では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
                    checkCudaErrors(cudaMemcpyAsync(hptrC, dptrC, size, cudaMemcpyDeviceToHost, streamToRunOn));
                }
                else {
                    checkCudaErrors(cudaMemcpy(hptrC, dptrC, size, cudaMemcpyDeviceToHost));
                }
            }
            sdkStopTimer(&gpuTransferCallsTimer);
            gpuTransferFromCallsTimes[i] += sdkGetAverageTimerValue(&gpuTransferCallsTimer);
            sdkResetTimer(&gpuTransferCallsTimer);
        }
        gpuLaunchAndTransferCallsTimes[i] =
            gpuLaunchCallsTimes[i] + gpuTransferToCallsTimes[i] + gpuTransferFromCallsTimes[i];
        gpuLaunchTransferSyncTimes[i] = gpuLaunchAndTransferCallsTimes[i];
        if (isAsync) {
            sdkStartTimer(&gpuSyncTimer);
            {
                if (hintsRequired) {
                    *latch = 1;
                }
                // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
                checkCudaErrors(cudaStreamSynchronize(streamToRunOn));
            }
            sdkStopTimer(&gpuSyncTimer);
            gpuLaunchTransferSyncTimes[i] += sdkGetAverageTimerValue(&gpuSyncTimer);
            sdkResetTimer(&gpuSyncTimer);
        }

        sdkStartTimer(&cpuAccessTimer);
        {
            verifyMatrixData((i & 0x1 == 0) ? randValuesVerifyXmulY : randValuesVerifyYmulX, hptrC, matrixDim);
        }
        sdkStopTimer(&cpuAccessTimer);
        cpuAccessTimes[i] += sdkGetAverageTimerValue(&cpuAccessTimer);
        sdkResetTimer(&cpuAccessTimer);
        overallTimes[i] = cpuAccessTimes[i] + gpuLaunchTransferSyncTimes[i];
    }

    switch (allocType) {
    case USE_HOST_PAGEABLE_AND_DEVICE_MEMORY:
    case USE_HOST_PAGEABLE_AND_DEVICE_MEMORY_ASYNC:
        free(hptrA);
        free(hptrB);
        free(hptrC);
        // JP: この連続する anchor 群では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
        checkCudaErrors(cudaFree(dptrA));
        checkCudaErrors(cudaFree(dptrB));
        checkCudaErrors(cudaFree(dptrC));
        break;

    case USE_HOST_PAGELOCKED_AND_DEVICE_MEMORY:
    case USE_HOST_PAGELOCKED_AND_DEVICE_MEMORY_ASYNC:
        checkCudaErrors(cudaFreeHost(hptrA));
        checkCudaErrors(cudaFreeHost(hptrB));
        checkCudaErrors(cudaFreeHost(hptrC));
        checkCudaErrors(cudaFree(dptrA));
        checkCudaErrors(cudaFree(dptrB));
        checkCudaErrors(cudaFree(dptrC));
        break;

    case USE_ZERO_COPY:
        checkCudaErrors(cudaFreeHost(hptrA));
        checkCudaErrors(cudaFreeHost(hptrB));
        checkCudaErrors(cudaFreeHost(hptrC));
        break;

    case USE_MANAGED_MEMORY:
    case USE_MANAGED_MEMORY_WITH_HINTS:
    case USE_MANAGED_MEMORY_WITH_HINTS_ASYNC:
        checkCudaErrors(cudaFree(dptrA));
        checkCudaErrors(cudaFree(dptrB));
        checkCudaErrors(cudaFree(dptrC));
        break;

    default:
        exit(EXIT_FAILURE); // exit due to error
    }

    checkCudaErrors(cudaStreamDestroy(streamToRunOn));
    checkCudaErrors(cudaFreeHost(latch));
    free(randValuesX);
    free(randValuesY);
    free(randValuesVerifyXmulY);
    free(randValuesVerifyYmulX);
    sdkDeleteTimer(&gpuLaunchCallsTimer);
    sdkDeleteTimer(&gpuTransferCallsTimer);
    sdkDeleteTimer(&gpuSyncTimer);
    sdkDeleteTimer(&cpuAccessTimer);
}

void matrixMultiplyPerfRunner(bool reportAsBandwidth,
                              bool print_launch_transfer_results,
                              bool print_std_deviation,
                              int  device_id)
{
    int          i;
    unsigned int minMatrixDim  = 32;
    unsigned int multiplierDim = 2;
    unsigned int matrixDim;
    unsigned int minSize    = minMatrixDim * minMatrixDim * sizeof(float);
    unsigned int maxSize    = (maxSampleSizeInMb * ONE_MB) / 4; // 3 buffers are used, but dividing by 4 (power of 2)
    unsigned int multiplier = multiplierDim * multiplierDim;
    unsigned int numSizesToTest;

    struct testResults *results;
    struct resultsData *gpuLaunchCallsTimes;
    struct resultsData *gpuTransferToCallsTimes;
    struct resultsData *gpuTransferFromCallsTimes;
    struct resultsData *gpuLaunchAndTransferCallsTimes;
    struct resultsData *gpuLaunchTransferSyncTimes;
    struct resultsData *cpuAccessTimes;
    struct resultsData *overallTimes;
    unsigned long      *sizesToTest;
    unsigned int        j;

    numSizesToTest = findNumSizesToTest(minSize, maxSize, multiplier);

    createAndInitTestResults(&results, "matrixMultiplyPerf", numKernelRuns, numSizesToTest);

    sizesToTest = getPtrSizesToTest(results);

    createResultDataAndAddToTestResults(
        &gpuLaunchCallsTimes, results, "GPU Kernel Launch Call Time", false, reportAsBandwidth);
    createResultDataAndAddToTestResults(
        &gpuTransferToCallsTimes, results, "CPU to GPU Transfer Calls Time", false, reportAsBandwidth);
    createResultDataAndAddToTestResults(
        &gpuTransferFromCallsTimes, results, "GPU to CPU Transfer Calls Time", false, reportAsBandwidth);
    createResultDataAndAddToTestResults(
        &gpuLaunchAndTransferCallsTimes, results, "GPU Launch and Transfer Calls Time", false, reportAsBandwidth);
    createResultDataAndAddToTestResults(
        &gpuLaunchTransferSyncTimes, results, "GPU Launch Transfer and Sync Time", false, reportAsBandwidth);
    createResultDataAndAddToTestResults(&cpuAccessTimes, results, "CPU Access Time", false, reportAsBandwidth);
    createResultDataAndAddToTestResults(&overallTimes, results, "Overall Time", false, reportAsBandwidth);

    printf("Running ");
    for (matrixDim = minMatrixDim, j = 0; matrixDim * matrixDim <= maxSize / sizeof(float);
         matrixDim *= multiplierDim, ++j) {
        sizesToTest[j] = matrixDim * matrixDim * sizeof(float);
        for (i = MEMALLOC_TYPE_START; i <= MEMALLOC_TYPE_END; i++) {
            printf(".");
            fflush(stdout);
            runMatrixMultiplyKernel(matrixDim,
                                    i,
                                    numKernelRuns,
                                    getPtrRunTimesInMs(gpuLaunchCallsTimes, i, j),
                                    getPtrRunTimesInMs(gpuTransferToCallsTimes, i, j),
                                    getPtrRunTimesInMs(gpuTransferFromCallsTimes, i, j),
                                    getPtrRunTimesInMs(gpuLaunchAndTransferCallsTimes, i, j),
                                    getPtrRunTimesInMs(gpuLaunchTransferSyncTimes, i, j),
                                    getPtrRunTimesInMs(cpuAccessTimes, i, j),
                                    getPtrRunTimesInMs(overallTimes, i, j),
                                    device_id);
        }
    }
    printf("\n");
    printResults(results, print_launch_transfer_results, print_std_deviation);
    freeTestResultsAndAllResultsData(results);
}

static void usage()
{
    printf("./cudaMemoryTypesPerf [-device=<device_id>] [-reportAsBandwidth] "
           "[-print-launch-transfer-results] [-print-std-deviation] [-verbose]\n");
    printf("Options:\n");
    printf("-reportAsBandwidth:             By default time taken is printed, this "
           "option allows to instead print bandwidth.\n");
    printf("-print-launch-transfer-results: By default overall results are printed, "
           "this option allows to print data transfers and kernel time as well.\n");
    printf("-print-std-deviation:           Prints std deviation of the results.\n");
    printf("-kernel-iterations=<num>:       Number of times the kernel tests should "
           "be run[default is 100 iterations].\n");
    printf("-device=<device_id>:            Allows to pass GPU Device ID on which "
           "the tests will be run.\n");
    printf("-verbose:                       Prints highly verbose output.\n");
}

int main(int argc, char **argv)
{
    bool reportAsBandwidth             = false;
    bool print_launch_transfer_results = false;
    bool print_std_deviation           = false;

    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "h")) {
        usage();
        printf("&&&& %s WAIVED\n", argv[0]);
        exit(EXIT_WAIVED);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "reportAsBandwidth")) {
        reportAsBandwidth = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "print-launch-transfer-results")) {
        print_launch_transfer_results = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "print-std-deviation")) {
        print_std_deviation = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "kernel-iterations")) {
        numKernelRuns = getCmdLineArgumentInt(argc, (const char **)argv, "kernel-iterations");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "verbose")) {
        verboseResults = 1;
    }

    // set device
    cudaDeviceProp device_prop;
    int            device_id = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, device_id));

    if (!device_prop.managedMemory) {
        // This samples requires being run on a device that supports Unified Memory
        fprintf(stderr, "Unified Memory not supported on this device\n");

        exit(EXIT_WAIVED);
    }

    matrixMultiplyPerfRunner(reportAsBandwidth, print_launch_transfer_results, print_std_deviation, device_id);

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. "
           "Results may vary when GPU Boost is enabled.\n");
    exit(EXIT_SUCCESS);
}
