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

/*
 * This sample demonstrates stream ordered memory allocation on a GPU using
 * cudaMallocAsync and cudaMemPool family of APIs.
 *
 * basicStreamOrderedAllocation(): demonstrates stream ordered allocation using
 * cudaMallocAsync/cudaFreeAsync APIs with default settings.
 *
 * streamOrderedAllocationPostSync(): demonstrates if there's a synchronization
 * in between allocations, then setting the release threshold on the pool will
 * make sure the synchronize will not free memory back to the OS.
 */

// System includes
#include <assert.h>
#include <climits>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#define MAX_ITER 20

/* Add two vectors on the GPU */
__global__ void vectorAddGPU(const float *a, const float *b, float *c, int N)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int basicStreamOrderedAllocation(const int dev, const int nelem, const float *a, const float *b, float *c)
{
    float *d_a, *d_b, *d_c; // Device buffers
    float  errorNorm, refNorm, ref, diff;
    size_t bytes = nelem * sizeof(float);

    // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    cudaStream_t stream;
    printf("Starting basicStreamOrderedAllocation()\n");
    checkCudaErrors(cudaSetDevice(dev));
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    checkCudaErrors(cudaMallocAsync(&d_a, bytes, stream));
    checkCudaErrors(cudaMallocAsync(&d_b, bytes, stream));
    checkCudaErrors(cudaMallocAsync(&d_c, bytes, stream));
    // JP: `cudaMemcpyAsync`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpyAsync(d_a, a, bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_b, b, bytes, cudaMemcpyHostToDevice, stream));

    dim3 block(256);
    dim3 grid((unsigned int)ceil(nelem / (float)block.x));
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    vectorAddGPU<<<grid, block, 0, stream>>>(d_a, d_b, d_c, nelem);

    // JP: `cudaFreeAsync`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFreeAsync(d_a, stream));
    checkCudaErrors(cudaFreeAsync(d_b, stream));
    checkCudaErrors(cudaMemcpyAsync(c, d_c, bytes, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaFreeAsync(d_c, stream));
    // JP: `cudaStreamSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaStreamSynchronize(stream));

    /* Compare the results */
    printf("> Checking the results from vectorAddGPU() ...\n");
    errorNorm = 0.f;
    refNorm   = 0.f;

    for (int n = 0; n < nelem; n++) {
        ref  = a[n] + b[n];
        diff = c[n] - ref;
        errorNorm += diff * diff;
        refNorm += ref * ref;
    }

    errorNorm = (float)sqrt((double)errorNorm);
    refNorm   = (float)sqrt((double)refNorm);
    if (errorNorm / refNorm < 1.e-6f)
        printf("basicStreamOrderedAllocation PASSED\n");

    // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaStreamDestroy(stream));

    return errorNorm / refNorm < 1.e-6f ? EXIT_SUCCESS : EXIT_FAILURE;
}

// streamOrderedAllocationPostSync(): demonstrates If the application wants the
// memory to persist in the pool beyond synchronization, then it sets the
// release threshold on the pool. This way, when the application reaches the
// "steady state", it is no longer allocating/freeing memory from the OS.
int streamOrderedAllocationPostSync(const int dev, const int nelem, const float *a, const float *b, float *c)
{
    float *d_a, *d_b, *d_c; // Device buffers
    float  errorNorm, refNorm, ref, diff;
    size_t bytes = nelem * sizeof(float);

    // JP: この連続する anchor 群では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    cudaStream_t  stream;
    cudaMemPool_t memPool;
    cudaEvent_t   start, end;
    printf("Starting streamOrderedAllocationPostSync()\n");
    checkCudaErrors(cudaSetDevice(dev));
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));

    checkCudaErrors(cudaDeviceGetDefaultMemPool(&memPool, dev));
    uint64_t thresholdVal = ULONG_MAX;
    // set high release threshold on the default pool so that cudaFreeAsync will
    // not actually release memory to the system. By default, the release
    // threshold for a memory pool is set to zero. This implies that the CUDA
    // driver is allowed to release a memory chunk back to the system as long as
    // it does not contain any active suballocations.
    checkCudaErrors(cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, (void *)&thresholdVal));

    // Record the start event
    // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaEventRecord(start, stream));
    for (int i = 0; i < MAX_ITER; i++) {
        checkCudaErrors(cudaMallocAsync(&d_a, bytes, stream));
        checkCudaErrors(cudaMallocAsync(&d_b, bytes, stream));
        checkCudaErrors(cudaMallocAsync(&d_c, bytes, stream));
        // JP: この連続する anchor 群では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
        checkCudaErrors(cudaMemcpyAsync(d_a, a, bytes, cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(d_b, b, bytes, cudaMemcpyHostToDevice, stream));

        dim3 block(256);
        dim3 grid((unsigned int)ceil(nelem / (float)block.x));
        // JP: この anchor では kernel launch の grid/block/shared-memory/stream 指定です。後続の sync/error check と完了確認を対応させます。
        vectorAddGPU<<<grid, block, 0, stream>>>(d_a, d_b, d_c, nelem);

        // JP: この連続する anchor 群では CUDA resource lifetime end です。未完了 work が残っていないか確認し、確保/作成/登録と対応する API で閉じます。
        checkCudaErrors(cudaFreeAsync(d_a, stream));
        checkCudaErrors(cudaFreeAsync(d_b, stream));
        // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
        checkCudaErrors(cudaMemcpyAsync(c, d_c, bytes, cudaMemcpyDeviceToHost, stream));
        // JP: この anchor では CUDA resource lifetime end です。未完了 work が残っていないか確認し、確保/作成/登録と対応する API で閉じます。
        checkCudaErrors(cudaFreeAsync(d_c, stream));
        // JP: この連続する anchor 群では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
    checkCudaErrors(cudaEventRecord(end, stream));
    // Wait for the end event to complete
    checkCudaErrors(cudaEventSynchronize(end));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, end));
    printf("Total elapsed time = %f ms over %d iterations\n", msecTotal, MAX_ITER);

    /* Compare the results */
    printf("> Checking the results from vectorAddGPU() ...\n");
    errorNorm = 0.f;
    refNorm   = 0.f;

    for (int n = 0; n < nelem; n++) {
        ref  = a[n] + b[n];
        diff = c[n] - ref;
        errorNorm += diff * diff;
        refNorm += ref * ref;
    }

    errorNorm = (float)sqrt((double)errorNorm);
    refNorm   = (float)sqrt((double)refNorm);
    if (errorNorm / refNorm < 1.e-6f)
        printf("streamOrderedAllocationPostSync PASSED\n");

    // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaStreamDestroy(stream));

    return errorNorm / refNorm < 1.e-6f ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char **argv)
{
    int    nelem;
    int    dev = 0; // use default device 0
    size_t bytes;
    float *a, *b, *c; // Host

    if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
        printf("Usage:  streamOrderedAllocation [OPTION]\n\n");
        printf("Options:\n");
        printf("  --device=[device #]  Specify the device to be used\n");
        return EXIT_SUCCESS;
    }

    dev = findCudaDevice(argc, (const char **)argv);

    int isMemPoolSupported = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&isMemPoolSupported, cudaDevAttrMemoryPoolsSupported, dev));
    if (!isMemPoolSupported) {
        printf("Waiving execution as device does not support Memory Pools\n");
        exit(EXIT_WAIVED);
    }

    // Allocate CPU memory.
    nelem = 1048576;
    bytes = nelem * sizeof(float);

    a = (float *)malloc(bytes);
    b = (float *)malloc(bytes);
    c = (float *)malloc(bytes);
    /* Initialize the vectors. */
    for (int n = 0; n < nelem; n++) {
        a[n] = rand() / (float)RAND_MAX;
        b[n] = rand() / (float)RAND_MAX;
    }

    int ret1 = basicStreamOrderedAllocation(dev, nelem, a, b, c);
    int ret2 = streamOrderedAllocationPostSync(dev, nelem, a, b, c);

    /* Memory clean up */
    free(a);
    free(b);
    free(c);

    return ((ret1 == EXIT_SUCCESS && ret2 == EXIT_SUCCESS) ? EXIT_SUCCESS : EXIT_FAILURE);
}
