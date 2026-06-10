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

/* A simple program demonstrating trivial use of global memory atomic
 * device functions (atomic*() functions).
 */

// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h> // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h> // helper functions for CUDA error check

// Includes, kernels
#include "simpleAtomicIntrinsics_kernel.cuh"

const char *sampleName = "simpleAtomicIntrinsics";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

extern "C" bool computeGold(int *gpuData, const int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("%s starting...\n", sampleName);

    runTest(argc, argv);

    printf("%s completed, returned %s\n", sampleName, testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv)
{
    // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    cudaStream_t stream;
    // This will pick the best possible CUDA capable device
    findCudaDevice(argc, (const char **)argv);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    unsigned int numThreads = 256;
    unsigned int numBlocks  = 64;
    unsigned int numData    = 11;
    unsigned int memSize    = sizeof(int) * numData;

    // allocate mem for the result on host side
    int *hOData;
    // JP: `cudaMallocHost`: page-locked host memory は DMA/async copy を安定させます。通常の free ではなく対応する CUDA API で解放します。
    checkCudaErrors(cudaMallocHost(&hOData, memSize));

    // initialize the memory
    for (unsigned int i = 0; i < numData; i++)
        hOData[i] = 0;

    // To make the AND and XOR tests generate something other than 0...
    hOData[8] = hOData[10] = 0xff;

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // allocate device memory for result
    int *dOData;
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc((void **)&dOData, memSize));
    // copy host memory to device to initialize to zero
    // JP: `cudaMemcpyAsync`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpyAsync(dOData, hOData, memSize, cudaMemcpyHostToDevice, stream));

    // execute the kernel
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    testKernel<<<numBlocks, numThreads, 0, stream>>>(dOData);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(hOData, dOData, memSize, cudaMemcpyDeviceToHost, stream));
    // JP: `cudaStreamSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaStreamSynchronize(stream));

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    // Compute reference solution
    testResult = computeGold(hOData, numThreads * numBlocks);

    // Cleanup memory
    // JP: `cudaFreeHost`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFreeHost(hOData));
    checkCudaErrors(cudaFree(dOData));
}
