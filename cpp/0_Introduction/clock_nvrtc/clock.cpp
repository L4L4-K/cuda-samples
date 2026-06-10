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
// JP: この file では stream/event による非同期実行と同期、Runtime/Driver/NVRTC の境界、performance measurement と memory access pattern を確認します。英語の識別子/API/出力文字列は保持します。

/*
 * This example shows how to use the clock function to measure the performance
 * of block of threads of a kernel accurately. Blocks are executed in parallel
 * and out of order. Since there's no synchronization mechanism between blocks,
 * we measure the clock once for each block. The clock samples are written to
 * device memory.
 */

// System includes
#include <assert.h>
#include <cuda_runtime.h>
#include <nvrtc_helper.h>
#include <stdint.h>
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>

#define NUM_BLOCKS 64

#define NUM_THREADS 256

// It's interesting to change the number of blocks and the number of threads to
// understand how to keep the hardware busy.
//

// Here are some numbers I get on my G80:
//    blocks - clocks
//    1 - 3096
//    8 - 3232
//    16 - 3364
//    32 - 4615
//    64 - 9981

//
// With less than 16 blocks some of the multiprocessors of the device are idle.
// With
// more than 16 you are using all the multiprocessors, but there's only one
// block per
// multiprocessor and that doesn't allow you to hide the latency of the memory.
// With
// more than 32 the speed scales linearly.

// Start the main CUDA Sample here

int main(int argc, char **argv)
{
    printf("CUDA Clock sample\n");

    typedef long clock_t;

    clock_t timer[NUM_BLOCKS * 2];

    float input[NUM_THREADS * 2];

    for (int i = 0; i < NUM_THREADS * 2; i++) {
        input[i] = (float)i;
    }

    char  *cubin, *kernel_file;
    size_t cubinSize;

    kernel_file = sdkFindFilePath("clock_kernel.cu", argv[0]);
    compileFileToCUBIN(kernel_file, argc, argv, &cubin, &cubinSize, 0);

    // JP: driver_api: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
    CUmodule   module = loadCUBIN(cubin, argc, argv);
    CUfunction kernel_addr;

    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "timedReduction"));

    dim3 cudaBlockSize(NUM_THREADS, 1, 1);
    dim3 cudaGridSize(NUM_BLOCKS, 1, 1);

    CUdeviceptr dinput, doutput, dtimer;
    // JP: `cuMemAlloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cuMemAlloc(&dinput, sizeof(float) * NUM_THREADS * 2));
    checkCudaErrors(cuMemAlloc(&doutput, sizeof(float) * NUM_BLOCKS));
    checkCudaErrors(cuMemAlloc(&dtimer, sizeof(clock_t) * NUM_BLOCKS * 2));
    // JP: `cuMemcpyHtoD`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cuMemcpyHtoD(dinput, input, sizeof(float) * NUM_THREADS * 2));

    void *arr[] = {(void *)&dinput, (void *)&doutput, (void *)&dtimer};

    // JP: `cuLaunchKernel`: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    checkCudaErrors(cuLaunchKernel(kernel_addr,
                                   cudaGridSize.x,
                                   cudaGridSize.y,
                                   cudaGridSize.z, /* grid dim */
                                   cudaBlockSize.x,
                                   cudaBlockSize.y,
                                   cudaBlockSize.z, /* block dim */
                                   sizeof(float) * 2 * NUM_THREADS,
                                   0,       /* shared mem, stream */
                                   &arr[0], /* arguments */
                                   0));

    checkCudaErrors(cuCtxSynchronize());
    checkCudaErrors(cuMemcpyDtoH(timer, dtimer, sizeof(clock_t) * NUM_BLOCKS * 2));
    // JP: `cuMemFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cuMemFree(dinput));
    checkCudaErrors(cuMemFree(doutput));
    checkCudaErrors(cuMemFree(dtimer));

    long double avgElapsedClocks = 0;

    for (int i = 0; i < NUM_BLOCKS; i++) {
        avgElapsedClocks += (long double)(timer[i + NUM_BLOCKS] - timer[i]);
    }

    avgElapsedClocks = avgElapsedClocks / NUM_BLOCKS;
    printf("Average clocks/block = %Lf\n", avgElapsedClocks);

    return EXIT_SUCCESS;
}
