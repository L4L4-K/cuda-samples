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
 * Demonstration of inline PTX (assembly language) usage in CUDA kernels
 */

// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

__global__ void sequence_gpu(int *d_ptr, int length)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int elemID = blockIdx.x * blockDim.x + threadIdx.x;

    if (elemID < length) {
        unsigned int laneid;
        // This command gets the lane ID within the current warp
        asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
        d_ptr[elemID] = laneid;
    }
}


void sequence_cpu(int *h_ptr, int length)
{
    for (int elemID = 0; elemID < length; elemID++) {
        h_ptr[elemID] = elemID % 32;
    }
}

int main(int argc, char **argv)
{
    printf("CUDA inline PTX assembler sample\n");

    const int N = 1000;

    int dev = findCudaDevice(argc, (const char **)argv);

    if (dev == -1) {
        return EXIT_FAILURE;
    }

    int *d_ptr;
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc(&d_ptr, N * sizeof(int)));

    int *h_ptr;
    // JP: `cudaMallocHost`: page-locked host memory は DMA/async copy を安定させます。通常の free ではなく対応する CUDA API で解放します。
    checkCudaErrors(cudaMallocHost(&h_ptr, N * sizeof(int)));

    dim3 cudaBlockSize(256, 1, 1);
    dim3 cudaGridSize((N + cudaBlockSize.x - 1) / cudaBlockSize.x, 1, 1);
    // JP: `cudaGridSize`, `cudaBlockSize`: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    sequence_gpu<<<cudaGridSize, cudaBlockSize>>>(d_ptr, N);
    checkCudaErrors(cudaGetLastError());
    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaDeviceSynchronize());

    sequence_cpu(h_ptr, N);

    int *h_d_ptr;
    checkCudaErrors(cudaMallocHost(&h_d_ptr, N * sizeof(int)));
    // JP: `cudaMemcpy`, `cudaMemcpyDeviceToHost`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpy(h_d_ptr, d_ptr, N * sizeof(int), cudaMemcpyDeviceToHost));

    bool bValid = true;

    for (int i = 0; i < N && bValid; i++) {
        if (h_ptr[i] != h_d_ptr[i]) {
            bValid = false;
        }
    }

    printf("Test %s.\n", bValid ? "Successful" : "Failed");

    // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFree(d_ptr));
    checkCudaErrors(cudaFreeHost(h_ptr));
    checkCudaErrors(cudaFreeHost(h_d_ptr));

    return bValid ? EXIT_SUCCESS : EXIT_FAILURE;
}
