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

//
// DESCRIPTION:   Simple CUDA consumer rendering sample app
//

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#include "eglstrm_common.h"

extern bool isCrossDevice;

__device__ static unsigned int numErrors = 0, errorFound = 0;
__device__ void                checkProducerDataGPU(char *data, int size, char expectedVal, int frameNumber)
{
    // JP: `blockDim`, `blockIdx`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    if ((data[blockDim.x * blockIdx.x + threadIdx.x] != expectedVal) && (!errorFound)) {
        printf("Producer FOUND:%d expected: %d at %d for trial %d %d\n",
               data[blockDim.x * blockIdx.x + threadIdx.x],
               expectedVal,
               (blockDim.x * blockIdx.x + threadIdx.x),
               frameNumber,
               numErrors);
        numErrors++;
        errorFound = 1;
        return;
    }
}

__device__ void checkConsumerDataGPU(char *data, int size, char expectedVal, int frameNumber)
{
    if ((data[blockDim.x * blockIdx.x + threadIdx.x] != expectedVal) && (!errorFound)) {
        printf("Consumer FOUND:%d expected: %d at %d for trial %d %d\n",
               data[blockDim.x * blockIdx.x + threadIdx.x],
               expectedVal,
               (blockDim.x * blockIdx.x + threadIdx.x),
               frameNumber,
               numErrors);
        numErrors++;
        errorFound = 1;
        return;
    }
}

__global__ void writeDataToBuffer(char *pSrc, char newVal) { pSrc[blockDim.x * blockIdx.x + threadIdx.x] = newVal; }

__global__ void testKernelConsumer(char *pSrc, char size, char expectedVal, char newVal, int frameNumber)
{
    checkConsumerDataGPU(pSrc, size, expectedVal, frameNumber);
}

__global__ void testKernelProducer(char *pSrc, char size, char expectedVal, char newVal, int frameNumber)
{
    checkProducerDataGPU(pSrc, size, expectedVal, frameNumber);
}
__global__ void getNumErrors(int *numErr) { *numErr = numErrors; }

// JP: `cudaError_t`, `cudaProducer_filter`, `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
cudaError_t cudaProducer_filter(cudaStream_t pStream,
                                char        *pSrc,
                                int          width,
                                int          height,
                                char         expectedVal,
                                char         newVal,
                                int          frameNumber)
{
    // in case where consumer is on dgpu and producer is on igpu when return is
    // called the frame is not copied back to igpu. So the consumer changes is not
    // visible to producer
    if (isCrossDevice == 0) {
        // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
        testKernelProducer<<<(width * height) / 1024, 1024, 1, pStream>>>(
            pSrc, width * height, expectedVal, newVal, frameNumber);
    }
    writeDataToBuffer<<<(width * height) / 1024, 1024, 1, pStream>>>(pSrc, newVal);
    return cudaSuccess;
};

cudaError_t cudaConsumer_filter(cudaStream_t cStream,
                                char        *pSrc,
                                int          width,
                                int          height,
                                char         expectedVal,
                                char         newVal,
                                int          frameNumber)
{
    testKernelConsumer<<<(width * height) / 1024, 1024, 1, cStream>>>(
        pSrc, width * height, expectedVal, newVal, frameNumber);
    writeDataToBuffer<<<(width * height) / 1024, 1024, 1, cStream>>>(pSrc, newVal);
    return cudaSuccess;
};

cudaError_t cudaGetValueMismatch()
{
    int         numErr_h;
    int        *numErr_d = NULL;
    cudaError_t err      = cudaSuccess;
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    err                  = cudaMalloc(&numErr_d, sizeof(int));
    if (err != cudaSuccess) {
        printf("Cuda Main: cudaMalloc failed with %s\n", cudaGetErrorString(err));
        return err;
    }
    getNumErrors<<<1, 1>>>(numErr_d);
    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Cuda Main: cudaDeviceSynchronize failed with %s\n", cudaGetErrorString(err));
    }
    // JP: `cudaMemcpy`, `cudaMemcpyDeviceToHost`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    err = cudaMemcpy(&numErr_h, numErr_d, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Cuda Main: cudaMemcpy failed with %s\n", cudaGetErrorString(err));
        // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        cudaFree(numErr_d);
        return err;
    }
    err = cudaFree(numErr_d);
    if (err != cudaSuccess) {
        printf("Cuda Main: cudaFree failed with %s\n", cudaGetErrorString(err));
        return err;
    }
    if (numErr_h > 0) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}
