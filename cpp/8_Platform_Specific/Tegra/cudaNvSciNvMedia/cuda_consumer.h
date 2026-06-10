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
// JP: この file では memory ownership と host/device transfer、stream/event による非同期実行と同期、同期範囲、atomic、memory ordering を確認します。英語の識別子/API/出力文字列は保持します。

#ifndef __CUDA_BUFIMPORT_KERNEL_H__
#define __CUDA_BUFIMPORT_KERNEL_H__

#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "nvmedia_image_nvscibuf.h"
#include "nvmedia_utils/cmdline.h"
#include "nvscisync.h"

struct cudaExternalResInterop
{
    cudaMipmappedArray_t   *d_mipmapArray;
    cudaArray_t            *d_mipLevelArray;
    cudaSurfaceObject_t    *cudaSurfaceNvmediaBuf;
    // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    cudaStream_t            stream;
    cudaExternalMemory_t    extMemImageBuf;
    cudaExternalSemaphore_t waitSem;
    cudaExternalSemaphore_t signalSem;

    int32_t       planeCount;
    uint64_t     *planeOffset;
    int32_t      *imageWidth;
    int32_t      *imageHeight;
    unsigned int *d_outputImage;
};

struct cudaResources
{
    cudaArray_t         *d_yuvArray;
    // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    cudaStream_t         stream;
    cudaSurfaceObject_t *cudaSurfaceNvmediaBuf;
    unsigned int        *d_outputImage;
};

void runCudaOperation(cudaExternalResInterop &cudaExtResObj,
                      NvSciSyncFence         *fence,
                      NvSciSyncFence         *cudaSignalfence,
                      int                     deviceId,
                      int                     iterations);
void runCudaOperation(Blit2DTest *ctx, cudaResources &cudaResObj, int deviceId);

void setupCuda(cudaExternalResInterop &cudaExtResObj,
               NvSciBufObj            &inputBufObj,
               NvSciSyncObj           &syncObj,
               NvSciSyncObj           &cudaSignalerSyncObj,
               int                     deviceId);
void setupCuda(Blit2DTest *ctx, cudaResources &cudaResObj, int deviceId);
void cleanupCuda(cudaExternalResInterop &cudaObjs);
void cleanupCuda(Blit2DTest *ctx, cudaResources &cudaResObj);

#endif
