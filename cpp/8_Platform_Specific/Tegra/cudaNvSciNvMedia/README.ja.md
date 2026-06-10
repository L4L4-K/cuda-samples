# cudaNvSciNvMedia - NvMedia CUDA Interop - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates CUDA-NvMedia interop via NvSciBuf/NvSciSync APIs. Note that this sample only supports cross build from x86_64 to aarch64, aarch64 native build is not supported. For detailed workflow of the sample please check cudaNvSciNvMedia_Readme.pdf in the sample directory.

CUDA NvSci Interop, Data Parallel Algorithms, Image Processing

Original README headings: `cudaNvSciNvMedia - NvMedia CUDA Interop`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `cudaNvSciNvMedia` as a focused example of the CUDA concepts used in `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia`.
> **日本語**
> この sample の目的は、`cudaNvSciNvMedia` の小さな実装を通して Runtime, Driver, And NVRTC, Streams And Events, Synchronization And Atomics, Memory, Kernel Launch And Indexing を具体的に追うことです。
>
> **学習メモ**
> 最初に `cuda_consumer.cu, cuda_consumer.h, main.cpp, nvmedia_producer.cpp, nvmedia_producer.h` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- The graphics, display, or platform stack named by the English README

> **日本語**
> 必要条件は英語 README と CMake/requirements を優先します。この guide は条件を置き換えず、読むべき確認点を追加します。
>
> **学習メモ**
> 実行できない場合は、source を変える前に driver、toolkit、GPU feature、library、platform guard、Python package version を確認します。

## Files

- `.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cudaNvSciNvMedia_Readme.pdf`: Supporting file used by `cudaNvSciNvMedia_Readme.pdf`.
- `cuda_consumer.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cuda_consumer.h`: Host/device declarations, helper types, constants, or library wrappers.
- `main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `nvmedia_producer.cpp`: Host-side setup, API calls, validation, and cleanup.
- `nvmedia_producer.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nvmedia_utils/cmdline.cpp`: Host-side setup, API calls, validation, and cleanup.
- `nvmedia_utils/cmdline.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nvmedia_utils/config_parser.cpp`: Host-side setup, API calls, validation, and cleanup.
- `nvmedia_utils/config_parser.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nvmedia_utils/image_utils.cpp`: Host-side setup, API calls, validation, and cleanup.
- `nvmedia_utils/image_utils.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nvmedia_utils/log_utils.cpp`: Host-side setup, API calls, validation, and cleanup.
- `nvmedia_utils/log_utils.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nvmedia_utils/misc_utils.cpp`: Host-side setup, API calls, validation, and cleanup.
- `nvmedia_utils/misc_utils.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nvsci_setup.cpp`: Host-side setup, API calls, validation, and cleanup.
- `nvsci_setup.h`: Host/device declarations, helper types, constants, or library wrappers.
- `sample.cfg`: Supporting file used by `sample.cfg`.
- `teapot.rgba`: Supporting file used by `teapot.rgba`.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `cuda_consumer.cu` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.
- Move, map, or expose input data so GPU work can read the intended values.
- Launch the kernel, graph, library call, or Python CUDA operation with the documented configuration.
- Synchronize only at the required correctness or timing boundary.
- Validate results against the CPU/reference path, generated artifact, or expected status message.
- Release CUDA, library, framework, graphics, or external resources in the reverse ownership order.

> **日本語**
> 実行の流れは setup、visibility、GPU work、sync、validation、cleanup の順に読みます。非同期 API がある場合は、host がいつ待つかを別に記録します。
>
> **学習メモ**
> CUDA の bug は kernel 本体だけでなく、copy direction、descriptor、stream dependency、cleanup order にも出ます。

## Concrete Reading Path

- `cuda_consumer.cu`: focus on `cudaExtResObj`, `cudaResObj`, `cudaSurfaceNvmediaBuf`, `cudaSurfaceObject_t`, `launch`.
- `cuda_consumer.h`: focus on `cudaExternalResInterop`, `cudaResources`, `cudaStream_t`, `cudaResObj`, `cudaArray_t`.
- `main.cpp`: focus on `cudaDeviceId`, `CUDA`, `cudaSignalerSyncObj`, `cudaResObj`, `cudaExtResObj`.
- `nvmedia_producer.cpp`: focus on `cudaDeviceId`, `launch`, `atomic`.
- `nvmedia_producer.h`: focus on `atomic`, `cudaDeviceId`.
- `nvmedia_utils/cmdline.cpp`: focus on `cudaNvSciNvMedia`, `CUDA`.
- `nvmedia_utils/cmdline.h`: focus on control flow and helper functions.
- `nvmedia_utils/config_parser.cpp`: focus on control flow and helper functions.
- `nvmedia_utils/config_parser.h`: focus on control flow and helper functions.
- `nvmedia_utils/image_utils.cpp`: focus on control flow and helper functions.
- Additional source files: 7 more support files. Use the same setup/work/sync/cleanup lens.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/Modules")

project(cudaNvSciNvMedia LANGUAGES C CXX CUDA)

# JP: `find_package`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 87 110)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cuda_consumer.cu`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/cuda_consumer.cu:29-47
```cuda
#include <cuda_runtime.h>
#include <helper_image.h>
#include <iostream>

#include "cuda_consumer.h"
#include "nvmedia_image_nvscibuf.h"
#include "nvmedia_utils/cmdline.h"

// Enable this to 1 if require cuda processed output to ppm file.
#define WRITE_OUTPUT_IMAGE 0

#define checkNvSciErrors(call)                                   \
    do {                                                         \
        NvSciError _status = call;                               \
        if (NvSciError_Success != _status) {                     \
            printf("NVSCI call in file '%s' in line %i returned" \
                   " %d, expected %d\n",                         \
                   __FILE__,                                     \
                   __LINE__,                                     \
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/cuda_consumer.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/cuda_consumer.cu:53-72
```cuda
    } while (0)

__global__ static void
yuvToGrayscale(cudaSurfaceObject_t surfaceObject, unsigned int *dstImage, int32_t imageWidth, int32_t imageHeight)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    uchar4 *dstImageUchar4 = (uchar4 *)dstImage;
    for (; x < imageWidth && y < imageHeight; x += gridDim.x * blockDim.x, y += gridDim.y * blockDim.y) {
        int           colInBytes   = x * sizeof(unsigned char);
        unsigned char luma         = surf2Dread<unsigned char>(surfaceObject, colInBytes, y);
        uchar4        grayscalePix = make_uchar4(luma, luma, luma, 0);

        dstImageUchar4[y * imageWidth + x] = grayscalePix;
    }
}

static void cudaImportNvSciSync(cudaExternalSemaphore_t &extSem, NvSciSyncObj &syncObj)
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/cuda_consumer.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/cuda_consumer.cu:91-145
```cuda
    waitParams.flags                  = 0;

    checkCudaErrors(cudaWaitExternalSemaphoresAsync(&waitSem, &waitParams, 1, stream));
}

// JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
static void signalExternalSemaphore(cudaExternalSemaphore_t &signalSem, NvSciSyncFence *fence, cudaStream_t stream)
{
    cudaExternalSemaphoreSignalParams signalParams;
    memset(&signalParams, 0, sizeof(signalParams));
    // For cross-process signaler-waiter applications need to use NvSciIpc
    // and NvSciSync[Export|Import] utilities to share the NvSciSyncFence
    // across process. This step is optional in single-process.
    signalParams.params.nvSciSync.fence = (void *)fence;
    signalParams.flags                  = 0;

    checkCudaErrors(cudaSignalExternalSemaphoresAsync(&signalSem, &signalParams, 1, stream));
}

static void yuvToGrayscaleCudaKernel(cudaExternalResInterop &cudaExtResObj, int32_t imageWidth, int32_t imageHeight)
{
#if WRITE_OUTPUT_IMAGE
    unsigned int *h_dstImage;
    // JP: `cudaMallocHost`: page-locked host memory は DMA/async copy を安定させます。通常の free ではなく対応する CUDA API で解放します。
    checkCudaErrors(cudaMallocHost(&h_dstImage, sizeof(unsigned int) * imageHeight * imageWidth));
#endif
    dim3 block(16, 16, 1);
    dim3 grid((imageWidth / block.x) + 1, (imageHeight / block.y) + 1, 1);

    // JP: `cudaExtResObj`: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    yuvToGrayscale<<<grid, block, 0, cudaExtResObj.stream>>>(
        cudaExtResObj.cudaSurfaceNvmediaBuf[0], cudaExtResObj.d_outputImage, imageWidth, imageHeight);

#if WRITE_OUTPUT_IMAGE
    // JP: `cudaMemcpyAsync`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpyAsync(h_dstImage,
                                    cudaExtResObj.d_outputImage,
                                    sizeof(unsigned int) * imageHeight * imageWidth,
                                    cudaMemcpyDeviceToHost,
                                    cudaExtResObj.stream));
    // JP: `cudaStreamSynchronize`, `cudaExtResObj`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaStreamSynchronize(cudaExtResObj.stream));
    char        outputFilename[1024];
    std::string image_filename = "Grayscale";
    strcpy(outputFilename, image_filename.c_str());
    strcpy(outputFilename + image_filename.length(), "_nvsci_out.ppm");
    sdkSavePPM4ub(outputFilename, (unsigned char *)h_dstImage, imageWidth, imageHeight);
    printf("Wrote '%s'\n", outputFilename);
    // JP: `cudaFreeHost`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFreeHost(h_dstImage));
#endif
}

static void cudaImportNvSciImage(cudaExternalResInterop &cudaExtResObj, NvSciBufObj &inputBufObj)
{
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/cuda_consumer.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/cuda_consumer.cu:242-261
```cuda
}

// JP: この連続する anchor 群では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
static cudaStream_t createCudaStream(int deviceId)
{
    checkCudaErrors(cudaSetDevice(deviceId));
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    return stream;
}

// CUDA setup buffers/synchronization objects for interop via NvSci API.
void setupCuda(cudaExternalResInterop &cudaExtResObj,
               NvSciBufObj            &inputBufObj,
               NvSciSyncObj           &syncObj,
               NvSciSyncObj           &cudaSignalerSyncObj,
               int                     deviceId)
{
    checkCudaErrors(cudaSetDevice(deviceId));
    cudaImportNvSciSync(cudaExtResObj.waitSem, syncObj);
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/cuda_consumer.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cuda_consumer.h`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/cuda_consumer.h:29-82
```cpp
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
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/cuda_consumer.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `main.cpp`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/main.cpp:30-48
```cpp
#include <iostream>
#include <signal.h>
#include <string.h>
#include <thread>

/* Nvidia headers */
#include <nvscisync.h>

#include "cuda_consumer.h"
#include "nvmedia_2d.h"
#include "nvmedia_2d_nvscisync.h"
#include "nvmedia_image.h"
#include "nvmedia_image_nvscibuf.h"
#include "nvmedia_producer.h"
#include "nvmedia_surface.h"
#include "nvmedia_utils/cmdline.h"
#include "nvmedia_utils/image_utils.h"
#include "nvsci_setup.h"

```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/main.cpp:73-92
```cpp
    if (status != NVMEDIA_STATUS_OK) {
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    TestArgs       args;
    Blit2DTest     ctx;
    NvMediaStatus  status               = NVMEDIA_STATUS_ERROR;
    NvSciSyncFence nvMediaSignalerFence = NvSciSyncFenceInitializer;
    NvSciSyncFence cudaSignalerFence    = NvSciSyncFenceInitializer;

    int      cudaDeviceId;
    uint64_t startTime, endTime;
    uint64_t operationStartTime, operationEndTime;
    double   processingTime;

    /* Read configuration from command line and config file */
    memset(&args, 0, sizeof(TestArgs));
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvmedia_producer.cpp`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_producer.cpp:29-47
```cpp
#include <iostream>
#include <string.h>
/* Nvidia headers */
#include "nvmedia_2d.h"
#include "nvmedia_2d_nvscisync.h"
#include "nvmedia_image.h"
#include "nvmedia_image_nvscibuf.h"
#include "nvmedia_producer.h"
#include "nvmedia_surface.h"
#include "nvmedia_utils/cmdline.h"
#include "nvmedia_utils/image_utils.h"
#include "nvsci_setup.h"

NvMediaImage *NvMediaImageCreateUsingNvScibuf(NvMediaDevice              *device,
                                              NvMediaSurfaceType          type,
                                              const NvMediaSurfAllocAttr *attrs,
                                              uint32_t                    numAttrs,
                                              uint32_t                    flags,
                                              NvSciBufObj                &bufobj,
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_producer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_producer.cpp:350-369
```cpp

void cleanupNvMedia(Blit2DTest *ctx)
{
    cleanup(ctx);
    // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    free(ctx->dstBuffPitches);
    free(ctx->dstBuffer);
    free(ctx->dstBuff);
}

void setupNvMedia(TestArgs     *args,
                  Blit2DTest   *ctx,
                  NvSciBufObj  &srcNvSciBufobj,
                  NvSciBufObj  &dstNvSciBufobj,
                  NvSciSyncObj &syncObj,
                  NvSciSyncObj &preSyncObj,
                  int           cudaDeviceId)
{
    NvMediaStatus status;
    status = NvMediaImageNvSciBufInit();
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_producer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvmedia_producer.h`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_producer.h:29-55
```cpp
#ifndef __NVMEDIA_PRODUCER_H__
#define __NVMEDIA_PRODUCER_H__
#include "nvmedia_2d.h"
#include "nvmedia_image.h"
#include "nvmedia_image_nvscibuf.h"
#include "nvmedia_surface.h"
#include "nvmedia_utils/cmdline.h"
#include "nvmedia_utils/image_utils.h"
#include "nvscisync.h"

void runNvMediaBlit2D(TestArgs       *args,
                      Blit2DTest     *ctx,
                      NvSciSyncObj   &syncObj,
                      NvSciSyncFence *preSyncFence,
                      NvSciSyncFence *fence);
void runNvMediaBlit2D(TestArgs *args, Blit2DTest *ctx);
void setupNvMedia(TestArgs     *args,
                  Blit2DTest   *ctx,
                  NvSciBufObj  &srcNvSciBufobj,
                  NvSciBufObj  &dstNvSciBufobj,
                  NvSciSyncObj &syncObj,
                  NvSciSyncObj &preSyncObj,
                  int           cudaDeviceId);
void setupNvMedia(TestArgs *args, Blit2DTest *ctx);
void cleanupNvMedia(Blit2DTest *ctx, NvSciSyncObj &syncObj, NvSciSyncObj &preSyncObj);
void cleanupNvMedia(Blit2DTest *ctx);
#endif
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_producer.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvmedia_utils/cmdline.cpp`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/cmdline.cpp:31-49
```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Nvidia headers */
#include "cmdline.h"
#include "config_parser.h"
#include "helper_cuda.h"
#include "log_utils.h"
#include "misc_utils.h"

/* see cmdline.h for details */
void PrintUsage()
{
    printf("cudaNvSciNvMedia\n");
    printf("Usage: cudaNvSciNvMedia [options]\n");
    printf("Options:\n");
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/cmdline.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvmedia_utils/cmdline.h`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/cmdline.h:30-93
```cpp
#ifndef _NVMEDIA_2D_CMD_LINE_H_
#define _NVMEDIA_2D_CMD_LINE_H_

/* Include header containing NvMedia2D declarations */
#include "nvmedia_2d.h"

/* Maximum length of the path including file name */
#define FILE_NAME_SIZE 1024

/* TestArgs contains all arguments required to run the 2D test */
typedef struct _TestArgs
{
    char inputFileName[FILE_NAME_SIZE];

    NvMediaSurfAllocAttr srcSurfAllocAttrs[NVM_SURF_ALLOC_ATTR_MAX];
    NvMediaSurfAllocAttr dstSurfAllocAttrs[NVM_SURF_ALLOC_ATTR_MAX];
    uint32_t             numSurfAllocAttrs;

    NvMediaSurfFormatAttr srcSurfFormatAttrs[NVM_SURF_FMT_ATTR_MAX];
    NvMediaSurfFormatAttr dstSurfFormatAttrs[NVM_SURF_FMT_ATTR_MAX];

    NvMediaRect             srcRect;
    NvMediaRect             dstRect;
    NvMedia2DBlitParameters blitParams;
    size_t                  iterations;
} TestArgs;

typedef struct
{
    NvMediaDevice *device;
    /* I2D for 2D blit processing */
    NvMedia2D    *i2d;
    NvMediaImage *srcImage;
    NvMediaImage *dstImage;
    NvMediaRect  *srcRect;
    NvMediaRect  *dstRect;
    uint8_t     **dstBuff;
    uint32_t     *dstBuffPitches;
    uint8_t      *dstBuffer;
    uint32_t      numSurfaces;
    uint32_t      bytesPerPixel;
    uint32_t      heightSurface;
    uint32_t      widthSurface;
    float        *xScalePtr;
    float        *yScalePtr;

} Blit2DTest;

/* Prints application usage options */
void PrintUsage(void);

/* Parses command line arguments.
 * Also parses any configuration files supplied in the command line arguments.
 * Arguments:
 * argc
 *    (in) Number of tokens in the command line
 * argv
 *    (in) Command line tokens
 * args
 *    (out) Pointer to test arguments structure
 */
int ParseArgs(int argc, char **argv, TestArgs *args);

#endif /* _NVMEDIA_2D_CMD_LINE_H_ */
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/cmdline.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvmedia_utils/config_parser.cpp`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/config_parser.cpp:29-47
```cpp
#include "config_parser.h"

#include <stdlib.h>

#include "log_utils.h"
#if defined(__QNX__)
#include <strings.h>
#endif

static NvMediaStatus GetParamIndex(ConfigParamsMap *paramsMap, char *paramName, unsigned int *index)
{
    int i = 0;

    while (paramsMap[i].paramName != NULL) {
        if (strcasecmp(paramsMap[i].paramName, paramName) == 0) {
            *index = i;
            return NVMEDIA_STATUS_OK;
        }
        else {
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/config_parser.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/config_parser.cpp:113-132
```cpp
    if (fseek(file, 0, SEEK_SET) != 0) {
        printf("Parser_GetFileContent: Cannot fseek in configuration file %s\n", filename);
        return NVMEDIA_STATUS_ERROR;
    }

    fileCotent = (char *)malloc(fileSize + 1);
    if (fileCotent == NULL) {
        printf("Parser_GetFileContent: Failed allocating buffer for file Content\n");
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    fileSize             = (long)fread(fileCotent, 1, fileSize, file);
    fileCotent[fileSize] = '\0';
    *fileContentOut      = fileCotent;

    fclose(file);

    return NVMEDIA_STATUS_OK;
}

```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/config_parser.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/config_parser.cpp:252-271
```cpp
               __func__,
               numParams,
               numSetsInSection);
        if (configContentBuf) {
            // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
            free(configContentBuf);
        }
        return NVMEDIA_STATUS_ERROR;
    }

    // Stage 2: Go through the list of items and save their values in parameters map
    for (i = 0; i < itemsCount; i += 3) {
        if (ConfigParser_GetSectionIndexByName(sectionsMap, items[i], &currItemIndex) == NVMEDIA_STATUS_OK) {
            currSectionId = atoi(items[i + 1]);
            currSectionId--;
            LOG_DBG("ConfigParser_ParseFile: Parsing section %s index %d\n", items[i], currSectionId);
            i -= 1;
            continue;
        }

```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/config_parser.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvmedia_utils/config_parser.h`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/config_parser.h:29-47
```cpp
#ifndef _NVMEDIA_TEST_CONFIG_PARSER_H_
#define _NVMEDIA_TEST_CONFIG_PARSER_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "nvmedia_core.h"
#include "nvmedia_surface.h"

#define MAX_ITEMS_TO_PARSE 10000

    typedef enum _ParamType {
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/config_parser.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvmedia_utils/image_utils.cpp`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/image_utils.cpp:30-48
```cpp
#include "image_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "misc_utils.h"
#include "nvmedia_surface.h"

#define MAXM_NUM_SURFACES 6

typedef struct
{
    float        heightFactor[6];
    float        widthFactor[6];
    unsigned int numSurfaces;
} ImgUtilSurfParams;

```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/image_utils.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/image_utils.cpp:384-403
```cpp
        printf("%s: NvMediaImageLock() failed\n", __func__);
        return status;
    }
    NvMediaImageUnlock(image);

    ctx->dstBuff = (uint8_t **)malloc(sizeof(uint8_t *) * MAXM_NUM_SURFACES);
    if (!ctx->dstBuff) {
        printf("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    ctx->dstBuffPitches = (uint32_t *)calloc(1, sizeof(uint32_t) * MAXM_NUM_SURFACES);
    if (!ctx->dstBuffPitches) {
        printf("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    ctx->heightSurface = surfaceMap.height;
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/image_utils.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/image_utils.cpp:632-651
```cpp
    }

done:
    if (pBuff) {
        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(pBuff);
    }

    if (pBuffPitches) {
        free(pBuffPitches);
    }

    if (pBuffer) {
        free(pBuffer);
    }

    if (file) {
        fclose(file);
    }

```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/image_utils.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvmedia_utils/image_utils.h`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/image_utils.h:29-47
```cpp
#ifndef _NVMEDIA_TEST_IMAGE_UTILS_H_
#define _NVMEDIA_TEST_IMAGE_UTILS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "cmdline.h"
#include "misc_utils.h"
#include "nvmedia_core.h"
#include "nvmedia_image.h"
#include "nvmedia_surface.h"

#if (NV_IS_SAFETY == 1)
#include "nvmedia_image_internal.h"
#endif

#define PACK_RGBA(R, G, B, A) (((uint32_t)(A) << 24) | ((uint32_t)(B) << 16) | ((uint32_t)(G) << 8) | (uint32_t)(R))
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/image_utils.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvmedia_utils/log_utils.cpp`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/log_utils.cpp:30-48
```cpp
#include <stdio.h>
#include <string.h>
#ifdef NVMEDIA_ANDROID
#define LOG_TAG    "nvmedia_common"
#define LOG_NDEBUG 1
#include <utils/Log.h>
#endif
#ifdef NVMEDIA_QNX
#include <sys/slog.h>
#endif

#include "log_utils.h"

#ifdef NVMEDIA_QNX
#define NV_SLOGCODE 0xAAAA
#endif
#define MAX_STATS_LEN 500

#define LOG_BUFFER_BYTES 1024
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/log_utils.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/log_utils.cpp:114-133
```cpp

    va_start(ap, format);
    vsnprintf(str + strlen(str), sizeof(str) - strlen(str), format, ap);

    if (msg_style == LOG_STYLE_NORMAL) {
        // Add trailing new line char
        if (strlen(str) && str[strlen(str) - 1] != '\n')
            strcat(str, "\n");
    }
    else if (msg_style == LOG_STYLE_FUNCTION_LINE) {
        // Remove trailing new line char
        if (strlen(str) && str[strlen(str) - 1] == '\n')
            str[strlen(str) - 1] = 0;

        // Add function and line info
        snprintf(str + +strlen(str), sizeof(str) - strlen(str), " at %s():%d\n", functionName, lineNumber);
    }

#ifdef NVMEDIA_ANDROID
    switch (msg_level) {
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/log_utils.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvmedia_utils/log_utils.h`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/log_utils.h:29-47
```cpp
#ifndef _NVMEDIA_TEST_LOG_UTILS_H_
#define _NVMEDIA_TEST_LOG_UTILS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdarg.h>
#include <stdio.h>

    enum LogLevel {
        LEVEL_ERR  = 0,
        LEVEL_WARN = 1,
        LEVEL_INFO = 2,
        LEVEL_DBG  = 3,
    };

    enum LogStyle { LOG_STYLE_NORMAL = 0, LOG_STYLE_FUNCTION_LINE };
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/log_utils.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvmedia_utils/misc_utils.cpp`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/misc_utils.cpp:29-61
```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#if defined(__QNX__)
#include <sys/time.h>
#endif
#include "misc_utils.h"

uint32_t u32(const uint8_t *ptr) { return ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24); }

NvMediaStatus GetTimeMicroSec(uint64_t *uTime)
{
    struct timespec t;
#if !(defined(CLOCK_MONOTONIC) && defined(_POSIX_MONOTONIC_CLOCK) && _POSIX_MONOTONIC_CLOCK >= 0 && _POSIX_TIMERS > 0)
    struct timeval tv;
#endif

    if (!uTime)
        return NVMEDIA_STATUS_BAD_PARAMETER;

#if !(defined(CLOCK_MONOTONIC) && defined(_POSIX_MONOTONIC_CLOCK) && _POSIX_MONOTONIC_CLOCK >= 0 && _POSIX_TIMERS > 0)
    gettimeofday(&tv, NULL);
    t.tv_sec  = tv.tv_sec;
    t.tv_nsec = tv.tv_usec * 1000L;
#else
    clock_gettime(CLOCK_MONOTONIC, &t);
#endif

    *uTime = (uint64_t)t.tv_sec * 1000000LL + (uint64_t)t.tv_nsec / 1000LL;
    return NVMEDIA_STATUS_OK;
}
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/misc_utils.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvmedia_utils/misc_utils.h`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/misc_utils.h:29-74
```cpp
#ifndef _NVMEDIA_TEST_MISC_UTILS_H_
#define _NVMEDIA_TEST_MISC_UTILS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "nvmedia_common.h"
#include "nvmedia_core.h"

#ifndef __INTEGRITY
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

    typedef enum { LSB_ALIGNED, MSB_ALIGNED } PixelAlignment;


    //  u32
    //
    //    u32()  Reads 4 bytes from buffer and returns the read value
    //
    //  Arguments:
    //
    //   ptr
    //      (in) Input buffer

    uint32_t u32(const uint8_t *ptr);

    //  GetTimeMicroSec
    //
    //    GetTimeMicroSec()  Returns current time in microseconds
    //
    //  Arguments:
    //
    //   uTime
    //      (out) Pointer to current time in microseconds

    NvMediaStatus GetTimeMicroSec(uint64_t *uTime);

#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_TEST_MISC_UTILS_H_ */
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/misc_utils.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvsci_setup.cpp`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvsci_setup.cpp:29-47
```cpp
#include "nvsci_setup.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "nvmedia_2d_nvscisync.h"
#include "nvmedia_utils/cmdline.h"

#define checkNvSciErrors(call)                                   \
    do {                                                         \
        NvSciError _status = call;                               \
        if (NvSciError_Success != _status) {                     \
            printf("NVSCI call in file '%s' in line %i returned" \
                   " %d, expected %d\n",                         \
                   __FILE__,                                     \
                   __LINE__,                                     \
                   _status,                                      \
                   NvSciError_Success);                          \
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvsci_setup.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvsci_setup.cpp:65-84
```cpp
    if (status != NVMEDIA_STATUS_OK) {
        printf("%s: NvMedia2DFillNvSciSyncAttrList failed\n", __func__);
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaSetDevice(cudaDeviceId));
    checkCudaErrors(cudaDeviceGetNvSciSyncAttributes(waiterAttrList, cudaDeviceId, cudaNvSciSyncAttrWait));

    syncUnreconciledList[0] = signalerAttrList;
    syncUnreconciledList[1] = waiterAttrList;
    checkNvSciErrors(NvSciSyncAttrListReconcile(syncUnreconciledList, 2, &syncReconciledList, &syncConflictList));
    checkNvSciErrors(NvSciSyncObjAlloc(syncReconciledList, &syncObj));

    NvSciSyncAttrListFree(signalerAttrList);
    NvSciSyncAttrListFree(waiterAttrList);
    if (syncConflictList != nullptr) {
        NvSciSyncAttrListFree(syncConflictList);
    }
}

```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvsci_setup.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvsci_setup.h`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvsci_setup.h:29-41
```cpp
#ifndef __NVSCI_SETUP_H__
#define __NVSCI_SETUP_H__
#include <nvscibuf.h>
#include <nvscisync.h>

#include "nvmedia_utils/cmdline.h"

void setupNvMediaSignalerNvSciSync(Blit2DTest *ctx, NvSciSyncObj &syncObj, int cudaDeviceId);
void setupCudaSignalerNvSciSync(Blit2DTest *ctx, NvSciSyncObj &syncObj, int cudaDeviceId);
void setupNvSciBuf(NvSciBufObj &bufobj, NvSciBufAttrList &nvmediaAttrlist, int cudaDeviceId);
void cleanupNvSciBuf(NvSciBufObj &Bufobj);
void cleanupNvSciSync(NvSciSyncObj &syncObj);
#endif
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvsci_setup.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaExtResObj` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaResObj` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDeviceId` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaSurfaceNvmediaBuf` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaExternalResInterop` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaSurfaceObject_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaSignalerSyncObj` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaResources` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaStream_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaStreamSynchronize` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaSetDevice` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaCreateChannelDesc` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaArray_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- Runtime/Driver/NVRTC sample では、compile/load した module と launch する kernel 名の対応が重要です。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。
- 同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target cudaNvSciNvMedia
ctest --test-dir build -R cudaNvSciNvMedia
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

Run the sample as documented and compare its output with the original README, validation message, generated file, or reference result.
> **日本語**
> 期待結果は英語の出力文字列、README の validation、生成 file、または reference result と照合します。
>
> **学習メモ**
> `PASS`、`Test passed`、error code、timing label などの出力文字列は翻訳せず、source と同じ表記で確認します。

## Common Mistakes

- API 名や target 名を翻訳してしまい、README や build command と対応できなくなる。
- allocation size を byte で渡す API と element count で考える loop を混同する。
- kernel launch が非同期であることを忘れ、同期前の結果を host 側で読んでしまう。
- different stream 間に依存があるのに event や explicit sync を置かない。
- JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaExtResObj` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Runtime, Driver, And NVRTC](../../../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [Streams And Events](../../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Memory](../../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
