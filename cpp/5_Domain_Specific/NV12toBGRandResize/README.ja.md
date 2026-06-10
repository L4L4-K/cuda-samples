# NV12toBGRandResize - NV12toBGRandResize - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This code shows two ways to convert and resize NV12 frames to BGR 3 planars frames using CUDA in batch. Way-1, Convert NV12 Input to BGR @ Input Resolution-1, then Resize to Resolution#2. Way-2, resize NV12 Input to Resolution#2 then convert it to BGR Output. NVIDIA HW Decoder, both dGPU and Tegra, normally outputs NV12 pitch format frames. For the inference using TensorRT, the input frame needs to be BGR planar format with possibly different size. So, conversion and resizing from NV12 to BGR planar is usually required for the inference following decoding. This CUDA code provides a reference implementation for conversion and resizing.

Graphics Interop, Image Processing, Video Processing

Original README headings: `NV12toBGRandResize - NV12toBGRandResize`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/NV12toBGRandResize` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `NV12toBGRandResize` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/NV12toBGRandResize`.
> **日本語**
> この sample の目的は、`NV12toBGRandResize` の小さな実装を通して CUDA Graphs, Streams And Events, Unified Memory, Synchronization And Atomics, Performance を具体的に追うことです。
>
> **学習メモ**
> 最初に `bgr_resize.cu, nv12_resize.cu, nv12_to_bgr_planar.cu, resize_convert.h, resize_convert_main.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit

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
- `bgr_resize.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `data/test1280x720.nv12`: Supporting file used by `data/test1280x720.nv12`.
- `data/test1920x1080.nv12`: Supporting file used by `data/test1920x1080.nv12`.
- `data/test640x480.nv12`: Supporting file used by `data/test640x480.nv12`.
- `nv12_resize.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `nv12_to_bgr_planar.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `resize_convert.h`: Host/device declarations, helper types, constants, or library wrappers.
- `resize_convert_main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `utils.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `utils.h`: Host/device declarations, helper types, constants, or library wrappers.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `bgr_resize.cu` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Capture or build CUDA Graph nodes, instantiate the graph, then launch the executable graph.
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

- `bgr_resize.cu`: focus on `blockIdx`, `launch`, `threadIdx`, `blockDim`, `cudaStream_t`.
- `nv12_resize.cu`: focus on `cudaTextureObject_t`, `blockIdx`, `launch`, `threadIdx`, `blockDim`.
- `nv12_to_bgr_planar.cu`: focus on `threadIdx`, `blockIdx`, `blockDim`, `launch`, `cudaStream_t`.
- `resize_convert.h`: focus on `cudaStream_t`.
- `resize_convert_main.cpp`: focus on `cudaEventRecord`, `cudaMalloc`, `CUDA`, `cudaEventSynchronize`, `cudaFree`.
- `utils.cu`: focus on `cudaMemcpy`, `cudaMemcpyDeviceToHost`, `launch`, `threadIdx`, `blockIdx`.
- `utils.h`: focus on control flow and helper functions.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/NV12toBGRandResize/CMakeLists.txt:1-47
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(NV12toBGRandResize LANGUAGES C CXX CUDA)

# JP: `find_package`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for NV12toBGRandResize
add_executable(NV12toBGRandResize bgr_resize.cu nv12_resize.cu nv12_to_bgr_planar.cu resize_convert_main.cpp utils.cu)

target_compile_options(NV12toBGRandResize PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(NV12toBGRandResize PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(NV12toBGRandResize PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(NV12toBGRandResize PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

add_custom_command(TARGET NV12toBGRandResize POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    ${CMAKE_CURRENT_BINARY_DIR}/data
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/5_Domain_Specific/NV12toBGRandResize/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bgr_resize.cu`

Source: cpp/5_Domain_Specific/NV12toBGRandResize/bgr_resize.cu:32-64
```cuda
#include <cuda.h>
#include <cuda_runtime.h>

#include "resize_convert.h"

__global__ void resizeBGRplanarBatchKernel(cudaTextureObject_t texSrc,
                                           float              *pDst,
                                           int                 nDstPitch,
                                           int                 nDstHeight,
                                           int                 nSrcHeight,
                                           int                 batch,
                                           float               scaleX,
                                           float               scaleY,
                                           int                 cropX,
                                           int                 cropY,
                                           int                 cropW,
                                           int                 cropH)
{
    // JP: `threadIdx`, `blockIdx`, `blockDim`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= (int)(cropW / scaleX) || y >= (int)(cropH / scaleY))
        return;

    int    frameSize = nDstPitch * nDstHeight;
    float *p         = NULL;
    for (int i = blockIdx.z; i < batch; i += gridDim.z) {
#pragma unroll
        for (int channel = 0; channel < 3; channel++) {
            p  = pDst + i * 3 * frameSize + y * nDstPitch + x + channel * frameSize;
            *p = tex2D<float>(texSrc, x * scaleX + cropX, ((3 * i + channel) * nSrcHeight + y * scaleY + cropY));
        }
```

> JP: この抜粋は `cpp/5_Domain_Specific/NV12toBGRandResize/bgr_resize.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/NV12toBGRandResize/bgr_resize.cu:139-158
```cuda
        resizeBGRplanarBatchKernel<<<grid, block, 0, stream>>>(
            texSrc[iTile], dpDstNew, nDstPitch, nDstHeight, nSrcHeight, bs, scaleX, scaleY, cropX, cropY, cropW, cropH);
    }

    for (iTile = 0; iTile < nTiles; ++iTile)
        // JP: `cudaDestroyTextureObject`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        checkCudaErrors(cudaDestroyTextureObject(texSrc[iTile]));
}

void resizeBGRplanarBatch(float       *dpSrc,
                          int          nSrcPitch,
                          int          nSrcWidth,
                          int          nSrcHeight,
                          float       *dpDst,
                          int          nDstPitch,
                          int          nDstWidth,
                          int          nDstHeight,
                          int          nBatchSize,
                          // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
                          cudaStream_t stream,
```

> JP: この抜粋は `cpp/5_Domain_Specific/NV12toBGRandResize/bgr_resize.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nv12_resize.cu`

Source: cpp/5_Domain_Specific/NV12toBGRandResize/nv12_resize.cu:31-60
```cuda
#include <cuda.h>
#include <cuda_runtime.h>

#include "resize_convert.h"

__global__ static void resizeNV12BatchKernel(cudaTextureObject_t texSrcLuma,
                                             cudaTextureObject_t texSrcChroma,
                                             uint8_t            *pDstNv12,
                                             int                 nSrcWidth,
                                             int                 nSrcHeight,
                                             int                 nDstPitch,
                                             int                 nDstWidth,
                                             int                 nDstHeight,
                                             int                 nBatchSize)
{
    // JP: `threadIdx`, `blockIdx`, `blockDim`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int px = x * 2, py = y * 2;

    if ((px + 1) >= nDstWidth || (py + 1) >= nDstHeight)
        return;

    float fxScale = 1.0f * nSrcWidth / nDstWidth;
    float fyScale = 1.0f * nSrcHeight / nDstHeight;

    uint8_t *p            = pDstNv12 + px + py * nDstPitch;
    int      hh           = nDstHeight * 3 / 2;
    int      nByte        = nDstPitch * hh;
```

> JP: この抜粋は `cpp/5_Domain_Specific/NV12toBGRandResize/nv12_resize.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/NV12toBGRandResize/nv12_resize.cu:120-128
```cuda
    dim3 grid((nDstWidth / 2 + block.x) / block.x, (nDstHeight / 2 + block.y) / block.y, blockDimZ);
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    resizeNV12BatchKernel<<<grid, block, 0, stream>>>(
        texLuma, texChroma, dpDst, nSrcWidth, nSrcHeight, nDstPitch, nDstWidth, nDstHeight, nBatchSize);

    // JP: `cudaDestroyTextureObject`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaDestroyTextureObject(texLuma));
    checkCudaErrors(cudaDestroyTextureObject(texChroma));
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/NV12toBGRandResize/nv12_resize.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nv12_to_bgr_planar.cu`

Source: cpp/5_Domain_Specific/NV12toBGRandResize/nv12_to_bgr_planar.cu:32-67
```cuda
#include <cuda.h>
#include <cuda_runtime.h>

#include "resize_convert.h"

#define CONV_THREADS_X 64
#define CONV_THREADS_Y 10

__forceinline__ __device__ static float clampF(float x, float lower, float upper)
{
    return x < lower ? lower : (x > upper ? upper : x);
}

__global__ static void nv12ToBGRplanarBatchKernel(const uint8_t *pNv12,
                                                  int            nNv12Pitch,
                                                  float         *pBgr,
                                                  int            nRgbPitch,
                                                  int            nWidth,
                                                  int            nHeight,
                                                  int            nBatchSize)
{
    // JP: `threadIdx`, `blockIdx`, `blockDim`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x << 2) + 1 > nWidth || (y << 1) + 1 > nHeight)
        return;

    const uint8_t *__restrict__ pSrc = pNv12;

    for (int i = blockIdx.z; i < nBatchSize; i += gridDim.z) {
        pSrc = pNv12 + i * ((nHeight * nNv12Pitch * 3) >> 1) + (x << 2) + (y << 1) * nNv12Pitch;
        uchar4 luma2x01, luma2x23, uv2;
        *(uint32_t *)&luma2x01 = *(uint32_t *)pSrc;
        *(uint32_t *)&luma2x23 = *(uint32_t *)(pSrc + nNv12Pitch);
        *(uint32_t *)&uv2      = *(uint32_t *)(pSrc + (nHeight - y) * nNv12Pitch);
```

> JP: この抜粋は `cpp/5_Domain_Specific/NV12toBGRandResize/nv12_to_bgr_planar.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `resize_convert.h`

Source: cpp/5_Domain_Specific/NV12toBGRandResize/resize_convert.h:30-77
```cpp
#ifndef __H_RESIZE_CONVERT__
#define __H_RESIZE_CONVERT__

#include <helper_cuda.h>
#include <iostream>

// nv12 resize
extern "C" void resizeNV12Batch(uint8_t     *dpSrc,
                                int          nSrcPitch,
                                int          nSrcWidth,
                                int          nSrcHeight,
                                uint8_t     *dpDst,
                                int          nDstPitch,
                                int          nDstWidth,
                                int          nDstHeight,
                                int          nBatchSize,
                                // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
                                cudaStream_t stream = 0);

// bgr resize
extern "C" void resizeBGRplanarBatch(float       *dpSrc,
                                     int          nSrcPitch,
                                     int          nSrcWidth,
                                     int          nSrcHeight,
                                     float       *dpDst,
                                     int          nDstPitch,
                                     int          nDstWidth,
                                     int          nDstHeight,
                                     int          nBatchSize,
                                     // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
                                     cudaStream_t stream            = 0,
                                     int          cropX             = 0,
                                     int          cropY             = 0,
                                     int          cropW             = 0,
                                     int          cropH             = 0,
                                     bool         whSameResizeRatio = false);

// NV12 to bgr planar
extern "C" void nv12ToBGRplanarBatch(uint8_t     *pNv12,
                                     int          nNv12Pitch,
                                     float       *pRgb,
                                     int          nRgbPitch,
                                     int          nWidth,
                                     int          nHeight,
                                     int          nBatchSize,
                                     // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
                                     cudaStream_t stream = 0);
#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/NV12toBGRandResize/resize_convert.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `resize_convert_main.cpp`

Source: cpp/5_Domain_Specific/NV12toBGRandResize/resize_convert_main.cpp:31-49
```cpp
NVIDIA HW Decoder, both dGPU and Tegra, normally outputs NV12 pitch format
frames. For the inference using TensorRT, the input frame needs to be BGR planar
format with possibly different size. So, conversion and resizing from NV12 to
BGR planar is usually required for the inference following decoding.
This CUDA code is to provide a reference implementation for conversion and
resizing.

Limitaion
=========
    NV12resize needs the height to be a even value.

Note
====
    Resize function needs the pitch of image buffer to be 32 alignment.

Run
====
./NV12toBGRandResize
   OR
```

> JP: この抜粋は `cpp/5_Domain_Specific/NV12toBGRandResize/resize_convert_main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/NV12toBGRandResize/resize_convert_main.cpp:202-221
```cpp
    frameSize = g_ctx.pitch * g_ctx.ctx_heights;

#if USE_UVM_MEM
    pNV12FrameData = d_inputNV12;
#else
    pNV12FrameData = (unsigned char *)malloc(frameSize);
    if (pNV12FrameData == NULL) {
        std::cerr << "Failed to malloc pNV12FrameData\n";
        return -1;
    }
#endif

    nv12File.read((char *)pNV12FrameData, frameSize);

    if (nv12File.gcount() < frameSize) {
        std::cerr << "can't get one frame!\n";
        return -1;
    }

#if USE_UVM_MEM
```

> JP: この抜粋は `cpp/5_Domain_Specific/NV12toBGRandResize/resize_convert_main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/NV12toBGRandResize/resize_convert_main.cpp:225-258
```cpp
#endif

    // expand one frame to multi frames for batch processing
    d_nv12 = d_inputNV12;
    for (int i = 0; i < g_ctx.batch; i++) {
        // JP: `cudaMemcpy2D`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
        checkCudaErrors(cudaMemcpy2D((void *)d_nv12,
                                     g_ctx.ctx_pitch,
                                     pNV12FrameData,
                                     g_ctx.width,
                                     g_ctx.width,
                                     g_ctx.ctx_heights,
                                     cudaMemcpyHostToDevice));

        d_nv12 += g_ctx.ctx_pitch * g_ctx.ctx_heights;
    }

#if (USE_UVM_MEM == 0)
    // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    free(pNV12FrameData);
#endif
    nv12File.close();

    return 0;
}

/*
  1. resize interlace nv12 to target size
  2. convert nv12 to bgr 3 progressive planars
 */
void nv12ResizeAndNV12ToBGR(unsigned char *d_inputNV12)
{
    unsigned char *d_resizedNV12;
    float         *d_outputBGR;
```

> JP: この抜粋は `cpp/5_Domain_Specific/NV12toBGRandResize/resize_convert_main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/NV12toBGRandResize/resize_convert_main.cpp:290-309
```cpp
                        g_ctx.dst_width,
                        g_ctx.dst_height,
                        g_ctx.batch);
    }
    cudaEventRecord(stop, 0);
    // JP: `cudaEventSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("  CUDA resize nv12(%dx%d --> %dx%d), batch: %d,"
           " average time: %.3f ms ==> %.3f ms/frame\n",
           g_ctx.width,
           g_ctx.height,
           g_ctx.dst_width,
           g_ctx.dst_height,
           g_ctx.batch,
           (elapsedTime / (TEST_LOOP * 1.0f)),
           (elapsedTime / (TEST_LOOP * 1.0f)) / g_ctx.batch);

    sprintf(filename, "resized_nv12_%dx%d", g_ctx.dst_width, g_ctx.dst_height);
```

> JP: この抜粋は `cpp/5_Domain_Specific/NV12toBGRandResize/resize_convert_main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `utils.cu`

Source: cpp/5_Domain_Specific/NV12toBGRandResize/utils.cu:29-56
```cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "resize_convert.h"
#include "utils.h"

__global__ void floatToChar(float *src, unsigned char *dst, int height, int width, int batchSize)
{
    // JP: `threadIdx`, `blockIdx`, `blockDim`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x >= height * width)
        return;

    int offset = height * width * 3;

    for (int j = 0; j < batchSize; j++) {
        // b
        *(dst + j * offset + x * 3 + 0) = (unsigned char)*(src + j * offset + height * width * 0 + x);
        // g
        *(dst + j * offset + x * 3 + 1) = (unsigned char)*(src + j * offset + height * width * 1 + x);
        // r
        *(dst + j * offset + x * 3 + 2) = (unsigned char)*(src + j * offset + height * width * 2 + x);
```

> JP: この抜粋は `cpp/5_Domain_Specific/NV12toBGRandResize/utils.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/NV12toBGRandResize/utils.cu:78-122
```cuda
#endif

    int ret = system(mkdir_cmd);

    frameSize = width * height * 3 * sizeof(float);
    bgr       = (float *)malloc(frameSize);
    if (bgr == NULL) {
        std::cerr << "Failed malloc for bgr\n";
        return;
    }

    d_bgr = d_srcBGR;
    for (int i = 0; i < batchSize; i++) {
        char           filename[256];
        std::ofstream *outputFile;

        // JP: `cudaMemcpy`, `cudaMemcpyDeviceToHost`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
        checkCudaErrors(cudaMemcpy((void *)bgr, (void *)d_bgr, frameSize, cudaMemcpyDeviceToHost));
        snprintf(filename, sizeof(filename), "%s/%s_%d.raw", directory, tag, (i + 1));

        outputFile = new std::ofstream(filename);
        if (outputFile) {
            outputFile->write((char *)bgr, frameSize);
            // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
            delete outputFile;
        }

        d_bgr += pitch * height * 3;
    }

    free(bgr);
}

void dumpBGR(float *d_srcBGR, int pitch, int width, int height, int batchSize, char *folder, char *tag)
{
    dumpRawBGR(d_srcBGR, pitch, width, height, batchSize, folder, tag);
}

void dumpYUV(unsigned char *d_nv12, int size, char *folder, char *tag)
{
    unsigned char *nv12Data;
    std::ofstream *nv12File;
    char           filename[256];
    char           directory[60];
    char           mkdir_cmd[256];
```

> JP: この抜粋は `cpp/5_Domain_Specific/NV12toBGRandResize/utils.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `utils.h`

Source: cpp/5_Domain_Specific/NV12toBGRandResize/utils.h:30-35
```cpp
#ifndef __H_UTIL_
#define __H_UTIL_

extern "C" void dumpBGR(float *d_srcBGR, int pitch, int width, int height, int batchSize, char *folder, char *tag);
extern "C" void dumpYUV(unsigned char *d_nv12, int size, char *folder, char *tag);
#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/NV12toBGRandResize/utils.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaStream_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaEventRecord` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaDestroyTextureObject` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaEventSynchronize` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaTextureObject_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaEventCreate` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaEventElapsedTime` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaEventDestroy` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- CUDA Graph は一連の work を node と依存関係として再利用します。capture 対象と buffer lifetime を確認します。
- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。
- 同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。
- Unified Memory は pointer を共有しますが、migration、prefetch、同期の理解は必要です。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target NV12toBGRandResize
ctest --test-dir build -R NV12toBGRandResize
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

The sample prints timing, bandwidth, latency, throughput, or comparison data; exact values depend on hardware and driver.
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
- graph capture 後に buffer lifetime や node dependency が変わったことを見落とす。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `threadIdx` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- graph node の依存関係を箇条書きにし、どの buffer lifetime が graph 実行全体をまたぐか確認する。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [CUDA Graphs](../../../docs_ja/themes/graphs.md): capture、node dependency、replay、graph update を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Unified Memory](../../../docs_ja/themes/unified_memory.md): managed memory、migration、prefetch の意味を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Performance](../../../docs_ja/themes/performance.md): memory traffic、occupancy、overlap、launch overhead、timing を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
