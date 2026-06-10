# imageDenoising - Image denoising - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates two adaptive image denoising techniques: KNN and NLM, based on computation of both geometric and color distance between texels. While both techniques are implemented in the DirectX SDK using shaders, massively speeded up variation of the latter technique, taking advantage of shared memory, is implemented in addition to DirectX counterparts.

Image Processing

Original README headings: `imageDenoising - Image denoising`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/2_Concepts_and_Techniques/imageDenoising` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `imageDenoising` as a focused example of the CUDA concepts used in `cpp/2_Concepts_and_Techniques/imageDenoising`.
> **日本語**
> この sample の目的は、`imageDenoising` の小さな実装を通して CUDA Graphs, Shared Memory, Streams And Events, Synchronization And Atomics, Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `bmploader.cpp, imageDenoising.cu, imageDenoising.h, imageDenoisingGL.cpp, imageDenoising_copy_kernel.cuh` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `bmploader.cpp`: Host-side setup, API calls, validation, and cleanup.
- `data/portrait_noise.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `data/ref_knn.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/ref_nlm.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/ref_nlm2.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/ref_passthru.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/NLM_lg.png`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/NLM_md.png`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/NLM_sm.png`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/imageDenoising.doc`: Supporting file used by `doc/imageDenoising.doc`.
- `doc/imageDenoising.pdf`: Supporting file used by `doc/imageDenoising.pdf`.
- `imageDenoising.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `imageDenoising.h`: Host/device declarations, helper types, constants, or library wrappers.
- `imageDenoisingGL.cpp`: Host-side setup, API calls, validation, and cleanup.
- `imageDenoising_copy_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `imageDenoising_knn_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `imageDenoising_nlm2_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `imageDenoising_nlm_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `bmploader.cpp` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Capture or build CUDA Graph nodes, instantiate the graph, then launch the executable graph.
- Inside the kernel, map thread/block indexes to tile elements and check the barrier around shared memory reuse.
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

- `bmploader.cpp`: focus on control flow and helper functions.
- `imageDenoising.cu`: focus on `cudaError_t`, `cudaMallocArray`, `cudaMemcpyHostToDevice`, `cudaResourceDesc`, `cudaTextureDesc`.
- `imageDenoising.h`: focus on `cudaTextureObject_t`, `CUDA`, `cudaError_t`, `launch`, `CUDA_MallocArray`.
- `imageDenoisingGL.cpp`: focus on `CUDA`, `cudaGraphicsResource`, `launch`, `cudaGraphicsMapResources`, `cudaGraphicsResourceGetMappedPointer`.
- `imageDenoising_copy_kernel.cuh`: focus on `launch`, `cudaTextureObject_t`, `blockDim`, `blockIdx`, `threadIdx`.
- `imageDenoising_knn_kernel.cuh`: focus on `blockDim`, `blockIdx`, `threadIdx`, `cudaTextureObject_t`, `launch`.
- `imageDenoising_nlm2_kernel.cuh`: focus on `threadIdx`, `blockDim`, `blockIdx`, `cudaTextureObject_t`, `launch`.
- `imageDenoising_nlm_kernel.cuh`: focus on `blockDim`, `blockIdx`, `threadIdx`, `cudaTextureObject_t`, `launch`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/2_Concepts_and_Techniques/imageDenoising/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(imageDenoising LANGUAGES C CXX CUDA)

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
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/imageDenoising/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bmploader.cpp`

Source: cpp/2_Concepts_and_Techniques/imageDenoising/bmploader.cpp:29-47
```cpp
#include <stdio.h>
#include <stdlib.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#pragma warning(disable : 4996) // disable deprecated warning
#endif

#pragma pack(1)

typedef struct
{
    short type;
    int   size;
    short reserved1;
    short reserved2;
    int   offset;
} BMPHeader;

typedef struct
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/imageDenoising/bmploader.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `imageDenoising.cu`

Source: cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising.cu:39-57
```cuda
#include <helper_cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imageDenoising.h"

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
float Max(float x, float y) { return (x > y) ? x : y; }

float Min(float x, float y) { return (x < y) ? x : y; }

int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

__device__ float lerpf(float a, float b, float c) { return a + (b - a) * c; }

__device__ float vecLen(float4 a, float4 b)
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising.cu:84-105
```cuda

extern "C" cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH)
{
    cudaError_t error;

    // JP: `cudaMallocArray`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    error = cudaMallocArray(&a_Src, &uchar4tex, imageW, imageH);
    error = cudaMemcpy2DToArray(
        // JP: `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
        a_Src, 0, 0, *h_Src, sizeof(uchar4) * imageW, sizeof(uchar4) * imageW, imageH, cudaMemcpyHostToDevice);

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType         = cudaResourceTypeArray;
    texRes.res.array.array = a_Src;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModeLinear;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising.cu:110-116
```cuda
    checkCudaErrors(cudaCreateTextureObject(&texImage, &texRes, &texDescr, NULL));

    return error;
}

// JP: `cudaError_t`, `cudaFreeArray`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
extern "C" cudaError_t CUDA_FreeArray() { return cudaFreeArray(a_Src); }
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `imageDenoising.h`

Source: cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising.h:29-83
```cpp
#ifndef IMAGE_DENOISING_H
#define IMAGE_DENOISING_H

typedef unsigned int TColor;

////////////////////////////////////////////////////////////////////////////////
// Filter configuration
////////////////////////////////////////////////////////////////////////////////
#define KNN_WINDOW_RADIUS   3
#define NLM_WINDOW_RADIUS   3
#define NLM_BLOCK_RADIUS    3
#define KNN_WINDOW_AREA     ((2 * KNN_WINDOW_RADIUS + 1) * (2 * KNN_WINDOW_RADIUS + 1))
#define NLM_WINDOW_AREA     ((2 * NLM_WINDOW_RADIUS + 1) * (2 * NLM_WINDOW_RADIUS + 1))
#define INV_KNN_WINDOW_AREA (1.0f / (float)KNN_WINDOW_AREA)
#define INV_NLM_WINDOW_AREA (1.0f / (float)NLM_WINDOW_AREA)

#define KNN_WEIGHT_THRESHOLD 0.02f
#define KNN_LERP_THRESHOLD   0.79f
#define NLM_WEIGHT_THRESHOLD 0.10f
#define NLM_LERP_THRESHOLD   0.10f

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

#ifndef MAX
#define MAX(a, b) ((a < b) ? b : a)
#endif
#ifndef MIN
#define MIN(a, b) ((a < b) ? a : b)
#endif

// functions to load images
extern "C" void LoadBMPFile(uchar4 **dst, int *width, int *height, const char *name);

// CUDA wrapper functions for allocation/freeing texture arrays
extern "C" cudaTextureObject_t texImage;

extern "C" cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH);
extern "C" cudaError_t CUDA_FreeArray();

// CUDA kernel functions
extern "C" void cuda_Copy(TColor *d_dst, int imageW, int imageH, cudaTextureObject_t texImage);
extern "C" void cuda_KNN(TColor *d_dst, int imageW, int imageH, float Noise, float lerpC, cudaTextureObject_t texImage);
extern "C" void
cuda_KNNdiag(TColor *d_dst, int imageW, int imageH, float Noise, float lerpC, cudaTextureObject_t texImage);
extern "C" void cuda_NLM(TColor *d_dst, int imageW, int imageH, float Noise, float lerpC, cudaTextureObject_t texImage);
extern "C" void
cuda_NLMdiag(TColor *d_dst, int imageW, int imageH, float Noise, float lerpC, cudaTextureObject_t texImage);

extern "C" void
cuda_NLM2(TColor *d_dst, int imageW, int imageH, float Noise, float LerpC, cudaTextureObject_t texImage);
extern "C" void
cuda_NLM2diag(TColor *d_dst, int imageW, int imageH, float Noise, float LerpC, cudaTextureObject_t texImage);

#endif
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `imageDenoisingGL.cpp`

Source: cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoisingGL.cpp:40-58
```cpp
#include <helper_gl.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imageDenoising.h"

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoisingGL.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoisingGL.cpp:184-203
```cpp
    if (frameCounter++ == 0) {
        sdkResetTimer(&timer);
    }

    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    getLastCudaError("cudaGraphicsMapResources failed");
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_dst, &num_bytes, cuda_pbo_resource));
    getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

    runImageFilters(d_dst);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    // Common display code path
    {
        glClear(GL_COLOR_BUFFER_BIT);

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));
        glBegin(GL_TRIANGLES);
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoisingGL.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoisingGL.cpp:415-434
```cpp
}

void cleanup()
{
    // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    free(h_Src);
    checkCudaErrors(CUDA_FreeArray());
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

    glDeleteProgramsARB(1, &shader);

    sdkDeleteTimer(&timer);
}

void runAutoTest(int argc, char **argv, const char *filename, int kernel_param)
{
    printf("[%s] - (automated testing w/ readback)\n", sSDKsample);

    int devID = findCudaDevice(argc, (const char **)argv);

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoisingGL.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoisingGL.cpp:448-497
```cpp

    checkCudaErrors(CUDA_MallocArray(&h_Src, imageW, imageH));

    TColor        *d_dst = NULL;
    unsigned char *h_dst = NULL;
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc((void **)&d_dst, imageW * imageH * sizeof(TColor)));
    h_dst = (unsigned char *)malloc(imageH * imageW * 4);

    {
        g_Kernel = kernel_param;
        printf("[AutoTest]: %s <%s>\n", sSDKsample, filterMode[g_Kernel]);

        runImageFilters(d_dst);

        // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
        checkCudaErrors(cudaDeviceSynchronize());

        // JP: `cudaMemcpy`, `cudaMemcpyDeviceToHost`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
        checkCudaErrors(cudaMemcpy(h_dst, d_dst, imageW * imageH * sizeof(TColor), cudaMemcpyDeviceToHost));
        sdkSavePPM4ub(filename, h_dst, imageW, imageH);
    }

    checkCudaErrors(CUDA_FreeArray());
    free(h_Src);

    // JP: この anchor では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaFree(d_dst));
    free(h_dst);

    printf("\n[%s] -> Kernel %d, Saved: %s\n", sSDKsample, kernel_param, filename);

    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    char *dump_file = NULL;

#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    pArgc = &argc;
    pArgv = argv;

    printf("%s Starting...\n\n", sSDKsample);

    if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
        getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&dump_file);
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoisingGL.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `imageDenoising_copy_kernel.cuh`

Source: cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_copy_kernel.cuh:29-51
```cuda
__global__ void Copy(TColor *dst, int imageW, int imageH, cudaTextureObject_t texImage)
{
    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if (ix < imageW && iy < imageH) {
        float4 fresult        = tex2D<float4>(texImage, x, y);
        dst[imageW * iy + ix] = make_color(fresult.x, fresult.y, fresult.z, 0);
    }
}

extern "C" void cuda_Copy(TColor *d_dst, int imageW, int imageH, cudaTextureObject_t texImage)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    Copy<<<grid, threads>>>(d_dst, imageW, imageH, texImage);
}
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_copy_kernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `imageDenoising_knn_kernel.cuh`

Source: cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_knn_kernel.cuh:29-50
```cuda
////////////////////////////////////////////////////////////////////////////////
// KNN kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void KNN(TColor *dst, int imageW, int imageH, float Noise, float lerpC, cudaTextureObject_t texImage)
{
    // JP: `blockDim`, `blockIdx`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if (ix < imageW && iy < imageH) {
        // Normalized counter for the weight threshold
        float fCount = 0;
        // Total sum of pixel weights
        float sumWeights = 0;
        // Result accumulator
        float3 clr = {0, 0, 0};
        // Center of the KNN window
        float4 clr00 = tex2D<float4>(texImage, x, y);

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_knn_kernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `imageDenoising_nlm2_kernel.cuh`

Source: cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_nlm2_kernel.cuh:48-73
```cuda
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void NLM2(TColor *dst, int imageW, int imageH, float Noise, float lerpC, cudaTextureObject_t texImage)
{
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block cta = cg::this_thread_block();

    // Weights cache
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    __shared__ float fWeights[BLOCKDIM_X * BLOCKDIM_Y];

    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // Add half of a texel to always address exact texel centers
    const float x  = (float)ix + 0.5f;
    const float y  = (float)iy + 0.5f;
    const float cx = blockDim.x * blockIdx.x + NLM_WINDOW_RADIUS + 0.5f;
    const float cy = blockDim.x * blockIdx.y + NLM_WINDOW_RADIUS + 0.5f;

    if (ix < imageW && iy < imageH) {
        // Find color distance from current texel to the center of NLM window
        float weight = 0;

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_nlm2_kernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `imageDenoising_nlm_kernel.cuh`

Source: cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_nlm_kernel.cuh:29-50
```cuda
////////////////////////////////////////////////////////////////////////////////
// NLM kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void NLM(TColor *dst, int imageW, int imageH, float Noise, float lerpC, cudaTextureObject_t texImage)
{
    // JP: `blockDim`, `blockIdx`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if (ix < imageW && iy < imageH) {
        // Normalized counter for the NLM weight threshold
        float fCount = 0;
        // Total sum of pixel weights
        float sumWeights = 0;
        // Result accumulator
        float3 clr = {0, 0, 0};

        // Cycle through NLM window, surrounding (x, y) texel
        for (float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS; i++)
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_nlm_kernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaTextureObject_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaError_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CUDA_MallocArray` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaMallocArray` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFreeArray` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `CUDA_FreeArray` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaGraphicsMapResources` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphicsResourceGetMappedPointer` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- CUDA Graph は一連の work を node と依存関係として再利用します。capture 対象と buffer lifetime を確認します。
- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。
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
cmake --build build --target imageDenoising
ctest --test-dir build -R imageDenoising
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

The sample may display a window or produce/validate image-like output; exact visuals depend on platform support.
> **日本語**
> 期待結果は英語の出力文字列、README の validation、生成 file、または reference result と照合します。
>
> **学習メモ**
> `PASS`、`Test passed`、error code、timing label などの出力文字列は翻訳せず、source と同じ表記で確認します。

## Common Mistakes

- API 名や target 名を翻訳してしまい、README や build command と対応できなくなる。
- allocation size を byte で渡す API と element count で考える loop を混同する。
- kernel launch が非同期であることを忘れ、同期前の結果を host 側で読んでしまう。
- shared memory を書いた thread と読む thread の間に必要な barrier を見落とす。
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
- shared memory tile の producer、consumer、barrier を図にする。
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
- [Shared Memory](../../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
