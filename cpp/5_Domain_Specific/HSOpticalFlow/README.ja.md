# HSOpticalFlow - Optical Flow - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

Variational optical flow estimation example.  Uses textures for image operations. Shows how simple PDE solver can be accelerated with CUDA.

Image Processing, Data Parallel Algorithms

Original README headings: `HSOpticalFlow - Optical Flow`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/HSOpticalFlow` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `HSOpticalFlow` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/HSOpticalFlow`.
> **日本語**
> この sample の目的は、`HSOpticalFlow` の小さな実装を通して Shared Memory, Memory, Kernel Launch And Indexing, Execution Model, Debugging, Profiling, And Testing を具体的に追うことです。
>
> **学習メモ**
> 最初に `addKernel.cuh, common.h, derivativesKernel.cuh, downscaleKernel.cuh, flowCUDA.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `FlowCPU.flo`: Supporting file used by `FlowCPU.flo`.
- `FlowGPU.flo`: Supporting file used by `FlowGPU.flo`.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `addKernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `data/frame10.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/frame11.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `derivativesKernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `doc/OpticalFlow.docx`: Supporting file used by `doc/OpticalFlow.docx`.
- `doc/OpticalFlow.pdf`: Supporting file used by `doc/OpticalFlow.pdf`.
- `downscaleKernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `flowCUDA.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `flowCUDA.h`: Host/device declarations, helper types, constants, or library wrappers.
- `flowGold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `flowGold.h`: Host/device declarations, helper types, constants, or library wrappers.
- `main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `solverKernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `upscaleKernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `warpingKernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `addKernel.cuh` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
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

- `addKernel.cuh`: focus on `launch`, `threadIdx`, `blockIdx`, `blockDim`, `CUDA`.
- `common.h`: focus on control flow and helper functions.
- `derivativesKernel.cuh`: focus on `cudaTextureObject_t`, `threadIdx`, `blockIdx`, `blockDim`, `cudaResourceDesc`.
- `downscaleKernel.cuh`: focus on `threadIdx`, `blockIdx`, `blockDim`, `launch`, `cudaTextureObject_t`.
- `flowCUDA.cu`: focus on `cudaMalloc`, `cudaFree`, `cudaMemset`, `cudaMemcpy`, `cudaMemcpyHostToDevice`.
- `flowCUDA.h`: focus on control flow and helper functions.
- `flowGold.cpp`: focus on control flow and helper functions.
- `flowGold.h`: focus on control flow and helper functions.
- `main.cpp`: focus on control flow and helper functions.
- `solverKernel.cuh`: focus on `threadIdx`, `blockIdx`, `blockDim`, `__shared__`, `launch`.
- Additional source files: 2 more support files. Use the same setup/work/sync/cleanup lens.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/HSOpticalFlow/CMakeLists.txt:1-47
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(HSOpticalFlow LANGUAGES C CXX CUDA)

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
# Add target for HSOpticalFlow
add_executable(HSOpticalFlow flowCUDA.cu flowGold.cpp main.cpp)

target_compile_options(HSOpticalFlow PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(HSOpticalFlow PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(HSOpticalFlow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(HSOpticalFlow PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

add_custom_command(TARGET HSOpticalFlow POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    ${CMAKE_CURRENT_BINARY_DIR}/data
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `addKernel.cuh`

Source: cpp/5_Domain_Specific/HSOpticalFlow/addKernel.cuh:29-65
```cuda
#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief add two vectors of size _count_
///
/// CUDA kernel
/// \param[in]  op1   term one
/// \param[in]  op2   term two
/// \param[in]  count vector size
/// \param[out] sum   result
///////////////////////////////////////////////////////////////////////////////
__global__ void AddKernel(const float *op1, const float *op2, int count, float *sum)
{
    // JP: `threadIdx`, `blockIdx`, `blockDim`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    const int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos >= count)
        return;

    sum[pos] = op1[pos] + op2[pos];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief add two vectors of size _count_
/// \param[in]  op1   term one
/// \param[in]  op2   term two
/// \param[in]  count vector size
/// \param[out] sum   result
///////////////////////////////////////////////////////////////////////////////
static void Add(const float *op1, const float *op2, int count, float *sum)
{
    dim3 threads(256);
    dim3 blocks(iDivUp(count, threads.x));

    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    AddKernel<<<blocks, threads>>>(op1, op2, count, sum);
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/addKernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `common.h`

Source: cpp/5_Domain_Specific/HSOpticalFlow/common.h:33-77
```cpp
#ifndef COMMON_H
#define COMMON_H

///////////////////////////////////////////////////////////////////////////////
// Common includes
///////////////////////////////////////////////////////////////////////////////

#include <helper_cuda.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

///////////////////////////////////////////////////////////////////////////////
// Common constants
///////////////////////////////////////////////////////////////////////////////
const int StrideAlignment = 32;

///////////////////////////////////////////////////////////////////////////////
// Common functions
///////////////////////////////////////////////////////////////////////////////

// Align up n to the nearest multiple of m
inline int iAlignUp(int n, int m = StrideAlignment)
{
    int mod = n % m;

    if (mod)
        return n + m - mod;
    else
        return n;
}

// round up n/m
inline int iDivUp(int n, int m) { return (n + m - 1) / m; }

// swap two values
template <typename T> inline void Swap(T &a, T &b)
{
    T t = a;
    a   = b;
    b   = t;
}
#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/common.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `derivativesKernel.cuh`

Source: cpp/5_Domain_Specific/HSOpticalFlow/derivativesKernel.cuh:29-65
```cuda
#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] Ix      x derivative
/// \param[out] Iy      y derivative
/// \param[out] Iz      temporal derivative
///////////////////////////////////////////////////////////////////////////////
__global__ void ComputeDerivativesKernel(int                 width,
                                         int                 height,
                                         int                 stride,
                                         float              *Ix,
                                         float              *Iy,
                                         float              *Iz,
                                         cudaTextureObject_t texSource,
                                         cudaTextureObject_t texTarget)
{
    // JP: `threadIdx`, `blockIdx`, `blockDim`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * stride;

    if (ix >= width || iy >= height)
        return;

    float dx = 1.0f / (float)width;
    float dy = 1.0f / (float)height;

    float x = ((float)ix + 0.5f) * dx;
    float y = ((float)iy + 0.5f) * dy;

```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/derivativesKernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `downscaleKernel.cuh`

Source: cpp/5_Domain_Specific/HSOpticalFlow/downscaleKernel.cuh:29-56
```cuda
#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief downscale image
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
__global__ void DownscaleKernel(int width, int height, int stride, float *out, cudaTextureObject_t texFine)
{
    // JP: `threadIdx`, `blockIdx`, `blockDim`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height) {
        return;
    }

    float dx = 1.0f / (float)width;
    float dy = 1.0f / (float)height;

    float x = ((float)ix + 0.5f) * dx;
    float y = ((float)iy + 0.5f) * dy;

    out[ix + iy * stride] = 0.25f
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/downscaleKernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `flowCUDA.cu`

Source: cpp/5_Domain_Specific/HSOpticalFlow/flowCUDA.cu:29-47
```cuda
#include "common.h"

// include kernels
#include "addKernel.cuh"
#include "derivativesKernel.cuh"
#include "downscaleKernel.cuh"
#include "solverKernel.cuh"
#include "upscaleKernel.cuh"
#include "warpingKernel.cuh"

///////////////////////////////////////////////////////////////////////////////
/// \brief method logic
///
/// handles memory allocations, control flow
/// \param[in]  I0           source image
/// \param[in]  I1           tracked image
/// \param[in]  width        images width
/// \param[in]  height       images height
/// \param[in]  stride       images stride
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/flowCUDA.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/HSOpticalFlow/flowCUDA.cu:65-84
```cuda
                     float       *v)
{
    printf("Computing optical flow on GPU...\n");

    // pI0 and pI1 will hold device pointers
    const float **pI0 = new const float *[nLevels];
    const float **pI1 = new const float *[nLevels];

    int *pW = new int[nLevels];
    int *pH = new int[nLevels];
    int *pS = new int[nLevels];

    // device memory pointers
    float *d_tmp;
    float *d_du0;
    float *d_dv0;
    float *d_du1;
    float *d_dv1;

    float *d_Ix;
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/flowCUDA.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/HSOpticalFlow/flowCUDA.cu:113-132
```cuda
    int currentLevel = nLevels - 1;
    // allocate GPU memory for input images
    checkCudaErrors(cudaMalloc(pI0 + currentLevel, dataSize));
    checkCudaErrors(cudaMalloc(pI1 + currentLevel, dataSize));

    // JP: `cudaMemcpy`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpy((void *)pI0[currentLevel], I0, dataSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void *)pI1[currentLevel], I1, dataSize, cudaMemcpyHostToDevice));

    pW[currentLevel] = width;
    pH[currentLevel] = height;
    pS[currentLevel] = stride;

    for (; currentLevel > 0; --currentLevel) {
        int nw = pW[currentLevel] / 2;
        int nh = pH[currentLevel] / 2;
        int ns = iAlignUp(nw);

        // JP: この連続する anchor 群では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
        checkCudaErrors(cudaMalloc(pI0 + currentLevel - 1, ns * nh * sizeof(float)));
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/flowCUDA.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/HSOpticalFlow/flowCUDA.cu:232-251
```cuda
    checkCudaErrors(cudaMemcpy(u, d_u, dataSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(v, d_v, dataSize, cudaMemcpyDeviceToHost));

    // cleanup
    for (int i = 0; i < nLevels; ++i) {
        // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        checkCudaErrors(cudaFree((void *)pI0[i]));
        checkCudaErrors(cudaFree((void *)pI1[i]));
    }

    delete[] pI0;
    delete[] pI1;
    delete[] pW;
    delete[] pH;
    delete[] pS;

    // JP: この連続する anchor 群では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaFree(d_tmp));
    checkCudaErrors(cudaFree(d_du0));
    checkCudaErrors(cudaFree(d_dv0));
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/flowCUDA.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `flowCUDA.h`

Source: cpp/5_Domain_Specific/HSOpticalFlow/flowCUDA.h:29-43
```cpp
#ifndef FLOW_CUDA_H
#define FLOW_CUDA_H

void ComputeFlowCUDA(const float *I0,           // source frame
                     const float *I1,           // tracked frame
                     int          width,        // frame width
                     int          height,       // frame height
                     int          stride,       // row access stride
                     float        alpha,        // smoothness coefficient
                     int          nLevels,      // number of levels in pyramid
                     int          nWarpIters,   // number of warping iterations per pyramid level
                     int          nSolverIters, // number of solver iterations (for linear system)
                     float       *u,            // output horizontal flow
                     float       *v);                 // output vertical flow
#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/flowCUDA.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `flowGold.cpp`

Source: cpp/5_Domain_Specific/HSOpticalFlow/flowGold.cpp:29-47
```cpp
#include "flowGold.h"

#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief host texture fetch
///
/// read from arbitrary position within image using bilinear interpolation
/// out of range coords are mirrored
/// \param[in]  t   texture raw data
/// \param[in]  w   texture width
/// \param[in]  h   texture height
/// \param[in]  s   texture stride
/// \param[in]  x   x coord of the point to fetch value at
/// \param[in]  y   y coord of the point to fetch value at
/// \return fetched value
///////////////////////////////////////////////////////////////////////////////
inline float Tex2D(const float *t, int w, int h, int s, float x, float y)
{
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/flowGold.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/HSOpticalFlow/flowGold.cpp:122-141
```cpp
/// \brief resize image
/// \param[in]  src         image to downscale
/// \param[in]  width       image width
/// \param[in]  height      image height
/// \param[in]  stride      image stride
/// \param[in]  newWidth    image new width
/// \param[in]  newHeight   image new height
/// \param[in]  newStride   image new stride
/// \param[out] out         downscaled image data
///////////////////////////////////////////////////////////////////////////////
static void
Downscale(const float *src, int width, int height, int stride, int newWidth, int newHeight, int newStride, float *out)
{
    for (int i = 0; i < newHeight; ++i) {
        for (int j = 0; j < newWidth; ++j) {
            const int srcX = j * 2;
            const int srcY = i * 2;
            // average 4 neighbouring pixels
            float sum;
            sum = Tex2Di(src, width, height, stride, srcX + 0, srcY + 0);
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/flowGold.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `flowGold.h`

Source: cpp/5_Domain_Specific/HSOpticalFlow/flowGold.h:29-44
```cpp
#ifndef FLOW_GOLD_H
#define FLOW_GOLD_H

void ComputeFlowGold(const float *I0,         // source frame
                     const float *I1,         // tracked frame
                     int          width,      // frame width
                     int          height,     // frame height
                     int          stride,     // row access stride
                     float        alpha,      // smoothness coefficient
                     int          nLevels,    // number of levels in pyramid
                     int          nWarpIters, // number of warping iterations per pyramid level
                     int          nIters,     // number of solver iterations (for linear system)
                     float       *u,          // output horizontal flow
                     float       *v);               // output vertical flow

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/flowGold.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `main.cpp`

Source: cpp/5_Domain_Specific/HSOpticalFlow/main.cpp:29-47
```cpp
const static char *const sSDKsample = "HSOpticalFlow";

// CPU-GPU discrepancy threshold for self-test
const float THRESHOLD = 0.05f;

#include <cuda_runtime.h>
#include <helper_functions.h>

#include "common.h"
#include "flowCUDA.h"
#include "flowGold.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief save optical flow in format described on vision.middlebury.edu/flow
/// \param[in] name output file name
/// \param[in] w    optical flow field width
/// \param[in] h    optical flow field height
/// \param[in] s    optical flow field row stride
/// \param[in] u    horizontal displacement
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/HSOpticalFlow/main.cpp:106-140
```cpp

    img_w = w;
    img_h = h;
    img_s = iAlignUp(img_w);

    img_data = new float[img_s * h];

    // source is 4 channel image
    const int widthStep = 4 * img_w;

    for (int i = 0; i < img_h; ++i) {
        for (int j = 0; j < img_w; ++j) {
            img_data[j + i * img_s] = ((float)data[j * 4 + i * widthStep]) / 255.0f;
        }
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief compare given flow field with gold (L1 norm)
/// \param[in] width    optical flow field width
/// \param[in] height   optical flow field height
/// \param[in] stride   optical flow field row stride
/// \param[in] h_uGold  horizontal displacement, gold
/// \param[in] h_vGold  vertical displacement, gold
/// \param[in] h_u      horizontal displacement
/// \param[in] h_v      vertical displacement
/// \return true if discrepancy is lower than a given threshold
///////////////////////////////////////////////////////////////////////////////
bool CompareWithGold(int          width,
                     int          height,
                     int          stride,
                     const float *h_uGold,
                     const float *h_vGold,
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/HSOpticalFlow/main.cpp:158-177
```cpp
}

///////////////////////////////////////////////////////////////////////////////
/// application entry point
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // welcome message
    printf("%s Starting...\n\n", sSDKsample);

    // pick GPU
    findCudaDevice(argc, (const char **)argv);

    // find images
    const char *const sourceFrameName = "frame10.ppm";
    const char *const targetFrameName = "frame11.ppm";

    // image dimensions
    int width;
    int height;
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `solverKernel.cuh`

Source: cpp/5_Domain_Specific/HSOpticalFlow/solverKernel.cuh:29-83
```cuda
#include <cooperative_groups.h>

#include "common.h"

namespace cg = cooperative_groups;

///////////////////////////////////////////////////////////////////////////////
/// \brief one iteration of classical Horn-Schunck method, CUDA kernel.
///
/// It is one iteration of Jacobi method for a corresponding linear system.
/// Template parameters are describe CTA size
/// \param[in]  du0     current horizontal displacement approximation
/// \param[in]  dv0     current vertical displacement approximation
/// \param[in]  Ix      image x derivative
/// \param[in]  Iy      image y derivative
/// \param[in]  Iz      temporal derivative
/// \param[in]  w       width
/// \param[in]  h       height
/// \param[in]  s       stride
/// \param[in]  alpha   degree of smoothness
/// \param[out] du1     new horizontal displacement approximation
/// \param[out] dv1     new vertical displacement approximation
///////////////////////////////////////////////////////////////////////////////
template <int bx, int by>
__global__ void JacobiIteration(const float *du0,
                                const float *dv0,
                                const float *Ix,
                                const float *Iy,
                                const float *Iz,
                                int          w,
                                int          h,
                                int          s,
                                float        alpha,
                                float       *du1,
                                float       *dv1)
{
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block cta = cg::this_thread_block();

    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    volatile __shared__ float du[(bx + 2) * (by + 2)];
    volatile __shared__ float dv[(bx + 2) * (by + 2)];

    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // position within global memory array
    const int pos = min(ix, w - 1) + min(iy, h - 1) * s;

    // position within shared memory array
    const int shMemPos = threadIdx.x + 1 + (threadIdx.y + 1) * (bx + 2);

    // Load data to shared memory.
    // load tile being processed
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/solverKernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `upscaleKernel.cuh`

Source: cpp/5_Domain_Specific/HSOpticalFlow/upscaleKernel.cuh:29-76
```cuda
#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief upscale one component of a displacement field, CUDA kernel
/// \param[in]  width   field width
/// \param[in]  height  field height
/// \param[in]  stride  field stride
/// \param[in]  scale   scale factor (multiplier)
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
__global__ void UpscaleKernel(int width, int height, int stride, float scale, float *out, cudaTextureObject_t texCoarse)
{
    // JP: `threadIdx`, `blockIdx`, `blockDim`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height)
        return;

    float x = ((float)ix + 0.5f) / (float)width;
    float y = ((float)iy + 0.5f) / (float)height;

    // exploit hardware interpolation
    // and scale interpolated vector to match next pyramid level resolution
    out[ix + iy * stride] = tex2D<float>(texCoarse, x, y) * scale;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief upscale one component of a displacement field, kernel wrapper
/// \param[in]  src         field component to upscale
/// \param[in]  width       field current width
/// \param[in]  height      field current height
/// \param[in]  stride      field current stride
/// \param[in]  newWidth    field new width
/// \param[in]  newHeight   field new height
/// \param[in]  newStride   field new stride
/// \param[in]  scale       value scale factor (multiplier)
/// \param[out] out         upscaled field component
///////////////////////////////////////////////////////////////////////////////
static void Upscale(const float *src,
                    int          width,
                    int          height,
                    int          stride,
                    int          newWidth,
                    int          newHeight,
                    int          newStride,
                    float        scale,
                    float       *out)
```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/upscaleKernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `warpingKernel.cuh`

Source: cpp/5_Domain_Specific/HSOpticalFlow/warpingKernel.cuh:29-62
```cuda
#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief warp image with a given displacement field, CUDA kernel.
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[in]  u       horizontal displacement
/// \param[in]  v       vertical displacement
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
__global__ void WarpingKernel(int                 width,
                              int                 height,
                              int                 stride,
                              const float        *u,
                              const float        *v,
                              float              *out,
                              cudaTextureObject_t texToWarp)
{
    // JP: `threadIdx`, `blockIdx`, `blockDim`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * stride;

    if (ix >= width || iy >= height)
        return;

    float x = ((float)ix + u[pos] + 0.5f) / (float)width;
    float y = ((float)iy + v[pos] + 0.5f) / (float)height;

    out[pos] = tex2D<float>(texToWarp, x, y);
}

```

> JP: この抜粋は `cpp/5_Domain_Specific/HSOpticalFlow/warpingKernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaTextureObject_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaResourceDesc` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaTextureDesc` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaAddressModeMirror` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMemset` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaResourceTypePitch2D` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaCreateChannelDesc` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target HSOpticalFlow
ctest --test-dir build -R HSOpticalFlow
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
- shared memory を書いた thread と読む thread の間に必要な barrier を見落とす。

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
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Shared Memory](../../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
