# cudaNvSci - CUDA NvSciBuf/NvSciSync Interop - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates CUDA-NvSciBuf/NvSciSync Interop. Two CPU threads import the NvSciBuf and NvSciSync into CUDA to perform two image processing algorithms on a ppm image - image rotation in 1st thread &amp; rgba to grayscale conversion of rotated image in 2nd thread. Currently only supported on Ubuntu 18.04

CUDA NvSci Interop, Data Parallel Algorithms, Image Processing

Original README headings: `cudaNvSci - CUDA NvSciBuf/NvSciSync Interop`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/4_CUDA_Libraries/cudaNvSci` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `cudaNvSci` as a focused example of the CUDA concepts used in `cpp/4_CUDA_Libraries/cudaNvSci`.
> **日本語**
> この sample の目的は、`cudaNvSci` の小さな実装を通して Runtime, Driver, And NVRTC, Streams And Events, Synchronization And Atomics, Performance, Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `cudaNvSci.cpp, cudaNvSci.h, imageKernels.cu, main.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `cudaNvSci.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaNvSci.h`: Host/device declarations, helper types, constants, or library wrappers.
- `imageKernels.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `teapot1024.ppm`: Input, reference, generated-data description, or documentation used by the sample.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `cudaNvSci.cpp` first and locate the host-side setup or Python entry point.
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

- `cudaNvSci.cpp`: focus on `cudaSetDevice`, `cudaNvSci`, `cudaNvSciSignal`, `cudaNvSciWait`, `cuDeviceGetUuid_v2`.
- `cudaNvSci.h`: focus on `CUDANVSCI_H`, `cudaStream_t`, `cudaNvSci`, `atomic`, `cudaTextureObject_t`.
- `imageKernels.cu`: focus on `blockDim`, `blockIdx`, `threadIdx`, `launch`, `cudaStream_t`.
- `main.cpp`: focus on `cudaNvSci`, `cudaDeviceGetAttribute`, `cudaNvSciApp`, `cudaGetDeviceCount`, `cudaDevAttrComputeCapabilityMajor`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/4_CUDA_Libraries/cudaNvSci/CMakeLists.txt:1-70
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(cudaNvSci LANGUAGES C CXX CUDA)

# JP: `find_package`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 90 110)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

find_package(NVSCI)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(NVSCI_FOUND)
        message(STATUS "FOUND NVSCI libs: ${NVSCIBUF_LIB} ${NVSCISYNC_LIB}")
        message(STATUS "Using NVSCI headers path: ${NVSCIBUF_INCLUDE_DIR} ${NVSCIBUF_INCLUDE_DIR}")
        # Source file
        # Add target for cudaNvSci
        add_executable(cudaNvSci imageKernels.cu cudaNvSci.cpp main.cpp)

        target_compile_options(cudaNvSci PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(cudaNvSci PRIVATE cxx_std_17 cuda_std_17)

        set_target_properties(cudaNvSci PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(cudaNvSci PUBLIC
            ${CUDAToolkit_INCLUDE_DIRS}
            ${NVSCI_INCLUDE_DIRS}
        )

        target_link_libraries(cudaNvSci
            CUDA::cuda_driver
            ${NVSCI_LIBRARIES}
        )
        # Copy teapot1024.ppm to the output directory
        add_custom_command(TARGET cudaNvSci POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/teapot1024.ppm ${CMAKE_CURRENT_BINARY_DIR}/teapot1024.ppm
        )

        # Specify additional clean files
        set_target_properties(cudaNvSci PROPERTIES
            ADDITIONAL_CLEAN_FILES "teapot1024_out.ppm"
        )
    else()
        message(STATUS "NvSCI not found - will not build sample 'cudaNvSci'")
    endif()
else()
    message(STATUS "Will not build sample cudaNvSci - requires Linux OS")
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cudaNvSci/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cudaNvSci.cpp`

Source: cpp/4_CUDA_Libraries/cudaNvSci/cudaNvSci.cpp:29-47
```cpp
#include "cudaNvSci.h"

#include <condition_variable>
#include <cuda.h>
#include <iostream>
#include <thread>

std::mutex              m_mutex;
std::condition_variable m_condVar;
bool                    workSubmitted = false;

class cudaNvSciSignal
{
private:
    NvSciSyncModule m_syncModule;
    NvSciBufModule  m_bufModule;

    NvSciSyncAttrList m_syncAttrList;
    NvSciSyncFence   *m_fence;
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cudaNvSci/cudaNvSci.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/cudaNvSci/cudaNvSci.cpp:95-115
```cpp
        checkCudaErrors(cudaDeviceGetNvSciSyncAttributes(m_syncAttrList, m_cudaDeviceId, cudaNvSciSyncAttrSignal));
    }

    ~cudaNvSciSignal()
    {
        checkCudaErrors(cudaSetDevice(m_cudaDeviceId));
        // JP: `cudaFreeMipmappedArray`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        checkCudaErrors(cudaFreeMipmappedArray(d_mipmapArray));
        // JP: `cudaFree`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
        checkCudaErrors(cudaFree(d_outputBuf));
        checkCudaErrors(cudaDestroyExternalSemaphore(signalSem));
        checkCudaErrors(cudaDestroyExternalMemory(extMemRawBuf));
        checkCudaErrors(cudaDestroyExternalMemory(extMemImageBuf));
        // JP: この連続する anchor 群では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
        checkCudaErrors(cudaDestroyTextureObject(texObject));
        checkCudaErrors(cudaStreamDestroy(streamToRun));
    }

    void initCuda()
    {
        checkCudaErrors(cudaSetDevice(m_cudaDeviceId));
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cudaNvSci/cudaNvSci.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/cudaNvSci/cudaNvSci.cpp:321-340
```cpp
    void copyDataToImageArray(unsigned char *imageData)
    {
        uint32_t mipLevelId = 0;
        checkCudaErrors(cudaGetMipmappedArrayLevel(&d_mipLevelArray, d_mipmapArray, mipLevelId));

        checkCudaErrors(cudaMemcpy2DToArrayAsync(d_mipLevelArray,
                                                 0,
                                                 0,
                                                 imageData,
                                                 m_imageWidth * sizeof(unsigned int),
                                                 m_imageWidth * sizeof(unsigned int),
                                                 m_imageHeight,
                                                 // JP: `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
                                                 cudaMemcpyHostToDevice,
                                                 streamToRun));
    }

    void createTexture()
    {
        cudaResourceDesc texRes;
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cudaNvSci/cudaNvSci.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cudaNvSci.h`

Source: cpp/4_CUDA_Libraries/cudaNvSci/cudaNvSci.h:29-47
```cpp
#ifndef CUDANVSCI_H
#define CUDANVSCI_H

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <nvscibuf.h>
#include <nvscisync.h>
#include <vector>

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

> JP: この抜粋は `cpp/4_CUDA_Libraries/cudaNvSci/cudaNvSci.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/cudaNvSci/cudaNvSci.h:53-72
```cpp
extern void rotateKernel(cudaTextureObject_t &texObj,
                         const float          angle,
                         unsigned int        *d_outputData,
                         const int            imageWidth,
                         const int            imageHeight,
                         // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
                         cudaStream_t         stream);
extern void launchGrayScaleKernel(unsigned int *d_rgbaImage,
                                  std::string   image_filename,
                                  size_t        imageWidth,
                                  size_t        imageHeight,
                                  cudaStream_t  stream);

class cudaNvSci
{
private:
    int            m_isMultiGPU;
    int            m_cudaNvSciSignalDeviceId;
    int            m_cudaNvSciWaitDeviceId;
    unsigned char *image_data;
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cudaNvSci/cudaNvSci.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `imageKernels.cu`

Source: cpp/4_CUDA_Libraries/cudaNvSci/imageKernels.cu:29-66
```cuda
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_image.h>

// convert floating point rgba color to 32-bit integer
__device__ unsigned int rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x); // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return ((unsigned int)(rgba.w * 255.0f) << 24) | ((unsigned int)(rgba.z * 255.0f) << 16)
         | ((unsigned int)(rgba.y * 255.0f) << 8) | ((unsigned int)(rgba.x * 255.0f));
}

////////////////////////////////////////////////////////////////////////////////
//! Rotate an image using texture lookups
//! @param outputData  output data in global memory
////////////////////////////////////////////////////////////////////////////////
static __global__ void
transformKernel(unsigned int *outputData, int width, int height, float theta, cudaTextureObject_t tex)
{
    // calculate normalized texture coordinates
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u  = (float)x - (float)width / 2;
    float v  = (float)y - (float)height / 2;
    float tu = u * cosf(theta) - v * sinf(theta);
    float tv = v * cosf(theta) + u * sinf(theta);

    tu /= (float)width;
    tv /= (float)height;

    // read from texture and write to global memory
    float4       pix          = tex2D<float4>(tex, tu + 0.5f, tv + 0.5f);
    unsigned int pixelInt     = rgbaFloatToInt(pix);
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cudaNvSci/imageKernels.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/cudaNvSci/imageKernels.cu:94-127
```cuda

    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    rgbToGrayscaleKernel<<<numOfBlocks, numThreadsPerBlock, 0, stream>>>(d_rgbaImage, imageWidth, imageHeight);

    unsigned int *outputData;
    // JP: `cudaMallocHost`: page-locked host memory は DMA/async copy を安定させます。通常の free ではなく対応する CUDA API で解放します。
    checkCudaErrors(cudaMallocHost((void **)&outputData, sizeof(unsigned int) * imageWidth * imageHeight));
    // JP: `cudaMemcpyAsync`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpyAsync(
        outputData, d_rgbaImage, sizeof(unsigned int) * imageWidth * imageHeight, cudaMemcpyDeviceToHost, stream));
    // JP: `cudaStreamSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaStreamSynchronize(stream));

    char outputFilename[1024];
    strcpy(outputFilename, image_filename.c_str());
    strcpy(outputFilename + image_filename.length() - 4, "_out.ppm");
    sdkSavePPM4ub(outputFilename, (unsigned char *)outputData, imageWidth, imageHeight);
    printf("Wrote '%s'\n", outputFilename);

    // JP: `cudaFreeHost`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFreeHost(outputData));
}

void rotateKernel(cudaTextureObject_t &texObj,
                  const float          angle,
                  unsigned int        *d_outputData,
                  const int            imageWidth,
                  const int            imageHeight,
                  // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
                  cudaStream_t         stream)
{
    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(imageWidth / dimBlock.x, imageHeight / dimBlock.y, 1);

```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cudaNvSci/imageKernels.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `main.cpp`

Source: cpp/4_CUDA_Libraries/cudaNvSci/main.cpp:29-47
```cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_image.h>
#include <vector>

#include "cudaNvSci.h"

void loadImageData(const std::string &filename,
                   const char       **argv,
                   unsigned char    **image_data,
                   uint32_t          &imageWidth,
                   uint32_t          &imageHeight)
{
    // load image (needed so we can get the width and height before we create
    // the window
    char *image_path = sdkFindFilePath(filename.c_str(), argv[0]);

    if (image_path == 0) {
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cudaNvSci/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/cudaNvSci/main.cpp:57-76
```cpp
    }

    printf("Loaded '%s', %d x %d pixels\n", image_path, imageWidth, imageHeight);
}

int main(int argc, const char **argv)
{
    int              numOfGPUs = 0;
    std::vector<int> deviceIds;
    checkCudaErrors(cudaGetDeviceCount(&numOfGPUs));

    printf("%d GPUs found\n", numOfGPUs);
    if (!numOfGPUs) {
        exit(EXIT_WAIVED);
    }
    else {
        for (int devID = 0; devID < numOfGPUs; devID++) {
            int major = 0, minor = 0;
            checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
            checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cudaNvSci/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaNvSci` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaSetDevice` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaStream_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaNvSciSignal` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDeviceGetAttribute` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaNvSciWait` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDeviceId` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaImportNvSciSemaphore` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaImportNvSciRawBuf` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaDestroyExternalMemory` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cuDeviceGetUuid_v2` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaGetDeviceCount` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaImportExternalMemory` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
- Runtime/Driver/NVRTC sample では、compile/load した module と launch する kernel 名の対応が重要です。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。
- 同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Sample-Specific Notes

- library sample では、CUDA kernel を直接書かなくても library call が device work を投入します。handle/descriptor/workspace の lifetime を kernel launch と同じ厳しさで追います。

> **日本語**
> この section は同じ template ではなく、sample 名、path、検出した API から読みどころを絞っています。
>
> **学習メモ**
> 似た名前の sample は Runtime 版、Driver 版、NVRTC 版、library 版の違いを比較すると学習効果が高くなります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target cudaNvSci
ctest --test-dir build -R cudaNvSci
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
- JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaNvSci` の直前と直後で、どの memory/resource が有効になったかをメモする。
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

- [Runtime, Driver, And NVRTC](../../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Performance](../../../docs_ja/themes/performance.md): memory traffic、occupancy、overlap、launch overhead、timing を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
