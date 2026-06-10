# dct8x8 - DCT8x8 - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates how Discrete Cosine Transform (DCT) for blocks of 8 by 8 pixels can be performed using CUDA: a naive implementation by definition and a more traditional approach used in many libraries. As opposed to implementing DCT in a fragment shader, CUDA allows for an easier and more efficient implementation.

Image Processing, Video Compression

Original README headings: `dct8x8 - DCT8x8`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/2_Concepts_and_Techniques/dct8x8` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `dct8x8` as a focused example of the CUDA concepts used in `cpp/2_Concepts_and_Techniques/dct8x8`.
> **日本語**
> この sample の目的は、`dct8x8` の小さな実装を通して Shared Memory, Performance, Memory, Kernel Launch And Indexing, Execution Model を具体的に追うことです。
>
> **学習メモ**
> 最初に `BmpUtil.cpp, BmpUtil.h, Common.h, DCT8x8_Gold.cpp, DCT8x8_Gold.h` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `BmpUtil.cpp`: Host-side setup, API calls, validation, and cleanup.
- `BmpUtil.h`: Host/device declarations, helper types, constants, or library wrappers.
- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `Common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `DCT8x8_Gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `DCT8x8_Gold.h`: Host/device declarations, helper types, constants, or library wrappers.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `data/teapot512.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `data/teapot512.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `dct8x8_kernel1.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `dct8x8_kernel2.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `dct8x8_kernel_quantization.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `dct8x8_kernel_short.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `doc/BarbaraBlocks1.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/BarbaraBlocks2.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/BarbaraBlocks3.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/CosineBasis.png`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/Cosines.xls`: Supporting file used by `doc/Cosines.xls`.
- `doc/DctJpeg.png`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/barbara.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/barbara_lg.png`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/barbara_md.png`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/barbara_sm.png`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/dct8x8.doc`: Supporting file used by `doc/dct8x8.doc`.
- `doc/dct8x8.pdf`: Supporting file used by `doc/dct8x8.pdf`.
- `teapot512_cuda1.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `teapot512_cuda2.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `teapot512_cuda_short.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `teapot512_gold1.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `teapot512_gold2.bmp`: Input, reference, generated-data description, or documentation used by the sample.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `BmpUtil.cpp` first and locate the host-side setup or Python entry point.
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

- `BmpUtil.cpp`: focus on control flow and helper functions.
- `BmpUtil.h`: focus on control flow and helper functions.
- `Common.h`: focus on `CUDA`.
- `DCT8x8_Gold.cpp`: focus on control flow and helper functions.
- `DCT8x8_Gold.h`: focus on control flow and helper functions.
- `dct8x8.cu`: focus on `CUDA`, `launch`, `cudaDeviceSynchronize`, `cudaMemcpy2D`, `cudaMallocPitch`.
- `dct8x8_kernel1.cuh`: focus on `blockIdx`, `threadIdx`, `__shared__`, `CUDA`, `cudaTextureObject_t`.
- `dct8x8_kernel2.cuh`: focus on `threadIdx`, `blockIdx`, `CUDAsubroutineInplaceDCTvector`, `CUDAsubroutineInplaceIDCTvector`, `__shared__`.
- `dct8x8_kernel_quantization.cuh`: focus on `blockIdx`, `threadIdx`, `launch`, `Device`, `CUDA`.
- `dct8x8_kernel_short.cuh`: focus on `threadIdx`, `CUDAshortInplaceDCT`, `CUDAshortInplaceIDCT`, `blockIdx`, `__shared__`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `BmpUtil.cpp`

Source: cpp/2_Concepts_and_Techniques/dct8x8/BmpUtil.cpp:38-56
```cpp
#include "BmpUtil.h"

#include "Common.h"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#pragma warning(disable : 4996) // disable deprecated warning
#endif

/**
**************************************************************************
*  The routine clamps the input value to integer byte range [0, 255]
*
* \param x          [IN] - Input value
*
* \return Pointer to the created plane
*/
int clamp_0_255(int x) { return (x < 0) ? 0 : ((x > 255) ? 255 : x); }

/**
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/BmpUtil.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/dct8x8/BmpUtil.cpp:195-214
```cpp
    //       _aligned_free(ptr);
    //   }
    // #else
    if (ptr) {
        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(ptr);
    }

    // #endif
}

/**
**************************************************************************
*  Performs addition of given value to each pixel in the plane
*
* \param Value              [IN] - Value to add
* \param ImgSrcDst          [IN/OUT] - Source float plane
* \param StrideF            [IN] - Source plane stride
* \param Size               [IN] - Size of area to copy
*
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/BmpUtil.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `BmpUtil.h`

Source: cpp/2_Concepts_and_Techniques/dct8x8/BmpUtil.h:38-56
```cpp
#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#pragma pack(push)
#endif

#pragma pack(1)

typedef char           int8;
typedef short          int16;
typedef int            int32;
typedef unsigned char  uint8;
typedef unsigned short uint16;
typedef unsigned int   uint32;

/**
 * \brief Bitmap file header structure
 *
 *  Bitmap file header structure
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/BmpUtil.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `CMakeLists.txt`

Source: cpp/2_Concepts_and_Techniques/dct8x8/CMakeLists.txt:1-62
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(dct8x8 LANGUAGES C CXX CUDA)

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
# Add target for dct8x8
add_executable(dct8x8 dct8x8.cu BmpUtil.cpp DCT8x8_Gold.cpp)

target_compile_options(dct8x8 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(dct8x8 PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(dct8x8 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(dct8x8 PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)

file(GLOB REF_DATA "teapot512*")
add_custom_command(TARGET dct8x8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${REF_DATA}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Copy data files to output directory
add_custom_command(TARGET dct8x8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/teapot512.ppm
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Copy data files to output directory
add_custom_command(TARGET dct8x8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/teapot512.bmp
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `Common.h`

Source: cpp/2_Concepts_and_Techniques/dct8x8/Common.h:37-88
```cpp
#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>      // helper functions for CUDA timing and initialization
#include <helper_functions.h> // helper functions for timing, string parsing
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 *  The dimension of pixels block
 */
#define BLOCK_SIZE 8

/**
 *  Square of dimension of pixels block
 */
#define BLOCK_SIZE2 64

/**
 *  log_2{BLOCK_SIZE), used for quick multiplication or division by the
 *  pixels block dimension via shifting
 */
#define BLOCK_SIZE_LOG2 3

/**
 *  log_2{BLOCK_SIZE*BLOCK_SIZE), used for quick multiplication or division by
 * the
 *  square of pixels block via shifting
 */
#define BLOCK_SIZE2_LOG2 6

/**
 *  This macro states that __mul24 operation is performed faster that traditional
 *  multiplication for two integers on CUDA. Please undefine if it appears to be
 *  wrong on your system
 */
#define __MUL24_FASTER_THAN_ASTERIX

/**
 *  Wrapper to the fastest integer multiplication function on CUDA
 */
#ifdef __MUL24_FASTER_THAN_ASTERIX
#define FMUL(x, y) (__mul24(x, y))
#else
#define FMUL(x, y) ((x) * (y))
#endif

/**
 *  This macro allows using aligned memory management
 */
// #define __ALLOW_ALIGNED_MEMORY_MANAGEMENT
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/Common.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `DCT8x8_Gold.cpp`

Source: cpp/2_Concepts_and_Techniques/dct8x8/DCT8x8_Gold.cpp:42-60
```cpp
#include "BmpUtil.h"
#include "Common.h"

/**
 *  This unitary matrix performs DCT of rows of the matrix to the left
 */
const float DCTv8matrix[BLOCK_SIZE2] = {
    0.3535533905932738f,  0.4903926402016152f,  0.4619397662556434f,  0.4157348061512726f,  0.3535533905932738f,
    0.2777851165098011f,  0.1913417161825449f,  0.0975451610080642f,  0.3535533905932738f,  0.4157348061512726f,
    0.1913417161825449f,  -0.0975451610080641f, -0.3535533905932737f, -0.4903926402016152f, -0.4619397662556434f,
    -0.2777851165098011f, 0.3535533905932738f,  0.2777851165098011f,  -0.1913417161825449f, -0.4903926402016152f,
    -0.3535533905932738f, 0.0975451610080642f,  0.4619397662556433f,  0.4157348061512727f,  0.3535533905932738f,
    0.0975451610080642f,  -0.4619397662556434f, -0.2777851165098011f, 0.3535533905932737f,  0.4157348061512727f,
    -0.1913417161825450f, -0.4903926402016153f, 0.3535533905932738f,  -0.0975451610080641f, -0.4619397662556434f,
    0.2777851165098009f,  0.3535533905932738f,  -0.4157348061512726f, -0.1913417161825453f, 0.4903926402016152f,
    0.3535533905932738f,  -0.2777851165098010f, -0.1913417161825452f, 0.4903926402016153f,  -0.3535533905932733f,
    -0.0975451610080649f, 0.4619397662556437f,  -0.4157348061512720f, 0.3535533905932738f,  -0.4157348061512727f,
    0.1913417161825450f,  0.0975451610080640f,  -0.3535533905932736f, 0.4903926402016152f,  -0.4619397662556435f,
    0.2777851165098022f,  0.3535533905932738f,  -0.4903926402016152f, 0.4619397662556433f,  -0.4157348061512721f,
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/DCT8x8_Gold.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `DCT8x8_Gold.h`

Source: cpp/2_Concepts_and_Techniques/dct8x8/DCT8x8_Gold.h:39-51
```cpp
#pragma once

#include "BmpUtil.h"

extern "C"
{
    void computeDCT8x8Gold1(const float *fSrc, float *fDst, int Stride, ROI Size);
    void computeIDCT8x8Gold1(const float *fSrc, float *fDst, int Stride, ROI Size);
    void quantizeGoldFloat(float *fSrcDst, int Stride, ROI Size);
    void quantizeGoldShort(short *fSrcDst, int Stride, ROI Size);
    void computeDCT8x8Gold2(const float *fSrc, float *fDst, int Stride, ROI Size);
    void computeIDCT8x8Gold2(const float *fSrc, float *fDst, int Stride, ROI Size);
}
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/DCT8x8_Gold.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `dct8x8.cu`

Source: cpp/2_Concepts_and_Techniques/dct8x8/dct8x8.cu:43-61
```cuda
#include "BmpUtil.h"
#include "Common.h"
#include "DCT8x8_Gold.h"

/**
 *  The number of DCT kernel calls
 */
#define BENCHMARK_SIZE 10

/**
 *  The PSNR values over this threshold indicate images equality
 */
#define PSNR_THRESHOLD_EQUAL 40

// includes kernels
#include "dct8x8_kernel1.cuh"
#include "dct8x8_kernel2.cuh"
#include "dct8x8_kernel_quantization.cuh"
#include "dct8x8_kernel_short.cuh"
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/dct8x8/dct8x8.cu:93-112
```cuda
        sdkStartTimer(&timerGold);
        computeDCT8x8Gold1(ImgF1, ImgF2, StrideF, Size);
        sdkStopTimer(&timerGold);
    }

    // stop and destroy CUDA timer
    float TimerGoldSpan = sdkGetAverageTimerValue(&timerGold);
    sdkDeleteTimer(&timerGold);

    // perform quantization
    quantizeGoldFloat(ImgF2, StrideF, Size);

    // perform block-wise IDCT processing
    computeIDCT8x8Gold1(ImgF2, ImgF1, StrideF, Size);

    // convert image back to byte representation
    AddFloatPlane(128.0f, ImgF1, StrideF, Size);
    CopyFloat2Byte(ImgF1, StrideF, ImgDst, Stride, Size);

    // free float buffers
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/dct8x8/dct8x8.cu:193-224
```cuda

    // allocate device memory
    cudaArray *Src;
    float     *Dst;
    size_t     DstStride;
    // JP: `cudaMallocArray`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMallocArray(&Src, &floattex, Size.width, Size.height));
    checkCudaErrors(cudaMallocPitch((void **)(&Dst), &DstStride, Size.width * sizeof(float), Size.height));
    DstStride /= sizeof(float);

    // convert source image to float representation
    int    ImgSrcFStride;
    float *ImgSrcF = MallocPlaneFloat(Size.width, Size.height, &ImgSrcFStride);
    CopyByte2Float(ImgSrc, Stride, ImgSrcF, ImgSrcFStride, Size);
    AddFloatPlane(-128.0f, ImgSrcF, ImgSrcFStride, Size);

    // copy from host memory to device
    checkCudaErrors(cudaMemcpy2DToArray(Src,
                                        0,
                                        0,
                                        ImgSrcF,
                                        ImgSrcFStride * sizeof(float),
                                        Size.width * sizeof(float),
                                        Size.height,
                                        // JP: `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
                                        cudaMemcpyHostToDevice));

    // setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

    // create and start CUDA timer
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/dct8x8/dct8x8.cu:246-266
```cuda
    checkCudaErrors(cudaCreateTextureObject(&TexSrc, &texRes, &texDescr, NULL));

    for (int i = 0; i < BENCHMARK_SIZE; i++) {
        sdkStartTimer(&timerCUDA);
        // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
        CUDAkernel1DCT<<<grid, threads>>>(Dst, (int)DstStride, 0, 0, TexSrc);
        // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
        checkCudaErrors(cudaDeviceSynchronize());
        sdkStopTimer(&timerCUDA);
    }

    getLastCudaError("Kernel execution failed");

    // finalize CUDA timer
    float TimerCUDASpan = sdkGetAverageTimerValue(&timerCUDA);
    sdkDeleteTimer(&timerCUDA);

    // execute Quantization kernel
    // JP: この連続する anchor 群では kernel launch の grid/block/shared-memory/stream 指定です。後続の sync/error check と完了確認を対応させます。
    CUDAkernelQuantizationFloat<<<grid, threads>>>(Dst, (int)DstStride);
    getLastCudaError("Kernel execution failed");
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `dct8x8_kernel1.cuh`

Source: cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel1.cuh:44-84
```cuda
#pragma once
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "Common.h"

/**
 *  This unitary matrix performs discrete cosine transform of rows of the matrix
 * to the left
 */
__constant__ float DCTv8matrix[] = {
    0.3535533905932738f,  0.4903926402016152f,  0.4619397662556434f,  0.4157348061512726f,  0.3535533905932738f,
    0.2777851165098011f,  0.1913417161825449f,  0.0975451610080642f,  0.3535533905932738f,  0.4157348061512726f,
    0.1913417161825449f,  -0.0975451610080641f, -0.3535533905932737f, -0.4903926402016152f, -0.4619397662556434f,
    -0.2777851165098011f, 0.3535533905932738f,  0.2777851165098011f,  -0.1913417161825449f, -0.4903926402016152f,
    -0.3535533905932738f, 0.0975451610080642f,  0.4619397662556433f,  0.4157348061512727f,  0.3535533905932738f,
    0.0975451610080642f,  -0.4619397662556434f, -0.2777851165098011f, 0.3535533905932737f,  0.4157348061512727f,
    -0.1913417161825450f, -0.4903926402016153f, 0.3535533905932738f,  -0.0975451610080641f, -0.4619397662556434f,
    0.2777851165098009f,  0.3535533905932738f,  -0.4157348061512726f, -0.1913417161825453f, 0.4903926402016152f,
    0.3535533905932738f,  -0.2777851165098010f, -0.1913417161825452f, 0.4903926402016153f,  -0.3535533905932733f,
    -0.0975451610080649f, 0.4619397662556437f,  -0.4157348061512720f, 0.3535533905932738f,  -0.4157348061512727f,
    0.1913417161825450f,  0.0975451610080640f,  -0.3535533905932736f, 0.4903926402016152f,  -0.4619397662556435f,
    0.2777851165098022f,  0.3535533905932738f,  -0.4903926402016152f, 0.4619397662556433f,  -0.4157348061512721f,
    0.3535533905932733f,  -0.2777851165098008f, 0.1913417161825431f,  -0.0975451610080625f};

// Temporary blocks
// JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
__shared__ float CurBlockLocal1[BLOCK_SIZE2];
__shared__ float CurBlockLocal2[BLOCK_SIZE2];

/**
**************************************************************************
*  Performs 1st implementation of 8x8 block-wise Forward Discrete Cosine
*Transform of the given
*  image plane and outputs result to the array of coefficients.
*
* \param Dst            [OUT] - Coefficients plane
* \param ImgWidth       [IN] - Stride of Dst
* \param OffsetXBlocks  [IN] - Offset along X in blocks from which to perform
*processing
* \param OffsetYBlocks  [IN] - Offset along Y in blocks from which to perform
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel1.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `dct8x8_kernel2.cuh`

Source: cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel2.cuh:42-60
```cuda
#pragma once

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include "Common.h"

// Used in forward and inverse DCT
#define C_a 1.387039845322148f //!< a = (2^0.5) * cos(    pi / 16);
#define C_b 1.306562964876377f //!< b = (2^0.5) * cos(    pi /  8);
#define C_c 1.175875602419359f //!< c = (2^0.5) * cos(3 * pi / 16);
#define C_d 0.785694958387102f //!< d = (2^0.5) * cos(5 * pi / 16);
#define C_e 0.541196100146197f //!< e = (2^0.5) * cos(3 * pi /  8);
#define C_f 0.275899379282943f //!< f = (2^0.5) * cos(7 * pi / 16);

/**
 *  Normalization constant that is used in forward and inverse DCT
 */
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel2.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel2.cuh:195-214
```cuda
{
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block cta = cg::this_thread_block();

    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    __shared__ float block[KER2_BLOCK_HEIGHT * KER2_SMEMBLOCK_STRIDE];

    int OffsThreadInRow = threadIdx.y * BLOCK_SIZE + threadIdx.x;
    int OffsThreadInCol = threadIdx.z * BLOCK_SIZE;
    src += FMUL(blockIdx.y * KER2_BLOCK_HEIGHT + OffsThreadInCol, ImgStride) + blockIdx.x * KER2_BLOCK_WIDTH
         + OffsThreadInRow;
    dst += FMUL(blockIdx.y * KER2_BLOCK_HEIGHT + OffsThreadInCol, ImgStride) + blockIdx.x * KER2_BLOCK_WIDTH
         + OffsThreadInRow;
    float *bl_ptr = block + OffsThreadInCol * KER2_SMEMBLOCK_STRIDE + OffsThreadInRow;

#pragma unroll

    for (unsigned int i = 0; i < BLOCK_SIZE; i++)
        bl_ptr[i * KER2_SMEMBLOCK_STRIDE] = src[i * ImgStride];
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel2.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `dct8x8_kernel_quantization.cuh`

Source: cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel_quantization.cuh:38-78
```cuda
#pragma once
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "Common.h"

/**
 *  JPEG quality=0_of_12 quantization matrix
 */
__constant__ short Q[] = {32, 33, 51, 81, 66, 39, 34, 17, 33, 36, 48, 47, 28, 23, 12, 12, 51, 48, 47, 28, 23, 12,
                          12, 12, 81, 47, 28, 23, 12, 12, 12, 12, 66, 28, 23, 12, 12, 12, 12, 12, 39, 23, 12, 12,
                          12, 12, 12, 12, 34, 12, 12, 12, 12, 12, 12, 12, 17, 12, 12, 12, 12, 12, 12, 12};

/**
**************************************************************************
*  Performs in-place quantization of given DCT coefficients plane using
*  predefined quantization matrices (for floats plane). Unoptimized.
*
* \param SrcDst         [IN/OUT] - DCT coefficients plane
* \param Stride         [IN] - Stride of SrcDst
*
* \return None
*/
__global__ void CUDAkernelQuantizationFloat(float *SrcDst, int Stride)
{
    // Block index
    // JP: `blockIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index (current coefficient)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // copy current coefficient to the local variable
    float curCoef  = SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx)];
    float curQuant = (float)Q[ty * BLOCK_SIZE + tx];

    // quantize the current coefficient
    float quantized = roundf(curCoef / curQuant);
    curCoef         = quantized * curQuant;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel_quantization.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `dct8x8_kernel_short.cuh`

Source: cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel_short.cuh:42-60
```cuda
#pragma once

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "Common.h"

/**
 *  Width of data block (short kernel)
 */
#define KERS_BLOCK_WIDTH 32

/**
 *  Height of data block (short kernel)
 */
#define KERS_BLOCK_HEIGHT 32

/**
 *  LOG2 of width of data block (short kernel)
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel_short.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel_short.cuh:443-462
```cuda
__global__ void CUDAkernelShortDCT(short *SrcDst, int ImgStride)
{
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block cta = cg::this_thread_block();
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    __shared__ short block[KERS_BLOCK_HEIGHT * KERS_SMEMBLOCK_STRIDE];
    int              OffsThreadInRow = FMUL(threadIdx.y, BLOCK_SIZE) + threadIdx.x;
    int              OffsThreadInCol = FMUL(threadIdx.z, BLOCK_SIZE);
    int              OffsThrRowPermuted =
        (OffsThreadInRow & 0xFFFFFFE0) | ((OffsThreadInRow << 1) | (OffsThreadInRow >> 4) & 0x1) & 0x1F;

    SrcDst += IMAD(IMAD(blockIdx.y, KERS_BLOCK_HEIGHT, OffsThreadInCol),
                   ImgStride,
                   IMAD(blockIdx.x, KERS_BLOCK_WIDTH, OffsThreadInRow * 2));
    short *bl_ptr = block + IMAD(OffsThreadInCol, KERS_SMEMBLOCK_STRIDE, OffsThreadInRow * 2);

    // load data to shared memory (only first half of threads in each row performs
    // data moving (each thread moves 2 shorts)
    if (OffsThreadInRow < KERS_BLOCK_WIDTH_HALF) {
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel_short.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `__shared__` | block 内共有 memory または同期境界です。producer/consumer の順序を確認します。 |
| `cudaMallocPitch` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaMemcpy2D` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMallocArray` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaMemcpyHostToDevice` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaDestroyTextureObject` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `CUDAshortInplaceDCT` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CUDAshortInplaceIDCT` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `Device` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
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
cmake --build build --target dct8x8
ctest --test-dir build -R dct8x8
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
