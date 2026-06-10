# eigenvalues - Eigenvalues - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

The computation of all or a subset of all eigenvalues is an important problem in Linear Algebra, statistics, physics, and many other fields. This sample demonstrates a parallel implementation of a bisection algorithm for the computation of all eigenvalues of a tridiagonal symmetric matrix of arbitrary size with CUDA.

Linear Algebra

Original README headings: `eigenvalues - Eigenvalues`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/2_Concepts_and_Techniques/eigenvalues` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `eigenvalues` as a focused example of the CUDA concepts used in `cpp/2_Concepts_and_Techniques/eigenvalues`.
> **日本語**
> この sample の目的は、`eigenvalues` の小さな実装を通して Shared Memory, Synchronization And Atomics, Memory, Kernel Launch And Indexing, Execution Model を具体的に追うことです。
>
> **学習メモ**
> 最初に `bisect_kernel_large.cuh, bisect_kernel_large_multi.cuh, bisect_kernel_large_onei.cuh, bisect_kernel_small.cuh, bisect_large.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `bisect_kernel_large.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `bisect_kernel_large_multi.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `bisect_kernel_large_onei.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `bisect_kernel_small.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `bisect_large.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `bisect_large.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `bisect_small.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `bisect_small.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `bisect_util.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `config.h`: Host/device declarations, helper types, constants, or library wrappers.
- `data/diagonal.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `data/reference.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `data/superdiagonal.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/eigenvalues.doc`: Supporting file used by `doc/eigenvalues.doc`.
- `doc/eigenvalues.pdf`: Supporting file used by `doc/eigenvalues.pdf`.
- `gerschgorin.cpp`: Host-side setup, API calls, validation, and cleanup.
- `gerschgorin.h`: Host/device declarations, helper types, constants, or library wrappers.
- `main.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `matlab.cpp`: Host-side setup, API calls, validation, and cleanup.
- `matlab.h`: Host/device declarations, helper types, constants, or library wrappers.
- `structs.h`: Host/device declarations, helper types, constants, or library wrappers.
- `util.h`: Host/device declarations, helper types, constants, or library wrappers.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `bisect_kernel_large.cuh` first and locate the host-side setup or Python entry point.
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

- `bisect_kernel_large.cuh`: focus on `__shared__`, `threadIdx`, `blockDim`, `launch`, `atomicExch`.
- `bisect_kernel_large_multi.cuh`: focus on `__shared__`, `threadIdx`, `blockIdx`, `launch`, `blockDim`.
- `bisect_kernel_large_onei.cuh`: focus on `__shared__`, `threadIdx`, `atomicExch`, `launch`, `blockDim`.
- `bisect_kernel_small.cuh`: focus on `threadIdx`, `__shared__`, `launch`.
- `bisect_large.cu`: focus on `cudaMemcpy`, `cudaMalloc`, `cudaMemcpyHostToDevice`, `cudaFree`, `cudaMemcpyDeviceToHost`.
- `bisect_large.cuh`: focus on control flow and helper functions.
- `bisect_small.cu`: focus on `cudaMemcpy`, `cudaMalloc`, `cudaMemcpyHostToDevice`, `cudaFree`, `launch`.
- `bisect_small.cuh`: focus on control flow and helper functions.
- `bisect_util.cu`: focus on `threadIdx`, `blockDim`, `atomicExch`, `launch`.
- `config.h`: focus on control flow and helper functions.
- Additional source files: 7 more support files. Use the same setup/work/sync/cleanup lens.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/CMakeLists.txt:1-58
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(eigenvalues LANGUAGES C CXX CUDA)

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
# Add target for eigenvalues
add_executable(eigenvalues bisect_large.cu bisect_small.cu bisect_util.cu gerschgorin.cpp main.cu matlab.cpp)

target_compile_options(eigenvalues PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(eigenvalues PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(eigenvalues PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Copy data files to output directory
add_custom_command(TARGET eigenvalues POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/diagonal.dat
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Copy data files to output directory
add_custom_command(TARGET eigenvalues POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/superdiagonal.dat
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Copy data files to output directory
add_custom_command(TARGET eigenvalues POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/reference.dat
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bisect_kernel_large.cuh`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large.cuh:30-48
```cuda
  step of the computation. */

#ifndef _BISECT_KERNEL_LARGE_H_
#define _BISECT_KERNEL_LARGE_H_
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
// includes, project
#include "config.h"
#include "util.h"

// additional kernel
#include "bisect_util.cu"

// declaration, forward

////////////////////////////////////////////////////////////////////////////////
//! Write data to global memory
////////////////////////////////////////////////////////////////////////////////
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large.cuh:188-207
```cuda
                                  unsigned int      *g_blocks_mult_sum)
{
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block   cta = cg::this_thread_block();
    const unsigned int tid = threadIdx.x;

    // intervals (store left and right because the subdivision tree is in general
    // not dense
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    __shared__ float s_left[2 * MAX_THREADS_BLOCK + 1];
    __shared__ float s_right[2 * MAX_THREADS_BLOCK + 1];

    // number of eigenvalues that are smaller than s_left / s_right
    // (correspondence is realized via indices)
    __shared__ unsigned short s_left_count[2 * MAX_THREADS_BLOCK + 1];
    __shared__ unsigned short s_right_count[2 * MAX_THREADS_BLOCK + 1];

    // helper for stream compaction
    __shared__ unsigned short s_compaction_list[2 * MAX_THREADS_BLOCK + 1];
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bisect_kernel_large_multi.cuh`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large_multi.cuh:33-51
```cuda
#ifndef _BISECT_KERNEL_LARGE_MULTI_H_
#define _BISECT_KERNEL_LARGE_MULTI_H_

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
// includes, project
#include "config.h"
#include "util.h"

// additional kernel
#include "bisect_util.cu"

////////////////////////////////////////////////////////////////////////////////
//! Perform second step of bisection algorithm for large matrices for
//! intervals that after the first step contained more than one eigenvalue
//! @param  g_d  diagonal elements of symmetric, tridiagonal matrix
//! @param  g_s  superdiagonal elements of symmetric, tridiagonal matrix
//! @param  n    matrix size
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large_multi.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large_multi.cuh:76-95
```cuda
                                                float              precision)
{
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block   cta = cg::this_thread_block();
    const unsigned int tid = threadIdx.x;

    // left and right limits of interval
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    __shared__ float s_left[2 * MAX_THREADS_BLOCK];
    __shared__ float s_right[2 * MAX_THREADS_BLOCK];

    // number of eigenvalues smaller than interval limits
    __shared__ unsigned int s_left_count[2 * MAX_THREADS_BLOCK];
    __shared__ unsigned int s_right_count[2 * MAX_THREADS_BLOCK];

    // helper array for chunk compaction of second chunk
    __shared__ unsigned int s_compaction_list[2 * MAX_THREADS_BLOCK + 1];
    // compaction list helper for exclusive scan
    unsigned int *s_compaction_list_exc = s_compaction_list + 1;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large_multi.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bisect_kernel_large_onei.cuh`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large_onei.cuh:33-51
```cuda
#ifndef _BISECT_KERNEL_LARGE_ONEI_H_
#define _BISECT_KERNEL_LARGE_ONEI_H_

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// includes, project
#include "config.h"
#include "util.h"

// additional kernel
#include "bisect_util.cu"

////////////////////////////////////////////////////////////////////////////////
//! Determine eigenvalues for large matrices for intervals that after
//! the first step contained one eigenvalue
//! @param  g_d  diagonal elements of symmetric, tridiagonal matrix
//! @param  g_s  superdiagonal elements of symmetric, tridiagonal matrix
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large_onei.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large_onei.cuh:68-87
```cuda
                                               float              precision)
{
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block   cta  = cg::this_thread_block();
    const unsigned int gtid = (blockDim.x * blockIdx.x) + threadIdx.x;

    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    __shared__ float s_left_scratch[MAX_THREADS_BLOCK];
    __shared__ float s_right_scratch[MAX_THREADS_BLOCK];

    // active interval of thread
    // left and right limit of current interval
    float left, right;
    // number of threads smaller than the right limit (also corresponds to the
    // global index of the eigenvalues contained in the active interval)
    unsigned int right_count;
    // flag if current thread converged
    unsigned int converged = 0;
    // midpoint when current interval is subdivided
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large_onei.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bisect_kernel_small.cuh`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_small.cuh:31-49
```cuda
#ifndef _BISECT_KERNEL_SMALL_H_
#define _BISECT_KERNEL_SMALL_H_

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// includes, project
#include "config.h"
#include "util.h"

// additional kernel
#include "bisect_util.cu"

////////////////////////////////////////////////////////////////////////////////
//! Bisection to find eigenvalues of a real, symmetric, and tridiagonal matrix
//! @param  g_d  diagonal elements in global memory
//! @param  g_s  superdiagonal elements in global elements (stored so that the
//!              element *(g_s - 1) can be accessed an equals 0
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_small.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_small.cuh:70-89
```cuda
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block cta = cg::this_thread_block();
    // intervals (store left and right because the subdivision tree is in general
    // not dense
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    __shared__ float s_left[MAX_THREADS_BLOCK_SMALL_MATRIX];
    __shared__ float s_right[MAX_THREADS_BLOCK_SMALL_MATRIX];

    // number of eigenvalues that are smaller than s_left / s_right
    // (correspondence is realized via indices)
    __shared__ unsigned int s_left_count[MAX_THREADS_BLOCK_SMALL_MATRIX];
    __shared__ unsigned int s_right_count[MAX_THREADS_BLOCK_SMALL_MATRIX];

    // helper for stream compaction
    __shared__ unsigned int s_compaction_list[MAX_THREADS_BLOCK_SMALL_MATRIX + 1];

    // state variables for whole block
    // if 0 then compaction of second chunk of child intervals is not necessary
    // (because all intervals had exactly one non-dead child)
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_small.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bisect_large.cu`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_large.cu:32-50
```cuda
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include "bisect_large.cuh"
#include "config.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "matlab.h"
#include "structs.h"
#include "util.h"

// includes, kernels
#include "bisect_kernel_large.cuh"
#include "bisect_kernel_large_multi.cuh"
#include "bisect_kernel_large_onei.cuh"
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_large.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_large.cu:59-89
```cuda
    // helper variables to initialize memory
    unsigned int zero        = 0;
    unsigned int mat_size_f  = sizeof(float) * mat_size;
    unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;

    float        *tempf  = (float *)malloc(mat_size_f);
    unsigned int *tempui = (unsigned int *)malloc(mat_size_ui);

    for (unsigned int i = 0; i < mat_size; ++i) {
        tempf[i]  = 0.0f;
        tempui[i] = 0;
    }

    // number of intervals containing only one eigenvalue after the first step
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc((void **)&result.g_num_one, sizeof(unsigned int)));
    // JP: `cudaMemcpy`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpy(result.g_num_one, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // number of (thread) blocks of intervals with multiple eigenvalues after
    // the first iteration
    // JP: この anchor では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaMalloc((void **)&result.g_num_blocks_mult, sizeof(unsigned int)));
    // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpy(result.g_num_blocks_mult, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // JP: この連続する anchor 群では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaMalloc((void **)&result.g_left_one, mat_size_f));
    checkCudaErrors(cudaMalloc((void **)&result.g_right_one, mat_size_f));
    checkCudaErrors(cudaMalloc((void **)&result.g_pos_one, mat_size_ui));

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_large.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_large.cu:125-144
```cuda
//! Cleanup result memory
//! @param result  handles to memory
////////////////////////////////////////////////////////////////////////////////
void cleanupResultDataLargeMatrix(ResultDataLarge &result)
{
    // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFree(result.g_num_one));
    checkCudaErrors(cudaFree(result.g_num_blocks_mult));
    checkCudaErrors(cudaFree(result.g_left_one));
    checkCudaErrors(cudaFree(result.g_right_one));
    checkCudaErrors(cudaFree(result.g_pos_one));
    checkCudaErrors(cudaFree(result.g_left_mult));
    checkCudaErrors(cudaFree(result.g_right_mult));
    checkCudaErrors(cudaFree(result.g_left_count_mult));
    checkCudaErrors(cudaFree(result.g_right_count_mult));
    checkCudaErrors(cudaFree(result.g_blocks_mult));
    checkCudaErrors(cudaFree(result.g_blocks_mult_sum));
    checkCudaErrors(cudaFree(result.g_lambda_mult));
    checkCudaErrors(cudaFree(result.g_pos_mult));
}
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_large.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_large.cu:177-217
```cuda

    // do for multiple iterations to improve timing accuracy
    for (unsigned int iter = 0; iter < iterations; ++iter) {
        sdkStartTimer(&timer_step1);
        // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
        bisectKernelLarge<<<blocks, threads>>>(input.g_a,
                                               input.g_b,
                                               mat_size,
                                               lg,
                                               ug,
                                               0,
                                               mat_size,
                                               precision,
                                               result.g_num_one,
                                               result.g_num_blocks_mult,
                                               result.g_left_one,
                                               result.g_right_one,
                                               result.g_pos_one,
                                               result.g_left_mult,
                                               result.g_right_mult,
                                               result.g_left_count_mult,
                                               result.g_right_count_mult,
                                               result.g_blocks_mult,
                                               result.g_blocks_mult_sum);

        getLastCudaError("Kernel launch failed.");
        // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
        checkCudaErrors(cudaDeviceSynchronize());
        sdkStopTimer(&timer_step1);

        // get the number of intervals containing one eigenvalue after the first
        // processing step
        unsigned int num_one_intervals;
        // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
        checkCudaErrors(cudaMemcpy(&num_one_intervals, result.g_num_one, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        dim3 grid_onei;
        grid_onei.x = getNumBlocksLinear(num_one_intervals, MAX_THREADS_BLOCK);
        dim3 threads_onei;
        // use always max number of available threads to better balance load times
        // for matrix data
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_large.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bisect_large.cuh`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_large.cuh:31-86
```cuda
#ifndef _BISECT_LARGE_CUH_
#define _BISECT_LARGE_CUH_

#include "structs.h"

extern "C"
{

    ////////////////////////////////////////////////////////////////////////////////
    //! Run the kernels to compute the eigenvalues for large matrices
    //! @param  input   handles to input data
    //! @param  result  handles to result data
    //! @param  mat_size  matrix size
    //! @param  precision  desired precision of eigenvalues
    //! @param  lg  lower limit of Gerschgorin interval
    //! @param  ug  upper limit of Gerschgorin interval
    //! @param  iterations  number of iterations (for timing)
    ////////////////////////////////////////////////////////////////////////////////
    void computeEigenvaluesLargeMatrix(const InputData       &input,
                                       const ResultDataLarge &result,
                                       const unsigned int     mat_size,
                                       const float            precision,
                                       const float            lg,
                                       const float            ug,
                                       const unsigned int     iterations);

    ////////////////////////////////////////////////////////////////////////////////
    //! Initialize variables and memory for result
    //! @param  result handles to memory
    //! @param  matr_size  size of the matrix
    ////////////////////////////////////////////////////////////////////////////////
    void initResultDataLargeMatrix(ResultDataLarge &result, const unsigned int mat_size);

    ////////////////////////////////////////////////////////////////////////////////
    //! Cleanup result memory
    //! @param result  handles to memory
    ////////////////////////////////////////////////////////////////////////////////
    void cleanupResultDataLargeMatrix(ResultDataLarge &result);

    ////////////////////////////////////////////////////////////////////////////////
    //! Process the result, that is obtain result from device and do simple sanity
    //! checking
    //! @param  input   handles to input data
    //! @param  result  handles to result data
    //! @param  mat_size  matrix size
    //! @param  filename  output filename
    ////////////////////////////////////////////////////////////////////////////////
    bool processResultDataLargeMatrix(const InputData       &input,
                                      const ResultDataLarge &result,
                                      const unsigned int     mat_size,
                                      const char            *filename,
                                      const unsigned int     user_defined,
                                      char                  *exec_path);
};

#endif // #ifndef _BISECT_LARGE_CUH_
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_large.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bisect_small.cu`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_small.cu:32-50
```cuda
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include "config.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "matlab.h"
#include "structs.h"

// includes, kernels
#include "bisect_kernel_small.cuh"

// includes, file
#include "bisect_small.cuh"

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_small.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_small.cu:74-164
```cuda
    for (unsigned int i = 0; i < iterations; ++i) {
        dim3 blocks(1, 1, 1);
        dim3 threads(MAX_THREADS_BLOCK_SMALL_MATRIX, 1, 1);

        // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
        bisectKernel<<<blocks, threads>>>(input.g_a,
                                          input.g_b,
                                          mat_size,
                                          result.g_left,
                                          result.g_right,
                                          result.g_left_count,
                                          result.g_right_count,
                                          lg,
                                          ug,
                                          0,
                                          mat_size,
                                          precision);
    }

    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    getLastCudaError("Kernel launch failed");
    printf("Average time: %f ms (%i iterations)\n", sdkGetTimerValue(&timer) / (float)iterations, iterations);

    sdkDeleteTimer(&timer);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize variables and memory for the result for small matrices
//! @param result  handles to the necessary memory
//! @param  mat_size  matrix_size
////////////////////////////////////////////////////////////////////////////////
void initResultSmallMatrix(ResultDataSmall &result, const unsigned int mat_size)
{
    result.mat_size_f  = sizeof(float) * mat_size;
    result.mat_size_ui = sizeof(unsigned int) * mat_size;

    result.eigenvalues = (float *)malloc(result.mat_size_f);

    // helper variables
    result.zero_f  = (float *)malloc(result.mat_size_f);
    result.zero_ui = (unsigned int *)malloc(result.mat_size_ui);

    for (unsigned int i = 0; i < mat_size; ++i) {
        result.zero_f[i]  = 0.0f;
        result.zero_ui[i] = 0;

        result.eigenvalues[i] = 0.0f;
    }

    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc((void **)&result.g_left, result.mat_size_f));
    checkCudaErrors(cudaMalloc((void **)&result.g_right, result.mat_size_f));

    checkCudaErrors(cudaMalloc((void **)&result.g_left_count, result.mat_size_ui));
    checkCudaErrors(cudaMalloc((void **)&result.g_right_count, result.mat_size_ui));

    // initialize result memory
    // JP: `cudaMemcpy`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpy(result.g_left, result.zero_f, result.mat_size_f, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(result.g_right, result.zero_f, result.mat_size_f, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(result.g_right_count, result.zero_ui, result.mat_size_ui, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(result.g_left_count, result.zero_ui, result.mat_size_ui, cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////////////////////////
//! Cleanup memory and variables for result for small matrices
//! @param  result  handle to variables
////////////////////////////////////////////////////////////////////////////////
void cleanupResultSmallMatrix(ResultDataSmall &result)
{
    freePtr(result.eigenvalues);
    freePtr(result.zero_f);
    freePtr(result.zero_ui);

    // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFree(result.g_left));
    checkCudaErrors(cudaFree(result.g_right));
    checkCudaErrors(cudaFree(result.g_left_count));
    checkCudaErrors(cudaFree(result.g_right_count));
}

////////////////////////////////////////////////////////////////////////////////
//! Process the result obtained on the device, that is transfer to host and
//! perform basic sanity checking
//! @param  input  handles to input data
//! @param  result  handles to result data
//! @param  mat_size   matrix size
//! @param  filename  output filename
////////////////////////////////////////////////////////////////////////////////
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_small.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bisect_small.cuh`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_small.cuh:31-83
```cuda
#ifndef _BISECT_SMALL_CUH_
#define _BISECT_SMALL_CUH_

extern "C"
{

    ////////////////////////////////////////////////////////////////////////////////
    //! Determine eigenvalues for matrices smaller than MAX_SMALL_MATRIX
    //! @param TimingIterations  number of iterations for timing
    //! @param  input  handles to input data of kernel
    //! @param  result handles to result of kernel
    //! @param  mat_size  matrix size
    //! @param  lg  lower limit of Gerschgorin interval
    //! @param  ug  upper limit of Gerschgorin interval
    //! @param  precision  desired precision of eigenvalues
    //! @param  iterations  number of iterations for timing
    ////////////////////////////////////////////////////////////////////////////////
    void computeEigenvaluesSmallMatrix(const InputData   &input,
                                       ResultDataSmall   &result,
                                       const unsigned int mat_size,
                                       const float        lg,
                                       const float        ug,
                                       const float        precision,
                                       const unsigned int iterations);

    ////////////////////////////////////////////////////////////////////////////////
    //! Initialize variables and memory for the result for small matrices
    //! @param result  handles to the necessary memory
    //! @param  mat_size  matrix_size
    ////////////////////////////////////////////////////////////////////////////////
    void initResultSmallMatrix(ResultDataSmall &result, const unsigned int mat_size);

    ////////////////////////////////////////////////////////////////////////////////
    //! Cleanup memory and variables for result for small matrices
    //! @param  result  handle to variables
    ////////////////////////////////////////////////////////////////////////////////
    void cleanupResultSmallMatrix(ResultDataSmall &result);

    ////////////////////////////////////////////////////////////////////////////////
    //! Process the result obtained on the device, that is transfer to host and
    //! perform basic sanity checking
    //! @param  input   handles to input data
    //! @param  result  handles to result variables
    //! @param  mat_size   matrix size
    //! @param  filename  output filename
    ////////////////////////////////////////////////////////////////////////////////
    void processResultSmallMatrix(const InputData       &input,
                                  const ResultDataSmall &result,
                                  const unsigned int     mat_size,
                                  const char            *filename);
}

#endif // #ifndef _BISECT_SMALL_CUH_
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_small.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bisect_util.cu`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_util.cu:31-49
```cuda
#ifndef _BISECT_UTIL_H_
#define _BISECT_UTIL_H_

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// includes, project
#include "config.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////
//! Compute the next lower power of two of n
//! @param  n  number for which next higher power of two is sought
////////////////////////////////////////////////////////////////////////////////
__device__ inline int floorPow2(int n)
{
    // early out if already power of two
    if (0 == (n & (n - 1))) {
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_util.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_util.cu:146-165
```cuda
//! @param  g_d  diagonal elements stored in global memory
//! @param  g_s  superdiagonal elements stored in global memory
//! @param  n    size of matrix
//! @param  x    value for which the number of eigenvalues that are smaller is
//!              seeked
//! @param  tid  thread identified (e.g. threadIdx.x or gtid)
//! @param  num_intervals_active  number of active intervals / threads that
//!                               currently process an interval
//! @param  s_d  scratch space to store diagonal entries of the tridiagonal
//!              matrix in shared memory
//! @param  s_s  scratch space to store superdiagonal entries of the tridiagonal
//!              matrix in shared memory
//! @param  converged  flag if the current thread is already converged (that
//!         is count does not have to be computed)
////////////////////////////////////////////////////////////////////////////////
__device__ inline unsigned int computeNumSmallerEigenvals(float             *g_d,
                                                          float             *g_s,
                                                          const unsigned int n,
                                                          const float        x,
                                                          const unsigned int tid,
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_util.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/eigenvalues/bisect_util.cu:237-256
```cuda
    // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    for (unsigned int i = 0; i < n; i += blockDim.x) {
        // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
        cg::sync(cta);

        // read new chunk of data into shared memory
        // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
        if ((i + threadIdx.x) < n) {
            s_d[threadIdx.x] = *(g_d + i + threadIdx.x);
            s_s[threadIdx.x] = *(g_s + i + threadIdx.x - 1);
        }

        // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
        cg::sync(cta);

        if (tid < num_intervals_active) {
            // perform (optimized) Gaussian elimination to determine the number
            // of eigenvalues that are smaller than n
            // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
            for (unsigned int k = 0; k < min(rem, blockDim.x); ++k) {
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_util.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `config.h`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/config.h:31-42
```cpp
#ifndef _CONFIG_H_
#define _CONFIG_H_

// should be power of two
#define MAX_THREADS_BLOCK 256

#define MAX_SMALL_MATRIX               512
#define MAX_THREADS_BLOCK_SMALL_MATRIX 512

#define MIN_ABS_INTERVAL 5.0e-37

#endif // #ifndef _CONFIG_H_
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/config.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `gerschgorin.cpp`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/gerschgorin.cpp:32-85
```cpp
#include "gerschgorin.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "util.h"

////////////////////////////////////////////////////////////////////////////////
//! Compute Gerschgorin interval for symmetric, tridiagonal matrix
//! @param  d  diagonal elements
//! @param  s  superdiagonal elements
//! @param  n  size of matrix
//! @param  lg  lower limit of Gerschgorin interval
//! @param  ug  upper limit of Gerschgorin interval
////////////////////////////////////////////////////////////////////////////////
void computeGerschgorin(float *d, float *s, unsigned int n, float &lg, float &ug)
{
    lg = FLT_MAX;
    ug = -FLT_MAX;

    // compute bounds
    for (unsigned int i = 1; i < (n - 1); ++i) {
        // sum over the absolute values of all elements of row i
        float sum_abs_ni = fabsf(s[i - 1]) + fabsf(s[i]);

        lg = min(lg, d[i] - sum_abs_ni);
        ug = max(ug, d[i] + sum_abs_ni);
    }

    // first and last row, only one superdiagonal element

    // first row
    lg = min(lg, d[0] - fabsf(s[0]));
    ug = max(ug, d[0] + fabsf(s[0]));

    // last row
    lg = min(lg, d[n - 1] - fabsf(s[n - 2]));
    ug = max(ug, d[n - 1] + fabsf(s[n - 2]));

    // increase interval to avoid side effects of fp arithmetic
    float bnorm = max(fabsf(ug), fabsf(lg));

    // these values depend on the implementation of floating count that is
    // employed in the following
    float psi_0 = 11 * FLT_EPSILON * bnorm;
    float psi_n = 11 * FLT_EPSILON * bnorm;

    lg = lg - bnorm * 2 * n * FLT_EPSILON - psi_0;
    ug = ug + bnorm * 2 * n * FLT_EPSILON + psi_n;

    ug = max(lg, ug);
}
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/gerschgorin.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `gerschgorin.h`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/gerschgorin.h:31-44
```cpp
#ifndef _GERSCHGORIN_H_
#define _GERSCHGORIN_H_

////////////////////////////////////////////////////////////////////////////////
//! Compute Gerschgorin interval for symmetric, tridiagonal matrix
//! @param  d  diagonal elements
//! @param  s  superdiagonal elements
//! @param  n  size of matrix
//! @param  lg  lower limit of Gerschgorin interval
//! @param  ug  upper limit of Gerschgorin interval
////////////////////////////////////////////////////////////////////////////////
extern "C" void computeGerschgorin(float *d, float *s, unsigned int n, float &lg, float &ug);

#endif // #ifndef _GERSCHGORIN_H_
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/gerschgorin.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `main.cu`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/main.cu:29-74
```cuda
/* Computation of eigenvalues of symmetric, tridiagonal matrix using
 * bisection.
 */

// includes, system
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>

#include "bisect_large.cuh"
#include "bisect_small.cuh"
#include "config.h"
#include "gerschgorin.h"
#include "matlab.h"
#include "structs.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    bool bQAResults = false;

    printf("Starting eigenvalues\n");

    bQAResults = runTest(argc, argv);
    printf("Test %s\n", bQAResults ? "Succeeded!" : "Failed!");

    exit(bQAResults ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize the input data to the algorithm
//! @param input  handles to the input data
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/main.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/eigenvalues/main.cu:78-97
```cuda
//!                      0 if the default size
////////////////////////////////////////////////////////////////////////////////
void initInputData(InputData &input, char *exec_path, const unsigned int mat_size, const unsigned int user_defined)
{
    // allocate memory
    input.a = (float *)malloc(sizeof(float) * mat_size);
    input.b = (float *)malloc(sizeof(float) * mat_size);

    if (1 == user_defined) {
        // initialize diagonal and superdiagonal entries with random values
        srand(278217421);

        // srand( clock());
        for (unsigned int i = 0; i < mat_size; ++i) {
            input.a[i] = (float)(2.0 * (((double)rand() / (double)RAND_MAX) - 0.5));
            input.b[i] = (float)(2.0 * (((double)rand() / (double)RAND_MAX) - 0.5));
        }

        // the first element of s is used as padding on the device (thus the
        // whole vector is copied to the device but the kernels are launched
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/main.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/eigenvalues/main.cu:109-138
```cuda
        char *sdiag_path = sdkFindFilePath("superdiagonal.dat", exec_path);
        assert(NULL != sdiag_path);
        sdkReadFile(sdiag_path, &(input.b), &input_data_size, false);

        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(diag_path);
        free(sdiag_path);
    }

    // allocate device memory for input
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc((void **)&(input.g_a), sizeof(float) * mat_size));
    checkCudaErrors(cudaMalloc((void **)&(input.g_b_raw), sizeof(float) * mat_size));

    // copy data to device
    // JP: `cudaMemcpy`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpy(input.g_a, input.a, sizeof(float) * mat_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(input.g_b_raw, input.b, sizeof(float) * mat_size, cudaMemcpyHostToDevice));

    input.g_b = input.g_b_raw + 1;
}

////////////////////////////////////////////////////////////////////////////////
//! Clean up input data, in particular allocated memory
//! @param input  handles to the input data
////////////////////////////////////////////////////////////////////////////////
void cleanupInputData(InputData &input)
{
    freePtr(input.a);
    freePtr(input.b);
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/main.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `matlab.cpp`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/matlab.cpp:30-71
```cpp
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

// includes, projcet
#include "matlab.h"

// namespace, unnamed
namespace {
} // namespace

///////////////////////////////////////////////////////////////////////////////
//! Write a tridiagonal, symmetric matrix in vector representation and
//! it's eigenvalues
//! @param  filename  name of output file
//! @param  d  diagonal entries of the matrix
//! @param  s  superdiagonal entries of the matrix (len = n - 1)
//! @param  eigenvals  eigenvalues of the matrix
//! @param  indices  vector of len n containing the position of the eigenvalues
//!                  if these are sorted in ascending order
//! @param  n  size of the matrix
///////////////////////////////////////////////////////////////////////////////
void writeTridiagSymMatlab(const char *filename, float *d, float *s, float *eigenvals, const unsigned int n)
{
    std::ofstream file(filename, std::ios::out);

    // write diagonal entries
    writeVectorMatlab(file, "d", d, n);

    // write superdiagonal entries
    writeVectorMatlab(file, "s", s, n - 1);

    // write eigenvalues
    writeVectorMatlab(file, "eigvals", eigenvals, n);

    file.close();
}
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/matlab.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `matlab.h`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/matlab.h:33-51
```cpp
#ifndef _MATLAB_H_
#define _MATLAB_H_

// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project

////////////////////////////////////////////////////////////////////////////////
//! Write a tridiagonal, symmetric matrix in vector representation and
//! it's eigenvalues
//! @param  filename  name of output file
//! @param  d  diagonal entries of the matrix
//! @param  s  superdiagonal entries of the matrix (len = n - 1)
//! @param  eigenvals  eigenvalues of the matrix
//! @param  indices  vector of len n containing the position of the eigenvalues
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/matlab.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `structs.h`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/structs.h:31-49
```cpp
#ifndef _STRUCTS_H_
#define _STRUCTS_H_

struct InputData
{
    //! host side representation of diagonal
    float *a;
    //! host side representation superdiagonal
    float *b;

    //! device side representation of diagonal
    float *g_a;
    //! device side representation of superdiagonal
    float *g_b;
    //! helper variable pointing to the mem allocated for g_b which provides
    //! space for one additional element of padding at the beginning
    float *g_b_raw;
};

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/structs.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `util.h`

Source: cpp/2_Concepts_and_Techniques/eigenvalues/util.h:31-55
```cpp
#ifndef _UTIL_H_
#define _UTIL_H_

////////////////////////////////////////////////////////////////////////////////
//! Safely free() for pointer
////////////////////////////////////////////////////////////////////////////////
template <class T> inline void freePtr(T *&ptr)
{
    if (NULL != ptr) {
        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(ptr);
        ptr = NULL;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Minimum
////////////////////////////////////////////////////////////////////////////////
template <class T>
#ifdef __CUDACC__
__host__
    __device__
#endif
        T
        min(const T &lhs, const T &rhs)
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/eigenvalues/util.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `__shared__` | block 内共有 memory または同期境界です。producer/consumer の順序を確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaMemcpyHostToDevice` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaMemcpyDeviceToHost` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `atomicExch` | 複数 thread が同じ address を更新する箇所です。競合と順序の意味を確認します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `Program` | 実行時 compile/link の境界です。log、module、kernel name の対応を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。
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
cmake --build build --target eigenvalues
ctest --test-dir build -R eigenvalues
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
