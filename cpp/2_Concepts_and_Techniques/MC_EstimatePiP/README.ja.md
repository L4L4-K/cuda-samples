# MC_EstimatePiP - Monte Carlo Estimation of Pi (batch PRNG) - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample uses Monte Carlo simulation for Estimation of Pi (using batch PRNG).  This sample also uses the NVIDIA CURAND library.

Random Number Generator, Computational Finance, CURAND Library

Original README headings: `MC_EstimatePiP - Monte Carlo Estimation of Pi (batch PRNG)`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/2_Concepts_and_Techniques/MC_EstimatePiP` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `MC_EstimatePiP` as a focused example of the CUDA concepts used in `cpp/2_Concepts_and_Techniques/MC_EstimatePiP`.
> **日本語**
> この sample の目的は、`MC_EstimatePiP` の小さな実装を通して CUDA Libraries, Shared Memory, Memory, Kernel Launch And Indexing, Execution Model を具体的に追うことです。
>
> **学習メモ**
> 最初に `cudasharedmem.h, piestimator.h, test.h, main.cpp, piestimator.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- CUDA library components used by this sample, such as cuBLAS, cuFFT, cuSolver, NPP, CUB, or nvJPEG

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
- `inc/cudasharedmem.h`: Host/device declarations, helper types, constants, or library wrappers.
- `inc/piestimator.h`: Host/device declarations, helper types, constants, or library wrappers.
- `inc/test.h`: Host/device declarations, helper types, constants, or library wrappers.
- `src/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `src/piestimator.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `src/test.cpp`: Host-side setup, API calls, validation, and cleanup.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `cudasharedmem.h` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Create library handles, descriptors, plans, or workspaces before the library call.
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

- `inc/cudasharedmem.h`: focus on `__shared__`, `CUDASHAREDMEM_H`.
- `inc/piestimator.h`: focus on control flow and helper functions.
- `inc/test.h`: focus on control flow and helper functions.
- `src/main.cpp`: focus on `cudaResult`, `cudaSuccess`, `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaError_t`.
- `src/piestimator.cu`: focus on `cudaResult`, `curandResult`, `cudaSuccess`, `cudaGetErrorString`, `threadIdx`.
- `src/test.cpp`: focus on `cudaResult`, `cudaDeviceProp`, `cudaError_t`, `cudaGetDeviceProperties`, `cudaSuccess`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/2_Concepts_and_Techniques/MC_EstimatePiP/CMakeLists.txt:1-47
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(MC_EstimatePiP LANGUAGES C CXX CUDA)

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
# Add target for MC_EstimatePiP
add_executable(MC_EstimatePiP src/main.cpp src/piestimator.cu src/test.cpp)

target_compile_options(MC_EstimatePiP PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(MC_EstimatePiP PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(MC_EstimatePiP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(MC_EstimatePiP PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(MC_EstimatePiP PUBLIC
    # JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
    CUDA::curand
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `inc/cudasharedmem.h`

Source: cpp/2_Concepts_and_Techniques/MC_EstimatePiP/inc/cudasharedmem.h:29-60
```cpp
#ifndef CUDASHAREDMEM_H
#define CUDASHAREDMEM_H

//****************************************************************************
// Because dynamically sized shared memory arrays are declared "extern",
// we can't templatize them directly.  To get around this, we declare a
// simple wrapper struct that will declare the extern array with a different
// name depending on the type.  This avoids compiler errors about duplicate
// definitions.
//
// To use dynamically allocated shared memory in a templatized __global__ or
// __device__ function, just replace code like this:
//      template<class T>
//      __global__ void
//      foo( T* g_idata, T* g_odata)
//      {
//          // Shared mem size is determined by the host app at run time
//          extern __shared__  T sdata[];
//          ...
//          x = sdata[i];
//          sdata[i] = x;
//          ...
//      }
//
// With this:
//      template<class T>
//      __global__ void
//      foo( T* g_idata, T* g_odata)
//      {
//          // Shared mem size is determined by the host app at run time
//          SharedMemory<T> sdata;
//          ...
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/inc/cudasharedmem.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `inc/piestimator.h`

Source: cpp/2_Concepts_and_Techniques/MC_EstimatePiP/inc/piestimator.h:29-45
```cpp
#ifndef PIESTIMATOR_H
#define PIESTIMATOR_H

template <typename Real> class PiEstimator
{
public:
    PiEstimator(unsigned int numSims, unsigned int device, unsigned int threadBlockSize, unsigned int seed);
    Real operator()();

private:
    unsigned int m_seed;
    unsigned int m_numSims;
    unsigned int m_device;
    unsigned int m_threadBlockSize;
};

#endif
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/inc/piestimator.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `inc/test.h`

Source: cpp/2_Concepts_and_Techniques/MC_EstimatePiP/inc/test.h:29-61
```cpp
#ifndef TEST_H
#define TEST_H

template <typename Real> struct Test
{
    Test()
        : pass(false) {};

    int          device;
    unsigned int numSims;
    unsigned int threadBlockSize;
    unsigned int seed;

    bool   pass;
    double elapsedTime;

    bool operator()();
};

// Defaults are arbitrary to give sensible runtime
#define k_sims_min  100000
#define k_sims_max  10000000
#define k_sims_def  100000
#define k_sims_qa   100000
#define k_bsize_min 32
#define k_bsize_def 128
#define k_bsize_qa  128
#define k_seed_def  1234

// Target value
#define PI 3.14159265359

#endif
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/inc/test.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `src/main.cpp`

Source: cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/main.cpp:43-74
```cpp
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdexcept>

#include "../inc/test.h"

// SDK information
static const char *printfFile = "MonteCarloEstimatePiP.txt";

// Forward declarations
void                          showHelp(const int argc, const char **argv);
template <typename Real> void runTest(int argc, const char **argv);

int main(int argc, char **argv)
{
    using std::invalid_argument;
    using std::string;

    // Open the log file
    printf("Monte Carlo Estimate Pi (with batch PRNG)\n");
    printf("=========================================\n\n");

    // If help flag is set, display help and exit immediately
    if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
        printf("Displaying help on console\n");
        showHelp(argc, (const char **)argv);
        exit(EXIT_SUCCESS);
    }
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/main.cpp:80-99
```cpp
        if (getCmdLineArgumentString(argc, (const char **)argv, "precision", &value)) {
            // Check requested precision is valid
            string prec(value);

            // JP: validation: GPU result を CPU/reference と比較する検証地点です。失敗時は transfer、indexing、sync の順に疑います。
            if (prec.compare("single") == 0 || prec.compare("\"single\"") == 0) {
                runTest<float>(argc, (const char **)argv);
            }
            else if (prec.compare("double") == 0 || prec.compare("\"double\"") == 0) {
                runTest<double>(argc, (const char **)argv);
            }
            else {
                printf("specified precision (%s) is invalid, must be \"single\".\n", value);
                throw invalid_argument("precision");
            }
        }
        else {
            runTest<float>(argc, (const char **)argv);
        }
    }
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `src/piestimator.cu`

Source: cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/piestimator.cu:29-61
```cuda
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#include "../inc/piestimator.h"

namespace cg = cooperative_groups;
#include <curand.h>

using std::string;
using std::vector;

__device__ unsigned int reduce_sum(unsigned int in, cg::thread_block cta)
{
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    extern __shared__ unsigned int sdata[];

    // Perform first level of reduction:
    // - Write to shared memory
    // JP: `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    unsigned int ltid = threadIdx.x;

    sdata[ltid] = in;
    // JP: sync: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    cg::sync(cta);

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (ltid < s) {
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/piestimator.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/piestimator.cu:140-159
```cuda
        && (deviceProperties.major < 1 || (deviceProperties.major == 1 && deviceProperties.minor < 3))) {
        throw std::runtime_error("Device does not have double precision support");
    }

    // Attach to GPU
    cudaResult = cudaSetDevice(m_device);

    if (cudaResult != cudaSuccess) {
        string msg("Could not set CUDA device: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Determine how to divide the work between cores
    dim3 block;
    dim3 grid;
    block.x = m_threadBlockSize;
    grid.x  = (m_numSims + m_threadBlockSize - 1) / m_threadBlockSize;

    // Aim to launch around ten or more times as many blocks as there
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/piestimator.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/piestimator.cu:188-207
```cuda
    }

    // Allocate memory for points
    // Each simulation has two random numbers to give X and Y coordinate
    Real *d_points = 0;
    // JP: `cudaResult`, `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    cudaResult     = cudaMalloc((void **)&d_points, 2 * m_numSims * sizeof(Real));

    if (cudaResult != cudaSuccess) {
        string msg("Could not allocate memory on device for random numbers: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Allocate memory for result
    // Each thread block will produce one result
    unsigned int *d_results = 0;
    // JP: この anchor では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    cudaResult              = cudaMalloc((void **)&d_results, grid.x * sizeof(unsigned int));

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/piestimator.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/piestimator.cu:249-279
```cuda
    }

    curandResult = curandDestroyGenerator(prng);

    if (curandResult != CURAND_STATUS_SUCCESS) {
        string msg("Could not destroy pseudo-random number generator: ");
        msg += curandResult;
        throw std::runtime_error(msg);
    }

    // Count the points inside unit quarter-circle
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    computeValue<Real><<<grid, block, block.x * sizeof(unsigned int)>>>(d_results, d_points, m_numSims);

    // Copy partial results back
    vector<unsigned int> results(grid.x);
    // JP: `cudaResult`, `cudaMemcpy`, `cudaMemcpyDeviceToHost`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    cudaResult = cudaMemcpy(&results[0], d_results, grid.x * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    if (cudaResult != cudaSuccess) {
        string msg("Could not copy partial results to host: ");
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }

    // Complete sum-reduction on host
    Real value = static_cast<Real>(std::accumulate(results.begin(), results.end(), 0));

    // Determine the proportion of points inside the quarter-circle,
    // i.e. the area of the unit quarter-circle
    value /= m_numSims;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/piestimator.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `src/test.cpp`

Source: cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/test.cpp:29-47
```cpp
#include "../inc/test.h"

#include <cassert>
#include <cuda_runtime.h>
#include <helper_timer.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <typeinfo>

#include "../inc/piestimator.h"

template <typename Real> bool Test<Real>::operator()()
{
    using std::endl;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/test.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/test.cpp:67-86
```cpp
    sdkStartTimer(&timer);
    Real result = estimator();
    sdkStopTimer(&timer);
    elapsedTime = sdkGetAverageTimerValue(&timer) / 1000.0f;

    // Tolerance to compare result with expected
    // This is just to check that nothing has gone very wrong with the
    // test, the actual accuracy of the result depends on the number of
    // Monte Carlo trials
    const Real tolerance = static_cast<Real>(0.01);

    // Display results
    Real abserror = fabs(result - static_cast<float>(PI));
    Real relerror = abserror / static_cast<float>(PI);
    printf("Precision:      %s\n", (typeid(Real) == typeid(double)) ? "double" : "single");
    printf("Number of sims: %d\n", numSims);
    printf("Tolerance:      %e\n", tolerance);
    printf("GPU result:     %e\n", result);
    printf("Expected:       %e\n", PI);
    printf("Absolute error: %e\n", abserror);
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/test.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaResult` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `curandResult` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaGetErrorString` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGetDeviceProperties` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CURAND` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `__shared__` | block 内共有 memory または同期境界です。producer/consumer の順序を確認します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaGetDeviceCount` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `CURAND_STATUS_SUCCESS` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaError_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDeviceProp` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- library sample では handle、descriptor、plan、workspace が GPU work の外側の resource です。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。

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
cmake --build build --target MC_EstimatePiP
ctest --test-dir build -R MC_EstimatePiP
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

The sample validates the library result against a CPU/reference path or reports the documented success status.
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
- leading dimension、stride、descriptor、workspace size を host 配列の見た目だけで判断する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaResult` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- shared memory tile の producer、consumer、barrier を図にする。
- handle/descriptor/workspace の作成、利用、破棄を対応表にする。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [CUDA Libraries](../../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。
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
