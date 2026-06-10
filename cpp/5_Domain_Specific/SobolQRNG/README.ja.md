# SobolQRNG - Sobol Quasirandom Number Generator - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample implements Sobol Quasirandom Sequence Generator.

Computational Finance

Original README headings: `SobolQRNG - Sobol Quasirandom Number Generator`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/SobolQRNG` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `SobolQRNG` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/SobolQRNG`.
> **日本語**
> この sample の目的は、`SobolQRNG` の小さな実装を通して Shared Memory, Memory, Kernel Launch And Indexing, Execution Model, Debugging, Profiling, And Testing を具体的に追うことです。
>
> **学習メモ**
> 最初に `sobol.cpp, sobol.h, sobol_gold.cpp, sobol_gold.h, sobol_gpu.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `sobol.cpp`: Host-side setup, API calls, validation, and cleanup.
- `sobol.h`: Host/device declarations, helper types, constants, or library wrappers.
- `sobol_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `sobol_gold.h`: Host/device declarations, helper types, constants, or library wrappers.
- `sobol_gpu.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `sobol_gpu.h`: Host/device declarations, helper types, constants, or library wrappers.
- `sobol_primitives.cpp`: Host-side setup, API calls, validation, and cleanup.
- `sobol_primitives.h`: Host/device declarations, helper types, constants, or library wrappers.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `sobol.cpp` first and locate the host-side setup or Python entry point.
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

- `sobol.cpp`: focus on `cudaResult`, `CUDA`, `cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`.
- `sobol.h`: focus on `CUDA`.
- `sobol_gold.cpp`: focus on `CUDA`.
- `sobol_gold.h`: focus on `CUDA`.
- `sobol_gpu.cu`: focus on `threadIdx`, `blockIdx`, `launch`, `__shared__`, `blockDim`.
- `sobol_gpu.h`: focus on `CUDA`.
- `sobol_primitives.cpp`: focus on `CUDA`.
- `sobol_primitives.h`: focus on `CUDA`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/SobolQRNG/CMakeLists.txt:1-41
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(SobolQRNG LANGUAGES C CXX CUDA)

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
# Add target for SobolQRNG
add_executable(SobolQRNG sobol_gold.cpp sobol_gpu.cu sobol_primitives.cpp sobol.cpp)

target_compile_options(SobolQRNG PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(SobolQRNG PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(SobolQRNG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(SobolQRNG PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/5_Domain_Specific/SobolQRNG/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `sobol.cpp`

Source: cpp/5_Domain_Specific/SobolQRNG/sobol.cpp:57-75
```cpp
#include "sobol.h"

#include <cuda_runtime.h>     // CUDA Runtime Functions
#include <helper_cuda.h>      // helper functions for CUDA error checking and initialization
#include <helper_functions.h> // helper functions
#include <iostream>
#include <math.h>
#include <stdexcept>

#include "sobol_gold.h"
#include "sobol_gpu.h"

#define L1ERROR_TOLERANCE (1e-6)

const char *sSDKsample = "Sobol Quasi-Random Number Generator";

void printHelp(int argc, char *argv[])
{
    if (argc > 0) {
```

> JP: この抜粋は `cpp/5_Domain_Specific/SobolQRNG/sobol.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/SobolQRNG/sobol.cpp:84-103
```cpp
    std::cout << "\t--dimensions=N  specify number of dimensions (required)\n";
    std::cout << "\t                Each vector will consist of N components\n\n";
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    bool ok = true;

    // We will generate n_vectors vectors of n_dimensions numbers
    int n_vectors    = 100000;
    int n_dimensions = 100;

    printf("%s Starting...\n\n", sSDKsample);

    // Print help if requested
    if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
        printHelp(argc, argv);
        return 0;
    }
```

> JP: この抜粋は `cpp/5_Domain_Specific/SobolQRNG/sobol.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/SobolQRNG/sobol.cpp:157-176
```cpp
    unsigned int *h_directions = 0;
    float        *h_outputCPU  = 0;
    float        *h_outputGPU  = 0;

    try {
        h_directions = new unsigned int[n_dimensions * n_directions];
        h_outputCPU  = new float[n_vectors * n_dimensions];
        h_outputGPU  = new float[n_vectors * n_dimensions];
    }
    catch (std::exception e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        std::cerr << "Unable to allocate CPU memory (try running with fewer "
                     "vectors/dimensions)"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Allocating GPU memory..." << std::endl;
    unsigned int *d_directions;
    float        *d_output;
```

> JP: この抜粋は `cpp/5_Domain_Specific/SobolQRNG/sobol.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/SobolQRNG/sobol.cpp:202-224
```cpp
    std::cout << "Initializing direction numbers..." << std::endl;
    initSobolDirectionVectors(n_dimensions, h_directions);

    // Copy the direction numbers to the device
    std::cout << "Copying direction numbers to device..." << std::endl;
    // JP: `cudaMemcpy`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpy(
        d_directions, h_directions, n_dimensions * n_directions * sizeof(unsigned int), cudaMemcpyHostToDevice));
    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaDeviceSynchronize());

    // Execute the QRNG on the device
    std::cout << "Executing QRNG on GPU..." << std::endl;
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    sobolGPU(n_vectors, n_dimensions, d_directions, d_output);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    time = sdkGetTimerValue(&hTimer);

    if (time < 1e-6) {
        std::cout << "Gsamples/s: problem size too small to measure, try "
                     "increasing number of vectors or dimensions"
```

> JP: この抜粋は `cpp/5_Domain_Specific/SobolQRNG/sobol.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `sobol.h`

Source: cpp/5_Domain_Specific/SobolQRNG/sobol.h:57-63
```cpp
#ifndef SOBOL_H
#define SOBOL_H

// Number of direction vectors is fixed to 32
#define n_directions 32

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/SobolQRNG/sobol.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `sobol_gold.cpp`

Source: cpp/5_Domain_Specific/SobolQRNG/sobol_gold.cpp:57-75
```cpp
#include "sobol_gold.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sobol.h"
#include "sobol_primitives.h"

#define k_2powneg32 2.3283064E-10F

// Windows does not provide ffs (find first set) so here is a
// fairly simple implementation.
// WIN32 is defined on 32 and 64 bit Windows
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
int ffs(const unsigned int &i)
{
    unsigned int v = i;
```

> JP: この抜粋は `cpp/5_Domain_Specific/SobolQRNG/sobol_gold.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `sobol_gold.h`

Source: cpp/5_Domain_Specific/SobolQRNG/sobol_gold.h:58-64
```cpp
#ifndef SOBOL_GOLD_H
#define SOBOL_GOLD_H

void initSobolDirectionVectors(int n_dimensions, unsigned int *directions);
void sobolCPU(int n_vectors, int n_dimensions, unsigned int *directions, float *output);

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/SobolQRNG/sobol_gold.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `sobol_gpu.cu`

Source: cpp/5_Domain_Specific/SobolQRNG/sobol_gpu.cu:58-87
```cuda
#include <cooperative_groups.h>

#include "sobol.h"
#include "sobol_gpu.h"

namespace cg = cooperative_groups;
#include <helper_cuda.h>

#define k_2powneg32 2.3283064E-10F

__global__ void sobolGPU_kernel(unsigned n_vectors, unsigned n_dimensions, unsigned *d_directions, float *d_output)
{
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block        cta = cg::this_thread_block();
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    __shared__ unsigned int v[n_directions];

    // Offset into the correct dimension as specified by the
    // block y coordinate
    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    d_directions = d_directions + n_directions * blockIdx.y;
    d_output     = d_output + n_vectors * blockIdx.y;

    // Copy the direction numbers for this dimension into shared
    // memory - there are only 32 direction numbers so only the
    // first 32 (n_directions) threads need participate.
    if (threadIdx.x < n_directions) {
        v[threadIdx.x] = d_directions[threadIdx.x];
    }
```

> JP: この抜粋は `cpp/5_Domain_Specific/SobolQRNG/sobol_gpu.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `sobol_gpu.h`

Source: cpp/5_Domain_Specific/SobolQRNG/sobol_gpu.h:58-63
```cpp
#ifndef SOBOL_GPU_H
#define SOBOL_GPU_H

extern "C" void sobolGPU(int n_vectors, int n_dimensions, unsigned int *d_directions, float *d_output);

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/SobolQRNG/sobol_gpu.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `sobol_primitives.cpp`

Source: cpp/5_Domain_Specific/SobolQRNG/sobol_primitives.cpp:58-76
```cpp
#include "sobol_primitives.h"

// Each primitive is stored as a struct where
//  dimension is the dimension number of the polynomial (unused)
//  degree is the degree of the polynomial
//  a is a binary word representing the coefficients
//  m is the array of m values

// The primitives are based on those generated by Stephen Joe and
// Frances Kuo in the joe-kuo-6.10200 set.
// c.f. http://web.maths.unsw.edu.au/~fkuo/sobol/index.html
const struct primitive sobol_primitives[] = {
    // First dimension is a special case so this entry is actually ignored
    {1, 0, 0, {}},
    {2, 1, 0, {1}},
    {3, 2, 1, {1, 3}},
    {4, 3, 1, {1, 3, 1}},
    {5, 3, 2, {1, 1, 1}},
    {6, 4, 1, {1, 1, 3, 3}},
```

> JP: この抜粋は `cpp/5_Domain_Specific/SobolQRNG/sobol_primitives.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `sobol_primitives.h`

Source: cpp/5_Domain_Specific/SobolQRNG/sobol_primitives.h:58-78
```cpp
#ifndef SOBOL_PRIMITIVES_H
#define SOBOL_PRIMITIVES_H

#define max_m 17

// Each primitive is stored as a struct where
//  dimension is the dimension number of the polynomial (unused)
//  degree is the degree of the polynomial
//  a is a binary word representing the coefficients
//  m is the array of m values
struct primitive
{
    unsigned int dimension;
    unsigned int degree;
    unsigned int a;
    unsigned int m[max_m];
};

extern const struct primitive sobol_primitives[];

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/SobolQRNG/sobol_primitives.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaResult` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGetErrorString` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaGetDevice` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGetDeviceProperties` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `__shared__` | block 内共有 memory または同期境界です。producer/consumer の順序を確認します。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaError_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

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
cmake --build build --target SobolQRNG
ctest --test-dir build -R SobolQRNG
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

- `cudaResult` の直前と直後で、どの memory/resource が有効になったかをメモする。
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
