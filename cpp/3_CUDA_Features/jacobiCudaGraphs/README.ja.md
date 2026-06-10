# jacobiCudaGraphs - Jacobi CUDA Graphs - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

Demonstrates Instantiated CUDA Graph Update with Jacobi Iterative Method using cudaGraphExecKernelNodeSetParams() and cudaGraphExecUpdate() approach.

CUDA Graphs, Stream Capture, Instantiated CUDA Graph Update, Cooperative Groups

Original README headings: `jacobiCudaGraphs - Jacobi CUDA Graphs`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/3_CUDA_Features/jacobiCudaGraphs` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `jacobiCudaGraphs` as a focused example of the CUDA concepts used in `cpp/3_CUDA_Features/jacobiCudaGraphs`.
> **日本語**
> この sample の目的は、`jacobiCudaGraphs` の小さな実装を通して CUDA Graphs, Cooperative Groups, Shared Memory, Streams And Events, Synchronization And Atomics を具体的に追うことです。
>
> **学習メモ**
> 最初に `jacobi.cu, jacobi.h, main.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `jacobi.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `jacobi.h`: Host/device declarations, helper types, constants, or library wrappers.
- `main.cpp`: Host-side setup, API calls, validation, and cleanup.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `jacobi.cu` first and locate the host-side setup or Python entry point.
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

- `jacobi.cu`: focus on `launch`, `threadIdx`, `blockDim`, `cudaMemcpyDeviceToHost`, `CUDA`.
- `jacobi.h`: focus on control flow and helper functions.
- `main.cpp`: focus on `CUDA`, `cudaStream_t`, `cudaMalloc`, `cudaFree`, `cudaMallocHost`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/3_CUDA_Features/jacobiCudaGraphs/CMakeLists.txt:1-41
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(jacobiCudaGraphs LANGUAGES C CXX CUDA)

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
# Add target for jacobiCudaGraphs
add_executable(jacobiCudaGraphs jacobi.cu main.cpp)

target_compile_options(jacobiCudaGraphs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(jacobiCudaGraphs PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(jacobiCudaGraphs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(jacobiCudaGraphs PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/3_CUDA_Features/jacobiCudaGraphs/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `jacobi.cu`

Source: cpp/3_CUDA_Features/jacobiCudaGraphs/jacobi.cu:29-47
```cuda
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <vector>

#include "jacobi.h"

namespace cg = cooperative_groups;

// 8 Rows of square-matrix A processed by each CTA.
// This can be max 32 and only power of 2 (i.e., 2/4/8/16/32).
#define ROWS_PER_CTA 8

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int  old            = *address_as_ull, assumed;
```

> JP: この抜粋は `cpp/3_CUDA_Features/jacobiCudaGraphs/jacobi.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/jacobiCudaGraphs/jacobi.cu:62-81
```cuda
JacobiMethod(const float *A, const double *b, const float conv_threshold, double *x, double *x_new, double *sum)
{
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block  cta = cg::this_thread_block();
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    __shared__ double x_shared[N_ROWS]; // N_ROWS == n
    __shared__ double b_shared[ROWS_PER_CTA + 1];

    for (int i = threadIdx.x; i < N_ROWS; i += blockDim.x) {
        x_shared[i] = x[i];
    }

    if (threadIdx.x < ROWS_PER_CTA) {
        int k = threadIdx.x;
        for (int i = k + (blockIdx.x * ROWS_PER_CTA); (k < ROWS_PER_CTA) && (i < N_ROWS);
             k += ROWS_PER_CTA, i += ROWS_PER_CTA) {
            b_shared[i % (ROWS_PER_CTA + 1)] = b[i];
        }
    }
```

> JP: この抜粋は `cpp/3_CUDA_Features/jacobiCudaGraphs/jacobi.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/jacobiCudaGraphs/jacobi.cu:195-219
```cuda
    cudaGraph_t     graph;
    cudaGraphExec_t graphExec = NULL;

    double  sum   = 0.0;
    double *d_sum = NULL;
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc(&d_sum, sizeof(double)));

    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t              memcpyNode, jacobiKernelNode, memsetNode;
    cudaMemcpy3DParms            memcpyParams = {0};
    cudaMemsetParams             memsetParams = {0};

    memsetParams.dst   = (void *)d_sum;
    memsetParams.value = 0;
    memsetParams.pitch = 0;
    // elementSize can be max 4 bytes, so we take sizeof(float) and width=2
    memsetParams.elementSize = sizeof(float);
    memsetParams.width       = 2;
    memsetParams.height      = 1;

    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphCreate(&graph, 0));
    checkCudaErrors(cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));
    nodeDependencies.push_back(memsetNode);
```

> JP: この抜粋は `cpp/3_CUDA_Features/jacobiCudaGraphs/jacobi.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/jacobiCudaGraphs/jacobi.cu:266-285
```cuda
        // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
        checkCudaErrors(cudaGraphExecKernelNodeSetParams(
            graphExec, jacobiKernelNode, ((k & 1) == 0) ? &NodeParams0 : &NodeParams1));
        checkCudaErrors(cudaGraphLaunch(graphExec, stream));
        // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
        checkCudaErrors(cudaStreamSynchronize(stream));

        if (sum <= conv_threshold) {
            // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
            checkCudaErrors(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
            nblocks.x            = (N_ROWS / nthreads.x) + 1;
            size_t sharedMemSize = ((nthreads.x / 32) + 1) * sizeof(double);
            if ((k & 1) == 0) {
                // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
                finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x_new, d_sum);
            }
            else {
                finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x, d_sum);
            }

```

> JP: この抜粋は `cpp/3_CUDA_Features/jacobiCudaGraphs/jacobi.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `jacobi.h`

Source: cpp/3_CUDA_Features/jacobiCudaGraphs/jacobi.h:29-34
```cpp
#ifndef JACOBI_H
#define JACOBI_H

#define N_ROWS 512

#endif
```

> JP: この抜粋は `cpp/3_CUDA_Features/jacobiCudaGraphs/jacobi.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `main.cpp`

Source: cpp/3_CUDA_Features/jacobiCudaGraphs/main.cpp:41-59
```cpp
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "jacobi.h"

// Run the Jacobi method for A*x = b on GPU with CUDA Graph -
// cudaGraphExecKernelNodeSetParams().
extern double JacobiMethodGpuCudaGraphExecKernelSetParams(const float  *A,
                                                          const double *b,
                                                          const float   conv_threshold,
                                                          const int     max_iter,
                                                          double       *x,
                                                          double       *x_new,
                                                          // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
                                                          cudaStream_t  stream);
```

> JP: この抜粋は `cpp/3_CUDA_Features/jacobiCudaGraphs/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/jacobiCudaGraphs/main.cpp:64-104
```cpp
                                                 const double *b,
                                                 const float   conv_threshold,
                                                 const int     max_iter,
                                                 double       *x,
                                                 double       *x_new,
                                                 // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
                                                 cudaStream_t  stream);

// Run the Jacobi method for A*x = b on GPU without CUDA Graph.
extern double JacobiMethodGpu(const float  *A,
                              const double *b,
                              const float   conv_threshold,
                              const int     max_iter,
                              double       *x,
                              double       *x_new,
                              // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
                              cudaStream_t  stream);

// creates N_ROWS x N_ROWS matrix A with N_ROWS+1 on the diagonal and 1
// elsewhere. The elements of the right hand side b all equal 2*n, hence the
// exact solution x to A*x = b is a vector of ones.
void createLinearSystem(float *A, double *b);

// Run the Jacobi method for A*x = b on CPU.
void JacobiMethodCPU(float *A, double *b, float conv_threshold, int max_iter, int *numit, double *x);

int main(int argc, char **argv)
{
    if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
        printf("Command line: jacobiCudaGraphs [-option]\n");
        printf("Valid options:\n");
        printf("-gpumethod=<0,1 or 2>  : 0 - [Default] "
               "JacobiMethodGpuCudaGraphExecKernelSetParams\n");
        printf("                       : 1 - JacobiMethodGpuCudaGraphExecUpdate\n");
        printf("                       : 2 - JacobiMethodGpu - Non CUDA Graph\n");
        printf("-device=device_num     : cuda device id");
        printf("-help         : Output a help message\n");
        exit(EXIT_SUCCESS);
    }

    int gpumethod = 0;
```

> JP: この抜粋は `cpp/3_CUDA_Features/jacobiCudaGraphs/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/jacobiCudaGraphs/main.cpp:113-132
```cpp

    int dev = findCudaDevice(argc, (const char **)argv);

    double *b = NULL;
    float  *A = NULL;
    // JP: `cudaMallocHost`: page-locked host memory は DMA/async copy を安定させます。通常の free ではなく対応する CUDA API で解放します。
    checkCudaErrors(cudaMallocHost(&b, N_ROWS * sizeof(double)));
    memset(b, 0, N_ROWS * sizeof(double));
    checkCudaErrors(cudaMallocHost(&A, N_ROWS * N_ROWS * sizeof(float)));
    memset(A, 0, N_ROWS * N_ROWS * sizeof(float));

    createLinearSystem(A, b);
    double *x = NULL;
    // start with array of all zeroes
    x = (double *)calloc(N_ROWS, sizeof(double));

    float conv_threshold = 1.0e-2;
    int   max_iter       = 4 * N_ROWS * N_ROWS;
    int   cnt            = 0;

```

> JP: この抜粋は `cpp/3_CUDA_Features/jacobiCudaGraphs/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/jacobiCudaGraphs/main.cpp:156-175
```cpp
    checkCudaErrors(cudaMalloc(&d_b, sizeof(double) * N_ROWS));
    checkCudaErrors(cudaMalloc(&d_A, sizeof(float) * N_ROWS * N_ROWS));
    checkCudaErrors(cudaMalloc(&d_x, sizeof(double) * N_ROWS));
    checkCudaErrors(cudaMalloc(&d_x_new, sizeof(double) * N_ROWS));

    // JP: `cudaMemsetAsync`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemsetAsync(d_x, 0, sizeof(double) * N_ROWS, stream1));
    checkCudaErrors(cudaMemsetAsync(d_x_new, 0, sizeof(double) * N_ROWS, stream1));
    checkCudaErrors(cudaMemcpyAsync(d_A, A, sizeof(float) * N_ROWS * N_ROWS, cudaMemcpyHostToDevice, stream1));
    checkCudaErrors(cudaMemcpyAsync(d_b, b, sizeof(double) * N_ROWS, cudaMemcpyHostToDevice, stream1));

    sdkCreateTimer(&timerGpu);
    sdkStartTimer(&timerGpu);

    double sumGPU = 0.0;
    if (gpumethod == 0) {
        sumGPU = JacobiMethodGpuCudaGraphExecKernelSetParams(d_A, d_b, conv_threshold, max_iter, d_x, d_x_new, stream1);
    }
    else if (gpumethod == 1) {
        sumGPU = JacobiMethodGpuCudaGraphExecUpdate(d_A, d_b, conv_threshold, max_iter, d_x, d_x_new, stream1);
```

> JP: この抜粋は `cpp/3_CUDA_Features/jacobiCudaGraphs/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaMemsetAsync` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaStream_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaMemcpyAsync` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaStreamSynchronize` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaMemcpyDeviceToHost` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaGraphExecKernelNodeSetParams` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphExecUpdate` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphInstantiate` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaMallocHost` | pinned host memory を作り、async copy や DMA の前提を作る API です。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- Cooperative Groups は協調する単位を明示します。grid/block/warp のどれを同期しているかを確認します。
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

## Sample-Specific Notes

- graph sample では、最初の capture/instantiate と繰り返し launch を分けて読みます。launch overhead を減らす代わりに依存関係と buffer lifetime が固定されます。

> **日本語**
> この section は同じ template ではなく、sample 名、path、検出した API から読みどころを絞っています。
>
> **学習メモ**
> 似た名前の sample は Runtime 版、Driver 版、NVRTC 版、library 版の違いを比較すると学習効果が高くなります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target jacobiCudaGraphs
ctest --test-dir build -R jacobiCudaGraphs
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
- different stream 間に依存があるのに event や explicit sync を置かない。
- graph capture 後に buffer lifetime や node dependency が変わったことを見落とす。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `launch` の直前と直後で、どの memory/resource が有効になったかをメモする。
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
- [Cooperative Groups](../../../docs_ja/themes/cooperative_groups.md): block/grid/warp 単位の協調と同期を読むための基礎です。
- [Shared Memory](../../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
