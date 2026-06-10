# matrixMulDrv - Matrix Multiplication (CUDA Driver API Version) - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample implements matrix multiplication and uses the new CUDA 4.0 kernel launch Driver API. It has been written for clarity of exposition to illustrate various CUDA programming principles, not with the goal of providing the most performant generic kernel for matrix multiplication. CUBLAS provides high-performance matrix multiplication.

CUDA Driver API, Matrix Multiply

Original README headings: `matrixMulDrv - Matrix Multiplication (CUDA Driver API Version)`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/0_Introduction/matrixMulDrv` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `matrixMulDrv` as a focused example of the CUDA concepts used in `cpp/0_Introduction/matrixMulDrv`.
> **日本語**
> この sample の目的は、`matrixMulDrv` の小さな実装を通して Runtime, Driver, And NVRTC, CUDA Libraries, Shared Memory, Synchronization And Atomics, Performance を具体的に追うことです。
>
> **学習メモ**
> 最初に `matrixMul.h, matrixMulDrv.cpp, matrixMul_kernel.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `matrixMul.h`: Host/device declarations, helper types, constants, or library wrappers.
- `matrixMulDrv.cpp`: Host-side setup, API calls, validation, and cleanup.
- `matrixMul_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `matrixMul.h` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Create library handles, descriptors, plans, or workspaces before the library call.
- Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.
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

- `matrixMul.h`: focus on control flow and helper functions.
- `matrixMulDrv.cpp`: focus on `CUDA`, `CUdeviceptr`, `cuDevice`, `launch`, `CUfunction`.
- `matrixMul_kernel.cu`: focus on `blockIdx`, `__shared__`, `threadIdx`, `__syncthreads`, `launch`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/0_Introduction/matrixMulDrv/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(matrixMulDrv LANGUAGES C CXX CUDA)

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

> JP: この抜粋は `cpp/0_Introduction/matrixMulDrv/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `matrixMul.h`

Source: cpp/0_Introduction/matrixMulDrv/matrixMul.h:29-41
```cpp
#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA (4 * block_size) // Matrix A width
#define HA (6 * block_size) // Matrix A height
#define WB (4 * block_size) // Matrix B width
#define HB WA               // Matrix B height
#define WC WB               // Matrix C width
#define HC HA               // Matrix C height

#endif // _MATRIXMUL_H_
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDrv/matrixMul.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `matrixMulDrv.cpp`

Source: cpp/0_Introduction/matrixMulDrv/matrixMulDrv.cpp:49-67
```cpp
#include <builtin_types.h>
#include <cstring>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project, CUDA
#include <cstring>
#include <cuda.h>
#include <helper_cuda_drvapi.h>
#include <helper_image.h>
#include <helper_string.h>
#include <helper_timer.h>
#include <iostream>
#include <string>

#include "matrixMul.h"
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDrv/matrixMulDrv.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/0_Introduction/matrixMulDrv/matrixMulDrv.cpp:100-119
```cpp
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("[ %s ]\n", sSDKsample);

    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv)
{
    // initialize CUDA
    // JP: この anchor では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
    CUfunction matrixMul  = NULL;
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDrv/matrixMulDrv.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/0_Introduction/matrixMulDrv/matrixMulDrv.cpp:125-144
```cpp
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A     = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float       *h_A        = reinterpret_cast<float *>(malloc(mem_size_A));
    unsigned int size_B     = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float       *h_B        = reinterpret_cast<float *>(malloc(mem_size_B));

    // initialize host memory
    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // allocate device memory
    CUdeviceptr d_A;
    // JP: `cuMemAlloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cuMemAlloc(&d_A, mem_size_A));
    CUdeviceptr d_B;
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDrv/matrixMulDrv.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/0_Introduction/matrixMulDrv/matrixMulDrv.cpp:178-197
```cpp
        // Launching (simplier method)
        size_t Matrix_Width_A = (size_t)WA;
        size_t Matrix_Width_B = (size_t)WB;
        void  *args[5]        = {&d_C, &d_A, &d_B, &Matrix_Width_A, &Matrix_Width_B};
        // new CUDA 4.0 Driver API Kernel launch call
        // JP: `cuLaunchKernel`: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
        checkCudaErrors(cuLaunchKernel(matrixMul,
                                       grid.x,
                                       grid.y,
                                       grid.z,
                                       block.x,
                                       block.y,
                                       block.z,
                                       2 * block_size * block_size * sizeof(float),
                                       NULL,
                                       args,
                                       NULL));
    }
    else {
        // This is the new CUDA 4.0 API for Kernel Parameter passing and Kernel
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDrv/matrixMulDrv.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `matrixMul_kernel.cu`

Source: cpp/0_Introduction/matrixMulDrv/matrixMul_kernel.cu:33-63
```cuda
#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
template <int block_size, typename size_type>
__device__ void matrixMul(float *C, float *A, float *B, size_type wA, size_type wB)
{
    // Block index
    // JP: `blockIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    size_type bx = blockIdx.x;
    size_type by = blockIdx.y;

    // Thread index
    size_type tx = threadIdx.x;
    size_type ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    size_type aBegin = wA * block_size * by;

    // Index of the last sub-matrix of A processed by the block
    size_type aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDrv/matrixMul_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/0_Introduction/matrixMulDrv/matrixMul_kernel.cu:91-110
```cuda
        AS(ty, tx) = A[a + wA * ty + tx];
        BS(ty, tx) = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (size_type k = 0; k < block_size; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
        __syncthreads();
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDrv/matrixMul_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `CUdeviceptr` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cuDevice` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuMemAlloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cuMemcpyHtoD` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cuLaunchKernel` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuMemFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `CUfunction` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuModule` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuFunction` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuDeviceGetAttribute` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUBLAS` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuContext` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuMemcpyDtoH` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- library sample では handle、descriptor、plan、workspace が GPU work の外側の resource です。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
- Runtime/Driver/NVRTC sample では、compile/load した module と launch する kernel 名の対応が重要です。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。
- 同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Sample-Specific Notes

- matrixMul 系では global memory の値を shared memory tile に移し、barrier 後に再利用します。性能の主役は arithmetic だけでなく memory reuse です。
- cuBLAS/Tensor Core 版がある場合は、同じ数学でも API、data layout、precision、workspace の責任分担が変わります。
- NVRTC/Driver/JIT 系では、compile/load/link と kernel launch が別の段階です。compile log、module、function handle、launch parameter の対応が重要です。
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
cmake --build build --target matrixMulDrv
ctest --test-dir build -R matrixMulDrv
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
- leading dimension、stride、descriptor、workspace size を host 配列の見た目だけで判断する。
- JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `CUdeviceptr` の直前と直後で、どの memory/resource が有効になったかをメモする。
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

- [Runtime, Driver, And NVRTC](../../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [CUDA Libraries](../../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。
- [Shared Memory](../../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
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
