# simpleDrvRuntime - Simple Driver-Runtime Interaction - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

A simple example which demonstrates how CUDA Driver and Runtime APIs can work together to load cuda fatbinary of vector add kernel and performing vector addition.

CUDA Driver API, CUDA Runtime API, Vector Addition

Original README headings: `simpleDrvRuntime - Simple Driver-Runtime Interaction`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/0_Introduction/simpleDrvRuntime` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `simpleDrvRuntime` as a focused example of the CUDA concepts used in `cpp/0_Introduction/simpleDrvRuntime`.
> **日本語**
> この sample の目的は、`simpleDrvRuntime` の小さな実装を通して Runtime, Driver, And NVRTC, Streams And Events, Performance, Memory, Kernel Launch And Indexing を具体的に追うことです。
>
> **学習メモ**
> 最初に `simpleDrvRuntime.cpp, vectorAdd_kernel.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `simpleDrvRuntime.cpp`: Host-side setup, API calls, validation, and cleanup.
- `vectorAdd_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `simpleDrvRuntime.cpp` first and locate the host-side setup or Python entry point.
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

- `simpleDrvRuntime.cpp`: focus on `cuContext`, `CUDA`, `cuModule`, `cudaMallocHost`, `cudaMalloc`.
- `vectorAdd_kernel.cu`: focus on `blockDim`, `blockIdx`, `threadIdx`, `launch`, `Device`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/0_Introduction/simpleDrvRuntime/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleDrvRuntime LANGUAGES C CXX CUDA)

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

> JP: この抜粋は `cpp/0_Introduction/simpleDrvRuntime/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `simpleDrvRuntime.cpp`

Source: cpp/0_Introduction/simpleDrvRuntime/simpleDrvRuntime.cpp:37-55
```cpp
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>

// includes, CUDA
#include <builtin_types.h>

using namespace std;

#ifndef FATBIN_FILE
#define FATBIN_FILE "vectorAdd_kernel64.fatbin"
#endif
```

> JP: この抜粋は `cpp/0_Introduction/simpleDrvRuntime/simpleDrvRuntime.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/0_Introduction/simpleDrvRuntime/simpleDrvRuntime.cpp:77-96
```cpp
}

#define checkCudaDrvErrors(val) check((val), #val, __FILE__, __LINE__)

// Host code
int main(int argc, char **argv)
{
    printf("simpleDrvRuntime..\n");
    int               N = 50000, devID = 0;
    size_t            size = N * sizeof(float);
    // JP: この連続する anchor 群では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
    CUdevice          cuDevice;
    CUfunction        vecAdd_kernel;
    CUmodule          cuModule = 0;
    CUcontext         cuContext;
    CUctxCreateParams ctxCreateParams = {};

    // Initialize
    checkCudaDrvErrors(cuInit(0));

```

> JP: この抜粋は `cpp/0_Introduction/simpleDrvRuntime/simpleDrvRuntime.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/0_Introduction/simpleDrvRuntime/simpleDrvRuntime.cpp:119-174
```cpp

    // Get function handle from module
    checkCudaDrvErrors(cuModuleGetFunction(&vecAdd_kernel, cuModule, "VecAdd_kernel"));

    // Allocate input vectors h_A and h_B in host memory
    // JP: `cudaMallocHost`: page-locked host memory は DMA/async copy を安定させます。通常の free ではなく対応する CUDA API で解放します。
    checkCudaErrors(cudaMallocHost(&h_A, size));
    checkCudaErrors(cudaMallocHost(&h_B, size));
    checkCudaErrors(cudaMallocHost(&h_C, size));

    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // Allocate vectors in device memory
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc((void **)(&d_A), size));
    checkCudaErrors(cudaMalloc((void **)(&d_B), size));
    checkCudaErrors(cudaMalloc((void **)(&d_C), size));

    // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // Copy vectors from host memory to device memory
    // JP: `cudaMemcpyAsync`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));

    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    void *args[] = {&d_A, &d_B, &d_C, &N};

    // Launch the CUDA kernel
    checkCudaDrvErrors(
        // JP: `cuLaunchKernel`: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
        cuLaunchKernel(vecAdd_kernel, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, stream, args, NULL));

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));
    // JP: `cudaStreamSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaStreamSynchronize(stream));
    // Verify result
    int i;

    for (i = 0; i < N; ++i) {
        float sum = h_A[i] + h_B[i];

        if (fabs(h_C[i] - sum) > 1e-7f) {
            break;
        }
    }

    checkCudaDrvErrors(cuModuleUnload(cuModule));
    CleanupNoFailure(cuContext);
```

> JP: この抜粋は `cpp/0_Introduction/simpleDrvRuntime/simpleDrvRuntime.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/0_Introduction/simpleDrvRuntime/simpleDrvRuntime.cpp:180-199
```cpp

// JP: この anchor では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
int CleanupNoFailure(CUcontext &cuContext)
{
    // Free device memory
    // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    // Free host memory
    if (h_A) {
        checkCudaErrors(cudaFreeHost(h_A));
    }

    if (h_B) {
        checkCudaErrors(cudaFreeHost(h_B));
    }

    if (h_C) {
```

> JP: この抜粋は `cpp/0_Introduction/simpleDrvRuntime/simpleDrvRuntime.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `vectorAdd_kernel.cu`

Source: cpp/0_Introduction/simpleDrvRuntime/vectorAdd_kernel.cu:38-45
```cuda
extern "C" __global__ void VecAdd_kernel(const float *A, const float *B, float *C, int N)
{
    // JP: `blockDim`, `blockIdx`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}
```

> JP: この抜粋は `cpp/0_Introduction/simpleDrvRuntime/vectorAdd_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cuContext` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaMallocHost` | pinned host memory を作り、async copy や DMA の前提を作る API です。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaMemcpyAsync` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaFreeHost` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cuModule` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuLaunchKernel` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaStreamSynchronize` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `CUcontext` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuDevice` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuInit` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuCtxCreate` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuModuleLoadData` | Driver API の handle 境界です。context/module/function と error code を追います。 |

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

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Sample-Specific Notes

- NVRTC/Driver/JIT 系では、compile/load/link と kernel launch が別の段階です。compile log、module、function handle、launch parameter の対応が重要です。

> **日本語**
> この section は同じ template ではなく、sample 名、path、検出した API から読みどころを絞っています。
>
> **学習メモ**
> 似た名前の sample は Runtime 版、Driver 版、NVRTC 版、library 版の違いを比較すると学習効果が高くなります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target simpleDrvRuntime
ctest --test-dir build -R simpleDrvRuntime
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

- `cuContext` の直前と直後で、どの memory/resource が有効になったかをメモする。
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
