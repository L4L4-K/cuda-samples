# cudaCompressibleMemory - CUDA Compressible Memory - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates the compressible memory allocation using cuMemMap API.

CUDA Driver API, Compressible Memory, MMAP

Original README headings: `cudaCompressibleMemory - CUDA Compressible Memory`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/3_CUDA_Features/cudaCompressibleMemory` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `cudaCompressibleMemory` as a focused example of the CUDA concepts used in `cpp/3_CUDA_Features/cudaCompressibleMemory`.
> **日本語**
> この sample の目的は、`cudaCompressibleMemory` の小さな実装を通して Runtime, Driver, And NVRTC, Multi-GPU, P2P, And IPC, Streams And Events, Performance, Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `compMalloc.cpp, compMalloc.h, saxpy.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- The device topology required by the README, such as multiple GPUs, peer access, IPC, MPI, or process support

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
- `compMalloc.cpp`: Host-side setup, API calls, validation, and cleanup.
- `compMalloc.h`: Host/device declarations, helper types, constants, or library wrappers.
- `saxpy.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `compMalloc.cpp` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.
- Enumerate devices, enable peer or IPC access, and record which device/process owns each resource.
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

- `compMalloc.cpp`: focus on `CUDA_SUCCESS`, `cudaErrorMemoryAllocation`, `cudaSuccess`, `cudaError_t`, `CUmemAllocationProp`.
- `compMalloc.h`: focus on `cudaError_t`.
- `saxpy.cu`: focus on `cudaMemcpy`, `blockDim`, `cuDeviceGetAttribute`, `blockIdx`, `threadIdx`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/3_CUDA_Features/cudaCompressibleMemory/CMakeLists.txt:1-45
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(cudaCompressibleMemory LANGUAGES C CXX CUDA)

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
# Add target for cudaCompressibleMemory
add_executable(cudaCompressibleMemory compMalloc.cpp saxpy.cu)

target_compile_options(cudaCompressibleMemory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cudaCompressibleMemory PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(cudaCompressibleMemory PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cudaCompressibleMemory PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cudaCompressibleMemory PRIVATE
    CUDA::cuda_driver
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/3_CUDA_Features/cudaCompressibleMemory/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `compMalloc.cpp`

Source: cpp/3_CUDA_Features/cudaCompressibleMemory/compMalloc.cpp:29-53
```cpp
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <string.h>

cudaError_t setProp(CUmemAllocationProp *prop, bool UseCompressibleMemory)
{
    // JP: driver_api: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
    CUdevice currentDevice;
    if (cuCtxGetDevice(&currentDevice) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    memset(prop, 0, sizeof(CUmemAllocationProp));
    prop->type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop->location.id   = currentDevice;

    if (UseCompressibleMemory)
        prop->allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

    return cudaSuccess;
}

cudaError_t allocateCompressible(void **adr, size_t size, bool UseCompressibleMemory)
```

> JP: この抜粋は `cpp/3_CUDA_Features/cudaCompressibleMemory/compMalloc.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `compMalloc.h`

Source: cpp/3_CUDA_Features/cudaCompressibleMemory/compMalloc.h:29-35
```cpp
#ifndef COMP_MALLOC_H
#define COMP_MALLOC_H

cudaError_t allocateCompressible(void **adr, size_t size, bool UseCompressibleMemory);
cudaError_t freeCompressible(void *ptr, size_t size, bool UseCompressibleMemory);

#endif
```

> JP: この抜粋は `cpp/3_CUDA_Features/cudaCompressibleMemory/compMalloc.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `saxpy.cu`

Source: cpp/3_CUDA_Features/cudaCompressibleMemory/saxpy.cu:35-57
```cuda
#include <cuda.h>
#include <stdio.h>
#define CUDA_DRIVER_API
#include "compMalloc.h"
#include "helper_cuda.h"

__global__ void saxpy(const float a, const float4 *x, const float4 *y, float4 *z, const size_t n)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float4 x4 = x[i];
        const float4 y4 = y[i];
        z[i]            = make_float4(a * x4.x + y4.x, a * x4.y + y4.y, a * x4.z + y4.z, a * x4.w + y4.w);
    }
}

__global__ void init(float4 *x, float4 *y, const float val, const size_t n)
{
    const float4 val4 = make_float4(val, val, val, val);
    // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        x[i] = y[i] = val4;
    }
```

> JP: この抜粋は `cpp/3_CUDA_Features/cudaCompressibleMemory/saxpy.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/cudaCompressibleMemory/saxpy.cu:71-136
```cuda
    int         blockSize;
    int         minGridSize;
    dim3        threads, blocks;

    if (!compressibleZbuf) {
        // We are on config where compressible buffer can only be initialized through cudaMemcpy
        // hence, x & y buffers are allocated as compressible and initialized via cudaMemcpy
        // whereas z buffer is allocated as non-compressible.
        float4 *h_x = (float4 *)malloc(sizeof(float4) * n);
        float4 *h_y = (float4 *)malloc(sizeof(float4) * n);
        for (int i = 0; i < n; i++) {
            h_x[i].x = h_x[i].y = h_x[i].z = h_x[i].w = init_val;
            h_y[i].x = h_y[i].y = h_y[i].z = h_y[i].w = init_val;
        }
        // JP: `cudaMemcpy`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
        checkCudaErrors(cudaMemcpy(x, h_x, sizeof(float4) * n, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(y, h_y, sizeof(float4) * n, cudaMemcpyHostToDevice));
        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(h_x);
        free(h_y);
    }
    else {
        checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)init));
        threads = dim3(blockSize, 1, 1);
        blocks  = dim3(minGridSize, 1, 1);
        // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
        init<<<blocks, threads>>>(x, y, init_val, n);
    }

    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)saxpy));
    threads = dim3(blockSize, 1, 1);
    blocks  = dim3(minGridSize, 1, 1);

    // JP: この連続する anchor 群では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));
    saxpy<<<blocks, threads>>>(a, x, y, z, n);
    checkCudaErrors(cudaEventRecord(stop));
    // JP: `cudaEventSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));

    const size_t size = n * sizeof(float4);
    printf("Running saxpy with %d blocks x %d threads = %.3f ms %.3f TB/s\n",
           blocks.x,
           threads.x,
           ms,
           (size * 3) / ms / 1e9);
}

int main(int argc, char **argv)
{
    const size_t n = 10485760;

    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        exit(EXIT_SUCCESS);
    }

    findCudaDevice(argc, (const char **)argv);
    // JP: driver_api: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
    CUdevice currentDevice;
    checkCudaErrors(cuCtxGetDevice(&currentDevice));

    // Check that the selected device supports virtual memory management
```

> JP: この抜粋は `cpp/3_CUDA_Features/cudaCompressibleMemory/saxpy.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `CUDA_SUCCESS` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaErrorMemoryAllocation` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaError_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuDeviceGetAttribute` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUmemAllocationProp` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cuMemMap` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cuCtxGetDevice` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuMemGetAllocationGranularity` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuMemCreate` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaOccupancyMaxPotentialBlockSize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaEventCreate` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaEventRecord` | 非同期 work の順序、overlap、計測範囲を表す API です。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- multi-GPU sample では、device 選択、peer capability、context/IPC handle の寿命を分けて読みます。
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
- Runtime/Driver/NVRTC sample では、compile/load した module と launch する kernel 名の対応が重要です。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target cudaCompressibleMemory
ctest --test-dir build -R cudaCompressibleMemory
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
- peer access が有効な device pair と、単に複数 GPU が存在することを混同する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `CUDA_SUCCESS` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- device ごとの ownership と、peer/IPC で共有される resource を分けて書く。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Runtime, Driver, And NVRTC](../../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [Multi-GPU, P2P, And IPC](../../../docs_ja/themes/multi_gpu_p2p_ipc.md): device topology、peer access、IPC handle、multi-process 境界を読むための基礎です。
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
