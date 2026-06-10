# shfl_scan - CUDA Parallel Prefix Sum with Shuffle Intrinsics (SHFL_Scan) - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This example demonstrates how to use the shuffle intrinsic __shfl_up_sync to perform a scan operation across a thread block.

Data-Parallel Algorithms, Performance Strategies

Original README headings: `shfl_scan - CUDA Parallel Prefix Sum with Shuffle Intrinsics (SHFL_Scan)`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/2_Concepts_and_Techniques/shfl_scan` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `shfl_scan` as a focused example of the CUDA concepts used in `cpp/2_Concepts_and_Techniques/shfl_scan`.
> **日本語**
> この sample の目的は、`shfl_scan` の小さな実装を通して Shared Memory, Streams And Events, Synchronization And Atomics, Performance, Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `shfl_integral_image.cuh, shfl_scan.cu, util.h` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `shfl_integral_image.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `shfl_scan.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `util.h`: Host/device declarations, helper types, constants, or library wrappers.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `shfl_integral_image.cuh` first and locate the host-side setup or Python entry point.
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

- `shfl_integral_image.cuh`: focus on `threadIdx`, `__syncthreads`, `__shared__`, `blockIdx`, `blockDim`.
- `shfl_scan.cu`: focus on `blockDim`, `threadIdx`, `cudaEventRecord`, `launch`, `blockIdx`.
- `util.h`: focus on `CUDA`, `cudaSuccess`, `cudaGetErrorString`, `cudaError_t`, `launch`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/2_Concepts_and_Techniques/shfl_scan/CMakeLists.txt:1-37
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(shfl_scan LANGUAGES C CXX CUDA)

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
# Add target for shfl_scan
add_executable(shfl_scan shfl_scan.cu)

target_compile_options(shfl_scan PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(shfl_scan PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(shfl_scan PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/shfl_scan/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `shfl_integral_image.cuh`

Source: cpp/2_Concepts_and_Techniques/shfl_scan/shfl_integral_image.cuh:29-65
```cuda
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Utility function to extract unsigned chars from an
// unsigned integer
__device__ uchar4 uint_to_uchar4(const unsigned int in)
{
    return make_uchar4(
        (in & 0x000000ff) >> 0, (in & 0x0000ff00) >> 8, (in & 0x00ff0000) >> 16, (in & 0xff000000) >> 24);
}

// Utility for dealing with vector data at different levels.
struct packed_result
{
    uint4 x, y, z, w;
};

__device__ packed_result get_prefix_sum(const uint4 &data, const cg::thread_block &cta)
{
    const auto tile = cg::tiled_partition<32>(cta);

    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    __shared__ unsigned int sums[128];
    const unsigned int      lane_id = tile.thread_rank();
    const unsigned int      warp_id = tile.meta_group_rank();

    unsigned int result[16] = {};
    {
        const uchar4 a = uint_to_uchar4(data.x);
        const uchar4 b = uint_to_uchar4(data.y);
        const uchar4 c = uint_to_uchar4(data.z);
        const uchar4 d = uint_to_uchar4(data.w);

        result[0] = a.x;
        result[1] = a.x + a.y;
        result[2] = a.x + a.y + a.z;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/shfl_scan/shfl_integral_image.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/shfl_scan/shfl_integral_image.cuh:127-146
```cuda
    if (tile.thread_rank() == (tile.size() - 1)) {
        sums[warp_id] = result[15];
    }

    // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
    __syncthreads();

    if (warp_id == 0) {
        unsigned int warp_sum = sums[lane_id];

#pragma unroll
        for (unsigned int i = 1; i <= 16; i *= 2) {
            const unsigned int n = tile.shfl_up(warp_sum, i);

            if (lane_id >= i)
                warp_sum += n;
        }

        sums[lane_id] = warp_sum;
    }
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/shfl_scan/shfl_integral_image.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `shfl_scan.cu`

Source: cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu:37-68
```cuda
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdio.h>

#include "shfl_integral_image.cuh"

// Scan using shfl - takes log2(n) steps
// This function demonstrates basic use of the shuffle intrinsic, __shfl_up,
// to perform a scan operation across a block.
// First, it performs a scan (prefix sum in this case) inside a warp
// Then to continue the scan operation across the block,
// each warp's sum is placed into shared memory.  A single warp
// then performs a shuffle scan on that shared memory.  The results
// are then uniformly added to each warp's threads.
// This pyramid type approach is continued by placing each block's
// final sum in global memory and prefix summing that via another kernel call,
// then uniformly adding across the input data via the uniform_add<<<>>> kernel.

__global__ void shfl_scan_test(int *data, int width, int *partial_sums = NULL)
{
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    extern __shared__ int sums[];
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int                   id      = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int                   lane_id = id % warpSize;
    // determine a warp_id within a block
    int warp_id = threadIdx.x / warpSize;

    // Below is the basic structure of using a shfl instruction
    // for a scan.
    // Record "value" as a variable - we accumulate it along the way
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu:92-111
```cuda
    if (threadIdx.x % warpSize == warpSize - 1) {
        sums[warp_id] = value;
    }

    // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
    __syncthreads();

    //
    // scan sum the warp sums
    // the same shfl scan operation, but performed on warp sums
    //
    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    if (warp_id == 0 && lane_id < (blockDim.x / warpSize)) {
        int warp_sum = sums[lane_id];

        int mask = (1 << (blockDim.x / warpSize)) - 1;
        for (int i = 1; i <= (blockDim.x / warpSize); i *= 2) {
            int n = __shfl_up_sync(mask, warp_sum, i, (blockDim.x / warpSize));

            if (lane_id >= i)
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu:244-263
```cuda
        printf("> __shfl() intrinsic requires device SM 3.0+\n");
        printf("> Waiving test.\n");
        exit(EXIT_WAIVED);
    }

    // JP: `cudaMallocHost`: page-locked host memory は DMA/async copy を安定させます。通常の free ではなく対応する CUDA API で解放します。
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_data), sizeof(int) * n_elements));
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_result), sizeof(int) * n_elements));

    // initialize data:
    printf("Computing Simple Sum test\n");
    printf("---------------------------------------------------\n");

    printf("Initialize test data [1, 1, 1...]\n");

    for (int i = 0; i < n_elements; i++) {
        h_data[i] = 1;
    }

    int blockSize     = 256;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu:282-301
```cuda
    float inc = 0;

    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_data), sz));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_partial_sums), partial_sz));
    // JP: `cudaMemset`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemset(d_partial_sums, 0, partial_sz));

    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&h_partial_sums), partial_sz));
    checkCudaErrors(cudaMemcpy(d_data, h_data, sz, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventRecord(start, 0));
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    shfl_scan_test<<<gridSize, blockSize, shmem_sz>>>(d_data, 32, d_partial_sums);
    shfl_scan_test<<<p_gridSize, p_blockSize, shmem_sz>>>(d_partial_sums, 32);
    uniform_add<<<gridSize - 1, blockSize>>>(d_data + blockSize, d_partial_sums, n_elements);
    checkCudaErrors(cudaEventRecord(stop, 0));
    // JP: この連続する anchor 群では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&inc, start, stop));
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `util.h`

Source: cpp/2_Concepts_and_Techniques/shfl_scan/util.h:29-62
```cpp
#ifndef SAMPLES_SHFL_SCAN_UTIL_H_
#define SAMPLES_SHFL_SCAN_UTIL_H_

// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                                                                        \
    do {                                                                                                            \
        /* Check synchronous errors, i.e. pre-launch */                                                             \
        cudaError_t err = cudaGetLastError();                                                                       \
        if (cudaSuccess != err) {                                                                                   \
            fprintf(                                                                                                \
                stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                     \
        }                                                                                                           \
        /* Check asynchronous errors, i.e. kernel failed (ULF) */                                                   \
        err = cudaDeviceSynchronize();                                                                              \
        if (cudaSuccess != err) {                                                                                   \
            fprintf(                                                                                                \
                stderr, "Cuda error in file '%s' in line %i : %s!\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                     \
        }                                                                                                           \
    } while (0)

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_CHECK(call)                                                                                            \
    do {                                                                                                            \
        cudaError_t err = call;                                                                                     \
        if (cudaSuccess != err) {                                                                                   \
            fprintf(                                                                                                \
                stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                     \
        }                                                                                                           \
    } while (0)

#endif // SAMPLES_SHFL_SCAN_UTIL_H_
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/shfl_scan/util.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `__syncthreads` | block 内共有 memory または同期境界です。producer/consumer の順序を確認します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaEventRecord` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaMallocHost` | pinned host memory を作り、async copy や DMA の前提を作る API です。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaFreeHost` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `__shared__` | block 内共有 memory または同期境界です。producer/consumer の順序を確認します。 |
| `cudaEventCreate` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaMemset` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |

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
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。
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
cmake --build build --target shfl_scan
ctest --test-dir build -R shfl_scan
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
- different stream 間に依存があるのに event や explicit sync を置かない。

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
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Shared Memory](../../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
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
