# segmentationTreeThrust - CUDA Segmentation Tree Thrust Library - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates an approach to the image segmentation trees construction.  This method is based on Boruvka's MST algorithm.

Data-Parallel Algorithms, Performance Strategies

Original README headings: `segmentationTreeThrust - CUDA Segmentation Tree Thrust Library`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/2_Concepts_and_Techniques/segmentationTreeThrust` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `segmentationTreeThrust` as a focused example of the CUDA concepts used in `cpp/2_Concepts_and_Techniques/segmentationTreeThrust`.
> **日本語**
> この sample の目的は、`segmentationTreeThrust` の小さな実装を通して CUDA Libraries, Streams And Events, Performance, Memory, Kernel Launch And Indexing を具体的に追うことです。
>
> **学習メモ**
> 最初に `common.cuh, kernels.cuh, segmentationTree.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `common.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `data/ref_00.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/ref_09.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/test.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `kernels.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `segmentationTree.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `common.cuh` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Create library handles, descriptors, plans, or workspaces before the library call.
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

- `common.cuh`: focus on control flow and helper functions.
- `kernels.cuh`: focus on `blockIdx`, `blockDim`, `threadIdx`, `CUDA`, `launch`.
- `segmentationTree.cu`: focus on `thrust::device_ptr`, `launch`, `cudaMemcpy`, `thrust::make_zip_iterator`, `thrust::system_error`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/2_Concepts_and_Techniques/segmentationTreeThrust/CMakeLists.txt:1-58
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(segmentationTreeThrust LANGUAGES C CXX CUDA)

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
# Add target for segmentationTreeThrust
add_executable(segmentationTreeThrust segmentationTree.cu)

target_compile_options(segmentationTreeThrust PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(segmentationTreeThrust PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(segmentationTreeThrust PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Copy data files to output directory
add_custom_command(TARGET segmentationTreeThrust POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/test.ppm
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Copy data files to output directory
add_custom_command(TARGET segmentationTreeThrust POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/ref_00.ppm
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Copy data files to output directory
add_custom_command(TARGET segmentationTreeThrust POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data/ref_09.ppm
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/segmentationTreeThrust/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `common.cuh`

Source: cpp/2_Concepts_and_Techniques/segmentationTreeThrust/common.cuh:29-36
```cuda
#ifndef _COMMON_CUH_
#define _COMMON_CUH_

typedef unsigned char          uchar;
typedef unsigned int           uint;
typedef unsigned long long int ullint;

#endif // #ifndef _COMMON_CUH_
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/segmentationTreeThrust/common.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `kernels.cuh`

Source: cpp/2_Concepts_and_Techniques/segmentationTreeThrust/kernels.cuh:34-71
```cuda
#ifndef _KERNELS_H_
#define _KERNELS_H_

#include <stdio.h>

#include "common.cuh"

// Functors used with thrust library.
template <typename Input> struct IsGreaterEqualThan
{
    __host__ __device__ IsGreaterEqualThan(uint upperBound)
        : upperBound_(upperBound)
    {
    }

    __host__ __device__ bool operator()(const Input &value) const { return value >= upperBound_; }

    uint upperBound_;
};

// CUDA kernels.
__global__ void addScalar(uint *array, int scalar, uint size)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        array[tid] += scalar;
    }
}

__global__ void markSegments(const uint *verticesOffsets, uint *flags, uint verticesCount)
{
    // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < verticesCount) {
        flags[verticesOffsets[tid]] = 1;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/segmentationTreeThrust/kernels.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `segmentationTree.cu`

Source: cpp/2_Concepts_and_Techniques/segmentationTreeThrust/segmentationTree.cu:41-59
```cuda
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// STL includes.
#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <vector>

// Thrust library includes.
#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/segmentationTreeThrust/segmentationTree.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/segmentationTreeThrust/segmentationTree.cu:240-259
```cuda
        {
        }

        void buildFromDeviceData(thrust::device_ptr<uint> superVerticesOffsets, thrust::device_ptr<uint> verticesIDs)
        {
            // JP: `cudaMemcpy`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
            checkCudaErrors(cudaMemcpy(&(superNodesOffsets_[0]),
                                       superVerticesOffsets.get(),
                                       sizeof(uint) * superNodesOffsets_.size(),
                                       cudaMemcpyDeviceToHost));

            checkCudaErrors(
                cudaMemcpy(&(nodes_[0]), verticesIDs.get(), sizeof(uint) * nodes_.size(), cudaMemcpyDeviceToHost));
        }

    private:
        friend class Pyramid;

        // The pair of the following vectors describes the
        // relation between the consecutive levels.
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/segmentationTreeThrust/segmentationTree.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/segmentationTreeThrust/segmentationTree.cu:329-348
```cuda
            colors[colorIndex * 3]     = myrand() % 256;
            colors[colorIndex * 3 + 1] = myrand() % 256;
            colors[colorIndex * 3 + 2] = myrand() % 256;
        }

        uchar *image = new uchar[width * height * 3];

        while (!nodesQueue.empty()) {
            std::pair<uint, uint> currentNode = nodesQueue.front();
            nodesQueue.pop_front();

            uint pixelIndex   = currentNode.first;
            uint pixelSegment = currentNode.second;

            image[pixelIndex * 3]     = colors[pixelSegment * 3];
            image[pixelIndex * 3 + 1] = colors[pixelSegment * 3 + 1];
            image[pixelIndex * 3 + 2] = colors[pixelSegment * 3 + 2];
        }

        __savePPM(filename, image, width, height, 3);
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/segmentationTreeThrust/segmentationTree.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/segmentationTreeThrust/segmentationTree.cu:406-425
```cuda
            cout << "Algorithm failed (" << e.what() << ")" << endl;
            exit(EXIT_FAILURE);
        }

        cudaEventRecord(stop, 0);
        // JP: `cudaEventSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
        cudaEventSynchronize(stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        return elapsedTime;
    }

private:
    void printMemoryUsage()
    {
        size_t availableMemory, totalMemory, usedMemory;

        cudaMemGetInfo(&availableMemory, &totalMemory);
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/segmentationTreeThrust/segmentationTree.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `thrust::device_ptr` | CUDA library call です。handle/descriptor/workspace と data layout を確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `thrust::make_zip_iterator` | CUDA library call です。handle/descriptor/workspace と data layout を確認します。 |
| `cudaMemset` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `thrust::system_error` | CUDA library call です。handle/descriptor/workspace と data layout を確認します。 |
| `cudaEventCreate` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaEventRecord` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaEventSynchronize` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaMemcpyDeviceToHost` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaEventElapsedTime` | 非同期 work の順序、overlap、計測範囲を表す API です。 |

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
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。

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
cmake --build build --target segmentationTreeThrust
ctest --test-dir build -R segmentationTreeThrust
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
- leading dimension、stride、descriptor、workspace size を host 配列の見た目だけで判断する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `thrust::device_ptr` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- handle/descriptor/workspace の作成、利用、破棄を対応表にする。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [CUDA Libraries](../../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。
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
