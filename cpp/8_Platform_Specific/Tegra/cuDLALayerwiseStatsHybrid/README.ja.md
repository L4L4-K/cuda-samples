# cuDLALayerwiseStatsHybrid - cuDLA Layerwise statistics HybridMode - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample is used to provide layerwise statistics to the application in the cuDLA hybrid mode wherein DLA is programmed using CUDA.

cuDLA, Data Parallel Algorithms, Image Processing

Original README headings: `cuDLALayerwiseStatsHybrid - cuDLA Layerwise statistics HybridMode`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsHybrid` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `cuDLALayerwiseStatsHybrid` as a focused example of the CUDA concepts used in `cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsHybrid`.
> **日本語**
> この sample の目的は、`cuDLALayerwiseStatsHybrid` の小さな実装を通して Streams And Events, Memory, Execution Model, Debugging, Profiling, And Testing を具体的に追うことです。
>
> **学習メモ**
> 最初に `main.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `main.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `main.cu` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
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

- `main.cu`: focus on `cudaSuccess`, `cudaFree`, `CUDA`, `cudaGetErrorName`, `cuDLA`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsHybrid/CMakeLists.txt:1-58
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/Modules")

# JP: `cuDLALayerwiseStatsHybrid`: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
project(cuDLALayerwiseStatsHybrid LANGUAGES C CXX CUDA)

# JP: `find_package`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 87 110)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../../Common)

find_library(CUDLA_LIB cudla PATHS ${CUDAToolkit_LIBRARY_DIR} ${CMAKE_LIBRARY_PATH})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CUDLA_LIB)
        # Source file
        # Add target for cuDLALayerwiseStatsHybrid
        # JP: この連続する anchor 群では CMake CUDA target/link/architecture wiring です。source、target、optional dependency、platform condition を確認します。
        add_executable(cuDLALayerwiseStatsHybrid main.cu)

        target_compile_options(cuDLALayerwiseStatsHybrid PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(cuDLALayerwiseStatsHybrid PRIVATE cxx_std_17 cuda_std_17)

        set_target_properties(cuDLALayerwiseStatsHybrid PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(cuDLALayerwiseStatsHybrid PUBLIC
            ${CUDAToolkit_INCLUDE_DIRS}
        )

        target_link_libraries(cuDLALayerwiseStatsHybrid
            ${CUDLA_LIB}
        )
    else()
        message(STATUS "CUDLA not found - will not build sample 'cuDLALayerwiseStatsHybrid'")
    endif()
else()
    message(STATUS "Will not build sample cuDLALayerwiseStatsHybrid - requires Linux OS")
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsHybrid/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `main.cu`

Source: cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsHybrid/main.cu:29-47
```cuda
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

#include "cuda_runtime.h"
#include "cudla.h"
#include "cudlaExternalEtbl.hpp"

#define MAX_FILENAME_LEN    200
#define RESERVED_SUFFIX_LEN 10

#define DPRINTF(...) printf(__VA_ARGS__)

static void printTensorDesc(cudlaModuleTensorDescriptor *tensorDesc)
{
    DPRINTF("\tTENSOR NAME : %s\n", tensorDesc->name);
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsHybrid/main.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsHybrid/main.cu:91-110
```cuda
void cleanUp(ResourceList *resourceList)
{
    uint32_t ii = 0;
    if (resourceList->inputTensorDesc != NULL) {
        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(resourceList->inputTensorDesc);
        resourceList->inputTensorDesc = NULL;
    }

    if (resourceList->outputTensorDesc != NULL) {
        free(resourceList->outputTensorDesc);
        resourceList->outputTensorDesc = NULL;
    }

    if (resourceList->outputTaskStatisticsDesc != NULL) {
        free(resourceList->outputTaskStatisticsDesc);
        resourceList->outputTaskStatisticsDesc = NULL;
    }

    if (resourceList->loadableData != NULL) {
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsHybrid/main.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsHybrid/main.cu:231-250
```cuda
    resourceList->numInputTensors         = 0;
    resourceList->numOutputTensors        = 0;
    resourceList->numOutputTaskStatistics = 0;
}

int main(int argc, char **argv)
{
    cudlaDevHandle devHandle;
    cudlaModule    moduleHandle;
    cudlaStatus    err;
    uint32_t       statSupport  = 0;
    uint32_t       dlaFreqInMHz = 0;
    FILE          *fp           = NULL;
    struct stat    st;
    size_t         file_size;
    size_t         actually_read = 0;
    unsigned char *loadableData  = NULL;
    char           filename[MAX_FILENAME_LEN];
    const char    *suffix = ".csv";

```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsHybrid/main.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsHybrid/main.cu:287-306
```cuda
    DPRINTF("The file size = %ld\n", file_size);

    dlaFreqInMHz = atoi(argv[2]);
    statSupport  = atoi(argv[3]);

    loadableData = (unsigned char *)malloc(file_size);
    if (loadableData == NULL) {
        DPRINTF("Cannot Allocate memory for loadable\n");
        return 1;
    }

    actually_read = fread(loadableData, 1, file_size, fp);
    if (actually_read != file_size) {
        free(loadableData);
        DPRINTF("Read wrong size\n");
        return 1;
    }
    fclose(fp);

    resourceList.loadableData = loadableData;
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsHybrid/main.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cuDLA` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaGetErrorName` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMemcpyAsync` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaStreamSynchronize` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaStreamDestroy` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaSetDevice` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMemsetAsync` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaStream_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaStreamCreateWithFlags` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cuDLALayerwiseStatsHybrid` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `Device` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaMemcpyDeviceToHost` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
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
cmake --build build --target cuDLALayerwiseStatsHybrid
ctest --test-dir build -R cuDLALayerwiseStatsHybrid
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
- different stream 間に依存があるのに event や explicit sync を置かない。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cuDLA` の直前と直後で、どの memory/resource が有効になったかをメモする。
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

- [Streams And Events](../../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Memory](../../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Execution Model](../../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
