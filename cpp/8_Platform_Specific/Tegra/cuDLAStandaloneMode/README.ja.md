# cuDLAStandaloneMode - cuDLA Standalone Mode - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates cuDLA standalone mode wherein DLA can be programmed without using CUDA.

cuDLA, Data Parallel Algorithms, Image Processing

Original README headings: `cuDLAStandaloneMode - cuDLA Standalone Mode`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/8_Platform_Specific/Tegra/cuDLAStandaloneMode` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `cuDLAStandaloneMode` as a focused example of the CUDA concepts used in `cpp/8_Platform_Specific/Tegra/cuDLAStandaloneMode`.
> **日本語**
> この sample の目的は、`cuDLAStandaloneMode` の小さな実装を通して Streams And Events, Synchronization And Atomics, Memory, Execution Model, Debugging, Profiling, And Testing を具体的に追うことです。
>
> **学習メモ**
> 最初に `main.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `main.cpp`: Host-side setup, API calls, validation, and cleanup.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `main.cpp` first and locate the host-side setup or Python entry point.
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

- `main.cpp`: focus on `cuDLA`, `cuDLAStandaloneMode`, `Device`, `CUDA`, `CUDLA_NVSCISYNC_FENCE`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/8_Platform_Specific/Tegra/cuDLAStandaloneMode/CMakeLists.txt:1-65
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/Modules")

# JP: `cuDLAStandaloneMode`: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
project(cuDLAStandaloneMode LANGUAGES C CXX CUDA)

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

# JP: この連続する anchor 群では CMake CUDA target/link/architecture wiring です。source、target、optional dependency、platform condition を確認します。
find_package(NVSCI)
find_library(CUDLA_LIB cudla PATHS ${CUDAToolkit_LIBRARY_DIR} ${CMAKE_LIBRARY_PATH})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CUDLA_LIB)
        if(NVSCI_FOUND)
            # Source file
            # Add target for cuDLAStandaloneMode
            add_executable(cuDLAStandaloneMode main.cpp)

            target_compile_options(cuDLAStandaloneMode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

            target_compile_features(cuDLAStandaloneMode PRIVATE cxx_std_17 cuda_std_17)

            set_target_properties(cuDLAStandaloneMode PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

            target_include_directories(cuDLAStandaloneMode PUBLIC
                ${CUDAToolkit_INCLUDE_DIRS}
                ${NVSCI_INCLUDE_DIRS}
            )

            target_link_libraries(cuDLAStandaloneMode
                ${CUDLA_LIB}
                ${NVSCI_LIBRARIES}
            )
        else()
            message(STATUS "NvSCI not found - will not build sample 'cuDLAStandaloneMode'")
        endif()
    else()
        message(STATUS "CUDLA not found - will not build sample 'cuDLAStandaloneMode'")
    endif()
else()
    message(STATUS "Will not build sample cuDLAStandaloneMode - requires Linux OS")
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cuDLAStandaloneMode/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `main.cpp`

Source: cpp/8_Platform_Specific/Tegra/cuDLAStandaloneMode/main.cpp:29-47
```cpp
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

#include "cudla.h"
#include "nvscibuf.h"
#include "nvscierror.h"
#include "nvscisync.h"

#define DPRINTF(...) printf(__VA_ARGS__)

static void printTensorDesc(cudlaModuleTensorDescriptor *tensorDesc)
{
    DPRINTF("\tTENSOR NAME : %s\n", tensorDesc->name);
    DPRINTF("\tsize: %lu\n", tensorDesc->size);

```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cuDLAStandaloneMode/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cuDLAStandaloneMode/main.cpp:216-235
```cpp
        resourceList->syncModule = NULL;
    }

    if (resourceList->waitEvents != NULL) {
        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(resourceList->waitEvents);
        resourceList->waitEvents = NULL;
    }

    if (resourceList->preFences != NULL) {
        free(resourceList->preFences);
        resourceList->preFences = NULL;
    }

    if (resourceList->signalEvents != NULL) {
        if (resourceList->signalEvents->eofFences != NULL) {
            free(resourceList->signalEvents->eofFences);
            resourceList->signalEvents->eofFences = NULL;
        }

```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cuDLAStandaloneMode/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cuDLAStandaloneMode/main.cpp:356-375
```cpp
    keyValue[1].len             = sizeof(cpuPerm);

    return NvSciSyncAttrListSetAttrs(list, keyValue, 2);
}

int main(int argc, char **argv)
{
    cudlaDevHandle devHandle;
    cudlaModule    moduleHandle;
    cudlaStatus    err;
    FILE          *fp = NULL;
    struct stat    st;
    size_t         file_size;
    size_t         actually_read = 0;
    unsigned char *loadableData  = NULL;

    ResourceList resourceList;

    memset(&resourceList, 0x00, sizeof(ResourceList));

```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cuDLAStandaloneMode/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cuDLAStandaloneMode/main.cpp:392-411
```cpp
    }

    file_size = st.st_size;
    DPRINTF("The file size = %ld\n", file_size);

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

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cuDLAStandaloneMode/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cuDLA` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuDLAStandaloneMode` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `Device` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `CUDLA_NVSCISYNC_FENCE` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `atomic` | 複数 thread が同じ address を更新する箇所です。競合と順序の意味を確認します。 |
| `CUDLA_STANDALONE` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUDLA_NUM_INPUT_TENSORS` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUDLA_NUM_OUTPUT_TENSORS` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUDLA_INPUT_TENSOR_DESCRIPTORS` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUDLA_OUTPUT_TENSOR_DESCRIPTORS` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUDLA_NVSCISYNC_ATTR_WAIT` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUDLA_NVSCISYNC_ATTR_SIGNAL` | Driver API の handle 境界です。context/module/function と error code を追います。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
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
cmake --build build --target cuDLAStandaloneMode
ctest --test-dir build -R cuDLAStandaloneMode
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
- [Synchronization And Atomics](../../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Memory](../../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Execution Model](../../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
