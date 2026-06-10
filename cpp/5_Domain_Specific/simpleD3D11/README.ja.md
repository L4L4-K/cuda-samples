# simpleD3D11 - Simple D3D11 - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

Simple program which demonstrates  how to use the CUDA D3D11 External Resource Interoperability APIs to update D3D11 buffers from CUDA and synchronize between D3D11 and CUDA with Keyed Mutexes.

Graphics Interop, Image Processing

Original README headings: `simpleD3D11 - Simple D3D11`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/simpleD3D11` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `simpleD3D11` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/simpleD3D11`.
> **日本語**
> この sample の目的は、`simpleD3D11` の小さな実装を通して CUDA Graphs, Streams And Events, Synchronization And Atomics, Memory, Kernel Launch And Indexing を具体的に追うことです。
>
> **学習メモ**
> 最初に `ShaderStructs.h, simpleD3D11.cpp, sinewave_cuda.cu, sinewave_cuda.h` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- The graphics, display, or platform stack named by the English README

> **日本語**
> 必要条件は英語 README と CMake/requirements を優先します。この guide は条件を置き換えず、読むべき確認点を追加します。
>
> **学習メモ**
> 実行できない場合は、source を変える前に driver、toolkit、GPU feature、library、platform guard、Python package version を確認します。

## Files

- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `ShaderStructs.h`: Host/device declarations, helper types, constants, or library wrappers.
- `data/ref_simpleD3D11.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleD3D11.cpp`: Host-side setup, API calls, validation, and cleanup.
- `sinewave_cuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `sinewave_cuda.h`: Host/device declarations, helper types, constants, or library wrappers.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `ShaderStructs.h` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Capture or build CUDA Graph nodes, instantiate the graph, then launch the executable graph.
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

- `ShaderStructs.h`: focus on `cudaStream_t`, `cudaDevVertptr`.
- `simpleD3D11.cpp`: focus on `CUDA`, `cuStatus`, `Device`, `cudaError`, `cuDevice`.
- `sinewave_cuda.cu`: focus on `cudaDevVertptr`, `cudaExternalSemaphore_t`, `cudaStream_t`, `blockIdx`, `blockDim`.
- `sinewave_cuda.h`: focus on `cudaExternalSemaphore_t`, `cudaStream_t`, `atomic`, `cudaDevVertptr`, `cudaImportVertexBuffer`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/simpleD3D11/CMakeLists.txt:1-62
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleD3D11 LANGUAGES C CXX CUDA)

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

if(WIN32)
    # Source file
    # Add target for simpleD3D11
    add_executable(simpleD3D11
        simpleD3D11.cpp
        sinewave_cuda.cu
        ../../../Common/rendercheck_d3d11.cpp
    )

    target_compile_options(simpleD3D11 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(simpleD3D11 PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(simpleD3D11 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(simpleD3D11 PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(simpleD3D11 PRIVATE
        d3d11
        dxgi
        dxguid
        d3dcompiler
    )

    add_custom_command(TARGET simpleD3D11 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        ${CMAKE_CURRENT_BINARY_DIR}/data
    )
else()
    message(STATUS "Sample 'simpleD3D11' is Windows-only - skipping")
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `ShaderStructs.h`

Source: cpp/5_Domain_Specific/simpleD3D11/ShaderStructs.h:29-50
```cpp
#pragma once

// #include "stdafx.h"
#include <DirectXMath.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"

using namespace DirectX;

struct Vertex
{
    XMFLOAT3 position;
    XMFLOAT4 color;
};

void RunSineWaveKernel(size_t       mesh_width,
                       size_t       mesh_height,
                       Vertex      *cudaDevVertptr,
                       // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
                       cudaStream_t streamToRun,
                       float        AnimTime);
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11/ShaderStructs.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `simpleD3D11.cpp`

Source: cpp/5_Domain_Specific/simpleD3D11/simpleD3D11.cpp:33-51
```cpp
#pragma warning(disable : 4312)

// includes for Windows
#include <windows.h>

// includes for multimedia
#include <mmsystem.h>

// This header inclues all the necessary D3D11 and CUDA includes
#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <dynlink_d3d11.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // includes cuda.h and cuda_runtime_api.h
#include <rendercheck_d3d11.h>
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11/simpleD3D11.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleD3D11/simpleD3D11.cpp:143-162
```cpp
        printf("> There are no device(s) supporting CUDA\n");
        return false;
    }
    else {
        // JP: python_cuda: Python object が CUDA resource を包みます。Python から見えても device memory/stream/context の寿命と順序は CUDA 側で管理します。
        printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
    }

    return true;
}

bool findDXDevice(char *dev_name)
{
    HRESULT   hr = S_OK;
    // JP: `cudaError`, `cuStatus`: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
    cudaError cuStatus;
    int       cuda_dev = -1;

    // Iterate through the candidate adapters
    IDXGIFactory1 *pFactory;
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11/simpleD3D11.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleD3D11/simpleD3D11.cpp:207-226
```cpp
    DXGI_ADAPTER_DESC adapterDesc;
    g_pCudaCapableAdapter->GetDesc(&adapterDesc);
    wcstombs(dev_name, adapterDesc.Description, 128);

    checkCudaErrors(cudaSetDevice(cuda_dev));
    // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));

    printf("> Found 1 D3D11 Adapater(s) /w Compute capability.\n");
    printf("> %s\n", dev_name);

    return true;
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11/simpleD3D11.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleD3D11/simpleD3D11.cpp:330-349
```cpp
                    const char *cur_image_path = "simpleD3D11.ppm";

                    // Save a reference of our current test run image
                    CheckRenderD3D11::ActiveRenderTargetToPPM(g_pd3dDevice, cur_image_path);

                    // compare to offical reference image, printing PASS or FAIL.
                    g_bPassed = CheckRenderD3D11::PPMvsPPM(cur_image_path, ref_file, argv[0], MAX_EPSILON, 0.15f);

                    g_bDone = true;

                    Cleanup();

                    PostQuitMessage(0);
                }
                else {
                    g_bPassed = true;
                }
            }
        }
    };
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11/simpleD3D11.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `sinewave_cuda.cu`

Source: cpp/5_Domain_Specific/simpleD3D11/sinewave_cuda.cu:29-51
```cuda
#include <stdio.h>

#include "ShaderStructs.h"
#include "helper_cuda.h"
#include "sinewave_cuda.h"

__global__ void sinewave_gen_kernel(Vertex *vertices, unsigned int width, unsigned int height, float time)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate uv coordinates
    float u = x / (float)width;
    float v = y / (float)height;
    u       = u * 2.0f - 1.0f;
    v       = v * 2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w    = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

    if (y < height && x < width) {
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11/sinewave_cuda.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleD3D11/sinewave_cuda.cu:108-127
```cuda
    extSemWaitParams.params.keyedMutex.timeoutMs = timeoutMs;

    checkCudaErrors(cudaWaitExternalSemaphoresAsync(&extSemaphore, &extSemWaitParams, 1, streamToRun));
}

// JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
void cudaReleaseSync(cudaExternalSemaphore_t &extSemaphore, uint64_t key, cudaStream_t streamToRun)
{
    cudaExternalSemaphoreSignalParams extSemSigParams;
    memset(&extSemSigParams, 0, sizeof(extSemSigParams));
    extSemSigParams.params.keyedMutex.key = key;

    checkCudaErrors(cudaSignalExternalSemaphoresAsync(&extSemaphore, &extSemSigParams, 1, streamToRun));
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void RunSineWaveKernel(cudaExternalSemaphore_t &extSemaphore,
                       uint64_t                &key,
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11/sinewave_cuda.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `sinewave_cuda.h`

Source: cpp/5_Domain_Specific/simpleD3D11/sinewave_cuda.h:29-47
```cpp
#ifndef SINEWAVE_CUDA_H
#define SINEWAVE_CUDA_H

#include <stdio.h>

#include "ShaderStructs.h"
#include "helper_cuda.h"

void    RunSineWaveKernel(cudaExternalSemaphore_t &extSemaphore,
                          uint64_t                &key,
                          unsigned int             timeoutMs,
                          size_t                   mesh_width,
                          size_t                   mesh_height,
                          Vertex                  *cudaDevVertptr,
                          // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
                          cudaStream_t             streamToRun);
Vertex *cudaImportVertexBuffer(void *sharedHandle, cudaExternalMemory_t &externalMemory, int meshWidth, int meshHeight);
void    cudaImportKeyedMutex(void *sharedHandle, cudaExternalSemaphore_t &extSemaphore);
#endif // !
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11/sinewave_cuda.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaStream_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaDevVertptr` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaExternalSemaphore_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuStatus` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaImportVertexBuffer` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaImportKeyedMutex` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `Device` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaError` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaAcquireSync` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaReleaseSync` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaExternalMemory_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGetDeviceCount` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuDevice` | Driver API の handle 境界です。context/module/function と error code を追います。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- CUDA Graph は一連の work を node と依存関係として再利用します。capture 対象と buffer lifetime を確認します。
- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
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
cmake --build build --target simpleD3D11
ctest --test-dir build -R simpleD3D11
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

The sample may display a window or produce/validate image-like output; exact visuals depend on platform support.
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
- graph capture 後に buffer lifetime や node dependency が変わったことを見落とす。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaStream_t` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
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
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
