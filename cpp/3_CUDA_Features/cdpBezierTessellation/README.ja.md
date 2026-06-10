# cdpBezierTessellation - Bezier Line Tessellation (CUDA Dynamic Parallelism) - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates bezier tessellation of lines implemented using CUDA Dynamic Parallelism.  This sample requires devices with compute capability 3.5 or higher.

CUDA Dynamic Parallelism

Original README headings: `cdpBezierTessellation - Bezier Line Tessellation (CUDA Dynamic Parallelism)`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/3_CUDA_Features/cdpBezierTessellation` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `cdpBezierTessellation` as a focused example of the CUDA concepts used in `cpp/3_CUDA_Features/cdpBezierTessellation`.
> **日本語**
> この sample の目的は、`cdpBezierTessellation` の小さな実装を通して Memory, Kernel Launch And Indexing, Execution Model, Debugging, Profiling, And Testing を具体的に追うことです。
>
> **学習メモ**
> 最初に `BezierLineCDP.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `BezierLineCDP.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `BezierLineCDP.cu` first and locate the host-side setup or Python entry point.
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

- `BezierLineCDP.cu`: focus on `threadIdx`, `blockDim`, `blockIdx`, `cudaMalloc`, `cudaFree`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `BezierLineCDP.cu`

Source: cpp/3_CUDA_Features/cdpBezierTessellation/BezierLineCDP.cu:29-47
```cuda
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <string.h>

__forceinline__ __device__ float2 operator+(float2 a, float2 b)
{
    float2 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

__forceinline__ __device__ float2 operator-(float2 a, float2 b)
{
    float2 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
```

> JP: この抜粋は `cpp/3_CUDA_Features/cdpBezierTessellation/BezierLineCDP.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/cdpBezierTessellation/BezierLineCDP.cu:65-84
```cuda
    int     nVertices;
};

__global__ void computeBezierLinePositions(int lidx, BezierLine *bLines, int nTessPoints)
{
    // JP: `threadIdx`, `blockDim`, `blockIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < nTessPoints) {
        float u   = (float)idx / (float)(nTessPoints - 1);
        float omu = 1.0f - u;

        float B3u[3];

        B3u[0] = omu * omu;
        B3u[1] = 2.0f * u * omu;
        B3u[2] = u * u;

        float2 position = {0, 0};

```

> JP: この抜粋は `cpp/3_CUDA_Features/cdpBezierTessellation/BezierLineCDP.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/cdpBezierTessellation/BezierLineCDP.cu:100-135
```cuda
                        / length(bLines[lidx].CP[2] - bLines[lidx].CP[0]);
        int nTessPoints = min(max((int)(curvature * 16.0f), 4), MAX_TESSELLATION);

        if (bLines[lidx].vertexPos == NULL) {
            bLines[lidx].nVertices = nTessPoints;
            // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
            cudaMalloc((void **)&bLines[lidx].vertexPos, nTessPoints * sizeof(float2));
        }

        // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
        computeBezierLinePositions<<<ceilf((float)bLines[lidx].nVertices / 32.0f), 32>>>(
            lidx, bLines, bLines[lidx].nVertices);
    }
}

__global__ void freeVertexMem(BezierLine *bLines, int nLines)
{
    // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    int lidx = threadIdx.x + blockDim.x * blockIdx.x;

    if (lidx < nLines)
        // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        cudaFree(bLines[lidx].vertexPos);
}

unsigned int checkCapableSM35Device(int argc, char **argv)
{
    // Get device properties
    cudaDeviceProp properties;
    int            device_count = 0, device = -1;

    if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
        device = getCmdLineArgumentInt(argc, (const char **)argv, "device");

        cudaDeviceProp properties;
        checkCudaErrors(cudaGetDeviceProperties(&properties, device));
```

> JP: この抜粋は `cpp/3_CUDA_Features/cdpBezierTessellation/BezierLineCDP.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/cdpBezierTessellation/BezierLineCDP.cu:169-188
```cuda
    return EXIT_SUCCESS;
}

#define N_LINES   256
#define BLOCK_DIM 64
int main(int argc, char **argv)
{
    BezierLine *bLines_h = new BezierLine[N_LINES];

    float2 last = {0, 0};

    for (int i = 0; i < N_LINES; i++) {
        bLines_h[i].CP[0] = last;

        for (int j = 1; j < 3; j++) {
            bLines_h[i].CP[j].x = (float)rand() / (float)RAND_MAX;
            bLines_h[i].CP[j].y = (float)rand() / (float)RAND_MAX;
        }

        last                  = bLines_h[i].CP[2];
```

> JP: この抜粋は `cpp/3_CUDA_Features/cdpBezierTessellation/BezierLineCDP.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `CMakeLists.txt`

Source: cpp/3_CUDA_Features/cdpBezierTessellation/CMakeLists.txt:1-46
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(cdpBezierTessellation LANGUAGES C CXX CUDA)

# JP: `find_package`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The aarch64/sbsa_aarch64 CUDA toolkit are support on Tegra since 13.0, so need to check which version of the toolkit is installed
string(FIND "${CUDAToolkit_INCLUDE_DIRS}" "aarch64-linux" _aarch64_linux_ctk)
string(FIND "${CUDAToolkit_INCLUDE_DIRS}" "aarch64-qnx" _aarch64_qnx_ctk)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" AND (NOT _aarch64_linux_ctk EQUAL -1 OR NOT _aarch64_qnx_ctk EQUAL -1))
    set(CMAKE_CUDA_ARCHITECTURES 87 110)
else()
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 100 110 120)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for cdpBezierTessellation
add_executable(cdpBezierTessellation BezierLineCDP.cu)

target_compile_options(cdpBezierTessellation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cdpBezierTessellation PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(cdpBezierTessellation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/3_CUDA_Features/cdpBezierTessellation/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaGetDeviceProperties` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaGetDeviceCount` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaDeviceProp` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMemcpyHostToDevice` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target cdpBezierTessellation
ctest --test-dir build -R cdpBezierTessellation
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

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaMalloc` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
