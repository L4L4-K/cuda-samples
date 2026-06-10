# randomFog - Random Fog - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample illustrates pseudo- and quasi- random numbers produced by CURAND.

3D Graphics, CURAND Library

Original README headings: `randomFog - Random Fog`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/4_CUDA_Libraries/randomFog` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `randomFog` as a focused example of the CUDA concepts used in `cpp/4_CUDA_Libraries/randomFog`.
> **日本語**
> この sample の目的は、`randomFog` の小さな実装を通して CUDA Libraries, CUDA Graphs, Streams And Events, Synchronization And Atomics, Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `randomFog.cpp, rng.cpp, rng.h` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `data/ref_randomFog.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `randomFog.cpp`: Host-side setup, API calls, validation, and cleanup.
- `rng.cpp`: Host-side setup, API calls, validation, and cleanup.
- `rng.h`: Host/device declarations, helper types, constants, or library wrappers.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `randomFog.cpp` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Create library handles, descriptors, plans, or workspaces before the library call.
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

- `randomFog.cpp`: focus on `CUDA`, `curand`, `cuBe`, `CURAND`.
- `rng.cpp`: focus on `curandResult`, `cudaResult`, `CURAND_STATUS_SUCCESS`, `CUDA`, `curandStatus_t`.
- `rng.h`: focus on `curandGenerator_t`, `curand`, `CUDA`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/4_CUDA_Libraries/randomFog/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(randomFog LANGUAGES CXX)

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

> JP: この抜粋は `cpp/4_CUDA_Libraries/randomFog/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/randomFog/CMakeLists.txt:65-84
```cmake
        )

        target_link_libraries(randomFog
            ${OPENGL_LIBRARIES}
            ${GLUT_LIBRARIES}
            # JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
            CUDA::curand
            CUDA::cudart
        )
        # Need to add X11 and other libraries for Debian13 or later explicitly
        if(DEBIAN)
            target_link_libraries(randomFog
                X11
                Xi
                Xxf86vm
                Xext
            )
        endif()

        # Copy data files to the output directory
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/randomFog/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `randomFog.cpp`

Source: cpp/4_CUDA_Libraries/randomFog/randomFog.cpp:30-48
```cpp
#include <helper_gl.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA Library Headers
#include <cuda_gl_interop.h>
#include <curand.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <rendercheck_gl.h>

// System includes
#include <iomanip>
#include <math.h>
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/randomFog/randomFog.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/randomFog/randomFog.cpp:282-301
```cpp
    // Check if shape is visible
    if (x == 0 || y == 0) {
        return;
    }

    // Set a new projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // Adjust fit
    if (y > x) {
        xScale = 1.0f;
        yScale = (float)y / x;
    }
    else {
        xScale = (float)x / y;
        yScale = 1.0f;
    }

    // Angle of view:40 degrees
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/randomFog/randomFog.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/randomFog/randomFog.cpp:638-657
```cpp
    ss << "\t" << setw(10) << "q/[ESC]"
       << "Quit the application.\n\n";
    puts(ss.str().c_str());
}

int main(int argc, char **argv)
{
    using std::runtime_error;

    try {
        bool bQA = false;

        // Open the log file
        printf("Random Fog\n");
        printf("==========\n\n");

        // Check QA mode
        if (checkCmdLineFlag(argc, (const char **)argv, "qatest")) {
            bQA = true;

```

> JP: この抜粋は `cpp/4_CUDA_Libraries/randomFog/randomFog.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `rng.cpp`

Source: cpp/4_CUDA_Libraries/randomFog/rng.cpp:32-50
```cpp
#include "rng.h"

#include <curand.h>
#include <sstream>
#include <stdexcept>

// Shared Library Test Functions
#include <helper_cuda.h>
#include <helper_timer.h>

const unsigned int RNG::s_maxQrngDimensions = 20000;

RNG::RNG(unsigned long prngSeed, unsigned int qrngDimensions, unsigned int nSamples)
    : m_prngSeed(prngSeed)
    , m_qrngDimensions(qrngDimensions)
    , m_nSamplesBatchTarget(nSamples)
    , m_nSamplesRemaining(0)
{
    using std::invalid_argument;
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/randomFog/rng.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/randomFog/rng.cpp:65-84
```cpp

    if (m_nSamplesBatchTarget < s_maxQrngDimensions) {
        throw invalid_argument("RNG batch size must be greater than RNG::s_maxQrngDimensions");
    }

    // JP: `curandStatus_t`, `curandResult`: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
    curandStatus_t curandResult;
    cudaError_t    cudaResult;

    // Allocate sample array in host mem
    m_h_samples = (float *)malloc(m_nSamplesBatchTarget * sizeof(float));

    if (m_h_samples == NULL) {
        throw runtime_error("Could not allocate host memory for RNG::m_h_samples");
    }

    // Allocate sample array in device mem
    // JP: `cudaResult`, `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    cudaResult = cudaMalloc((void **)&m_d_samples, m_nSamplesBatchTarget * sizeof(float));

```

> JP: この抜粋は `cpp/4_CUDA_Libraries/randomFog/rng.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/randomFog/rng.cpp:129-148
```cpp
    curandDestroyGenerator(m_prng);
    curandDestroyGenerator(m_qrng);
    curandDestroyGenerator(m_sqrng);

    if (m_d_samples) {
        // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        cudaFree(m_d_samples);
    }

    if (m_h_samples) {
        free(m_h_samples);
    }
}

void RNG::generateBatch(void)
{
    using std::runtime_error;
    using std::string;

    cudaError_t    cudaResult;
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/randomFog/rng.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/randomFog/rng.cpp:157-176
```cpp
        msg += curandResult;
        throw runtime_error(msg);
    }

    // Copy random numbers to host
    // JP: `cudaResult`, `cudaMemcpy`, `cudaMemcpyDeviceToHost`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    cudaResult = cudaMemcpy(m_h_samples, m_d_samples, m_nSamplesBatchActual * sizeof(float), cudaMemcpyDeviceToHost);

    if (cudaResult != cudaSuccess) {
        string msg("Could not copy random numbers to host: ");
        msg += cudaGetErrorString(cudaResult);
        throw runtime_error(msg);
    }
}

float RNG::getNextU01(void)
{
    if (m_nSamplesRemaining == 0) {
        generateBatch();
        m_nSamplesRemaining = m_nSamplesBatchActual;
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/randomFog/rng.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `rng.h`

Source: cpp/4_CUDA_Libraries/randomFog/rng.h:29-74
```cpp
#include <curand.h>
#include <string>

// RNGs
class RNG
{
public:
    enum RngType { Pseudo, Quasi, ScrambledQuasi };
    RNG(unsigned long prngSeed, unsigned int qrngDimensions, unsigned int nSamples);
    virtual ~RNG();

    float getNextU01(void);
    void  getInfoString(std::string &msg);
    void  selectRng(RngType type);
    void  resetSeed(void);
    void  resetDimensions(void);
    void  incrementDimensions(void);

private:
    // Generators
    // JP: `curandGenerator_t`: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
    curandGenerator_t *m_pCurrent;
    curandGenerator_t  m_prng;
    curandGenerator_t  m_qrng;
    curandGenerator_t  m_sqrng;

    // Parameters
    unsigned long m_prngSeed;
    unsigned int  m_qrngDimensions;

    // Batches
    const unsigned int m_nSamplesBatchTarget;
    unsigned int       m_nSamplesBatchActual;
    unsigned int       m_nSamplesRemaining;
    void               generateBatch(void);

    // Helpers
    void updateDimensions(void);
    void setBatchSize(void);

    // Buffers
    float *m_h_samples;
    float *m_d_samples;

    static const unsigned int s_maxQrngDimensions;
};
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/randomFog/rng.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `curandResult` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaResult` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CURAND_STATUS_SUCCESS` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CURAND` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `curand` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `curandStatus_t` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `curandGenerator_t` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaGetErrorString` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `curandCreateGenerator` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `curandDestroyGenerator` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `curandSetGeneratorOffset` | Driver API の handle 境界です。context/module/function と error code を追います。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- CUDA Graph は一連の work を node と依存関係として再利用します。capture 対象と buffer lifetime を確認します。
- library sample では handle、descriptor、plan、workspace が GPU work の外側の resource です。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。
- 同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。

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
cmake --build build --target randomFog
ctest --test-dir build -R randomFog
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

The sample validates the library result against a CPU/reference path or reports the documented success status.
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
- leading dimension、stride、descriptor、workspace size を host 配列の見た目だけで判断する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `curandResult` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- graph node の依存関係を箇条書きにし、どの buffer lifetime が graph 実行全体をまたぐか確認する。
- handle/descriptor/workspace の作成、利用、破棄を対応表にする。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [CUDA Libraries](../../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。
- [CUDA Graphs](../../../docs_ja/themes/graphs.md): capture、node dependency、replay、graph update を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
