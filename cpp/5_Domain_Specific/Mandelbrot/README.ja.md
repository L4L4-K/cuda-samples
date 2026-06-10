# Mandelbrot - Mandelbrot - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample uses CUDA to compute and display the Mandelbrot or Julia sets interactively. It also illustrates the use of "double single" arithmetic to improve precision when zooming a long way into the pattern. This sample uses double precision.  Thanks to Mark Granger of NewTek who submitted this code sample.!

Graphics Interop, Data Parallel Algorithms

Original README headings: `Mandelbrot - Mandelbrot`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/Mandelbrot` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `Mandelbrot` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/Mandelbrot`.
> **日本語**
> この sample の目的は、`Mandelbrot` の小さな実装を通して CUDA Graphs, Streams And Events, Synchronization And Atomics, Memory, Kernel Launch And Indexing を具体的に追うことです。
>
> **学習メモ**
> 最初に `Mandelbrot.cpp, Mandelbrot_cuda.cu, Mandelbrot_gold.cpp, Mandelbrot_gold.h, Mandelbrot_kernel.cuh` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `Mandelbrot.cpp`: Host-side setup, API calls, validation, and cleanup.
- `Mandelbrot_cuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `Mandelbrot_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `Mandelbrot_gold.h`: Host/device declarations, helper types, constants, or library wrappers.
- `Mandelbrot_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `Mandelbrot_kernel.h`: Host/device declarations, helper types, constants, or library wrappers.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `data/Mandelbrot_fp32.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/Mandelbrot_fp64.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/params.txt`: Input, reference, generated-data description, or documentation used by the sample.
- `data/referenceJulia_fp32.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/referenceJulia_fp64.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/sshot_lg.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/sshot_md.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/sshot_sm.JPG`: Input, reference, generated-data description, or documentation used by the sample.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `Mandelbrot.cpp` first and locate the host-side setup or Python entry point.
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

- `Mandelbrot.cpp`: focus on `CUDA`, `launch`, `cudaMemcpy`, `cudaMalloc`, `cudaGraphicsResource`.
- `Mandelbrot_cuda.cu`: focus on `blockDim`, `threadIdx`, `launch`, `blockIdx`, `gridDim`.
- `Mandelbrot_gold.cpp`: focus on control flow and helper functions.
- `Mandelbrot_gold.h`: focus on control flow and helper functions.
- `Mandelbrot_kernel.cuh`: focus on `CUDA`, `launch`.
- `Mandelbrot_kernel.h`: focus on control flow and helper functions.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/Mandelbrot/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(Mandelbrot LANGUAGES C CXX CUDA)

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

> JP: この抜粋は `cpp/5_Domain_Specific/Mandelbrot/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `Mandelbrot.cpp`

Source: cpp/5_Domain_Specific/Mandelbrot/Mandelbrot.cpp:30-48
```cpp
  Mandelbrot sample
  submitted by Mark Granger, NewTek

  CUDA 2.0 SDK - updated with double precision support
  CUDA 2.1 SDK - updated to demonstrate software block scheduling using
  atomics
  CUDA 2.2 SDK - updated with drawing of Julia sets by Konstantin Kolchin,
  NVIDIA
*/

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
```

> JP: この抜粋は `cpp/5_Domain_Specific/Mandelbrot/Mandelbrot.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/Mandelbrot/Mandelbrot.cpp:265-284
```cpp

            if (bUseOpenGL) {
                // DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_dst,
                // gl_PBO));
                // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
                checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
                size_t num_bytes;
                checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_dst, &num_bytes, cuda_pbo_resource));
            }

            // Get the anti-alias sub-pixel sample location
            GetSample(pass & 127, xs, ys);

            // Get the pixel scale and offset
            double s = scale / (double)imageW;
            double x = (xs - (double)imageW * 0.5f) * s + xOff;
            double y = (ys - (double)imageH * 0.5f) * s + yOff;

            // Run the mandelbrot generator
            // Use the adaptive sampling version when animating.
```

> JP: この抜粋は `cpp/5_Domain_Specific/Mandelbrot/Mandelbrot.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/Mandelbrot/Mandelbrot.cpp:501-520
```cpp

void cleanup()
{
    if (h_Src) {
        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(h_Src);
        h_Src = 0;
    }

    sdkStopTimer(&hTimer);
    sdkDeleteTimer(&hTimer);

    // DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
    // JP: この anchor では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glDeleteBuffers(1, &gl_PBO);
    glDeleteTextures(1, &gl_Tex);
    glDeleteProgramsARB(1, &gl_Shader);
```

> JP: この抜粋は `cpp/5_Domain_Specific/Mandelbrot/Mandelbrot.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/Mandelbrot/Mandelbrot.cpp:964-983
```cpp
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &gl_PBO);
        gl_PBO = 0;
    }

    // allocate new buffers
    h_Src = (uchar4 *)malloc(w * h * 4);

    printf("Creating GL texture...\n");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
    printf("Texture created.\n");

    printf("Creating PBO...\n");
```

> JP: この抜粋は `cpp/5_Domain_Specific/Mandelbrot/Mandelbrot.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `Mandelbrot_cuda.cu`

Source: cpp/5_Domain_Specific/Mandelbrot/Mandelbrot_cuda.cu:29-69
```cuda
#include <stdio.h>

#include "Mandelbrot_kernel.cuh"
#include "Mandelbrot_kernel.h"
#include "helper_cuda.h"

// The Mandelbrot CUDA GPU thread function

template <class T>
__global__ void Mandelbrot0(uchar4      *dst,
                            const int    imageW,
                            const int    imageH,
                            const int    crunch,
                            const T      xOff,
                            const T      yOff,
                            const T      xJP,
                            const T      yJP,
                            const T      scale,
                            const uchar4 colors,
                            const int    frame,
                            const int    animationFrame,
                            const int    gridWidth,
                            const int    numBlocks,
                            const bool   isJ)
{
    // loop until all blocks completed
    // JP: `blockIdx`, `gridDim`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    for (unsigned int blockIndex = blockIdx.x; blockIndex < numBlocks; blockIndex += gridDim.x) {
        unsigned int blockX = blockIndex % gridWidth;
        unsigned int blockY = blockIndex / gridWidth;

        // process this block
        const int ix = blockDim.x * blockX + threadIdx.x;
        const int iy = blockDim.y * blockY + threadIdx.y;

        if ((ix < imageW) && (iy < imageH)) {
            // Calculate the location
            const T xPos = (T)ix * scale + xOff;
            const T yPos = (T)iy * scale + yOff;

            // Calculate the Mandelbrot index for the current location
```

> JP: この抜粋は `cpp/5_Domain_Specific/Mandelbrot/Mandelbrot_cuda.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `Mandelbrot_gold.cpp`

Source: cpp/5_Domain_Specific/Mandelbrot/Mandelbrot_gold.cpp:29-47
```cpp
#include "Mandelbrot_gold.h"

#define ABS(n) ((n) < 0 ? -(n) : (n))

/* dfloat class declaration */
class dfloat
{
private:
    float val[2];

public:
    dfloat() { val[0] = val[1] = 0; }
    dfloat(float a, float b)
    {
        val[0] = a;
        val[1] = b;
    }
    dfloat(double b);
    inline float operator[](unsigned idx) const { return val[idx]; }
```

> JP: この抜粋は `cpp/5_Domain_Specific/Mandelbrot/Mandelbrot_gold.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `Mandelbrot_gold.h`

Source: cpp/5_Domain_Specific/Mandelbrot/Mandelbrot_gold.h:29-87
```cpp
#ifndef _MANDELBROT_GOLD_h_
#define _MANDELBROT_GOLD_h_

#include <vector_types.h>

extern "C" void RunMandelbrotGold0(uchar4      *dst,
                                   const int    imageW,
                                   const int    imageH,
                                   const int    crunch,
                                   const float  xOff,
                                   const float  yOff,
                                   const float  xJParam,
                                   const float  yJParam,
                                   const float  scale,
                                   const uchar4 colors,
                                   const int    frame,
                                   const int    animationFrame,
                                   const bool   isJulia);
extern "C" void RunMandelbrotDSGold0(uchar4      *dst,
                                     const int    imageW,
                                     const int    imageH,
                                     const int    crunch,
                                     const double xOff,
                                     const double yOff,
                                     const double xJParam,
                                     const double yJParam,
                                     const double scale,
                                     const uchar4 colors,
                                     const int    frame,
                                     const int    animationFrame,
                                     const bool   isJulia);
extern "C" void RunMandelbrotGold1(uchar4      *dst,
                                   const int    imageW,
                                   const int    imageH,
                                   const int    crunch,
                                   const float  xOff,
                                   const float  yOff,
                                   const float  xJParam,
                                   const float  yJParam,
                                   const float  scale,
                                   const uchar4 colors,
                                   const int    frame,
                                   const int    animationFrame,
                                   const bool   isJulia);
extern "C" void RunMandelbrotDSGold1(uchar4      *dst,
                                     const int    imageW,
                                     const int    imageH,
                                     const int    crunch,
                                     const double xOff,
                                     const double yOff,
                                     const double xJParam,
                                     const double yJParam,
                                     const double scale,
                                     const uchar4 colors,
                                     const int    frame,
                                     const int    animationFrame,
                                     const bool   isJulia);

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/Mandelbrot/Mandelbrot_gold.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `Mandelbrot_kernel.cuh`

Source: cpp/5_Domain_Specific/Mandelbrot/Mandelbrot_kernel.cuh:29-67
```cuda
#include <stdio.h>

#include "Mandelbrot_kernel.h"
#include "helper_cuda.h"

// The dimensions of the thread block
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

#define ABS(n) ((n) < 0 ? -(n) : (n))

// Double single functions based on DSFUN90 package:
// http://crd.lbl.gov/~dhbailey/mpdist/index.html

// This function sets the DS number A equal to the double precision floating
// point number B.
inline void dsdeq(float &a0, float &a1, double b)
{
    a0 = (float)b;
    a1 = (float)(b - a0);
} // dsdcp

// This function sets the DS number A equal to the single precision floating
// point number B.
__device__ inline void dsfeq(float &a0, float &a1, float b)
{
    a0 = b;
    a1 = 0.0f;
} // dsfeq

// This function computes c = a + b.
__device__ inline void dsadd(float &c0, float &c1, const float a0, const float a1, const float b0, const float b1)
{
    // Compute dsa + dsb using Knuth's trick.
    float t1 = a0 + b0;
    float e  = t1 - a0;
    float t2 = ((b0 - e) + (a0 - (t1 - e))) + a1 + b1;

    // The result is t1 + t2, after normalization.
```

> JP: この抜粋は `cpp/5_Domain_Specific/Mandelbrot/Mandelbrot_kernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `Mandelbrot_kernel.h`

Source: cpp/5_Domain_Specific/Mandelbrot/Mandelbrot_kernel.h:29-67
```cpp
#ifndef _MANDELBROT_KERNEL_h_
#define _MANDELBROT_KERNEL_h_

#include <vector_types.h>

extern "C" void RunMandelbrot0(uchar4      *dst,
                               const int    imageW,
                               const int    imageH,
                               const int    crunch,
                               const double xOff,
                               const double yOff,
                               const double xjp,
                               const double yjp,
                               const double scale,
                               const uchar4 colors,
                               const int    frame,
                               const int    animationFrame,
                               const int    mode,
                               const int    numSMs,
                               const bool   isJ,
                               int          version = 13);
extern "C" void RunMandelbrot1(uchar4      *dst,
                               const int    imageW,
                               const int    imageH,
                               const int    crunch,
                               const double xOff,
                               const double yOff,
                               const double xjp,
                               const double yjp,
                               const double scale,
                               const uchar4 colors,
                               const int    frame,
                               const int    animationFrame,
                               const int    mode,
                               const int    numSMs,
                               const bool   isJ,
                               int          version = 13);

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/Mandelbrot/Mandelbrot_kernel.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `gridDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaGLMapBufferObject` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGraphicsMapResources` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphicsResourceGetMappedPointer` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGLUnmapBufferObject` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGraphicsUnmapResources` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGLUnregisterBufferObject` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGraphicsUnregisterResource` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |

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
cmake --build build --target Mandelbrot
ctest --test-dir build -R Mandelbrot
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
- graph capture 後に buffer lifetime や node dependency が変わったことを見落とす。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `launch` の直前と直後で、どの memory/resource が有効になったかをメモする。
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
