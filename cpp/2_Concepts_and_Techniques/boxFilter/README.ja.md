# boxFilter - Box Filter - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

Fast image box filter using CUDA with OpenGL rendering.

Graphics Interop, Image Processing

Original README headings: `boxFilter - Box Filter`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/2_Concepts_and_Techniques/boxFilter` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `boxFilter` as a focused example of the CUDA concepts used in `cpp/2_Concepts_and_Techniques/boxFilter`.
> **日本語**
> この sample の目的は、`boxFilter` の小さな実装を通して CUDA Graphs, Streams And Events, Synchronization And Atomics, Memory, Kernel Launch And Indexing を具体的に追うことです。
>
> **学習メモ**
> 最初に `boxFilter.cpp, boxFilter_cpu.cpp, boxFilter_kernel.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `boxFilter.cpp`: Host-side setup, API calls, validation, and cleanup.
- `boxFilter_cpu.cpp`: Host-side setup, API calls, validation, and cleanup.
- `boxFilter_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `data/ref_14.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/ref_22.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/teapot1024.ppm`: Input, reference, generated-data description, or documentation used by the sample.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `boxFilter.cpp` first and locate the host-side setup or Python entry point.
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

- `boxFilter.cpp`: focus on `CUDA`, `cudaMalloc`, `cudaDeviceSynchronize`, `launch`, `cudaGraphicsResource`.
- `boxFilter_cpu.cpp`: focus on control flow and helper functions.
- `boxFilter_kernel.cu`: focus on `cudaTextureObject_t`, `blockIdx`, `blockDim`, `threadIdx`, `CUDA`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/2_Concepts_and_Techniques/boxFilter/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(boxFilter LANGUAGES C CXX CUDA)

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

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/boxFilter/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `boxFilter.cpp`

Source: cpp/2_Concepts_and_Techniques/boxFilter/boxFilter.cpp:30-48
```cpp
  Image box filtering example

  This sample uses CUDA to perform a simple box filter on an image
  and uses OpenGL to display the results.

  It processes rows and columns of the image in parallel.

  The box filter is implemented such that it has a constant cost,
  regardless of the filter width.

  Press '=' to increment the filter radius, '-' to decrease it

  Version 1.1 - modified to process 8-bit RGBA images
*/

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/boxFilter/boxFilter.cpp:177-196
```cpp

    // execute filter, writing results to pbo
    unsigned int *d_result;

    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_result, &num_bytes, cuda_pbo_resource));
    boxFilterRGBA(d_temp, d_result, width, height, filter_radius, iterations, nthreads, kernel_timer);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    // OpenGL display code path
    {
        glClear(GL_COLOR_BUFFER_BIT);

        // load texture from pbo
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
        glBindTexture(GL_TEXTURE_2D, texid);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/boxFilter/boxFilter.cpp:299-335
```cpp
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void initCuda(bool useRGBA)
{
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc((void **)&d_temp, (width * height * sizeof(unsigned int))));

    // Refer to boxFilter_kernel.cu for implementation
    initTexture(width, height, h_img, useRGBA);

    sdkCreateTimer(&timer);
    sdkCreateTimer(&kernel_timer);
}

void cleanup()
{
    sdkDeleteTimer(&timer);
    sdkDeleteTimer(&kernel_timer);

    if (h_img) {
        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(h_img);
        h_img = NULL;
    }

    if (d_temp) {
        // JP: この anchor では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
        cudaFree(d_temp);
        d_temp = NULL;
    }

    // Refer to boxFilter_kernel.cu for implementation
    freeTextures();

    // JP: この anchor では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/boxFilter/boxFilter.cpp:426-445
```cpp
    // JP: この anchor では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaMalloc((void **)&d_result, width * height * sizeof(unsigned int)));

    // warm-up
    boxFilterRGBA(d_temp, d_temp, width, height, filter_radius, iterations, nthreads, kernel_timer);
    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaDeviceSynchronize());

    sdkStartTimer(&kernel_timer);
    // Start round-trip timer and process iCycles loops on the GPU
    iterations                = 1; // standard 1-pass filtering
    const int iCycles         = 150;
    double    dProcessingTime = 0.0;
    printf("\nRunning BoxFilterGPU for %d cycles...\n\n", iCycles);

    for (int i = 0; i < iCycles; i++) {
        dProcessingTime +=
            boxFilterRGBA(d_temp, d_temp, width, height, filter_radius, iterations, nthreads, kernel_timer);
    }

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `boxFilter_cpu.cpp`

Source: cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_cpu.cpp:31-49
```cpp
extern "C" void computeGold(float *id, float *od, int w, int h, int r);

// CPU implementation
void hboxfilter_x(float *id, float *od, int w, int h, int r)
{
    float scale = 1.0f / (2 * r + 1);

    for (int y = 0; y < h; y++) {
        float t;
        // do left edge
        t = id[y * w] * r;

        for (int x = 0; x < r + 1; x++) {
            t += id[y * w + x];
        }

        od[y * w] = t * scale;

        for (int x = 1; x < r + 1; x++) {
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_cpu.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `boxFilter_kernel.cu`

Source: cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_kernel.cu:29-47
```cuda
#ifndef _BOXFILTER_KERNEL_CH_
#define _BOXFILTER_KERNEL_CH_

#include <helper_functions.h>
#include <helper_math.h>

cudaTextureObject_t tex;
cudaTextureObject_t texTempArray;
cudaTextureObject_t rgbaTex;
cudaTextureObject_t rgbaTexTempArray;
cudaArray          *d_array, *d_tempArray;

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host
// call returns an error
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_kernel.cu:55-74
```cuda

/*
  Perform a fast box filter using the sliding window method.

  As the kernel moves from left to right, we add in the contribution of the
  new sample on the right, and subtract the value of the exiting sample on the
  left. This only requires 2 adds and a mul per output value, independent of the
  filter radius. The box filter is separable, so to perform a 2D box filter we
  perform the filter in the x direction, followed by the same filter in the y
  direction. Applying multiple iterations of the box filter converges towards a
  Gaussian blur. Using CUDA, rows or columns of the image are processed in
  parallel. This version duplicates edge pixels.

  Note that the x (row) pass suffers from uncoalesced global memory reads,
  since each thread is reading from a different row. For this reason it is
  better to use texture lookups for the x pass.
  The y (column) pass is perfectly coalesced.

  Parameters
  id - pointer to input data in global memory
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_kernel.cu:162-181
```cuda
    }
}

__global__ void d_boxfilter_x_global(float *id, float *od, int w, int h, int r)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
    d_boxfilter_x(&id[y * w], &od[y * w], w, h, r);
}

__global__ void d_boxfilter_y_global(float *id, float *od, int w, int h, int r)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    d_boxfilter_y(&id[x], &od[x], w, h, r);
}

// texture version
// texture fetches automatically clamp to edge of image
__global__ void d_boxfilter_x_tex(float *od, int w, int h, int r, cudaTextureObject_t tex)
{
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_kernel.cu:321-340
```cuda
    }
    // JP: `cudaMallocArray`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMallocArray(&d_array, &channelDesc, width, height));

    size_t bytesPerElem = (useRGBA ? sizeof(uchar4) : sizeof(float));
    checkCudaErrors(cudaMemcpy2DToArray(
        // JP: `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
        d_array, 0, 0, pImage, width * bytesPerElem, width * bytesPerElem, height, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMallocArray(&d_tempArray, &channelDesc, width, height));

    // set texture parameters
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType         = cudaResourceTypeArray;
    texRes.res.array.array = d_array;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaTextureObject_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaDestroyTextureObject` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaAddressModeWrap` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaCreateTextureObject` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMallocArray` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaResourceDesc` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaTextureDesc` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |

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
cmake --build build --target boxFilter
ctest --test-dir build -R boxFilter
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

- `cudaDeviceSynchronize` の直前と直後で、どの memory/resource が有効になったかをメモする。
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
