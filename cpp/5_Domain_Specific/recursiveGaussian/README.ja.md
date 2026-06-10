# recursiveGaussian - Recursive Gaussian Filter - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample implements a Gaussian blur using Deriche's recursive method. The advantage of this method is that the execution time is independent of the filter width.

Graphics Interop, Image Processing

Original README headings: `recursiveGaussian - Recursive Gaussian Filter`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/recursiveGaussian` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `recursiveGaussian` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/recursiveGaussian`.
> **日本語**
> この sample の目的は、`recursiveGaussian` の小さな実装を通して CUDA Graphs, Shared Memory, Streams And Events, Synchronization And Atomics, Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `recursiveGaussian.cpp, recursiveGaussian_cuda.cu, recursiveGaussian_kernel.cuh` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `data/ref_10.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/ref_14.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/ref_18.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/ref_22.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/teapot512.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `recursiveGaussian.cpp`: Host-side setup, API calls, validation, and cleanup.
- `recursiveGaussian_cuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `recursiveGaussian_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `recursiveGaussian.cpp` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Capture or build CUDA Graph nodes, instantiate the graph, then launch the executable graph.
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

- `recursiveGaussian.cpp`: focus on `CUDA`, `cudaFree`, `cudaDeviceSynchronize`, `cudaMalloc`, `launch`.
- `recursiveGaussian_cuda.cu`: focus on `launch`, `CUDA`.
- `recursiveGaussian_kernel.cuh`: focus on `threadIdx`, `blockIdx`, `__shared__`, `blockDim`, `launch`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/recursiveGaussian/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(recursiveGaussian LANGUAGES C CXX CUDA)

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

> JP: この抜粋は `cpp/5_Domain_Specific/recursiveGaussian/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `recursiveGaussian.cpp`

Source: cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian.cpp:30-48
```cpp
  Recursive Gaussian filter
  sgreen 8/1/08

  This code sample implements a Gaussian blur using Deriche's recursive method:
  http://citeseer.ist.psu.edu/deriche93recursively.html

  This is similar to the box filter sample in the SDK, but it uses the previous
  outputs of the filter as well as the previous inputs. This is also known as an
  IIR (infinite impulse response) filter, since its response to an input impulse
  can last forever.

  The main advantage of this method is that the execution time is independent of
  the filter width.

  The GPU processes columns of the image in parallel. To avoid uncoalesced reads
  for the row pass we transpose the image and then transpose it back again
  afterwards.

  The implementation is based on code from the CImg library:
```

> JP: この抜粋は `cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian.cpp:158-177
```cpp
    sdkStartTimer(&timer);

    // execute filter, writing results to pbo
    unsigned int *d_result;
    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_result, &num_bytes, cuda_vbo_resource));
    gaussianFilterRGBA(d_img, d_result, d_temp, width, height, sigma, order, nthreads);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

    // load texture from pbo
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, texid);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

```

> JP: この抜粋は `cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian.cpp:204-223
```cpp

void cleanup()
{
    sdkDeleteTimer(&timer);

    // JP: `cudaFree`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。 ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFree(d_img));
    checkCudaErrors(cudaFree(d_temp));

    if (!runBenchmark) {
        if (pbo) {
            // unregister this buffer object with CUDA
            // JP: この anchor では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
            checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
            glDeleteBuffers(1, &pbo);
        }

        if (texid) {
            glDeleteTextures(1, &texid);
        }
```

> JP: この抜粋は `cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian.cpp:300-319
```cpp
{
    unsigned int size = width * height * sizeof(unsigned int);

    // allocate device memory
    // JP: この anchor では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaMalloc((void **)&d_img, size));
    checkCudaErrors(cudaMalloc((void **)&d_temp, size));

    // JP: `cudaMemcpy`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice));

    sdkCreateTimer(&timer);
}

void initGLBuffers()
{
    // create pixel buffer object to store final image
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(GLubyte) * 4, h_img, GL_STREAM_DRAW_ARB);
```

> JP: この抜粋は `cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `recursiveGaussian_cuda.cu`

Source: cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian_cuda.cu:30-48
```cuda
  Recursive Gaussian filter
  sgreen 8/1/08

  This code sample implements a Gaussian blur using Deriche's recursive method:
  http://citeseer.ist.psu.edu/deriche93recursively.html

  This is similar to the box filter sample in the SDK, but it uses the previous
  outputs of the filter as well as the previous inputs. This is also known as an
  IIR (infinite impulse response) filter, since its response to an input impulse
  can last forever.

  The main advantage of this method is that the execution time is independent of
  the filter width.

  The GPU processes columns of the image in parallel. To avoid uncoalesced reads
  for the row pass we transpose the image and then transpose it back again
  afterwards.

  The implementation is based on code from the CImg library:
```

> JP: この抜粋は `cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian_cuda.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian_cuda.cu:70-89
```cuda
extern "C" void transpose(uint *d_src, uint *d_dest, uint width, int height)
{
    dim3 grid(iDivUp(width, BLOCK_DIM), iDivUp(height, BLOCK_DIM), 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    d_transpose<<<grid, threads>>>(d_dest, d_src, width, height);
    getLastCudaError("Kernel execution failed");
}

/*
  Perform Gaussian filter on a 2D image using CUDA

  Parameters:
  d_src  - pointer to input image in device memory
  d_dest - pointer to destination image in device memory
  d_temp - pointer to temporary storage in device memory
  width  - image width
  height - image height
  sigma  - sigma of Gaussian
  order  - filter order (0, 1 or 2)
```

> JP: この抜粋は `cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian_cuda.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `recursiveGaussian_kernel.cuh`

Source: cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian_kernel.cuh:30-70
```cuda
  Recursive Gaussian filter
*/

#ifndef _RECURSIVEGAUSSIAN_KERNEL_CU_
#define _RECURSIVEGAUSSIAN_KERNEL_CU_

#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace cg = cooperative_groups;

#include <helper_cuda.h>
#include <helper_math.h>

#define BLOCK_DIM     16
#define CLAMP_TO_EDGE 1

// Transpose kernel (see transpose CUDA Sample for details)
__global__ void d_transpose(uint *odata, uint *idata, int width, int height)
{
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block cta = cg::this_thread_block();

    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    __shared__ uint block[BLOCK_DIM][BLOCK_DIM + 1];

    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

    if ((xIndex < width) && (yIndex < height)) {
        unsigned int index_in           = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    // JP: sync: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    cg::sync(cta);

```

> JP: この抜粋は `cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian_kernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaGraphicsMapResources` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphicsResourceGetMappedPointer` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphicsUnmapResources` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphicsUnregisterResource` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphicsGLRegisterBuffer` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGetDevice` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGetDeviceProperties` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- CUDA Graph は一連の work を node と依存関係として再利用します。capture 対象と buffer lifetime を確認します。
- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
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
cmake --build build --target recursiveGaussian
ctest --test-dir build -R recursiveGaussian
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
- shared memory を書いた thread と読む thread の間に必要な barrier を見落とす。
- different stream 間に依存があるのに event や explicit sync を置かない。
- graph capture 後に buffer lifetime や node dependency が変わったことを見落とす。

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
- graph node の依存関係を箇条書きにし、どの buffer lifetime が graph 実行全体をまたぐか確認する。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [CUDA Graphs](../../../docs_ja/themes/graphs.md): capture、node dependency、replay、graph update を読むための基礎です。
- [Shared Memory](../../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
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
