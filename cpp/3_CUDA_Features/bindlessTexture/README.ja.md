# bindlessTexture - Bindless Texture - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This example demonstrates use of cudaSurfaceObject, cudaTextureObject, and MipMap support in CUDA.  A GPU with Compute Capability SM 3.0 is required to run the sample.

Graphics Interop, Texture

Original README headings: `bindlessTexture - Bindless Texture`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/3_CUDA_Features/bindlessTexture` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `bindlessTexture` as a focused example of the CUDA concepts used in `cpp/3_CUDA_Features/bindlessTexture`.
> **日本語**
> この sample の目的は、`bindlessTexture` の小さな実装を通して CUDA Graphs, Streams And Events, Synchronization And Atomics, Memory, Kernel Launch And Indexing を具体的に追うことです。
>
> **学習メモ**
> 最初に `bindlessTexture.cpp, bindlessTexture.h, bindlessTexture_kernel.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `bindlessTexture.cpp`: Host-side setup, API calls, validation, and cleanup.
- `bindlessTexture.h`: Host/device declarations, helper types, constants, or library wrappers.
- `bindlessTexture_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `data/flower.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/person.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/ref_bindlessTexture.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `data/sponge.ppm`: Input, reference, generated-data description, or documentation used by the sample.
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
- Open `bindlessTexture.cpp` first and locate the host-side setup or Python entry point.
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

- `bindlessTexture.cpp`: focus on `CUDA`, `launch`, `cudaTextureObjects`, `cudaExtent`, `cudaGraphicsResource`.
- `bindlessTexture.h`: focus on `CUDA`, `cudaExtent`, `cudaResourceType`, `cudaArray_t`, `cudaMipmappedArray_t`.
- `bindlessTexture_kernel.cu`: focus on `cudaAddressModeClamp`, `cudaTextureObject_t`, `cudaResourceDesc`, `cudaTextureDesc`, `blockIdx`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/3_CUDA_Features/bindlessTexture/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(bindlessTexture LANGUAGES C CXX CUDA)

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

> JP: この抜粋は `cpp/3_CUDA_Features/bindlessTexture/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bindlessTexture.cpp`

Source: cpp/3_CUDA_Features/bindlessTexture/bindlessTexture.cpp:30-48
```cpp
  Bindless Texture/Surface

  This sample generates a few 2D textures and uses cudaTextureObjects to
  perform pseudo virtual texturing for display. One 2D texture stores
  references to other textures.
  Furthermore use of mip mapping is shown using both cudaTextureObjects
  and cudaSurfaceObjects.

  Look into the bindlessTexture_kernel.cu file for most relevant code.
*/

#include <helper_gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
```

> JP: この抜粋は `cpp/3_CUDA_Features/bindlessTexture/bindlessTexture.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/bindlessTexture/bindlessTexture.cpp:124-143
```cpp
// render image using CUDA
void render()
{
    // map PBO to get CUDA device pointer
    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes, cuda_pbo_resource));

    // call CUDA kernel, writing results to PBO
    renderAtlasImage(windowGridSize, windowBlockSize, d_output, windowSize.x, windowSize.y, lod);

    getLastCudaError("render_kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// display results using OpenGL (called by GLUT)
void display()
{
```

> JP: この抜粋は `cpp/3_CUDA_Features/bindlessTexture/bindlessTexture.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/bindlessTexture/bindlessTexture.cpp:260-279
```cpp
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    uchar *data = (uchar *)malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

    printf("Read '%s', %zu bytes\n", filename, read);

    return data;
}

void initGL(int *argc, char **argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(windowSize.x, windowSize.y);
```

> JP: この抜粋は `cpp/3_CUDA_Features/bindlessTexture/bindlessTexture.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/bindlessTexture/bindlessTexture.cpp:297-333
```cpp
    checkCudaErrors(cudaMalloc((void **)&d_output, windowBytes));

    // render the volumeData
    renderAtlasImage(windowGridSize, windowBlockSize, d_output, windowSize.x, windowSize.y, lod);

    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("render_kernel failed");

    void *h_output = malloc(windowBytes);
    // JP: `cudaMemcpy`, `cudaMemcpyDeviceToHost`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpy(h_output, d_output, windowBytes, cudaMemcpyDeviceToHost));
    sdkDumpBin(h_output, (unsigned int)windowBytes, "bindlessTexture.bin");

    // JP: validation: GPU result を CPU/reference と比較する検証地点です。失敗時は transfer、indexing、sync の順に疑います。
    bool bTestResult = sdkCompareBin2BinFloat("bindlessTexture.bin",
                                              sdkFindFilePath(ref_file, exec_path),
                                              windowSize.x * windowSize.y,
                                              MAX_EPSILON_ERROR,
                                              THRESHOLD,
                                              exec_path);

    // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFree(d_output));
    free(h_output);
    deinitAtlasAndImages();

    sdkStopTimer(&timer);
    sdkDeleteTimer(&timer);

    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void loadImageData(const char *exe_path)
{
    std::vector<Image> images;

```

> JP: この抜粋は `cpp/3_CUDA_Features/bindlessTexture/bindlessTexture.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bindlessTexture.h`

Source: cpp/3_CUDA_Features/bindlessTexture/bindlessTexture.h:29-67
```cpp
#ifndef _BINDLESSTEXTURE_CU_
#define _BINDLESSTEXTURE_CU_

// includes, cuda
#include <cuda_runtime.h>
#include <vector_types.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <vector_types.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

#pragma pack(push, 4)
struct Image
{
    void                *h_data;
    cudaExtent           size;
    cudaResourceType     type;
    cudaArray_t          dataArray;
    cudaMipmappedArray_t mipmapArray;
    cudaTextureObject_t  textureObject;

    Image() { memset(this, 0, sizeof(Image)); }
};
#pragma pack(pop)

inline void _checkHost(bool test, const char *condition, const char *file, int line, const char *func)
{
    if (!test) {
        fprintf(stderr, "HOST error at %s:%d (%s) \"%s\" \n", file, line, condition, func);
        exit(EXIT_FAILURE);
    }
}

#define checkHost(condition) _checkHost(condition, #condition, __FILE__, __LINE__, __FUNCTION__)

#endif
```

> JP: この抜粋は `cpp/3_CUDA_Features/bindlessTexture/bindlessTexture.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bindlessTexture_kernel.cu`

Source: cpp/3_CUDA_Features/bindlessTexture/bindlessTexture_kernel.cu:30-49
```cuda
  This sample has two kernels, one doing the rendering every frame, and
  another one used to generate the mip map levels at startup.

  For rendering we use a "virtual" texturing approach, where one 2d texture
  stores pointers to the actual textures used. This can be achieved by the
  new cudaTextureObject introduced in CUDA 5.0 and requiring sm3+ hardware.

  The mipmap generation kernel uses cudaSurfaceObject and cudaTextureObject
  passed as kernel arguments to compute the higher mip map level based on
  the lower.
*/

#ifndef _BINDLESSTEXTURE_KERNEL_CU_
#define _BINDLESSTEXTURE_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
```

> JP: この抜粋は `cpp/3_CUDA_Features/bindlessTexture/bindlessTexture_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/bindlessTexture/bindlessTexture_kernel.cu:90-109
```cuda
// the atlas texture stores the 64 bit cudaTextureObjects
// we use it for "virtual" texturing

__global__ void d_render(uchar4 *d_output, uint imageW, uint imageH, float lod, cudaTextureObject_t atlasTexture)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / (float)imageW;
    float v = y / (float)imageH;

    if ((x < imageW) && (y < imageH)) {
        // read from 2D atlas texture and decode texture object
        uint2               texCoded = tex2D<uint2>(atlasTexture, u, v);
        cudaTextureObject_t tex      = decodeTextureObject(texCoded);

        // read from cuda texture object, use template to specify what data will be
        // returned. tex2DLod allows us to pass the lod (mip map level) directly.
        // There is other functions with CUDA 5, e.g. tex2DGrad, that allow you
```

> JP: この抜粋は `cpp/3_CUDA_Features/bindlessTexture/bindlessTexture_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/bindlessTexture/bindlessTexture_kernel.cu:233-263
```cuda
        dim3 blockSize(16, 16, 1);
        dim3 gridSize(((uint)width + blockSize.x - 1) / blockSize.x, ((uint)height + blockSize.y - 1) / blockSize.y, 1);

        d_mipmap<<<gridSize, blockSize>>>(surfOutput, texInput, (uint)width, (uint)height);

        // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        // JP: `cudaDestroySurfaceObject`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        checkCudaErrors(cudaDestroySurfaceObject(surfOutput));

        checkCudaErrors(cudaDestroyTextureObject(texInput));

#ifdef SHOW_MIPMAPS
        // we blit the current mipmap back into first level
        cudaMemcpy3DParms copyParams = {0};
        copyParams.dstArray          = levelFirst;
        copyParams.srcArray          = levelTo;
        copyParams.extent            = make_cudaExtent(width, height, 1);
        copyParams.kind              = cudaMemcpyDeviceToDevice;
        // JP: `cudaMemcpy3D`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
        checkCudaErrors(cudaMemcpy3D(&copyParams));
#endif

        level++;
    }
}

uint getMipMapLevels(cudaExtent size)
{
```

> JP: この抜粋は `cpp/3_CUDA_Features/bindlessTexture/bindlessTexture_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaExtent` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaTextureObject_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaAddressModeClamp` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaResourceDesc` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGetMipmappedArrayLevel` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaTextureDesc` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaArray_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaCreateTextureObject` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDestroyTextureObject` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |

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
cmake --build build --target bindlessTexture
ctest --test-dir build -R bindlessTexture
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

- `cudaExtent` の直前と直後で、どの memory/resource が有効になったかをメモする。
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
