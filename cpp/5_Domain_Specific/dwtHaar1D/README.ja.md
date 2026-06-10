# dwtHaar1D - 1D Discrete Haar Wavelet Decomposition - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

Discrete Haar wavelet decomposition for 1D signals with a length which is a power of 2.

Image Processing, Video Compression

Original README headings: `dwtHaar1D - 1D Discrete Haar Wavelet Decomposition`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/dwtHaar1D` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `dwtHaar1D` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/dwtHaar1D`.
> **日本語**
> この sample の目的は、`dwtHaar1D` の小さな実装を通して Shared Memory, Memory, Kernel Launch And Indexing, Execution Model, Debugging, Profiling, And Testing を具体的に追うことです。
>
> **学習メモ**
> 最初に `dwtHaar1D.cu, dwtHaar1D_kernel.cuh` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `data/regression.gold.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `data/regression_2_14.gold.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `data/regression_2_18.gold.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `data/signal.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `data/signal_2_14.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `data/signal_2_18.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `dwtHaar1D.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `dwtHaar1D_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `dwtHaar1D.cu` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
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

- `dwtHaar1D.cu`: focus on `cudaMemcpy`, `cudaMalloc`, `launch`, `cudaFree`, `cudaMemcpyHostToDevice`.
- `dwtHaar1D_kernel.cuh`: focus on `blockIdx`, `blockDim`, `threadIdx`, `__shared__`, `launch`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/dwtHaar1D/CMakeLists.txt:1-43
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(dwtHaar1D LANGUAGES C CXX CUDA)

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
# Add target for dwtHaar1D
add_executable(dwtHaar1D dwtHaar1D.cu)

target_compile_options(dwtHaar1D PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(dwtHaar1D PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(dwtHaar1D PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_custom_command(TARGET dwtHaar1D POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    ${CMAKE_CURRENT_BINARY_DIR}/data
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/5_Domain_Specific/dwtHaar1D/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `dwtHaar1D.cu`

Source: cpp/5_Domain_Specific/dwtHaar1D/dwtHaar1D.cu:62-102
```cuda
decomposition.

-------------------------------------------------
| a_0 | a_0 | a_0 | a_0 | a_0 | a_0 | a_0 | a_0 |
-------------------------------------------------

-------------------------------------------------
| a_1 | a_1 | a_1 | a_1 | d_1 | d_1 | d_1 | d_1 |
-------------------------------------------------

-------------------------------------------------
| a_2 | a_2 | d_2 | d_2 | d_1 | d_1 | d_1 | d_1 |
-------------------------------------------------

-------------------------------------------------
| a_3 | d_3 | d_2 | d_2 | d_1 | d_1 | d_1 | d_1 |
-------------------------------------------------

* Host code.
*/

#ifdef _WIN32
#define NOMINMAX
#endif

// includes, system
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>

// constants which are used in host and device code
#define INV_SQRT_2 0.70710678118654752440f;
const unsigned int LOG_NUM_BANKS = 4;
const unsigned int NUM_BANKS     = 16;

```

> JP: この抜粋は `cpp/5_Domain_Specific/dwtHaar1D/dwtHaar1D.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/dwtHaar1D/dwtHaar1D.cu:110-129
```cuda
bool getLevels(unsigned int len, unsigned int *levels);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // run test
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Perform the wavelet decomposition
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv)
{
    bool bResult = false; // flag for final validation of the results

    char      *s_fname = NULL, *r_gold_fname = NULL;
    char       r_fname[256];
```

> JP: この抜粋は `cpp/5_Domain_Specific/dwtHaar1D/dwtHaar1D.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/dwtHaar1D/dwtHaar1D.cu:195-238
```cuda
    if (true != getLevels(slength, &dlevels_complete)) {
        // error message
        fprintf(stderr, "Signal length not supported.\n");
        // cleanup and abort
        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(signal);
        exit(EXIT_FAILURE);
    }

    // device in data
    float *d_idata = NULL;
    // device out data
    float *d_odata = NULL;
    // device approx_final data
    float *approx_final = NULL;
    // The very final approximation coefficient has to be written to the output
    // data, all others are reused as input data in the next global step and
    // therefore have to be written to the input data again.
    // The following flag indicates where to copy approx_final data
    //   - 0 is input, 1 is output
    int approx_is_input;

    // allocate device mem
    const unsigned int smem_size = sizeof(float) * slength;
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc((void **)&d_idata, smem_size));
    checkCudaErrors(cudaMalloc((void **)&d_odata, smem_size));
    checkCudaErrors(cudaMalloc((void **)&approx_final, smem_size));
    // copy input data to device
    // JP: `cudaMemcpy`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpy(d_idata, signal, smem_size, cudaMemcpyHostToDevice));

    // total number of threads
    // in the first decomposition step always one thread computes the average and
    // detail signal for one pair of adjacent values
    unsigned int num_threads_total_left = slength / 2;
    // decomposition levels performed in the current / next step
    unsigned int dlevels_step = dlevels_complete;

    // 1D signal so the arrangement of elements is also 1D
    dim3 block_size;
    dim3 grid_size;

    // number of decomposition levels left after one iteration on the device
```

> JP: この抜粋は `cpp/5_Domain_Specific/dwtHaar1D/dwtHaar1D.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/dwtHaar1D/dwtHaar1D.cu:258-277
```cuda
        approx_is_input = 1;
    }

    // Initialize d_odata to 0.0f
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    initValue<<<grid_size, block_size>>>(d_odata, 0.0f);

    // do until full decomposition is accomplished
    while (0 != num_threads_total_left) {
        // double the number of threads as bytes
        unsigned int mem_shared = (2 * block_size.x) * sizeof(float);
        // extra memory requirements to avoid bank conflicts
        mem_shared += ((2 * block_size.x) / NUM_BANKS) * sizeof(float);

        // run kernel
        // JP: この anchor では kernel launch の grid/block/shared-memory/stream 指定です。後続の sync/error check と完了確認を対応させます。
        dwtHaar1D<<<grid_size, block_size, mem_shared>>>(
            d_idata, d_odata, approx_final, dlevels_step, num_threads_total_left, block_size.x);

        // Copy approx_final to appropriate location
```

> JP: この抜粋は `cpp/5_Domain_Specific/dwtHaar1D/dwtHaar1D.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `dwtHaar1D_kernel.cuh`

Source: cpp/5_Domain_Specific/dwtHaar1D/dwtHaar1D_kernel.cuh:62-80
```cuda
decomposition.

-------------------------------------------------
| a_0 | a_0 | a_0 | a_0 | a_0 | a_0 | a_0 | a_0 |
-------------------------------------------------

-------------------------------------------------
| a_1 | a_1 | a_1 | a_1 | d_1 | d_1 | d_1 | d_1 |
-------------------------------------------------

-------------------------------------------------
| a_2 | a_2 | d_2 | d_2 | d_1 | d_1 | d_1 | d_1 |
-------------------------------------------------

-------------------------------------------------
| a_3 | d_3 | d_2 | d_2 | d_1 | d_1 | d_1 | d_1 |
-------------------------------------------------

* Device Code.
```

> JP: この抜粋は `cpp/5_Domain_Specific/dwtHaar1D/dwtHaar1D_kernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/dwtHaar1D/dwtHaar1D_kernel.cuh:96-115
```cuda
{
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block cta = cg::this_thread_block();
    // position of write into global memory
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    od[index] = value;

    // sync after each decomposition step
    // JP: sync: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    cg::sync(cta);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute partial wavelet decomposition on the GPU using Haar basis
//! For each thread block the full decomposition is computed but these results
//! have to be combined
//! Use one thread to perform the full decomposition
//! @param id  input data
```

> JP: この抜粋は `cpp/5_Domain_Specific/dwtHaar1D/dwtHaar1D_kernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaMemcpyHostToDevice` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMemcpyDeviceToDevice` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `__shared__` | block 内共有 memory または同期境界です。producer/consumer の順序を確認します。 |
| `Program` | 実行時 compile/link の境界です。log、module、kernel name の対応を確認します。 |
| `cudaMemcpyDeviceToHost` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `Device` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `gridDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target dwtHaar1D
ctest --test-dir build -R dwtHaar1D
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

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaMemcpy` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- shared memory tile の producer、consumer、barrier を図にする。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Shared Memory](../../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
