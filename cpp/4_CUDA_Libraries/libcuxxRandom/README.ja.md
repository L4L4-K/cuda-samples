# libcuxxRandom - libcu++ Random Distributions - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates the random-number facilities added to libcu++ in CCCL. `<cuda/std/random>` now offers host- and device-compatible implementations of the standard C++ distributions (uniform, normal, Poisson, Bernoulli, and more) and backports the C++26 `cuda::std::philox4x32` / `philox4x64` engines. `<cuda/random>` adds `cuda::pcg64` as an NVIDIA extension (the same generator NumPy uses by default). A kernel draws samples on each thread and the host computes empirical statistics, comparing them to the theoretical mean / variance / probability.

CCCL 3.3, libcu++ Random, PCG, Philox, Device-Side PRNG

Original README headings: `libcuxxRandom - libcu++ Random Distributions`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CCCL libcu++](https://nvidia.github.io/cccl/libcudacxx/)`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/4_CUDA_Libraries/libcuxxRandom` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `libcuxxRandom` as a focused example of the CUDA concepts used in `cpp/4_CUDA_Libraries/libcuxxRandom`.
> **日本語**
> この sample の目的は、`libcuxxRandom` の小さな実装を通して Multi-GPU, P2P, And IPC, Memory, Kernel Launch And Indexing, Execution Model, Debugging, Profiling, And Testing を具体的に追うことです。
>
> **学習メモ**
> 最初に `libcuxxRandom.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- The device topology required by the README, such as multiple GPUs, peer access, IPC, MPI, or process support

> **日本語**
> 必要条件は英語 README と CMake/requirements を優先します。この guide は条件を置き換えず、読むべき確認点を追加します。
>
> **学習メモ**
> 実行できない場合は、source を変える前に driver、toolkit、GPU feature、library、platform guard、Python package version を確認します。

## Files

- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `libcuxxRandom.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `libcuxxRandom.cu` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Enumerate devices, enable peer or IPC access, and record which device/process owns each resource.
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

- `libcuxxRandom.cu`: focus on `cudaMalloc`, `cudaMemcpy`, `cudaMemcpyDeviceToHost`, `cudaFree`, `blockDim`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/4_CUDA_Libraries/libcuxxRandom/CMakeLists.txt:1-68
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(libcuxxRandom LANGUAGES C CXX CUDA)

# Disable response file for libraries on QNX as qcc does not support lib paths with double quotes
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
endif()

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

# Fetch CCCL (CUB + libcu++ + Thrust) via CPM. The toolkit that ships
# with CUDA 13.2 bundles CCCL 3.2, but this sample uses APIs added in
# CCCL 3.3. Pinning the tag here lets the sample build on any toolkit
# with a usable nvcc. Override with -DCCCL_SOURCE_DIR=/path/to/cccl
# to use a local checkout instead of fetching from GitHub.
set(CCCL_SAMPLES_CCCL_TAG "v3.3.3" CACHE STRING
    "Tag/branch of NVIDIA/cccl to fetch for the CCCL samples")

if(NOT TARGET CCCL::CCCL)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CPM.cmake")
    if(DEFINED CCCL_SOURCE_DIR AND NOT CCCL_SOURCE_DIR STREQUAL "")
        CPMAddPackage(NAME CCCL SOURCE_DIR "${CCCL_SOURCE_DIR}")
    else()
        CPMAddPackage(
            NAME CCCL
            GIT_REPOSITORY "https://github.com/NVIDIA/cccl"
            GIT_TAG "${CCCL_SAMPLES_CCCL_TAG}"
        )
    endif()
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for libcuxxRandom
add_executable(libcuxxRandom libcuxxRandom.cu)

target_compile_options(libcuxxRandom PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(libcuxxRandom PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(libcuxxRandom PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(libcuxxRandom PRIVATE
    CUDA::cudart
    CCCL::CCCL
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/libcuxxRandom/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `libcuxxRandom.cu`

Source: cpp/4_CUDA_Libraries/libcuxxRandom/libcuxxRandom.cu:42-60
```cuda
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <helper_cuda.h>

/* Includes, cccl */
#include <cuda/random>
#include <cuda/std/random>

#define THREADS_PER_BLOCK  256
#define SAMPLES_PER_THREAD 256

/* Per-thread kernel: seed a PCG engine, draw samples from four
 * distributions, and also pull Philox output through a Bernoulli dist
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/libcuxxRandom/libcuxxRandom.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/libcuxxRandom/libcuxxRandom.cu:64-83
```cuda
                              float             *uniform_out,
                              float             *normal_out,
                              int               *poisson_out,
                              int               *bernoulli_out)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    const int tid           = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    cuda::pcg64 rng(base_seed + static_cast<unsigned long long>(tid));

    cuda::std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
    cuda::std::normal_distribution<float>       normal_dist(0.0f, 1.0f);
    cuda::std::poisson_distribution<int>        poisson_dist(4.0);
    cuda::std::bernoulli_distribution           bernoulli_dist(0.25);

    cuda::std::philox4x32 philox(static_cast<cuda::std::uint32_t>(base_seed + 17u + tid));

    for (int i = 0; i < num_samples_per_thread; ++i) {
        const int idx      = i * total_threads + tid;
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/libcuxxRandom/libcuxxRandom.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/libcuxxRandom/libcuxxRandom.cu:118-137
```cuda
        ones += v;
    const double p = static_cast<double>(ones) / static_cast<double>(samples.size());
    printf("%-24s n=%zu  p(1)=%.4f (exp %.4f)\n", "bernoulli(0.25):", samples.size(), p, expected_p);
}

int main(int argc, char **argv)
{
    int num_blocks = 64;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "--blocks") == 0)
            num_blocks = atoi(argv[i + 1]);
    }
    if (num_blocks <= 0)
        num_blocks = 1;

    int devID = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    printf("Device: %s (Compute Capability %d.%d)\n\n", props.name, props.major, props.minor);

```

> JP: この抜粋は `cpp/4_CUDA_Libraries/libcuxxRandom/libcuxxRandom.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/libcuxxRandom/libcuxxRandom.cu:145-187
```cuda

    float *d_uniform   = nullptr;
    float *d_normal    = nullptr;
    int   *d_poisson   = nullptr;
    int   *d_bernoulli = nullptr;
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc(&d_uniform, n * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_normal, n * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_poisson, n * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_bernoulli, n * sizeof(int)));

    const unsigned long long seed = 0xC0FFEE00ULL;
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    sample_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        seed, SAMPLES_PER_THREAD, d_uniform, d_normal, d_poisson, d_bernoulli);
    checkCudaErrors(cudaGetLastError());
    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<float> uniform(n), normal(n);
    std::vector<int>   poisson(n), bernoulli(n);
    // JP: `cudaMemcpy`, `cudaMemcpyDeviceToHost`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpy(uniform.data(), d_uniform, n * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(normal.data(), d_normal, n * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(poisson.data(), d_poisson, n * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(bernoulli.data(), d_bernoulli, n * sizeof(int), cudaMemcpyDeviceToHost));

    summarize(uniform, /*mean=*/0.5, /*var=*/1.0 / 12.0, "uniform(0,1):");
    summarize(normal, /*mean=*/0.0, /*var=*/1.0, "normal(0,1):");
    summarize(poisson, /*mean=*/4.0, /*var=*/4.0, "poisson(lambda=4):");
    summarize_bernoulli(bernoulli, /*p=*/0.25);

    printf("\nEngines exercised: cuda::pcg64 (NumPy-compatible) and cuda::std::philox4x32 (C++26)\n");

    // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFree(d_uniform));
    checkCudaErrors(cudaFree(d_normal));
    checkCudaErrors(cudaFree(d_poisson));
    checkCudaErrors(cudaFree(d_bernoulli));

    printf("Done\n");
    return EXIT_SUCCESS;
}
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/libcuxxRandom/libcuxxRandom.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaMemcpyDeviceToHost` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaGetDeviceProperties` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `Device` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `gridDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaDeviceProp` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGetLastError` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- multi-GPU sample では、device 選択、peer capability、context/IPC handle の寿命を分けて読みます。

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
cmake --build build --target libcuxxRandom
ctest --test-dir build -R libcuxxRandom
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
- peer access が有効な device pair と、単に複数 GPU が存在することを混同する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaMalloc` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- device ごとの ownership と、peer/IPC で共有される resource を分けて書く。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Multi-GPU, P2P, And IPC](../../../docs_ja/themes/multi_gpu_p2p_ipc.md): device topology、peer access、IPC handle、multi-process 境界を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
