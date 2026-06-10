# tileLayerNorm - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates a persistent layer-norm forward pass using CUDA Tile C++: `y = (x - mean) * rsqrt(var + eps) * weight + bias`. The grid launches `NUM_SMS` persistent blocks; each block walks the row dimension with a grid-stride loop, processing `BLOCK_N` rows by `BLOCK_D` cols per iteration and striding by `NUM_SMS * BLOCK_N` rows between iterations. Per-row mean and inverse standard deviation are reduced across the column

dimension with `cuda::tiles` row reductions and saved to float32 side buffers, while the weight and bias tiles are loaded once and broadcast across rows.

Original README headings: `tileLayerNorm`, `Description`, `Expected Output`, `Prerequisites`

> **日本語**
> `cpp/9_CUDA_Tile/tileLayerNorm` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `tileLayerNorm` as a focused example of the CUDA concepts used in `cpp/9_CUDA_Tile/tileLayerNorm`.
> **日本語**
> この sample の目的は、`tileLayerNorm` の小さな実装を通して Runtime, Driver, And NVRTC, Multi-GPU, P2P, And IPC, Tensor Cores And WMMA, Shared Memory, Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `tileLayerNorm.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `tileLayerNorm.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `tileLayerNorm.cu` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.
- Enumerate devices, enable peer or IPC access, and record which device/process owns each resource.
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

- `tileLayerNorm.cu`: focus on `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemcpyDeviceToHost`, `launch`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/9_CUDA_Tile/tileLayerNorm/CMakeLists.txt:1-32
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(tileLayerNorm LANGUAGES C CXX CUDA)

# JP: `find_package`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90 100 110 120)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --enable-tile")

if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
add_executable(tileLayerNorm tileLayerNorm.cu)

target_compile_features(tileLayerNorm PRIVATE cxx_std_20 cuda_std_20)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileLayerNorm/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `tileLayerNorm.cu`

Source: cpp/9_CUDA_Tile/tileLayerNorm/tileLayerNorm.cu:43-67
```cuda
#include "helper_cuda.h"
#include "cuda_tile.h"
#include "cuda_fp16.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

/* SIMT initializer for X (N x D), W (D,), B (D,) with deterministic data. */
__global__ void initializeInputs(__half* X, __half* W, __half* B,
                                 int N, int D) {
  // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
  auto idx   = blockIdx.x * blockDim.x + threadIdx.x;
  auto total = N * D;
  if (idx < total) {
    int m = idx / D;
    int n = idx - m * D;
    X[idx] = __half{float((m + n) % 7) - 3.5f};
  }
  if (idx < D) {
    W[idx] = __half{1.0f + 0.1f * float(idx % 5)};
    B[idx] = __half{0.1f * float(idx % 3)};
  }
}

template<typename T,
```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileLayerNorm/tileLayerNorm.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/9_CUDA_Tile/tileLayerNorm/tileLayerNorm.cu:188-238
```cuda
  constexpr int   NUM_SMS = 132;
  constexpr float EPS     = 1e-5f;

  __half *d_X = nullptr, *d_Y = nullptr, *d_W = nullptr, *d_B = nullptr;
  float  *d_Mean = nullptr, *d_Rstd = nullptr;
  // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
  checkCudaErrors(cudaMalloc(&d_X,    N * D * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_Y,    N * D * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_W,    D     * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_B,    D     * sizeof(__half)));
  checkCudaErrors(cudaMalloc(&d_Mean, N     * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_Rstd, N     * sizeof(float)));

  int init_threads = 256, init_blocks = 1 + ((N * D - 1) / init_threads);
  // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
  initializeInputs<<<init_blocks, init_threads>>>(d_X, d_W, d_B, N, D);
  checkCudaErrors(cudaGetLastError());

  /* NUM_SMS is a compile-time NTTP that doubles as the persistent-loop
   * stride; the launch grid x must equal NUM_SMS for correctness.
   * Adjust the constant (and recompile) for one block per SM on a
   * device with a different SM count. */
  persistent_layer_norm_fwd_kernel<__half, BLOCK_N, BLOCK_D,
                                   /*TRAINING=*/true, /*COMPUTE_MEAN_AND_RSTD=*/true,
                                   N, D, NUM_SMS, EPS>
      <<<dim3(NUM_SMS, 1, 1)>>>(d_X, d_Y, d_W, d_B, d_Mean, d_Rstd);
  checkCudaErrors(cudaGetLastError());
  // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
  checkCudaErrors(cudaDeviceSynchronize());

  __half* h_Y        = new __half[N * D];
  __half* h_Y_ref    = new __half[N * D];
  float*  h_Mean     = new float[N];
  float*  h_Rstd     = new float[N];
  float*  h_Mean_ref = new float[N];
  float*  h_Rstd_ref = new float[N];
  // JP: `cudaMemcpy`, `cudaMemcpyDeviceToHost`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
  checkCudaErrors(cudaMemcpy(h_Y,    d_Y,    N * D * sizeof(__half), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_Mean, d_Mean, N     * sizeof(float),  cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_Rstd, d_Rstd, N     * sizeof(float),  cudaMemcpyDeviceToHost));

  /* CPU reference in double precision; compare with 1e-1 fp16 tolerance
   * for Y and 1e-3 for the float32 Mean/Rstd outputs. */
  for (int m = 0; m < N; ++m) {
    double sum = 0.0, sumsq = 0.0;
    for (int n = 0; n < D; ++n) {
      double x = double(float((m + n) % 7) - 3.5f);
      sum += x; sumsq += x * x;
    }
    double mu      = sum / double(D);
    double var     = sumsq / double(D) - mu * mu;
```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileLayerNorm/tileLayerNorm.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/9_CUDA_Tile/tileLayerNorm/tileLayerNorm.cu:265-277
```cuda
    }
  }

  printf("Success! Persistent LayerNorm matches expected results.\n");

  // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
  checkCudaErrors(cudaFree(d_X));   checkCudaErrors(cudaFree(d_Y));
  checkCudaErrors(cudaFree(d_W));   checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_Mean)); checkCudaErrors(cudaFree(d_Rstd));
  delete[] h_Y;        delete[] h_Y_ref;
  delete[] h_Mean;     delete[] h_Rstd;
  delete[] h_Mean_ref; delete[] h_Rstd_ref;
}
```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileLayerNorm/tileLayerNorm.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMemcpyDeviceToHost` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaGetLastError` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- multi-GPU sample では、device 選択、peer capability、context/IPC handle の寿命を分けて読みます。
- Runtime/Driver/NVRTC sample では、compile/load した module と launch する kernel 名の対応が重要です。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。
- Tensor Core sample では tile size、alignment、precision、accumulator の型が正しさと性能を決めます。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target tileLayerNorm
ctest --test-dir build -R tileLayerNorm
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
- JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。
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
- shared memory tile の producer、consumer、barrier を図にする。
- device ごとの ownership と、peer/IPC で共有される resource を分けて書く。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Runtime, Driver, And NVRTC](../../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [Multi-GPU, P2P, And IPC](../../../docs_ja/themes/multi_gpu_p2p_ipc.md): device topology、peer access、IPC handle、multi-process 境界を読むための基礎です。
- [Tensor Cores And WMMA](../../../docs_ja/themes/tensor_cores_wmma.md): Tensor Core、tile、precision、fragment の制約を読むための基礎です。
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
