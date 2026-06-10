# 9. CUDA Tile - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

A CUDA Tile C++ sample demonstrating basic usage of tile kernels. This sample shows how to launch a tile kernel and how data can be passed between SIMT and Tile kernels through global device memory.

This sample demonstrates a simple vector addition using CUDA Tile C++. The vector addition is performed by splitting the dataset into blocks which process 1024 elements at a time. The cuda::tiles::partition_view type is used to partition the data into chunks of size 1024. Each block loads its respective chunk from 'a' and 'b', performs an elementwise addition, then stores it to the corresponding chunk of 'c'. Masked loads and stores are used to ensure that the last chunk

Original README headings: `9. CUDA Tile`, `[helloTile](./helloTile)`, `[tileVectorAdd](./tileVectorAdd)`, `[tileTranspose](./tileTranspose)`, `[tileMatmul](./tileMatmul)`, `[tileMatmulAutotuner](./tileMatmulAutotuner)`, `[tileBmm](./tileBmm)`, `[tileLayerNorm](./tileLayerNorm)`, `[tileRope](./tileRope)`, `[tileSpMV](./tileSpMV)`

> **日本語**
> `cpp/9_CUDA_Tile` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `9_CUDA_Tile` as a focused example of the CUDA concepts used in `cpp/9_CUDA_Tile`.
> **日本語**
> この sample の目的は、`9_CUDA_Tile` の小さな実装を通して Runtime, Driver, And NVRTC, Multi-GPU, P2P, And IPC, Tensor Cores And WMMA, Shared Memory, Streams And Events を具体的に追うことです。
>
> **学習メモ**
> 最初に `benchmark.h, matmul_benchmark.h, helloTile.cu, tileBmm.cu, tileLayerNorm.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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

- `Benchmark_Common/benchmark.h`: Host/device declarations, helper types, constants, or library wrappers.
- `Benchmark_Common/matmul_benchmark.h`: Host/device declarations, helper types, constants, or library wrappers.
- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `helloTile/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `helloTile/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `helloTile/helloTile.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `tileBmm/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `tileBmm/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `tileBmm/tileBmm.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `tileLayerNorm/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `tileLayerNorm/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `tileLayerNorm/tileLayerNorm.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `tileMatmul/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `tileMatmul/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `tileMatmul/tileMatmul.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `tileMatmulAutotuner/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `tileMatmulAutotuner/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `tileMatmulAutotuner/autotuner_search_space.conf`: Supporting file used by `tileMatmulAutotuner/autotuner_search_space.conf`.
- `tileMatmulAutotuner/backend_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `tileMatmulAutotuner/backend_nvcc.h`: Host/device declarations, helper types, constants, or library wrappers.
- `tileMatmulAutotuner/backend_nvrtc.h`: Host/device declarations, helper types, constants, or library wrappers.
- `tileMatmulAutotuner/matmul.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `tileMatmulAutotuner/matmul_autotuner.cpp`: Host-side setup, API calls, validation, and cleanup.
- `tileRope/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `tileRope/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `tileRope/tileRope.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `tileSpMV/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `tileSpMV/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `tileSpMV/tileSpMV.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `tileTranspose/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `tileTranspose/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `tileTranspose/tileTranspose.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `tileVectorAdd/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `tileVectorAdd/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `tileVectorAdd/tileVectorAdd.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `benchmark.h` first and locate the host-side setup or Python entry point.
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

- `Benchmark_Common/benchmark.h`: focus on `CUDA`, `CUDA_TILE_BENCHMARK_H`, `cudaEventCreate`, `cudaEventDestroy`, `cudaError_t`.
- `Benchmark_Common/matmul_benchmark.h`: focus on `CUDA_TILE_MATMUL_BENCHMARK_H`.
- `helloTile/helloTile.cu`: focus on `cudaDeviceSynchronize`, `launch`, `cudaGetLastError`, `cudaMalloc`, `cudaMemset`.
- `tileBmm/tileBmm.cu`: focus on `launch`, `cudaMalloc`, `cudaMemcpy`, `cudaMemcpyDeviceToHost`, `cudaFree`.
- `tileLayerNorm/tileLayerNorm.cu`: focus on `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemcpyDeviceToHost`, `launch`.
- `tileMatmul/tileMatmul.cu`: focus on `cudaMalloc`, `cudaMemcpy`, `cudaFree`, `launch`, `cudaMemcpyHostToDevice`.
- `tileMatmulAutotuner/backend_common.h`: focus on `nvrtc`, `launch`.
- `tileMatmulAutotuner/backend_nvcc.h`: focus on `CUDA`, `CUDA_INCLUDE_PATH`.
- `tileMatmulAutotuner/backend_nvrtc.h`: focus on `CUDA`, `nvrtc`, `nvrtcResult`, `nvrtcCreateProgram`, `nvrtcCompileProgram`.
- `tileMatmulAutotuner/matmul.cu`: focus on `CUDA`.
- Additional source files: 5 more support files. Use the same setup/work/sync/cleanup lens.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/9_CUDA_Tile/CMakeLists.txt:1-19
```cmake
# GCC auto-enables _FORTIFY_SOURCE at -O1 and above, which routes printf through
# a __host__ __device__ wrapper that tile kernels can't call. Turn it off for
# the tile category.
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-U_FORTIFY_SOURCE>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-D_FORTIFY_SOURCE=0>
)

add_subdirectory(helloTile)
add_subdirectory(tileVectorAdd)
add_subdirectory(tileTranspose)
add_subdirectory(tileMatmulAutotuner)
add_subdirectory(tileMatmul)
add_subdirectory(tileBmm)
add_subdirectory(tileLayerNorm)
add_subdirectory(tileRope)
add_subdirectory(tileSpMV)
```

> JP: この抜粋は `cpp/9_CUDA_Tile/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMemcpyDeviceToHost` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGetLastError` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMemcpyHostToDevice` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `nvrtc` | 実行時 compile/link の境界です。log、module、kernel name の対応を確認します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `CUdeviceptr` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuMemAlloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- multi-GPU sample では、device 選択、peer capability、context/IPC handle の寿命を分けて読みます。
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
- Runtime/Driver/NVRTC sample では、compile/load した module と launch する kernel 名の対応が重要です。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。
- 同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。
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
cmake --build build --target 9_CUDA_Tile
ctest --test-dir build -R 9_CUDA_Tile
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

The sample prints timing, bandwidth, latency, throughput, or comparison data; exact values depend on hardware and driver.
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
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- device ごとの ownership と、peer/IPC で共有される resource を分けて書く。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Runtime, Driver, And NVRTC](../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [Multi-GPU, P2P, And IPC](../../docs_ja/themes/multi_gpu_p2p_ipc.md): device topology、peer access、IPC handle、multi-process 境界を読むための基礎です。
- [Tensor Cores And WMMA](../../docs_ja/themes/tensor_cores_wmma.md): Tensor Core、tile、precision、fragment の制約を読むための基礎です。
- [Shared Memory](../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Streams And Events](../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Performance](../../docs_ja/themes/performance.md): memory traffic、occupancy、overlap、launch overhead、timing を読むための基礎です。
- [Memory](../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
