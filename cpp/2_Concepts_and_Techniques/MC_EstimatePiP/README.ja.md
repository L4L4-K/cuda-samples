# MC_EstimatePiP - Monte Carlo Estimation of Pi (batch PRNG) - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample uses Monte Carlo simulation for Estimation of Pi (using batch PRNG).  This sample also uses the NVIDIA CURAND library.

Random Number Generator, Computational Finance, CURAND Library

Original README headings: `MC_EstimatePiP - Monte Carlo Estimation of Pi (batch PRNG)`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/2_Concepts_and_Techniques/MC_EstimatePiP` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `MC_EstimatePiP` as a focused example of the CUDA concepts used in `cpp/2_Concepts_and_Techniques/MC_EstimatePiP`.
> **日本語**
> この sample の目的は、`MC_EstimatePiP` の小さな実装を通して CUDA Libraries, Shared Memory, Memory, Kernel Launch And Indexing, Execution Model を具体的に追うことです。
>
> **学習メモ**
> 最初に `cudasharedmem.h, piestimator.h, test.h, main.cpp, piestimator.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `inc/cudasharedmem.h`: Host/device declarations, helper types, constants, or library wrappers.
- `inc/piestimator.h`: Host/device declarations, helper types, constants, or library wrappers.
- `inc/test.h`: Host/device declarations, helper types, constants, or library wrappers.
- `src/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `src/piestimator.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `src/test.cpp`: Host-side setup, API calls, validation, and cleanup.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `cudasharedmem.h` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Create library handles, descriptors, plans, or workspaces before the library call.
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

- `inc/cudasharedmem.h`: focus on `__shared__`, `CUDASHAREDMEM_H`.
- `inc/piestimator.h`: focus on control flow and helper functions.
- `inc/test.h`: focus on control flow and helper functions.
- `src/main.cpp`: focus on `cudaResult`, `cudaSuccess`, `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaError_t`.
- `src/piestimator.cu`: focus on `cudaResult`, `curandResult`, `cudaSuccess`, `cudaGetErrorString`, `threadIdx`.
- `src/test.cpp`: focus on `cudaResult`, `cudaDeviceProp`, `cudaError_t`, `cudaGetDeviceProperties`, `cudaSuccess`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaResult` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `curandResult` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaGetErrorString` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGetDeviceProperties` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CURAND` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `__shared__` | block 内共有 memory または同期境界です。producer/consumer の順序を確認します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaGetDeviceCount` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `CURAND_STATUS_SUCCESS` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaError_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDeviceProp` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- library sample では handle、descriptor、plan、workspace が GPU work の外側の resource です。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。

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
cmake --build build --target MC_EstimatePiP
ctest --test-dir build -R MC_EstimatePiP
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
- shared memory を書いた thread と読む thread の間に必要な barrier を見落とす。
- leading dimension、stride、descriptor、workspace size を host 配列の見た目だけで判断する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaResult` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- shared memory tile の producer、consumer、barrier を図にする。
- handle/descriptor/workspace の作成、利用、破棄を対応表にする。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [CUDA Libraries](../../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。
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
