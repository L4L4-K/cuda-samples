# bicubicTexture - Bicubic B-spline Interoplation - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates how to efficiently implement a Bicubic B-spline interpolation filter with CUDA texture.

Graphics Interop, Image Processing

Original README headings: `bicubicTexture - Bicubic B-spline Interoplation`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/bicubicTexture` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `bicubicTexture` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/bicubicTexture`.
> **日本語**
> この sample の目的は、`bicubicTexture` の小さな実装を通して CUDA Graphs, Streams And Events, Synchronization And Atomics, Memory, Kernel Launch And Indexing を具体的に追うことです。
>
> **学習メモ**
> 最初に `bicubicTexture.cpp, bicubicTexture_cuda.cu, bicubicTexture_kernel.cuh` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `bicubicTexture.cpp`: Host-side setup, API calls, validation, and cleanup.
- `bicubicTexture_cuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `bicubicTexture_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `data/0_nearest.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/1_bilinear.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/2_bicubic.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/3_fastbicubic.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/4_catmull-rom.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/teapot512.pgm`: Input, reference, generated-data description, or documentation used by the sample.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `bicubicTexture.cpp` first and locate the host-side setup or Python entry point.
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

- `bicubicTexture.cpp`: focus on `CUDA`, `cudaDeviceSynchronize`, `cudaGraphicsResource`, `cudaGraphicsMapResources`, `cudaGraphicsResourceGetMappedPointer`.
- `bicubicTexture_cuda.cu`: focus on `cudaAddressModeClamp`, `cudaTextureDesc`, `launch`, `cudaMallocArray`, `cudaMemcpyHostToDevice`.
- `bicubicTexture_kernel.cuh`: focus on `cudaTextureObject_t`, `blockIdx`, `blockDim`, `threadIdx`, `launch`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaTextureObject_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGraphicsMapResources` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphicsResourceGetMappedPointer` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphicsUnmapResources` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphicsUnregisterResource` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGetDeviceProperties` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaMallocArray` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |

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
cmake --build build --target bicubicTexture
ctest --test-dir build -R bicubicTexture
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

- `cudaTextureObject_t` の直前と直後で、どの memory/resource が有効になったかをメモする。
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
