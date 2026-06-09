# simpleCUFFT_MGPU - Simple CUFFT_MGPU - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

Example of using CUFFT. In this example, CUFFT is used to compute the 1D-convolution of some signal with some filter by transforming both into frequency domain, multiplying them together, and transforming the signal back to time domain on Multiple GPU.

Original README headings: `simpleCUFFT_MGPU - Simple CUFFT_MGPU`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`

> **日本語**
> `cpp/4_CUDA_Libraries/simpleCUFFT_MGPU` は `4_CUDA_Libraries` に属する CUDA sample です。英語 README の目的と手順を残し、ここでは日本語で実装の読み方を補足します。
>
> **学習メモ**
> まず英語 README の前提と実行方法を確認し、次に source file で memory ownership、transfer、kernel/library call、sync、validation の順に追います。

## Purpose

English anchor: this sample demonstrates `simpleCUFFT_MGPU - Simple CUFFT_MGPU` and should be read together with the original README and source files.

> **日本語**
> この sample の目的は、CUDA の機能や周辺 library/API の使い方を小さな実行単位で確認することです。
>
> **学習メモ**
> sample は production code ではなく、概念を分離して見せる教材です。error check や validation がどこにあるかも含めて読みます。

## Prerequisites

- CUDA Toolkit matching the repository branch
- CMake 3.20 or newer
- C++/CUDA compiler supported by the toolkit
- CUDA library components used by the sample, such as cuBLAS/cuFFT/cuSolver/NPP/CUB/nvJPEG
- Multiple compatible GPUs or peer/IPC support when the sample requires it

> **日本語**
> 必要な環境は英語 README と CMake/requirements を優先します。GPU 機能、driver、toolkit、OS、display stack の条件で実行できない sample があります。
>
> **学習メモ**
> build できない場合は、source code の前に CMake 条件、必要 library、platform guard、Python package version を確認します。

## Files

- `CMakeLists.txt`
- `README.md`
- `simpleCUFFT_MGPU.cu`

> **日本語**
> code file は実装、`README.md` は実行手順、`CMakeLists.txt` や `requirements.txt` は環境と依存関係を表します。
>
> **学習メモ**
> data/reference file がある sample では、GPU 計算結果を比較するための入力または期待値として扱われます。

## Execution Flow

- Read command-line options and choose the CUDA device or library configuration.
- Allocate host/device resources and initialize input data.
- Move, map, or migrate data so GPU work sees the intended inputs.
- Create handles/descriptors/plans and call the CUDA library operation.
- Use streams/events or graph dependencies to order asynchronous work.
- Synchronize at the required boundary, copy or expose results, and validate against a reference path.
- Release CUDA, library, framework, or external resources in the matching cleanup order.

> **日本語**
> 実行の流れは、入力準備、GPU が見える memory への配置、kernel または library work の投入、同期、結果確認、後片付けです。
>
> **学習メモ**
> host と device は同時に進むことがあります。結果を CPU が読む直前に、どの API が完了を保証しているかを確認します。

## Key APIs And Concepts

- `cudaXtFree`
- `cudaSetDevice`
- `cudaGetDeviceCount`
- `cudaDeviceSynchronize`
- `cudaGetDeviceProperties`
- `cudaLibXtDesc`
- `cudaDeviceProp`
- `cufftXt`
- `cufftComplex`
- `cufftCreate`
- `cufftResult`
- `cufftHandle`
- `cufftXtSetGPUs`
- `cufftMakePlan1d`
- `cufftXtMalloc`
- `cufftXtMemcpy`
- `cufftXtExecDescriptorC2C`
- `cufftExecC2C`

> **日本語**
> API 名は翻訳せず、呼び出しが「確保」「転送」「起動」「同期」「破棄」「library 実行」のどれかに分類できるかを確認します。
>
> **学習メモ**
> helper macro に包まれた API も、実際には CUDA Runtime/Driver/library call です。error handling の境界を見落とさないようにします。

## Memory, Synchronization, And Performance Notes

- kernel launch、grid/block/thread の実行階層を確認します。
- stream/event による非同期 work と待ち合わせを確認します。
- handle、descriptor、workspace、library call の device work を確認します。
- device selection、peer access、IPC handle、集約同期を確認します。
- memory traffic、occupancy、overlap、launch overhead、計測範囲を確認します。
- thread index から data index への対応を確認します。
- host/device/managed/external memory の所有権と lifetime を確認します。
- 同期範囲、fence、atomic 更新の必要性を確認します。

> **日本語**
> memory の所有者、転送方向、同期 point、計測範囲を分けて読むと、sample の意図が分かりやすくなります。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Build And Run

English commands are in the original README. Typical CMake flow from the repository root:

```bash
cmake -S . -B build
cmake --build build --target simpleCUFFT_MGPU
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と CMake を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても元の sample を汚しにくくなります。

## Expected Behavior

English expectation: run the sample as documented and compare its output with the README, validation message, generated file, or reference result.

> **日本語**
> 多くの sample は `PASS`/`Test passed` 相当の検証、数値誤差の比較、画像/データの生成、または性能値の表示を行います。GUI/interop sample は window や外部 API の状態にも依存します。
>
> **学習メモ**
> 出力文字列は test runner やドキュメントと対応するため翻訳しません。日本語メモでは、何を確認すべきかだけを補います。

## Common Mistakes

- Mixing element counts and byte counts in allocation or copy sizes.
- Forgetting bounds checks when grid size is rounded up.
- Assuming async work is complete before the matching stream/event synchronization.
- Passing a wrong leading dimension, stride, descriptor, handle, or workspace size to a CUDA library.
- Allocating on one device and launching or freeing on another without checking current device.

> **日本語**
> 間違いを探すときは、API の戻り値、現在の device、memory size、同期の位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> CUDA の bug は kernel 内だけでなく、host 側の setup と cleanup の順序にもよく現れます。

## Exercises

- Trace every allocation and write down which API owns the memory and which API releases it.
- Change the problem size and verify the bounds check still prevents out-of-range access.
- Draw a stream timeline and mark where host code must wait for correctness.
- Locate every descriptor/plan parameter and match it to the input data layout.
- Measure separately: setup, transfer, kernel/library execution, and validation.
- Compare this sample with one related theme guide and note one repeated CUDA pattern.

> **日本語**
> 演習では code の挙動を変える前に、読み取った仮説をコメントやメモとして整理します。
>
> **学習メモ**
> 変更する場合は、元の出力と validation を壊していないかを小さく確認します。

## Related Themes

- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を確認します。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): stream/event による非同期 work と待ち合わせを確認します。
- [CUDA Libraries](../../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の device work を確認します。
- [Multi-GPU, P2P, And IPC](../../../docs_ja/themes/multi_gpu_p2p_ipc.md): device selection、peer access、IPC handle、集約同期を確認します。
- [Performance](../../../docs_ja/themes/performance.md): memory traffic、occupancy、overlap、launch overhead、計測範囲を確認します。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応を確認します。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を確認します。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): 同期範囲、fence、atomic 更新の必要性を確認します。

> **日本語**
> 関連テーマを先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かりやすくなります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
