# Prefix Sum (Scan) - Japanese Learning Guide

English source: [README.md](README.md)
Python requirements: [requirements.txt](requirements.txt)

## English Overview

Demonstrates parallel prefix sum (scan) algorithms using cuda.compute with cuda.core stream management.

Original README headings: `Prefix Sum (Scan)`, `Overview`, `Requirements`, `Hardware`, `Software`, `Usage`, `Create and activate virtual environment`, `venv\Scripts\activate # Windows`

> **日本語**
> `python/2_CoreConcepts/prefixSum` は `2_CoreConcepts` に属する CUDA sample です。英語 README の目的と手順を残し、ここでは日本語で実装の読み方を補足します。
>
> **学習メモ**
> まず英語 README の前提と実行方法を確認し、次に source file で memory ownership、transfer、kernel/library call、sync、validation の順に追います。

## Purpose

English anchor: this sample demonstrates `Prefix Sum (Scan)` and should be read together with the original README and source files.

> **日本語**
> この sample の目的は、CUDA の機能や周辺 library/API の使い方を小さな実行単位で確認することです。
>
> **学習メモ**
> sample は production code ではなく、概念を分離して見せる教材です。error check や validation がどこにあるかも含めて読みます。

## Prerequisites

- Python 3 environment
- `requirements.txt` packages for this sample
- NVIDIA driver and CUDA-capable GPU

> **日本語**
> 必要な環境は英語 README と CMake/requirements を優先します。GPU 機能、driver、toolkit、OS、display stack の条件で実行できない sample があります。
>
> **学習メモ**
> build できない場合は、source code の前に CMake 条件、必要 library、platform guard、Python package version を確認します。

## Files

- `prefixSum.py`
- `README.md`
- `requirements.txt`

> **日本語**
> code file は実装、`README.md` は実行手順、`CMakeLists.txt` や `requirements.txt` は環境と依存関係を表します。
>
> **学習メモ**
> data/reference file がある sample では、GPU 計算結果を比較するための入力または期待値として扱われます。

## Execution Flow

- Read command-line options and choose the CUDA device or library configuration.
- Create CPU/GPU arrays or framework tensors through Python libraries.
- Move, map, or migrate data so GPU work sees the intended inputs.
- Launch CUDA kernels or JIT-compiled device code with an explicit execution configuration.
- Use streams/events or graph dependencies to order asynchronous work.
- Synchronize at the required boundary, copy or expose results, and validate against a reference path.
- Release CUDA, library, framework, or external resources in the matching cleanup order.

> **日本語**
> 実行の流れは、入力準備、GPU が見える memory への配置、kernel または library work の投入、同期、結果確認、後片付けです。
>
> **学習メモ**
> host と device は同時に進むことがあります。結果を CPU が読む直前に、どの API が完了を保証しているかを確認します。

## Key APIs And Concepts

- `Device`
- `cp.cuda`
- `cp.asarray`
- `cp.empty_like`
- `cp.asnumpy`
- `cp.ones`

> **日本語**
> API 名は翻訳せず、呼び出しが「確保」「転送」「起動」「同期」「破棄」「library 実行」のどれかに分類できるかを確認します。
>
> **学習メモ**
> helper macro に包まれた API も、実際には CUDA Runtime/Driver/library call です。error handling の境界を見落とさないようにします。

## Memory, Synchronization, And Performance Notes

- Python library が所有する GPU buffer、JIT、implicit sync を確認します。
- stream/event による非同期 work と待ち合わせを確認します。
- 同期範囲、fence、atomic 更新の必要性を確認します。
- memory traffic、occupancy、overlap、launch overhead、計測範囲を確認します。
- host/device/managed/external memory の所有権と lifetime を確認します。
- error check、reference validation、profiling/timing の範囲を確認します。

> **日本語**
> memory の所有者、転送方向、同期 point、計測範囲を分けて読むと、sample の意図が分かりやすくなります。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Build And Run

English commands are in the original README. Typical local flow:

```bash
pip install -r requirements.txt
python prefixSum.py
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
- Assuming async work is complete before the matching stream/event synchronization.
- Triggering hidden host-device transfers while timing Python code.

> **日本語**
> 間違いを探すときは、API の戻り値、現在の device、memory size、同期の位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> CUDA の bug は kernel 内だけでなく、host 側の setup と cleanup の順序にもよく現れます。

## Exercises

- Trace every allocation and write down which API owns the memory and which API releases it.
- Draw a stream timeline and mark where host code must wait for correctness.
- Measure separately: setup, transfer, kernel/library execution, and validation.
- Identify which arrays are CPU-backed and which are GPU-backed before each operation.
- Compare this sample with one related theme guide and note one repeated CUDA pattern.

> **日本語**
> 演習では code の挙動を変える前に、読み取った仮説をコメントやメモとして整理します。
>
> **学習メモ**
> 変更する場合は、元の出力と validation を壊していないかを小さく確認します。

## Related Themes

- [Python CUDA](../../../docs_ja/themes/python_cuda.md): Python library が所有する GPU buffer、JIT、implicit sync を確認します。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): stream/event による非同期 work と待ち合わせを確認します。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): 同期範囲、fence、atomic 更新の必要性を確認します。
- [Performance](../../../docs_ja/themes/performance.md): memory traffic、occupancy、overlap、launch overhead、計測範囲を確認します。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を確認します。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を確認します。

> **日本語**
> 関連テーマを先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かりやすくなります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
