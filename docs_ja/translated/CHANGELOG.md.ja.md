# CHANGELOG.md Japanese Companion

Original English document: [`CHANGELOG.md`](../../CHANGELOG.md)

## English Reference

The changelog records CUDA Samples updates by CUDA Toolkit release, including new samples, removed samples, API migrations, platform changes, and dependency changes.

> **日本語**
> changelog は release ごとの差分を読む file です。どの sample が追加され、どの API が更新され、どの platform や dependency の前提が変わったかを確認します。
>
> **学習メモ**
> build できない sample がある場合、source を疑う前に changelog の toolkit version、platform condition、dependency change を確認します。

## How To Read Releases

English anchor: release sections are chronological and tied to CUDA Toolkit versions.

> **日本語**
> release section は「いつ入った sample か」「どの toolkit feature を示す sample か」を知る入口です。新しい API を使う sample は古い Toolkit や driver では build/run できないことがあります。
>
> **学習メモ**
> `cpp/9_CUDA_Tile`、CUDA Python samples、CCCL/CUB/libcu++ samples などは version 前提が強いので README と requirements も合わせて読みます。

## API Migration Notes

English anchor: changelog entries mention deprecated fields, updated signatures, and moved functionality.

> **日本語**
> deprecated field や updated signature の記録は、古い tutorial と現在の sample code が違って見える理由を説明します。Driver API、Graph API、Unified Memory、library API は toolkit version によって見た目が変わることがあります。
>
> **学習メモ**
> API 名は翻訳せず、old/new の対応を英語のままメモします。検索や公式 documentation との照合がしやすくなります。

## Build Impact

English anchor: changes can affect CMake, dependencies, optional libraries, platform guards, and tests.

> **日本語**
> changelog の変更は build graph に影響します。sample 追加は new target、dependency 追加は find/link 設定、platform change は build exclusion に関係します。
>
> **学習メモ**
> build failure を調べるときは `CMakeLists.txt` companion、sample `README.ja.md`、theme guide の Runtime/Driver/NVRTC や Libraries を合わせて読みます。

## Learning Use

English anchor: changelog is not a tutorial, but it helps choose which samples to compare.

> **日本語**
> changelog は教材本文ではありませんが、同じ concept の sample を release ごとに比較する手がかりになります。たとえば Graph、Tensor Core、CUDA Tile、Python CUDA は関連 sample を横断して読むと変化が見えます。
>
> **学習メモ**
> 「なぜこの sample が追加されたか」を考えると、CUDA Toolkit がどの programming model や library を強化しているかが分かります。
