# CHANGELOG.md Japanese Companion

Original English document: [`CHANGELOG.md`](../../CHANGELOG.md)

## English Reference

The changelog records CUDA Samples updates by CUDA Toolkit release, including CUDA Tile C++ samples, CCCL feature samples, CUDA Python samples, API migrations, and platform changes.

> **日本語**
> changelog は CUDA Toolkit release ごとの sample 追加、API 変更、directory 構成変更、library/sample 更新を記録します。
>
> **学習メモ**
> どの sample が新しい API や toolkit version に依存するかを知る手がかりです。build できない sample がある場合、changelog の version 条件を確認します。

## Recent Themes

English reference: CUDA 13.3 adds CUDA Tile C++ and CCCL 3.3 feature samples; CUDA 13.2 adds CUDA Python samples under `python/`.

> **日本語**
> 最近の更新では、`cpp/9_CUDA_Tile`、CCCL/CUB/libcu++ 系 sample、`python/` 以下の CUDA Python sample が増えています。
>
> **学習メモ**
> 新しい sample は toolkit と package version の前提が強い場合があります。README、`requirements.txt`、CMake の fetch 設定を合わせて読みます。

## API Migration Notes

English reference: entries mention replacement of deprecated CUDA fields and updated API signatures.

> **日本語**
> deprecated field や updated API signature の移行履歴は、古い tutorial や blog と現在の code が違って見える理由を説明します。
>
> **学習メモ**
> `cudaDeviceProp` field、Driver API context creation、Graph API、Unified Memory advice/prefetch などは toolkit version によって表記が変わることがあります。
