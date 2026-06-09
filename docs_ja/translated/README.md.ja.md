# README.md Japanese Companion

Original English document: [`README.md`](../../README.md)

## English Reference

CUDA Samples provides sample programs for CUDA developers and this branch tracks CUDA Toolkit 13.3. The original README explains prerequisites, cloning, CMake-based builds on Linux and Windows, on-GPU debugging, and sample execution.

> **日本語**
> CUDA Samples は CUDA Toolkit の機能を学ぶためのサンプル集です。元の README は、CUDA Toolkit の導入、repository の取得、CMake による Linux/Windows build、on-GPU debugging、各 sample の実行方法を説明しています。
>
> **学習メモ**
> この companion ではコマンド名や target 名を翻訳しません。実際に入力するコマンドは英語 README を source of truth とし、日本語側では「なぜその手順が必要か」を補います。

## Build Flow

English flow: install CUDA Toolkit and CMake, create a build directory, configure with CMake, build, then run samples from the build output or individual sample directories.

> **日本語**
> build は「CUDA Toolkit と CMake を用意する」「source tree とは別に build directory を作る」「CMake で generator と設定を確定する」「build tool で target を作る」「生成物を実行する」という流れです。
>
> **学習メモ**
> CMake configure は project の構成を決める段階、build は実際に compile/link する段階です。CUDA architecture、optional library、platform-specific sample の有無は configure 時に影響します。

## CUDA Debugging Note

English reference: on-GPU debugging can be enabled through cuda-gdb and the `ENABLE_CUDA_DEBUG` CMake option, but it can significantly affect performance.

> **日本語**
> device 側 debug を有効にすると、最適化が制限されるため性能が大きく変わります。debug build の計測値を performance 判断に使わないようにします。
>
> **学習メモ**
> correctness を見る debug 実行と、性能を見る profiling 実行は分けて考えます。

## Local Study Overlay

English source files remain unchanged except for `JP:` comments. Japanese sample guides live beside sample READMEs as `README.ja.md`; cross-cutting explanations live under `docs_ja/`.

> **日本語**
> この fork の日本語化は、元の英語を置き換えず横に追加する方式です。sample ごとの読み方は各 directory の `README.ja.md`、共通知識は `docs_ja/themes/` と `docs_ja/glossary/` を参照します。
>
> **学習メモ**
> 元 README、`README.ja.md`、source code の順に読むと、目的、実行手順、実装詳細を段階的に追えます。
