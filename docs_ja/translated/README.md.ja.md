# README.md Japanese Companion

Original English document: [`README.md`](../../README.md)

## English Reference

CUDA Samples provides sample programs for CUDA developers. The root README explains prerequisites, repository cloning, CMake-based builds on Linux and Windows, on-GPU debugging, and sample execution.

> **日本語**
> CUDA Samples は CUDA Toolkit の機能を学ぶための sample collection です。root README は「何を install するか」「どう clone するか」「Linux/Windows でどう configure/build するか」「debug option をどう扱うか」「生成した sample をどう実行するか」を説明します。
>
> **学習メモ**
> command、target、directory name は英語のまま読みます。日本語側では「その手順が何を決めるか」を補います。

## Prerequisites And Environment

English anchor: the original README names CUDA Toolkit, CMake, compiler, OS, and platform requirements.

> **日本語**
> prerequisites は build と実行の前提です。CUDA Toolkit は compiler、headers、libraries、tools を提供し、driver と GPU capability は実行可能性を決めます。CMake と host compiler は sample target を生成するために必要です。
>
> **学習メモ**
> build が失敗した場合、source code の前に Toolkit version、driver、CMake version、host compiler、optional dependency、platform-specific guard を確認します。

## Clone And Repository Layout

English anchor: the README describes obtaining the repository and using the directory structure.

> **日本語**
> repository layout は学習 map です。`cpp/` は C++/CUDA samples、`python/` は CUDA Python samples、`Common/` は helper utilities、`cmake/` は build support、`docs_ja/` はこの fork の日本語 companion です。
>
> **学習メモ**
> sample directory では `README.md`、`README.ja.md`、`CMakeLists.txt` または `requirements.txt`、source files を一緒に読みます。

## CMake Build Flow

English anchor: the README shows configure and build commands through CMake.

```bash
cmake -S . -B build
cmake --build build
```

> **日本語**
> configure は build graph を作る段階、build は compile/link する段階です。source tree と build tree を分けることで、生成物を削除しても original source や日本語 companion を壊しにくくなります。
>
> **学習メモ**
> CMake option、CUDA architecture、optional library、platform guard は configure 時に効きます。実際の target name は各 sample の `CMakeLists.txt` を確認します。

## Running Samples And Tests

English anchor: generated binaries or Python scripts are run according to each sample README.

> **日本語**
> 実行は sample ごとに前提が違います。console sample は validation message を出し、graphics/interop sample は window や external API を必要とし、performance sample は hardware-dependent な timing を出します。
>
> **学習メモ**
> output strings は test runner や README と対応するため翻訳しません。`PASS`、`FAIL`、timing label、error string は source と同じ文字列で確認します。

## Debugging And Profiling

English anchor: the README references on-GPU debugging and CMake debug options.

> **日本語**
> debugging option は correctness を調べるためのものです。optimization や timing が変わるため、debug build の数値を performance 判断に使わないようにします。
>
> **学習メモ**
> correctness debugging、profiling、benchmarking は別の目的です。`docs_ja/themes/debugging_profiling_testing.md` と `docs_ja/themes/performance.md` を分けて読みます。

## Local Japanese Overlay

English anchor: original files remain authoritative; Japanese files are companion study material.

> **日本語**
> この fork の日本語資料は sample behavior を変えません。英語 source、identifier、API、command、target、expected output、LICENSE、attribution を維持し、説明だけを追加します。
>
> **学習メモ**
> 迷った場合は original README と source を優先し、日本語 companion は理解を補う secondary material として扱います。
