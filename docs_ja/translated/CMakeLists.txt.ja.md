# CMakeLists.txt Japanese Companion

Original English/build file: [`CMakeLists.txt`](../../CMakeLists.txt)

## English Reference

The root `CMakeLists.txt` configures the CUDA Samples project, establishes global build behavior, and includes subdirectories that contain sample targets.

> **日本語**
> root の `CMakeLists.txt` は、repository 全体の build 設定を決め、各 sample directory の CMake 設定を読み込む入口です。
>
> **学習メモ**
> CMake file は実行時の CUDA logic ではありませんが、どの source が compile され、どの library と link され、どの platform 条件で除外されるかを決めます。

## Reading Points

- Project and language settings determine CUDA/C++ build mode.
- Options and cache variables control optional behavior.
- Subdirectories map repository structure to build targets.
- Platform checks protect samples that require graphics, Tegra, Windows, or special libraries.

> **日本語**
> build できない sample を調べるときは、source code だけでなく CMake の条件分岐を確認します。
>
> **学習メモ**
> `add_subdirectory` は sample 群を build graph に追加する境界です。個別 target の詳細は各 sample の `CMakeLists.txt` にあります。
