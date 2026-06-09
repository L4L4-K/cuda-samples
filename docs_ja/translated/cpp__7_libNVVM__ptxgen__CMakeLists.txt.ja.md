# cpp/7_libNVVM/ptxgen/CMakeLists.txt Japanese Companion

Original English/source document: [`cpp/7_libNVVM/ptxgen/CMakeLists.txt`](../../cpp/7_libNVVM/ptxgen/CMakeLists.txt)

Role: CMake build configuration

## English Reference

Headings:
- `Copyright (c) 1993-2023, NVIDIA CORPORATION. All rights reserved.`
- `Redistribution and use in source and binary forms, with or without`
- `modification, are permitted provided that the following conditions`
- `are met:`
- `* Redistributions of source code must retain the above copyright`
- `notice, this list of conditions and the following disclaimer.`
- `* Redistributions in binary form must reproduce the above copyright`
- `notice, this list of conditions and the following disclaimer in the`
- `documentation and/or other materials provided with the distribution.`
- `* Neither the name of NVIDIA CORPORATION nor the names of its`

> **日本語**
> CMake の build 設定です。どの source が target に入り、どの CUDA library と link され、どの platform 条件で有効になるかを確認します。
>
> **学習メモ**
> 英語の command、API、target、expected output、license、attribution は翻訳せず、ここでは読む順序と注意点を補います。

## How To Read This With The Code

- `project` で language と target scope を確認します。
- `find_package` と `target_link_libraries` で依存 library を確認します。
- `add_executable` や `add_subdirectory` で build graph に入る source/child directory を確認します。
- platform guard がある場合は、実行できない理由が environment 条件か code 条件かを切り分けます。

> **日本語**
> document 単体で理解しようとせず、同じ directory の source、CMake、README、data/reference file と対応付けます。
>
> **学習メモ**
> build/run document は挙動を決めることがありますが、この companion は説明だけで build output や sample output を変更しません。

## Related Japanese Material

- [docs_ja/README.md](../README.md)
- [Build And Run Cheatsheet](../glossary/build_run.md)
- [Debugging, Profiling, And Testing](../themes/debugging_profiling_testing.md)
