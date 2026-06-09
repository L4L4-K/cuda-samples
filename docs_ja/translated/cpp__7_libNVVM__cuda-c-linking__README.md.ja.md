# cpp/7_libNVVM/cuda-c-linking/README.md Japanese Companion

Original English/source document: [`cpp/7_libNVVM/cuda-c-linking/README.md`](../../cpp/7_libNVVM/cuda-c-linking/README.md)

Role: README / run instructions

## English Reference

Key lines:
- Introduction
- ============
- This sample demonstrates linking a libnvvm-generated module with an existing
- CUDA C library. The LLVM C++ API is used to generate an LLVM IR module that
- conforms to the NVVM IR specification and contains a call to an externally-
- defined function, and this module is compiled to PTX with libnvvm. The JIT
- linker (part of the CUDA Driver API) is then used to assemble the PTX and link
- it with the math library, creating a linked CUBIN image. This image is then

> **日本語**
> sample または directory の目的、前提、実行手順、期待動作を説明する英語 README です。
>
> **学習メモ**
> 英語の command、API、target、expected output、license、attribution は翻訳せず、ここでは読む順序と注意点を補います。

## How To Read This With The Code

- 目的と前提条件を確認します。
- build/run command を元の英語のまま辿ります。
- expected output と validation 条件を確認します。
- 隣の `README.ja.md` がある場合は、日本語の flow と mistakes も合わせて読みます。

> **日本語**
> document 単体で理解しようとせず、同じ directory の source、CMake、README、data/reference file と対応付けます。
>
> **学習メモ**
> build/run document は挙動を決めることがありますが、この companion は説明だけで build output や sample output を変更しません。

## Related Japanese Material

- [docs_ja/README.md](../README.md)
- [Build And Run Cheatsheet](../glossary/build_run.md)
- [Debugging, Profiling, And Testing](../themes/debugging_profiling_testing.md)
