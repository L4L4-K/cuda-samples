# cpp/0_Introduction/matrixMulDynlinkJIT/extras/README.TXT Japanese Companion

Original English/source document: [`cpp/0_Introduction/matrixMulDynlinkJIT/extras/README.TXT`](../../cpp/0_Introduction/matrixMulDynlinkJIT/extras/README.TXT)

Role: README / run instructions

## English Reference

Key lines:
- The auto-generated pair of files named matrixMul_ptxdump.c and
- matrixMul_ptxdump.h can be acquired by treating matrixMul_kernel.ptx
- as binary file and representing its contents as an array of chars.
- An example of the script written in Python language that performs
- such translation can be found in "extras" directory of the sample.
- The matrixMul_kernel.ptx contains the same PTX code as the file in
- "data" directory of matrixMulDrv sample after compilation by nvcc.
- The command line for generation using Python script is as follows:

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
