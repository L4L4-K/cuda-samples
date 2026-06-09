# cpp/6_Performance/alignedTypes/doc/alignedTypes.txt Japanese Companion

Original English/source document: [`cpp/6_Performance/alignedTypes/doc/alignedTypes.txt`](../../cpp/6_Performance/alignedTypes/doc/alignedTypes.txt)

Role: Supplemental technical document

## English Reference

Key lines:
- CUDA programming language, being a C with extensions, offers the ability to use arbitrary data structures in GPU programs. But in order for the hardware to perform efficient global
- Take a look at this structure definition:
- typedef struct{
- float a;
- float b;
- } testStructure;
- Without alignment specification the compiler will not automatically use a single 64-bit global memory load/store instruction, but will emit two 32-bit load instructions instead.
- This significantly impacts aggregate load/store bandwidth, since the latter breaks coalescing rules because of incontiguous memory access pattern. Refer to section 5.1.2.1 of the P

> **日本語**
> 補足技術資料です。binary 形式の場合は原文 file を変更せず、この companion で読み方を補います。
>
> **学習メモ**
> 英語の command、API、target、expected output、license、attribution は翻訳せず、ここでは読む順序と注意点を補います。

## How To Read This With The Code

- この document が input、reference、design note、build note のどれかを確認します。
- 関連する source file と README を同じ directory から探します。
- binary document は変換せず、必要な時だけ原文 viewer で確認します。

> **日本語**
> document 単体で理解しようとせず、同じ directory の source、CMake、README、data/reference file と対応付けます。
>
> **学習メモ**
> build/run document は挙動を決めることがありますが、この companion は説明だけで build output や sample output を変更しません。

## Related Japanese Material

- [docs_ja/README.md](../README.md)
- [Build And Run Cheatsheet](../glossary/build_run.md)
- [Debugging, Profiling, And Testing](../themes/debugging_profiling_testing.md)
