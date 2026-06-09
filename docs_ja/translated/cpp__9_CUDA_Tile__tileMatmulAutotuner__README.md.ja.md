# cpp/9_CUDA_Tile/tileMatmulAutotuner/README.md Japanese Companion

Original English/source document: [`cpp/9_CUDA_Tile/tileMatmulAutotuner/README.md`](../../cpp/9_CUDA_Tile/tileMatmulAutotuner/README.md)

Role: README / run instructions

## English Reference

Headings:
- `tileMatmulAutotuner`
- `Description`
- `Running`
- `Run autotuner. Validation is disabled by default.`
- `Select a backend`
- `Enable CPU validation`
- `To run faster, skip warmups and set iteration to 1`
- `Show all options`
- `Command-Line Options`
- `Search Space Configuration`

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
