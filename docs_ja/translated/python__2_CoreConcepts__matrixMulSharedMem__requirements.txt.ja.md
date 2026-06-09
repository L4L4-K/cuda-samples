# python/2_CoreConcepts/matrixMulSharedMem/requirements.txt Japanese Companion

Original English/source document: [`python/2_CoreConcepts/matrixMulSharedMem/requirements.txt`](../../python/2_CoreConcepts/matrixMulSharedMem/requirements.txt)

Role: Python dependency list

## English Reference

Headings:
- `Matrix Multiplication with Shared Memory (GEMM) Requirements`
- `IMPORTANT: this sample pins older versions of cuda-core and nvmath-python`
- `on purpose. nvmath-python 0.9.0 (the current CUDA-13 release at the time`
- `of CTK 13.3) calls cuda-core's pre-1.0 API name `EventOptions(enable_timing=...)``
- `in its own internals. With cuda-core 1.0+ that kwarg was renamed to`
- ``timing_enabled` and the old name is rejected, so any cuda-core>=1.0 +`
- `nvmath-python 0.9.0 combination raises a TypeError at runtime.`
- `Until nvmath-python ships a release that targets the cuda-core 1.0 naming`
- `audit, this sample requires the older cuda-core 0.7 line. Installing this`
- `requirements.txt into the same environment as the other samples will`

> **日本語**
> Python sample の依存 package を示す file です。実行前に version と CUDA 対応 wheel を確認します。
>
> **学習メモ**
> 英語の command、API、target、expected output、license、attribution は翻訳せず、ここでは読む順序と注意点を補います。

## How To Read This With The Code

- 依存 package が GPU array、JIT、framework interop、test helper のどれを担うか確認します。
- CUDA version と wheel name が一致しているか確認します。
- install 成功と sample 実行成功は別なので、実行時 error も記録します。

> **日本語**
> document 単体で理解しようとせず、同じ directory の source、CMake、README、data/reference file と対応付けます。
>
> **学習メモ**
> build/run document は挙動を決めることがありますが、この companion は説明だけで build output や sample output を変更しません。

## Related Japanese Material

- [docs_ja/README.md](../README.md)
- [Build And Run Cheatsheet](../glossary/build_run.md)
- [Debugging, Profiling, And Testing](../themes/debugging_profiling_testing.md)
