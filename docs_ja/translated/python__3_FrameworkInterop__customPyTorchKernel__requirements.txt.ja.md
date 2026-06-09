# python/3_FrameworkInterop/customPyTorchKernel/requirements.txt Japanese Companion

Original English/source document: [`python/3_FrameworkInterop/customPyTorchKernel/requirements.txt`](../../python/3_FrameworkInterop/customPyTorchKernel/requirements.txt)

Role: Python dependency list

## English Reference

Headings:
- `Custom PyTorch Kernel Sample Requirements`
- `NOTE: On Windows, the default `torch` wheel from PyPI is CPU-only and the`
- `sample will fail with "Torch not compiled with CUDA enabled". Install a`
- `CUDA-enabled torch from PyTorch's wheel index first (see README.md):`
- `pip install torch --index-url https://download.pytorch.org/whl/cu128`

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
