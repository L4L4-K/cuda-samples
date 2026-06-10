# Build And Run Cheat Sheet

English command or file | 日本語 | 学習メモ
--- | --- | ---
`README.md` | original English instructions | command、target、expected output の source of truth です。
`README.ja.md` | Japanese learning guide | 目的、流れ、注意点、関連 theme を補います。
`CMakeLists.txt` | CMake build definition | source、target、library、platform condition を読みます。
`cmake -S . -B build` | configure | generator と cache を作る段階です。
`cmake --build build` | build | compile/link を実行します。
`ctest --test-dir build` | CTest execution | build tree 内の tests を実行します。
`requirements.txt` | Python dependencies | Python CUDA sample の package 前提です。
`python sample.py` | Python sample execution | package、driver、GPU、stream sync を確認します。
`run_tests.py` | sample test runner | skip と success を分けて記録します。
`nvidia-smi` / `nvcc --version` | environment check | GPU/driver/toolkit availability を確認します。

## Build Reading Pattern

1. Start from root `README.md` for general commands.
2. Open the sample `README.md` and `README.ja.md` for target-specific prerequisites.
3. Open `CMakeLists.txt` or `requirements.txt` to find dependency and platform conditions.
4. Build only when CUDA Toolkit, compiler, GPU, and optional dependencies are available.
5. Record skipped checks honestly when the environment is missing.

> **日本語**
> configure、build、run、test は別の段階です。configure で止まるなら CMake/dependency、build で止まるなら compiler/link、run で止まるなら driver/GPU/runtime condition を疑います。
>
> **学習メモ**
> この日本語 overlay では sample behavior を変えないため、build/run command や output strings は翻訳しません。

## Common Mistakes

- source directory の中に generated files を混ぜて cleanup を難しくする。
- optional library がないのに library sample の source error だと思う。
- graphics/interop sample を headless environment で実行しようとする。
- Python requirements を install せずに CUDA Python sample を実行する。
- skipped build/test を success として記録する。

## Exercises

- 任意の C++ sample で target name、source files、linked libraries を `CMakeLists.txt` から抜き出す。
- Python sample で requirements と import boundary を対応させる。
- build failure を configure/build/run のどの段階か分類する。
