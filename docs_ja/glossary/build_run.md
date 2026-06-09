# Build And Run Cheat Sheet

English command or file | 日本語 | 学習メモ
--- | --- | ---
`README.md` | original English instructions | command、target、expected output の source of truth です。
`README.ja.md` | Japanese learning guide | 目的、流れ、注意点、関連 theme を補います。
`CMakeLists.txt` | CMake build definition | source、target、library、platform condition を読みます。
`cmake -S . -B build` | configure | generator と cache を作る段階です。
`cmake --build build` | build | compile/link を実行します。
`requirements.txt` | Python dependencies | Python CUDA sample の package 前提です。
`run_tests.py` | sample test runner | skip と success を分けて記録します。

> **日本語**
> 実際の command は英語 README を優先し、日本語 guide は「なぜその手順があるか」を補います。
>
> **学習メモ**
> build 成功、test 成功、skip は別の状態として final report に残します。
