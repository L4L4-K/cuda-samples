# CMakeLists.txt Japanese Companion

Original English/build file: [`CMakeLists.txt`](../../CMakeLists.txt)

## English Reference

The root `CMakeLists.txt` configures the CUDA Samples project, sets global build behavior, discovers packages, and includes subdirectories that contain sample targets.

> **日本語**
> root `CMakeLists.txt` は repository 全体の build entry point です。ここで project configuration、optional package discovery、global compile behavior、sample category subdirectories の読み込みが決まります。
>
> **学習メモ**
> CMake は runtime CUDA logic ではありませんが、どの source が compile され、どの library と link され、どの platform 条件で sample が含まれるかを決めます。

## Project And Language Setup

English anchor: project settings establish C++/CUDA build mode and repository-wide assumptions.

> **日本語**
> project/language 設定は、CUDA compiler と host compiler の組み合わせ、C++ standard、CUDA language handling を決めます。sample code を読む前に、CMake が CUDA を language として扱うのか、toolkit package として扱うのかを確認します。
>
> **学習メモ**
> CMake configure の失敗は source の syntax error ではなく、compiler discovery、toolkit path、generator、architecture option の問題であることがあります。

## Subdirectories And Targets

English anchor: `add_subdirectory` connects repository directories to build targets.

> **日本語**
> root file は category directory を build graph に追加します。個別 target の source list、libraries、compile options は各 sample directory の `CMakeLists.txt` にあります。
>
> **学習メモ**
> sample が build されない場合、root で directory が含まれているか、category CMake で除外されていないか、sample CMake で platform guard がないかを順に確認します。

## Dependencies And Optional Libraries

English anchor: CUDA samples may depend on CUDA Toolkit libraries, graphics stacks, platform SDKs, Python packages, or downloaded dependencies.

> **日本語**
> CMake の `find_package`、`target_link_libraries`、option guard は library sample の実行条件を示します。cuBLAS/cuFFT/cuSolver/NPP/nvJPEG/CUB/graphics interop などは source だけでなく link 設定も読まないと理解できません。
>
> **学習メモ**
> library handle や descriptor を source で見つけたら、CMake でどの library target と link しているかも確認します。

## Build Configuration Reading Checklist

- Identify which subdirectory brings the sample into the build.
- Check target name, source files, include directories, linked libraries, and compile definitions.
- Check CUDA architecture settings and platform-specific guards.
- Distinguish configure-time dependency discovery from compile/link-time errors.
- Compare root settings with per-sample `CMakeLists.txt` and `README.ja.md`.

> **日本語**
> CMake file を読むときは「target が存在するか」「source が入っているか」「link が足りているか」「platform guard で除外されていないか」を分けます。
>
> **学習メモ**
> CUDA code の error と build system の error を混ぜないことが重要です。compiler が呼ばれる前に止まっているなら CMake/configuration の問題です。
