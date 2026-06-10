# interval - Interval Computing - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

Interval arithmetic operators example.  Uses various C++ features (templates and recursion).  The recursive mode requires Compute SM 2.0 capabilities.

Recursion, Templates

Original README headings: `interval - Interval Computing`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/2_Concepts_and_Techniques/interval` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `interval` as a focused example of the CUDA concepts used in `cpp/2_Concepts_and_Techniques/interval`.
> **日本語**
> この sample の目的は、`interval` の小さな実装を通して Streams And Events, Performance, Memory, Kernel Launch And Indexing, Execution Model を具体的に追うことです。
>
> **学習メモ**
> 最初に `config.hpp, borland_prefix.hpp, borland_suffix.hpp, msvc_prefix.hpp, msvc_suffix.hpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit

> **日本語**
> 必要条件は英語 README と CMake/requirements を優先します。この guide は条件を置き換えず、読むべき確認点を追加します。
>
> **学習メモ**
> 実行できない場合は、source を変える前に driver、toolkit、GPU feature、library、platform guard、Python package version を確認します。

## Files

- `.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `boost/config.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/abi/borland_prefix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/abi/borland_suffix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/abi/msvc_prefix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/abi/msvc_suffix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/abi_prefix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/abi_suffix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/auto_link.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/borland.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/codegear.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/comeau.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/common_edg.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/compaq_cxx.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/digitalmars.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/gcc.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/gcc_xml.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/greenhills.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/hp_acc.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/intel.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/kai.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/metrowerks.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/mpw.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/pgi.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/sgi_mipspro.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/sunpro_cc.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/vacpp.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/compiler/visualc.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/no_tr1/cmath.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/no_tr1/complex.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/no_tr1/functional.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/no_tr1/memory.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/no_tr1/utility.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/platform/aix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/platform/amigaos.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/platform/beos.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/platform/bsd.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/platform/cygwin.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/platform/hpux.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/platform/irix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/platform/linux.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/platform/macos.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/platform/qnxnto.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/platform/solaris.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/platform/vxworks.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/platform/win32.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/posix_features.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/requires_threads.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/select_compiler_config.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/select_platform_config.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/select_stdlib_config.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/stdlib/dinkumware.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/stdlib/libcomo.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/stdlib/libstdcpp3.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/stdlib/modena.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/stdlib/msl.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/stdlib/roguewave.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/stdlib/sgi.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/stdlib/stlport.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/stdlib/vacpp.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/suffix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/user.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/config/warning_disable.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/limits.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/arith.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/arith2.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/arith3.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/checking.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/compare.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/compare/certain.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/compare/explicit.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/compare/lexicographic.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/compare/possible.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/compare/set.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/compare/tribool.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/constants.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/detail/alpha_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/detail/bcc_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/detail/bugs.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/detail/c99_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/detail/c99sub_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/detail/division.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/detail/ia64_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/detail/interval_prototype.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/detail/msvc_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/detail/ppc_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/detail/sparc_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/detail/test_input.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/detail/x86_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/detail/x86gcc_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/ext/integer.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/ext/x86_fast_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/hw_rounding.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/interval.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/io.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/limits.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/policies.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/rounded_arith.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/rounded_transc.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/rounding.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/transc.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `boost/numeric/interval/utility.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `cpu_interval.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cuda_interval.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cuda_interval_lib.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cuda_interval_rounded_arith.h`: Host/device declarations, helper types, constants, or library wrappers.
- `interval.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `interval.h`: Host/device declarations, helper types, constants, or library wrappers.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `config.hpp` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Move, map, or expose input data so GPU work can read the intended values.
- Launch the kernel, graph, library call, or Python CUDA operation with the documented configuration.
- Synchronize only at the required correctness or timing boundary.
- Validate results against the CPU/reference path, generated artifact, or expected status message.
- Release CUDA, library, framework, graphics, or external resources in the reverse ownership order.

> **日本語**
> 実行の流れは setup、visibility、GPU work、sync、validation、cleanup の順に読みます。非同期 API がある場合は、host がいつ待つかを別に記録します。
>
> **学習メモ**
> CUDA の bug は kernel 本体だけでなく、copy direction、descriptor、stream dependency、cleanup order にも出ます。

## Concrete Reading Path

- `boost/config.hpp`: focus on control flow and helper functions.
- `boost/config/abi/borland_prefix.hpp`: focus on control flow and helper functions.
- `boost/config/abi/borland_suffix.hpp`: focus on control flow and helper functions.
- `boost/config/abi/msvc_prefix.hpp`: focus on control flow and helper functions.
- `boost/config/abi/msvc_suffix.hpp`: focus on control flow and helper functions.
- `boost/config/abi_prefix.hpp`: focus on control flow and helper functions.
- `boost/config/abi_suffix.hpp`: focus on control flow and helper functions.
- `boost/config/auto_link.hpp`: focus on control flow and helper functions.
- `boost/config/compiler/borland.hpp`: focus on control flow and helper functions.
- `boost/config/compiler/codegear.hpp`: focus on control flow and helper functions.
- Additional source files: 98 more support files. Use the same setup/work/sync/cleanup lens.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaEventDestroy` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaEventCreate` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaEventRecord` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaGetDeviceProperties` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaSetDevice` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaFuncSetCacheConfig` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDeviceSetLimit` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGetLastError` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMemcpyDeviceToHost` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target interval
ctest --test-dir build -R interval
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

The sample prints timing, bandwidth, latency, throughput, or comparison data; exact values depend on hardware and driver.
> **日本語**
> 期待結果は英語の出力文字列、README の validation、生成 file、または reference result と照合します。
>
> **学習メモ**
> `PASS`、`Test passed`、error code、timing label などの出力文字列は翻訳せず、source と同じ表記で確認します。

## Common Mistakes

- API 名や target 名を翻訳してしまい、README や build command と対応できなくなる。
- allocation size を byte で渡す API と element count で考える loop を混同する。
- kernel launch が非同期であることを忘れ、同期前の結果を host 側で読んでしまう。
- different stream 間に依存があるのに event や explicit sync を置かない。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaMalloc` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Performance](../../../docs_ja/themes/performance.md): memory traffic、occupancy、overlap、launch overhead、timing を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
