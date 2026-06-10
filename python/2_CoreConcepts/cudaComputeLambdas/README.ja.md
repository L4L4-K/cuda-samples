# cudaComputeLambdas (Python) - Japanese Learning Guide

English source: [README.md](README.md)
Python requirements: [requirements.txt](requirements.txt)

## English Overview

This sample demonstrates how **cuda.compute** (from the `cuda-cccl` package) accepts plain Python callables, including lambdas, as the operators that drive device-wide reductions, transforms, and scans. Internally `cuda.compute` JIT-compiles the callable through Numba for the GPU, so you can iterate on the operator in pure Python and still get a fused device-wide kernel.

The sample exercises three algorithm families:

Original README headings: `cudaComputeLambdas (Python)`, `Description`, `What You'll Learn`, `Key Libraries`, `Key APIs`, `From `cuda.compute``, `From `cuda_samples_utils``, `Requirements`, `Hardware`, `Software`, `Installation`, `How to Run`

> **日本語**
> `python/2_CoreConcepts/cudaComputeLambdas` は `Python` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `cudaComputeLambdas` as a focused example of the CUDA concepts used in `python/2_CoreConcepts/cudaComputeLambdas`.
> **日本語**
> この sample の目的は、`cudaComputeLambdas` の小さな実装を通して Python CUDA, Runtime, Driver, And NVRTC, Multi-GPU, P2P, And IPC, Memory, Kernel Launch And Indexing を具体的に追うことです。
>
> **学習メモ**
> 最初に `cudaComputeLambdas.py` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- Python 3 environment
- `requirements.txt` packages when the file is present
- The device topology required by the README, such as multiple GPUs, peer access, IPC, MPI, or process support

> **日本語**
> 必要条件は英語 README と CMake/requirements を優先します。この guide は条件を置き換えず、読むべき確認点を追加します。
>
> **学習メモ**
> 実行できない場合は、source を変える前に driver、toolkit、GPU feature、library、platform guard、Python package version を確認します。

## Files

- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cudaComputeLambdas.py`: Python entry point or helper using CUDA Python, CuPy, framework interop, or subprocess logic.
- `requirements.txt`: Python package prerequisites for the sample.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `cudaComputeLambdas.py` first and locate the host-side setup or Python entry point.
- Create or select the CUDA device/context and construct Python objects that wrap CUDA resources.
- Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.
- Enumerate devices, enable peer or IPC access, and record which device/process owns each resource.
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

- `cudaComputeLambdas.py`: focus on `CUDA`, `Device`, `cp.arange`, `cp.empty_like`, `cupy`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `cudaComputeLambdas.py`

Source: python/2_CoreConcepts/cudaComputeLambdas/cudaComputeLambdas.py:2-20
```python
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
```

> JP: この抜粋は `python/2_CoreConcepts/cudaComputeLambdas/cudaComputeLambdas.py` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: python/2_CoreConcepts/cudaComputeLambdas/cudaComputeLambdas.py:83-102
```python
    expected = int(d_in.get().sum())
    ok = got == expected
    print(
        f"reduce_into(lambda a,b: a+b) over 1..10 -> {got} "
        # JP: validation: GPU result を CPU/reference と比較する検証地点です。失敗時は transfer、indexing、sync の順に疑います。
        f"(expected {expected})  {'OK' if ok else 'FAIL'}"
    )
    return ok


def demo_unary_transform_lambda() -> bool:
    """unary_transform driven by a lambda: y = x*x + 1."""
    # JP: この連続する anchor 群では Python object と CUDA resource/context/stream の境界です。hidden sync と lifetime を確認します。
    d_in = cp.arange(8, dtype=cp.int32)
    d_out = cp.empty_like(d_in)

    cuda.compute.unary_transform(
        d_in=d_in,
        d_out=d_out,
        num_items=int(d_in.size),
```

> JP: この抜粋は `python/2_CoreConcepts/cudaComputeLambdas/cudaComputeLambdas.py` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: python/2_CoreConcepts/cudaComputeLambdas/cudaComputeLambdas.py:151-170
```python
    )
    return ok


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Drive cuda.compute device algorithms with Python lambdas / callables"
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    args = parser.parse_args()

    # JP: この anchor では Python object と CUDA resource/context/stream の境界です。hidden sync と lifetime を確認します。
    device = Device(args.device)
    device.set_current()
    print_gpu_info(device)
    print()

    ok = True
```

> JP: この抜粋は `python/2_CoreConcepts/cudaComputeLambdas/cudaComputeLambdas.py` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaComputeLambdas` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cupy` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `Device` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `CUDA_HOME` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cp.arange` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cp.empty_like` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cp.empty` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cp.int32` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cp.array` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- multi-GPU sample では、device 選択、peer capability、context/IPC handle の寿命を分けて読みます。
- Python sample では、Python object の lifetime と CUDA stream/context の lifetime が別であることを意識します。
- Runtime/Driver/NVRTC sample では、compile/load した module と launch する kernel 名の対応が重要です。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Sample-Specific Notes

- Python sample では、Python object が CUDA pointer、stream、module を包んでいます。見た目は Python でも、同期と lifetime は CUDA の規則で決まります。

> **日本語**
> この section は同じ template ではなく、sample 名、path、検出した API から読みどころを絞っています。
>
> **学習メモ**
> 似た名前の sample は Runtime 版、Driver 版、NVRTC 版、library 版の違いを比較すると学習効果が高くなります。

## Build And Run

English commands remain authoritative. Typical local Python flow:

```bash
pip install -r requirements.txt
python cudaComputeLambdas.py
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

Run the sample as documented and compare its output with the original README, validation message, generated file, or reference result.
> **日本語**
> 期待結果は英語の出力文字列、README の validation、生成 file、または reference result と照合します。
>
> **学習メモ**
> `PASS`、`Test passed`、error code、timing label などの出力文字列は翻訳せず、source と同じ表記で確認します。

## Common Mistakes

- API 名や target 名を翻訳してしまい、README や build command と対応できなくなる。
- allocation size を byte で渡す API と element count で考える loop を混同する。
- kernel launch が非同期であることを忘れ、同期前の結果を host 側で読んでしまう。
- JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。
- peer access が有効な device pair と、単に複数 GPU が存在することを混同する。
- CuPy/NumPy/DLPack 変換で hidden copy や hidden sync が起きる可能性を見落とす。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaComputeLambdas` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- device ごとの ownership と、peer/IPC で共有される resource を分けて書く。
- Python object、DLPack/CuPy view、CUDA buffer の lifetime を別々に書く。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Python CUDA](../../../docs_ja/themes/python_cuda.md): Python object が CUDA resource を包む境界と hidden sync を読むための基礎です。
- [Runtime, Driver, And NVRTC](../../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [Multi-GPU, P2P, And IPC](../../../docs_ja/themes/multi_gpu_p2p_ipc.md): device topology、peer access、IPC handle、multi-process 境界を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
