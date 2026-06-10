# tmaTensorMap (Python) - Japanese Learning Guide

English source: [README.md](README.md)
Python requirements: [requirements.txt](requirements.txt)

## English Overview

This sample demonstrates how to use Tensor Memory Accelerator (TMA) descriptors with `cuda.core` on Hopper and later GPUs (compute capability >= 9.0). TMA enables efficient bulk data movement between global and shared memory using hardware-managed tensor map descriptors, which are a key building block for modern GEMM kernels and large shared-memory tile loads.

The sample:

Original README headings: `tmaTensorMap (Python)`, `Description`, `What You'll Learn`, `Key Libraries`, `Key APIs`, `From `cuda.core``, `From `cuda.pathfinder``, `From `cuda_samples_utils``, `Requirements`, `Hardware`, `Software`, `Installation`

> **日本語**
> `python/2_CoreConcepts/tmaTensorMap` は `Python` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `tmaTensorMap` as a focused example of the CUDA concepts used in `python/2_CoreConcepts/tmaTensorMap`.
> **日本語**
> この sample の目的は、`tmaTensorMap` の小さな実装を通して Python CUDA, Runtime, Driver, And NVRTC, Multi-GPU, P2P, And IPC, Shared Memory, Streams And Events を具体的に追うことです。
>
> **学習メモ**
> 最初に `tmaTensorMap.py` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `requirements.txt`: Python package prerequisites for the sample.
- `tmaTensorMap.py`: Python entry point or helper using CUDA Python, CuPy, framework interop, or subprocess logic.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `tmaTensorMap.py` first and locate the host-side setup or Python entry point.
- Create or select the CUDA device/context and construct Python objects that wrap CUDA resources.
- Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.
- Enumerate devices, enable peer or IPC access, and record which device/process owns each resource.
- Inside the kernel, map thread/block indexes to tile elements and check the barrier around shared memory reuse.
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

- `tmaTensorMap.py`: focus on `CUDA`, `launch`, `CUDA_HOME`, `cp.float32`, `CUDA_PATH`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `tmaTensorMap.py`

Source: python/2_CoreConcepts/tmaTensorMap/tmaTensorMap.py:2-20
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

> JP: この抜粋は `python/2_CoreConcepts/tmaTensorMap/tmaTensorMap.py` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: python/2_CoreConcepts/tmaTensorMap/tmaTensorMap.py:39-58
```python
  1. Creates a TMA tiled descriptor from a CuPy device array via
     ``StridedMemoryView.from_any_interface(...).as_tensor_map(...)``.
  2. Passes the descriptor by value (as ``__grid_constant__``) to a
     kernel that uses libcudacxx TMA/barrier wrappers to bulk-load a
     tile into shared memory.
  3. Reuses the same descriptor against a new source tensor with
     ``replace_address()`` to avoid rebuilding it.

On GPUs older than Hopper (sm < 90), the sample prints a diagnostic
and exits cleanly.

Ported from ``cuda_core/examples/tma_tensor_map.py`` in the
`cuda-python` repository.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))
```

> JP: この抜粋は `python/2_CoreConcepts/tmaTensorMap/tmaTensorMap.py` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: python/2_CoreConcepts/tmaTensorMap/tmaTensorMap.py:93-122
```python
__global__ void tma_copy(
    const __grid_constant__ TensorMap tensor_map,
    float* output,
    int N)
{
    __shared__ __align__(128) float smem[TILE_SIZE];
    __shared__ TmaBarrier bar;

    const int tid        = threadIdx.x;
    const int tile_start = blockIdx.x * TILE_SIZE;

    if (tid == 0)
    {
        init(&bar, 1);
    }
    __syncthreads();

    if (tid == 0)
    {
        cuda::device::experimental::cp_async_bulk_tensor_1d_global_to_shared(
            smem,
            reinterpret_cast<const CUtensorMap*>(&tensor_map),
            tile_start,
            bar);
        bar.wait(cuda::device::barrier_arrive_tx(bar, 1, TILE_SIZE * sizeof(float)));
    }
    __syncthreads();

    if (tid < TILE_SIZE)
    {
```

> JP: この抜粋は `python/2_CoreConcepts/tmaTensorMap/tmaTensorMap.py` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: python/2_CoreConcepts/tmaTensorMap/tmaTensorMap.py:177-196
```python
        sys.exit(1)
    return include_path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Use a TMA tensor map to bulk-copy data on Hopper+ GPUs"
    )
    parser.add_argument(
        "--elements",
        type=int,
        default=1024,
        help="Total number of float32 elements (must be a multiple of 128)",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    args = parser.parse_args()

    if args.elements % TILE_SIZE != 0:
```

> JP: この抜粋は `python/2_CoreConcepts/tmaTensorMap/tmaTensorMap.py` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cupy` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `Device` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `CUBIN` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `Program` | 実行時 compile/link の境界です。log、module、kernel name の対応を確認します。 |
| `ProgramOptions` | 実行時 compile/link の境界です。log、module、kernel name の対応を確認します。 |
| `CUDA_HOME` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cp.float32` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `CUDA_PATH` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `LaunchConfig` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `__shared__` | block 内共有 memory または同期境界です。producer/consumer の順序を確認します。 |
| `__syncthreads` | block 内共有 memory または同期境界です。producer/consumer の順序を確認します。 |
| `CUtensorMap` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cp.zeros` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |

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
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。
- 同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。
- Unified Memory は pointer を共有しますが、migration、prefetch、同期の理解は必要です。

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
python tmaTensorMap.py
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
- shared memory を書いた thread と読む thread の間に必要な barrier を見落とす。
- different stream 間に依存があるのに event や explicit sync を置かない。
- JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。
- peer access が有効な device pair と、単に複数 GPU が存在することを混同する。
- CuPy/NumPy/DLPack 変換で hidden copy や hidden sync が起きる可能性を見落とす。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `launch` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- shared memory tile の producer、consumer、barrier を図にする。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
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
- [Shared Memory](../../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Unified Memory](../../../docs_ja/themes/unified_memory.md): managed memory、migration、prefetch の意味を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
