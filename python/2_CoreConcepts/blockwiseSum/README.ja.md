# Sample: Block-wise Array Sum (Python) - Japanese Learning Guide

English source: [README.md](README.md)
Python requirements: [requirements.txt](requirements.txt)

## English Overview

Demonstrates fundamental CUDA thread cooperation: thread/block indexing, strided loops, and block-wise reduction using shared memory. This sample shows three progressively complex kernel patterns using the **cuda.core API**:

1. **Simple indexing** - One thread per element 2. **Strided loop** - Each thread processes multiple elements 3. **Block partial sum** - Shared memory reduction within each block

Original README headings: `Sample: Block-wise Array Sum (Python)`, `Description`, `What You'll Learn`, `Key Concepts`, `Thread and Block Indexing`, `Strided Loop Pattern`, `Key APIs`, `From `cuda.core`:`, `From CuPy:`, `Requirements`, `Hardware:`, `Software:`

> **日本語**
> `python/2_CoreConcepts/blockwiseSum` は `Python` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `blockwiseSum` as a focused example of the CUDA concepts used in `python/2_CoreConcepts/blockwiseSum`.
> **日本語**
> この sample の目的は、`blockwiseSum` の小さな実装を通して Python CUDA, Runtime, Driver, And NVRTC, Multi-GPU, P2P, And IPC, Shared Memory, Streams And Events を具体的に追うことです。
>
> **学習メモ**
> 最初に `blockwiseSum.py` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `blockwiseSum.py`: Python entry point or helper using CUDA Python, CuPy, framework interop, or subprocess logic.
- `requirements.txt`: Python package prerequisites for the sample.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `blockwiseSum.py` first and locate the host-side setup or Python entry point.
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

- `blockwiseSum.py`: focus on `CUDA`, `launch`, `blockDim`, `blockIdx`, `threadIdx`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `blockwiseSum.py`

Source: python/2_CoreConcepts/blockwiseSum/blockwiseSum.py:2-20
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

> JP: この抜粋は `python/2_CoreConcepts/blockwiseSum/blockwiseSum.py` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: python/2_CoreConcepts/blockwiseSum/blockwiseSum.py:29-48
```python
Block-wise Array Sum with Threaded Access

Demonstrates thread/block indexing, strided loops, and block-wise reduction.

Key Concepts:
    Global Thread ID = blockIdx.x * blockDim.x + threadIdx.x
    Stride = blockDim.x * gridDim.x
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))
from cuda_samples_utils import verify_array_result

try:
    # JP: python_cuda: Python object が CUDA resource を包みます。Python から見えても device memory/stream/context の寿命と順序は CUDA 側で管理します。
    import cupy as cp
    import numpy as np
    from cuda.core import (
```

> JP: この抜粋は `python/2_CoreConcepts/blockwiseSum/blockwiseSum.py` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: python/2_CoreConcepts/blockwiseSum/blockwiseSum.py:92-111
```python
    float sum = 0.0f;
    for (size_t i = tid; i < N; i += stride) {
        sum += input[i];
    }
    sdata[local_tid] = sum;
    __syncthreads();

    // Block-level tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_tid < s) {
            sdata[local_tid] += sdata[local_tid + s];
        }
        __syncthreads();
    }

    if (local_tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}
"""
```

> JP: この抜粋は `python/2_CoreConcepts/blockwiseSum/blockwiseSum.py` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: python/2_CoreConcepts/blockwiseSum/blockwiseSum.py:129-148
```python
    """
    threads_per_block = 256
    num_blocks = 64

    # JP: この anchor では Python object と CUDA resource/context/stream の境界です。hidden sync と lifetime を確認します。
    device = Device(device_id)
    device.set_current()
    stream = device.create_stream()

    arch = f"sm_{device.arch}"
    print(f"Device: {device.name}")
    print(f"Compute Capability: {arch}")
    print(f"Array size: {num_elements:,} elements\n")

    try:
        # Make CuPy use our stream
        # JP: この anchor では Python object と CUDA resource/context/stream の境界です。hidden sync と lifetime を確認します。
        cp.cuda.Stream.from_external(stream).use()

        # Compile kernels
```

> JP: この抜粋は `python/2_CoreConcepts/blockwiseSum/blockwiseSum.py` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `Device` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `LaunchConfig` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `gridDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `Program` | 実行時 compile/link の境界です。log、module、kernel name の対応を確認します。 |
| `ProgramOptions` | 実行時 compile/link の境界です。log、module、kernel name の対応を確認します。 |
| `__syncthreads` | block 内共有 memory または同期境界です。producer/consumer の順序を確認します。 |
| `cp.asarray` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cp.uint64` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cupy` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cp.zeros_like` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- multi-GPU sample では、device 選択、peer capability、context/IPC handle の寿命を分けて読みます。
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
- Python sample では、Python object の lifetime と CUDA stream/context の lifetime が別であることを意識します。
- Runtime/Driver/NVRTC sample では、compile/load した module と launch する kernel 名の対応が重要です。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。
- 同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。

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
python blockwiseSum.py
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

- `blockDim` の直前と直後で、どの memory/resource が有効になったかをメモする。
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
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Performance](../../../docs_ja/themes/performance.md): memory traffic、occupancy、overlap、launch overhead、timing を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
