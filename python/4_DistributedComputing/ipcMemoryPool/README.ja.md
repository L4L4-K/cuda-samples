# ipcMemoryPool (Python) - Japanese Learning Guide

English source: [README.md](README.md)
Python requirements: [requirements.txt](requirements.txt)

## English Overview

This sample demonstrates how to share GPU memory between Python processes using CUDA Inter-Process Communication (IPC) and `cuda.core`'s IPC-enabled memory pools.

By default each process has its own CUDA virtual address space and cannot see allocations made by another process. With an IPC-enabled `DeviceMemoryResource` the parent allocates once, and the child process maps that same physical GPU memory into its own address space so both read and write the same bytes. The sample performs a round-trip test:

Original README headings: `ipcMemoryPool (Python)`, `Description`, `What You'll Learn`, `Key Libraries`, `Key APIs`, `From `cuda.core``, `From `cuda_samples_utils``, `Requirements`, `Hardware`, `Software`, `Installation`, `How to Run`

> **日本語**
> `python/4_DistributedComputing/ipcMemoryPool` は `Python` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `ipcMemoryPool` as a focused example of the CUDA concepts used in `python/4_DistributedComputing/ipcMemoryPool`.
> **日本語**
> この sample の目的は、`ipcMemoryPool` の小さな実装を通して Python CUDA, Multi-GPU, P2P, And IPC, Streams And Events, Performance, Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `ipcMemoryPool.py` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `ipcMemoryPool.py`: Python entry point or helper using CUDA Python, CuPy, framework interop, or subprocess logic.
- `requirements.txt`: Python package prerequisites for the sample.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `ipcMemoryPool.py` first and locate the host-side setup or Python entry point.
- Create or select the CUDA device/context and construct Python objects that wrap CUDA resources.
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

- `ipcMemoryPool.py`: focus on `CUDA`, `DeviceMemoryResource`, `Device`, `cp.float32`, `cp.arange`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `ipcMemoryPool.py`

Source: python/4_DistributedComputing/ipcMemoryPool/ipcMemoryPool.py:2-20
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

> JP: この抜粋は `python/4_DistributedComputing/ipcMemoryPool/ipcMemoryPool.py` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: python/4_DistributedComputing/ipcMemoryPool/ipcMemoryPool.py:30-49
```python

Share GPU memory between Python processes using CUDA Inter-Process
Communication (IPC) and cuda.core's IPC-enabled memory pools. By default
each worker process has its own CUDA virtual address space and cannot see
allocations made by another process. With an IPC-enabled
``DeviceMemoryResource`` the parent can allocate once, and the child
process can map that same physical GPU memory into its own address space
so both read and write the same bytes.

The sample does a round-trip test:

  1. Parent creates an IPC-enabled ``DeviceMemoryResource`` and allocates
     a ``Buffer``.
  2. Parent fills the buffer with a known pattern.
  3. Parent sends the ``Buffer`` to a child process through an
     ``mp.Queue`` - cuda.core's pickle reducers take care of re-creating
     the memory resource and mapping the buffer in the child.
  4. Child verifies the parent's pattern, writes a new pattern, and
     signals completion.
  5. Parent verifies the child's writes.
```

> JP: この抜粋は `python/4_DistributedComputing/ipcMemoryPool/ipcMemoryPool.py` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: python/4_DistributedComputing/ipcMemoryPool/ipcMemoryPool.py:98-136
```python


def child_worker(q_in, q_out, n_elements, parent_seed, child_seed):
    """Runs in a separate process. Verifies and modifies the shared buffer."""
    # JP: この anchor では Python object と CUDA resource/context/stream の境界です。hidden sync と lifetime を確認します。
    device = Device(0)
    device.set_current()
    pid = mp.current_process().pid

    # The Buffer (and its MR) are reconstructed and mapped in this process
    # when the queued object is unpickled. Both ``is_mapped`` flags are
    # True here.
    buffer = q_in.get(timeout=CHILD_TIMEOUT_SEC)
    print(
        f"[child pid={pid}] received buffer: is_mapped={buffer.is_mapped}, "
        f"size={buffer.size}"
    )

    # Build a zero-copy CuPy view of the shared device memory.
    # JP: この連続する anchor 群では Python object と CUDA resource/context/stream の境界です。hidden sync と lifetime を確認します。
    arr = cp.from_dlpack(buffer).view(dtype=cp.float32)

    # Verify the parent's pattern.
    expected_parent = cp.arange(n_elements, dtype=cp.float32) + float(parent_seed)
    if not cp.allclose(arr, expected_parent):
        print("[child] ERROR: parent's pattern did not match expectation")
        buffer.close()
        q_out.put("fail")
        return

    # Write a new pattern for the parent to verify.
    arr[:] = cp.arange(n_elements, dtype=cp.float32) * float(child_seed)
    device.sync()

    buffer.close()
    q_out.put("done")


def main() -> int:
```

> JP: この抜粋は `python/4_DistributedComputing/ipcMemoryPool/ipcMemoryPool.py` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `DeviceMemoryResource` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `Device` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cupy` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cp.float32` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cp.arange` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cp.from_dlpack` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cp.allclose` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- multi-GPU sample では、device 選択、peer capability、context/IPC handle の寿命を分けて読みます。
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
- Python sample では、Python object の lifetime と CUDA stream/context の lifetime が別であることを意識します。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Sample-Specific Notes

- multi-GPU/P2P/IPC 系では、どの process/thread/device が resource を所有しているかを先に分けると読みやすくなります。
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
python ipcMemoryPool.py
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
- peer access が有効な device pair と、単に複数 GPU が存在することを混同する。
- CuPy/NumPy/DLPack 変換で hidden copy や hidden sync が起きる可能性を見落とす。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `DeviceMemoryResource` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
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
- [Multi-GPU, P2P, And IPC](../../../docs_ja/themes/multi_gpu_p2p_ipc.md): device topology、peer access、IPC handle、multi-process 境界を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Performance](../../../docs_ja/themes/performance.md): memory traffic、occupancy、overlap、launch overhead、timing を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
