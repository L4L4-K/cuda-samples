# CUDA Terms Glossary

English term | 日本語 | 学習メモ
--- | --- | ---
Host | CPU 側 | CUDA API 呼び出し、入力準備、結果検証を行う側です。
Device | GPU 側 | kernel や CUDA library work が実行される側です。
Kernel | GPU で実行される関数 | `<<<...>>>`、Driver API、Python `launch` で起動されます。
Grid | block の集合 | 1 回の kernel launch で投入される全 work です。
Block | thread の集合 | shared memory と `__syncthreads()` の基本範囲です。
Thread | 最小実行単位 | `threadIdx` などで担当 data を決めます。
Warp | 通常 32 thread の実行単位 | warp-level API、分岐、coalescing で重要です。
Stream | 非同期 work queue | 同じ stream 内は投入順、別 stream 間は明示依存が必要です。
Event | stream timeline marker | 完了確認、依存、timing に使います。
Pinned Memory | page-locked host memory | async transfer と overlap の条件になります。
Managed Memory | CPU/GPU 共有 pointer | migration、prefetch、sync を読みます。
Shared Memory | block 内 scratchpad | producer/consumer と barrier を対応させます。
Atomic | 競合 update の保護 | correctness を守りますが contention で遅くなります。
Descriptor | library resource | data layout や operation shape を library に渡します。
Workspace | temporary storage | library/algorithm が一時的に使う device memory です。

## Reading Notes

> **日本語**
> 用語は単独で覚えるより、source 上の API と対応させて読みます。たとえば `Block` は launch shape、shared memory、barrier、occupancy の複数 theme にまたがります。
>
> **学習メモ**
> English term を翻訳しすぎると公式 docs や source search と対応しにくくなります。term は英語のまま、意味を日本語で補います。

## Common Confusions

- Host pointer と device pointer は同じ pointer 型に見えても access できる processor が違います。
- Stream ordering と host synchronization は別です。stream に入った順序と host が待つ位置を分けます。
- Shared memory と managed memory は名前が似ていてもまったく別です。shared memory は block 内、managed memory は CPU/GPU 共有 pointer です。
- Atomic は race を防ぎますが、algorithm 全体の実行順を deterministic にするわけではありません。

## Exercises

- 任意の sample で Host、Device、Kernel、Stream、Validation の行を 1 つずつ探す。
- `vectorAdd` と `matrixMul` で Block/Thread の意味がどう変わるか説明する。
- library sample で Descriptor と Workspace の作成、利用、破棄を表にする。
