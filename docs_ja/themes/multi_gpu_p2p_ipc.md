# Multi-GPU, P2P, And IPC

English anchor: multi-GPU samples coordinate devices, peer access, IPC handles, and external sync.

> **日本語**
> multi-GPU では device selection、peer access、IPC handle、data partition、集約同期を追います。
>
> **学習メモ**
> `cudaSetDevice` の位置を追うと、後続の allocation と launch がどの GPU に属するか分かります。

## What To Look For

- P2P capability は GPU 組み合わせごとに確認します。
- IPC は pointer 共有ではなく handle の export/import と lifetime が重要です。
- 全 device/stream/process の完了を待ってから validation します。

> **日本語**
> code を読むときは、API 名を英語のまま保ち、その API が何を所有し、何を待ち、何を計測しているかを日本語で補います。
>
> **学習メモ**
> 同じ theme を持つ複数 sample を比較すると、基本 pattern と例外が見えます。


## Reading Checklist

- Identify who owns each allocation and which API releases it.
- Find the host-to-device or mapping point that makes input visible to GPU work.
- Find the kernel launch, library call, graph launch, or Python framework call that performs device work.
- Find the synchronization boundary before host-side validation or output.
- Decide whether the sample is mainly about correctness, interoperability, or performance.

> **日本語**
> CUDA サンプルは、所有権、転送、起動、同期、検証、後片付けの順に読むと構造が見えます。
>
> **学習メモ**
> 速さを読む前に、まず「どの memory を誰がいつ読むか」を確認します。
