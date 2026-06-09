# Shared Memory

English anchor: shared memory is a programmer-managed block-local cache/workspace.

> **日本語**
> shared memory は block 内 thread が共有する高速な作業領域です。
>
> **学習メモ**
> `__shared__` を見つけたら、誰が書き、どの barrier の後で誰が読むかを追います。

## What To Look For

- matrix tile、stencil、convolution、reduction、transpose でよく使われます。
- bank conflict と occupancy の tradeoff を確認します。
- dynamic shared memory は kernel launch の第三引数に size が出ます。

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
