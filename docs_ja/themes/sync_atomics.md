# Synchronization And Atomics

English anchor: synchronization controls ordering and atomics protect shared updates.

> **日本語**
> 同期は「いつ読んでよいか」を決め、atomic は複数 thread が同じ場所を更新する競合を防ぎます。
>
> **学習メモ**
> `__syncthreads()` は block 内だけです。grid 全体の同期ではありません。

## What To Look For

- shared memory を読む前に producer thread が書き終わっているか見ます。
- atomic は lost update を防ぎますが、高 contention では遅くなります。
- memory fence は順序を作りますが、全 thread の到着待ちとは限りません。

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
