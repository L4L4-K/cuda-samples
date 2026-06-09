# Unified Memory

English anchor: managed allocations are visible to CPU and GPU while CUDA handles migration.

> **日本語**
> Unified Memory は CPU/GPU から同じ pointer を使える allocation ですが、同期と migration は消えません。
>
> **学習メモ**
> `cudaMallocManaged` は簡単さを提供しますが、page fault や prefetch が性能に影響します。

## What To Look For

- GPU が書いた managed data を CPU が読む前に同期が必要です。
- first-touch や prefetch の timing を測定範囲から分けます。
- multi-GPU では access pattern と device support を確認します。

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
