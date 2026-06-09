# Performance

English anchor: performance depends on memory traffic, launch overhead, occupancy, overlap, and instruction mix.

> **日本語**
> CUDA performance は演算だけでなく memory traffic、launch overhead、occupancy、overlap、同期で決まります。
>
> **学習メモ**
> 測っている範囲が setup、transfer、kernel、validation のどれを含むかを最初に確認します。

## What To Look For

- CUDA event timing と CPU timer の違いを読みます。
- occupancy は指標であり、単独で性能を保証しません。
- coalescing、shared memory reuse、library plan reuse を探します。

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
