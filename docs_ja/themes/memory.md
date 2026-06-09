# CUDA Memory

English anchor: samples use host, device, pinned, mapped, managed, array, and external memory.

> **日本語**
> CUDA の memory は、どこに確保され、誰が読めて、いつ最新になるかで意味が変わります。
>
> **学習メモ**
> `cudaMalloc` と `malloc` は同じ pointer 取得に見えても、所有者と access できる processor が違います。

## What To Look For

- device memory は CUDA API で確保し、対応する API で解放します。
- pinned memory は async copy と overlap の前提になることがあります。
- external memory は CUDA 以外の API が所有する lifetime も読みます。

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
