# CUDA Libraries

English anchor: libraries expose optimized GPU operations through handles, descriptors, plans, and workspaces.

> **日本語**
> CUDA library sample では、handle、descriptor、plan、workspace、stream、data layout を API に渡します。
>
> **学習メモ**
> library call は直接書いた kernel の代わりに device work を発行する境界です。

## What To Look For

- cuBLAS/cuFFT/cuSolver/NPP/CUB/nvJPEG などは API contract が重要です。
- leading dimension、stride、pitch、batch size を混同しないようにします。
- workspace と descriptor の cleanup を allocation と対応付けます。

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
