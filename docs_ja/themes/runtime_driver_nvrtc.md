# Runtime, Driver, And NVRTC

English anchor: samples may use Runtime API, Driver API, and runtime compilation.

> **日本語**
> Runtime API、Driver API、NVRTC は kernel を準備して起動する層が異なります。
>
> **学習メモ**
> Driver/NVRTC sample では context、module、function、compile log の扱いが増えます。

## What To Look For

- Runtime API は `cuda` prefix と `<<<...>>>` launch が中心です。
- Driver API は `cuModuleLoad` や `cuLaunchKernel` で module/function を明示します。
- NVRTC は実行時 compile から PTX/module loading までを読みます。

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
