# Streams And Events

English anchor: streams order asynchronous work and events mark points in stream timelines.

> **日本語**
> stream は work の順序を持つ queue、event は stream 上の時点を記録する marker です。
>
> **学習メモ**
> 別 stream の work は重なる可能性がありますが、正しさには event や synchronize による依存関係が必要です。

## What To Look For

- `cudaMemcpyAsync` は stream と pinned memory の条件を合わせて読みます。
- `cudaEventRecord` と `cudaEventSynchronize` は timing と待ち合わせで役割が違います。
- `cudaDeviceSynchronize` は全体待ちなので overlap を消すことがあります。

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
