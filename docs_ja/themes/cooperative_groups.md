# Cooperative Groups

English anchor: cooperative groups make participating thread sets explicit.

> **日本語**
> cooperative groups は、協調する thread 集合と同期範囲を object として明示します。
>
> **学習メモ**
> group に参加する thread と collective を呼ぶ thread が一致するかを確認します。

## What To Look For

- `thread_block`、`grid_group`、`coalesced_group` の scope を読み分けます。
- grid-wide sync には cooperative launch の条件があります。
- warp aggregation や reduction の範囲を code 上で追います。

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
