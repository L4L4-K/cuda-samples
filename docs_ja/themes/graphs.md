# CUDA Graphs

English anchor: graphs capture or build dependency workflows and replay them efficiently.

> **日本語**
> CUDA Graph は kernel、copy、memset、依存関係を graph としてまとめ、繰り返し実行の overhead を減らします。
>
> **学習メモ**
> graph は定義、instantiate、launch の段階を分けて読みます。

## What To Look For

- stream capture は capture 可能な API に制限があります。
- `cudaGraph_t` と `cudaGraphExec_t` の lifetime を追います。
- parameter update と graph topology change は別の操作です。

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
