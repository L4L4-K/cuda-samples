# Debugging, Profiling, And Testing

English anchor: samples combine API checks, reference validation, and timing/profiling hooks.

> **日本語**
> debug では API error、kernel launch error、非同期 error、結果 validation を分けて確認します。
>
> **学習メモ**
> kernel error は launch 直後ではなく同期時に表面化することがあります。

## What To Look For

- helper macro がどの error を捕まえるか読みます。
- reference CPU path や expected image/data と比較します。
- skipped check は成功ではないため、理由を final report に残します。

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
