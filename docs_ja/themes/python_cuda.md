# Python CUDA

English anchor: Python samples use CUDA through cuda-python, CuPy, frameworks, and JIT helpers.

> **日本語**
> Python CUDA でも host/device memory、転送、kernel/library work、同期、validation の考え方は同じです。
>
> **学習メモ**
> 短い Python code の中でも、内部で device allocation、JIT、implicit synchronization が起きます。

## What To Look For

- NumPy array と GPU array/tensor の所有者を区別します。
- `.get()`、`asnumpy`、framework tensor copy は host-device transfer です。
- 初回 JIT と通常実行の timing を分けて測ります。

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
