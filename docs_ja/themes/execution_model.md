# CUDA Execution Model

English anchor: host code launches grids of blocks and threads on GPU SMs.

> **日本語**
> CPU 側の host code が GPU 側の kernel を起動し、grid、block、thread の階層で device work が進みます。
>
> **学習メモ**
> `<<<grid, block>>>` は普通の関数呼び出しではなく、GPU の実行キューに仕事を投入する境界です。

## What To Look For

- `dim3 grid` と `dim3 block` が並列度を決めます。
- `blockIdx`、`blockDim`、`threadIdx` が thread と data の対応を決めます。
- 範囲外 access を避けるため、丸めた launch には bounds check が必要です。

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
