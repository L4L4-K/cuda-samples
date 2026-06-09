# Kernel Launch And Indexing

English anchor: kernels map launch geometry to array, image, matrix, or volume indices.

> **日本語**
> kernel の index 計算は、各 GPU thread がどの data element を担当するかを決めます。
>
> **学習メモ**
> `threadIdx.x` は block 内でしか一意ではありません。global index は `blockIdx` と組み合わせます。

## What To Look For

- 1D vector は `blockIdx.x * blockDim.x + threadIdx.x` が典型形です。
- 2D image/matrix は `x`、`y` と pitch/width の関係を確認します。
- grid-stride loop は 1 thread が複数 element を処理する形です。

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
