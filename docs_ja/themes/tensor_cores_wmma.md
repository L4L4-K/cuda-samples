# Tensor Cores And WMMA

English anchor: Tensor Core paths use specialized matrix instructions or library kernels.

> **日本語**
> Tensor Core は行列演算などを高速化する専用実行 path で、WMMA/MMA または library 経由で使います。
>
> **学習メモ**
> data type、tile shape、alignment、architecture guard が正しさと性能を左右します。

## What To Look For

- fragment、matrix_a、matrix_b、accumulator の役割を確認します。
- TF32、FP16、BF16、INT8、FP64 は対応 GPU と精度が違います。
- custom kernel と cuBLAS path の比較では測定範囲を合わせます。

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
