# CUDA Terms Glossary

English term | 日本語 | 学習メモ
--- | --- | ---
Host | CPU 側 | CUDA API 呼び出し、入力準備、結果検証を行う側です。
Device | GPU 側 | kernel や CUDA library work が実行される側です。
Kernel | GPU で実行される関数 | `<<<...>>>` や Driver/Python launch API で起動されます。
Grid | block の集合 | 1 回の kernel launch で投入される全 work です。
Block | thread の集合 | shared memory と `__syncthreads()` の基本範囲です。
Thread | 最小実行単位 | `threadIdx` などで担当 data を決めます。
Warp | 通常 32 thread の実行単位 | warp-level API、分岐、coalescing で重要です。
Pinned Memory | page-locked host memory | async transfer と overlap の条件になります。
Managed Memory | Unified Memory allocation | 同じ pointer を CPU/GPU から使えますが同期は必要です。
Stream | work ordering queue | 同じ stream 内は順序化され、別 stream は重なる可能性があります。
Event | stream 上の marker | timing と待ち合わせに使います。
Descriptor | data shape/layout object | library API に matrix/image/tensor などの形を伝えます。
Workspace | 一時作業 memory | library algorithm や変換処理の scratch 領域です。

> **日本語**
> 用語は英語名のまま覚え、日本語で役割を補うと code と documentation を照合しやすくなります。
>
> **学習メモ**
> 「どの単語が scope を表すか」を意識すると、同期や memory ownership の読み間違いが減ります。
