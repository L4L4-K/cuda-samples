# Memory Transfer Cheat Sheet

English pattern | 日本語 | 学習メモ
--- | --- | ---
Host to Device | CPU 入力を GPU へ渡す | kernel/library call 前の準備です。
Device to Host | GPU 結果を CPU へ戻す | validation や file output の前に出ます。
Device to Device | GPU 内 copy | staging、peer copy、multi-buffer で使います。
Pinned host buffer | page-locked CPU buffer | async transfer と overlap で重要です。
Mapped host buffer | GPU から見える host buffer | zero-copy は transfer を省けますが latency に注意します。
CUDA array | texture/surface 向け allocation | image/texture sample で出ます。
Pitch | 行ごとの byte stride | width と byte stride を混同しないようにします。
Prefetch | managed memory を先に移動 | page fault を減らす performance 対策です。
Peer copy | GPU 間 transfer | P2P support と current device を確認します。
External map | 他 API memory を CUDA に見せる | owner、map/unmap、fence の順序を読みます。

> **日本語**
> transfer direction は sample の data flow そのものです。入力、計算、結果確認の境界として読みます。
>
> **学習メモ**
> byte 数、element 数、pitch、stride の取り違えは典型的な bug です。
