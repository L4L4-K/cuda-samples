# Memory Transfer Cheat Sheet

English pattern | 日本語 | 学習メモ
--- | --- | ---
Host to Device | CPU 入力を GPU へ渡す | kernel/library call 前の visibility を作ります。
Device to Host | GPU 結果を CPU へ戻す | validation や file output の前に出ます。
Device to Device | GPU 内 copy | staging、peer copy、multi-buffer で使います。
Pinned host buffer | page-locked CPU buffer | async transfer と overlap で重要です。
Mapped host buffer | GPU から見える host buffer | zero-copy は copy を省けますが latency に注意します。
CUDA array | texture/surface 向け allocation | image/texture sample で出ます。
Pitch | 行ごとの byte stride | width と byte stride を混同しないようにします。
Prefetch | managed memory を先に移動 | page fault を減らす performance 対策です。
Peer copy | GPU 間 copy | topology と peer access capability が必要です。
External memory | CUDA 以外の API が持つ resource | import/map/unmap と owner lifetime を読みます。

## Direction Checklist

- Source pointer と destination pointer が host/device/peer/external のどれかを確認します。
- Size が byte count か element count かを確認します。
- Async API なら stream と pinned memory 条件を確認します。
- Transfer 後に kernel/library が読むのか、validation が読むのかを確認します。
- Cleanup 前に未完了 work がないか確認します。

> **日本語**
> transfer は data を動かすだけでなく、「どの processor から最新値が見えるか」を変える境界です。direction、size、stream、sync を一緒に読みます。
>
> **学習メモ**
> Unified Memory sample では明示 copy が見えないことがあります。その場合も migration、prefetch、sync の位置を transfer 相当の境界として読みます。

## Common Mistakes

- `cudaMemcpyHostToDevice` と `cudaMemcpyDeviceToHost` を逆にする。
- `sizeof(T) * count` が必要なところで count だけ渡す。
- Async copy があるのに stream dependency がない。
- peer copy で source/destination device を取り違える。
- image pitch を element stride として扱う。

## Exercises

- vectorAdd の H2D/D2H copy を table にする。
- simpleStreams で copy と kernel が overlap する条件を書く。
- p2pBandwidthLatencyTest で peer copy の device pair を追う。
