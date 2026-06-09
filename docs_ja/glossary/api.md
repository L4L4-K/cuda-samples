# CUDA API Cheat Sheet

English API | 日本語 | 学習メモ
--- | --- | ---
`cudaGetDeviceCount` | GPU 数確認 | skip 条件や multi-GPU sample の入口です。
`cudaSetDevice` | 現在 device 選択 | allocation と launch の所属先を決めます。
`cudaMalloc` / `cudaFree` | device memory 確保/解放 | byte 数と lifetime を確認します。
`cudaMallocHost` | pinned host memory 確保 | async copy の性能条件です。
`cudaMallocManaged` | managed memory 確保 | pointer は共有でも同期は必要です。
`cudaMemcpy` | 同期 copy | direction enum で data flow を読みます。
`cudaMemcpyAsync` | stream へ非同期 copy 投入 | stream と pinned memory を確認します。
`cudaStreamCreate` | stream 作成 | work ordering lane を作ります。
`cudaEventRecord` | event 記録 | timing と dependency の基点です。
`cudaDeviceSynchronize` | device 全体待ち | host validation 前によく出ます。
`cuModuleLoad` | Driver module 読み込み | PTX/CUBIN/NVRTC output を実行可能にします。
`cuLaunchKernel` | Driver API launch | Runtime の `<<<...>>>` に対応する低レベル起動です。
`nvrtcCompileProgram` | runtime compile | compile option と log を確認します。

> **日本語**
> API は「確保」「転送」「起動」「同期」「破棄」「library 実行」に分類して読みます。
>
> **学習メモ**
> helper macro に包まれていても、実際に失敗する API と error check の位置を追います。
