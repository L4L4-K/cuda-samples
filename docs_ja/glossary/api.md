# CUDA API Cheat Sheet

English API | 日本語 | 学習メモ
--- | --- | ---
`cudaGetDeviceCount` | GPU 数確認 | skip 条件や multi-GPU sample の入口です。
`cudaSetDevice` | current device 選択 | allocation と launch の所属先を決めます。
`cudaMalloc` / `cudaFree` | device memory 確保/解放 | byte 数と lifetime を確認します。
`cudaMallocHost` / `cudaFreeHost` | pinned host memory | async copy と overlap の条件です。
`cudaMallocManaged` | managed memory 確保 | pointer は共有でも同期と migration は必要です。
`cudaMemcpy` | 同期 copy | direction enum で data flow を読みます。
`cudaMemcpyAsync` | stream へ非同期 copy 投入 | stream と pinned memory を確認します。
`cudaStreamCreate` / `cudaStreamDestroy` | stream resource | ordering lane の作成と cleanup を対応させます。
`cudaEventRecord` / `cudaEventElapsedTime` | event timing | stream 上の測定範囲です。
`cudaDeviceSynchronize` | device-wide host wait | correctness 待ちか measurement 待ちかを分けます。
`cuInit` / `cuCtxCreate` | Driver API setup | CU* handle lifetime を追います。
`cuModuleLoad` / `cuModuleGetFunction` | module/function lookup | PTX/cubin と kernel symbol を対応させます。
`cuLaunchKernel` | Driver API launch | argument 配列、grid/block、stream を確認します。
`nvrtcCompileProgram` | runtime compile | option、architecture、compile log を確認します。
`cublasCreate` / `cufftPlan*` / `cusolver*` | library resource/work | handle、plan、descriptor、workspace を追います。
`LaunchConfig` / `launch` | Python CUDA launch | Python object の lifetime と CUDA stream を確認します。

## API Reading Pattern

1. Identify which family the API belongs to: Runtime, Driver, NVRTC/JIT, library, Python CUDA, or external interop.
2. Decide whether the API creates a resource, moves data, launches work, synchronizes, validates, or destroys a resource.
3. Find the matching cleanup or synchronization boundary.
4. Check whether error handling is immediate or appears at a later synchronization point.

> **日本語**
> API 名は英語のまま読み、役割を分類します。`create`/`malloc` は ownership、`copy`/`map` は visibility、`launch`/library call は device work、`sync` は host wait、`destroy`/`free` は lifetime end です。
>
> **学習メモ**
> helper macro の中に API call が隠れている場合もあります。source annotation の `JP:` comment と対応させると見落としにくくなります。

## Common Mistakes

- `cudaGetLastError` を実行完了の同期だと思う。
- Runtime API の `cudaError_t` と Driver API の `CUresult` を混同する。
- library call が必ず同期的に終わると思い、validation 前の sync を忘れる。
- Python `launch` の戻りを GPU 完了と誤解する。

## Exercises

- 任意の sample で API を create/move/launch/sync/validate/cleanup に分類する。
- Driver API sample で CU* handle の lifetime を表にする。
- library sample で workspace query、allocation、library call、cleanup を追う。
