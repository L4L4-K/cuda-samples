# Japanese CUDA Theme Guides

English anchor: this directory groups Japanese learning guides by CUDA concept.

> **日本語**
> sample を個別に読む前に、概念ごとの mental model、API、よくある間違い、演習を確認するための入口です。
>
> **学習メモ**
> 迷ったら `memory`、`execution_model`、`debugging_profiling_testing` から読み、sample の `Related Themes` で戻ってきます。

## Guides

- [CUDA Execution Model](execution_model.md): CUDA samples launch work as grids of thread blocks, where each thread runs the same kernel code on different data.
- [CUDA Memory](memory.md): CUDA samples move data among host memory, device memory, pinned memory, managed memory, arrays, mapped memory, and external resources.
- [Kernel Launch And Indexing](kernel_indexing.md): CUDA kernels map grid, block, and thread coordinates to data indexes.
- [CUDA Streams And Events](streams_events.md): Streams order asynchronous CUDA work, and events mark points in stream timelines.
- [Synchronization And Atomics](sync_atomics.md): Synchronization creates ordering, while atomics protect shared updates to one memory location.
- [CUDA Shared Memory](shared_memory.md): Shared memory is a fast per-block scratchpad shared by threads in the same block.
- [CUDA Unified Memory](unified_memory.md): Unified Memory lets CPU and GPU use one managed pointer while the driver migrates pages.
- [CUDA Cooperative Groups](cooperative_groups.md): Cooperative Groups makes the group of threads participating in collective operations explicit.
- [CUDA Graphs](graphs.md): CUDA Graphs record a dependency graph of GPU work and replay it efficiently.
- [CUDA Runtime, Driver, And NVRTC](runtime_driver_nvrtc.md): Samples use Runtime API, Driver API, NVRTC, JIT linking, PTX, and NVVM at different abstraction levels.
- [CUDA Libraries](libraries.md): CUDA libraries submit optimized device work through handles, descriptors, plans, and workspaces.
- [Tensor Cores And WMMA](tensor_cores_wmma.md): Tensor Core samples use specialized matrix instructions, tile APIs, and precision formats for high-throughput math.
- [Multi-GPU, P2P, And IPC](multi_gpu_p2p_ipc.md): Multi-GPU samples coordinate memory, contexts, peer access, processes, and interconnect topology across devices.
- [CUDA Performance](performance.md): Performance samples measure memory traffic, launch overhead, occupancy, overlap, bandwidth, latency, and math throughput.
- [Debugging, Profiling, And Testing](debugging_profiling_testing.md): Samples use error checks, reference validation, profiling ranges, assertions, printf, and device capability checks.
- [Python CUDA](python_cuda.md): Python CUDA samples use CUDA Python, CuPy, framework interop, DLPack, streams, memory resources, and distributed tools.

## Suggested Paths

- First CUDA kernel: `execution_model` -> `kernel_indexing` -> `memory` -> `debugging_profiling_testing`.
- Memory and overlap: `memory` -> `streams_events` -> `performance` -> `unified_memory`.
- Advanced launch/runtime: `runtime_driver_nvrtc` -> `graphs` -> `cooperative_groups`.
- Libraries and math: `libraries` -> `tensor_cores_wmma` -> `performance`.
- Python workflow: `python_cuda` -> `memory` -> `streams_events` -> `debugging_profiling_testing`.
