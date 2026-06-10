# 3. CUDA Features - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

A CUDA sample demonstrating __nv_bfloat16 (e8m7) GEMM computation using the Warp Matrix Multiply and Accumulate (WMMA) API introduced with CUDA 11 in Ampere chip family tensor cores for faster matrix operations. This sample also uses async copy provided by cuda pipeline interface for gmem to shmem async loads which improves kernel performance and reduces register presssure.

This sample is a simple code that illustrates binary partition cooperative groups and reduce within the thread block.

Original README headings: `3. CUDA Features`, `[bf16TensorCoreGemm](./bf16TensorCoreGemm)`, `[binaryPartitionCG](./binaryPartitionCG)`, `[bindlessTexture](./bindlessTexture)`, `[cdpAdvancedQuicksort](./cdpAdvancedQuicksort)`, `[cdpBezierTessellation](./cdpBezierTessellation)`, `[cdpQuadtree](./cdpQuadtree)`, `[cdpSimplePrint](./cdpSimplePrint)`, `[cdpSimpleQuicksort](./cdpSimpleQuicksort)`, `[cudaCompressibleMemory](./cudaCompressibleMemory)`, `[cudaTensorCoreGemm](./cudaTensorCoreGemm)`, `[dmmaTensorCoreGemm](./dmmaTensorCoreGemm)`

> **日本語**
> `cpp/3_CUDA_Features` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `3_CUDA_Features` as a focused example of the CUDA concepts used in `cpp/3_CUDA_Features`.
> **日本語**
> この sample の目的は、`3_CUDA_Features` の小さな実装を通して Runtime, Driver, And NVRTC, CUDA Libraries, CUDA Graphs, Multi-GPU, P2P, And IPC, Tensor Cores And WMMA を具体的に追うことです。
>
> **学習メモ**
> 最初に `StreamPriorities.cu, bf16TensorCoreGemm.cu, binaryPartitionCG.cu, bindlessTexture.cpp, bindlessTexture.h` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- CUDA library components used by this sample, such as cuBLAS, cuFFT, cuSolver, NPP, CUB, or nvJPEG
- The device topology required by the README, such as multiple GPUs, peer access, IPC, MPI, or process support

> **日本語**
> 必要条件は英語 README と CMake/requirements を優先します。この guide は条件を置き換えず、読むべき確認点を追加します。
>
> **学習メモ**
> 実行できない場合は、source を変える前に driver、toolkit、GPU feature、library、platform guard、Python package version を確認します。

## Files

- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `StreamPriorities/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `StreamPriorities/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `StreamPriorities/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `StreamPriorities/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `StreamPriorities/StreamPriorities.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `bf16TensorCoreGemm/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `bf16TensorCoreGemm/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `bf16TensorCoreGemm/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `bf16TensorCoreGemm/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `bf16TensorCoreGemm/bf16TensorCoreGemm.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `binaryPartitionCG/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `binaryPartitionCG/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `binaryPartitionCG/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `binaryPartitionCG/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `binaryPartitionCG/binaryPartitionCG.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `bindlessTexture/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `bindlessTexture/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `bindlessTexture/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `bindlessTexture/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `bindlessTexture/bindlessTexture.cpp`: Host-side setup, API calls, validation, and cleanup.
- `bindlessTexture/bindlessTexture.h`: Host/device declarations, helper types, constants, or library wrappers.
- `bindlessTexture/bindlessTexture_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `bindlessTexture/data/flower.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `bindlessTexture/data/person.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `bindlessTexture/data/ref_bindlessTexture.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `bindlessTexture/data/sponge.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `bindlessTexture/doc/sshot_lg.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `bindlessTexture/doc/sshot_md.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `bindlessTexture/doc/sshot_sm.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `cdpAdvancedQuicksort/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cdpAdvancedQuicksort/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cdpAdvancedQuicksort/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cdpAdvancedQuicksort/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cdpAdvancedQuicksort/cdpAdvancedQuicksort.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cdpAdvancedQuicksort/cdpBitonicSort.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cdpAdvancedQuicksort/cdpQuicksort.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cdpBezierTessellation/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cdpBezierTessellation/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cdpBezierTessellation/BezierLineCDP.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cdpBezierTessellation/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cdpBezierTessellation/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cdpQuadtree/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cdpQuadtree/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cdpQuadtree/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cdpQuadtree/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cdpQuadtree/cdpQuadtree.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cdpSimplePrint/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cdpSimplePrint/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cdpSimplePrint/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cdpSimplePrint/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cdpSimplePrint/cdpSimplePrint.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cdpSimpleQuicksort/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cdpSimpleQuicksort/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cdpSimpleQuicksort/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cdpSimpleQuicksort/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cdpSimpleQuicksort/cdpSimpleQuicksort.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cudaCompressibleMemory/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cudaCompressibleMemory/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cudaCompressibleMemory/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cudaCompressibleMemory/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cudaCompressibleMemory/compMalloc.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaCompressibleMemory/compMalloc.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cudaCompressibleMemory/saxpy.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cudaTensorCoreGemm/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cudaTensorCoreGemm/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cudaTensorCoreGemm/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cudaTensorCoreGemm/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cudaTensorCoreGemm/cudaTensorCoreGemm.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `dmmaTensorCoreGemm/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `dmmaTensorCoreGemm/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `dmmaTensorCoreGemm/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `dmmaTensorCoreGemm/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `dmmaTensorCoreGemm/dmmaTensorCoreGemm.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `globalToShmemAsyncCopy/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `globalToShmemAsyncCopy/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `globalToShmemAsyncCopy/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `globalToShmemAsyncCopy/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `graphConditionalNodes/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `graphConditionalNodes/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `graphConditionalNodes/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `graphConditionalNodes/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `graphConditionalNodes/graphConditionalNodes.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `graphMemoryFootprint/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `graphMemoryFootprint/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `graphMemoryFootprint/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `graphMemoryFootprint/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `graphMemoryFootprint/graphMemoryFootprint.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `graphMemoryNodes/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `graphMemoryNodes/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `graphMemoryNodes/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `graphMemoryNodes/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `graphMemoryNodes/graphMemoryNodes.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `immaTensorCoreGemm/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `immaTensorCoreGemm/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `immaTensorCoreGemm/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `immaTensorCoreGemm/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `immaTensorCoreGemm/immaTensorCoreGemm.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `jacobiCudaGraphs/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `jacobiCudaGraphs/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `jacobiCudaGraphs/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `jacobiCudaGraphs/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `jacobiCudaGraphs/jacobi.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `jacobiCudaGraphs/jacobi.h`: Host/device declarations, helper types, constants, or library wrappers.
- `jacobiCudaGraphs/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `memMapIPCDrv/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `memMapIPCDrv/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `memMapIPCDrv/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `memMapIPCDrv/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `memMapIPCDrv/memMapIpc.cpp`: Host-side setup, API calls, validation, and cleanup.
- `memMapIPCDrv/memMapIpc_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `newdelete/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `newdelete/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `newdelete/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `newdelete/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `newdelete/container.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `newdelete/newdelete.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `ptxjit/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `ptxjit/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `ptxjit/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `ptxjit/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `ptxjit/ptxjit.cpp`: Host-side setup, API calls, validation, and cleanup.
- `ptxjit/ptxjit_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleCudaGraphs/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCudaGraphs/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCudaGraphs/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleCudaGraphs/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleCudaGraphs/simpleCudaGraphs.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `tf32TensorCoreGemm/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `tf32TensorCoreGemm/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `tf32TensorCoreGemm/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `tf32TensorCoreGemm/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `tf32TensorCoreGemm/tf32TensorCoreGemm.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `warpAggregatedAtomicsCG/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `warpAggregatedAtomicsCG/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `warpAggregatedAtomicsCG/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `warpAggregatedAtomicsCG/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `warpAggregatedAtomicsCG/warpAggregatedAtomicsCG.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `StreamPriorities.cu` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Create library handles, descriptors, plans, or workspaces before the library call.
- Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.
- Capture or build CUDA Graph nodes, instantiate the graph, then launch the executable graph.
- Enumerate devices, enable peer or IPC access, and record which device/process owns each resource.
- Inside the kernel, map thread/block indexes to tile elements and check the barrier around shared memory reuse.
- Move, map, or expose input data so GPU work can read the intended values.
- Launch the kernel, graph, library call, or Python CUDA operation with the documented configuration.
- Synchronize only at the required correctness or timing boundary.
- Validate results against the CPU/reference path, generated artifact, or expected status message.
- Release CUDA, library, framework, graphics, or external resources in the reverse ownership order.

> **日本語**
> 実行の流れは setup、visibility、GPU work、sync、validation、cleanup の順に読みます。非同期 API がある場合は、host がいつ待つかを別に記録します。
>
> **学習メモ**
> CUDA の bug は kernel 本体だけでなく、copy direction、descriptor、stream dependency、cleanup order にも出ます。

## Concrete Reading Path

- `StreamPriorities/StreamPriorities.cu`: focus on `cudaMalloc`, `cudaMemcpy`, `cudaEvent_t`, `cudaEventCreate`, `cudaEventRecord`.
- `bf16TensorCoreGemm/bf16TensorCoreGemm.cu`: focus on `__syncthreads`, `blockDim`, `threadIdx`, `gridDim`, `cudaMemcpy`.
- `binaryPartitionCG/binaryPartitionCG.cu`: focus on `cudaMallocHost`, `cudaMalloc`, `cudaMemcpyAsync`, `cudaFreeHost`, `atomicAdd`.
- `bindlessTexture/bindlessTexture.cpp`: focus on `CUDA`, `cudaTextureObjects`, `cudaExtent`, `cudaGraphicsResource`, `cudaMalloc`.
- `bindlessTexture/bindlessTexture.h`: focus on `CUDA`, `cudaExtent`, `cudaResourceType`, `cudaArray_t`, `cudaMipmappedArray_t`.
- `bindlessTexture/bindlessTexture_kernel.cu`: focus on `cudaAddressModeClamp`, `cudaTextureObject_t`, `cudaResourceDesc`, `cudaTextureDesc`, `blockIdx`.
- `cdpAdvancedQuicksort/cdpAdvancedQuicksort.cu`: focus on `atomicData`, `launch`, `atomicDataStack`, `atomicAdd`, `atomic`.
- `cdpAdvancedQuicksort/cdpBitonicSort.cu`: focus on `threadIdx`, `blockDim`, `__shared__`, `launch`.
- `cdpAdvancedQuicksort/cdpQuicksort.h`: focus on `atomic`, `launch`, `atomicData`.
- `cdpBezierTessellation/BezierLineCDP.cu`: focus on `threadIdx`, `blockDim`, `blockIdx`, `cudaMalloc`, `cudaFree`.
- Additional source files: 25 more support files. Use the same setup/work/sync/cleanup lens.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `__syncthreads` | block 内共有 memory または同期境界です。producer/consumer の順序を確認します。 |
| `gridDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaMemcpyHostToDevice` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `__shared__` | block 内共有 memory または同期境界です。producer/consumer の順序を確認します。 |
| `cudaMemcpyDeviceToHost` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaStream_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaGraphNode_t` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- Cooperative Groups は協調する単位を明示します。grid/block/warp のどれを同期しているかを確認します。
- CUDA Graph は一連の work を node と依存関係として再利用します。capture 対象と buffer lifetime を確認します。
- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- library sample では handle、descriptor、plan、workspace が GPU work の外側の resource です。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- multi-GPU sample では、device 選択、peer capability、context/IPC handle の寿命を分けて読みます。
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
- Runtime/Driver/NVRTC sample では、compile/load した module と launch する kernel 名の対応が重要です。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。
- 同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。
- Tensor Core sample では tile size、alignment、precision、accumulator の型が正しさと性能を決めます。
- Unified Memory は pointer を共有しますが、migration、prefetch、同期の理解は必要です。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Sample-Specific Notes

- library sample では、CUDA kernel を直接書かなくても library call が device work を投入します。handle/descriptor/workspace の lifetime を kernel launch と同じ厳しさで追います。

> **日本語**
> この section は同じ template ではなく、sample 名、path、検出した API から読みどころを絞っています。
>
> **学習メモ**
> 似た名前の sample は Runtime 版、Driver 版、NVRTC 版、library 版の違いを比較すると学習効果が高くなります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target 3_CUDA_Features
ctest --test-dir build -R 3_CUDA_Features
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

The sample prints timing, bandwidth, latency, throughput, or comparison data; exact values depend on hardware and driver.
> **日本語**
> 期待結果は英語の出力文字列、README の validation、生成 file、または reference result と照合します。
>
> **学習メモ**
> `PASS`、`Test passed`、error code、timing label などの出力文字列は翻訳せず、source と同じ表記で確認します。

## Common Mistakes

- API 名や target 名を翻訳してしまい、README や build command と対応できなくなる。
- allocation size を byte で渡す API と element count で考える loop を混同する。
- kernel launch が非同期であることを忘れ、同期前の結果を host 側で読んでしまう。
- shared memory を書いた thread と読む thread の間に必要な barrier を見落とす。
- different stream 間に依存があるのに event や explicit sync を置かない。
- graph capture 後に buffer lifetime や node dependency が変わったことを見落とす。
- leading dimension、stride、descriptor、workspace size を host 配列の見た目だけで判断する。
- JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。
- peer access が有効な device pair と、単に複数 GPU が存在することを混同する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `threadIdx` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- shared memory tile の producer、consumer、barrier を図にする。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- graph node の依存関係を箇条書きにし、どの buffer lifetime が graph 実行全体をまたぐか確認する。
- handle/descriptor/workspace の作成、利用、破棄を対応表にする。
- device ごとの ownership と、peer/IPC で共有される resource を分けて書く。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Runtime, Driver, And NVRTC](../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [CUDA Libraries](../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。
- [CUDA Graphs](../../docs_ja/themes/graphs.md): capture、node dependency、replay、graph update を読むための基礎です。
- [Multi-GPU, P2P, And IPC](../../docs_ja/themes/multi_gpu_p2p_ipc.md): device topology、peer access、IPC handle、multi-process 境界を読むための基礎です。
- [Tensor Cores And WMMA](../../docs_ja/themes/tensor_cores_wmma.md): Tensor Core、tile、precision、fragment の制約を読むための基礎です。
- [Cooperative Groups](../../docs_ja/themes/cooperative_groups.md): block/grid/warp 単位の協調と同期を読むための基礎です。
- [Shared Memory](../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Streams And Events](../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
