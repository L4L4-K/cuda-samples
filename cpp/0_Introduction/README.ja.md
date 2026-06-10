# 0. Introduction - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample illustrates the usage of CUDA events for both GPU timing and overlapping CPU and GPU execution. Events are inserted into a stream of CUDA calls. Since CUDA stream calls are asynchronous, the CPU can perform computations while GPU is executing (including DMA memcopies between the host and device). CPU can query CUDA events to determine whether GPU has completed tasks.

This example shows how to use the clock function to measure the performance of block of threads of a kernel accurately.

Original README headings: `0. Introduction`, `[asyncAPI](./asyncAPI)`, `[clock](./clock)`, `[clock_nvrtc](./clock_nvrtc)`, `[cudaOpenMP](./cudaOpenMP)`, `[fp16ScalarProduct](./fp16ScalarProduct)`, `[matrixMul](./matrixMul)`, `[matrixMul_nvrtc](./matrixMul_nvrtc)`, `[matrixMulDrv](./matrixMulDrv)`, `[matrixMulDynlinkJIT](./matrixMulDynlinkJIT)`, `[mergeSort](./mergeSort)`, `[simpleAssert](./simpleAssert)`

> **日本語**
> `cpp/0_Introduction` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `0_Introduction` as a focused example of the CUDA concepts used in `cpp/0_Introduction`.
> **日本語**
> この sample の目的は、`0_Introduction` の小さな実装を通して Runtime, Driver, And NVRTC, CUDA Libraries, CUDA Graphs, Multi-GPU, P2P, And IPC, Tensor Cores And WMMA を具体的に追うことです。
>
> **学習メモ**
> 最初に `UnifiedMemoryStreams.cu, asyncAPI.cu, clock.cu, clock.cpp, clock_kernel.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `UnifiedMemoryStreams/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `UnifiedMemoryStreams/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `UnifiedMemoryStreams/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `UnifiedMemoryStreams/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `UnifiedMemoryStreams/UnifiedMemoryStreams.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `asyncAPI/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `asyncAPI/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `asyncAPI/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `asyncAPI/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `asyncAPI/asyncAPI.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `clock/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `clock/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `clock/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `clock/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `clock/clock.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `clock_nvrtc/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `clock_nvrtc/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `clock_nvrtc/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `clock_nvrtc/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `clock_nvrtc/clock.cpp`: Host-side setup, API calls, validation, and cleanup.
- `clock_nvrtc/clock_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cudaOpenMP/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cudaOpenMP/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cudaOpenMP/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cudaOpenMP/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cudaOpenMP/cudaOpenMP.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `fp16ScalarProduct/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `fp16ScalarProduct/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `fp16ScalarProduct/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `fp16ScalarProduct/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `fp16ScalarProduct/fp16ScalarProduct.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `matrixMul/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `matrixMul/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `matrixMul/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `matrixMul/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `matrixMul/matrixMul.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `matrixMulDrv/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `matrixMulDrv/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `matrixMulDrv/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `matrixMulDrv/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `matrixMulDrv/matrixMul.h`: Host/device declarations, helper types, constants, or library wrappers.
- `matrixMulDrv/matrixMulDrv.cpp`: Host-side setup, API calls, validation, and cleanup.
- `matrixMulDrv/matrixMul_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `matrixMulDynlinkJIT/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `matrixMulDynlinkJIT/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `matrixMulDynlinkJIT/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `matrixMulDynlinkJIT/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `matrixMulDynlinkJIT/cuda_drvapi_dynlink.c`: Host-side setup, API calls, validation, and cleanup.
- `matrixMulDynlinkJIT/cuda_drvapi_dynlink.h`: Host/device declarations, helper types, constants, or library wrappers.
- `matrixMulDynlinkJIT/cuda_drvapi_dynlink_cuda.h`: Host/device declarations, helper types, constants, or library wrappers.
- `matrixMulDynlinkJIT/extras/README.TXT`: Input, reference, generated-data description, or documentation used by the sample.
- `matrixMulDynlinkJIT/extras/matrixMul_kernel_32.ptx`: Supporting file used by `matrixMulDynlinkJIT/extras/matrixMul_kernel_32.ptx`.
- `matrixMulDynlinkJIT/extras/matrixMul_kernel_64.ptx`: Supporting file used by `matrixMulDynlinkJIT/extras/matrixMul_kernel_64.ptx`.
- `matrixMulDynlinkJIT/extras/ptx2c.py`: Python entry point or helper using CUDA Python, CuPy, framework interop, or subprocess logic.
- `matrixMulDynlinkJIT/helper_cuda_drvapi.h`: Host/device declarations, helper types, constants, or library wrappers.
- `matrixMulDynlinkJIT/matrixMul.h`: Host/device declarations, helper types, constants, or library wrappers.
- `matrixMulDynlinkJIT/matrixMulDynlinkJIT.cpp`: Host-side setup, API calls, validation, and cleanup.
- `matrixMulDynlinkJIT/matrixMul_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `matrixMulDynlinkJIT/matrixMul_kernel_32_ptxdump.c`: Host-side setup, API calls, validation, and cleanup.
- `matrixMulDynlinkJIT/matrixMul_kernel_32_ptxdump.h`: Host/device declarations, helper types, constants, or library wrappers.
- `matrixMulDynlinkJIT/matrixMul_kernel_64_ptxdump.c`: Host-side setup, API calls, validation, and cleanup.
- `matrixMulDynlinkJIT/matrixMul_kernel_64_ptxdump.h`: Host/device declarations, helper types, constants, or library wrappers.
- `matrixMul_nvrtc/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `matrixMul_nvrtc/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `matrixMul_nvrtc/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `matrixMul_nvrtc/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `matrixMul_nvrtc/matrixMul.cpp`: Host-side setup, API calls, validation, and cleanup.
- `matrixMul_nvrtc/matrixMul_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `mergeSort/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `mergeSort/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `mergeSort/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `mergeSort/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `mergeSort/bitonic.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `mergeSort/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `mergeSort/mergeSort.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `mergeSort/mergeSort_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `mergeSort/mergeSort_host.cpp`: Host-side setup, API calls, validation, and cleanup.
- `mergeSort/mergeSort_validate.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleAWBarrier/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleAWBarrier/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleAWBarrier/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleAWBarrier/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleAWBarrier/simpleAWBarrier.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleAssert/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleAssert/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleAssert/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleAssert/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleAssert/simpleAssert.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleAssert_nvrtc/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleAssert_nvrtc/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleAssert_nvrtc/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleAssert_nvrtc/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleAssert_nvrtc/simpleAssert.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleAssert_nvrtc/simpleAssert_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleAtomicIntrinsics/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleAtomicIntrinsics/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleAtomicIntrinsics/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleAtomicIntrinsics/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleAtomicIntrinsics/simpleAtomicIntrinsics.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleAtomicIntrinsics/simpleAtomicIntrinsics_cpu.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleAtomicIntrinsics/simpleAtomicIntrinsics_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `simpleAtomicIntrinsics_nvrtc/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleAtomicIntrinsics_nvrtc/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleAtomicIntrinsics_nvrtc/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleAtomicIntrinsics_nvrtc/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleAtomicIntrinsics_nvrtc/simpleAtomicIntrinsics.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleAtomicIntrinsics_nvrtc/simpleAtomicIntrinsics_cpu.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleAtomicIntrinsics_nvrtc/simpleAtomicIntrinsics_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `simpleAttributes/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleAttributes/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleAttributes/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleAttributes/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleAttributes/simpleAttributes.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleCUDA2GL/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUDA2GL/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUDA2GL/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleCUDA2GL/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleCUDA2GL/data/ref_simpleCUDA2GL.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleCUDA2GL/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleCUDA2GL/simpleCUDA2GL.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleCallback/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCallback/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCallback/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleCallback/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleCallback/multithreading.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleCallback/multithreading.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleCallback/simpleCallback.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleCooperativeGroups/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCooperativeGroups/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCooperativeGroups/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleCooperativeGroups/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleCooperativeGroups/simpleCooperativeGroups.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleCubemapTexture/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCubemapTexture/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCubemapTexture/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleCubemapTexture/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleCubemapTexture/simpleCubemapTexture.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleDrvRuntime/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleDrvRuntime/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleDrvRuntime/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleDrvRuntime/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleDrvRuntime/simpleDrvRuntime.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleDrvRuntime/vectorAdd_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleHyperQ/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleHyperQ/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleHyperQ/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleHyperQ/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleHyperQ/doc/HyperQ.docx`: Supporting file used by `simpleHyperQ/doc/HyperQ.docx`.
- `simpleHyperQ/doc/HyperQ.pdf`: Supporting file used by `simpleHyperQ/doc/HyperQ.pdf`.
- `simpleHyperQ/simpleHyperQ.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleIPC/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleIPC/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleIPC/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleIPC/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleIPC/simpleIPC.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleLayeredTexture/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleLayeredTexture/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleLayeredTexture/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleLayeredTexture/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleLayeredTexture/simpleLayeredTexture.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleMPI/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleMPI/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleMPI/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleMPI/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleMPI/simpleMPI.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleMPI/simpleMPI.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleMPI/simpleMPI.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleMultiCopy/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleMultiCopy/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleMultiCopy/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleMultiCopy/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleMultiCopy/simpleMultiCopy.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleMultiGPU/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleMultiGPU/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleMultiGPU/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleMultiGPU/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleMultiGPU/simpleMultiGPU.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleMultiGPU/simpleMultiGPU.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleOccupancy/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleOccupancy/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleOccupancy/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleOccupancy/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleOccupancy/simpleOccupancy.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleP2P/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleP2P/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleP2P/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleP2P/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleP2P/simpleP2P.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simplePitchLinearTexture/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simplePitchLinearTexture/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simplePitchLinearTexture/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simplePitchLinearTexture/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simplePitchLinearTexture/simplePitchLinearTexture.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simplePrintf/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simplePrintf/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simplePrintf/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simplePrintf/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simplePrintf/simplePrintf.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleStreams/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleStreams/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleStreams/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleStreams/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleStreams/simpleStreams.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleSurfaceWrite/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleSurfaceWrite/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleSurfaceWrite/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleSurfaceWrite/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleSurfaceWrite/data/ref_rotated.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleSurfaceWrite/data/teapot512.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleSurfaceWrite/simpleSurfaceWrite.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleTemplates/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleTemplates/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleTemplates/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleTemplates/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleTemplates/sharedmem.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `simpleTemplates/simpleTemplates.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleTexture/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleTexture/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleTexture/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleTexture/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleTexture/data/ref_rotated.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleTexture/data/teapot512.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleTexture/data/teapot512_out.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleTexture/simpleTexture.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleTexture3D/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleTexture3D/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleTexture3D/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleTexture3D/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleTexture3D/data/Bucky.raw`: Supporting file used by `simpleTexture3D/data/Bucky.raw`.
- `simpleTexture3D/data/ref_texture3D.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleTexture3D/doc/sshot_lg.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleTexture3D/doc/sshot_md.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleTexture3D/doc/sshot_sm.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleTexture3D/simpleTexture3D.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleTexture3D/simpleTexture3D_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleTextureDrv/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleTextureDrv/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleTextureDrv/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleTextureDrv/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleTextureDrv/data/ref_rotated.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleTextureDrv/data/teapot512.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleTextureDrv/data/teapot512_out.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleTextureDrv/simpleTextureDrv.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleTextureDrv/simpleTexture_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleVoteIntrinsics/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleVoteIntrinsics/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleVoteIntrinsics/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleVoteIntrinsics/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleVoteIntrinsics/simpleVoteIntrinsics.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleVoteIntrinsics/simpleVote_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `simpleZeroCopy/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleZeroCopy/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleZeroCopy/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleZeroCopy/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleZeroCopy/simpleZeroCopy.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `systemWideAtomics/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `systemWideAtomics/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `systemWideAtomics/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `systemWideAtomics/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `systemWideAtomics/systemWideAtomics.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `template/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `template/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `template/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `template/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `template/template.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `template/template_cpu.cpp`: Host-side setup, API calls, validation, and cleanup.
- `vectorAdd/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `vectorAdd/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `vectorAdd/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `vectorAdd/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `vectorAdd/vectorAdd.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `vectorAddDrv/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `vectorAddDrv/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `vectorAddDrv/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `vectorAddDrv/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `vectorAddDrv/vectorAddDrv.cpp`: Host-side setup, API calls, validation, and cleanup.
- `vectorAddDrv/vectorAdd_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `vectorAddMMAP/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `vectorAddMMAP/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `vectorAddMMAP/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `vectorAddMMAP/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `vectorAddMMAP/multidevicealloc_memmap.cpp`: Host-side setup, API calls, validation, and cleanup.
- `vectorAddMMAP/multidevicealloc_memmap.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `vectorAddMMAP/vectorAddMMAP.cpp`: Host-side setup, API calls, validation, and cleanup.
- `vectorAddMMAP/vectorAdd_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `vectorAdd_nvrtc/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `vectorAdd_nvrtc/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `vectorAdd_nvrtc/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `vectorAdd_nvrtc/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `vectorAdd_nvrtc/vectorAdd.cpp`: Host-side setup, API calls, validation, and cleanup.
- `vectorAdd_nvrtc/vectorAdd_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `UnifiedMemoryStreams.cu` first and locate the host-side setup or Python entry point.
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

- `UnifiedMemoryStreams/UnifiedMemoryStreams.cu`: focus on `cudaStreamAttachMemAsync`, `CUDA`, `cudaMallocManaged`, `cudaStream_t`, `cublasHandle_t`.
- `asyncAPI/asyncAPI.cu`: focus on `CUDA`, `launch`, `cudaEventDestroy`, `blockIdx`, `blockDim`.
- `clock/clock.cu`: focus on `CUDA`, `blockDim`, `cudaMalloc`, `cudaFree`, `__shared__`.
- `clock_nvrtc/clock.cpp`: focus on `cudaBlockSize`, `cudaGridSize`, `cuMemAlloc`, `cuMemFree`, `CUDA`.
- `clock_nvrtc/clock_kernel.cu`: focus on `blockDim`, `__shared__`, `threadIdx`, `__syncthreads`, `launch`.
- `cudaOpenMP/cudaOpenMP.cu`: focus on `CUDA`, `launch`, `blockIdx`, `blockDim`, `threadIdx`.
- `fp16ScalarProduct/fp16ScalarProduct.cu`: focus on `threadIdx`, `__syncthreads`, `blockDim`, `blockIdx`, `cudaMemcpy`.
- `matrixMul/matrixMul.cu`: focus on `CUDA`, `cudaMallocHost`, `cudaMalloc`, `cudaMemcpyAsync`, `cudaFreeHost`.
- `matrixMulDrv/matrixMul.h`: focus on control flow and helper functions.
- `matrixMulDrv/matrixMulDrv.cpp`: focus on `CUDA`, `CUdeviceptr`, `cuDevice`, `launch`, `CUfunction`.
- Additional source files: 78 more support files. Use the same setup/work/sync/cleanup lens.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/0_Introduction/CMakeLists.txt:1-48
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

add_subdirectory(UnifiedMemoryStreams)
add_subdirectory(asyncAPI)
add_subdirectory(clock)
add_subdirectory(clock_nvrtc)
add_subdirectory(cudaOpenMP)
add_subdirectory(fp16ScalarProduct)
add_subdirectory(matrixMul)
add_subdirectory(matrixMulDrv)
add_subdirectory(matrixMulDynlinkJIT)
add_subdirectory(matrixMul_nvrtc)
add_subdirectory(mergeSort)
add_subdirectory(simpleAWBarrier)
add_subdirectory(simpleAssert)
add_subdirectory(simpleAssert_nvrtc)
add_subdirectory(simpleAtomicIntrinsics)
add_subdirectory(simpleAtomicIntrinsics_nvrtc)
add_subdirectory(simpleAttributes)
add_subdirectory(simpleCUDA2GL)
add_subdirectory(simpleCallback)
add_subdirectory(simpleCooperativeGroups)
add_subdirectory(simpleCubemapTexture)
add_subdirectory(simpleDrvRuntime)
add_subdirectory(simpleHyperQ)
add_subdirectory(simpleIPC)
add_subdirectory(simpleLayeredTexture)
add_subdirectory(simpleMPI)
add_subdirectory(simpleMultiCopy)
add_subdirectory(simpleMultiGPU)
add_subdirectory(simpleOccupancy)
add_subdirectory(simpleP2P)
add_subdirectory(simplePitchLinearTexture)
add_subdirectory(simplePrintf)
add_subdirectory(simpleStreams)
add_subdirectory(simpleSurfaceWrite)
add_subdirectory(simpleTemplates)
add_subdirectory(simpleTexture)
add_subdirectory(simpleTexture3D)
add_subdirectory(simpleTextureDrv)
add_subdirectory(simpleVoteIntrinsics)
add_subdirectory(simpleZeroCopy)
add_subdirectory(template)
add_subdirectory(systemWideAtomics)
add_subdirectory(vectorAdd)
add_subdirectory(vectorAddDrv)
add_subdirectory(vectorAddMMAP)
add_subdirectory(vectorAdd_nvrtc)
```

> JP: この抜粋は `cpp/0_Introduction/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `CUresult` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `CUDAAPI` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `CUdeviceptr` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `__shared__` | block 内共有 memory または同期境界です。producer/consumer の順序を確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMemcpyHostToDevice` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cuDevice` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

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
cmake --build build --target 0_Introduction
ctest --test-dir build -R 0_Introduction
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
