# 4. CUDA Libraries - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

A CUDA Sample that demonstrates how using batched CUBLAS API calls to improve overall performance.

A NPP CUDA Sample that demonstrates how to use NPP FilterBox function to perform a Box Filter.

Original README headings: `4. CUDA Libraries`, `[batchCUBLAS](./batchCUBLAS)`, `[boxFilterNPP](./boxFilterNPP)`, `[cannyEdgeDetectorNPP](./cannyEdgeDetectorNPP)`, `[conjugateGradient](./conjugateGradient)`, `[conjugateGradientCudaGraphs](./conjugateGradientCudaGraphs)`, `[conjugateGradientMultiBlockCG](./conjugateGradientMultiBlockCG)`, `[conjugateGradientMultiDeviceCG](./conjugateGradientMultiDeviceCG)`, `[conjugateGradientPrecond](./conjugateGradientPrecond)`, `[conjugateGradientUM](./conjugateGradientUM)`, `[cudaNvSci](./cudaNvSci)`, `[cuSolverDn_LinearSolver](./cuSolverDn_LinearSolver)`

> **日本語**
> `cpp/4_CUDA_Libraries` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `4_CUDA_Libraries` as a focused example of the CUDA concepts used in `cpp/4_CUDA_Libraries`.
> **日本語**
> この sample の目的は、`4_CUDA_Libraries` の小さな実装を通して Runtime, Driver, And NVRTC, CUDA Libraries, CUDA Graphs, Multi-GPU, P2P, And IPC, Cooperative Groups を具体的に追うことです。
>
> **学習メモ**
> 最初に `FilterBorderControlNPP.cpp, MersenneTwister.cpp, batchCUBLAS.cpp, batchCUBLAS.h, boxFilterNPP.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `FilterBorderControlNPP/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `FilterBorderControlNPP/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `FilterBorderControlNPP/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `FilterBorderControlNPP/FilterBorderControlNPP.cpp`: Host-side setup, API calls, validation, and cleanup.
- `FilterBorderControlNPP/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `FilterBorderControlNPP/data/teapot512.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `MersenneTwisterGP11213/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `MersenneTwisterGP11213/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `MersenneTwisterGP11213/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `MersenneTwisterGP11213/MersenneTwister.cpp`: Host-side setup, API calls, validation, and cleanup.
- `MersenneTwisterGP11213/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `batchCUBLAS/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `batchCUBLAS/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `batchCUBLAS/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `batchCUBLAS/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `batchCUBLAS/batchCUBLAS.cpp`: Host-side setup, API calls, validation, and cleanup.
- `batchCUBLAS/batchCUBLAS.h`: Host/device declarations, helper types, constants, or library wrappers.
- `boxFilterNPP/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `boxFilterNPP/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `boxFilterNPP/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `boxFilterNPP/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `boxFilterNPP/boxFilterNPP.cpp`: Host-side setup, API calls, validation, and cleanup.
- `boxFilterNPP/teapot512.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `cannyEdgeDetectorNPP/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cannyEdgeDetectorNPP/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cannyEdgeDetectorNPP/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cannyEdgeDetectorNPP/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cannyEdgeDetectorNPP/cannyEdgeDetectorNPP.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cannyEdgeDetectorNPP/teapot512.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `conjugateGradient/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `conjugateGradient/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `conjugateGradient/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `conjugateGradient/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `conjugateGradient/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `conjugateGradientCudaGraphs/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `conjugateGradientCudaGraphs/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `conjugateGradientCudaGraphs/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `conjugateGradientCudaGraphs/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `conjugateGradientMultiBlockCG/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `conjugateGradientMultiBlockCG/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `conjugateGradientMultiBlockCG/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `conjugateGradientMultiBlockCG/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `conjugateGradientMultiBlockCG/conjugateGradientMultiBlockCG.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `conjugateGradientMultiDeviceCG/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `conjugateGradientMultiDeviceCG/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `conjugateGradientMultiDeviceCG/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `conjugateGradientMultiDeviceCG/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `conjugateGradientMultiDeviceCG/conjugateGradientMultiDeviceCG.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `conjugateGradientPrecond/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `conjugateGradientPrecond/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `conjugateGradientPrecond/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `conjugateGradientPrecond/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `conjugateGradientPrecond/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `conjugateGradientUM/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `conjugateGradientUM/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `conjugateGradientUM/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `conjugateGradientUM/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `conjugateGradientUM/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cuSolverDn_LinearSolver/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuSolverDn_LinearSolver/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuSolverDn_LinearSolver/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cuSolverDn_LinearSolver/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cuSolverDn_LinearSolver/cuSolverDn_LinearSolver.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cuSolverDn_LinearSolver/gr_900_900_crg.mtx`: Supporting file used by `cuSolverDn_LinearSolver/gr_900_900_crg.mtx`.
- `cuSolverDn_LinearSolver/lap3D_7pt_n20.mtx`: Supporting file used by `cuSolverDn_LinearSolver/lap3D_7pt_n20.mtx`.
- `cuSolverDn_LinearSolver/mmio.c`: Host-side setup, API calls, validation, and cleanup.
- `cuSolverDn_LinearSolver/mmio.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cuSolverDn_LinearSolver/mmio_wrapper.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cuSolverRf/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuSolverRf/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuSolverRf/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cuSolverRf/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cuSolverRf/cuSolverRf.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cuSolverRf/lap2D_5pt_n100.mtx`: Supporting file used by `cuSolverRf/lap2D_5pt_n100.mtx`.
- `cuSolverRf/lap3D_7pt_n20.mtx`: Supporting file used by `cuSolverRf/lap3D_7pt_n20.mtx`.
- `cuSolverRf/mmio.c`: Host-side setup, API calls, validation, and cleanup.
- `cuSolverRf/mmio.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cuSolverRf/mmio_wrapper.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cuSolverSp_LinearSolver/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuSolverSp_LinearSolver/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuSolverSp_LinearSolver/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cuSolverSp_LinearSolver/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cuSolverSp_LinearSolver/cuSolverSp_LinearSolver.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cuSolverSp_LinearSolver/lap2D_5pt_n100.mtx`: Supporting file used by `cuSolverSp_LinearSolver/lap2D_5pt_n100.mtx`.
- `cuSolverSp_LinearSolver/lap3D_7pt_n20.mtx`: Supporting file used by `cuSolverSp_LinearSolver/lap3D_7pt_n20.mtx`.
- `cuSolverSp_LinearSolver/mmio.c`: Host-side setup, API calls, validation, and cleanup.
- `cuSolverSp_LinearSolver/mmio.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cuSolverSp_LinearSolver/mmio_wrapper.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cuSolverSp_LowlevelCholesky/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuSolverSp_LowlevelCholesky/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuSolverSp_LowlevelCholesky/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cuSolverSp_LowlevelCholesky/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cuSolverSp_LowlevelCholesky/cuSolverSp_LowlevelCholesky.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cuSolverSp_LowlevelCholesky/lap2D_5pt_n100.mtx`: Supporting file used by `cuSolverSp_LowlevelCholesky/lap2D_5pt_n100.mtx`.
- `cuSolverSp_LowlevelCholesky/lap3D_7pt_n20.mtx`: Supporting file used by `cuSolverSp_LowlevelCholesky/lap3D_7pt_n20.mtx`.
- `cuSolverSp_LowlevelCholesky/mmio.c`: Host-side setup, API calls, validation, and cleanup.
- `cuSolverSp_LowlevelCholesky/mmio.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cuSolverSp_LowlevelCholesky/mmio_wrapper.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cuSolverSp_LowlevelQR/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuSolverSp_LowlevelQR/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuSolverSp_LowlevelQR/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cuSolverSp_LowlevelQR/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cuSolverSp_LowlevelQR/cuSolverSp_LowlevelQR.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cuSolverSp_LowlevelQR/lap2D_5pt_n100.mtx`: Supporting file used by `cuSolverSp_LowlevelQR/lap2D_5pt_n100.mtx`.
- `cuSolverSp_LowlevelQR/lap2D_5pt_n32.mtx`: Supporting file used by `cuSolverSp_LowlevelQR/lap2D_5pt_n32.mtx`.
- `cuSolverSp_LowlevelQR/lap3D_7pt_n20.mtx`: Supporting file used by `cuSolverSp_LowlevelQR/lap3D_7pt_n20.mtx`.
- `cuSolverSp_LowlevelQR/mmio.c`: Host-side setup, API calls, validation, and cleanup.
- `cuSolverSp_LowlevelQR/mmio.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cuSolverSp_LowlevelQR/mmio_wrapper.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cubDeviceFind/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cubDeviceFind/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cubDeviceFind/cubDeviceFind.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cubDeviceSegmentedScan/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cubDeviceSegmentedScan/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cubDeviceSegmentedScan/cubDeviceSegmentedScan.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cubDeviceTransform/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cubDeviceTransform/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cubDeviceTransform/cubDeviceTransform.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cudaNvSci/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cudaNvSci/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cudaNvSci/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cudaNvSci/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cudaNvSci/cudaNvSci.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaNvSci/cudaNvSci.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cudaNvSci/imageKernels.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cudaNvSci/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaNvSci/teapot1024.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `freeImageInteropNPP/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `freeImageInteropNPP/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `freeImageInteropNPP/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `freeImageInteropNPP/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `freeImageInteropNPP/freeImageInteropNPP.cpp`: Host-side setup, API calls, validation, and cleanup.
- `histEqualizationNPP/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `histEqualizationNPP/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `histEqualizationNPP/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `histEqualizationNPP/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `histEqualizationNPP/histEqualizationNPP.cpp`: Host-side setup, API calls, validation, and cleanup.
- `jitLto/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `jitLto/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `jitLto/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `jitLto/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `jitLto/jitLto.cpp`: Host-side setup, API calls, validation, and cleanup.
- `libcuxxMdspan/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `libcuxxMdspan/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `libcuxxMdspan/libcuxxMdspan.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `libcuxxRandom/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `libcuxxRandom/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `libcuxxRandom/libcuxxRandom.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `lineOfSight/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `lineOfSight/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `lineOfSight/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `lineOfSight/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `lineOfSight/lineOfSight.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `matrixMulCUBLAS/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `matrixMulCUBLAS/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `matrixMulCUBLAS/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `matrixMulCUBLAS/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `matrixMulCUBLAS/matrixMulCUBLAS.cpp`: Host-side setup, API calls, validation, and cleanup.
- `nvJPEG/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `nvJPEG/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `nvJPEG/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `nvJPEG/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `nvJPEG/images/img1.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG/images/img2.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG/images/img3.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG/images/img4.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG/images/img5.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG/images/img6.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG/images/img7.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG/images/img8.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG/nvJPEG.cpp`: Host-side setup, API calls, validation, and cleanup.
- `nvJPEG_encoder/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `nvJPEG_encoder/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `nvJPEG_encoder/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `nvJPEG_encoder/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `nvJPEG_encoder/encode_output/img1.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/encode_output/img2.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/encode_output/img3.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/encode_output/img4.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/encode_output/img5.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/encode_output/img6.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/encode_output/img7.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/encode_output/img8.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/images/img1.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/images/img2.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/images/img3.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/images/img4.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/images/img5.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/images/img6.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/images/img7.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/images/img8.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder/nvJPEG_encoder.cpp`: Host-side setup, API calls, validation, and cleanup.
- `oceanFFT/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `oceanFFT/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `oceanFFT/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `oceanFFT/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `oceanFFT/data/ocean.frag`: Supporting file used by `oceanFFT/data/ocean.frag`.
- `oceanFFT/data/ocean.vert`: Supporting file used by `oceanFFT/data/ocean.vert`.
- `oceanFFT/data/ref_slopeShading.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `oceanFFT/data/ref_spatialDomain.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `oceanFFT/data/reference.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `oceanFFT/doc/sshot_lg.png`: Input, reference, generated-data description, or documentation used by the sample.
- `oceanFFT/doc/sshot_md.png`: Input, reference, generated-data description, or documentation used by the sample.
- `oceanFFT/doc/sshot_sm.png`: Input, reference, generated-data description, or documentation used by the sample.
- `oceanFFT/oceanFFT.cpp`: Host-side setup, API calls, validation, and cleanup.
- `oceanFFT/oceanFFT_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `randomFog/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `randomFog/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `randomFog/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `randomFog/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `randomFog/data/ref_randomFog.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `randomFog/randomFog.cpp`: Host-side setup, API calls, validation, and cleanup.
- `randomFog/rng.cpp`: Host-side setup, API calls, validation, and cleanup.
- `randomFog/rng.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleCUBLAS/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUBLAS/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUBLAS/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleCUBLAS/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleCUBLAS/simpleCUBLAS.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleCUBLASXT/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUBLASXT/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUBLASXT/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleCUBLASXT/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleCUBLASXT/simpleCUBLASXT.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleCUBLAS_LU/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUBLAS_LU/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUBLAS_LU/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleCUBLAS_LU/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleCUBLAS_LU/simpleCUBLAS_LU.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleCUFFT/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUFFT/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUFFT/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleCUFFT/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleCUFFT/simpleCUFFT.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleCUFFT_2d_MGPU/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUFFT_2d_MGPU/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUFFT_2d_MGPU/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleCUFFT_2d_MGPU/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleCUFFT_2d_MGPU/simpleCUFFT_2d_MGPU.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleCUFFT_MGPU/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUFFT_MGPU/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUFFT_MGPU/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleCUFFT_MGPU/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleCUFFT_MGPU/simpleCUFFT_MGPU.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleCUFFT_callback/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUFFT_callback/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleCUFFT_callback/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleCUFFT_callback/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleCUFFT_callback/simpleCUFFT_callback.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `watershedSegmentationNPP/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `watershedSegmentationNPP/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `watershedSegmentationNPP/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `watershedSegmentationNPP/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `watershedSegmentationNPP/watershedSegmentationNPP.cpp`: Host-side setup, API calls, validation, and cleanup.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `FilterBorderControlNPP.cpp` first and locate the host-side setup or Python entry point.
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

- `FilterBorderControlNPP/FilterBorderControlNPP.cpp`: focus on `nppStreamCtx`, `npp`, `nppiFree`, `cudaError`, `CUDA`.
- `MersenneTwisterGP11213/MersenneTwister.cpp`: focus on `curandGenerator_t`, `curandGenerateUniform`, `cudaStreamSynchronize`, `curand`, `cudaStream_t`.
- `batchCUBLAS/batchCUBLAS.cpp`: focus on `cudaSuccess`, `cuGet`, `cudaError_t`, `cublasOperation_t`, `CUBLASTEST_FAILED`.
- `batchCUBLAS/batchCUBLAS.h`: focus on `cuGet`, `cuEqual`, `CUDA`, `cuRand`, `CUDA_ZNEW`.
- `boxFilterNPP/boxFilterNPP.cpp`: focus on `nppStreamCtx`, `cudaError`, `npp`, `CUDA`, `cudaSuccess`.
- `cannyEdgeDetectorNPP/cannyEdgeDetectorNPP.cpp`: focus on `nppStreamCtx`, `cudaError`, `npp`, `CUDA`, `cudaSuccess`.
- `conjugateGradient/main.cpp`: focus on `cublasHandle`, `cublasStatus`, `cudaMalloc`, `cudaFree`, `CUDA_R_32F`.
- `conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu`: focus on `cublasHandle`, `cudaMalloc`, `cusparseHandle`, `cudaMemcpyAsync`, `CUDA_R_32F`.
- `conjugateGradientMultiBlockCG/conjugateGradientMultiBlockCG.cu`: focus on `cudaFree`, `cudaMallocManaged`, `CUDA`, `__shared__`, `threadIdx`.
- `conjugateGradientMultiDeviceCG/conjugateGradientMultiDeviceCG.cu`: focus on `cudaMemAdvise`, `cudaMallocManaged`, `cudaFree`, `cudaSetDevice`, `cudaMemPrefetchAsync`.
- Additional source files: 51 more support files. Use the same setup/work/sync/cleanup lens.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cuGet` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `nppStreamCtx` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMemcpyHostToDevice` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cublasHandle` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaError` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuComplex` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuDoubleComplex` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaMemcpyDeviceToHost` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cusolverSpH` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cusparseHandle` | Driver API の handle 境界です。context/module/function と error code を追います。 |

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
cmake --build build --target 4_CUDA_Libraries
ctest --test-dir build -R 4_CUDA_Libraries
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

- `cuGet` の直前と直後で、どの memory/resource が有効になったかをメモする。
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
- [Cooperative Groups](../../docs_ja/themes/cooperative_groups.md): block/grid/warp 単位の協調と同期を読むための基礎です。
- [Shared Memory](../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Streams And Events](../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Unified Memory](../../docs_ja/themes/unified_memory.md): managed memory、migration、prefetch の意味を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
