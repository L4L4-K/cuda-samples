# 2. Concepts and Techniques - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

Fast image box filter using CUDA with OpenGL rendering.

This sample implements a separable convolution filter of a 2D signal with a gaussian kernel.

Original README headings: `2. Concepts and Techniques`, `[boxFilter](./boxFilter)`, `[convolutionSeparable](./convolutionSeparable)`, `[convolutionTexture](./convolutionTexture)`, `[dct8x8](./dct8x8)`, `[EGLStream_CUDA_CrossGPU](./EGLStream_CUDA_CrossGPU)`, `[EGLStream_CUDA_Interop](./EGLStream_CUDA_Interop)`, `[eigenvalues](./eigenvalues)`, `[FunctionPointers](./FunctionPointers)`, `[histogram](./histogram)`, `[imageDenoising](./imageDenoising)`, `[inlinePTX](./inlinePTX)`

> **日本語**
> `cpp/2_Concepts_and_Techniques` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `2_Concepts_and_Techniques` as a focused example of the CUDA concepts used in `cpp/2_Concepts_and_Techniques`.
> **日本語**
> この sample の目的は、`2_Concepts_and_Techniques` の小さな実装を通して Runtime, Driver, And NVRTC, CUDA Libraries, CUDA Graphs, Multi-GPU, P2P, And IPC, Cooperative Groups を具体的に追うことです。
>
> **学習メモ**
> 最初に `cuda_consumer.cpp, cuda_consumer.h, cuda_producer.cpp, cuda_producer.h, eglstrm_common.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `EGLStream_CUDA_CrossGPU/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `EGLStream_CUDA_CrossGPU/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `EGLStream_CUDA_CrossGPU/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `EGLStream_CUDA_CrossGPU/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `EGLStream_CUDA_CrossGPU/cuda_consumer.cpp`: Host-side setup, API calls, validation, and cleanup.
- `EGLStream_CUDA_CrossGPU/cuda_consumer.h`: Host/device declarations, helper types, constants, or library wrappers.
- `EGLStream_CUDA_CrossGPU/cuda_producer.cpp`: Host-side setup, API calls, validation, and cleanup.
- `EGLStream_CUDA_CrossGPU/cuda_producer.h`: Host/device declarations, helper types, constants, or library wrappers.
- `EGLStream_CUDA_CrossGPU/eglstrm_common.cpp`: Host-side setup, API calls, validation, and cleanup.
- `EGLStream_CUDA_CrossGPU/eglstrm_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `EGLStream_CUDA_CrossGPU/helper.h`: Host/device declarations, helper types, constants, or library wrappers.
- `EGLStream_CUDA_CrossGPU/kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `EGLStream_CUDA_CrossGPU/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `EGLStream_CUDA_Interop/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `EGLStream_CUDA_Interop/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `EGLStream_CUDA_Interop/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `EGLStream_CUDA_Interop/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `EGLStream_CUDA_Interop/cuda_consumer.cpp`: Host-side setup, API calls, validation, and cleanup.
- `EGLStream_CUDA_Interop/cuda_consumer.h`: Host/device declarations, helper types, constants, or library wrappers.
- `EGLStream_CUDA_Interop/cuda_f_1.yuv`: Supporting file used by `EGLStream_CUDA_Interop/cuda_f_1.yuv`.
- `EGLStream_CUDA_Interop/cuda_f_2.yuv`: Supporting file used by `EGLStream_CUDA_Interop/cuda_f_2.yuv`.
- `EGLStream_CUDA_Interop/cuda_producer.cpp`: Host-side setup, API calls, validation, and cleanup.
- `EGLStream_CUDA_Interop/cuda_producer.h`: Host/device declarations, helper types, constants, or library wrappers.
- `EGLStream_CUDA_Interop/cuda_yuv_f_1.yuv`: Supporting file used by `EGLStream_CUDA_Interop/cuda_yuv_f_1.yuv`.
- `EGLStream_CUDA_Interop/cuda_yuv_f_2.yuv`: Supporting file used by `EGLStream_CUDA_Interop/cuda_yuv_f_2.yuv`.
- `EGLStream_CUDA_Interop/eglstrm_common.cpp`: Host-side setup, API calls, validation, and cleanup.
- `EGLStream_CUDA_Interop/eglstrm_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `EGLStream_CUDA_Interop/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `FunctionPointers/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `FunctionPointers/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `FunctionPointers/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `FunctionPointers/FunctionPointers.cpp`: Host-side setup, API calls, validation, and cleanup.
- `FunctionPointers/FunctionPointers_kernels.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `FunctionPointers/FunctionPointers_kernels.h`: Host/device declarations, helper types, constants, or library wrappers.
- `FunctionPointers/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `FunctionPointers/data/ref_orig.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `FunctionPointers/data/ref_shared.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `FunctionPointers/data/ref_tex.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `FunctionPointers/data/teapot512.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `MC_EstimatePiInlineP/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `MC_EstimatePiInlineP/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `MC_EstimatePiInlineP/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `MC_EstimatePiInlineP/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `MC_EstimatePiInlineP/inc/cudasharedmem.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_EstimatePiInlineP/inc/piestimator.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_EstimatePiInlineP/inc/test.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_EstimatePiInlineP/src/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `MC_EstimatePiInlineP/src/piestimator.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `MC_EstimatePiInlineP/src/test.cpp`: Host-side setup, API calls, validation, and cleanup.
- `MC_EstimatePiInlineQ/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `MC_EstimatePiInlineQ/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `MC_EstimatePiInlineQ/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `MC_EstimatePiInlineQ/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `MC_EstimatePiInlineQ/inc/cudasharedmem.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_EstimatePiInlineQ/inc/piestimator.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_EstimatePiInlineQ/inc/test.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_EstimatePiInlineQ/src/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `MC_EstimatePiInlineQ/src/piestimator.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `MC_EstimatePiInlineQ/src/test.cpp`: Host-side setup, API calls, validation, and cleanup.
- `MC_EstimatePiP/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `MC_EstimatePiP/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `MC_EstimatePiP/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `MC_EstimatePiP/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `MC_EstimatePiP/inc/cudasharedmem.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_EstimatePiP/inc/piestimator.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_EstimatePiP/inc/test.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_EstimatePiP/src/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `MC_EstimatePiP/src/piestimator.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `MC_EstimatePiP/src/test.cpp`: Host-side setup, API calls, validation, and cleanup.
- `MC_EstimatePiQ/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `MC_EstimatePiQ/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `MC_EstimatePiQ/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `MC_EstimatePiQ/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `MC_EstimatePiQ/inc/cudasharedmem.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_EstimatePiQ/inc/piestimator.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_EstimatePiQ/inc/test.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_EstimatePiQ/src/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `MC_EstimatePiQ/src/piestimator.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `MC_EstimatePiQ/src/test.cpp`: Host-side setup, API calls, validation, and cleanup.
- `MC_SingleAsianOptionP/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `MC_SingleAsianOptionP/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `MC_SingleAsianOptionP/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `MC_SingleAsianOptionP/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `MC_SingleAsianOptionP/inc/asianoption.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_SingleAsianOptionP/inc/cudasharedmem.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_SingleAsianOptionP/inc/pricingengine.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_SingleAsianOptionP/inc/test.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MC_SingleAsianOptionP/src/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `MC_SingleAsianOptionP/src/pricingengine.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `MC_SingleAsianOptionP/src/test.cpp`: Host-side setup, API calls, validation, and cleanup.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `boxFilter/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `boxFilter/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `boxFilter/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `boxFilter/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `boxFilter/boxFilter.cpp`: Host-side setup, API calls, validation, and cleanup.
- `boxFilter/boxFilter_cpu.cpp`: Host-side setup, API calls, validation, and cleanup.
- `boxFilter/boxFilter_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `boxFilter/data/ref_14.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `boxFilter/data/ref_22.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `boxFilter/data/teapot1024.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `convolutionSeparable/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `convolutionSeparable/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `convolutionSeparable/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `convolutionSeparable/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `convolutionSeparable/convolutionSeparable.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `convolutionSeparable/convolutionSeparable_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `convolutionSeparable/convolutionSeparable_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `convolutionSeparable/doc/convolutionSeparable.doc`: Supporting file used by `convolutionSeparable/doc/convolutionSeparable.doc`.
- `convolutionSeparable/doc/convolutionSeparable.pdf`: Supporting file used by `convolutionSeparable/doc/convolutionSeparable.pdf`.
- `convolutionSeparable/doc/convolutionSeparable.vsd`: Supporting file used by `convolutionSeparable/doc/convolutionSeparable.vsd`.
- `convolutionSeparable/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `convolutionTexture/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `convolutionTexture/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `convolutionTexture/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `convolutionTexture/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `convolutionTexture/convolutionTexture.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `convolutionTexture/convolutionTexture_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `convolutionTexture/convolutionTexture_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `convolutionTexture/doc/Performance.xls`: Supporting file used by `convolutionTexture/doc/Performance.xls`.
- `convolutionTexture/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `dct8x8/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `dct8x8/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `dct8x8/BmpUtil.cpp`: Host-side setup, API calls, validation, and cleanup.
- `dct8x8/BmpUtil.h`: Host/device declarations, helper types, constants, or library wrappers.
- `dct8x8/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `dct8x8/Common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `dct8x8/DCT8x8_Gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `dct8x8/DCT8x8_Gold.h`: Host/device declarations, helper types, constants, or library wrappers.
- `dct8x8/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `dct8x8/data/teapot512.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8/data/teapot512.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8/dct8x8.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `dct8x8/dct8x8_kernel1.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `dct8x8/dct8x8_kernel2.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `dct8x8/dct8x8_kernel_quantization.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `dct8x8/dct8x8_kernel_short.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `dct8x8/doc/BarbaraBlocks1.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8/doc/BarbaraBlocks2.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8/doc/BarbaraBlocks3.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8/doc/CosineBasis.png`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8/doc/Cosines.xls`: Supporting file used by `dct8x8/doc/Cosines.xls`.
- `dct8x8/doc/DctJpeg.png`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8/doc/barbara.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8/doc/barbara_lg.png`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8/doc/barbara_md.png`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8/doc/barbara_sm.png`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8/doc/dct8x8.doc`: Supporting file used by `dct8x8/doc/dct8x8.doc`.
- `dct8x8/doc/dct8x8.pdf`: Supporting file used by `dct8x8/doc/dct8x8.pdf`.
- `dct8x8/teapot512_cuda1.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8/teapot512_cuda2.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8/teapot512_cuda_short.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8/teapot512_gold1.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `dct8x8/teapot512_gold2.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `eigenvalues/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `eigenvalues/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `eigenvalues/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `eigenvalues/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `eigenvalues/bisect_kernel_large.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `eigenvalues/bisect_kernel_large_multi.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `eigenvalues/bisect_kernel_large_onei.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `eigenvalues/bisect_kernel_small.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `eigenvalues/bisect_large.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `eigenvalues/bisect_large.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `eigenvalues/bisect_small.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `eigenvalues/bisect_small.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `eigenvalues/bisect_util.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `eigenvalues/config.h`: Host/device declarations, helper types, constants, or library wrappers.
- `eigenvalues/data/diagonal.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `eigenvalues/data/reference.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `eigenvalues/data/superdiagonal.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `eigenvalues/doc/eigenvalues.doc`: Supporting file used by `eigenvalues/doc/eigenvalues.doc`.
- `eigenvalues/doc/eigenvalues.pdf`: Supporting file used by `eigenvalues/doc/eigenvalues.pdf`.
- `eigenvalues/gerschgorin.cpp`: Host-side setup, API calls, validation, and cleanup.
- `eigenvalues/gerschgorin.h`: Host/device declarations, helper types, constants, or library wrappers.
- `eigenvalues/main.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `eigenvalues/matlab.cpp`: Host-side setup, API calls, validation, and cleanup.
- `eigenvalues/matlab.h`: Host/device declarations, helper types, constants, or library wrappers.
- `eigenvalues/structs.h`: Host/device declarations, helper types, constants, or library wrappers.
- `eigenvalues/util.h`: Host/device declarations, helper types, constants, or library wrappers.
- `histogram/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `histogram/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `histogram/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `histogram/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `histogram/doc/histogram.doc`: Supporting file used by `histogram/doc/histogram.doc`.
- `histogram/doc/histogram.pdf`: Supporting file used by `histogram/doc/histogram.pdf`.
- `histogram/doc/histogram.vsd`: Supporting file used by `histogram/doc/histogram.vsd`.
- `histogram/histogram256.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `histogram/histogram64.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `histogram/histogram_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `histogram/histogram_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `histogram/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `imageDenoising/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `imageDenoising/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `imageDenoising/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `imageDenoising/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `imageDenoising/bmploader.cpp`: Host-side setup, API calls, validation, and cleanup.
- `imageDenoising/data/portrait_noise.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `imageDenoising/data/ref_knn.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `imageDenoising/data/ref_nlm.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `imageDenoising/data/ref_nlm2.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `imageDenoising/data/ref_passthru.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `imageDenoising/doc/NLM_lg.png`: Input, reference, generated-data description, or documentation used by the sample.
- `imageDenoising/doc/NLM_md.png`: Input, reference, generated-data description, or documentation used by the sample.
- `imageDenoising/doc/NLM_sm.png`: Input, reference, generated-data description, or documentation used by the sample.
- `imageDenoising/doc/imageDenoising.doc`: Supporting file used by `imageDenoising/doc/imageDenoising.doc`.
- `imageDenoising/doc/imageDenoising.pdf`: Supporting file used by `imageDenoising/doc/imageDenoising.pdf`.
- `imageDenoising/imageDenoising.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `imageDenoising/imageDenoising.h`: Host/device declarations, helper types, constants, or library wrappers.
- `imageDenoising/imageDenoisingGL.cpp`: Host-side setup, API calls, validation, and cleanup.
- `imageDenoising/imageDenoising_copy_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `imageDenoising/imageDenoising_knn_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `imageDenoising/imageDenoising_nlm2_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `imageDenoising/imageDenoising_nlm_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `inlinePTX/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `inlinePTX/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `inlinePTX/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `inlinePTX/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `inlinePTX/inlinePTX.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `inlinePTX_nvrtc/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `inlinePTX_nvrtc/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `inlinePTX_nvrtc/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `inlinePTX_nvrtc/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `inlinePTX_nvrtc/inlinePTX.cpp`: Host-side setup, API calls, validation, and cleanup.
- `inlinePTX_nvrtc/inlinePTX_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `interval/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `interval/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `interval/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `interval/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `interval/boost/config.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/abi/borland_prefix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/abi/borland_suffix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/abi/msvc_prefix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/abi/msvc_suffix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/abi_prefix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/abi_suffix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/auto_link.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/borland.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/codegear.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/comeau.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/common_edg.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/compaq_cxx.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/digitalmars.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/gcc.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/gcc_xml.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/greenhills.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/hp_acc.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/intel.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/kai.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/metrowerks.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/mpw.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/pgi.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/sgi_mipspro.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/sunpro_cc.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/vacpp.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/compiler/visualc.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/no_tr1/cmath.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/no_tr1/complex.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/no_tr1/functional.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/no_tr1/memory.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/no_tr1/utility.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/platform/aix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/platform/amigaos.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/platform/beos.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/platform/bsd.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/platform/cygwin.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/platform/hpux.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/platform/irix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/platform/linux.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/platform/macos.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/platform/qnxnto.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/platform/solaris.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/platform/vxworks.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/platform/win32.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/posix_features.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/requires_threads.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/select_compiler_config.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/select_platform_config.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/select_stdlib_config.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/stdlib/dinkumware.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/stdlib/libcomo.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/stdlib/libstdcpp3.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/stdlib/modena.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/stdlib/msl.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/stdlib/roguewave.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/stdlib/sgi.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/stdlib/stlport.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/stdlib/vacpp.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/suffix.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/user.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/config/warning_disable.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/limits.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/arith.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/arith2.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/arith3.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/checking.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/compare.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/compare/certain.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/compare/explicit.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/compare/lexicographic.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/compare/possible.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/compare/set.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/compare/tribool.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/constants.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/detail/alpha_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/detail/bcc_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/detail/bugs.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/detail/c99_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/detail/c99sub_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/detail/division.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/detail/ia64_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/detail/interval_prototype.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/detail/msvc_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/detail/ppc_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/detail/sparc_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/detail/test_input.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/detail/x86_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/detail/x86gcc_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/ext/integer.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/ext/x86_fast_rounding_control.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/hw_rounding.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/interval.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/io.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/limits.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/policies.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/rounded_arith.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/rounded_transc.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/rounding.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/transc.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/boost/numeric/interval/utility.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/cpu_interval.h`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/cuda_interval.h`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/cuda_interval_lib.h`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/cuda_interval_rounded_arith.h`: Host/device declarations, helper types, constants, or library wrappers.
- `interval/interval.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `interval/interval.h`: Host/device declarations, helper types, constants, or library wrappers.
- `particles/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `particles/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `particles/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `particles/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `particles/data/ref_particles.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `particles/doc/particles.doc`: Supporting file used by `particles/doc/particles.doc`.
- `particles/doc/particles.pdf`: Supporting file used by `particles/doc/particles.pdf`.
- `particles/doc/screenshot_lg.png`: Input, reference, generated-data description, or documentation used by the sample.
- `particles/doc/screenshot_md.png`: Input, reference, generated-data description, or documentation used by the sample.
- `particles/doc/screenshot_sm.png`: Input, reference, generated-data description, or documentation used by the sample.
- `particles/particleSystem.cpp`: Host-side setup, API calls, validation, and cleanup.
- `particles/particleSystem.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `particles/particleSystem.h`: Host/device declarations, helper types, constants, or library wrappers.
- `particles/particleSystem_cuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `particles/particles.cpp`: Host-side setup, API calls, validation, and cleanup.
- `particles/particles_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `particles/particles_kernel_impl.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `particles/render_particles.cpp`: Host-side setup, API calls, validation, and cleanup.
- `particles/render_particles.h`: Host/device declarations, helper types, constants, or library wrappers.
- `particles/shaders.cpp`: Host-side setup, API calls, validation, and cleanup.
- `particles/shaders.h`: Host/device declarations, helper types, constants, or library wrappers.
- `radixSortThrust/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `radixSortThrust/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `radixSortThrust/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `radixSortThrust/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `radixSortThrust/doc/readme.txt`: Input, reference, generated-data description, or documentation used by the sample.
- `radixSortThrust/radixSortThrust.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `reduction/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `reduction/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `reduction/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `reduction/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `reduction/reduction.cpp`: Host-side setup, API calls, validation, and cleanup.
- `reduction/reduction.h`: Host/device declarations, helper types, constants, or library wrappers.
- `reduction/reduction_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `reductionMultiBlockCG/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `reductionMultiBlockCG/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `reductionMultiBlockCG/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `reductionMultiBlockCG/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `reductionMultiBlockCG/reductionMultiBlockCG.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `scalarProd/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `scalarProd/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `scalarProd/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `scalarProd/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `scalarProd/scalarProd.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `scalarProd/scalarProd_cpu.cpp`: Host-side setup, API calls, validation, and cleanup.
- `scalarProd/scalarProd_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `scan/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `scan/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `scan/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `scan/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `scan/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `scan/scan.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `scan/scan_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `scan/scan_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `segmentationTreeThrust/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `segmentationTreeThrust/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `segmentationTreeThrust/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `segmentationTreeThrust/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `segmentationTreeThrust/common.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `segmentationTreeThrust/data/ref_00.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `segmentationTreeThrust/data/ref_09.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `segmentationTreeThrust/data/test.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `segmentationTreeThrust/kernels.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `segmentationTreeThrust/segmentationTree.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `shfl_scan/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `shfl_scan/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `shfl_scan/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `shfl_scan/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `shfl_scan/shfl_integral_image.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `shfl_scan/shfl_scan.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `shfl_scan/util.h`: Host/device declarations, helper types, constants, or library wrappers.
- `sortingNetworks/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `sortingNetworks/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `sortingNetworks/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `sortingNetworks/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `sortingNetworks/bitonicSort.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `sortingNetworks/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `sortingNetworks/oddEvenMergeSort.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `sortingNetworks/sortingNetworks_common.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `sortingNetworks/sortingNetworks_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `sortingNetworks/sortingNetworks_validate.cpp`: Host-side setup, API calls, validation, and cleanup.
- `streamOrderedAllocation/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `streamOrderedAllocation/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `streamOrderedAllocation/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `streamOrderedAllocation/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `streamOrderedAllocation/streamOrderedAllocation.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `streamOrderedAllocationIPC/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `streamOrderedAllocationIPC/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `streamOrderedAllocationIPC/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `streamOrderedAllocationIPC/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `streamOrderedAllocationIPC/streamOrderedAllocationIPC.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `streamOrderedAllocationP2P/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `streamOrderedAllocationP2P/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `streamOrderedAllocationP2P/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `streamOrderedAllocationP2P/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `streamOrderedAllocationP2P/streamOrderedAllocationP2P.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `threadFenceReduction/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `threadFenceReduction/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `threadFenceReduction/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `threadFenceReduction/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `threadFenceReduction/threadFenceReduction.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `threadFenceReduction/threadFenceReduction.h`: Host/device declarations, helper types, constants, or library wrappers.
- `threadFenceReduction/threadFenceReduction_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `threadMigration/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `threadMigration/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `threadMigration/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `threadMigration/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `threadMigration/threadMigration.cpp`: Host-side setup, API calls, validation, and cleanup.
- `threadMigration/threadMigration_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `cuda_consumer.cpp` first and locate the host-side setup or Python entry point.
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

- `EGLStream_CUDA_CrossGPU/cuda_consumer.cpp`: focus on `cudaConsumer`, `cuStatus`, `CUDA_SUCCESS`, `CUresult`, `CUDA`.
- `EGLStream_CUDA_CrossGPU/cuda_consumer.h`: focus on `CUresult`, `cudaConsumer`, `CUstream`, `cudaError_t`, `CUDA`.
- `EGLStream_CUDA_CrossGPU/cuda_producer.cpp`: focus on `cudaProducer`, `cudaEgl`, `CUDA_SUCCESS`, `CUresult`, `cudaPtr`.
- `EGLStream_CUDA_CrossGPU/cuda_producer.h`: focus on `CUresult`, `CUdeviceptr`, `cudaProducer`, `CUeglFrame`, `cudaEgl`.
- `EGLStream_CUDA_CrossGPU/eglstrm_common.cpp`: focus on `CUDA`, `cudaDevIndexProd`, `Device`, `cudaDevIndexCons`.
- `EGLStream_CUDA_CrossGPU/eglstrm_common.h`: focus on `cudaEGL`, `cudaDevIndexCons`, `cudaDevIndexProd`.
- `EGLStream_CUDA_CrossGPU/helper.h`: focus on `cuInit`, `CUDA_SUCCESS`, `CUresult`, `cuDeviceGetCount`, `cuDeviceGetAttribute`.
- `EGLStream_CUDA_CrossGPU/kernel.cu`: focus on `blockDim`, `blockIdx`, `threadIdx`, `cudaSuccess`, `cudaError_t`.
- `EGLStream_CUDA_CrossGPU/main.cpp`: focus on `cudaConsumer`, `cudaProducer`, `CUDA_SUCCESS`, `CUDA_ERROR_UNKNOWN`, `cudaEgl1`.
- `EGLStream_CUDA_Interop/cuda_consumer.cpp`: focus on `cuStatus`, `cudaEgl`, `CUDA_SUCCESS`, `cudaConsumer`, `CUDA`.
- Additional source files: 245 more support files. Use the same setup/work/sync/cleanup lens.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaResult` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaProducer` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `__shared__` | block 内共有 memory または同期境界です。producer/consumer の順序を確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `CUDA_SUCCESS` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaConsumer` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMemcpyHostToDevice` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
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
cmake --build build --target 2_Concepts_and_Techniques
ctest --test-dir build -R 2_Concepts_and_Techniques
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
- [Cooperative Groups](../../docs_ja/themes/cooperative_groups.md): block/grid/warp 単位の協調と同期を読むための基礎です。
- [Shared Memory](../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Streams And Events](../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
