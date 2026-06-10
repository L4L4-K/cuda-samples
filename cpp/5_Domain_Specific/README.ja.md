# 5. Domain Specific - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates how to efficiently implement a Bicubic B-spline interpolation filter with CUDA texture.

Bilateral filter is an edge-preserving non-linear smoothing filter that is implemented with CUDA with OpenGL rendering. It can be used in image recovery and denoising. Each pixel is weight by considering both the spatial distance and color distance between its neighbors. Reference:"C. Tomasi, R. Manduchi, Bilateral Filtering for Gray and Color Images, proceeding of the ICCV, 1998, http://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf"

Original README headings: `5. Domain Specific`, `[bicubicTexture](./bicubicTexture)`, `[bilateralFilter](./bilateralFilter)`, `[binomialOptions](./binomialOptions)`, `[binomialOptions_nvrtc](./binomialOptions_nvrtc)`, `[BlackScholes](./BlackScholes)`, `[BlackScholes_nvrtc](./BlackScholes_nvrtc)`, `[convolutionFFT2D](./convolutionFFT2D)`, `[dwtHaar1D](./dwtHaar1D)`, `[dxtc](./dxtc)`, `[fastWalshTransform](./fastWalshTransform)`, `[FDTD3d](./FDTD3d)`

> **日本語**
> `cpp/5_Domain_Specific` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `5_Domain_Specific` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific`.
> **日本語**
> この sample の目的は、`5_Domain_Specific` の小さな実装を通して Runtime, Driver, And NVRTC, CUDA Libraries, CUDA Graphs, Multi-GPU, P2P, And IPC, Shared Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `BlackScholes.cu, BlackScholes_gold.cpp, BlackScholes_kernel.cuh, BlackScholes.cpp, BlackScholes_gold.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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

- `BlackScholes/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `BlackScholes/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `BlackScholes/BlackScholes.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `BlackScholes/BlackScholes_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `BlackScholes/BlackScholes_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `BlackScholes/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `BlackScholes/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `BlackScholes/doc/BlackScholes.doc`: Supporting file used by `BlackScholes/doc/BlackScholes.doc`.
- `BlackScholes/doc/BlackScholes.pdf`: Supporting file used by `BlackScholes/doc/BlackScholes.pdf`.
- `BlackScholes_nvrtc/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `BlackScholes_nvrtc/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `BlackScholes_nvrtc/BlackScholes.cpp`: Host-side setup, API calls, validation, and cleanup.
- `BlackScholes_nvrtc/BlackScholes_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `BlackScholes_nvrtc/BlackScholes_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `BlackScholes_nvrtc/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `BlackScholes_nvrtc/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `FDTD3d/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `FDTD3d/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `FDTD3d/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `FDTD3d/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `FDTD3d/inc/FDTD3d.h`: Host/device declarations, helper types, constants, or library wrappers.
- `FDTD3d/inc/FDTD3dGPU.h`: Host/device declarations, helper types, constants, or library wrappers.
- `FDTD3d/inc/FDTD3dGPUKernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `FDTD3d/inc/FDTD3dReference.h`: Host/device declarations, helper types, constants, or library wrappers.
- `FDTD3d/src/FDTD3d.cpp`: Host-side setup, API calls, validation, and cleanup.
- `FDTD3d/src/FDTD3dGPU.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `FDTD3d/src/FDTD3dReference.cpp`: Host-side setup, API calls, validation, and cleanup.
- `HSOpticalFlow/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `HSOpticalFlow/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `HSOpticalFlow/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `HSOpticalFlow/FlowCPU.flo`: Supporting file used by `HSOpticalFlow/FlowCPU.flo`.
- `HSOpticalFlow/FlowGPU.flo`: Supporting file used by `HSOpticalFlow/FlowGPU.flo`.
- `HSOpticalFlow/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `HSOpticalFlow/addKernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `HSOpticalFlow/common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `HSOpticalFlow/data/frame10.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `HSOpticalFlow/data/frame11.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `HSOpticalFlow/derivativesKernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `HSOpticalFlow/doc/OpticalFlow.docx`: Supporting file used by `HSOpticalFlow/doc/OpticalFlow.docx`.
- `HSOpticalFlow/doc/OpticalFlow.pdf`: Supporting file used by `HSOpticalFlow/doc/OpticalFlow.pdf`.
- `HSOpticalFlow/downscaleKernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `HSOpticalFlow/flowCUDA.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `HSOpticalFlow/flowCUDA.h`: Host/device declarations, helper types, constants, or library wrappers.
- `HSOpticalFlow/flowGold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `HSOpticalFlow/flowGold.h`: Host/device declarations, helper types, constants, or library wrappers.
- `HSOpticalFlow/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `HSOpticalFlow/solverKernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `HSOpticalFlow/upscaleKernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `HSOpticalFlow/warpingKernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `Mandelbrot/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `Mandelbrot/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `Mandelbrot/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `Mandelbrot/Mandelbrot.cpp`: Host-side setup, API calls, validation, and cleanup.
- `Mandelbrot/Mandelbrot_cuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `Mandelbrot/Mandelbrot_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `Mandelbrot/Mandelbrot_gold.h`: Host/device declarations, helper types, constants, or library wrappers.
- `Mandelbrot/Mandelbrot_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `Mandelbrot/Mandelbrot_kernel.h`: Host/device declarations, helper types, constants, or library wrappers.
- `Mandelbrot/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `Mandelbrot/data/Mandelbrot_fp32.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `Mandelbrot/data/Mandelbrot_fp64.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `Mandelbrot/data/params.txt`: Input, reference, generated-data description, or documentation used by the sample.
- `Mandelbrot/data/referenceJulia_fp32.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `Mandelbrot/data/referenceJulia_fp64.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `Mandelbrot/doc/sshot_lg.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `Mandelbrot/doc/sshot_md.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `Mandelbrot/doc/sshot_sm.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `MonteCarloMultiGPU/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `MonteCarloMultiGPU/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `MonteCarloMultiGPU/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `MonteCarloMultiGPU/MonteCarloMultiGPU.cpp`: Host-side setup, API calls, validation, and cleanup.
- `MonteCarloMultiGPU/MonteCarlo_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MonteCarloMultiGPU/MonteCarlo_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `MonteCarloMultiGPU/MonteCarlo_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `MonteCarloMultiGPU/MonteCarlo_reduction.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `MonteCarloMultiGPU/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `MonteCarloMultiGPU/doc/MonteCarlo.doc`: Supporting file used by `MonteCarloMultiGPU/doc/MonteCarlo.doc`.
- `MonteCarloMultiGPU/doc/MonteCarlo.pdf`: Supporting file used by `MonteCarloMultiGPU/doc/MonteCarlo.pdf`.
- `MonteCarloMultiGPU/multithreading.cpp`: Host-side setup, API calls, validation, and cleanup.
- `MonteCarloMultiGPU/multithreading.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MonteCarloMultiGPU/realtype.h`: Host/device declarations, helper types, constants, or library wrappers.
- `NV12toBGRandResize/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `NV12toBGRandResize/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `NV12toBGRandResize/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `NV12toBGRandResize/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `NV12toBGRandResize/bgr_resize.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `NV12toBGRandResize/data/test1280x720.nv12`: Supporting file used by `NV12toBGRandResize/data/test1280x720.nv12`.
- `NV12toBGRandResize/data/test1920x1080.nv12`: Supporting file used by `NV12toBGRandResize/data/test1920x1080.nv12`.
- `NV12toBGRandResize/data/test640x480.nv12`: Supporting file used by `NV12toBGRandResize/data/test640x480.nv12`.
- `NV12toBGRandResize/nv12_resize.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `NV12toBGRandResize/nv12_to_bgr_planar.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `NV12toBGRandResize/resize_convert.h`: Host/device declarations, helper types, constants, or library wrappers.
- `NV12toBGRandResize/resize_convert_main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `NV12toBGRandResize/utils.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `NV12toBGRandResize/utils.h`: Host/device declarations, helper types, constants, or library wrappers.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `SobelFilter/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `SobelFilter/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `SobelFilter/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `SobelFilter/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `SobelFilter/SobelFilter.cpp`: Host-side setup, API calls, validation, and cleanup.
- `SobelFilter/SobelFilter_kernels.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `SobelFilter/SobelFilter_kernels.h`: Host/device declarations, helper types, constants, or library wrappers.
- `SobelFilter/data/ref_orig.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `SobelFilter/data/ref_shared.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `SobelFilter/data/ref_tex.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `SobelFilter/data/teapot.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `SobelFilter/doc/sshot_lg.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `SobelFilter/doc/sshot_md.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `SobelFilter/doc/sshot_sm.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `SobolQRNG/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `SobolQRNG/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `SobolQRNG/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `SobolQRNG/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `SobolQRNG/sobol.cpp`: Host-side setup, API calls, validation, and cleanup.
- `SobolQRNG/sobol.h`: Host/device declarations, helper types, constants, or library wrappers.
- `SobolQRNG/sobol_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `SobolQRNG/sobol_gold.h`: Host/device declarations, helper types, constants, or library wrappers.
- `SobolQRNG/sobol_gpu.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `SobolQRNG/sobol_gpu.h`: Host/device declarations, helper types, constants, or library wrappers.
- `SobolQRNG/sobol_primitives.cpp`: Host-side setup, API calls, validation, and cleanup.
- `SobolQRNG/sobol_primitives.h`: Host/device declarations, helper types, constants, or library wrappers.
- `bicubicTexture/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `bicubicTexture/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `bicubicTexture/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `bicubicTexture/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `bicubicTexture/bicubicTexture.cpp`: Host-side setup, API calls, validation, and cleanup.
- `bicubicTexture/bicubicTexture_cuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `bicubicTexture/bicubicTexture_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `bicubicTexture/data/0_nearest.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `bicubicTexture/data/1_bilinear.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `bicubicTexture/data/2_bicubic.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `bicubicTexture/data/3_fastbicubic.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `bicubicTexture/data/4_catmull-rom.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `bicubicTexture/data/teapot512.pgm`: Input, reference, generated-data description, or documentation used by the sample.
- `bilateralFilter/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `bilateralFilter/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `bilateralFilter/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `bilateralFilter/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `bilateralFilter/bilateralFilter.cpp`: Host-side setup, API calls, validation, and cleanup.
- `bilateralFilter/bilateralFilter_cpu.cpp`: Host-side setup, API calls, validation, and cleanup.
- `bilateralFilter/bilateral_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `bilateralFilter/bmploader.cpp`: Host-side setup, API calls, validation, and cleanup.
- `bilateralFilter/data/nature_monte.bmp`: Input, reference, generated-data description, or documentation used by the sample.
- `bilateralFilter/data/ref_05.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `bilateralFilter/data/ref_06.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `bilateralFilter/data/ref_07.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `bilateralFilter/data/ref_08.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `binomialOptions/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `binomialOptions/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `binomialOptions/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `binomialOptions/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `binomialOptions/binomialOptions.cpp`: Host-side setup, API calls, validation, and cleanup.
- `binomialOptions/binomialOptions_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `binomialOptions/binomialOptions_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `binomialOptions/binomialOptions_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `binomialOptions/doc/binomialOptions.doc`: Supporting file used by `binomialOptions/doc/binomialOptions.doc`.
- `binomialOptions/doc/binomialOptions.pdf`: Supporting file used by `binomialOptions/doc/binomialOptions.pdf`.
- `binomialOptions/realtype.h`: Host/device declarations, helper types, constants, or library wrappers.
- `binomialOptions_nvrtc/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `binomialOptions_nvrtc/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `binomialOptions_nvrtc/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `binomialOptions_nvrtc/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `binomialOptions_nvrtc/binomialOptions.cpp`: Host-side setup, API calls, validation, and cleanup.
- `binomialOptions_nvrtc/binomialOptions_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `binomialOptions_nvrtc/binomialOptions_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `binomialOptions_nvrtc/binomialOptions_gpu.cpp`: Host-side setup, API calls, validation, and cleanup.
- `binomialOptions_nvrtc/binomialOptions_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `binomialOptions_nvrtc/common_gpu_header.h`: Host/device declarations, helper types, constants, or library wrappers.
- `binomialOptions_nvrtc/realtype.h`: Host/device declarations, helper types, constants, or library wrappers.
- `convolutionFFT2D/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `convolutionFFT2D/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `convolutionFFT2D/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `convolutionFFT2D/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `convolutionFFT2D/convolutionFFT2D.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `convolutionFFT2D/convolutionFFT2D.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `convolutionFFT2D/convolutionFFT2D_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `convolutionFFT2D/convolutionFFT2D_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `convolutionFFT2D/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `dwtHaar1D/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `dwtHaar1D/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `dwtHaar1D/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `dwtHaar1D/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `dwtHaar1D/data/regression.gold.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `dwtHaar1D/data/regression_2_14.gold.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `dwtHaar1D/data/regression_2_18.gold.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `dwtHaar1D/data/signal.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `dwtHaar1D/data/signal_2_14.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `dwtHaar1D/data/signal_2_18.dat`: Input, reference, generated-data description, or documentation used by the sample.
- `dwtHaar1D/dwtHaar1D.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `dwtHaar1D/dwtHaar1D_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `dxtc/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `dxtc/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `dxtc/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `dxtc/CudaMath.h`: Host/device declarations, helper types, constants, or library wrappers.
- `dxtc/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `dxtc/data/teapot512_ref.dds`: Supporting file used by `dxtc/data/teapot512_ref.dds`.
- `dxtc/data/teapot512_std.dds`: Supporting file used by `dxtc/data/teapot512_std.dds`.
- `dxtc/data/teapot512_std.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `dxtc/dds.h`: Host/device declarations, helper types, constants, or library wrappers.
- `dxtc/doc/cuda_dxtc.doc`: Supporting file used by `dxtc/doc/cuda_dxtc.doc`.
- `dxtc/doc/cuda_dxtc.pdf`: Supporting file used by `dxtc/doc/cuda_dxtc.pdf`.
- `dxtc/dxtc.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `dxtc/permutations.h`: Host/device declarations, helper types, constants, or library wrappers.
- `fastWalshTransform/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `fastWalshTransform/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `fastWalshTransform/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `fastWalshTransform/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `fastWalshTransform/doc/FWT.doc`: Supporting file used by `fastWalshTransform/doc/FWT.doc`.
- `fastWalshTransform/fastWalshTransform.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `fastWalshTransform/fastWalshTransform_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `fastWalshTransform/fastWalshTransform_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `fluidsGL/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `fluidsGL/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `fluidsGL/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `fluidsGL/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `fluidsGL/data/ref_fluidsGL.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `fluidsGL/defines.h`: Host/device declarations, helper types, constants, or library wrappers.
- `fluidsGL/doc/fluidsGL.doc`: Supporting file used by `fluidsGL/doc/fluidsGL.doc`.
- `fluidsGL/doc/fluidsGL.pdf`: Supporting file used by `fluidsGL/doc/fluidsGL.pdf`.
- `fluidsGL/doc/fluidsGL_lg.gif`: Supporting file used by `fluidsGL/doc/fluidsGL_lg.gif`.
- `fluidsGL/doc/fluidsGL_md.gif`: Supporting file used by `fluidsGL/doc/fluidsGL_md.gif`.
- `fluidsGL/doc/fluidsGL_sm.gif`: Supporting file used by `fluidsGL/doc/fluidsGL_sm.gif`.
- `fluidsGL/fluidsGL.cpp`: Host-side setup, API calls, validation, and cleanup.
- `fluidsGL/fluidsGL_kernels.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `fluidsGL/fluidsGL_kernels.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `fluidsGL/fluidsGL_kernels.h`: Host/device declarations, helper types, constants, or library wrappers.
- `marchingCubes/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `marchingCubes/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `marchingCubes/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `marchingCubes/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `marchingCubes/data/Bucky.raw`: Supporting file used by `marchingCubes/data/Bucky.raw`.
- `marchingCubes/data/compVoxelArray.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `marchingCubes/data/normalArray.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `marchingCubes/data/posArray.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `marchingCubes/data/ref_march_cubes.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `marchingCubes/defines.h`: Host/device declarations, helper types, constants, or library wrappers.
- `marchingCubes/doc/screenshot_lg.png`: Input, reference, generated-data description, or documentation used by the sample.
- `marchingCubes/doc/screenshot_md.png`: Input, reference, generated-data description, or documentation used by the sample.
- `marchingCubes/doc/screenshot_sm.png`: Input, reference, generated-data description, or documentation used by the sample.
- `marchingCubes/marchingCubes.cpp`: Host-side setup, API calls, validation, and cleanup.
- `marchingCubes/marchingCubes_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `marchingCubes/tables.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nbody/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `nbody/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `nbody/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `nbody/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `nbody/bodysystem.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nbody/bodysystemcpu.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nbody/bodysystemcpu_impl.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nbody/bodysystemcuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `nbody/bodysystemcuda.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nbody/bodysystemcuda_impl.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nbody/doc/nbody_gems3_ch31.pdf`: Supporting file used by `nbody/doc/nbody_gems3_ch31.pdf`.
- `nbody/doc/screenshot_lg.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nbody/doc/screenshot_md.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nbody/doc/screenshot_sm.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nbody/nbody.cpp`: Host-side setup, API calls, validation, and cleanup.
- `nbody/render_particles.cpp`: Host-side setup, API calls, validation, and cleanup.
- `nbody/render_particles.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nbody/tipsy.h`: Host/device declarations, helper types, constants, or library wrappers.
- `p2pBandwidthLatencyTest/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `p2pBandwidthLatencyTest/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `p2pBandwidthLatencyTest/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `p2pBandwidthLatencyTest/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `p2pBandwidthLatencyTest/p2pBandwidthLatencyTest.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `postProcessGL/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `postProcessGL/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `postProcessGL/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `postProcessGL/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `postProcessGL/data/teapot_2.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `postProcessGL/data/teapot_4.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `postProcessGL/data/teapot_8.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `postProcessGL/data/teapot_orig.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `postProcessGL/doc/postProcessGL_lg.gif`: Supporting file used by `postProcessGL/doc/postProcessGL_lg.gif`.
- `postProcessGL/doc/postProcessGL_md.gif`: Supporting file used by `postProcessGL/doc/postProcessGL_md.gif`.
- `postProcessGL/doc/postProcessGL_sm.gif`: Supporting file used by `postProcessGL/doc/postProcessGL_sm.gif`.
- `postProcessGL/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `postProcessGL/postProcessGL.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `quasirandomGenerator/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `quasirandomGenerator/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `quasirandomGenerator/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `quasirandomGenerator/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `quasirandomGenerator/quasirandomGenerator.cpp`: Host-side setup, API calls, validation, and cleanup.
- `quasirandomGenerator/quasirandomGenerator_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `quasirandomGenerator/quasirandomGenerator_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `quasirandomGenerator/quasirandomGenerator_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `quasirandomGenerator_nvrtc/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `quasirandomGenerator_nvrtc/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `quasirandomGenerator_nvrtc/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `quasirandomGenerator_nvrtc/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `quasirandomGenerator_nvrtc/quasirandomGenerator.cpp`: Host-side setup, API calls, validation, and cleanup.
- `quasirandomGenerator_nvrtc/quasirandomGenerator_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `quasirandomGenerator_nvrtc/quasirandomGenerator_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `quasirandomGenerator_nvrtc/quasirandomGenerator_gpu.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `quasirandomGenerator_nvrtc/quasirandomGenerator_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `recursiveGaussian/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `recursiveGaussian/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `recursiveGaussian/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `recursiveGaussian/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `recursiveGaussian/data/ref_10.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `recursiveGaussian/data/ref_14.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `recursiveGaussian/data/ref_18.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `recursiveGaussian/data/ref_22.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `recursiveGaussian/data/teapot512.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `recursiveGaussian/recursiveGaussian.cpp`: Host-side setup, API calls, validation, and cleanup.
- `recursiveGaussian/recursiveGaussian_cuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `recursiveGaussian/recursiveGaussian_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `simpleD3D11/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleD3D11/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleD3D11/ShaderStructs.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleD3D11/data/ref_simpleD3D11.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleD3D11/simpleD3D11.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleD3D11/sinewave_cuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleD3D11/sinewave_cuda.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleD3D11Texture/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleD3D11Texture/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleD3D11Texture/d3dx11effect/d3dx11effect.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleD3D11Texture/data/ref_simpleD3D11Texture.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleD3D11Texture/simpleD3D11Texture.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleD3D11Texture/texture_2d.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleD3D11Texture/texture_3d.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleD3D11Texture/texture_cube.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleD3D12/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleD3D12/DX12CudaSample.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleD3D12/DX12CudaSample.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleD3D12/DXSampleHelper.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleD3D12/Main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleD3D12/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleD3D12/ShaderStructs.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleD3D12/Win32Application.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleD3D12/Win32Application.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleD3D12/d3dx12.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleD3D12/shaders.hlsl`: Supporting file used by `simpleD3D12/shaders.hlsl`.
- `simpleD3D12/simpleD3D12.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleD3D12/simpleD3D12.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleD3D12/sinewave_cuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleD3D12/stdafx.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleD3D12/stdafx.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleGL/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleGL/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleGL/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleGL/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleGL/data/ref_simpleGL.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleGL/simpleGL.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleVulkan/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleVulkan/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleVulkan/Build_instructions.txt`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleVulkan/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleVulkan/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleVulkan/SineWaveSimulation.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleVulkan/SineWaveSimulation.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleVulkan/VulkanBaseApp.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleVulkan/VulkanBaseApp.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleVulkan/frag.spv`: Supporting file used by `simpleVulkan/frag.spv`.
- `simpleVulkan/linmath.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleVulkan/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleVulkan/sinewave.frag`: Supporting file used by `simpleVulkan/sinewave.frag`.
- `simpleVulkan/sinewave.vert`: Supporting file used by `simpleVulkan/sinewave.vert`.
- `simpleVulkan/vert.spv`: Supporting file used by `simpleVulkan/vert.spv`.
- `simpleVulkanMMAP/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleVulkanMMAP/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleVulkanMMAP/Build_instructions.txt`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleVulkanMMAP/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleVulkanMMAP/MonteCarloPi.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleVulkanMMAP/MonteCarloPi.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleVulkanMMAP/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleVulkanMMAP/VulkanBaseApp.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleVulkanMMAP/VulkanBaseApp.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleVulkanMMAP/VulkanCudaInterop.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleVulkanMMAP/frag.spv`: Supporting file used by `simpleVulkanMMAP/frag.spv`.
- `simpleVulkanMMAP/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleVulkanMMAP/montecarlo.frag`: Supporting file used by `simpleVulkanMMAP/montecarlo.frag`.
- `simpleVulkanMMAP/montecarlo.vert`: Supporting file used by `simpleVulkanMMAP/montecarlo.vert`.
- `simpleVulkanMMAP/vert.spv`: Supporting file used by `simpleVulkanMMAP/vert.spv`.
- `smokeParticles/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `smokeParticles/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `smokeParticles/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `smokeParticles/GLSLProgram.cpp`: Host-side setup, API calls, validation, and cleanup.
- `smokeParticles/GLSLProgram.h`: Host/device declarations, helper types, constants, or library wrappers.
- `smokeParticles/GpuArray.h`: Host/device declarations, helper types, constants, or library wrappers.
- `smokeParticles/ParticleSystem.cpp`: Host-side setup, API calls, validation, and cleanup.
- `smokeParticles/ParticleSystem.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `smokeParticles/ParticleSystem.h`: Host/device declarations, helper types, constants, or library wrappers.
- `smokeParticles/ParticleSystem_cuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `smokeParticles/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `smokeParticles/SmokeRenderer.cpp`: Host-side setup, API calls, validation, and cleanup.
- `smokeParticles/SmokeRenderer.h`: Host/device declarations, helper types, constants, or library wrappers.
- `smokeParticles/SmokeShaders.cpp`: Host-side setup, API calls, validation, and cleanup.
- `smokeParticles/SmokeShaders.h`: Host/device declarations, helper types, constants, or library wrappers.
- `smokeParticles/data/floortile.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `smokeParticles/data/ref_smokePart_pos.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `smokeParticles/data/ref_smokePart_vel.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `smokeParticles/doc/screenshot_lg.png`: Input, reference, generated-data description, or documentation used by the sample.
- `smokeParticles/doc/screenshot_md.png`: Input, reference, generated-data description, or documentation used by the sample.
- `smokeParticles/doc/screenshot_sm.png`: Input, reference, generated-data description, or documentation used by the sample.
- `smokeParticles/doc/smokeParticles.doc`: Supporting file used by `smokeParticles/doc/smokeParticles.doc`.
- `smokeParticles/doc/smokeParticles.pdf`: Supporting file used by `smokeParticles/doc/smokeParticles.pdf`.
- `smokeParticles/framebufferObject.cpp`: Host-side setup, API calls, validation, and cleanup.
- `smokeParticles/framebufferObject.h`: Host/device declarations, helper types, constants, or library wrappers.
- `smokeParticles/nvMath.h`: Host/device declarations, helper types, constants, or library wrappers.
- `smokeParticles/nvMatrix.h`: Host/device declarations, helper types, constants, or library wrappers.
- `smokeParticles/nvQuaternion.h`: Host/device declarations, helper types, constants, or library wrappers.
- `smokeParticles/nvVector.h`: Host/device declarations, helper types, constants, or library wrappers.
- `smokeParticles/particleDemo.cpp`: Host-side setup, API calls, validation, and cleanup.
- `smokeParticles/particles_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `smokeParticles/particles_kernel_device.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `smokeParticles/renderbuffer.cpp`: Host-side setup, API calls, validation, and cleanup.
- `smokeParticles/renderbuffer.h`: Host/device declarations, helper types, constants, or library wrappers.
- `stereoDisparity/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `stereoDisparity/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `stereoDisparity/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `stereoDisparity/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `stereoDisparity/data/stereo.im0.640x533.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `stereoDisparity/data/stereo.im1.640x533.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `stereoDisparity/stereoDisparity.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `stereoDisparity/stereoDisparity_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `volumeFiltering/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `volumeFiltering/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `volumeFiltering/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `volumeFiltering/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `volumeFiltering/data/Bucky.raw`: Supporting file used by `volumeFiltering/data/Bucky.raw`.
- `volumeFiltering/data/ref_volumefilter.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `volumeFiltering/doc/sshot_lg.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `volumeFiltering/doc/sshot_md.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `volumeFiltering/doc/sshot_sm.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `volumeFiltering/volume.cpp`: Host-side setup, API calls, validation, and cleanup.
- `volumeFiltering/volume.h`: Host/device declarations, helper types, constants, or library wrappers.
- `volumeFiltering/volumeFilter.h`: Host/device declarations, helper types, constants, or library wrappers.
- `volumeFiltering/volumeFilter_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `volumeFiltering/volumeFiltering.cpp`: Host-side setup, API calls, validation, and cleanup.
- `volumeFiltering/volumeRender.h`: Host/device declarations, helper types, constants, or library wrappers.
- `volumeFiltering/volumeRender_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `volumeRender/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `volumeRender/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `volumeRender/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `volumeRender/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `volumeRender/data/Bucky.raw`: Supporting file used by `volumeRender/data/Bucky.raw`.
- `volumeRender/data/ref_volume.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `volumeRender/doc/sshot_lg.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `volumeRender/doc/sshot_md.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `volumeRender/doc/sshot_sm.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `volumeRender/volume.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `volumeRender/volumeRender.cpp`: Host-side setup, API calls, validation, and cleanup.
- `volumeRender/volumeRender_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `vulkanImageCUDA/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `vulkanImageCUDA/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `vulkanImageCUDA/Build_instructions.txt`: Input, reference, generated-data description, or documentation used by the sample.
- `vulkanImageCUDA/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `vulkanImageCUDA/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `vulkanImageCUDA/frag.spv`: Supporting file used by `vulkanImageCUDA/frag.spv`.
- `vulkanImageCUDA/linmath.h`: Host/device declarations, helper types, constants, or library wrappers.
- `vulkanImageCUDA/shader.frag`: Supporting file used by `vulkanImageCUDA/shader.frag`.
- `vulkanImageCUDA/shader.vert`: Supporting file used by `vulkanImageCUDA/shader.vert`.
- `vulkanImageCUDA/teapot1024.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `vulkanImageCUDA/vert.spv`: Supporting file used by `vulkanImageCUDA/vert.spv`.
- `vulkanImageCUDA/vulkanImageCUDA.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `BlackScholes.cu` first and locate the host-side setup or Python entry point.
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

- `BlackScholes/BlackScholes.cu`: focus on `cudaMalloc`, `cudaMemcpy`, `cudaFree`, `cudaMemcpyHostToDevice`, `cudaDeviceSynchronize`.
- `BlackScholes/BlackScholes_gold.cpp`: focus on control flow and helper functions.
- `BlackScholes/BlackScholes_kernel.cuh`: focus on `blockDim`, `blockIdx`, `threadIdx`, `launch`, `gridDim`.
- `BlackScholes_nvrtc/BlackScholes.cpp`: focus on `cuMemAlloc`, `cuMemFree`, `cuMemcpyHtoD`, `cudaBlockSize`, `cudaGridSize`.
- `BlackScholes_nvrtc/BlackScholes_gold.cpp`: focus on control flow and helper functions.
- `BlackScholes_nvrtc/BlackScholes_kernel.cuh`: focus on `blockDim`, `blockIdx`, `threadIdx`, `launch`.
- `FDTD3d/inc/FDTD3d.h`: focus on control flow and helper functions.
- `FDTD3d/inc/FDTD3dGPU.h`: focus on `launch`.
- `FDTD3d/inc/FDTD3dGPUKernel.cuh`: focus on `blockDim`, `threadIdx`, `blockIdx`, `__shared__`, `launch`.
- `FDTD3d/inc/FDTD3dReference.h`: focus on control flow and helper functions.
- Additional source files: 184 more support files. Use the same setup/work/sync/cleanup lens.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/CMakeLists.txt:1-38
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

add_subdirectory(BlackScholes)
add_subdirectory(BlackScholes_nvrtc)
add_subdirectory(FDTD3d)
add_subdirectory(HSOpticalFlow)
add_subdirectory(Mandelbrot)
add_subdirectory(MonteCarloMultiGPU)
add_subdirectory(NV12toBGRandResize)
add_subdirectory(SobelFilter)
add_subdirectory(SobolQRNG)
add_subdirectory(bicubicTexture)
add_subdirectory(bilateralFilter)
add_subdirectory(binomialOptions)
add_subdirectory(binomialOptions_nvrtc)
add_subdirectory(convolutionFFT2D)
add_subdirectory(dwtHaar1D)
add_subdirectory(dxtc)
add_subdirectory(fastWalshTransform)
add_subdirectory(fluidsGL)
add_subdirectory(marchingCubes)
add_subdirectory(nbody)
add_subdirectory(p2pBandwidthLatencyTest)
add_subdirectory(postProcessGL)
add_subdirectory(quasirandomGenerator)
add_subdirectory(quasirandomGenerator_nvrtc)
add_subdirectory(recursiveGaussian)
add_subdirectory(simpleD3D11)
add_subdirectory(simpleD3D11Texture)
add_subdirectory(simpleD3D12)
add_subdirectory(simpleGL)
add_subdirectory(simpleVulkan)
add_subdirectory(simpleVulkanMMAP)
add_subdirectory(smokeParticles)
add_subdirectory(stereoDisparity)
add_subdirectory(volumeFiltering)
add_subdirectory(volumeRender)
add_subdirectory(vulkanImageCUDA)
```

> JP: この抜粋は `cpp/5_Domain_Specific/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaTextureObject_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaCheckError` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMemcpyHostToDevice` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaResourceDesc` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaStream_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaTextureDesc` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

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
cmake --build build --target 5_Domain_Specific
ctest --test-dir build -R 5_Domain_Specific
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
- [Shared Memory](../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Streams And Events](../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Unified Memory](../../docs_ja/themes/unified_memory.md): managed memory、migration、prefetch の意味を読むための基礎です。
- [Synchronization And Atomics](../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
