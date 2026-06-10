# 8. Platform_Specific/Tegra - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates CUDA-NvMedia interop via NvSciBuf/NvSciSync APIs. Note that this sample only supports cross build from x86_64 to aarch64, aarch64 native build is not supported. For detailed workflow of the sample please check cudaNvSciNvMedia_Readme.pdf in the sample directory.

This sample demonstrates CUDA-NvSciBuf Interop for Multiplanar images. A YUV 420 multiplanar image is flipped and allocated using NvSciBuf APIs and imported into CUDA with CUDA External Resource Interoperability. A CUDA surface is created from the corresponding mapped CUDA array and again bit flipping is performed on the surface. The result is copied back to a YUV image which is compared against the input.

Original README headings: `8. Platform_Specific/Tegra`, `[cudaNvSciNvMedia](./cudaNvSciNvMedia)`, `[cudaNvSciBufMultiplanar](./cudaNvSciBufMultiplanar)`, `[cuDLAErrorReporting](./cuDLAErrorReporting)`, `[cuDLAHybridMode](./cuDLAHybridMode)`, `[cuDLALayerwiseStatsHybrid](./cuDLALayerwiseStatsHybrid)`, `[cuDLALayerwiseStatsStandalone](./cuDLALayerwiseStatsStandalone)`, `[cuDLAStandaloneMode](./cuDLAStandaloneMode)`, `[EGLSync_CUDAEvent_Interop](./EGLSync_CUDAEvent_Interop)`, `[fluidsGLES](./fluidsGLES)`, `[nbody_opengles](./nbody_opengles)`, `[simpleGLES](./simpleGLES)`

> **日本語**
> `cpp/8_Platform_Specific/Tegra` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `Tegra` as a focused example of the CUDA concepts used in `cpp/8_Platform_Specific/Tegra`.
> **日本語**
> この sample の目的は、`Tegra` の小さな実装を通して Runtime, Driver, And NVRTC, CUDA Libraries, CUDA Graphs, Shared Memory, Streams And Events を具体的に追うことです。
>
> **学習メモ**
> 最初に `EGLSync_CUDAEvent_Interop.cu, egl_common.h, graphics_interface.h, main.cu, main.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- CUDA library components used by this sample, such as cuBLAS, cuFFT, cuSolver, NPP, CUB, or nvJPEG

> **日本語**
> 必要条件は英語 README と CMake/requirements を優先します。この guide は条件を置き換えず、読むべき確認点を追加します。
>
> **学習メモ**
> 実行できない場合は、source を変える前に driver、toolkit、GPU feature、library、platform guard、Python package version を確認します。

## Files

- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `EGLSync_CUDAEvent_Interop/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `EGLSync_CUDAEvent_Interop/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `EGLSync_CUDAEvent_Interop/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `EGLSync_CUDAEvent_Interop/EGLSync_CUDAEvent_Interop.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `EGLSync_CUDAEvent_Interop/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `EGLSync_CUDAEvent_Interop/egl_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `EGLSync_CUDAEvent_Interop/graphics_interface.h`: Host/device declarations, helper types, constants, or library wrappers.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cuDLAErrorReporting/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuDLAErrorReporting/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuDLAErrorReporting/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cuDLAErrorReporting/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cuDLAErrorReporting/main.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cuDLAHybridMode/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuDLAHybridMode/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuDLAHybridMode/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cuDLAHybridMode/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cuDLAHybridMode/main.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cuDLALayerwiseStatsHybrid/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuDLALayerwiseStatsHybrid/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuDLALayerwiseStatsHybrid/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cuDLALayerwiseStatsHybrid/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cuDLALayerwiseStatsHybrid/main.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cuDLALayerwiseStatsStandalone/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuDLALayerwiseStatsStandalone/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuDLALayerwiseStatsStandalone/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cuDLALayerwiseStatsStandalone/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cuDLALayerwiseStatsStandalone/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cuDLAStandaloneMode/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuDLAStandaloneMode/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cuDLAStandaloneMode/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cuDLAStandaloneMode/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cuDLAStandaloneMode/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaNvSciBufMultiplanar/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cudaNvSciBufMultiplanar/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cudaNvSciBufMultiplanar/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cudaNvSciBufMultiplanar/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cudaNvSciBufMultiplanar/imageKernels.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cudaNvSciBufMultiplanar/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaNvSciBufMultiplanar/yuv_planar_img1.yuv`: Supporting file used by `cudaNvSciBufMultiplanar/yuv_planar_img1.yuv`.
- `cudaNvSciNvMedia/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cudaNvSciNvMedia/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `cudaNvSciNvMedia/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `cudaNvSciNvMedia/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cudaNvSciNvMedia/cudaNvSciNvMedia_Readme.pdf`: Supporting file used by `cudaNvSciNvMedia/cudaNvSciNvMedia_Readme.pdf`.
- `cudaNvSciNvMedia/cuda_consumer.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `cudaNvSciNvMedia/cuda_consumer.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cudaNvSciNvMedia/main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaNvSciNvMedia/nvmedia_producer.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaNvSciNvMedia/nvmedia_producer.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cudaNvSciNvMedia/nvmedia_utils/cmdline.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaNvSciNvMedia/nvmedia_utils/cmdline.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cudaNvSciNvMedia/nvmedia_utils/config_parser.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaNvSciNvMedia/nvmedia_utils/config_parser.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cudaNvSciNvMedia/nvmedia_utils/image_utils.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaNvSciNvMedia/nvmedia_utils/image_utils.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cudaNvSciNvMedia/nvmedia_utils/log_utils.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaNvSciNvMedia/nvmedia_utils/log_utils.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cudaNvSciNvMedia/nvmedia_utils/misc_utils.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaNvSciNvMedia/nvmedia_utils/misc_utils.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cudaNvSciNvMedia/nvsci_setup.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaNvSciNvMedia/nvsci_setup.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cudaNvSciNvMedia/sample.cfg`: Supporting file used by `cudaNvSciNvMedia/sample.cfg`.
- `cudaNvSciNvMedia/teapot.rgba`: Supporting file used by `cudaNvSciNvMedia/teapot.rgba`.
- `fluidsGLES/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `fluidsGLES/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `fluidsGLES/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `fluidsGLES/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `fluidsGLES/data/ref_fluidsGLES.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `fluidsGLES/defines.h`: Host/device declarations, helper types, constants, or library wrappers.
- `fluidsGLES/fluidsGLES.cpp`: Host-side setup, API calls, validation, and cleanup.
- `fluidsGLES/fluidsGLES_kernels.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `fluidsGLES/fluidsGLES_kernels.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `fluidsGLES/fluidsGLES_kernels.h`: Host/device declarations, helper types, constants, or library wrappers.
- `fluidsGLES/graphics_interface.h`: Host/device declarations, helper types, constants, or library wrappers.
- `fluidsGLES/mesh.frag.glsl`: Supporting file used by `fluidsGLES/mesh.frag.glsl`.
- `fluidsGLES/mesh.vert.glsl`: Supporting file used by `fluidsGLES/mesh.vert.glsl`.
- `nbody_opengles/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `nbody_opengles/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `nbody_opengles/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `nbody_opengles/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `nbody_opengles/bodysystem.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nbody_opengles/bodysystemcpu.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nbody_opengles/bodysystemcpu_impl.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nbody_opengles/bodysystemcuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `nbody_opengles/bodysystemcuda.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nbody_opengles/bodysystemcuda_impl.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nbody_opengles/galaxy_20K.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `nbody_opengles/nbody_opengles.cpp`: Host-side setup, API calls, validation, and cleanup.
- `nbody_opengles/render_particles.cpp`: Host-side setup, API calls, validation, and cleanup.
- `nbody_opengles/render_particles.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nbody_opengles/tipsy.h`: Host/device declarations, helper types, constants, or library wrappers.
- `simpleGLES/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleGLES/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleGLES/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleGLES/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleGLES/data/ref_simpleGL.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleGLES/graphics_interface.c`: Host-side setup, API calls, validation, and cleanup.
- `simpleGLES/mesh.frag.glsl`: Supporting file used by `simpleGLES/mesh.frag.glsl`.
- `simpleGLES/mesh.vert.glsl`: Supporting file used by `simpleGLES/mesh.vert.glsl`.
- `simpleGLES/simpleGLES.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `simpleGLES_EGLOutput/.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleGLES_EGLOutput/.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `simpleGLES_EGLOutput/CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `simpleGLES_EGLOutput/README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `simpleGLES_EGLOutput/data/ref_simpleGLES_EGLOutput.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleGLES_EGLOutput/graphics_interface_egloutput_via_egl.c`: Host-side setup, API calls, validation, and cleanup.
- `simpleGLES_EGLOutput/mesh.frag.glsl`: Supporting file used by `simpleGLES_EGLOutput/mesh.frag.glsl`.
- `simpleGLES_EGLOutput/mesh.vert.glsl`: Supporting file used by `simpleGLES_EGLOutput/mesh.vert.glsl`.
- `simpleGLES_EGLOutput/simpleGLES_EGLOutput.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `EGLSync_CUDAEvent_Interop.cu` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Create library handles, descriptors, plans, or workspaces before the library call.
- Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.
- Capture or build CUDA Graph nodes, instantiate the graph, then launch the executable graph.
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

- `EGLSync_CUDAEvent_Interop/EGLSync_CUDAEvent_Interop.cu`: focus on `CUDA_SUCCESS`, `CUDA`, `cudaSuccess`, `CUsurfObject`, `cudaGetErrorString`.
- `EGLSync_CUDAEvent_Interop/egl_common.h`: focus on `cudaEGL`.
- `EGLSync_CUDAEvent_Interop/graphics_interface.h`: focus on `CUDA`.
- `cuDLAErrorReporting/main.cu`: focus on `cudaSuccess`, `cudaFree`, `CUDA`, `cudaGetErrorName`, `cudaStream_t`.
- `cuDLAHybridMode/main.cu`: focus on `cudaSuccess`, `cudaFree`, `CUDA`, `cudaGetErrorName`, `cudaStream_t`.
- `cuDLALayerwiseStatsHybrid/main.cu`: focus on `cudaSuccess`, `cudaFree`, `CUDA`, `cudaGetErrorName`, `cuDLA`.
- `cuDLALayerwiseStatsStandalone/main.cpp`: focus on `cuDLA`, `Device`, `CUDA`, `atomic`, `CUDLA_STANDALONE`.
- `cuDLAStandaloneMode/main.cpp`: focus on `cuDLA`, `cuDLAStandaloneMode`, `Device`, `CUDA`, `CUDLA_NVSCISYNC_FENCE`.
- `cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.cpp`: focus on `cudaNvSciBufMultiplanar`, `CUDA`, `cudaArray_t`, `cudaArr`, `cuCtxSynchronize`.
- `cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.h`: focus on `CUDA_NVSCIBUF_MULTIPLANAR_H`, `cudaArray_t`, `cudaNvSciBufMultiplanar`, `CUresult`, `CUDA_SUCCESS`.
- Additional source files: 39 more support files. Use the same setup/work/sync/cleanup lens.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/8_Platform_Specific/Tegra/CMakeLists.txt:1-18
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

add_subdirectory(cudaNvSciNvMedia)
add_subdirectory(cudaNvSciBufMultiplanar)
add_subdirectory(cuDLAErrorReporting)
add_subdirectory(cuDLAHybridMode)
add_subdirectory(cuDLALayerwiseStatsHybrid)
add_subdirectory(cuDLALayerwiseStatsStandalone)
add_subdirectory(cuDLAStandaloneMode)
add_subdirectory(EGLSync_CUDAEvent_Interop)
add_subdirectory(fluidsGLES)
add_subdirectory(nbody_opengles)
add_subdirectory(simpleGLES)
add_subdirectory(simpleGLES_EGLOutput)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaExtResObj` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaResObj` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaDeviceId` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CUDA_SUCCESS` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuDLA` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaGraphicsResource` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaSetDevice` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaStream_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |

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
cmake --build build --target Tegra
ctest --test-dir build -R Tegra
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

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaExtResObj` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- shared memory tile の producer、consumer、barrier を図にする。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- graph node の依存関係を箇条書きにし、どの buffer lifetime が graph 実行全体をまたぐか確認する。
- handle/descriptor/workspace の作成、利用、破棄を対応表にする。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Runtime, Driver, And NVRTC](../../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [CUDA Libraries](../../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。
- [CUDA Graphs](../../../docs_ja/themes/graphs.md): capture、node dependency、replay、graph update を読むための基礎です。
- [Shared Memory](../../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Performance](../../../docs_ja/themes/performance.md): memory traffic、occupancy、overlap、launch overhead、timing を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
