# cudaNvSciBufMultiplanar - CUDA NvSciBufMultiplanar Image Samples - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates CUDA-NvSciBuf Interop for Multiplanar images. A YUV 420 multiplanar image is flipped and allocated using NvSciBuf APIs and imported into CUDA with CUDA External Resource Interoperability. A CUDA surface is created from the corresponding mapped CUDA array and again bit flipping is performed on the surface. The result is copied back to a YUV image which is compared against the input.

CUDA NvSci Interop, Data Parallel Algorithms, Image Processing

Original README headings: `cudaNvSciBufMultiplanar - CUDA NvSciBufMultiplanar Image Samples`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `cudaNvSciBufMultiplanar` as a focused example of the CUDA concepts used in `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar`.
> **日本語**
> この sample の目的は、`cudaNvSciBufMultiplanar` の小さな実装を通して Runtime, Driver, And NVRTC, Streams And Events, Synchronization And Atomics, Performance, Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `cudaNvSciBufMultiplanar.cpp, cudaNvSciBufMultiplanar.h, imageKernels.cu, main.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- The graphics, display, or platform stack named by the English README

> **日本語**
> 必要条件は英語 README と CMake/requirements を優先します。この guide は条件を置き換えず、読むべき確認点を追加します。
>
> **学習メモ**
> 実行できない場合は、source を変える前に driver、toolkit、GPU feature、library、platform guard、Python package version を確認します。

## Files

- `.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cudaNvSciBufMultiplanar.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cudaNvSciBufMultiplanar.h`: Host/device declarations, helper types, constants, or library wrappers.
- `imageKernels.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `yuv_planar_img1.yuv`: Supporting file used by `yuv_planar_img1.yuv`.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `cudaNvSciBufMultiplanar.cpp` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.
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

- `cudaNvSciBufMultiplanar.cpp`: focus on `cudaNvSciBufMultiplanar`, `CUDA`, `cudaArray_t`, `cudaArr`, `cuCtxSynchronize`.
- `cudaNvSciBufMultiplanar.h`: focus on `CUDA_NVSCIBUF_MULTIPLANAR_H`, `cudaArray_t`, `cudaNvSciBufMultiplanar`, `CUresult`, `CUDA_SUCCESS`.
- `imageKernels.cu`: focus on `launch`, `blockIdx`, `blockDim`, `threadIdx`, `cudaSurfaceObject_t`.
- `main.cpp`: focus on `cudaNvSciBufMultiplanar`, `cudaDeviceGetAttribute`, `cudaNvSciBufMultiplanarApp`, `cudaGetDeviceCount`, `cudaDevAttrComputeCapabilityMajor`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/CMakeLists.txt:1-67
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/Modules")

project(cudaNvSciBufMultiplanar LANGUAGES C CXX CUDA)

# JP: `find_package`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 87 110)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../../Common)

find_package(NVSCI)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(NVSCI_FOUND)
        # Source file
        # Add target for cudaNvSciBufMultiplanar
        add_executable(cudaNvSciBufMultiplanar imageKernels.cu cudaNvSciBufMultiplanar.cpp main.cpp)

        target_compile_options(cudaNvSciBufMultiplanar PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(cudaNvSciBufMultiplanar PRIVATE cxx_std_17 cuda_std_17)

        set_target_properties(cudaNvSciBufMultiplanar PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(cudaNvSciBufMultiplanar PUBLIC
            ${CUDAToolkit_INCLUDE_DIRS}
            ${NVSCI_INCLUDE_DIRS}
        )

        target_link_libraries(cudaNvSciBufMultiplanar
            CUDA::cuda_driver
            ${NVSCI_LIBRARIES}
        )
        # Copy yuv_planar_img1.yuv to the output directory
        add_custom_command(TARGET cudaNvSciBufMultiplanar POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/yuv_planar_img1.yuv ${CMAKE_CURRENT_BINARY_DIR}/yuv_planar_img1.yuv
        )
        # Specify additional clean files
        set_target_properties(cudaNvSciBufMultiplanar PROPERTIES
            ADDITIONAL_CLEAN_FILES "image_out.yuv"
        )
    else()
        message(STATUS "NvSCI not found - will not build sample 'cudaNvSciBufMultiplanar'")
    endif()
else()
    message(STATUS "Will not build sample cudaNvSciBufMultiplanar - requires Linux OS")
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cudaNvSciBufMultiplanar.cpp`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.cpp:29-47
```cpp
#include "cudaNvSciBufMultiplanar.h"

NvSciBufModule module;
NvSciBufObj    buffObj;
CUuuid         uuid;

void flipBits(uint8_t *pBuff, uint32_t size)
{
    for (uint32_t i = 0; i < size; i++) {
        pBuff[i] = (~pBuff[i]);
    }
}

// Compare input and generated image files
// JP: validation: GPU result を CPU/reference と比較する検証地点です。失敗時は transfer、indexing、sync の順に疑います。
void compareFiles(std::string &path1, std::string &path2)
{
    bool  result = true;
    FILE *fp1, *fp2;
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.cpp:92-111
```cpp
}

void Caller::deinit()
{
    NvSciBufAttrListFree(attrList);
    // JP: `cudaDestroyExternalMemory`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaDestroyExternalMemory(extMem));
}

// Set NvSciBufImage attribute values in the attribute list
void Caller::setAttrListImageMultiPlanes(int imageWidth, int imageHeight)
{
    NvSciBufType                   bufType       = NvSciBufType_Image;
    NvSciBufAttrValImageLayoutType layout        = NvSciBufImage_BlockLinearType;
    bool                           cpuAccessFlag = false;
    NvSciBufAttrValAccessPerm      perm          = NvSciBufAccessPerm_ReadWrite;
    NvSciRmGpuId                   gpuid;
    bool                           vpr        = false;
    int32_t                        planeCount = PLANAR_NUM_PLANES;
    int                            drvVersion;
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.cpp:156-175
```cpp
void cudaNvSciBufMultiplanar::initCuda(int devId)
{
    int          major = 0, minor = 0, drvVersion;
    NvSciRmGpuId gpuid;

    checkCudaErrors(cudaSetDevice(mCudaDeviceId));
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, mCudaDeviceId));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, mCudaDeviceId));
    printf("[cudaNvSciBufMultiplanar] GPU Device %d: \"%s\" with compute capability "
           "%d.%d\n\n",
           mCudaDeviceId,
           _ConvertSMVer2ArchName(major, minor),
           major,
           minor);

    // JP: `cuDriverGetVersion`: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
    checkCudaDrvErrors(cuDriverGetVersion(&drvVersion));

    if (drvVersion <= 11030) {
        checkCudaDrvErrors(cuDeviceGetUuid(&uuid, devId));
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.cpp:371-390
```cpp
    copyWidthInBytes[2] = width / PLANAR_CHROMA_WIDTH_ORDER;
    uvOffset[1]         = width * height;
    uvOffset[2]         = uvOffset[1] + (width / PLANAR_CHROMA_WIDTH_ORDER) * (height / PLANAR_CHROMA_HEIGHT_ORDER);
    for (int i = 0; i < numPlanes; i++) {
        checkCudaDrvErrors(cuCtxSynchronize());
        checkCudaErrors(cudaMemcpy2DToArray(cudaArr[i],
                                            0,
                                            0,
                                            (void *)(pBuff + uvOffset[i]),
                                            copyWidthInBytes[i],
                                            copyWidthInBytes[i],
                                            copyHeight[i],
                                            // JP: `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
                                            cudaMemcpyHostToDevice));
    }

    if (fp) {
        fclose(fp);
        fp = NULL;
    }
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cudaNvSciBufMultiplanar.h`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.h:29-47
```cpp
#ifndef CUDA_NVSCIBUF_MULTIPLANAR_H
#define CUDA_NVSCIBUF_MULTIPLANAR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <nvscibuf.h>
#include <vector>

#define PLANAR_NUM_PLANES          3
#define PLANAR_CHROMA_WIDTH_ORDER  2
#define PLANAR_CHROMA_HEIGHT_ORDER 2

#define ATTR_SIZE   20
#define DEFAULT_GPU 0

#define checkNvSciErrors(call)                                   \
    do {                                                         \
        NvSciError _status = call;                               \
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.h:60-79
```cpp
#define checkCudaDrvErrors(call)                           \
    do {                                                   \
        CUresult err = call;                               \
        if (CUDA_SUCCESS != err) {                         \
            const char *errorStr = NULL;                   \
            cuGetErrorString(err, &errorStr);              \
            printf("checkCudaDrvErrors() Driver API error" \
                   " = %04d \"%s\" from file <%s>, "       \
                   "line %i.\n",                           \
                   err,                                    \
                   errorStr,                               \
                   __FILE__,                               \
                   __LINE__);                              \
            exit(EXIT_FAILURE);                            \
        }                                                  \
    } while (0)

extern void launchFlipSurfaceBitsKernel(cudaArray_t *levelArray,
                                        int32_t     *multiPlanarWidth,
                                        int32_t     *multiPlanarHeight,
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `imageKernels.cu`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/imageKernels.cu:29-68
```cuda
#include <cuda.h>
#include <helper_cuda.h>

static __global__ void flipSurfaceBits(cudaSurfaceObject_t surfObj, int width, int height)
{
    char         data;
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        // Read from input surface
        surf2Dread(&data, surfObj, x, y);
        // Write to output surface
        data = ~data;
        surf2Dwrite(data, surfObj, x, y);
    }
}

// Copy cudaArray to surface memory and launch the CUDA kernel
void launchFlipSurfaceBitsKernel(cudaArray_t *levelArray,
                                 int32_t     *multiPlanarWidth,
                                 int32_t     *multiPlanarHeight,
                                 int          numPlanes)
{

    cudaSurfaceObject_t surfObject[numPlanes] = {0};
    cudaResourceDesc    resDesc;

    for (int i = 0; i < numPlanes; i++) {
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = levelArray[i];
        checkCudaErrors(cudaCreateSurfaceObject(&surfObject[i], &resDesc));
        dim3 threadsperBlock(16, 16);
        dim3 numBlocks((multiPlanarWidth[i] + threadsperBlock.x - 1) / threadsperBlock.x,
                       (multiPlanarHeight[i] + threadsperBlock.y - 1) / threadsperBlock.y);
        // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
        flipSurfaceBits<<<numBlocks, threadsperBlock>>>(surfObject[i], multiPlanarWidth[i], multiPlanarHeight[i]);
    }
}
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/imageKernels.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `main.cpp`

Source: cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/main.cpp:29-74
```cpp
#include <cuda.h>
#include <helper_image.h>
#include <vector>

#include "cudaNvSciBufMultiplanar.h"

#define MAX_FILE_SIZE 100

int main(int argc, const char **argv)
{
    int              numOfGPUs = 0;
    std::vector<int> deviceIds;
    (cudaGetDeviceCount(&numOfGPUs));

    printf("%d GPUs found\n", numOfGPUs);
    if (!numOfGPUs) {
        exit(EXIT_WAIVED);
    }
    else {
        for (int devID = 0; devID < numOfGPUs; devID++) {
            int major = 0, minor = 0;
            (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
            (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
            if (major >= 6) {
                deviceIds.push_back(devID);
            }
        }
        if (deviceIds.size() == 0) {
            printf("cudaNvSciBufMultiplanar requires one or more GPUs of Pascal(SM 6.0) or higher "
                   "archs\nWaiving..\n");
            exit(EXIT_WAIVED);
        }
    }

    std::string image_filename     = sdkFindFilePath("yuv_planar_img1.yuv", argv[0]);
    std::string image_filename_out = "image_out.yuv";
    uint32_t    imageWidth         = 720;
    uint32_t    imageHeight        = 480;

    printf("input image %s , width = %d, height = %d\n", image_filename.c_str(), imageWidth, imageHeight);

    cudaNvSciBufMultiplanar cudaNvSciBufMultiplanarApp(imageWidth, imageHeight, deviceIds);
    cudaNvSciBufMultiplanarApp.runCudaNvSciBufPlanar(image_filename, image_filename_out);

    return EXIT_SUCCESS;
}
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaNvSciBufMultiplanar` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaArray_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDeviceGetAttribute` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuCtxSynchronize` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaDestroyExternalMemory` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cuDriverGetVersion` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaGetMipmappedArrayLevel` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaFreeMipmappedArray` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaArr` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaSetDevice` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuDeviceGetUuid` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaImportExternalMemory` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaCreateChannelDesc` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaExternalMemoryGetMappedMipmappedArray` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
- Runtime/Driver/NVRTC sample では、compile/load した module と launch する kernel 名の対応が重要です。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。
- 同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target cudaNvSciBufMultiplanar
ctest --test-dir build -R cudaNvSciBufMultiplanar
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
- different stream 間に依存があるのに event や explicit sync を置かない。
- JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaNvSciBufMultiplanar` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Runtime, Driver, And NVRTC](../../../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [Streams And Events](../../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Performance](../../../../docs_ja/themes/performance.md): memory traffic、occupancy、overlap、launch overhead、timing を読むための基礎です。
- [Memory](../../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
