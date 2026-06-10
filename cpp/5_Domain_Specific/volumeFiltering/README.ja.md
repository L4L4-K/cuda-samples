# volumeFiltering - Volumetric Filtering with 3D Textures and Surface Writes - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates 3D Volumetric Filtering using 3D Textures and 3D Surface Writes.

Graphics Interop, Image Processing, 3D Textures, Surface Writes

Original README headings: `volumeFiltering - Volumetric Filtering with 3D Textures and Surface Writes`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/volumeFiltering` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `volumeFiltering` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/volumeFiltering`.
> **日本語**
> この sample の目的は、`volumeFiltering` の小さな実装を通して CUDA Graphs, Streams And Events, Synchronization And Atomics, Memory, Kernel Launch And Indexing を具体的に追うことです。
>
> **学習メモ**
> 最初に `volume.cpp, volume.h, volumeFilter.h, volumeFilter_kernel.cu, volumeFiltering.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit

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
- `data/Bucky.raw`: Supporting file used by `data/Bucky.raw`.
- `data/ref_volumefilter.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/sshot_lg.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/sshot_md.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/sshot_sm.JPG`: Input, reference, generated-data description, or documentation used by the sample.
- `volume.cpp`: Host-side setup, API calls, validation, and cleanup.
- `volume.h`: Host/device declarations, helper types, constants, or library wrappers.
- `volumeFilter.h`: Host/device declarations, helper types, constants, or library wrappers.
- `volumeFilter_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `volumeFiltering.cpp`: Host-side setup, API calls, validation, and cleanup.
- `volumeRender.h`: Host/device declarations, helper types, constants, or library wrappers.
- `volumeRender_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `volume.cpp` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Capture or build CUDA Graph nodes, instantiate the graph, then launch the executable graph.
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

- `volume.cpp`: focus on `cudaResourceDesc`, `cudaAddressModeWrap`, `cudaMemcpyHostToDevice`, `cudaResourceTypeArray`, `cudaTextureDesc`.
- `volume.h`: focus on `cudaTextureReadMode`, `cudaExtent`, `cudaReadModeNormalizedFloat`, `cudaArray`, `cudaChannelFormatDesc`.
- `volumeFilter.h`: focus on control flow and helper functions.
- `volumeFilter_kernel.cu`: focus on `blockIdx`, `blockDim`, `threadIdx`, `launch`, `cudaExtent`.
- `volumeFiltering.cpp`: focus on `CUDA`, `launch`, `cudaDeviceSynchronize`, `cudaMemset`, `cudaGraphicsResource`.
- `volumeRender.h`: focus on `cudaTextureObject_t`.
- `volumeRender_kernel.cu`: focus on `cudaTextureObject_t`, `cudaResourceDesc`, `cudaTextureDesc`, `blockIdx`, `blockDim`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/volumeFiltering/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(volumeFiltering LANGUAGES C CXX CUDA)

# JP: `find_package`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)
```

> JP: この抜粋は `cpp/5_Domain_Specific/volumeFiltering/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `volume.cpp`

Source: cpp/5_Domain_Specific/volumeFiltering/volume.cpp:30-94
```cpp
#include <cuda_runtime.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_math.h>

#include "volume.h"

void Volume_init(Volume *vol, cudaExtent dataSize, void *h_data, int allowStore)
{
    // create 3D array
    vol->channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors(
        cudaMalloc3DArray(&vol->content, &vol->channelDesc, dataSize, allowStore ? cudaArraySurfaceLoadStore : 0));
    vol->size = dataSize;

    if (h_data) {
        // copy data to 3D array
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr =
            make_cudaPitchedPtr(h_data, dataSize.width * sizeof(VolumeType), dataSize.width, dataSize.height);
        copyParams.dstArray = vol->content;
        copyParams.extent   = dataSize;
        // JP: `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
        copyParams.kind     = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));
    }

    if (allowStore) {
        cudaResourceDesc surfRes;
        memset(&surfRes, 0, sizeof(cudaResourceDesc));
        surfRes.resType         = cudaResourceTypeArray;
        surfRes.res.array.array = vol->content;

        checkCudaErrors(cudaCreateSurfaceObject(&vol->volumeSurf, &surfRes));
    }

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType         = cudaResourceTypeArray;
    texRes.res.array.array = vol->content;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode       = cudaFilterModeLinear;
    texDescr.addressMode[0]   = cudaAddressModeWrap;
    texDescr.addressMode[1]   = cudaAddressModeWrap;
    texDescr.addressMode[2]   = cudaAddressModeWrap;
    texDescr.readMode         = cudaReadModeNormalizedFloat; // VolumeTypeInfo<VolumeType>::readMode;

    checkCudaErrors(cudaCreateTextureObject(&vol->volumeTex, &texRes, &texDescr, NULL));
}

void Volume_deinit(Volume *vol)
{
    // JP: `cudaDestroyTextureObject`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaDestroyTextureObject(vol->volumeTex));
    checkCudaErrors(cudaDestroySurfaceObject(vol->volumeSurf));
    // JP: `cudaFreeArray`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaFreeArray(vol->content));
    vol->content = 0;
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/volumeFiltering/volume.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `volume.h`

Source: cpp/5_Domain_Specific/volumeFiltering/volume.h:29-89
```cpp
#ifndef _VOLUME_H_
#define _VOLUME_H_

#include <cuda_runtime.h>

typedef unsigned char VolumeType;

extern "C"
{

    struct Volume
    {
        cudaArray            *content;
        cudaExtent            size;
        cudaChannelFormatDesc channelDesc;
        cudaTextureObject_t   volumeTex;
        cudaSurfaceObject_t   volumeSurf;
    };

    void Volume_init(Volume *vol, cudaExtent size, void *data, int allowStore);
    void Volume_deinit(Volume *vol);
};

//////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

/* Helper class to do popular integer storage to float conversions if required
 */

template <typename T> struct VolumeTypeInfo
{
};

template <> struct VolumeTypeInfo<unsigned char>
{
    static const cudaTextureReadMode           readMode = cudaReadModeNormalizedFloat;
    static __inline__ __device__ unsigned char convert(float sampled)
    {
        return (unsigned char)(__saturatef(sampled) * 255.0);
    }
};

template <> struct VolumeTypeInfo<unsigned short>
{
    static const cudaTextureReadMode            readMode = cudaReadModeNormalizedFloat;
    static __inline__ __device__ unsigned short convert(float sampled)
    {
        return (unsigned short)(__saturatef(sampled) * 65535.0);
    }
};

template <> struct VolumeTypeInfo<float>
{
    static const cudaTextureReadMode   readMode = cudaReadModeElementType;
    static __inline__ __device__ float convert(float sampled) { return sampled; }
};

#endif

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/volumeFiltering/volume.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `volumeFilter.h`

Source: cpp/5_Domain_Specific/volumeFiltering/volumeFilter.h:29-49
```cpp
#ifndef _VOLUMEFILTER_KERNEL_H_
#define _VOLUMEFILTER_KERNEL_H_

#define VOLUMEFILTER_MAXWEIGHTS 125

#include <cuda_runtime.h>

#include "volume.h"

extern "C"
{
    Volume *VolumeFilter_runFilter(Volume *input,
                                   Volume *output0,
                                   Volume *output1,
                                   int     iterations,
                                   int     numWeights,
                                   float4 *weights,
                                   float   postWeightOffset);
};

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/volumeFiltering/volumeFilter.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `volumeFilter_kernel.cu`

Source: cpp/5_Domain_Specific/volumeFiltering/volumeFilter_kernel.cu:29-63
```cuda
#ifndef _VOLUMEFILTER_KERNEL_CU_
#define _VOLUMEFILTER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>

#include "volumeFilter.h"

typedef unsigned int   uint;
typedef unsigned char  uchar;
typedef unsigned short ushort;

__constant__ float4 c_filterData[VOLUMEFILTER_MAXWEIGHTS];

__global__ void d_filter_surface3d(int                 filterSize,
                                   float               filter_offset,
                                   cudaExtent          volumeSize,
                                   cudaTextureObject_t volumeTexIn,
                                   cudaSurfaceObject_t volumeTexOut)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth) {
        return;
    }

    float  filtered  = 0;
    float4 basecoord = make_float4(x, y, z, 0);

    for (int i = 0; i < filterSize; i++) {
        float4 coord = basecoord + c_filterData[i];
        filtered += tex3D<float>(volumeTexIn, coord.x, coord.y, coord.z) * c_filterData[i].w;
```

> JP: この抜粋は `cpp/5_Domain_Specific/volumeFiltering/volumeFilter_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/volumeFiltering/volumeFilter_kernel.cu:95-114
```cuda
    unsigned int dim  = 32 / sizeof(VolumeType);
    dim3         blockSize(dim, dim, 1);
    dim3 gridSize(iDivUp(size.width, blockSize.x), iDivUp(size.height, blockSize.y), iDivUp(size.depth, blockSize.z));

    // set weights
    checkCudaErrors(cudaMemcpyToSymbol(c_filterData, weights, sizeof(float4) * numWeights));

    for (int i = 0; i < iterations; i++) {
        // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
        d_filter_surface3d<<<gridSize, blockSize>>>(
            numWeights, postWeightOffset, size, input->volumeTex, output0->volumeSurf);

        getLastCudaError("filter kernel failed");

        swap    = input;
        input   = output0;
        output0 = swap;

        if (i == 0) {
            output0 = output1;
```

> JP: この抜粋は `cpp/5_Domain_Specific/volumeFiltering/volumeFilter_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `volumeFiltering.cpp`

Source: cpp/5_Domain_Specific/volumeFiltering/volumeFiltering.cpp:29-47
```cpp
#include <helper_gl.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// CUDA Runtime and Interop
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

// Helper functions
#include <helper_functions.h>
#include <helper_timer.h>

```

> JP: この抜粋は `cpp/5_Domain_Specific/volumeFiltering/volumeFiltering.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/volumeFiltering/volumeFiltering.cpp:225-244
```cpp

    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes, cuda_pbo_resource));
    // printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    // JP: `cudaMemset`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemset(d_output, 0, width * height * 4));

    // call CUDA kernel, writing results to PBO
    VolumeRender_render(gridSize,
                        blockSize,
                        d_output,
                        width,
                        height,
```

> JP: この抜粋は `cpp/5_Domain_Specific/volumeFiltering/volumeFiltering.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/volumeFiltering/volumeFiltering.cpp:459-478
```cpp
{
    width  = w;
    height = h;
    initPixelBuffer();

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}


void initGL(int *argc, char **argv)
```

> JP: この抜粋は `cpp/5_Domain_Specific/volumeFiltering/volumeFiltering.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/volumeFiltering/volumeFiltering.cpp:601-620
```cpp
    void  *h_volume = loadRawFile(path, size);

    FilterKernel_init();
    Volume_init(&volumeOriginal, volumeSize, h_volume, 0);
    // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    free(h_volume);
    Volume_init(&volumeFilter0, volumeSize, NULL, 1);
    Volume_init(&volumeFilter1, volumeSize, NULL, 1);
    VolumeRender_init();
    VolumeRender_setPreIntegrated(preIntegrated);

    sdkCreateTimer(&timer);
    sdkCreateTimer(&animationTimer);
    sdkStartTimer(&animationTimer);

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
}

//////////////////////////////////////////////////////////////////////////
```

> JP: この抜粋は `cpp/5_Domain_Specific/volumeFiltering/volumeFiltering.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `volumeRender.h`

Source: cpp/5_Domain_Specific/volumeFiltering/volumeRender.h:29-56
```cpp
#ifndef _VOLUMERENDER__H_
#define _VOLUMERENDER__H_

#include <cuda_runtime.h>

#include "volume.h"

extern "C"
{
    void VolumeRender_init();
    void VolumeRender_deinit();

    void VolumeRender_setPreIntegrated(int state);
    void VolumeRender_setTextureFilterMode(bool bLinearFilter, Volume *volume);
    void VolumeRender_render(dim3                gridSize,
                             dim3                blockSize,
                             uint               *d_output,
                             uint                imageW,
                             uint                imageH,
                             float               density,
                             float               brightness,
                             float               transferOffset,
                             float               transferScale,
                             cudaTextureObject_t tex);
    void VolumeRender_copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);
};

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/volumeFiltering/volumeRender.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `volumeRender_kernel.cu`

Source: cpp/5_Domain_Specific/volumeFiltering/volumeRender_kernel.cu:31-49
```cuda
#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>

#include "volumeRender.h"

#define VOLUMERENDER_TFS            2
#define VOLUMERENDER_TF_PREINTSIZE  1024
#define VOLUMERENDER_TF_PREINTSTEPS 1024
#define VOLUMERENDER_TF_PREINTRAY   4

enum TFMode {
    TF_SINGLE_1D         = 0, // single 1D TF for everything
    TF_LAYERED_2D_PREINT = 1, // layered 2D TF uses pre-integration
    TF_LAYERED_2D        = 2, // layered 2D TF without pre-integration behavior
};

```

> JP: この抜粋は `cpp/5_Domain_Specific/volumeFiltering/volumeRender_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/volumeFiltering/volumeRender_kernel.cu:151-170
```cuda
    const float3 boxMin           = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax           = make_float3(1.0f, 1.0f, 1.0f);

    density *= rayscale;

    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH))
        return;

    float u = (x / (float)imageW) * 2.0f - 1.0f;
    float v = (y / (float)imageH) * 2.0f - 1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);
```

> JP: この抜粋は `cpp/5_Domain_Specific/volumeFiltering/volumeRender_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/volumeFiltering/volumeRender_kernel.cu:407-426
```cuda
//////////////////////////////////////////////////////////////////////////

void VolumeRender_setTextureFilterMode(bool bLinearFilter, Volume *vol)
{
    if (vol->volumeTex) {
        // JP: `cudaDestroyTextureObject`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        checkCudaErrors(cudaDestroyTextureObject(vol->volumeTex));
    }
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType         = cudaResourceTypeArray;
    texRes.res.array.array = vol->content;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode       = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;

```

> JP: この抜粋は `cpp/5_Domain_Specific/volumeFiltering/volumeRender_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/volumeFiltering/volumeRender_kernel.cu:451-472
```cuda
        checkCudaErrors(cudaFreeArray(d_transferFunc));
        d_transferFunc = 0;
    }

    cudaChannelFormatDesc channelFloat4 = cudaCreateChannelDesc<float4>();
    checkCudaErrors(cudaMallocArray(&d_transferFunc, &channelFloat4, numColors, 1));
    checkCudaErrors(
        // JP: `cudaMemcpy2DToArray`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
        cudaMemcpy2DToArray(d_transferFunc, 0, 0, colors, 0, sizeof(float4) * numColors, 1, cudaMemcpyHostToDevice));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType         = cudaResourceTypeArray;
    texRes.res.array.array = d_transferFunc;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode       = cudaFilterModeLinear;
    texDescr.addressMode[0]   = cudaAddressModeClamp;
```

> JP: この抜粋は `cpp/5_Domain_Specific/volumeFiltering/volumeRender_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaTextureObject_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaResourceDesc` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaExtent` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaTextureDesc` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDestroyTextureObject` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaFreeArray` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaResourceTypeArray` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaCreateTextureObject` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaAddressModeWrap` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaSurfaceObject_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- CUDA Graph は一連の work を node と依存関係として再利用します。capture 対象と buffer lifetime を確認します。
- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
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
cmake --build build --target volumeFiltering
ctest --test-dir build -R volumeFiltering
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

The sample may display a window or produce/validate image-like output; exact visuals depend on platform support.
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
- graph capture 後に buffer lifetime や node dependency が変わったことを見落とす。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaTextureObject_t` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- graph node の依存関係を箇条書きにし、どの buffer lifetime が graph 実行全体をまたぐか確認する。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [CUDA Graphs](../../../docs_ja/themes/graphs.md): capture、node dependency、replay、graph update を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
