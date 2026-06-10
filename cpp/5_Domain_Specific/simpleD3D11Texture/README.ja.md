# simpleD3D11Texture - Simple D3D11 Texture - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

Simple program which demonstrates Direct3D11 Texture interoperability with CUDA.  The program creates a number of D3D11 Textures (2D, 3D, and CubeMap) which are written to from CUDA kernels. Direct3D then renders the results on the screen.  A Direct3D Capable device is required.

Graphics Interop, Image Processing

Original README headings: `simpleD3D11Texture - Simple D3D11 Texture`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/simpleD3D11Texture` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `simpleD3D11Texture` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/simpleD3D11Texture`.
> **日本語**
> この sample の目的は、`simpleD3D11Texture` の小さな実装を通して CUDA Graphs, Streams And Events, Synchronization And Atomics, Memory, Kernel Launch And Indexing を具体的に追うことです。
>
> **学習メモ**
> 最初に `d3dx11effect.h, simpleD3D11Texture.cpp, texture_2d.cu, texture_3d.cu, texture_cube.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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

- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `d3dx11effect/d3dx11effect.h`: Host/device declarations, helper types, constants, or library wrappers.
- `data/ref_simpleD3D11Texture.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `simpleD3D11Texture.cpp`: Host-side setup, API calls, validation, and cleanup.
- `texture_2d.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `texture_3d.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `texture_cube.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `d3dx11effect.h` first and locate the host-side setup or Python entry point.
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

- `d3dx11effect/d3dx11effect.h`: focus on `Stream`, `Device`.
- `simpleD3D11Texture.cpp`: focus on `CUDA`, `cudaLinearMemory`, `cudaResource`, `launch`, `cuArray`.
- `texture_2d.cu`: focus on `launch`, `blockIdx`, `blockDim`, `threadIdx`, `cudaSuccess`.
- `texture_3d.cu`: focus on `launch`, `blockIdx`, `blockDim`, `threadIdx`, `cudaSuccess`.
- `texture_cube.cu`: focus on `launch`, `blockIdx`, `blockDim`, `threadIdx`, `cudaSuccess`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/simpleD3D11Texture/CMakeLists.txt:1-64
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleD3D11Texture LANGUAGES C CXX CUDA)

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

if(WIN32)
    # Source file
    # Add target for simpleD3D11Texture
    add_executable(simpleD3D11Texture
        simpleD3D11Texture.cpp
        ../../../Common/rendercheck_d3d11.cpp
        texture_2d.cu
        texture_3d.cu
        texture_cube.cu
    )

    target_compile_options(simpleD3D11Texture PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(simpleD3D11Texture PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(simpleD3D11Texture PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(simpleD3D11Texture PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(simpleD3D11Texture PRIVATE
        d3d11
        dxgi
        dxguid
        d3dcompiler
    )

    add_custom_command(TARGET simpleD3D11Texture POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        ${CMAKE_CURRENT_BINARY_DIR}/data
    )
else()
    message(STATUS "Sample 'simpleD3D11Texture' is Windows-only - skipping")
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11Texture/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `simpleD3D11Texture.cpp`

Source: cpp/5_Domain_Specific/simpleD3D11Texture/simpleD3D11Texture.cpp:33-51
```cpp
#pragma warning(disable : 4312)

// includes for Windows
#include <windows.h>

// includes for multimedia
#include <mmsystem.h>

// This header inclues all the necessary D3D11 and CUDA includes
#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>
#include <d3dcompiler.h>
#include <dynlink_d3d11.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // includes cuda.h and cuda_runtime_api.h
#include <rendercheck_d3d11.h>

```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11Texture/simpleD3D11Texture.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleD3D11Texture/simpleD3D11Texture.cpp:329-348
```cpp
        printf("> There are no device(s) supporting CUDA\n");
        return false;
    }
    else {
        // JP: python_cuda: Python object が CUDA resource を包みます。Python から見えても device memory/stream/context の寿命と順序は CUDA 側で管理します。
        printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
    }

    // Get CUDA device properties
    cudaDeviceProp deviceProp;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaGetDeviceProperties(&deviceProp, dev);
        STRCPY(devname, NAME_LEN, deviceProp.name);
        printf("> GPU %d: %s\n", dev, devname);
    }

    return true;
}

```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11Texture/simpleD3D11Texture.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleD3D11Texture/simpleD3D11Texture.cpp:524-549
```cpp
        getLastCudaError("cudaGraphicsD3D11RegisterResource (g_texture_2d) failed");
        // cuda cannot write into the texture directly : the texture is seen as a
        // cudaArray and can only be mapped as a texture
        // Create a buffer so that cuda can write into it
        // pixel fmt is DXGI_FORMAT_R32G32B32A32_FLOAT
        // JP: `cudaMallocPitch`, `cudaLinearMemory`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
        cudaMallocPitch(&g_texture_2d.cudaLinearMemory,
                        &g_texture_2d.pitch,
                        g_texture_2d.width * sizeof(float) * 4,
                        g_texture_2d.height);
        getLastCudaError("cudaMallocPitch (g_texture_2d) failed");
        // JP: `cudaMemset`, `cudaLinearMemory`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
        cudaMemset(g_texture_2d.cudaLinearMemory, 1, g_texture_2d.pitch * g_texture_2d.height);

        // CUBE
        cudaGraphicsD3D11RegisterResource(
            &g_texture_cube.cudaResource, g_texture_cube.pTexture, cudaGraphicsRegisterFlagsNone);
        getLastCudaError("cudaGraphicsD3D11RegisterResource (g_texture_cube) failed");
        // create the buffer. pixel fmt is DXGI_FORMAT_R8G8B8A8_SNORM
        // JP: この anchor では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
        cudaMallocPitch(
            &g_texture_cube.cudaLinearMemory, &g_texture_cube.pitch, g_texture_cube.size * 4, g_texture_cube.size);
        getLastCudaError("cudaMallocPitch (g_texture_cube) failed");
        // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
        cudaMemset(g_texture_cube.cudaLinearMemory, 1, g_texture_cube.pitch * g_texture_cube.size);
        getLastCudaError("cudaMemset (g_texture_cube) failed");
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11Texture/simpleD3D11Texture.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleD3D11Texture/simpleD3D11Texture.cpp:593-612
```cpp
                    const char *cur_image_path = "simpleD3D11Texture.ppm";

                    // Save a reference of our current test run image
                    CheckRenderD3D11::ActiveRenderTargetToPPM(g_pd3dDevice, cur_image_path);

                    // compare to offical reference image, printing PASS or FAIL.
                    g_bPassed = CheckRenderD3D11::PPMvsPPM(cur_image_path, ref_file, argv[0], MAX_EPSILON, 0.15f);

                    g_bDone = true;

                    Cleanup();

                    PostQuitMessage(0);
                }
                else {
                    g_bPassed = true;
                }
            }
        }
    };
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11Texture/simpleD3D11Texture.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `texture_2d.cu`

Source: cpp/5_Domain_Specific/simpleD3D11Texture/texture_2d.cu:29-81
```cuda
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.1415926536f

/*
 * Paint a 2D texture with a moving red/green hatch pattern on a
 * strobing blue background.  Note that this kernel reads to and
 * writes from the texture, hence why this texture was not mapped
 * as WriteDiscard.
 */
__global__ void cuda_kernel_texture_2d(unsigned char *surface, int width, int height, size_t pitch, float t)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int    x = blockIdx.x * blockDim.x + threadIdx.x;
    int    y = blockIdx.y * blockDim.y + threadIdx.y;
    float *pixel;

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (x >= width || y >= height)
        return;

    // get a pointer to the pixel at (x,y)
    pixel = (float *)(surface + y * pitch) + 4 * x;

    // populate it
    float value_x = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * x) / width - 1.0f));
    float value_y = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * y) / height - 1.0f));
    pixel[0]      = 0.5 * pixel[0] + 0.5 * pow(value_x, 3.0f); // red
    pixel[1]      = 0.5 * pixel[1] + 0.5 * pow(value_y, 3.0f); // green
    pixel[2]      = 0.5f + 0.5f * cos(t);                      // blue
    pixel[3]      = 1;                                         // alpha
}

extern "C" void cuda_texture_2d(void *surface, int width, int height, size_t pitch, float t)
{
    cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16); // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

    // JP: `cuda_kernel_texture_2d`: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    cuda_kernel_texture_2d<<<Dg, Db>>>((unsigned char *)surface, width, height, pitch, t);

    error = cudaGetLastError();

    if (error != cudaSuccess) {
        printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
    }
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11Texture/texture_2d.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `texture_3d.cu`

Source: cpp/5_Domain_Specific/simpleD3D11Texture/texture_3d.cu:29-83
```cuda
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Paint a 3D texture with a gradient in X (blue) and Z (green), and have every
 * other Z slice have full red.
 */
__global__ void cuda_kernel_texture_3d(unsigned char *surface,
                                       int            width,
                                       int            height,
                                       int            depth,
                                       size_t         pitch,
                                       size_t         pitchSlice,
                                       float          t)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (x >= width || y >= height)
        return;

    // walk across the Z slices of this texture.  it should be noted that
    // this is far from optimal data access.
    for (int z = 0; z < depth; ++z) {
        // get a pointer to this pixel
        unsigned char *pixel = surface + z * pitchSlice + y * pitch + 4 * x;
        pixel[0] = (unsigned char)(255.f * (0.5f + 0.5f * cos(t + (x * x + y * y + z * z) * 0.0001f * 3.14f))); // red
        pixel[1] = (unsigned char)(255.f * (0.5f + 0.5f * sin(t + (x * x + y * y + z * z) * 0.0001f * 3.14f))); // green
        pixel[2] = (unsigned char)0;                                                                            // blue
        pixel[3] = 255;                                                                                         // alpha
    }
}

extern "C" void
cuda_texture_3d(void *surface, int width, int height, int depth, size_t pitch, size_t pitchSlice, float t)
{
    cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16); // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

    // JP: `cuda_kernel_texture_3d`: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    cuda_kernel_texture_3d<<<Dg, Db>>>((unsigned char *)surface, width, height, depth, pitch, pitchSlice, t);

    error = cudaGetLastError();

    if (error != cudaSuccess) {
        printf("cuda_kernel_texture_3d() failed to launch error = %d\n", error);
    }
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11Texture/texture_3d.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `texture_cube.cu`

Source: cpp/5_Domain_Specific/simpleD3D11Texture/texture_cube.cu:29-95
```cuda
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.1415926536f

/*
 * Paint a 2D surface with a moving bulls-eye pattern.  The "face" parameter
 * selects
 * between 6 different colors to use.  We will use a different color on each
 * face of a
 * cube map.
 */
__global__ void cuda_kernel_texture_cube(char *surface, int width, int height, size_t pitch, int face, float t)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int            x = blockIdx.x * blockDim.x + threadIdx.x;
    int            y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned char *pixel;

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (x >= width || y >= height)
        return;

    // get a pointer to this pixel
    pixel = (unsigned char *)(surface + y * pitch) + 4 * x;

    // populate it
    float         theta_x = (2.0f * x) / width - 1.0f;
    float         theta_y = (2.0f * y) / height - 1.0f;
    float         theta   = 2.0f * PI * sqrt(theta_x * theta_x + theta_y * theta_y);
    unsigned char value   = 255 * (0.6f + 0.4f * cos(theta + t));

    pixel[3] = 255; // alpha

    if (face % 2) {
        pixel[0]        =      // blue
            pixel[1]    =      // green
            pixel[2]    = 0.5; // red
        pixel[face / 2] = value;
    }
    else {
        pixel[0]        =        // blue
            pixel[1]    =        // green
            pixel[2]    = value; // red
        pixel[face / 2] = 0.5;
    }
}

extern "C" void cuda_texture_cube(void *surface, int width, int height, size_t pitch, int face, float t)
{
    cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16); // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

    // JP: `cuda_kernel_texture_cube`: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    cuda_kernel_texture_cube<<<Dg, Db>>>((char *)surface, width, height, pitch, face, t);

    error = cudaGetLastError();

    if (error != cudaSuccess) {
        printf("cuda_kernel_texture_cube() failed to launch error = %d\n", error);
    }
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D11Texture/texture_cube.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaLinearMemory` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaResource` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMallocPitch` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cuArray` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaMemset` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaGraphicsSubResourceGetMappedArray` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphicsUnregisterResource` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaArray` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGraphicsD3D11RegisterResource` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |

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
cmake --build build --target simpleD3D11Texture
ctest --test-dir build -R simpleD3D11Texture
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

- `cudaLinearMemory` の直前と直後で、どの memory/resource が有効になったかをメモする。
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
