# simpleVulkan - Vulkan CUDA Interop Sinewave - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates Vulkan CUDA Interop. CUDA imports the Vulkan vertex buffer and operates on it to create sinewave, and synchronizes with Vulkan through vulkan semaphores imported by CUDA. This sample depends on Vulkan SDK, GLFW3 libraries, for building this sample please refer to "Build_instructions.txt" provided in this sample's directory

Graphics Interop, CUDA Vulkan Interop, Data Parallel Algorithms

Original README headings: `simpleVulkan - Vulkan CUDA Interop Sinewave`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/simpleVulkan` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `simpleVulkan` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/simpleVulkan`.
> **日本語**
> この sample の目的は、`simpleVulkan` の小さな実装を通して CUDA Graphs, Streams And Events, Synchronization And Atomics, Performance, Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `SineWaveSimulation.cu, SineWaveSimulation.h, VulkanBaseApp.cpp, VulkanBaseApp.h, linmath.h` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `Build_instructions.txt`: Input, reference, generated-data description, or documentation used by the sample.
- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `SineWaveSimulation.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `SineWaveSimulation.h`: Host/device declarations, helper types, constants, or library wrappers.
- `VulkanBaseApp.cpp`: Host-side setup, API calls, validation, and cleanup.
- `VulkanBaseApp.h`: Host/device declarations, helper types, constants, or library wrappers.
- `frag.spv`: Supporting file used by `frag.spv`.
- `linmath.h`: Host/device declarations, helper types, constants, or library wrappers.
- `main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `sinewave.frag`: Supporting file used by `sinewave.frag`.
- `sinewave.vert`: Supporting file used by `sinewave.vert`.
- `vert.spv`: Supporting file used by `vert.spv`.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `SineWaveSimulation.cu` first and locate the host-side setup or Python entry point.
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

- `SineWaveSimulation.cu`: focus on `CUDA`, `launch`, `blockDim`, `cudaGetDeviceProperties`, `gridDim`.
- `SineWaveSimulation.h`: focus on `cudaStream_t`, `launch`.
- `VulkanBaseApp.cpp`: focus on `CUDA`, `atomic`.
- `VulkanBaseApp.h`: focus on `atomic`, `CUDA`.
- `linmath.h`: focus on control flow and helper functions.
- `main.cpp`: focus on `cudaDestroyExternalSemaphore`, `cudaMem`, `cudaStream_t`, `cudaExternalSemaphore_t`, `cudaExternalMemory_t`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/simpleVulkan/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleVulkan LANGUAGES C CXX CUDA)

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

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkan/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `SineWaveSimulation.cu`

Source: cpp/5_Domain_Specific/simpleVulkan/SineWaveSimulation.cu:29-80
```cuda
#include <algorithm>
#include <helper_cuda.h>

#include "SineWaveSimulation.h"

__global__ void sinewave(float *heightMap, unsigned int width, unsigned int height, float time)
{
    const float  freq   = 4.0f;
    // JP: `gridDim`, `blockDim`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    const size_t stride = gridDim.x * blockDim.x;

    // Iterate through the entire array in a way that is
    // independent of the grid configuration
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < width * height; tid += stride) {
        // Calculate the x, y coordinates
        const size_t y = tid / width;
        const size_t x = tid - y * width;
        // Normalize x, y to [0,1]
        const float u = ((2.0f * x) / width) - 1.0f;
        const float v = ((2.0f * y) / height) - 1.0f;
        // Calculate the new height value
        const float w = 0.5f * sinf(u * freq + time) * cosf(v * freq + time);
        // Store this new height value
        heightMap[tid] = w;
    }
}

SineWaveSimulation::SineWaveSimulation(size_t width, size_t height)
    : m_heightMap(nullptr)
    , m_width(width)
    , m_height(height)
{
}

void SineWaveSimulation::initCudaLaunchConfig(int device)
{
    cudaDeviceProp prop = {};
    checkCudaErrors(cudaSetDevice(device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));

    // We don't need large block sizes, since there's not much inter-thread
    // communication
    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    m_threads = prop.warpSize;

    // Use the occupancy calculator and fill the gpu as best as we can
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&m_blocks, sinewave, prop.warpSize, 0));
    m_blocks *= prop.multiProcessorCount;

    // Go ahead and the clamp the blocks to the minimum needed for this
    // height/width
    m_blocks = std::min(m_blocks, (int)((m_width * m_height + m_threads - 1) / m_threads));
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkan/SineWaveSimulation.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `SineWaveSimulation.h`

Source: cpp/5_Domain_Specific/simpleVulkan/SineWaveSimulation.h:29-58
```cpp
#pragma once
#ifndef __SINESIM_H__
#define __SINESIM_H__

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <vector>

#include "linmath.h"

class SineWaveSimulation
{
    float *m_heightMap;
    size_t m_width, m_height;
    int    m_blocks, m_threads;

public:
    SineWaveSimulation(size_t width, size_t height);
    ~SineWaveSimulation();
    void initSimulation(float *heightMap);
    // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    void stepSimulation(float time, cudaStream_t stream = 0);
    void initCudaLaunchConfig(int device);
    int  initCuda(uint8_t *vkDeviceUUID, size_t UUID_SIZE);

    size_t getWidth() const { return m_width; }
    size_t getHeight() const { return m_height; }
};

#endif // __SINESIM_H__
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkan/SineWaveSimulation.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `VulkanBaseApp.cpp`

Source: cpp/5_Domain_Specific/simpleVulkan/VulkanBaseApp.cpp:35-53
```cpp
#include "VulkanBaseApp.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string.h>

#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <GLFW/glfw3.h>

#ifdef _WIN64
#include <VersionHelpers.h>
#include <aclapi.h>
#include <dxgi1_2.h>
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkan/VulkanBaseApp.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleVulkan/VulkanBaseApp.cpp:280-299
```cpp

WindowsSecurityAttributes::WindowsSecurityAttributes()
{
    m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
    if (!m_winPSecurityDescriptor) {
        // JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
        throw std::runtime_error("Failed to allocate memory for security descriptor");
    }

    PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

    InitializeSecurityDescriptor(m_winPSecurityDescriptor, SECURITY_DESCRIPTOR_REVISION);

    SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority = SECURITY_WORLD_SID_AUTHORITY;
    AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, ppSID);

    EXPLICIT_ACCESS explicitAccess;
    ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
    explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkan/VulkanBaseApp.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleVulkan/VulkanBaseApp.cpp:324-343
```cpp
    }
    if (*ppACL) {
        LocalFree(*ppACL);
    }
    // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    free(m_winPSecurityDescriptor);
}
#endif /* _WIN64 */

static VkFormat findSupportedFormat(VkPhysicalDevice             physicalDevice,
                                    const std::vector<VkFormat> &candidates,
                                    VkImageTiling                tiling,
                                    VkFormatFeatureFlags         features)
{
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        }
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkan/VulkanBaseApp.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `VulkanBaseApp.h`

Source: cpp/5_Domain_Specific/simpleVulkan/VulkanBaseApp.h:29-47
```cpp
#pragma once
#ifndef __VULKANBASEAPP_H__
#define __VULKANBASEAPP_H__

#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#ifdef _WIN64
#define NOMINMAX
// Add windows.h to the include path
#include <windows.h>
// Add vulkan_win32.h to the include path
#include <vulkan/vulkan_win32.h>
#endif /* _WIN64 */

/* remove _VK_TIMELINE_SEMAPHORE to use binary semaphores */
// use vulkan timeline semaphore
#define _VK_TIMELINE_SEMAPHORE

```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkan/VulkanBaseApp.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `linmath.h`

Source: cpp/5_Domain_Specific/simpleVulkan/linmath.h:22-40
```cpp
#ifndef LINMATH_H
#define LINMATH_H

#define _USE_MATH_DEFINES
#include <math.h>

// Converts degrees to radians.
#define degreesToRadians(angleDegrees) (angleDegrees * M_PI / 180.0)

// Converts radians to degrees.
#define radiansToDegrees(angleRadians) (angleRadians * 180.0 / M_PI)

typedef float      vec3[3];
static inline void vec3_add(vec3 r, vec3 const a, vec3 const b)
{
    int i;
    for (i = 0; i < 3; ++i)
        r[i] = a[i] + b[i];
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkan/linmath.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `main.cpp`

Source: cpp/5_Domain_Specific/simpleVulkan/main.cpp:29-47
```cpp
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "SineWaveSimulation.h"
#include "VulkanBaseApp.h"
#include "helper_cuda.h"
#include "linmath.h"

typedef float vec2[2];
std::string   execution_path;

#ifdef NDEBUG
#define ENABLE_VALIDATION (false)
#else
#define ENABLE_VALIDATION (true)
#endif

```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkan/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleVulkan/main.cpp:98-122
```cpp
        m_shaderFiles.push_back(std::make_pair(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_shader_path));
    }
    ~VulkanCudaSineWave()
    {
        // Make sure there's no pending work before we start tearing down
        // JP: `cudaStreamSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
        checkCudaErrors(cudaStreamSynchronize(m_stream));

#ifdef _VK_TIMELINE_SEMAPHORE
        if (m_vkTimelineSemaphore != VK_NULL_HANDLE) {
            // JP: `cudaDestroyExternalSemaphore`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
            checkCudaErrors(cudaDestroyExternalSemaphore(m_cudaTimelineSemaphore));
            vkDestroySemaphore(m_device, m_vkTimelineSemaphore, nullptr);
        }
#endif /* _VK_TIMELINE_SEMAPHORE */

        if (m_vkSignalSemaphore != VK_NULL_HANDLE) {
            checkCudaErrors(cudaDestroyExternalSemaphore(m_cudaSignalSemaphore));
            vkDestroySemaphore(m_device, m_vkSignalSemaphore, nullptr);
        }
        if (m_vkWaitSemaphore != VK_NULL_HANDLE) {
            checkCudaErrors(cudaDestroyExternalSemaphore(m_cudaWaitSemaphore));
            vkDestroySemaphore(m_device, m_vkWaitSemaphore, nullptr);
        }

```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkan/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleVulkan/main.cpp:520-532
```cpp
            m_lastTime  = currentTime;
        }
    }
};

int main(int argc, char **argv)
{
    execution_path = argv[0];
    VulkanCudaSineWave app((1ULL << 8ULL), (1ULL << 8ULL));
    app.init();
    app.mainLoop();
    return 0;
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkan/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `sinewave.frag`

Source: cpp/5_Domain_Specific/simpleVulkan/sinewave.frag:29-38
```glsl
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkan/sinewave.frag` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `sinewave.vert`

Source: cpp/5_Domain_Specific/simpleVulkan/sinewave.vert:28-43
```glsl
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
	mat4 modelViewProj;
} ubo;

layout(location = 0) in float height;
layout(location = 1) in vec2 xyPos;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = ubo.modelViewProj * vec4(xyPos.xy, height, 1.0f);
    fragColor = vec3(0.0f, (height + 0.5f), 0.0f);
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkan/sinewave.vert` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaStream_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaDestroyExternalSemaphore` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaGetDeviceProperties` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaSetDevice` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaStreamSynchronize` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaWaitExternalSemaphoresAsync` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaSignalExternalSemaphoresAsync` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGetDeviceCount` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDestroyExternalMemory` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaStreamCreateWithFlags` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaMem` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- CUDA Graph は一連の work を node と依存関係として再利用します。capture 対象と buffer lifetime を確認します。
- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
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
cmake --build build --target simpleVulkan
ctest --test-dir build -R simpleVulkan
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
- graph capture 後に buffer lifetime や node dependency が変わったことを見落とす。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaStream_t` の直前と直後で、どの memory/resource が有効になったかをメモする。
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
- [Performance](../../../docs_ja/themes/performance.md): memory traffic、occupancy、overlap、launch overhead、timing を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
