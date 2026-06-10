# simpleVulkanMMAP - Vulkan CUDA Interop PI Approximation - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates Vulkan CUDA Interop via cuMemMap APIs. CUDA exports buffers that Vulkan imports as vertex buffer. CUDA invokes kernels to operate on vertices and synchronizes with Vulkan through vulkan semaphores imported by CUDA. This sample depends on Vulkan SDK, GLFW3 libraries, for building this sample please refer to "Build_instructions.txt" provided in this sample's directory

cuMemMap IPC, MMAP, Graphics Interop, CUDA Vulkan Interop, Data Parallel Algorithms

Original README headings: `simpleVulkanMMAP - Vulkan CUDA Interop PI Approximation`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/simpleVulkanMMAP` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `simpleVulkanMMAP` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/simpleVulkanMMAP`.
> **日本語**
> この sample の目的は、`simpleVulkanMMAP` の小さな実装を通して Runtime, Driver, And NVRTC, CUDA Libraries, CUDA Graphs, Multi-GPU, P2P, And IPC, Streams And Events を具体的に追うことです。
>
> **学習メモ**
> 最初に `MonteCarloPi.cu, MonteCarloPi.h, VulkanBaseApp.cpp, VulkanBaseApp.h, VulkanCudaInterop.h` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- CUDA library components used by this sample, such as cuBLAS, cuFFT, cuSolver, NPP, CUB, or nvJPEG
- The device topology required by the README, such as multiple GPUs, peer access, IPC, MPI, or process support
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
- `MonteCarloPi.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `MonteCarloPi.h`: Host/device declarations, helper types, constants, or library wrappers.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `VulkanBaseApp.cpp`: Host-side setup, API calls, validation, and cleanup.
- `VulkanBaseApp.h`: Host/device declarations, helper types, constants, or library wrappers.
- `VulkanCudaInterop.h`: Host/device declarations, helper types, constants, or library wrappers.
- `frag.spv`: Supporting file used by `frag.spv`.
- `main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `montecarlo.frag`: Supporting file used by `montecarlo.frag`.
- `montecarlo.vert`: Supporting file used by `montecarlo.vert`.
- `vert.spv`: Supporting file used by `vert.spv`.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `MonteCarloPi.cu` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Create library handles, descriptors, plans, or workspaces before the library call.
- Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.
- Capture or build CUDA Graph nodes, instantiate the graph, then launch the executable graph.
- Enumerate devices, enable peer or IPC access, and record which device/process owns each resource.
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

- `MonteCarloPi.cu`: focus on `CUdeviceptr`, `cudaPositionHandle`, `cudaInCircleHandle`, `launch`, `CU_MEM_HANDLE_TYPE_WIN32`.
- `MonteCarloPi.h`: focus on `cudaStream_t`, `cudaDevice`, `curand`, `curand_kernel`, `cuMemMap`.
- `VulkanBaseApp.cpp`: focus on `CUDA`, `atomic`.
- `VulkanBaseApp.h`: focus on `atomic`, `CUDA`.
- `VulkanCudaInterop.h`: focus on `cudaDevice`, `cuDeviceGetAttribute`, `cudaInvalidDeviceId`, `CUDA_DRIVER_API`, `cudaGetDeviceCount`.
- `main.cpp`: focus on `cudaDevice`, `CUDA`, `cudaStream_t`, `cudaExternalSemaphore_t`, `cudaStreamSynchronize`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleVulkanMMAP LANGUAGES C CXX CUDA)

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

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `MonteCarloPi.cu`

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/MonteCarloPi.cu:33-51
```cuda
#include <algorithm>

#include "MonteCarloPi.h"
#define CUDA_DRIVER_API
#include <helper_cuda.h>
#include <iostream>

#define ROUND_UP_TO_GRANULARITY(x, n) (((x + n - 1) / n) * n)

// `ipcHandleTypeFlag` specifies the platform specific handle type this sample
// uses for importing and exporting memory allocation. On Linux this sample
// specifies the type as CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR meaning that
// file descriptors will be used. On Windows this sample specifies the type as
// CU_MEM_HANDLE_TYPE_WIN32 meaning that NT HANDLEs will be used. The
// ipcHandleTypeFlag variable is a convenience variable and is passed by value
// to individual requests.
#if defined(__linux__)
CUmemAllocationHandleType ipcHandleTypeFlag = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/MonteCarloPi.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/MonteCarloPi.cu:85-107
```cuda
                                   unsigned int numPoints,
                                   float        time)
{
    // JP: `gridDim`, `blockDim`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    const size_t stride = gridDim.x * blockDim.x;
    size_t       tid    = blockIdx.x * blockDim.x + threadIdx.x;
    float        count  = 0.0f;

    // JP: `curandState`: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
    curandState rgnState;
    curand_init((unsigned long long)time, tid, 0, &rgnState);

    for (; tid < numPoints; tid += stride) {
        float x          = curand_uniform(&rgnState);
        float y          = curand_uniform(&rgnState);
        x                = (2.0f * x) - 1.0f;
        y                = (2.0f * y) - 1.0f;
        xyVector[tid][0] = x;
        xyVector[tid][1] = y;

        // Compute the distance of this point form the center(0, 0)
        float dist = sqrtf((x * x) + (y * y));

```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/MonteCarloPi.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/MonteCarloPi.cu:123-142
```cuda
}

MonteCarloPiSimulation::~MonteCarloPiSimulation()
{
    if (m_numPointsInCircle) {
        // JP: `cudaFree`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。 ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        checkCudaErrors(cudaFree(m_numPointsInCircle));
        m_numPointsInCircle = nullptr;
    }
    if (m_hostNumPointsInCircle) {
        // JP: `cudaFreeHost`: page-locked host memory は DMA/async copy を安定させます。通常の free ではなく対応する CUDA API で解放します。
        checkCudaErrors(cudaFreeHost(m_hostNumPointsInCircle));
        m_hostNumPointsInCircle = nullptr;
    }

    cleanupSimulationAllocations();
}

// JP: `cudaDevice`, `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
void MonteCarloPiSimulation::initSimulation(int cudaDevice, cudaStream_t stream)
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/MonteCarloPi.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/MonteCarloPi.cu:156-175
```cuda
    checkCudaErrors(cudaMallocHost((float **)&m_hostNumPointsInCircle, sizeof(*m_hostNumPointsInCircle)));
}

void MonteCarloPiSimulation::stepSimulation(float time, cudaStream_t stream)
{
    // JP: `cudaMemsetAsync`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemsetAsync(m_numPointsInCircle, 0, sizeof(*m_numPointsInCircle), stream));

    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    monte_carlo_kernel<<<m_blocks, m_threads, 0, stream>>>(
        m_xyVector, m_pointsInsideCircle, m_numPointsInCircle, m_numPoints, time);
    getLastCudaError("Failed to launch CUDA simulation");

    // JP: この連続する anchor 群では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpyAsync(
        m_hostNumPointsInCircle, m_numPointsInCircle, sizeof(*m_numPointsInCircle), cudaMemcpyDeviceToHost, stream));

    // Queue up a stream callback to compute and print the PI value.
    checkCudaErrors(cudaLaunchHostFunc(stream, this->computePiCallback, (void *)this));
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/MonteCarloPi.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `MonteCarloPi.h`

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/MonteCarloPi.h:29-96
```cpp
#pragma once
#ifndef __PISIM_H__
#define __PISIM_H__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

#include "helper_multiprocess.h"

typedef float vec2[2];

class MonteCarloPiSimulation
{
    size_t m_numPoints;

    // Pointers to Cuda allocated buffers which are imported and used by vulkan as
    // vertex buffer
    vec2  *m_xyVector;
    float *m_pointsInsideCircle;

    // Pointers to device and host allocated memories storing number of points
    // that are inside the unit circle
    float *m_numPointsInCircle;
    float *m_hostNumPointsInCircle;

    int m_blocks, m_threads;

    // Total size of allocations created by cuMemMap Apis. This size is the sum of
    // sizes of m_xyVector and m_pointsInsideCircle buffers.
    size_t m_totalAllocationSize;

    // Shareable Handles(a file descriptor on Linux and NT Handle on Windows),
    // used for sharing cuda
    // allocated memory with Vulkan
    ShareableHandle m_posShareableHandle, m_inCircleShareableHandle;

    // Cuda Device corresponding to the Vulkan Physical device
    int m_cudaDevice;

    // Track and accumulate total points that have been simulated since start of
    // the sample. The idea is to get a closer approximation to PI with time.
    size_t m_totalPointsInsideCircle;
    size_t m_totalPointsSimulated;

    void setupSimulationAllocations();
    void cleanupSimulationAllocations();
    void getIdealExecutionConfiguration();

public:
    MonteCarloPiSimulation(size_t num_points);
    ~MonteCarloPiSimulation();
    // JP: `cudaDevice`, `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    void        initSimulation(int cudaDevice, cudaStream_t stream = 0);
    void        stepSimulation(float time, cudaStream_t stream = 0);
    static void computePiCallback(void *args);

    size_t getNumPoints() const { return m_numPoints; }

    float getNumPointsInCircle() const { return *m_hostNumPointsInCircle; }

    ShareableHandle &getPositionShareableHandle() { return m_posShareableHandle; }
    ShareableHandle &getInCircleShareableHandle() { return m_inCircleShareableHandle; }
};

#endif // __PISIM_H__
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/MonteCarloPi.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `VulkanBaseApp.cpp`

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/VulkanBaseApp.cpp:35-53
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

#include "VulkanCudaInterop.h"

#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <GLFW/glfw3.h>

#ifdef _WIN64
#include <VersionHelpers.h>
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/VulkanBaseApp.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/VulkanBaseApp.cpp:274-293
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

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/VulkanBaseApp.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/VulkanBaseApp.cpp:318-337
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

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/VulkanBaseApp.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `VulkanBaseApp.h`

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/VulkanBaseApp.h:29-47
```cpp
#pragma once
#ifndef __VULKANBASEAPP_H__
#define __VULKANBASEAPP_H__

#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#ifdef _WIN64
#define NOMINMAX
// Add windows.h to the include path firstly as dependency for other Windows headers
#include <windows.h>
// Add other Windows headers
#include <vulkan/vulkan_win32.h>
#endif /* _WIN64 */

struct GLFWwindow;

class VulkanBaseApp
{
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/VulkanBaseApp.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `VulkanCudaInterop.h`

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/VulkanCudaInterop.h:29-80
```cpp
#pragma once
#ifndef __VKCUDA_H__
#define __VKCUDA_H__

#include <cuda_runtime_api.h>

#include "cuda.h"
#define CUDA_DRIVER_API
#include <helper_cuda.h>

bool isDeviceCompatible(void *Uuid, size_t size)
{
    int cudaDevice = cudaInvalidDeviceId;
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp devProp = {};
        checkCudaErrors(cudaGetDeviceProperties(&devProp, i));
        if (!memcmp(&devProp.uuid, Uuid, size)) {
            cudaDevice = i;
            break;
        }
    }
    if (cudaDevice == cudaInvalidDeviceId) {
        return false;
    }

    int deviceSupportsHandle = 0;
    int attributeVal         = 0;
    int deviceComputeMode    = 0;

    // JP: `cuDeviceGetAttribute`, `cudaDevice`: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
    checkCudaErrors(cuDeviceGetAttribute(&deviceComputeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cudaDevice));
    checkCudaErrors(
        cuDeviceGetAttribute(&attributeVal, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, cudaDevice));

#if defined(__linux__)
    checkCudaErrors(cuDeviceGetAttribute(
        &deviceSupportsHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, cudaDevice));
#else
    checkCudaErrors(cuDeviceGetAttribute(
        &deviceSupportsHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, cudaDevice));
#endif

    if ((deviceComputeMode != CU_COMPUTEMODE_DEFAULT) || !attributeVal || !deviceSupportsHandle) {
        return false;
    }
    return true;
}

#endif // __VKCUDA_H__
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/VulkanCudaInterop.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `main.cpp`

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/main.cpp:36-54
```cpp
#include <algorithm>
#include <chrono>
#include <cuda.h>
#include <iomanip>
#include <iostream>

#include "MonteCarloPi.h"
#include "VulkanBaseApp.h"
#include "helper_cuda.h"
#include "helper_multiprocess.h"

// #define DEBUG
#ifdef NDEBUG
#define ENABLE_VALIDATION (false)
#else
#define ENABLE_VALIDATION (true)
#endif

#define NUM_SIMULATION_POINTS 50000
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/main.cpp:99-120
```cpp

    ~VulkanCudaPi()
    {
        if (m_stream) {
            // Make sure there's no pending work before we start tearing down
            // JP: `cudaStreamSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
            checkCudaErrors(cudaStreamSynchronize(m_stream));
            // JP: `cudaStreamDestroy`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
            checkCudaErrors(cudaStreamDestroy(m_stream));
        }

        if (m_vkSignalSemaphore != VK_NULL_HANDLE) {
            // JP: この連続する anchor 群では CUDA resource lifetime end です。未完了 work が残っていないか確認し、確保/作成/登録と対応する API で閉じます。
            checkCudaErrors(cudaDestroyExternalSemaphore(m_cudaSignalSemaphore));
            vkDestroySemaphore(m_device, m_vkSignalSemaphore, nullptr);
        }
        if (m_vkWaitSemaphore != VK_NULL_HANDLE) {
            // JP: wait semaphore 側の CUDA external semaphore も Vulkan semaphore と対で閉じます。signal 側と同じ cleanup pair として lifetime を確認します。
            checkCudaErrors(cudaDestroyExternalSemaphore(m_cudaWaitSemaphore));
            vkDestroySemaphore(m_device, m_vkWaitSemaphore, nullptr);
        }
        if (m_xyPositionBuffer != VK_NULL_HANDLE) {
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/main.cpp:213-232
```cpp
        if (cudaDevice == cudaInvalidDeviceId) {
            throw std::runtime_error("No Suitable device found!");
        }

        // On the corresponding cuda device, create the cuda stream we'll using
        checkCudaErrors(cudaSetDevice(cudaDevice));
        // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
        checkCudaErrors(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
        m_sim.initSimulation(cudaDevice, m_stream);

        importExternalBuffer((void *)(uintptr_t)m_sim.getPositionShareableHandle(),
                             getDefaultMemHandleType(),
                             nVerts * sizeof(vec2),
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             m_xyPositionBuffer,
                             m_xyPositionMemory);

        importExternalBuffer((void *)(uintptr_t)m_sim.getInCircleShareableHandle(),
                             getDefaultMemHandleType(),
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `montecarlo.frag`

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/montecarlo.frag:28-37
```glsl
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/montecarlo.frag` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `montecarlo.vert`

Source: cpp/5_Domain_Specific/simpleVulkanMMAP/montecarlo.vert:28-36
```glsl
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    float frame;
} ubo;

layout(location = 0) in float pointInsideCircle;
layout(location = 1) in vec2 xyPos;
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleVulkanMMAP/montecarlo.vert` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaDevice` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuMemMap` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaStream_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `CUdeviceptr` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaPositionHandle` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaInCircleHandle` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuMemCreate` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuMemRelease` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuDeviceGetAttribute` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaFreeHost` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaMemsetAsync` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaSetDevice` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDeviceGetAttribute` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

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
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。
- 同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Sample-Specific Notes

- multi-GPU/P2P/IPC 系では、どの process/thread/device が resource を所有しているかを先に分けると読みやすくなります。
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
cmake --build build --target simpleVulkanMMAP
ctest --test-dir build -R simpleVulkanMMAP
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
- leading dimension、stride、descriptor、workspace size を host 配列の見た目だけで判断する。
- JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。
- peer access が有効な device pair と、単に複数 GPU が存在することを混同する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaDevice` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
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

- [Runtime, Driver, And NVRTC](../../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [CUDA Libraries](../../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。
- [CUDA Graphs](../../../docs_ja/themes/graphs.md): capture、node dependency、replay、graph update を読むための基礎です。
- [Multi-GPU, P2P, And IPC](../../../docs_ja/themes/multi_gpu_p2p_ipc.md): device topology、peer access、IPC handle、multi-process 境界を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Performance](../../../docs_ja/themes/performance.md): memory traffic、occupancy、overlap、launch overhead、timing を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
