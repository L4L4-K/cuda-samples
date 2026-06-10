# vectorAddMMAP - Vector Addition cuMemMap - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample replaces the device allocation in the vectorAddDrv sample with cuMemMap-ed allocations.  This sample demonstrates that the cuMemMap api allows the user to specify the physical properties of their memory while retaining the contiguous nature of their access, thus not requiring a change in their program structure.

CUDA Driver API, Vector Addition, MMAP

Original README headings: `vectorAddMMAP - Vector Addition cuMemMap`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/0_Introduction/vectorAddMMAP` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `vectorAddMMAP` as a focused example of the CUDA concepts used in `cpp/0_Introduction/vectorAddMMAP`.
> **日本語**
> この sample の目的は、`vectorAddMMAP` の小さな実装を通して Runtime, Driver, And NVRTC, Multi-GPU, P2P, And IPC, Tensor Cores And WMMA, Memory, Kernel Launch And Indexing を具体的に追うことです。
>
> **学習メモ**
> 最初に `multidevicealloc_memmap.cpp, multidevicealloc_memmap.hpp, vectorAddMMAP.cpp, vectorAdd_kernel.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- The device topology required by the README, such as multiple GPUs, peer access, IPC, MPI, or process support

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
- `multidevicealloc_memmap.cpp`: Host-side setup, API calls, validation, and cleanup.
- `multidevicealloc_memmap.hpp`: Host/device declarations, helper types, constants, or library wrappers.
- `vectorAddMMAP.cpp`: Host-side setup, API calls, validation, and cleanup.
- `vectorAdd_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `multidevicealloc_memmap.cpp` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.
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

- `multidevicealloc_memmap.cpp`: focus on `CUDA_SUCCESS`, `CUresult`, `cuMemRelease`, `CUdeviceptr`, `CUdevice`.
- `multidevicealloc_memmap.hpp`: focus on `CUresult`, `CUdeviceptr`, `CUdevice`, `cuMemGetAllocationGranularity`, `cuMemCreate`.
- `vectorAddMMAP.cpp`: focus on `cuDevice`, `CUdevice`, `cuModule`, `CUDA`, `cuContext`.
- `vectorAdd_kernel.cu`: focus on `blockDim`, `blockIdx`, `threadIdx`, `launch`, `Device`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/0_Introduction/vectorAddMMAP/CMakeLists.txt:1-70
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(vectorAddMMAP LANGUAGES C CXX CUDA)

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

# Source file
# Add target for vectorAddMMAP
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    message(STATUS "Will not build sample vectorAddMMAP - not supported on aarch64")
else()
    add_executable(vectorAddMMAP vectorAddMMAP.cpp multidevicealloc_memmap.cpp)

    target_compile_options(vectorAddMMAP PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(vectorAddMMAP PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(vectorAddMMAP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(vectorAddMMAP PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(vectorAddMMAP PUBLIC
        CUDA::cuda_driver
    )

    set(CUDA_FATBIN_FILE "${CMAKE_CURRENT_BINARY_DIR}/vectorAdd_kernel64.fatbin")
    set(CUDA_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/vectorAdd_kernel.cu")

    # Construct GENCODE_FLAGS explicitly from CUDA architectures
    set(GENCODE_FLAGS "")
    foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
        list(APPEND GENCODE_FLAGS "-gencode=arch=compute_${arch},code=sm_${arch}")
    endforeach()

    add_custom_command(
        OUTPUT ${CUDA_FATBIN_FILE}
        COMMAND ${CMAKE_CUDA_COMPILER} ${INCLUDES} ${ALL_CCFLAGS} -Wno-deprecated-gpu-targets  ${GENCODE_FLAGS} -o ${CUDA_FATBIN_FILE} -fatbin ${CUDA_KERNEL_SOURCE}
        DEPENDS ${CUDA_KERNEL_SOURCE}
        COMMENT "Building CUDA fatbin: ${CUDA_FATBIN_FILE}"
    )

    # Create a dummy target for fatbin generation
    add_custom_target(generate_fatbin_vectorAddMMAP ALL DEPENDS ${CUDA_FATBIN_FILE})

    # Ensure matrixMulDrv depends on the fatbin
    add_dependencies(vectorAddMMAP generate_fatbin_vectorAddMMAP)
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/0_Introduction/vectorAddMMAP/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `multidevicealloc_memmap.cpp`

Source: cpp/0_Introduction/vectorAddMMAP/multidevicealloc_memmap.cpp:29-62
```cpp
#include "multidevicealloc_memmap.hpp"

static size_t round_up(size_t x, size_t y) { return ((x + y - 1) / y) * y; }

// JP: driver_api: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
CUresult simpleMallocMultiDeviceMmap(CUdeviceptr                 *dptr,
                                     size_t                      *allocationSize,
                                     size_t                       size,
                                     const std::vector<CUdevice> &residentDevices,
                                     const std::vector<CUdevice> &mappingDevices,
                                     size_t                       align)
{
    CUresult status          = CUDA_SUCCESS;
    size_t   min_granularity = 0;
    size_t   stripeSize;

    // Setup the properties common for all the chunks
    // The allocations will be device pinned memory.
    // This property structure describes the physical location where the memory
    // will be allocated via cuMemCreate allong with additional properties In this
    // case, the allocation will be pinnded device memory local to a given device.
    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;

    // Get the minimum granularity needed for the resident devices
    // (the max of the minimum granularity of each participating device)
    for (int idx = 0; idx < residentDevices.size(); idx++) {
        size_t granularity = 0;

        // get the minnimum granularity for residentDevices[idx]
        prop.location.id = residentDevices[idx];
        status           = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (status != CUDA_SUCCESS) {
```

> JP: この抜粋は `cpp/0_Introduction/vectorAddMMAP/multidevicealloc_memmap.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `multidevicealloc_memmap.hpp`

Source: cpp/0_Introduction/vectorAddMMAP/multidevicealloc_memmap.hpp:29-81
```cpp
#pragma once
#include <cuda.h>
#include <vector>

////////////////////////////////////////////////////////////////////////////
//! Allocate virtually contiguous memory backed on separate devices
//! @return CUresult error code on failure.
//! @param[out] dptr            Virtual address reserved for allocation
//! @param[out] allocationSize  Actual amount of virtual address space reserved.
//!                             AllocationSize is needed in the free operation.
//! @param[in] size             The minimum size to allocate (will be rounded up
//! to accomodate
//!                             required granularity).
//! @param[in] residentDevices  Specifies what devices the allocation should be
//! striped across.
//! @param[in] mappingDevices   Specifies what devices need to read/write to the
//! allocation.
//! @align                      Additional allignment requirement if desired.
//! @note       The VA mappings will look like the following:
//!
//!     v-stripeSize-v                v-rounding -v
//!     +-----------------------------------------+
//!     |      D1     |      D2     |      D3     |
//!     +-----------------------------------------+
//!     ^-- dptr                      ^-- dptr + size
//!
//! Each device in the residentDevices list will get an equal sized stripe.
//! Excess memory allocated will be  that meets the minimum
//! granularity requirements of all the devices.
//!
//! @note uses cuMemGetAllocationGranularity cuMemCreate cuMemMap and
//! cuMemSetAccess
//!   function calls to organize the va space
//!
//! @note uses cuMemRelease to release the allocationHandle.  The allocation
//! handle
//!   is not needed after its mappings are set up.
////////////////////////////////////////////////////////////////////////////
// JP: driver_api: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
CUresult simpleMallocMultiDeviceMmap(CUdeviceptr                 *dptr,
                                     size_t                      *allocationSize,
                                     size_t                       size,
                                     const std::vector<CUdevice> &residentDevices,
                                     const std::vector<CUdevice> &mappingDevices,
                                     size_t                       align = 0);

////////////////////////////////////////////////////////////////////////////
//! Frees resources allocated by simpleMallocMultiDeviceMmap
//! @CUresult CUresult error code on failure.
//! @param[in] dptr  Virtual address reserved by simpleMallocMultiDeviceMmap
//! @param[in] size  allocationSize returned by simpleMallocMultiDeviceMmap
////////////////////////////////////////////////////////////////////////////
CUresult simpleFreeMultiDeviceMmap(CUdeviceptr dptr, size_t size);
```

> JP: この抜粋は `cpp/0_Introduction/vectorAddMMAP/multidevicealloc_memmap.hpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `vectorAddMMAP.cpp`

Source: cpp/0_Introduction/vectorAddMMAP/vectorAddMMAP.cpp:40-58
```cpp
#include <cstring>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

// includes, project
#include <helper_cuda_drvapi.h>
#include <helper_functions.h>

// includes, CUDA
#include <builtin_types.h>

#include "multidevicealloc_memmap.hpp"

using namespace std;

// Variables
// JP: `cuDevice`: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
```

> JP: この抜粋は `cpp/0_Introduction/vectorAddMMAP/vectorAddMMAP.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/0_Introduction/vectorAddMMAP/vectorAddMMAP.cpp:114-133
```cpp
    }
    return backingDevices;
}

// Host code
int main(int argc, char **argv)
{
    printf("Vector Addition (Driver API)\n");
    int    N            = 50000;
    size_t size         = N * sizeof(float);
    int    attributeVal = 0;

    // Initialize
    // JP: この anchor では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
    checkCudaErrors(cuInit(0));

    cuDevice = findCudaDeviceDRV(argc, (const char **)argv);

    // Check that the selected device supports virtual address management
    checkCudaErrors(
```

> JP: この抜粋は `cpp/0_Introduction/vectorAddMMAP/vectorAddMMAP.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/0_Introduction/vectorAddMMAP/vectorAddMMAP.cpp:174-193
```cpp

    // Get function handle from module
    checkCudaErrors(cuModuleGetFunction(&vecAdd_kernel, cuModule, "VecAdd_kernel"));

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);


    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // Allocate vectors in device memory
    // note that a call to cuCtxEnablePeerAccess is not needed even though
    // the backing devices and mapping device are not the same.
    // This is because the cuMemSetAccess call explicitly specifies
    // the cross device mapping.
    // cuMemSetAccess is still subject to the constraints of cuDeviceCanAccessPeer
```

> JP: この抜粋は `cpp/0_Introduction/vectorAddMMAP/vectorAddMMAP.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/0_Introduction/vectorAddMMAP/vectorAddMMAP.cpp:208-263
```cpp
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    void *args[] = {&d_A, &d_B, &d_C, &N};

    // Launch the CUDA kernel
    // JP: `cuLaunchKernel`: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    checkCudaErrors(cuLaunchKernel(vecAdd_kernel, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, NULL, args, NULL));

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cuMemcpyDtoH(h_C, d_C, size));

    // Verify result
    int i;

    for (i = 0; i < N; ++i) {
        float sum = h_A[i] + h_B[i];

        if (fabs(h_C[i] - sum) > 1e-7f) {
            break;
        }
    }

    CleanupNoFailure();
    // JP: validation: GPU result を CPU/reference と比較する検証地点です。失敗時は transfer、indexing、sync の順に疑います。
    printf("%s\n", (i == N) ? "Result = PASS" : "Result = FAIL");

    exit((i == N) ? EXIT_SUCCESS : EXIT_FAILURE);
}

int CleanupNoFailure()
{
    // Free device memory
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_A, allocationSize));
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_B, allocationSize));
    checkCudaErrors(simpleFreeMultiDeviceMmap(d_C, allocationSize));

    // Free host memory
    if (h_A) {
        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(h_A);
    }

    if (h_B) {
        free(h_B);
    }

    if (h_C) {
        free(h_C);
    }

    // JP: この連続する anchor 群では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
    checkCudaErrors(cuModuleUnload(cuModule));
    checkCudaErrors(cuCtxDestroy(cuContext));

```

> JP: この抜粋は `cpp/0_Introduction/vectorAddMMAP/vectorAddMMAP.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `vectorAdd_kernel.cu`

Source: cpp/0_Introduction/vectorAddMMAP/vectorAdd_kernel.cu:38-45
```cuda
extern "C" __global__ void VecAdd_kernel(const float *A, const float *B, float *C, int N)
{
    // JP: `blockDim`, `blockIdx`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}
```

> JP: この抜粋は `cpp/0_Introduction/vectorAddMMAP/vectorAdd_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cuDevice` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUDA_SUCCESS` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuMemMap` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUresult` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUdevice` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUdeviceptr` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuMemRelease` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuMemSetAccess` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cuMemCreate` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuMemGetAllocationGranularity` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuDeviceCanAccessPeer` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuMemcpyHtoD` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cuMemAddressReserve` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuModule` | Driver API の handle 境界です。context/module/function と error code を追います。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- multi-GPU sample では、device 選択、peer capability、context/IPC handle の寿命を分けて読みます。
- Runtime/Driver/NVRTC sample では、compile/load した module と launch する kernel 名の対応が重要です。
- Tensor Core sample では tile size、alignment、precision、accumulator の型が正しさと性能を決めます。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Sample-Specific Notes

- vectorAdd では「1 thread が 1 element を担当する」形が最小単位です。rounded-up grid と `i < numElements` の境界チェックを必ず一緒に読みます。
- host 配列、device 配列、copy direction、kernel launch、D2H copy、validation、free の順序が CUDA Runtime API の基本形です。
- multi-GPU/P2P/IPC 系では、どの process/thread/device が resource を所有しているかを先に分けると読みやすくなります。

> **日本語**
> この section は同じ template ではなく、sample 名、path、検出した API から読みどころを絞っています。
>
> **学習メモ**
> 似た名前の sample は Runtime 版、Driver 版、NVRTC 版、library 版の違いを比較すると学習効果が高くなります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target vectorAddMMAP
ctest --test-dir build -R vectorAddMMAP
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

Run the sample as documented and compare its output with the original README, validation message, generated file, or reference result.
> **日本語**
> 期待結果は英語の出力文字列、README の validation、生成 file、または reference result と照合します。
>
> **学習メモ**
> `PASS`、`Test passed`、error code、timing label などの出力文字列は翻訳せず、source と同じ表記で確認します。

## Common Mistakes

- API 名や target 名を翻訳してしまい、README や build command と対応できなくなる。
- allocation size を byte で渡す API と element count で考える loop を混同する。
- kernel launch が非同期であることを忘れ、同期前の結果を host 側で読んでしまう。
- JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。
- peer access が有効な device pair と、単に複数 GPU が存在することを混同する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cuDevice` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- device ごとの ownership と、peer/IPC で共有される resource を分けて書く。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Runtime, Driver, And NVRTC](../../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [Multi-GPU, P2P, And IPC](../../../docs_ja/themes/multi_gpu_p2p_ipc.md): device topology、peer access、IPC handle、multi-process 境界を読むための基礎です。
- [Tensor Cores And WMMA](../../../docs_ja/themes/tensor_cores_wmma.md): Tensor Core、tile、precision、fragment の制約を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
