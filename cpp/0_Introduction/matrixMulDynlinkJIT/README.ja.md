# matrixMulDynlinkJIT - Matrix Multiplication (CUDA Driver API version with Dynamic Linking Version) - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample revisits matrix multiplication using the CUDA driver API. It demonstrates how to link to CUDA driver at runtime and how to use JIT (just-in-time) compilation from PTX code. It has been written for clarity of exposition to illustrate various CUDA programming principles, not with the goal of providing the most performant generic kernel for matrix multiplication. CUBLAS provides high-performance matrix multiplication.

CUDA Driver API, CUDA Dynamically Linked Library

Original README headings: `matrixMulDynlinkJIT - Matrix Multiplication (CUDA Driver API version with Dynamic Linking Version)`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/0_Introduction/matrixMulDynlinkJIT` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `matrixMulDynlinkJIT` as a focused example of the CUDA concepts used in `cpp/0_Introduction/matrixMulDynlinkJIT`.
> **日本語**
> この sample の目的は、`matrixMulDynlinkJIT` の小さな実装を通して Runtime, Driver, And NVRTC, CUDA Libraries, CUDA Graphs, Multi-GPU, P2P, And IPC, Shared Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `cuda_drvapi_dynlink.c, cuda_drvapi_dynlink.h, cuda_drvapi_dynlink_cuda.h, ptx2c.py, helper_cuda_drvapi.h` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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

- `.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `cuda_drvapi_dynlink.c`: Host-side setup, API calls, validation, and cleanup.
- `cuda_drvapi_dynlink.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cuda_drvapi_dynlink_cuda.h`: Host/device declarations, helper types, constants, or library wrappers.
- `extras/README.TXT`: Input, reference, generated-data description, or documentation used by the sample.
- `extras/matrixMul_kernel_32.ptx`: Supporting file used by `extras/matrixMul_kernel_32.ptx`.
- `extras/matrixMul_kernel_64.ptx`: Supporting file used by `extras/matrixMul_kernel_64.ptx`.
- `extras/ptx2c.py`: Python entry point or helper using CUDA Python, CuPy, framework interop, or subprocess logic.
- `helper_cuda_drvapi.h`: Host/device declarations, helper types, constants, or library wrappers.
- `matrixMul.h`: Host/device declarations, helper types, constants, or library wrappers.
- `matrixMulDynlinkJIT.cpp`: Host-side setup, API calls, validation, and cleanup.
- `matrixMul_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `matrixMul_kernel_32_ptxdump.c`: Host-side setup, API calls, validation, and cleanup.
- `matrixMul_kernel_32_ptxdump.h`: Host/device declarations, helper types, constants, or library wrappers.
- `matrixMul_kernel_64_ptxdump.c`: Host-side setup, API calls, validation, and cleanup.
- `matrixMul_kernel_64_ptxdump.h`: Host/device declarations, helper types, constants, or library wrappers.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `cuda_drvapi_dynlink.c` first and locate the host-side setup or Python entry point.
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

- `cuda_drvapi_dynlink.c`: focus on `CUDA`, `CUDADRIVER`, `CUresult`, `CUDA_ERROR_UNKNOWN`, `CUDA_SUCCESS`.
- `cuda_drvapi_dynlink.h`: focus on control flow and helper functions.
- `cuda_drvapi_dynlink_cuda.h`: focus on `CUresult`, `CUDAAPI`, `CUdeviceptr`, `CUDA`, `CUarray`.
- `extras/ptx2c.py`: focus on `CUDA`, `CUmodule`.
- `helper_cuda_drvapi.h`: focus on `CUDA`, `cuDevice`, `cuDeviceGetAttribute`, `Device`, `CUdevice`.
- `matrixMul.h`: focus on control flow and helper functions.
- `matrixMulDynlinkJIT.cpp`: focus on `CUDA`, `cuDevice`, `cuCtxDestroy`, `CUDA_SUCCESS`, `cuMemAlloc`.
- `matrixMul_gold.cpp`: focus on control flow and helper functions.
- `matrixMul_kernel_32_ptxdump.c`: focus on control flow and helper functions.
- `matrixMul_kernel_32_ptxdump.h`: focus on control flow and helper functions.
- Additional source files: 2 more support files. Use the same setup/work/sync/cleanup lens.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/0_Introduction/matrixMulDynlinkJIT/CMakeLists.txt:1-54
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(matrixMulDynlinkJIT LANGUAGES C CXX CUDA)

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
# Add target for matrixMulDynlinkJIT
add_executable(matrixMulDynlinkJIT cuda_drvapi_dynlink.c matrixMulDynlinkJIT.cpp matrixMul_gold.cpp matrixMul_kernel_32_ptxdump.c matrixMul_kernel_64_ptxdump.c)

target_compile_options(matrixMulDynlinkJIT PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(matrixMulDynlinkJIT PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(matrixMulDynlinkJIT PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
)

# Only add -no-pie for GCC or Clang
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")
endif()

target_link_libraries(matrixMulDynlinkJIT PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    target_link_libraries(matrixMulDynlinkJIT PUBLIC dl)
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDynlinkJIT/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `extras/matrixMul_kernel_32.ptx`

Source: cpp/0_Introduction/matrixMulDynlinkJIT/extras/matrixMul_kernel_32.ptx:1-19
```ptx
	.version 1.4
	.target sm_20, map_f64_to_f32
	// compiled with C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0\\bin/../open64/lib//be.exe
	// nvopencc 4.0 built on 2011-02-21

	//-----------------------------------------------------------
	// Compiling C:/Users/EYOUNG~1.COM/AppData/Local/Temp/tmpxft_000014c0_00000000-11_matrixMul_kernel.cpp3.i (C:/Users/EYOUNG~1.COM/AppData/Local/Temp/ccBI#.a04000)
	//-----------------------------------------------------------

	//-----------------------------------------------------------
	// Options:
	//-----------------------------------------------------------
	//  Target:ptx, ISA:sm_20, Endian:little, Pointer Size:32
	//  -O3	(Optimization level)
	//  -g0	(Debug level)
	//  -m2	(Report advisories)
	//-----------------------------------------------------------

	.shared .align 4 .b8 __cuda_local_var_87382_38_non_const_As__6[1024];
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDynlinkJIT/extras/matrixMul_kernel_32.ptx` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `extras/matrixMul_kernel_64.ptx`

Source: cpp/0_Introduction/matrixMulDynlinkJIT/extras/matrixMul_kernel_64.ptx:1-19
```ptx
	.version 1.4
	.target sm_20, map_f64_to_f32
	// compiled with C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0\\bin/../open64/lib//be.exe
	// nvopencc 4.0 built on 2011-02-21

	//-----------------------------------------------------------
	// Compiling C:/Users/EYOUNG~1.COM/AppData/Local/Temp/tmpxft_00000c2c_00000000-11_matrixMul_kernel.cpp3.i (C:/Users/EYOUNG~1.COM/AppData/Local/Temp/ccBI#.a04524)
	//-----------------------------------------------------------

	//-----------------------------------------------------------
	// Options:
	//-----------------------------------------------------------
	//  Target:ptx, ISA:sm_20, Endian:little, Pointer Size:64
	//  -O3	(Optimization level)
	//  -g0	(Debug level)
	//  -m2	(Report advisories)
	//-----------------------------------------------------------

	.shared .align 4 .b8 __cuda_local_var_87382_38_non_const_As__6[1024];
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDynlinkJIT/extras/matrixMul_kernel_64.ptx` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `extras/ptx2c.py`

Source: cpp/0_Introduction/matrixMulDynlinkJIT/extras/ptx2c.py:2-20
```python
# JP: この file では Python から CUDA work を起動する境界、stream/event による非同期実行と同期、Runtime/Driver/NVRTC の境界 を確認します。英語の識別子/API/出力文字列は保持します。

from string import *
import os, getopt, sys, platform

g_Header = '''/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDynlinkJIT/extras/ptx2c.py` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `helper_cuda_drvapi.h`

Source: cpp/0_Introduction/matrixMulDynlinkJIT/helper_cuda_drvapi.h:15-33
```cpp
#ifndef HELPER_CUDA_DRVAPI_H
#define HELPER_CUDA_DRVAPI_H

#include <helper_string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#ifndef HELPER_CUDA_DRVAPI_H
inline int ftoi(float value) { return (value >= 0 ? static_cast<int>(value + 0.5) : static_cast<int>(value - 0.5)); }
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDynlinkJIT/helper_cuda_drvapi.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/0_Introduction/matrixMulDynlinkJIT/helper_cuda_drvapi.h:137-156
```cpp
inline int gpuDeviceInitDRV(int ARGC, const char **ARGV)
{
    int cuDevice    = 0;
    int deviceCount = 0;
    // JP: この anchor では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
    checkCudaErrors(cuInit(0, __CUDA_API_VERSION));

    checkCudaErrors(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
        exit(EXIT_FAILURE);
    }

    int dev = 0;
    dev     = getCmdLineArgumentInt(ARGC, (const char **)ARGV, "device=");

    if (dev < 0) {
        dev = 0;
    }
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDynlinkJIT/helper_cuda_drvapi.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `matrixMul.h`

Source: cpp/0_Introduction/matrixMulDynlinkJIT/matrixMul.h:30-42
```cpp
#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA (4 * block_size) // Matrix A width
#define HA (6 * block_size) // Matrix A height
#define WB (4 * block_size) // Matrix B width
#define HB WA               // Matrix B height
#define WC WB               // Matrix C width
#define HC HA               // Matrix C height

#endif // _MATRIXMUL_H_
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDynlinkJIT/matrixMul.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `matrixMulDynlinkJIT.cpp`

Source: cpp/0_Introduction/matrixMulDynlinkJIT/matrixMulDynlinkJIT.cpp:47-65
```cpp
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, CUDA
#include "cuda_drvapi_dynlink.h"
#include "helper_cuda_drvapi.h"

// includes, project
#include "matrixMul.h"
#include "matrixMul_kernel_32_ptxdump.h"
#include "matrixMul_kernel_64_ptxdump.h"

extern "C" void computeGold(float *, const float *, const float *, unsigned int, unsigned int, unsigned int);

#if defined _MSC_VER
#pragma warning(disable : 4312)
#endif
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDynlinkJIT/matrixMulDynlinkJIT.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/0_Introduction/matrixMulDynlinkJIT/matrixMulDynlinkJIT.cpp:97-116
```cpp
    CUfunction cuFunction;
    int        major, minor, block_size, devID = 0;
    char       deviceName[256];

    // link to cuda driver dynamically
    checkCudaErrors(cuInit(0, __CUDA_API_VERSION));

    // This assumes that the user is attempting to specify a explicit device -device=n
    if (argc > 1) {
        bool bFound = false;

        for (int param = 0; param < argc; param++) {
            if (!strncmp(argv[param], "-device", 7)) {
                int i = (int)strlen(argv[1]);

                while (argv[1][i] != '=') {
                    i--;
                }

                devID  = atoi(&argv[1][++i]);
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDynlinkJIT/matrixMulDynlinkJIT.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/0_Introduction/matrixMulDynlinkJIT/matrixMulDynlinkJIT.cpp:166-185
```cpp

    // setup JIT compilation options and perform compilation
    {
        // in this branch we use compilation with parameters
        const unsigned int jitNumOptions = 3;
        CUjit_option      *jitOptions    = new CUjit_option[jitNumOptions];
        void             **jitOptVals    = new void *[jitNumOptions];

        // set up size of compilation log buffer
        jitOptions[0]        = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        int jitLogBufferSize = 1024;
        jitOptVals[0]        = (void *)(size_t)jitLogBufferSize;

        // set up pointer to the compilation log buffer
        jitOptions[1]      = CU_JIT_INFO_LOG_BUFFER;
        char *jitLogBuffer = new char[jitLogBufferSize];
        jitOptVals[1]      = jitLogBuffer;

        // set up pointer to set the Maximum # of registers for a particular kernel
        jitOptions[2]   = CU_JIT_MAX_REGISTERS;
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDynlinkJIT/matrixMulDynlinkJIT.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/0_Introduction/matrixMulDynlinkJIT/matrixMulDynlinkJIT.cpp:284-303
```cpp
        // This is the new CUDA 4.0 API for Kernel Parameter passing and Kernel Launching (simpler method)
        int   Matrix_Width_A = WA;
        int   Matrix_Width_B = WB;
        void *args[5]        = {&d_C, &d_A, &d_B, &Matrix_Width_A, &Matrix_Width_B};

        // JP: `cuLaunchKernel`: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
        checkCudaErrors(cuLaunchKernel(
            matrixMul, (WC / block_size), (HC / block_size), 1, block_size, block_size, 1, 0, NULL, args, NULL));
    }
#else // __CUDA_API_VERSION <= 3020
    {
        // This is the older CUDA Driver API for Kernel Parameter passing and Kernel Launching
        int offset = 0;
        {
            // setup execution parameters
            // JP: この連続する anchor 群では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
            checkCudaErrors(cuParamSetv(matrixMul, offset, &d_C, sizeof(d_C)));
            offset += sizeof(d_C);

            checkCudaErrors(cuParamSetv(matrixMul, offset, &d_A, sizeof(d_A)));
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDynlinkJIT/matrixMulDynlinkJIT.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `matrixMul_gold.cpp`

Source: cpp/0_Introduction/matrixMulDynlinkJIT/matrixMul_gold.cpp:32-57
```cpp
extern "C" void computeGold(float *, const float *, const float *, unsigned int, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void computeGold(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDynlinkJIT/matrixMul_gold.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `CUresult` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUDAAPI` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CUdeviceptr` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUarray` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUstream` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `Device` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `CUdevice` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUtexref` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUfunction` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuDevice` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUDA_SUCCESS` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CUcontext` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuDeviceGetAttribute` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuCtxDestroy` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |

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

- matrixMul 系では global memory の値を shared memory tile に移し、barrier 後に再利用します。性能の主役は arithmetic だけでなく memory reuse です。
- cuBLAS/Tensor Core 版がある場合は、同じ数学でも API、data layout、precision、workspace の責任分担が変わります。
- NVRTC/Driver/JIT 系では、compile/load/link と kernel launch が別の段階です。compile log、module、function handle、launch parameter の対応が重要です。
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
cmake --build build --target matrixMulDynlinkJIT
ctest --test-dir build -R matrixMulDynlinkJIT
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

- `CUresult` の直前と直後で、どの memory/resource が有効になったかをメモする。
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

- [Runtime, Driver, And NVRTC](../../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [CUDA Libraries](../../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。
- [CUDA Graphs](../../../docs_ja/themes/graphs.md): capture、node dependency、replay、graph update を読むための基礎です。
- [Multi-GPU, P2P, And IPC](../../../docs_ja/themes/multi_gpu_p2p_ipc.md): device topology、peer access、IPC handle、multi-process 境界を読むための基礎です。
- [Shared Memory](../../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Unified Memory](../../../docs_ja/themes/unified_memory.md): managed memory、migration、prefetch の意味を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
