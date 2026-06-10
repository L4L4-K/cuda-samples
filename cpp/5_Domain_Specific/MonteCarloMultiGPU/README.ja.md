# MonteCarloMultiGPU - Monte Carlo Option Pricing with Multi-GPU support - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample evaluates fair call price for a given set of European options using the Monte Carlo approach, taking advantage of all CUDA-capable GPUs installed in the system. This sample use double precision hardware if a GTX 200 class GPU is present.  The sample also takes advantage of CUDA 4.0 capability to supporting using a single CPU thread to control multiple GPUs

Random Number Generator, Computational Finance, CURAND Library

Original README headings: `MonteCarloMultiGPU - Monte Carlo Option Pricing with Multi-GPU support`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/MonteCarloMultiGPU` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `MonteCarloMultiGPU` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/MonteCarloMultiGPU`.
> **日本語**
> この sample の目的は、`MonteCarloMultiGPU` の小さな実装を通して CUDA Libraries, Multi-GPU, P2P, And IPC, Shared Memory, Streams And Events, Synchronization And Atomics を具体的に追うことです。
>
> **学習メモ**
> 最初に `MonteCarloMultiGPU.cpp, MonteCarlo_common.h, MonteCarlo_gold.cpp, MonteCarlo_kernel.cu, MonteCarlo_reduction.cuh` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `MonteCarloMultiGPU.cpp`: Host-side setup, API calls, validation, and cleanup.
- `MonteCarlo_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `MonteCarlo_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `MonteCarlo_kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `MonteCarlo_reduction.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `doc/MonteCarlo.doc`: Supporting file used by `doc/MonteCarlo.doc`.
- `doc/MonteCarlo.pdf`: Supporting file used by `doc/MonteCarlo.pdf`.
- `multithreading.cpp`: Host-side setup, API calls, validation, and cleanup.
- `multithreading.h`: Host/device declarations, helper types, constants, or library wrappers.
- `realtype.h`: Host/device declarations, helper types, constants, or library wrappers.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `MonteCarloMultiGPU.cpp` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Create library handles, descriptors, plans, or workspaces before the library call.
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

- `MonteCarloMultiGPU.cpp`: focus on `cudaSetDevice`, `cudaDeviceProp`, `cudaGetDeviceProperties`, `cudaCores`, `cudaDeviceSynchronize`.
- `MonteCarlo_common.h`: focus on `Device`, `curandState`, `cudaStream_t`, `curand_kernel`, `CUDA`.
- `MonteCarlo_gold.cpp`: focus on `curandGenerator_t`, `curand`, `curand_kernel`, `CUDA`, `curandCreateGeneratorHost`.
- `MonteCarlo_kernel.cu`: focus on `curandState`, `CUDA`, `threadIdx`, `blockIdx`, `cudaMalloc`.
- `MonteCarlo_reduction.cuh`: focus on `blockDim`, `launch`.
- `multithreading.cpp`: focus on `CUTThread`, `CUT_THREADROUTINE`.
- `multithreading.h`: focus on `CUTThread`, `CUT_THREADROUTINE`, `CUT_THREADPROC`, `CUT_THREADEND`.
- `realtype.h`: focus on control flow and helper functions.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/CMakeLists.txt:1-46
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(MonteCarloMultiGPU LANGUAGES C CXX CUDA)

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
# Add target for MonteCarloMultiGPU
add_executable(MonteCarloMultiGPU MonteCarlo_gold.cpp MonteCarlo_kernel.cu multithreading.cpp MonteCarloMultiGPU.cpp)

target_compile_options(MonteCarloMultiGPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(MonteCarloMultiGPU PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(MonteCarloMultiGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(MonteCarloMultiGPU PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(MonteCarloMultiGPU PRIVATE
    # JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
    CUDA::curand
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `MonteCarloMultiGPU.cpp`

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarloMultiGPU.cpp:35-53
```cpp
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <helper_cuda.h>      // helper functions (cuda error checking and initialization)
#include <helper_functions.h> // Helper functions (utilities, parsing, timing)
#include <multithreading.h>

#include "MonteCarlo_common.h"

int   *pArgc = NULL;
char **pArgv = NULL;

#ifdef WIN32
#define strcasecmp _strcmpi
#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarloMultiGPU.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarloMultiGPU.cpp:103-158
```cpp
StopWatchInterface **hTimer = NULL;

static CUT_THREADPROC solverThread(TOptionPlan *plan)
{
    // Init GPU
    checkCudaErrors(cudaSetDevice(plan->device));

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, plan->device));

    // Start the timer
    sdkStartTimer(&hTimer[plan->device]);

    // Allocate intermediate memory for MC integrator and initialize
    // RNG states
    initMonteCarloGPU(plan);

    // Main computation
    MonteCarloGPU(plan);

    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaDeviceSynchronize());

    // Stop the timer
    sdkStopTimer(&hTimer[plan->device]);

    // Shut down this GPU
    closeMonteCarloGPU(plan);

    // JP: `cudaStreamSynchronize`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    cudaStreamSynchronize(0);

    printf("solverThread() finished - GPU Device %d: %s\n", plan->device, deviceProp.name);

    CUT_THREADEND;
}

static void multiSolver(TOptionPlan *plan, int nPlans)
{
    // allocate and initialize an array of stream handles
    // JP: この連続する anchor 群では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    cudaStream_t *streams = (cudaStream_t *)malloc(nPlans * sizeof(cudaStream_t));
    cudaEvent_t  *events  = (cudaEvent_t *)malloc(nPlans * sizeof(cudaEvent_t));

    for (int i = 0; i < nPlans; i++) {
        checkCudaErrors(cudaSetDevice(plan[i].device));
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
        checkCudaErrors(cudaEventCreate(&(events[i])));
    }

    // Init Each GPU
    // In CUDA 4.0 we can call cudaSetDevice multiple times to target each device
    // Set the device desired, then perform initializations on that device

    for (int i = 0; i < nPlans; i++) {
        // set the target device to perform initialization on
```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarloMultiGPU.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `MonteCarlo_common.h`

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_common.h:29-47
```cpp
#ifndef MONTECARLO_COMMON_H
#define MONTECARLO_COMMON_H
#include "curand_kernel.h"
#include "realtype.h"

////////////////////////////////////////////////////////////////////////////////
// Global types
////////////////////////////////////////////////////////////////////////////////
typedef struct
{
    float S;
    float X;
    float T;
    float R;
    float V;
} TOptionData;

typedef struct
// #ifdef __CUDACC__
```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_common.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_common.h:82-101
```cpp

    // Intermediate device-side buffers
    void *d_Buffer;

    // random number generator states
    // JP: `curandState`: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
    curandState *rngStates;

    // Pseudorandom samples count
    int pathN;

    // Time stamp
    float time;

    int gridSize;
} TOptionPlan;

extern "C" void initMonteCarloGPU(TOptionPlan *plan);
// JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
extern "C" void MonteCarloGPU(TOptionPlan *plan, cudaStream_t stream = 0);
```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_common.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `MonteCarlo_gold.cpp`

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_gold.cpp:29-47
```cpp
#include <curand.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// #include "curand_kernel.h"
#include "helper_cuda.h"

////////////////////////////////////////////////////////////////////////////////
// Common types
////////////////////////////////////////////////////////////////////////////////
#include "MonteCarlo_common.h"

////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for Monte Carlo results validation
////////////////////////////////////////////////////////////////////////////////
#define A1       0.31938153
#define A2       -0.356563782
#define A3       1.781477937
```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_gold.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_gold.cpp:100-119
```cpp
    const double V        = optionData.V;
    const double MuByT    = (R - 0.5 * V * V) * T;
    const double VBySqrtT = V * sqrt(T);

    float            *samples;
    // JP: `curandGenerator_t`: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
    curandGenerator_t gen;

    checkCudaErrors(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    unsigned long long seed = 1234ULL;
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, seed));

    if (h_Samples != NULL) {
        samples = h_Samples;
    }
    else {
        samples = (float *)malloc(pathN * sizeof(float));
        checkCudaErrors(curandGenerateNormal(gen, samples, pathN, 0.0, 1.0));
    }

```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_gold.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_gold.cpp:128-144
```cpp
        sum2 += callValue * callValue;
    }

    if (h_Samples == NULL)
        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(samples);

    checkCudaErrors(curandDestroyGenerator(gen));

    // Derive average from the total sum and discount by riskfree rate
    callValue.Expected = (float)(exp(-R * T) * sum / (double)pathN);
    // Standard deviation
    double stdDev = sqrt(((double)pathN * sum2 - sum * sum) / ((double)pathN * (double)(pathN - 1)));
    // Confidence width; in 95% of all cases theoretical value lies within these
    // borders
    callValue.Confidence = (float)(exp(-R * T) * 1.96 * stdDev / sqrt((double)pathN));
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_gold.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `MonteCarlo_kernel.cu`

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_kernel.cu:32-50
```cuda
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>

namespace cg = cooperative_groups;
#include <curand_kernel.h>
#include <helper_cuda.h>

#include "MonteCarlo_common.h"

////////////////////////////////////////////////////////////////////////////////
// Helper reduction template
// Please see the "reduction" CUDA Sample for more information
////////////////////////////////////////////////////////////////////////////////
#include "MonteCarlo_reduction.cuh"

////////////////////////////////////////////////////////////////////////////////
// Internal GPU-side data structures
////////////////////////////////////////////////////////////////////////////////
```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_kernel.cu:79-111
```cuda
////////////////////////////////////////////////////////////////////////////////
// This kernel computes the integral over all paths using a single thread block
// per option. It is fastest when the number of thread blocks times the work per
// block is high enough to keep the GPU busy.
////////////////////////////////////////////////////////////////////////////////
// JP: `curandState`: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
static __global__ void MonteCarloOneBlockPerOption(curandState *__restrict rngStates,
                                                   const __TOptionData *__restrict d_OptionData,
                                                   __TOptionValue *__restrict d_CallValue,
                                                   int pathN,
                                                   int optionN)
{
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block          cta    = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    const int       SUM_N = THREAD_N;
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    __shared__ real s_SumCall[SUM_N];
    __shared__ real s_Sum2Call[SUM_N];

    // determine global thread id
    // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Copy random number state to local memory for efficiency
    // JP: この anchor では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    curandState localState = rngStates[tid];
    // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    for (int optionIndex = blockIdx.x; optionIndex < optionN; optionIndex += gridDim.x) {
        const real S        = d_OptionData[optionIndex].S;
        const real X        = d_OptionData[optionIndex].X;
```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_kernel.cu:164-183
```cuda
    checkCudaErrors(cudaMallocHost(&plan->h_OptionData, sizeof(__TOptionData) * (plan->optionCount)));
    // Allocate internal device memory
    checkCudaErrors(cudaMallocHost(&plan->h_CallValue, sizeof(__TOptionValue) * (plan->optionCount)));
    // Allocate states for pseudo random number generators
    checkCudaErrors(cudaMalloc((void **)&plan->rngStates, plan->gridSize * THREAD_N * sizeof(curandState)));
    // JP: `cudaMemset`, `curandState`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemset(plan->rngStates, 0, plan->gridSize * THREAD_N * sizeof(curandState)));

    // place each device pathN random numbers apart on the random number sequence
    rngSetupStates<<<plan->gridSize, THREAD_N>>>(plan->rngStates, plan->device);
    getLastCudaError("rngSetupStates kernel failed.\n");
}

// Compute statistics and deallocate internal device memory
extern "C" void closeMonteCarloGPU(TOptionPlan *plan)
{
    for (int i = 0; i < plan->optionCount; i++) {
        const double RT    = plan->optionData[i].R * plan->optionData[i].T;
        const double sum   = plan->h_CallValue[i].Expected;
        const double sum2  = plan->h_CallValue[i].Confidence;
```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_kernel.cu:189-208
```cuda
        // Confidence width; in 95% of all cases theoretical value lies within these
        // borders
        plan->callValue[i].Confidence = (float)(exp(-RT) * 1.96 * stdDev / sqrt(pathN));
    }

    // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFree(plan->rngStates));
    checkCudaErrors(cudaFreeHost(plan->h_CallValue));
    checkCudaErrors(cudaFreeHost(plan->h_OptionData));
    checkCudaErrors(cudaFree(plan->d_CallValue));
    checkCudaErrors(cudaFree(plan->d_OptionData));
}

// Main computations
// JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
extern "C" void MonteCarloGPU(TOptionPlan *plan, cudaStream_t stream)
{
    __TOptionValue *h_CallValue = plan->h_CallValue;

    if (plan->optionCount <= 0 || plan->optionCount > MAX_OPTIONS) {
```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `MonteCarlo_reduction.cuh`

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_reduction.cuh:29-83
```cuda
#ifndef MONTECARLO_REDUCTION_CUH
#define MONTECARLO_REDUCTION_CUH

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////
// This function calculates total sum for each of the two input arrays.
// SUM_N must be power of two
// Unrolling provides a bit of a performance improvement for small
// to medium path counts.
////////////////////////////////////////////////////////////////////////////////

template <class T, int SUM_N, int blockSize>
__device__ void
sumReduce(T *sum, T *sum2, cg::thread_block &cta, cg::thread_block_tile<32> &tile32, __TOptionValue *d_CallValue)
{
    const int VEC = 32;
    const int tid = cta.thread_rank();

    T beta  = sum[tid];
    T beta2 = sum2[tid];
    T temp, temp2;

    for (int i = VEC / 2; i > 0; i >>= 1) {
        if (tile32.thread_rank() < i) {
            temp  = sum[tid + i];
            temp2 = sum2[tid + i];
            beta += temp;
            beta2 += temp2;
            sum[tid]  = beta;
            sum2[tid] = beta2;
        }
        // JP: sync: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
        cg::sync(tile32);
    }
    cg::sync(cta);

    if (tid == 0) {
        beta  = 0;
        beta2 = 0;
        // JP: `blockDim`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
        for (int i = 0; i < blockDim.x; i += VEC) {
            beta += sum[i];
            beta2 += sum2[i];
        }
        __TOptionValue t = {beta, beta2};
        *d_CallValue     = t;
    }
    // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
    cg::sync(cta);
}

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_reduction.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `multithreading.cpp`

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/multithreading.cpp:29-75
```cpp
#include <multithreading.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// Create thread
CUTThread cutStartThread(CUT_THREADROUTINE func, void *data)
{
    return CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, data, 0, NULL);
}

// Wait for thread to finish
void cutEndThread(CUTThread thread)
{
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}

// Wait for multiple threads
void cutWaitForThreads(const CUTThread *threads, int num)
{
    WaitForMultipleObjects(num, threads, true, INFINITE);

    for (int i = 0; i < num; i++) {
        CloseHandle(threads[i]);
    }
}

#else
// Create thread
CUTThread cutStartThread(CUT_THREADROUTINE func, void *data)
{
    pthread_t thread;
    pthread_create(&thread, NULL, func, data);
    return thread;
}

// Wait for thread to finish
void cutEndThread(CUTThread thread) { pthread_join(thread, NULL); }

// Wait for multiple threads
void cutWaitForThreads(const CUTThread *threads, int num)
{
    for (int i = 0; i < num; i++) {
        cutEndThread(threads[i]);
    }
}

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/multithreading.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `multithreading.h`

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/multithreading.h:29-73
```cpp
#ifndef MULTITHREADING_H
#define MULTITHREADING_H

// Simple portable thread library.

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// Windows threads.
#include <windows.h>

typedef HANDLE CUTThread;
typedef unsigned(WINAPI *CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC unsigned WINAPI
#define CUT_THREADEND  return 0

#else
// POSIX threads.
#include <pthread.h>

typedef pthread_t CUTThread;
typedef void *(*CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC void
#define CUT_THREADEND
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    // Create thread.
    CUTThread cutStartThread(CUT_THREADROUTINE, void *data);

    // Wait for thread to finish.
    void cutEndThread(CUTThread thread);

    // Wait for multiple threads.
    void cutWaitForThreads(const CUTThread *threads, int num);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MULTITHREADING_H
```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/multithreading.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `realtype.h`

Source: cpp/5_Domain_Specific/MonteCarloMultiGPU/realtype.h:29-40
```cpp
#ifndef REALTYPE_H
#define REALTYPE_H

// #define DOUBLE_PRECISION

#ifndef DOUBLE_PRECISION
typedef float real;
#else
typedef double real;
#endif

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/MonteCarloMultiGPU/realtype.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `CUTThread` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaSetDevice` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `curandState` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaGetDeviceProperties` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaStream_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaDeviceProp` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `Device` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `CUT_THREADROUTINE` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaMallocHost` | pinned host memory を作り、async copy や DMA の前提を作る API です。 |
| `cudaCores` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- library sample では handle、descriptor、plan、workspace が GPU work の外側の resource です。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- multi-GPU sample では、device 選択、peer capability、context/IPC handle の寿命を分けて読みます。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。
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
cmake --build build --target MonteCarloMultiGPU
ctest --test-dir build -R MonteCarloMultiGPU
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

The sample validates the library result against a CPU/reference path or reports the documented success status.
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
- leading dimension、stride、descriptor、workspace size を host 配列の見た目だけで判断する。
- peer access が有効な device pair と、単に複数 GPU が存在することを混同する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `CUTThread` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- shared memory tile の producer、consumer、barrier を図にする。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- handle/descriptor/workspace の作成、利用、破棄を対応表にする。
- device ごとの ownership と、peer/IPC で共有される resource を分けて書く。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [CUDA Libraries](../../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。
- [Multi-GPU, P2P, And IPC](../../../docs_ja/themes/multi_gpu_p2p_ipc.md): device topology、peer access、IPC handle、multi-process 境界を読むための基礎です。
- [Shared Memory](../../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
